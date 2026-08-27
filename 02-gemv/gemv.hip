#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
#include "check.hpp"
#include "timing.hpp"
#include "device.hpp"

// native compiler types instead of wrappers
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef float    float4_t __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float dot2(half2_t a, half2_t b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

__device__ __forceinline__ half2_t as_h2(float v) {
    return __builtin_bit_cast(half2_t, v);
}

__device__ __forceinline__ float4_t mfma_16x16x16(half4_t a, half4_t b, float4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ float wave_reduce_shfl(float x) {
    for (int off = 32; off > 0; off >>= 1)
        x += __shfl_xor(x, off, 64);
    return x;
}

// DPP codes from ISA 13.3.9, table 93
#define DPP_ROW_SHR(n) (0x110 | (n))
#define DPP_ROW_BCAST15 0x142
#define DPP_ROW_BCAST31 0x143

// __builtin requires constants at compiling
template<int CTRL, int RMASK, int BMASK>
__device__ __forceinline__ float dpp_add(float x) {
    int xi = __builtin_bit_cast(int, x);
    int sh = __builtin_amdgcn_update_dpp(0, xi, CTRL, RMASK, BMASK, false);
    return x + __builtin_bit_cast(float, sh);
}

// Sum all 64 lanes, the result in lane 63

__device__ __forceinline__ float wave_reduce_dpp(float x) {
    x = dpp_add<DPP_ROW_SHR(1), 0xf, 0xf>(x);
    x = dpp_add<DPP_ROW_SHR(2), 0xf, 0xf>(x);
    x = dpp_add<DPP_ROW_SHR(4), 0xf, 0xe>(x);
    x = dpp_add<DPP_ROW_SHR(8), 0xf, 0xc>(x);
    x = dpp_add<DPP_ROW_BCAST15, 0xa, 0xf>(x);
    x = dpp_add<DPP_ROW_BCAST31, 0xc, 0xf>(x);
    int r = __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), 63);
    return __builtin_bit_cast(float, r);
}

__global__ __launch_bounds__(512)
void gemv_dot2(const half2_t* __restrict__ W, // [N][K/2], row-major
                const half2_t* __restrict__ x, // [K/2]
                float* __restrict__ y, // [N]
                int N, int K2) // K2 = K/2
{
    const int wave = threadIdx.y; // 0..7
    const int lane = threadIdx.x; // 0..63
    const int waves = blockDim.y;
    const int row0 = blockIdx.x * waves + wave;
    const int stride = gridDim.x * waves;

    for (int row = row0; row < N; row += stride) {
        const half2_t* wrow = W + size_t(row) * K2;

        // four independent accumulators
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        for (int i = lane; i < K2; i += 256) {
            half2_t w0 = wrow[i], x0 = x[i];
            half2_t w1 = wrow[i + 64], x1 = x[i + 64];
            half2_t w2 = wrow[i + 128], x2 = x[i + 128];
            half2_t w3 = wrow[i + 192], x3 = x[i + 192];
            acc0 = dot2(w0, x0, acc0);
            acc1 = dot2(w1, x1, acc1);
            acc2 = dot2(w2, x2, acc2);
            acc3 = dot2(w3, x3, acc3);
        }

        float acc = (acc0 + acc1) + (acc2 + acc3);
        float s = wave_reduce_dpp(acc);

        if (lane == 63) y[row] = s;
    }
}

// reference for double. Needs to compare it for the kernel
static void gemv_ref(const std::vector<_Float16>& W,
                    const std::vector<_Float16>& x,
                    std::vector<float>& y,
                    int N, int K)
{
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
            acc += double(float(W[size_t(r)*K + k])) * double(float(x[k]));
        y[r] = float(acc);
    }
}


__global__ __launch_bounds__(512)
void gemv_mfma(const half2_t* __restrict__ W,
               const half2_t* __restrict__ x,
               float* __restrict__ y,
               int N, int K2)
{
    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int waves  = blockDim.y;
    const int row0   = (blockIdx.x * waves + wave) * 16;   // 16 rows per wave
    const int stride = gridDim.x * waves * 16;

    for (int rb = row0; rb < N; rb += stride) {
        float4_t c = {0.f, 0.f, 0.f, 0.f};

        for (int k = 0; k < K2; k += 8) {          // 8 half2 = 16 values along K
            // A tile: row 0 is the activation, rows 1..15 are zeros.
            const int a_row = lane / 16;
            half4_t a = {0, 0, 0, 0};
            if (a_row == 0) {
                const int kk = k + (lane % 16) / 2 * 2;
                half2_t p0 = x[kk], p1 = x[kk + 1];
                a[0] = p0[0]; a[1] = p0[1]; a[2] = p1[0]; a[3] = p1[1];
            }

            // B tile: 16x16 block of weights.
            const int b_row = rb + lane % 16;
            const int bk    = k + (lane / 16) * 2;
            half2_t q0 = W[size_t(b_row) * K2 + bk];
            half2_t q1 = W[size_t(b_row) * K2 + bk + 1];
            half4_t b = {q0[0], q0[1], q1[0], q1[1]};

            c = mfma_16x16x16(a, b, c);
        }

        // Only row 0 of the C tile is useful.
        if (lane < 16 && rb + lane < N) y[rb + lane] = c[0];
    }
}

__global__ __launch_bounds__(512)
void gemv_dot2_wide(const half2_t* __restrict__ W,
                    const half2_t* __restrict__ x,
                    float* __restrict__ y,
                    int N, int K2)
{
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int waves = blockDim.y;
    const int row0 = blockIdx.x * waves + wave;
    const int stride = gridDim.x * waves;
    const int K4 = K2 / 4; // elements amount in f32x4 string

    const f32x4* x4 = (const f32x4*)x;

    for (int row = row0; row < N; row += stride) {
        const f32x4* w4 = (const f32x4*)(W + size_t(row) * K2);

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

        for (int j = lane; j < K4; j += 256) {
            f32x4 w0 = w4[j],       a0 = x4[j];
            f32x4 w1 = w4[j + 64],  a1 = x4[j + 64];
            f32x4 w2 = w4[j + 128], a2 = x4[j + 128];
            f32x4 w3 = w4[j + 192], a3 = x4[j + 192];

            acc0 = dot2(as_h2(w0.x), as_h2(a0.x), acc0);
            acc0 = dot2(as_h2(w0.y), as_h2(a0.y), acc0);
            acc0 = dot2(as_h2(w0.z), as_h2(a0.z), acc0);
            acc0 = dot2(as_h2(w0.w), as_h2(a0.w), acc0);

            acc1 = dot2(as_h2(w1.x), as_h2(a1.x), acc1);
            acc1 = dot2(as_h2(w1.y), as_h2(a1.y), acc1);
            acc1 = dot2(as_h2(w1.z), as_h2(a1.z), acc1);
            acc1 = dot2(as_h2(w1.w), as_h2(a1.w), acc1);

            acc2 = dot2(as_h2(w2.x), as_h2(a2.x), acc2);
            acc2 = dot2(as_h2(w2.y), as_h2(a2.y), acc2);
            acc2 = dot2(as_h2(w2.z), as_h2(a2.z), acc2);
            acc2 = dot2(as_h2(w2.w), as_h2(a2.w), acc2);

            acc3 = dot2(as_h2(w3.x), as_h2(a3.x), acc3);
            acc3 = dot2(as_h2(w3.y), as_h2(a3.y), acc3);
            acc3 = dot2(as_h2(w3.z), as_h2(a3.z), acc3);
            acc3 = dot2(as_h2(w3.w), as_h2(a3.w), acc3);
        }

        float acc = (acc0 + acc1) + (acc2 + acc3);
        float s = wave_reduce_dpp(acc);

        if (lane == 63) y[row] = s;
    }
}

int main(int argc, char** argv) {
    print_device();

    std::string mode = "dot2";
    int N = 4096, K = 4096;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mode") && i + 1 < argc)
            mode = argv[++i];
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc)
            N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc)
            K = std::atoi(argv[++i]);
    }


    if (K % 2048 != 0) {
        std::fprintf(stderr, "K must be a multiple of 2048\n");
        return 1;
    }

    const int K2 = K / 2;
    std::printf("\nmode=%s N=%d K=%d\n", mode.c_str(), N, K);

    // Hom much copies needed to outbound LLC
    const size_t w_bytes = size_t(N) * K * sizeof(_Float16);
    const size_t llc = 256ull << 20; // 256MB Infinity Cache
    const int copies = int((4 * llc + w_bytes - 1) / w_bytes); // >= 4x LLC
    std::printf("weights %.1f MB * %d copies = %.2f GB (LLC 256 MB)\n",
        w_bytes / 1e6, copies, copies * w_bytes / 1e9);

    // host data
    std::vector<_Float16> hW(size_t(N) * K), hx(K);
    for (size_t i = 0; i < hW.size(); ++i)
        hW[i] = _Float16((float(rand()) / float(RAND_MAX) - 0.5f) * 0.1f);
    for (int i = 0; i < K; ++i)
        hx[i] = _Float16((float(rand()) / float(RAND_MAX) - 0.5f) * 0.1f);


    std::vector<float> hy_ref(N);
    gemv_ref(hW, hx, hy_ref, N, K);

    // device
    half2_t *d_W = nullptr, *d_x = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_W, copies * w_bytes));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));

    for (int c = 0; c < copies; ++c)
        HIP_CHECK(hipMemcpy((char*)d_W + c * w_bytes, hW.data(),
            w_bytes, hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMemcpy(d_x, hx.data(), K * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, N * sizeof(float)));

    // run
    const dim3 block(64, 8); // one x, another as in Kog
    int gblocks = 256;
    auto launch_at = [&](int copy) {
        const half2_t* Wp = (const half2_t*)((const char*)d_W + size_t(copy) * w_bytes);
        if (mode == "dot2")
            hipLaunchKernelGGL(gemv_dot2, dim3(gblocks), block, 0, 0, Wp, d_x, d_y, N, K2);
        else if (mode == "wide")
            hipLaunchKernelGGL(gemv_dot2_wide, dim3(gblocks), block, 0, 0, Wp, d_x, d_y, N, K2);
        else
            hipLaunchKernelGGL(gemv_mfma, dim3(gblocks), block, 0, 0, Wp, d_x, d_y, N, K2);
    };

    launch_at(0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Correctness
    std::vector<float> hy(N);
    HIP_CHECK(hipMemcpy(hy.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    double max_rel = 0.0; int worst = -1; int bad = 0;

    for (int r = 0; r < N; ++r) {
        double ref = hy_ref[r];
        double den = std::max(1e-3, std::fabs(ref));
        double rel = std::fabs(hy[r] - ref) / den;
        if (rel > max_rel) {
            max_rel = rel;
            worst = r;
        }
        if (rel > 2e-2) ++bad;
    }

    std::printf("correctness: max rel err %.4e at row %d, %d/%d rows above 2e-2\n",
        max_rel, worst, bad, N);

    if (bad > 0) {
        std::printf("   ref[%d]=%.6f got[%d]=%.6f\n", worst, hy_ref[worst], worst, hy[worst]);
        std::fprintf(stderr, "FAILED - not measuring a wrong kernel\n");
        return 1;
    }

    // bench
    const double bytes_w = double(N) * K * sizeof(_Float16);
    const double bytes_x = double(K) * sizeof(_Float16);
    const double bytes_y = double(N) * sizeof(float);
    const double total   = bytes_w + bytes_x + bytes_y;
    const double ceiling = 4822.0;           // из 01-ceiling

    std::printf("bytes: weights %.1f MB (%.2f%%), x %.1f kB, y %.1f kB\n\n",
                bytes_w / 1e6, 100.0 * bytes_w / total,
                bytes_x / 1e3, bytes_y / 1e3);

    for (int g : {256, 512, 1024, 2048}) {
        gblocks = g;
        int c = 0;
        auto launch = [&]{ launch_at(c); c = (c + 1) % copies; };
        Stats st = time_kernel(launch, /*warmup=*/20, /*iters=*/100, /*reps=*/5);
        const double gbs = total / (st.median_ms * 1e-3) / 1e9;
        std::printf("%-6s blocks=%5d  %8.1f us  %8.1f GB/s  MBU %5.1f%%  (sd %.4f ms)\n",
                    mode.c_str(), g, st.median_ms * 1e3, gbs,
                    100.0 * gbs / ceiling, st.stddev_ms);
    }

    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    return 0;
}