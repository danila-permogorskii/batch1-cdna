#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include "check.hpp"
#include "device.hpp"

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(half2_t a, half2_t b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

// clock speed counter
__device__ __forceinline__ long long rtc() {
    return __builtin_amdgcn_s_memrealtime();
}

// calibration
__global__ void rtc_probe(long long* out) {
    if (threadIdx.x == 0) out[0] = rtc();
}

#define CHAIN_UNROLL 16

// clean chain. Every dot2 read and write the same accumulator
__device__ __forceinline__ float chain_pure(half2_t a, half2_t b, float acc, int n) {
    #pragma unroll CHAIN_UNROLL
    for (int i = 0; i < n; ++i) acc = dot2(a, b, acc);
    return acc;
}

__device__ __forceinline__ float chain_mixed(half2_t a, half2_t b, float acc, int n) {
    #pragma unroll CHAIN_UNROLL
    for (int i = 0; i < n; ++i) {
        acc = dot2(a, b, acc);
        acc = acc * 1.0000001f; // forwading interruption
    }
    return acc;
}

__device__ __forceinline__ float chain_dual(half2_t a, half2_t b, float acc, int n) {
    float acc2 = 0.f;
    #pragma unroll CHAIN_UNROLL
    for (int i = 0; i < n; ++i) {
        acc = dot2(a, b, acc);
        acc2 = dot2(b, a, acc2);
    }

    return acc + acc2;
}

__global__ __launch_bounds__(1024)
void bench_chain(float* sink, long long* ticks, int iters, int mode) {

    // from memory to avoid compiler constant making
    half2_t a, b;
    a[0] = _Float16(1.0009765625f); a[1] = _Float16(0.99951171875f);
    b[0] = _Float16(0.99951171875f); b[1] = _Float16(1.0009765625f);

    float acc = 0.f;

    __builtin_amdgcn_s_barrier();
    long long t0 = rtc();

    if (mode == 0) acc = chain_pure(a, b, acc, iters);
    else if (mode == 1) acc = chain_mixed(a, b, acc, iters);
    else acc = chain_dual(a, b, acc, iters);

    long long t1 = rtc();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ticks[blockIdx.x] = t1 - t0;
        if (acc == 1e30f) sink[0] = acc; // always false
    }
}

int main(int argc, char** argv)
{
    print_device();

    int iters = 100000;
    int waves = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--waves") && i + 1 < argc) waves = std::atoi(argv[++i]);
    }

    long long* d_ticks = nullptr;
    float*     d_sink  = nullptr;
    HIP_CHECK(hipMalloc(&d_ticks, sizeof(long long) * 4));
    HIP_CHECK(hipMalloc(&d_sink,  sizeof(float)));
    HIP_CHECK(hipMemset(d_sink, 0, sizeof(float)));

    // clock calibrating
    long long h0 = 0, h1 = 0;
    hipLaunchKernelGGL(rtc_probe, dim3(1), dim3(64), 0, 0, d_ticks);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h0, d_ticks, sizeof(long long), hipMemcpyDeviceToHost));
    usleep(200000);                                     // 0.2 s
    hipLaunchKernelGGL(rtc_probe, dim3(1), dim3(64), 0, 0, d_ticks);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h1, d_ticks, sizeof(long long), hipMemcpyDeviceToHost));

    const double rtc_hz = double(h1 - h0) / 0.2;
    std::printf("\nrtc frequency: %.2f MHz  (expected ~100)\n", rtc_hz / 1e6);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const double shader_hz = double(prop.clockRate) * 1e3;
    std::printf("shader clock : %.2f MHz\n", shader_hz / 1e6);
    std::printf("iterations   : %d\n\n", iters);

    // all chains
    const char* names[3] = { "pure", "mixed", "dual" };
    const int   ops[3]   = { 1, 2, 2 };     // instructions per cycle

    std::printf("waves/SIMD   : %d\n", waves);
    std::printf("%-6s %12s %14s %14s %10s\n",
                "chain", "ticks", "ns/instr", "shader cyc", "instr");

    for (int mode = 0; mode < 3; ++mode) {
        // warmup
        hipLaunchKernelGGL(bench_chain, dim3(1), dim3(64, waves), 0, 0,
                           d_sink, d_ticks, iters, mode);
        HIP_CHECK(hipDeviceSynchronize());

        long long best = 0;
        for (int rep = 0; rep < 5; ++rep) {
            hipLaunchKernelGGL(bench_chain, dim3(1), dim3(64,waves), 0, 0,
                               d_sink, d_ticks, iters, mode);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            long long t = 0;
            HIP_CHECK(hipMemcpy(&t, d_ticks, sizeof(long long), hipMemcpyDeviceToHost));
            if (rep == 0 || t < best) best = t;
        }

        const double n_instr = double(iters) * ops[mode];
        const double seconds = double(best) / rtc_hz;
        const double ns_per  = seconds / n_instr * 1e9;
        const double cycles  = ns_per * 1e-9 * shader_hz;

        std::printf("%-6s %12lld %14.4f %14.2f %10.0f\n",
                    names[mode], best, ns_per, cycles, n_instr);
    }

    HIP_CHECK(hipFree(d_ticks));
    HIP_CHECK(hipFree(d_sink));
    return 0;
}