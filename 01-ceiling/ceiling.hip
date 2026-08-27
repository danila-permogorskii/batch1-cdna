#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.hpp"
#include "timing.hpp"
#include "device.hpp"

typedef float f32x4 __attribute__((ext_vector_type(4)));

// float4 provide one global_load_dwordx4
template<bool NT>
__global__ __launch_bounds__(512) // block size for the compiler
void stream_read(const f32x4* __restrict__ src, size_t n4, float* sink) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x; // without size_t it may overflow the buffer

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    for (size_t i = idx; i < n4; i += stride) {
        f32x4 v = NT ? __builtin_nontemporal_load(src + i) : src[i]; // check in asm
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }

    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1e30f) sink[0] = s; // Always false, needs to avoid full cycle loading
}

int main(int argc, char** argv) {
    const size_t bytes = 8ull << 30; // 8 GiB ~ 8.59 GB
    const size_t n4 = bytes / sizeof(f32x4);

    f32x4* d_src = nullptr;
    float* d_sink = nullptr;

    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_sink, sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 1, bytes)); // avoiding exceptional handling of 0's at the allocator layer
    HIP_CHECK(hipMemset(d_sink, 0, sizeof(float)));

    std::printf("buffer: %.2f GB (LLC is 0.25 GB)\n", bytes / 1e9);

    const int threads = 512;
    const int blocks = 256;
    HIP_CHECK(hipMemset(d_sink, 0, sizeof(float)));
    hipLaunchKernelGGL((stream_read<false>), dim3(blocks), dim3(threads), 0, 0, d_src, n4, d_sink);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("kernel ran\n");

    auto run = [&](int blocks, bool nt) {
        auto launch = [&]{
            if(nt) hipLaunchKernelGGL((stream_read<true>), dim3(blocks), dim3(threads), 0, 0, d_src, n4, d_sink);
            else hipLaunchKernelGGL((stream_read<false>), dim3(blocks), dim3(threads), 0, 0, d_src, n4, d_sink);
        };
        Stats st = time_kernel(launch, /* warmup */10, /* iters */20, /* reps */5);
        double gbs = bytes / (st.median_ms * 1e-3) / 1e9;
        std::printf("blocks=%4d nt=%d %8.3f ms %8.1f GB/s (sd %.3f ms)\n",
            blocks, int(nt), st.median_ms, gbs, st.stddev_ms);
        return gbs;
    };

    for (int b : {64, 128, 256, 512, 1024}) { run(b, false); }
    std::printf("\n");
    for (int b : {256, 512, 1024}) { run(b, true); }
    return 0;
}