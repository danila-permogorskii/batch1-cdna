#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include "check.hpp"

inline void print_device() {
        hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("device        : %s\n", p.name);
    std::printf("arch          : %s\n", p.gcnArchName);
    std::printf("CUs           : %d\n", p.multiProcessorCount);
    std::printf("clock         : %d kHz\n", p.clockRate);
    std::printf("VRAM          : %.1f GB\n", p.totalGlobalMem / 1e9);
    std::printf("LDS per block : %zu KiB\n", p.sharedMemPerBlock / 1024);
    std::printf("warp size     : %d\n", p.warpSize);
}