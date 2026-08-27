#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x)                                                     \
    do {                                                                 \
        hipError_t _e = (x);                                             \
        if (_e != hipSuccess) {                                          \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",              \
                         hipGetErrorString(_e), __FILE__, __LINE__);     \
            std::exit(1);                                                \
        }                                                                \
    } while (0)