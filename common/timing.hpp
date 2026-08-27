#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include "check.hpp"

struct Stats { double median_ms, min_ms, max_ms, stddev_ms; };

inline Stats time_kernel(const std::function<void()>& launch,
    int warmup = 20, int iters = 200, int reps = 5) 
{
    for (int i = 0; i < warmup: ++i) lanch();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> per_rep;
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));

    for (int r = 0; r < reps; ++r) {
        HIP_CHECK(hipEventRecord(a));
        for (int i = 0; i < iters; ++i) launch();
        HIP_CHECK(hipEventRecord(b));
        HIP_CHECK(hipEventSynchronize(b));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        per_rep.push_back(double(ms) / iters);
    }

    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));

    std::sort(per_rep.begin(), per_rep.end());
    double mean = 0;
    for (double v : per_rep) mean += v;
    mean /= per_rep.size();

    double var = 0;
    for (double v : per_rep) var += (v-mean)*(v-mean);
    var /= per_rep.size();

    return {
        per_rep[per_rep.size()/2],
        per_rep.front(),
        per_rep.back(),
        std::sqrt(var)
    };
}