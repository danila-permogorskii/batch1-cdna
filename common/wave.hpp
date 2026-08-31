#pragma once
#include <hip/hip_runtime.h>

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// permanent 100 MHz, needs to use between CPU and XCD
__device__ __forceinline__ long long rtc() {
    return __builtin_amdgcn_s_memrealtime();
}

// 64 bit counter for kernels.
__device__ __forceinline__ long long clk() {
    return __builtin_amdgcn_s_memtime();
}

// dl ops
__device__ __forceinline__ float dot2(half2_t a , half2_t b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef float f32x16 __attribute__((ext_vector_type(16)));

__device__ __forceinline__ f32x4 mfma_16x16x16(half4_t a, half4_t b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ f32x16 mfma_32x32x8(half4_t a, half4_t b, f32x16 c) {
    return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
}

// DPP
#define DPP_ROW_SHR(n) (0x110 | (n))
#define DPP_ROW_BCAST15 0x142
#define DPP_ROW_BCAST31 0x143

template <int ctrl, int row_mask = 0xf, int bank_mask = 0xf>
__device__ __forceinline__ float dpp_add(float x) {
    const int src = __builtin_bit_cast(int, x);
    const int r = __builtin_amdgcn_update_dpp(0, src, ctrl, row_mask, bank_mask, false);
    return x + __builtin_bit_cast(float, r);
}

// six chains
__device__ __forceinline__ float wave_reduce_dpp_lane63(float x) {
    x = dpp_add<DPP_ROW_SHR(1)>(x);
    x = dpp_add<DPP_ROW_SHR(2)>(x);
    x = dpp_add<DPP_ROW_SHR(4)>(x);
    x = dpp_add<DPP_ROW_SHR(8)>(x);
    x = dpp_add<DPP_ROW_BCAST15, 0xa>(x);
    x = dpp_add<DPP_ROW_BCAST31, 0xc>(x);
    return x;
}

__device__ __forceinline__ float wave_reduce_dpp(float x) {
    const float t = wave_reduce_dpp_lane63(x);
    return __builtin_bit_cast(float,
     __builtin_amdgcn_readlane(__builtin_bit_cast(int, t), 63));
}

__device__ __forceinline__ float wave_reduce_shfl(float x) {
    #pragma unroll
    for (int off = 1; off < 64; off <<= 1) {
        const float o = __shfl_xor(x, off, 64);
        x += o;
    }
    return x;
}

__device__ __forceinline__ float wave_reduce_lds(float x, float* smem) {
    const int lane = threadIdx.x & 63;
    #pragma unroll
    for (int off = 1; off < 64; off <<= 1) {
        smem[lane] = x;
        __syncthreads();
        const float o = smem[lane ^ off];
        __syncthreads();
        x += o;
    }
    return x;
}