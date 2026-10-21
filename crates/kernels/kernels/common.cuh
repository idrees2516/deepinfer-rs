// common.cuh — shared device-side foundation for DeepInfer-rs kernels.
//
// ONE source set, two compilers:
//   * NVRTC  (NVIDIA: sm_80 … sm_100/sm_103, GB200 NVL72 / GB300)
//   * hipRTC (AMD: gfx942/gfx950, Instinct MI335X / MI350 families)
// selected with `#if defined(__HIP__)`.
//
// All numeric codecs are bit-exact ports of `deepinfer-core::format` so that
// GPU results can be golden-tested against the CPU reference crates.
// No vendor headers are included: everything is raw types + builtins.
// Kernels use 1-D blocks only, so `threadIdx.x % DI_WARP` is the lane id on
// both vendors (NVIDIA warp 32, AMD wavefront 64).

#pragma once

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#if defined(__HIP__)
#define DI_WARP 64 // CDNA wavefront size
#else
#define DI_WARP 32 // NVIDIA warp size
#endif

#define DI_FULL_MASK 0xffffffffu

// ---------------------------------------------------------------------------
// Bit-exact float codecs (ports of deepinfer_core::format).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float di_bf16_to_f32(u16 v) {
    return __int_as_float(((u32)v) << 16);
}

// f32 -> bf16, round-to-nearest-even (matches core::format::f32_to_bf16_rne).
__device__ __forceinline__ u16 di_f32_to_bf16(float x) {
    u32 bits = __float_as_uint(x);
    if (isnan(x)) {
        return (u16)(((bits >> 16) & 0x8000u) | 0x7FC0u);
    }
    if (isinf(x) || x == 0.0f) {
        return (u16)(bits >> 16);
    }
    u32 lsb = (bits >> 16) & 1u;
    u32 rounded = bits + 0x7FFFu + lsb;
    return (u16)(rounded >> 16);
}

// OCP E4M3FN byte -> f32 (exact).
__device__ __forceinline__ float di_e4m3_to_f32(u8 v) {
    u32 sign = ((u32)v >> 7) & 1u;
    u32 exp = ((u32)v >> 3) & 0xFu;
    u32 man = (u32)v & 7u;
    u32 bits;
    if (exp == 0 && man == 0) {
        bits = sign << 31;
    } else if (exp == 0) {
        // subnormal: man * 2^-9 (exact in f32 for man <= 7)
        bits = __float_as_uint((float)man * 1.953125e-3f) | (sign << 31);
    } else if (exp == 15 && man == 7) {
        bits = (sign << 31) | 0x7FC00000u; // NaN
    } else {
        bits = (sign << 31) | ((exp + 120u) << 23) | (man << 20);
    }
    return __uint_as_float(bits);
}

// f32 -> OCP E4M3FN, RNE, with DeepSeek's saturate policy (port of
// core::format::f32_to_e4m3_rne; `sat` mirrors the `saturate` flag).
__device__ __forceinline__ u8 di_f32_to_e4m3(float x, bool sat) {
    u32 sign = (__float_as_uint(x) >> 31) & 1u;
    if (isnan(x)) return 0x7Fu;
    float a = fabsf(x);
    if (a >= 448.0f) return (u8)((sign << 7) | (sat ? 0x7Eu : 0x7Fu));
    if (a < 0.015625f) { // 2^-6: subnormal region, step 2^-9
        float q = a * 512.0f;
        float m = rintf(q); // round-to-nearest-even
        if (m >= 8.0f) return (u8)((sign << 7) | 0x08u);
        return (u8)((sign << 7) | (u8)m);
    }
    u32 bits = __float_as_uint(a);
    u32 be = (bits >> 23) & 0xFFu;
    u32 mant = bits & 0x007FFFFFu;
    int e4 = (int)be - 127 + 7;
    u32 half = 0x00080000u; // 2^19
    u32 m3 = mant >> 20;
    u32 rem = mant & 0x000FFFFFu;
    u32 m = m3 + (u32)((rem > half || (rem == half && (m3 & 1u) != 0u)) ? 1u : 0u);
    if (m == 8u) { m = 0u; e4 += 1; }
    if (e4 > 15) return (u8)((sign << 7) | (sat ? 0x7Eu : 0x7Fu));
    return (u8)((sign << 7) | ((u8)e4 << 3) | (u8)m);
}

// E8M0 byte -> f32: 2^(v-127); 0xFF = NaN, 0x00 = 2^-127 (subnormal f32).
__device__ __forceinline__ float di_e8m0_to_f32(u8 v) {
    if (v == 0xFFu) return NAN;
    if (v == 0x00u) return __uint_as_float(1u << 22);
    return __uint_as_float(((u32)v) << 23);
}

// Strictly-positive f32 scale -> E8M0 byte, ceil policy, with the CPU's
// exact-power-of-two protection (core::format::f32_to_e8m0 with Ceil).
// Callers guarantee finite x > 0 (validated host-side).
__device__ __forceinline__ u8 di_f32_to_e8m0_ceil(float x) {
    double l2 = log2((double)x);
    double e = ceil(l2);
    if (fabs(e - l2) < 1e-9) e = floor(l2);
    if (e < -127.0) e = -127.0;
    if (e > 127.0) e = 127.0;
    return (u8)((int)e + 127);
}

// E2M1 nibble -> f32 (exact lattice).
__device__ __forceinline__ float di_e2m1_to_f32(u8 nib) {
    float mag;
    switch (nib & 7u) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return (nib & 8u) ? -mag : mag;
}

// f32 -> E2M1 nibble, RNE over the lattice (core::format::f32_to_e2m1_rne).
__device__ __forceinline__ u8 di_f32_to_e2m1(float x) {
    u8 sign = (__float_as_uint(x) >> 31) ? 8u : 0u;
    if (isnan(x)) return sign | 7u;
    float a = fabsf(x);
    u8 code;
    if (a <= 0.25f) code = 0u;
    else if (a < 0.75f) code = 1u;
    else if (a <= 1.25f) code = 2u;
    else if (a < 1.75f) code = 3u;
    else if (a <= 2.5f) code = 4u;
    else if (a < 3.5f) code = 5u;
    else if (a <= 5.0f) code = 6u;
    else code = 7u;
    return sign | code;
}

__device__ __forceinline__ float di_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// ---------------------------------------------------------------------------
// Portable reductions over DI_WARP consecutive lanes (1-D blocks only).
// ---------------------------------------------------------------------------

__device__ __forceinline__ int di_lane_id() {
    return (int)(threadIdx.x & (DI_WARP - 1));
}

__device__ __forceinline__ int di_warp_id() {
    return (int)(threadIdx.x / DI_WARP);
}

__device__ __forceinline__ float di_warp_reduce_sum(float v) {
#pragma unroll
    for (int off = DI_WARP / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(DI_FULL_MASK, v, off);
    }
    return v;
}

__device__ __forceinline__ float di_warp_reduce_max(float v) {
#pragma unroll
    for (int off = DI_WARP / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(DI_FULL_MASK, v, off));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Deterministic top-k key: (ordered score bits << 32) | position.
// Ordering: float DESC by value, ties by position ASC — the CPU `top_k_desc`
// contract. NaN scores are rejected host-side before keys are built.
// ---------------------------------------------------------------------------

__device__ __forceinline__ u64 di_score_pos_key(float score, u32 pos) {
    u32 b = __float_as_uint(score);
    // Total order for floats incl. -inf: flip all bits for negatives, set
    // top bit for non-negatives; then invert once more for DESC.
    u32 asc = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (((u64)(~asc)) << 32) | (u64)pos;
}

__device__ __forceinline__ u32 di_key_pos(u64 k) { return (u32)(k & 0xffffffffull); }

// ---------------------------------------------------------------------------
// Shared-memory bitonic sort of u64 keys, ASCENDING. `n` must be a power of
// two, `v` must have at least `n` slots; blockDim must cover n via striding.
// Classic CUDA-samples formulation; fully deterministic.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void di_bitonic_sort_u64(u64* v, int n) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
                int j = i ^ stride;
                if (j > i && j < n) {
                    bool down = ((i & size) != 0);
                    if (((v[i] > v[j]) != down)) {
                        u64 tmp = v[i];
                        v[i] = v[j];
                        v[j] = tmp;
                    }
                }
            }
        }
    }
    __syncthreads();
}
