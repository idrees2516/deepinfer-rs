// gemm.cu — NT GEMMs (D = C + A @ B^T) with DeepGEMM semantics.
//
// Layouts (all row-major):
//   A: [M, K]  (fp8: u8 e4m3 + sf [M, K/128] ue8m0;  bf16: u16)
//   B: [N, K]  (fp8: u8 e4m3 + sf [ceil(N/128), K/128] ue8m0;  bf16: u16)
//   D: [M, N] f32.
//
// FP8 scaling is the DeepGEMM recipe: activations 1x128, weights 128x128,
// per-128-k-block rescale, f32 accumulation between blocks (the SM90
// semantics; our CPU `deepinfer-gemm` crate is the golden reference).
//
// Both vendors share one skeleton: stage 64x32 bf16 tiles in shared memory,
// accumulate one 128-k block into `blk`, then fold `blk` into `acc` with the
// block scales. Only the per-tile core differs:
//   * NVIDIA: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 tensor
//     cores (fragment mapping per the PTX ISA tables — lane group g=lane>>2,
//     t=lane&3:
//       A a0=(row g,k 2t..2t+1) a1=(row g+8,k 2t..2t+1)
//          a2=(row g,k 2t+8..9) a3=(row g+8,k 2t+8..9)
//       B b0=(k 2t..2t+1,col g) b1=(k 2t+8..9,col g)
//       C c0=(row g,col 2t) c1=(row g,col 2t+1) c2=(row g+8,col 2t)
//          c3=(row g+8,col 2t+1) )
//     FP8 operands are decoded to bf16 during global->shared staging, so
//     memory traffic stays FP8 while compute runs at bf16 tensor rate
//     (documented v1 tradeoff; docs/GPU.md lists the e4m3-mma/tcgen05
//     upgrade path).
//   * AMD: LDS-tiled scalar f32 FMA per 4x4 subtile. Correct and
//     bandwidth-efficient; MFMA upgrade requires on-device layout
//     validation and is deliberately not shipped blind.

#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 32
#define GEMM_THREADS 256

__device__ __forceinline__ void gemm_stage_tile(
    const void* __restrict__ base_a, const void* __restrict__ base_b,
    int a_is_fp8, int m0, int n0, int k0, int M, int N, int K, long long b_row_stride,
    u16* __restrict__ as, u16* __restrict__ bs) {
    if (a_is_fp8) {
        const u8* a = (const u8*)base_a;
        const u8* b = (const u8*)base_b;
        for (int idx = (int)threadIdx.x; idx < GEMM_BM * GEMM_BK; idx += GEMM_THREADS) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int gm = m0 + r;
            int gk = k0 + c;
            u16 v = 0;
            if (gm < M && gk < K) v = di_f32_to_bf16(di_e4m3_to_f32(a[(u64)gm * K + gk]));
            as[idx] = v;
        }
        for (int idx = (int)threadIdx.x; idx < GEMM_BN * GEMM_BK; idx += GEMM_THREADS) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int gn = n0 + r;
            int gk = k0 + c;
            u16 v = 0;
            if (gn < N && gk < K) v = di_f32_to_bf16(di_e4m3_to_f32(b[(u64)gn * b_row_stride + gk]));
            bs[idx] = v;
        }
    } else {
        const u16* a = (const u16*)base_a;
        const u16* b = (const u16*)base_b;
        for (int idx = (int)threadIdx.x; idx < GEMM_BM * GEMM_BK; idx += GEMM_THREADS) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int gm = m0 + r;
            int gk = k0 + c;
            as[idx] = (gm < M && gk < K) ? a[(u64)gm * K + gk] : (u16)0;
        }
        for (int idx = (int)threadIdx.x; idx < GEMM_BN * GEMM_BK; idx += GEMM_THREADS) {
            int r = idx / GEMM_BK;
            int c = idx - r * GEMM_BK;
            int gn = n0 + r;
            int gk = k0 + c;
            bs[idx] = (gn < N && gk < K) ? b[(u64)gn * b_row_stride + gk] : (u16)0;
        }
    }
    __syncthreads();
}

#if !defined(__HIP__)

// NVIDIA tensor-core core: blk += A_tile @ B_tile^T (BK=32 -> 2 k16 steps).
// acc layout: 4 mma tiles (mi,ni in {0,1}x{0,1}) x 4 f32.
__device__ __forceinline__ void gemm_core_nv(const u16* __restrict__ as,
                                             const u16* __restrict__ bs,
                                             float* __restrict__ blk) {
    int warp = (int)(threadIdx.x / DI_WARP);
    int wm = warp >> 2; // 0..1
    int wn = warp & 3;  // 0..3
    int g = di_lane_id() >> 2;
    int t = di_lane_id() & 3;
#pragma unroll
    for (int kj = 0; kj < 2; ++kj) { // two k16 steps
        u32 a0[2], a1[2], a2[2], a3[2];
        u32 b0[2], b1[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int arow = wm * 32 + mi * 16 + g;
            int kc = kj * 16 + 2 * t;
            a0[mi] = (u32)as[arow * GEMM_BK + kc] | ((u32)as[arow * GEMM_BK + kc + 1] << 16);
            a1[mi] = (u32)as[(arow + 8) * GEMM_BK + kc] | ((u32)as[(arow + 8) * GEMM_BK + kc + 1] << 16);
            a2[mi] = (u32)as[arow * GEMM_BK + kc + 8] | ((u32)as[arow * GEMM_BK + kc + 9] << 16);
            a3[mi] = (u32)as[(arow + 8) * GEMM_BK + kc + 8] | ((u32)as[(arow + 8) * GEMM_BK + kc + 9] << 16);
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            int brow = wn * 16 + ni * 8 + g;
            int kc = kj * 16 + 2 * t;
            b0[ni] = (u32)bs[brow * GEMM_BK + kc] | ((u32)bs[brow * GEMM_BK + kc + 1] << 16);
            b1[ni] = (u32)bs[brow * GEMM_BK + kc + 8] | ((u32)bs[brow * GEMM_BK + kc + 9] << 16);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                int base = (mi * 2 + ni) * 4;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(blk[base + 0]), "+f"(blk[base + 1]), "+f"(blk[base + 2]),
                      "+f"(blk[base + 3])
                    : "r"(a0[mi]), "r"(a1[mi]), "r"(a2[mi]), "r"(a3[mi]), "r"(b0[ni]),
                      "r"(b1[ni]));
            }
        }
    }
}

__device__ __forceinline__ void gemm_zero_blk_nv(float* blk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) blk[i] = 0.0f;
}

// Fold the finished 128-k block: acc += blk * (sa[row] * sb[colblock]).
__device__ __forceinline__ void gemm_rescale_nv(float* __restrict__ acc,
                                                const float* __restrict__ blk,
                                                const u8* __restrict__ a_sf,
                                                const u8* __restrict__ b_sf,
                                                int m0, int n0, int kb,
                                                int kb_blocks) {
    int warp = (int)(threadIdx.x / DI_WARP);
    int wm = warp >> 2;
    int wn = warp & 3;
    int g = di_lane_id() >> 2;
    float sb = di_e8m0_to_f32(b_sf[(u64)((n0 + wn * 16) >> 7) * kb_blocks + kb]);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        int rlo = m0 + wm * 32 + mi * 16 + g;
        float sa0 = di_e8m0_to_f32(a_sf[(u64)rlo * kb_blocks + kb]);
        float sa1 = di_e8m0_to_f32(a_sf[(u64)(rlo + 8) * kb_blocks + kb]);
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            int base = (mi * 2 + ni) * 4;
            acc[base + 0] += blk[base + 0] * (sa0 * sb);
            acc[base + 1] += blk[base + 1] * (sa0 * sb);
            acc[base + 2] += blk[base + 2] * (sa1 * sb);
            acc[base + 3] += blk[base + 3] * (sa1 * sb);
        }
    }
}

// Store the 16 accumulators of this lane (guarded).
__device__ __forceinline__ void gemm_store_nv(const float* __restrict__ acc,
                                              const float* __restrict__ c,
                                              float* __restrict__ d,
                                              int m0, int n0, int M, int N) {
    int warp = (int)(threadIdx.x / DI_WARP);
    int wm = warp >> 2;
    int wn = warp & 3;
    int g = di_lane_id() >> 2;
    int t = di_lane_id() & 3;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const float* tile = &acc[(mi * 2 + ni) * 4];
            int rs[2] = {m0 + wm * 32 + mi * 16 + g, m0 + wm * 32 + mi * 16 + g + 8};
            int cs[2] = {n0 + wn * 16 + ni * 8 + 2 * t, n0 + wn * 16 + ni * 8 + 2 * t + 1};
#pragma unroll
            for (int ri = 0; ri < 2; ++ri) {
                if (rs[ri] >= M) continue;
#pragma unroll
                for (int ci = 0; ci < 2; ++ci) {
                    if (cs[ci] >= N) continue;
                    float v = tile[ri * 2 + ci];
                    if (c) v += c[(u64)rs[ri] * N + cs[ci]];
                    d[(u64)rs[ri] * N + cs[ci]] = v;
                }
            }
        }
    }
}

#else // __HIP__

// AMD scalar core: each of 256 threads owns a 4x4 subtile.
// acc/blk layout: [i*4 + j] for rows m0+ty*4+i, cols n0+tx*4+j.
__device__ __forceinline__ void gemm_core_amd(const u16* __restrict__ as,
                                              const u16* __restrict__ bs,
                                              float* __restrict__ blk) {
    int ty = (int)threadIdx.x >> 4;
    int tx = (int)threadIdx.x & 15;
    int m0 = ty * 4;
    int n0 = tx * 4;
    for (int k = 0; k < GEMM_BK; ++k) {
        float av0 = di_bf16_to_f32(as[(m0 + 0) * GEMM_BK + k]);
        float av1 = di_bf16_to_f32(as[(m0 + 1) * GEMM_BK + k]);
        float av2 = di_bf16_to_f32(as[(m0 + 2) * GEMM_BK + k]);
        float av3 = di_bf16_to_f32(as[(m0 + 3) * GEMM_BK + k]);
        float bv0 = di_bf16_to_f32(bs[(n0 + 0) * GEMM_BK + k]);
        float bv1 = di_bf16_to_f32(bs[(n0 + 1) * GEMM_BK + k]);
        float bv2 = di_bf16_to_f32(bs[(n0 + 2) * GEMM_BK + k]);
        float bv3 = di_bf16_to_f32(bs[(n0 + 3) * GEMM_BK + k]);
        blk[0] += av0 * bv0;
        blk[1] += av0 * bv1;
        blk[2] += av0 * bv2;
        blk[3] += av0 * bv3;
        blk[4] += av1 * bv0;
        blk[5] += av1 * bv1;
        blk[6] += av1 * bv2;
        blk[7] += av1 * bv3;
        blk[8] += av2 * bv0;
        blk[9] += av2 * bv1;
        blk[10] += av2 * bv2;
        blk[11] += av2 * bv3;
        blk[12] += av3 * bv0;
        blk[13] += av3 * bv1;
        blk[14] += av3 * bv2;
        blk[15] += av3 * bv3;
    }
}

__device__ __forceinline__ void gemm_zero_blk_amd(float* blk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) blk[i] = 0.0f;
}

__device__ __forceinline__ void gemm_rescale_amd(float* __restrict__ acc,
                                                 const float* __restrict__ blk,
                                                 const u8* __restrict__ a_sf,
                                                 const u8* __restrict__ b_sf,
                                                 int m0, int n0, int kb,
                                                 int kb_blocks) {
    int ty = (int)threadIdx.x >> 4;
    int tx = (int)threadIdx.x & 15;
    float sb = di_e8m0_to_f32(b_sf[(u64)((n0 + tx * 4) >> 7) * kb_blocks + kb]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gm = m0 + ty * 4 + i;
        float sa = di_e8m0_to_f32(a_sf[(u64)gm * kb_blocks + kb]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i * 4 + j] += blk[i * 4 + j] * (sa * sb);
        }
    }
}

__device__ __forceinline__ void gemm_store_amd(const float* __restrict__ acc,
                                               const float* __restrict__ c,
                                               float* __restrict__ d,
                                               int m0, int n0, int M, int N) {
    int ty = (int)threadIdx.x >> 4;
    int tx = (int)threadIdx.x & 15;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gm = m0 + ty * 4 + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gn = n0 + tx * 4 + j;
            if (gn >= N) continue;
            float v = acc[i * 4 + j];
            if (c) v += c[(u64)gm * N + gn];
            d[(u64)gm * N + gn] = v;
        }
    }
}

#endif

// ---------------------------------------------------------------------------
// Dense BF16 NT GEMM: D = C + A @ B^T (C optional, f32).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(GEMM_THREADS)
di_gemm_bf16_nt_kernel(const void* __restrict__ a,
                       const void* __restrict__ b,
                       const float* __restrict__ c,
                       float* __restrict__ d,
                       int M, int N, int K) {
    __shared__ u16 as[GEMM_BM * GEMM_BK];
    __shared__ u16 bs[GEMM_BN * GEMM_BK];
    int m0 = (int)blockIdx.y * GEMM_BM;
    int n0 = (int)blockIdx.x * GEMM_BN;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        gemm_stage_tile(a, b, 0, m0, n0, k0, M, N, K, K, as, bs);
#if defined(__HIP__)
        gemm_core_amd(as, bs, acc);
#else
        float blk[16];
        gemm_zero_blk_nv(blk);
        gemm_core_nv(as, bs, blk);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += blk[i];
#endif
        __syncthreads();
    }
#if defined(__HIP__)
    gemm_store_amd(acc, c, d, m0, n0, M, N);
#else
    gemm_store_nv(acc, c, d, m0, n0, M, N);
#endif
}

// ---------------------------------------------------------------------------
// Dense FP8 block-scaled NT GEMM. Requires K % 128 == 0 (CPU contract).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(GEMM_THREADS)
di_gemm_fp8_nt_kernel(const void* __restrict__ a,
                      const u8* __restrict__ a_sf,
                      const void* __restrict__ b,
                      const u8* __restrict__ b_sf,
                      float* __restrict__ d,
                      int M, int N, int K) {
    __shared__ u16 as[GEMM_BM * GEMM_BK];
    __shared__ u16 bs[GEMM_BN * GEMM_BK];
    int m0 = (int)blockIdx.y * GEMM_BM;
    int n0 = (int)blockIdx.x * GEMM_BN;
    int kb_blocks = K >> 7;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int kb = 0; kb < kb_blocks; ++kb) {
        int k0 = kb << 7;
#if defined(__HIP__)
        float blk[16];
        gemm_zero_blk_amd(blk);
#else
        float blk[16];
        gemm_zero_blk_nv(blk);
#endif
        for (int kk = 0; kk < 4; ++kk) { // 4 x BK=32 = 128
            gemm_stage_tile(a, b, 1, m0, n0, k0 + kk * GEMM_BK, M, N, K, K, as, bs);
#if defined(__HIP__)
            gemm_core_amd(as, bs, blk);
#else
            gemm_core_nv(as, bs, blk);
#endif
            __syncthreads();
        }
#if defined(__HIP__)
        gemm_rescale_amd(acc, blk, a_sf, b_sf, m0, n0, kb, kb_blocks);
#else
        gemm_rescale_nv(acc, blk, a_sf, b_sf, m0, n0, kb, kb_blocks);
#endif
    }
#if defined(__HIP__)
    gemm_store_amd(acc, NULL, d, m0, n0, M, N);
#else
    gemm_store_nv(acc, NULL, d, m0, n0, M, N);
#endif
}

// ---------------------------------------------------------------------------
// Grouped-contiguous FP8 GEMM (MoE prefill; DeepGEMM m_indices semantics).
// Segments (given by cumulative m_offsets[G+1], each a multiple of 128) tile
// [0, M_total) exactly; each 64-row CTA lies inside one segment. B rows are
// addressed with a per-group base pointer arithmetic:
//   b_g = b + g * (u64)N * K;  b_sf_g = b_sf + g * (N/128) * kb_blocks.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(GEMM_THREADS)
di_gemm_fp8_grouped_nt_kernel(const void* __restrict__ a,
                              const u8* __restrict__ a_sf,
                              const void* __restrict__ b,
                              const u8* __restrict__ b_sf,
                              const int* __restrict__ m_offsets,
                              float* __restrict__ d,
                              int M_total, int N, int K, int G) {
    __shared__ u16 as[GEMM_BM * GEMM_BK];
    __shared__ u16 bs[GEMM_BN * GEMM_BK];
    int m0 = (int)blockIdx.y * GEMM_BM;
    int n0 = (int)blockIdx.x * GEMM_BN;
    if (m0 >= M_total) return;
    // Binary search the segment containing row m0 (offsets ascending, G+1).
    int lo = 0, hi = G - 1, g = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (m_offsets[mid] <= m0) {
            g = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    int seg_end = m_offsets[g + 1];
    if (m0 >= seg_end) return; // zero-padded tail tile of the last segment
    int M_eff = seg_end;       // rows below seg_end belong to this group
    const u8* bg = (const u8*)b + (u64)g * N * K;
    int kb_blocks = K >> 7;
    int n_sf_rows = (N + 127) >> 7;
    const u8* bsf_g = b_sf + (u64)g * n_sf_rows * kb_blocks;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int kb = 0; kb < kb_blocks; ++kb) {
        int k0 = kb << 7;
#if defined(__HIP__)
        float blk[16];
        gemm_zero_blk_amd(blk);
#else
        float blk[16];
        gemm_zero_blk_nv(blk);
#endif
        for (int kk = 0; kk < 4; ++kk) {
            gemm_stage_tile(a, bg, 1, m0, n0, k0 + kk * GEMM_BK, M_eff, N, K, K, as, bs);
#if defined(__HIP__)
            gemm_core_amd(as, bs, blk);
#else
            gemm_core_nv(as, bs, blk);
#endif
            __syncthreads();
        }
#if defined(__HIP__)
        gemm_rescale_amd(acc, blk, a_sf, bsf_g, m0, n0, kb, kb_blocks);
#else
        gemm_rescale_nv(acc, blk, a_sf, bsf_g, m0, n0, kb, kb_blocks);
#endif
    }
#if defined(__HIP__)
    gemm_store_amd(acc, NULL, d, m0, n0, M_eff, N);
#else
    gemm_store_nv(acc, NULL, d, m0, n0, M_eff, N);
#endif
}
