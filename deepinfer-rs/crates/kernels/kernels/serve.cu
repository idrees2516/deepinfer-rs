// serve.cu — the paged-cache multi-sequence serving kernel family.
//
// The single-sequence DeviceModel carves contiguous row ranges out of flat
// per-layer arenas; this family replaces that with a **paged device cache**
// in the vLLM / FlashMLA sense: per-layer pools of fixed 64-token pages
// (the same granularity as the DSA sparse index `page_idx * 64 + offset`
// contract), per-sequence page tables in device memory, and kernels that
// translate every logical row `r` through
//     phys = tbl[r >> 6] * 64 + (r & 63)
// so many sequences share one pool and finish/free pages independently.
//
// One launch covers the WHOLE batch: grid.z (or a slot index folded into
// grid.x/y) runs over `B` sequence slots; each slot reads its geometry from
// a device-resident cell block written by `di_serve_fanout_kernel` from a
// small host upload (`{active, pos, len, sq, sqnb}` per slot), so no
// varying scalar is ever a kernel argument — the same graph-safe design
// rule as the `decode.cu` family.
//
// Cell ABI (i32 per slot; mirrors `ServeCell` in serve.rs):
//   0 active   1 pos (absolute first new row)   2 len (cached rows)
//   3 sq (rows this step)   4 sqnb (quant blocks already complete)
//   derived: 5 total  6 row0 (= len)  7 knb (total/qb)  8 topk_eff
//            9 swa_slot 10 swa_n 11 swa_qoff 12 xlo 13 xn 14 xqoff 15 pages

#define SERVE_PAGE 64

// Heads (decode) / query rows (prefill) per CTA — the same warp-per-
// (head, query) structure as attention.cu (which is a SEPARATE translation
// unit; the define is local to each).
#define ATT_WARPS 8

// Cell field indices (ABI with serve.rs).
#define SV_ACTIVE 0
#define SV_POS 1
#define SV_LEN 2
#define SV_SQ 3
#define SV_SQNB 4
#define SV_TOTAL 5
#define SV_ROW0 6
#define SV_KNB 7
#define SV_TOPK 8
#define SV_SWA_SLOT 9
#define SV_SWA_N 10
#define SV_SWA_QOFF 11
#define SV_XLO 12
#define SV_XN 13
#define SV_XQOFF 14
#define SV_PAGES 15
#define SV_COUNT 16

__device__ __forceinline__ int sv_phys(const int* __restrict__ tbl, int r) {
    return tbl[r >> 6] * SERVE_PAGE + (r & (SERVE_PAGE - 1));
}
__device__ __forceinline__ u8* sv_row(void* __restrict__ pool,
                                      const int* __restrict__ tbl, int r, int rb) {
    return (u8*)pool + (u64)sv_phys(tbl, r) * rb;
}
__device__ __forceinline__ const u8* sv_crow(const void* __restrict__ pool,
                                             const int* __restrict__ tbl, int r, int rb) {
    return (const u8*)pool + (u64)sv_phys(tbl, r) * rb;
}

// Paged KV row element decode (same three formats as att_load_k).
__device__ __forceinline__ float sv_load_k(const void* __restrict__ pool,
                                           const int* __restrict__ tbl,
                                           int fmt, int r, int d, int D) {
    if (fmt == 0) {
        const float* row = (const float*)sv_crow(pool, tbl, r, D * 4);
        return row[d];
    }
    if (fmt == 1) {
        const u8* row = sv_crow(pool, tbl, r, 528);
        return di_e4m3_to_f32(row[d]) * di_e8m0_to_f32(row[D + (d >> 5)]);
    }
    const u8* row = sv_crow(pool, tbl, r, 288);
    u8 nib = ((d & 1) == 0) ? (u8)(row[d >> 1] & 0x0Fu) : (u8)(row[d >> 1] >> 4);
    return di_e2m1_to_f32(nib) * di_e4m3_to_f32(row[256 + (d >> 4)]);
}

// Row byte stride of one cached row in the pool (by cache format).
__device__ __forceinline__ int sv_row_bytes(int fmt, int D) {
    return (fmt == 0) ? D * 4 : ((fmt == 1) ? 528 : 288);
}

// Element decode from a STAGED (shared-memory) copy of one cached row.
// Identical expressions to sv_load_k on the same bytes, so pipelined
// kernels are bit-identical to the direct-load family.
__device__ __forceinline__ float sv_deq_k(const u8* __restrict__ row,
                                          int fmt, int d, int D) {
    if (fmt == 0) return ((const float*)row)[d];
    if (fmt == 1) return di_e4m3_to_f32(row[d]) * di_e8m0_to_f32(row[D + (d >> 5)]);
    u8 nib = ((d & 1) == 0) ? (u8)(row[d >> 1] & 0x0Fu) : (u8)(row[d >> 1] >> 4);
    return di_e2m1_to_f32(nib) * di_e4m3_to_f32(row[256 + (d >> 4)]);
}

// ---------------------------------------------------------------------------
// Fanout: host cells {active, pos, len, sq, sqnb} -> full cell block.
// grid = (B,); block = 32.
// ---------------------------------------------------------------------------
extern "C" __global__ void di_serve_fanout_kernel(const int* __restrict__ in_cells,
                                                  int* __restrict__ cells,
                                                  int B, int cross_win, int swa_win,
                                                  int qb, int topk) {
    int b = blockIdx.x;
    if (b >= B || threadIdx.x != 0) return;
    const int* inr = in_cells + (u64)b * 5;
    int* c = cells + (u64)b * SV_COUNT;
    const int active = inr[SV_ACTIVE];
    const int pos = inr[SV_POS];
    const int len = inr[SV_LEN];
    const int sq = inr[SV_SQ];
    const int sqnb = inr[SV_SQNB];
    const int total = len + sq;
    const int swa_n = total < swa_win ? total : swa_win;
    const int xn = total < cross_win ? total : cross_win;
    c[SV_ACTIVE] = active;
    c[SV_POS] = pos;
    c[SV_LEN] = len;
    c[SV_SQ] = sq;
    c[SV_SQNB] = sqnb;
    c[SV_TOTAL] = total;
    c[SV_ROW0] = len;
    c[SV_KNB] = total / qb;
    c[SV_TOPK] = topk < total ? topk : total;
    c[SV_SWA_SLOT] = (total - 1) % (swa_win > 0 ? swa_win : 1);
    c[SV_SWA_N] = swa_n;
    c[SV_SWA_QOFF] = (pos + sq - 1) - (total - swa_n);
    c[SV_XLO] = total - xn;
    c[SV_XN] = xn;
    c[SV_XQOFF] = pos - (total - xn);
    c[SV_PAGES] = (total + SERVE_PAGE - 1) / SERVE_PAGE;
}

// ---------------------------------------------------------------------------
// RoPE position fill: pos_kv [B*sq], pos_q [B*sq*H].
// ---------------------------------------------------------------------------
extern "C" __global__ void di_serve_pos_fill_kernel(const int* __restrict__ cells,
                                                    u32* __restrict__ pos_kv,
                                                    u32* __restrict__ pos_q,
                                                    int B, int sq, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * sq) return;
    int b = idx / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = idx - b * sq;
    u32 p = (u32)(c[SV_POS] + i);
    pos_kv[idx] = p;
    for (int h = 0; h < H; ++h) pos_q[(u64)idx * H + h] = p;
}

// ---------------------------------------------------------------------------
// Generic f32 row append into a paged pool (k_raw, latent f32 caches).
// src rows: [B*sq, elems]; grid = (ceil(elems/256), B*sq).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_append_rows_kernel(const float* __restrict__ src,
                            void* __restrict__ pool,
                            const u64* __restrict__ tbl_ptrs,
                            const int* __restrict__ cells,
                            int elems, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int* tbl = (const int*)tbl_ptrs[b];
    float* dst = (float*)sv_row(pool, tbl, c[SV_ROW0] + i, elems * 4);
    const float* s = src + (u64)row * elems;
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < elems;
         d += gridDim.x * blockDim.x) {
        dst[d] = s[d];
    }
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm + paged append (k_norm).
// grid = (ceil(ihd/…)? one warp per (row, head)); block = 8 warps.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_append_norm_rows_kernel(const float* __restrict__ src,
                                  void* __restrict__ pool,
                                  const u64* __restrict__ tbl_ptrs,
                                  const int* __restrict__ cells,
                                  int nh, int ihd, float eps, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int* tbl = (const int*)tbl_ptrs[b];
    const int dim = nh * ihd;
    float* dst = (float*)sv_row(pool, tbl, c[SV_ROW0] + i, dim * 4);
    const float* s = src + (u64)row * dim;
    int wh = di_warp_id();
    for (int h = wh; h < nh; h += (int)(blockDim.x / DI_WARP)) {
        const float* sh = s + h * ihd;
        float ms = 0.0f;
        for (int d = di_lane_id(); d < ihd; d += DI_WARP) ms += sh[d] * sh[d];
        ms = di_warp_reduce_sum(ms) / (float)ihd;
        float rs = rsqrtf(ms + eps);
        float* dh = dst + h * ihd;
        for (int d = di_lane_id(); d < ihd; d += DI_WARP) dh[d] = sh[d] * rs;
    }
}

// ---------------------------------------------------------------------------
// KV-cache encode into the paged pool (fmt 0 = f32 copy, 1 = FP8 528 B,
// 2 = FP4 288 B). One warp per (row, group).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_kv_encode_kernel(const float* __restrict__ src,
                          void* __restrict__ pool,
                          const u64* __restrict__ tbl_ptrs,
                          const int* __restrict__ cells,
                          int hd, int fmt, int sq, int B) {
    int gw = blockIdx.x * (int)(blockDim.x / DI_WARP) + di_warp_id();
    int groups = (fmt == 2) ? (hd / 16) : (hd / 32);
    int rb = (fmt == 0) ? hd * 4 : (fmt == 1 ? 528 : 288);
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    if (gw >= groups) return;
    int i = row - b * sq;
    const int* tbl = (const int*)tbl_ptrs[b];
    u8* dst = sv_row(pool, tbl, c[SV_ROW0] + i, rb);
    const float* xs = src + (u64)row * hd + (u64)gw * (fmt == 2 ? 16 : 32);
    if (fmt == 0) {
        // f32 rows: each warp copies its own 32-element slice.
        for (int t = di_lane_id(); t < 32; t += DI_WARP) dst[gw * 32 + t] = xs[t];
        return;
    }
    if (fmt == 1) {
        const int g = gw * 32;
        float amax = 0.0f;
        for (int t = di_lane_id(); t < 32; t += DI_WARP) {
            float v = xs[t];
            if (isnan(v)) v = 0.0f;
            amax = fmaxf(amax, fabsf(v));
        }
        amax = di_warp_reduce_max(amax);
        float ratio = amax / 448.0f;
        u8 sfb = 0x00u;
        if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
        float scale = di_e8m0_to_f32(sfb);
        for (int t = di_lane_id(); t < 32; t += DI_WARP) {
            float v = xs[t];
            if (isnan(v)) v = 0.0f;
            dst[g + t] = di_f32_to_e4m3(v / scale, true);
        }
        if (di_lane_id() == 0) dst[hd + gw] = sfb;
        return;
    }
    // fmt 2: one e4m3 scale per 16 dims, two e2m1 per byte (even = low).
    const int g = gw * 16;
    float amax = 0.0f;
    for (int t = di_lane_id(); t < 16; t += DI_WARP) {
        float v = xs[t];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float ratio = amax / 6.0f;
    u8 sfb = 0x00u;
    if (ratio > 0.0f) sfb = di_f32_to_e4m3(ratio, true);
    float scale = di_e4m3_to_f32(sfb);
    if (scale > 0.0f) scale = 1.0f / scale; else scale = 0.0f;
    for (int t = di_lane_id(); t < 16; t += DI_WARP) {
        float v = xs[t];
        if (isnan(v)) v = 0.0f;
        u8 q = di_f32_to_e2m1(v * scale);
        u8 byte = dst[(g + t) >> 1];
        if (((g + t) & 1) == 0) byte = (u8)((byte & 0xF0u) | (q & 0x0Fu));
        else byte = (u8)((byte & 0x0Fu) | (u8)(q << 4));
        dst[(g + t) >> 1] = byte;
    }
    if (di_lane_id() == 0) dst[256 + gw] = sfb;
}

// ---------------------------------------------------------------------------
// Cascade conv features over paged k rows (per slot, per new row).
// grid = (ceil(idx_dim/256), B*sq).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_conv_feat_kernel(const void* __restrict__ k_pool,
                          void* __restrict__ feat_pool,
                          const u64* __restrict__ tbl_ptrs,
                          const int* __restrict__ cells,
                          const float* __restrict__ kw,
                          const int* __restrict__ kw_off,
                          int num_kernels, int hidden, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int j = c[SV_ROW0] + i; // logical (seq-local) row index
    const int* tbl = (const int*)tbl_ptrs[b];
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden;
         d += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int ck = 0; ck < num_kernels; ++ck) {
            int len = kw_off[ck + 1] - kw_off[ck];
            for (int t = 0; t < len; ++t) {
                if (t > j) break;
                const float* krow = (const float*)sv_crow(k_pool, tbl, j - t, hidden * 4);
                acc += kw[kw_off[ck] + t] * krow[d];
            }
        }
        float* frow = (float*)sv_row(feat_pool, tbl, j, hidden * 4);
        frow[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Gated RMSNorm of paged feat rows (in place). grid = (B*sq,).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_conv_gated_norm_kernel(void* __restrict__ feat_pool,
                                const u64* __restrict__ tbl_ptrs,
                                const int* __restrict__ cells,
                                const float* __restrict__ gain,
                                int hidden, int sq, int B) {
    int row = blockIdx.x;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int* tbl = (const int*)tbl_ptrs[b];
    float* frow = (float*)sv_row(feat_pool, tbl, c[SV_ROW0] + i, hidden * 4);
    float g = 0.0f, ms = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        g += frow[d] * gain[d];
        ms += frow[d] * frow[d];
    }
    __shared__ float red[2][256];
    red[0][threadIdx.x] = g;
    red[1][threadIdx.x] = ms;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[0][threadIdx.x] += red[0][threadIdx.x + s];
            red[1][threadIdx.x] += red[1][threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        g = di_sigmoid(red[0][0]);
        ms = red[1][0] / (float)hidden;
    }
    __shared__ float sh[2];
    if (threadIdx.x == 0) {
        sh[0] = g;
        sh[1] = rsqrtf(ms + 1e-6f);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        frow[d] *= sh[0] * sh[1];
    }
}

// ---------------------------------------------------------------------------
// Lightning block quant over the paged k_norm pool.
// One block per (slot, block); blocks [sqnb, knb) of the slot. k_sf lives
// in a per-sequence buffer (`sf_ptrs[b]`, [max_blocks] u8).
// Deterministic: block-local amax via a fixed tree.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_lightning_quant_kernel(const void* __restrict__ k_norm_pool,
                                void* __restrict__ k_u8_pool,
                                const u64* __restrict__ tbl_ptrs,
                                const u64* __restrict__ sf_ptrs,
                                const int* __restrict__ cells,
                                int idx_dim, int qb, int B, int max_blocks) {
    int b = blockIdx.y;
    int blk = blockIdx.x;
    if (b >= B) return;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    if (blk < c[SV_SQNB] || blk >= c[SV_KNB] || blk >= max_blocks) return;
    const int* tbl = (const int*)tbl_ptrs[b];
    const int r0 = blk * qb;
    // Fixed-tree amax over the block (qb rows x idx_dim dims).
    extern __shared__ float red[];
    float local = 0.0f;
    for (int r = 0; r < qb; ++r) {
        const float* row = (const float*)sv_crow(k_norm_pool, tbl, r0 + r, idx_dim * 4);
        for (int d = threadIdx.x; d < idx_dim; d += blockDim.x) {
            float v = fabsf(row[d]);
            if (v > local) local = v;
        }
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float scale;
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float amax = red[0];
        float ratio = amax / 448.0f;
        u8 sfb = 0x00u;
        if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
        ((u8*)sf_ptrs[b])[blk] = sfb;
        scale = di_e8m0_to_f32(sfb);
        inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    }
    __syncthreads();
    for (int r = 0; r < qb; ++r) {
        const float* row = (const float*)sv_crow(k_norm_pool, tbl, r0 + r, idx_dim * 4);
        u8* dst = sv_row(k_u8_pool, tbl, r0 + r, idx_dim);
        for (int d = threadIdx.x; d < idx_dim; d += blockDim.x) {
            dst[d] = di_f32_to_e4m3(row[d] * inv, true);
        }
    }
}

// ---------------------------------------------------------------------------
// Thunder logits over the paged k caches. grid = (gridx, B*sq); the query
// row is staged in shared memory, keys stream through the page table.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_thunder_logits_kernel(const float* __restrict__ q_f32,
                               const u8* __restrict__ q_u8,
                               const u8* __restrict__ q_sf,
                               const void* __restrict__ k_norm_pool,
                               const void* __restrict__ k_u8_pool,
                               const u64* __restrict__ tbl_ptrs,
                               const u64* __restrict__ sf_ptrs,
                               const int* __restrict__ cells,
                               const float* __restrict__ w,
                               float* __restrict__ scores,
                               int nh, int ihd, int qb, int q_nblocks,
                               int max_cols, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int dim = nh * ihd;
    extern __shared__ float sq_f32[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        if (row / qb < q_nblocks) {
            // Global row-block index (rows are [B*sq] across the batch).
            float scale = di_e8m0_to_f32(q_sf[row / qb]);
            sq_f32[d] = di_e4m3_to_f32(q_u8[(u64)row * dim + d]) * scale;
        } else {
            sq_f32[d] = q_f32[(u64)row * dim + d];
        }
    }
    __syncthreads();
    const int s_kv = c[SV_TOTAL];
    const int knb = c[SV_KNB];
    const int* tbl = (const int*)tbl_ptrs[b];
    const u8* ksf = (const u8*)sf_ptrs[b];
    const float* wrow = w + (u64)row * nh;
    float* outrow = scores + (u64)row * max_cols;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < s_kv;
         j += gridDim.x * blockDim.x) {
        bool kq = (j / qb) < knb;
        float kscale = 1.0f;
        if (kq) kscale = di_e8m0_to_f32(ksf[j / qb]);
        const u8* ku8 = (const u8*)sv_crow(k_u8_pool, tbl, j, dim);
        const float* kf32 = (const float*)sv_crow(k_norm_pool, tbl, j, dim * 4);
        float acc = 0.0f;
        for (int h = 0; h < nh; ++h) {
            float dot = 0.0f;
            int base = h * ihd;
            if (kq) {
                for (int d = 0; d < ihd; ++d) {
                    dot += sq_f32[base + d] * (di_e4m3_to_f32(ku8[base + d]) * kscale);
                }
            } else {
                for (int d = 0; d < ihd; ++d) {
                    dot += sq_f32[base + d] * kf32[base + d];
                }
            }
            acc += (dot > 0.0f ? dot : 0.0f) * wrow[h];
        }
        outrow[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// Conv-branch scores over the paged feat pool.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_dot_scores_kernel(const float* __restrict__ q,
                           const void* __restrict__ feat_pool,
                           const u64* __restrict__ tbl_ptrs,
                           const int* __restrict__ cells,
                           float* __restrict__ out,
                           int hidden, int max_cols, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    const int* tbl = (const int*)tbl_ptrs[b];
    const float* qi = q + (u64)row * hidden;
    float* outrow = out + (u64)row * max_cols;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < c[SV_TOTAL];
         j += gridDim.x * blockDim.x) {
        const float* fj = (const float*)sv_crow(feat_pool, tbl, j, hidden * 4);
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) acc += qi[d] * fj[d];
        outrow[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// Gate blend, per-slot (gt/gc indexed by the global row = b*sq + i).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_gate_blend_kernel(const float* __restrict__ gt,
                           const float* __restrict__ gc,
                           const float* __restrict__ conv,
                           float* __restrict__ thunder,
                           const int* __restrict__ cells,
                           int max_cols, int sq, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * sq * max_cols) return;
    int row = idx / max_cols;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int j = idx - row * max_cols;
    if (j >= c[SV_TOTAL]) {
        thunder[idx] = -INFINITY;
        return;
    }
    float th = thunder[idx];
    float cv = conv[idx];
    if (th == -INFINITY || cv == -INFINITY) {
        thunder[idx] = -INFINITY;
    } else {
        float denom = gt[row] + gc[row];
        thunder[idx] = (denom > 0.0f) ? ((gt[row] * th + gc[row] * cv) / denom) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Causal + extrapolation-window clip (crate contract: key j is masked for
// j > pos_row, where pos_row is the ABSOLUTE position of the query row).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_causal_clip_kernel(float* __restrict__ scores,
                            const int* __restrict__ cells,
                            int max_cols, int window, int sq, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * sq * max_cols) return;
    int row = idx / max_cols;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    int j = idx - row * max_cols;
    int pos_row = c[SV_POS] + i; // absolute position of the query row
    if (j >= c[SV_TOTAL] || j > pos_row || (window > 0 && j < pos_row + 1 - window)) {
        scores[idx] = -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Block preselect (fixed geometry num_blocks x per). grid = (num_blocks,
// B*sq). Candidate slots beyond the slot's total stay zero.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(1024)
di_serve_block_preselect_kernel(const float* __restrict__ scores,
                                u64* __restrict__ cands,
                                const int* __restrict__ cells,
                                int block, int per, int max_cols, int sq, int B) {
    extern __shared__ u64 keys[];
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int bs = blockIdx.x * block;
    if (bs >= c[SV_TOTAL]) return;
    int be = bs + block;
    if (be > c[SV_TOTAL]) be = c[SV_TOTAL];
    int n = be - bs;
    int pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    for (int t = threadIdx.x; t < pow2; t += blockDim.x) {
        keys[t] = (t < n) ? di_score_pos_key(scores[(u64)row * max_cols + bs + t], (u32)(bs + t))
                          : 0ull;
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, pow2);
    if (threadIdx.x < per) {
        u64 k = keys[pow2 - 1 - (int)threadIdx.x];
        if (k != 0ull) {
            cands[((u64)row * gridDim.x) * per + (u64)blockIdx.x * per + threadIdx.x] = k;
        }
    }
}

// ---------------------------------------------------------------------------
// Top-K merge + scatter over the preselect candidates (one block per row;
// the block loads all candidates, sorts, writes the top-k logical indices).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(1024)
di_serve_topk_kernel(const u64* __restrict__ cands,
                     int* __restrict__ sel,
                     const int* __restrict__ cells,
                     int num_blocks, int per, int topk_max, int sq, int B) {
    int row = blockIdx.x;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    const int n_cands = num_blocks * per;
    extern __shared__ u64 keys[];
    int pow2 = 1;
    while (pow2 < n_cands) pow2 <<= 1;
    for (int t = threadIdx.x; t < pow2; t += blockDim.x) {
        keys[t] = (t < n_cands) ? cands[(u64)row * n_cands + t] : 0ull;
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, pow2);
    const int k = c[SV_TOPK];
    if (threadIdx.x == 0) {
        for (int t = 0; t < topk_max; ++t) {
            int* s = sel + (u64)row * topk_max;
            s[t] = (t < k) ? (int)(keys[pow2 - 1 - t] & 0xffffffffull) : -1;
        }
    }
}

// ---------------------------------------------------------------------------
// MLA decode, split-KV partial pass over the paged KV pool.
// grid = (splits, ceil(H / ATT_WARPS), B*sq). Partials:
// pm/pl [B*sq*H, splits]; po [B*sq*H, splits, D].
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_mla_decode_partial_kernel(const float* __restrict__ q,
                                   const void* __restrict__ kv_pool,
                                   const u64* __restrict__ tbl_ptrs,
                                   const int* __restrict__ cells,
                                   const int* __restrict__ sel,
                                   float* __restrict__ pm,
                                   float* __restrict__ pl,
                                   float* __restrict__ po,
                                   int H, int D, int topk_max, int chunk,
                                   float sm_scale, int fmt, int sq, int B) {
    int split = blockIdx.x;
    int splits = gridDim.x;
    int hbase = blockIdx.y * ATT_WARPS;
    int row = blockIdx.z;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int warp = (int)(threadIdx.x / DI_WARP);
    int h = hbase + warp;
    if (h >= H) return;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const float* qv_ptr = q + ((u64)row * H + h) * D;
    const int* selrow = sel + (u64)row * topk_max;
    const int* tbl = (const int*)tbl_ptrs[b];
    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    for (int i = 0; i < per; ++i) qv[i] = qv_ptr[d0 + i];
    float m = -INFINITY;
    float l = 0.0f;
    const int c0 = split * chunk;
    const int c1 = min(c0 + chunk, c[SV_TOPK]);
    for (int ci = c0; ci < c1; ++ci) {
        int idx = selrow[ci];
        if (idx < 0) continue;
        float kv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
#pragma unroll 2
        for (int i = 0; i < per; ++i) kv[i] = sv_load_k(kv_pool, tbl, fmt, idx, d0 + i, D);
        float s = 0.0f;
        for (int i = 0; i < per; ++i) s += qv[i] * kv[i];
        s = di_warp_reduce_sum(s) * sm_scale;
        if (s > m) {
            float res = expf(m - s);
#pragma unroll
            for (int i = 0; i < 16; ++i) ov[i] *= res;
            l *= res;
            m = s;
        }
        float p = expf(s - m);
        l += p;
        for (int i = 0; i < per; ++i) ov[i] += p * kv[i];
    }
    const u64 hp = ((u64)row * H + h) * splits + split;
    float* porow = po + (hp * D);
    for (int i = 0; i < per; ++i) porow[d0 + i] = ov[i];
    if (lane == 0) {
        pm[hp] = m;
        pl[hp] = l;
    }
}

// Combine pass: sequential split-ascending online merge — identical
// structure and accumulation order to `di_mla_decode_combine_kernel`.
// grid = (ceil(D / 256), B*sq*H); block = 256.
extern "C" __global__ void __launch_bounds__(256)
di_serve_mla_decode_combine_kernel(const float* __restrict__ pm,
                                   const float* __restrict__ pl,
                                   const float* __restrict__ po,
                                   const float* __restrict__ sink, // [H] or NULL
                                   float* __restrict__ out,
                                   float* __restrict__ lse,
                                   const int* __restrict__ cells,
                                   int H, int D, int splits, int sq, int B) {
    int d = blockIdx.x * 256 + (int)threadIdx.x;
    int row_h = blockIdx.y; // = row * H + h
    if (d >= D) return;
    int row = row_h / H;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;

    float o = 0.0f, l = 0.0f, m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        u64 pbase = ((u64)row_h * splits) + s;
        float ms = pm[pbase];
        float ls = pl[pbase];
        float os = po[pbase * D + d];
        float m_new = fmaxf(m, ms);
        if (m_new == -INFINITY) continue; // both empty so far
        float so = expf(m - m_new);
        float sc = expf(ms - m_new);
        o = o * so + os * sc;
        l = l * so + ls * sc;
        m = m_new;
    }

    float lse_v;
    float v;
    if (l > 0.0f) {
        lse_v = m + logf(l);
        v = o / l;
    } else {
        lse_v = INFINITY;
        v = 0.0f;
    }
    if (sink) {
        // exp(lse)/(exp(lse)+exp(sink)) == 1/(1+exp(sink-lse)); +inf lse -> 1.
        v *= 1.0f / (1.0f + expf(sink[row_h % H] - lse_v));
    }
    out[((u64)row_h) * D + d] = v;
    if (threadIdx.x == 0 && lse) lse[(u64)row_h] = lse_v;
}

// ---------------------------------------------------------------------------
// Sparse prefill over the paged KV pool (DSA-selected indices).
// grid = (ceil(sq_rows / ATT_WARPS), H, B); one warp per query row.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_prefill_sparse_kernel(const float* __restrict__ q,
                               const void* __restrict__ kv_pool,
                               const u64* __restrict__ tbl_ptrs,
                               const int* __restrict__ cells,
                               const int* __restrict__ sel,
                               float* __restrict__ out,
                               float* __restrict__ lse,
                               int H, int D, int topk_max,
                               float sm_scale, int fmt, int sq, int B) {
    int b = blockIdx.z;
    if (b >= B) return;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int qrow = blockIdx.x * ATT_WARPS + (int)(threadIdx.x / DI_WARP);
    if (qrow >= sq) return;
    int h = blockIdx.y;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const int row = b * sq + qrow;
    const float* qv_ptr = q + ((u64)row * H + h) * D;
    const int* idxs = sel + (u64)row * topk_max;
    const int* tbl = (const int*)tbl_ptrs[b];
    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    for (int i = 0; i < per; ++i) qv[i] = qv_ptr[d0 + i];
    float m = -INFINITY;
    float l = 0.0f;
    const int lim = c[SV_TOPK];
    for (int ci = 0; ci < lim; ++ci) {
        int idx = idxs[ci];
        if (idx < 0) continue;
        float kv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
        for (int i = 0; i < per; ++i) kv[i] = sv_load_k(kv_pool, tbl, fmt, idx, d0 + i, D);
        float s = 0.0f;
        for (int i = 0; i < per; ++i) s += qv[i] * kv[i];
        s = di_warp_reduce_sum(s) * sm_scale;
        if (s > m) {
            float res = expf(m - s);
#pragma unroll
            for (int i = 0; i < 16; ++i) ov[i] *= res;
            l *= res;
            m = s;
        }
        float p = expf(s - m);
        l += p;
        for (int i = 0; i < per; ++i) ov[i] += p * kv[i];
    }
    float lse_v = (l > 0.0f) ? (m + logf(l)) : INFINITY;
    float* orow = out + ((u64)row * H + h) * D;
    for (int i = 0; i < per; ++i) orow[d0 + i] = (l > 0.0f) ? (ov[i] / l) : 0.0f;
    if (lse && lane == 0) lse[(u64)row * H + h] = lse_v;
}

// ---------------------------------------------------------------------------
// Dense-causal prefill over a paged window (the cross-attention path:
// keys [xlo, xlo + xn), query offset xqoff from the cell).
// grid = (ceil(sq / ATT_WARPS), H, B).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_prefill_dense_kernel(const float* __restrict__ q,
                              const void* __restrict__ kv_pool,
                              const u64* __restrict__ tbl_ptrs,
                              const int* __restrict__ cells,
                              float* __restrict__ out,
                              int H, int D, float sm_scale, int fmt,
                              int sq, int B) {
    int b = blockIdx.z;
    if (b >= B) return;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int qrow = blockIdx.x * ATT_WARPS + (int)(threadIdx.x / DI_WARP);
    if (qrow >= sq) return;
    int h = blockIdx.y;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const int row = b * sq + qrow;
    const float* qv_ptr = q + ((u64)row * H + h) * D;
    const int* tbl = (const int*)tbl_ptrs[b];
    const int xlo = c[SV_XLO];
    const int xn = c[SV_XN];
    const int qoff = c[SV_XQOFF];
    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    for (int i = 0; i < per; ++i) qv[i] = qv_ptr[d0 + i];
    float m = -INFINITY;
    float l = 0.0f;
    int jend = qoff + qrow + 1;
    if (jend > xn) jend = xn;
    for (int j = 0; j < jend; ++j) {
        float kv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
        for (int i = 0; i < per; ++i) {
            kv[i] = sv_load_k(kv_pool, tbl, fmt, xlo + j, d0 + i, D);
        }
        float s = 0.0f;
        for (int i = 0; i < per; ++i) s += qv[i] * kv[i];
        s = di_warp_reduce_sum(s) * sm_scale;
        if (s > m) {
            float res = expf(m - s);
#pragma unroll
            for (int i = 0; i < 16; ++i) ov[i] *= res;
            l *= res;
            m = s;
        }
        float p = expf(s - m);
        l += p;
        for (int i = 0; i < per; ++i) ov[i] += p * kv[i];
    }
    float* orow = out + ((u64)row * H + h) * D;
    for (int i = 0; i < per; ++i) orow[d0 + i] = (l > 0.0f) ? (ov[i] / l) : 0.0f;
}

// ---------------------------------------------------------------------------
// SWA ring append: x rows [B*sq, hidden] into per-(slot, cross-layer) rings
// at slots (pos + i) % win. grid = (B*sq, n_cross); block 128.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
di_serve_ring_append_kernel(const float* __restrict__ x,
                            const u64* __restrict__ ring_ptrs,
                            const int* __restrict__ cells,
                            int hidden, int win, int n_cross, int sq, int B) {
    int row = blockIdx.x;
    int l = blockIdx.y;
    if (row >= B * sq || l >= n_cross) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    int slot = (c[SV_POS] + i) % (win > 0 ? win : 1);
    float* ring = (float*)ring_ptrs[(u64)l * B + b];
    float* dst = ring + (u64)slot * hidden;
    const float* src = x + (u64)row * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) dst[d] = src[d];
}

// ---------------------------------------------------------------------------
// Sliding-window self-attention over the per-slot ring (single head,
// D = hidden). CTA per (slot, query row); the ring window starts at slot
// (total - swa_n) % win and the query offset comes from the cell. Same
// two-pass online-softmax structure and accumulation order as
// di_swa_attention_dev_kernel, with the ring slot indirection added.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_serve_swa_attention_kernel(const float* __restrict__ x,
                              const u64* __restrict__ ring_ptrs,
                              const int* __restrict__ cells,
                              float* __restrict__ out,
                              int hidden, int win, float sm_scale, int sq, int B) {
    int row = blockIdx.x;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int i = row - b * sq;
    const int n = c[SV_SWA_N];
    const int start = (c[SV_TOTAL] - n) % (win > 0 ? win : 1);
    const int q_offset = c[SV_SWA_QOFF] - (sq - 1 - i);
    const float* qrow = x + (u64)row * hidden;
    const float* ring = (const float*)ring_ptrs[(u64)b * gridDim.y + blockIdx.y];
    extern __shared__ float smem[];        // [hidden] ov, then [2*blockDim] red
    float* ov = smem;
    float* red = smem + hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) ov[d] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    int jend = q_offset + i + 1;
    if (jend > n) jend = n;
    for (int j = 0; j < jend; ++j) {
        const float* krow = ring + (u64)((start + j) % (win > 0 ? win : 1)) * hidden;
        float part = 0.0f;
        for (int d = threadIdx.x; d < hidden; d += blockDim.x) part += qrow[d] * krow[d];
        red[threadIdx.x] = part;
        __syncthreads();
        for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        float s_score = red[0] * sm_scale;
        __syncthreads(); // red reused next iteration
        if (s_score > m) {
            float res = expf(m - s_score);
            for (int d = threadIdx.x; d < hidden; d += blockDim.x) ov[d] *= res;
            l *= res;
            m = s_score;
        }
        float p = expf(s_score - m);
        l += p;
        for (int d = threadIdx.x; d < hidden; d += blockDim.x) ov[d] += p * krow[d];
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* orow = out + (u64)row * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) orow[d] = ov[d] * inv;
}

// ===========================================================================
// P1-8: cp.async-pipelined and vectorized variants of the hot decode walks.
//
// Design contract (docs/NEXT-UPGRADES.md P1-8):
//   * the MLA split-KV partial and the dense prefill stage their KV rows
//     through a STAGES-deep shared-memory pipeline of 16-byte `cp.async`
//     copies — the whole CTA (8 warps = 8 heads / query rows) consumes ONE
//     staged copy of each row instead of eight redundant global walks, and
//     tile t+1's bytes are in flight while tile t is being consumed. The
//     per-row online-softmax update order (ascending ci / j) and every
//     arithmetic expression are IDENTICAL to the direct-load kernels, so
//     results are bit-identical — the golden contract the legacy kernels
//     are held to carries over unchanged. On HIP / pre-Ampere the copy
//     helper degrades to synchronous loads (same bytes, no overlap).
//   * the DSA thunder-logits and conv dot-score kernels switch from one
//     thread per key row (32-way uncoalesced row streams) to ONE WARP per
//     key row with packed vector loads (u32 for the e4m3 rows, float4 for
//     the f32 rows): every load instruction becomes a full 128-byte
//     transaction. Per-head dots stay lane-sequential in d order and merge
//     through the fixed `di_warp_reduce_sum` tree — run-to-run
//     deterministic; versus the scalar kernel the association differs, so
//     these two are tolerance-goldens (the tcgen05-vs-portable class).
// ===========================================================================

#define SV_PIPE_STAGES 3

// Stage one tile of `rows_per_stage` cached rows into `slot`.
// grid-cooperative across the whole CTA; rows with invalid indices are
// skipped by BOTH the producer and the consumer (the consumer re-reads the
// index and guards identically).
__device__ __forceinline__ void sv_stage_tile(
        const void* __restrict__ kv_pool, const int* __restrict__ tbl,
        const int* __restrict__ idxs, int c_lo, int c_hi, u8* __restrict__ slot,
        int rows_per_stage, int row_bytes) {
    const int chunks = row_bytes >> 4; // 16-byte units (row_bytes % 16 == 0)
    const int total = rows_per_stage * chunks;
    for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
        int r = i / chunks;
        int c = i - r * chunks;
        int ci = c_lo + r;
        if (ci >= c_hi) break;
        // Decode path: ci indexes the selection list (idxs != NULL).
        // Dense path: ci IS the logical row id (idxs == NULL).
        int idx = idxs ? idxs[ci] : ci;
        if (idx < 0) continue;
        const u8* src = sv_crow(kv_pool, tbl, idx, row_bytes) + c * 16;
        di_cp16(slot + r * row_bytes + c * 16, src);
    }
    di_cp_commit();
}

// ---------------------------------------------------------------------------
// MLA decode, split-KV partial, PIPELINED. Same mapping and outputs as
// di_serve_mla_decode_partial_kernel; adds `rows_per_stage` and expects
// `STAGES * rows_per_stage * row_bytes(fmt, D)` dynamic shared bytes.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_mla_decode_pipe_kernel(const float* __restrict__ q,
                                const void* __restrict__ kv_pool,
                                const u64* __restrict__ tbl_ptrs,
                                const int* __restrict__ cells,
                                const int* __restrict__ sel,
                                float* __restrict__ pm,
                                float* __restrict__ pl,
                                float* __restrict__ po,
                                int H, int D, int topk_max, int chunk,
                                float sm_scale, int fmt, int sq, int B,
                                int rows_per_stage) {
    int split = blockIdx.x;
    int splits = gridDim.x;
    int hbase = blockIdx.y * ATT_WARPS;
    int row = blockIdx.z;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int warp = (int)(threadIdx.x / DI_WARP);
    int h = hbase + warp;
    // NOTE: no early return for h >= H — the tile loop's __syncthreads is
    // CTA-wide; over-range warps march through the barriers idle.
    const bool hok = h < H;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const float* qv_ptr = q + ((u64)row * H + h) * D;
    const int* selrow = sel + (u64)row * topk_max;
    const int* tbl = (const int*)tbl_ptrs[b];
    const int rb = sv_row_bytes(fmt, D);
    extern __shared__ u8 sv_sm[]; // byte view of the dynamic smem
    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    if (hok) {
        for (int i = 0; i < per; ++i) qv[i] = qv_ptr[d0 + i];
    }
    float m = -INFINITY;
    float l = 0.0f;
    const int c0 = split * chunk;
    const int c1 = min(c0 + chunk, c[SV_TOPK]);
    const int R = rows_per_stage;
    const int ntiles = (max(c1 - c0, 0) + R - 1) / R;
    // Prologue: fill STAGES-1 stages.
    for (int t = 0; t < SV_PIPE_STAGES - 1 && t < ntiles; ++t) {
        sv_stage_tile(kv_pool, tbl, selrow, c0 + t * R, c1,
                      sv_sm + t * R * rb, R, rb);
    }
    for (int t = 0; t < ntiles; ++t) {
        // The in-flight budget only holds when a NEWER group exists; the
        // final tile must drain completely (wait<0>).
        if (t + SV_PIPE_STAGES - 1 < ntiles) di_cp_wait<SV_PIPE_STAGES - 2>();
        else di_cp_wait<0>();
        __syncthreads();
        const u8* slot = sv_sm + (t % SV_PIPE_STAGES) * R * rb;
        const int lo = c0 + t * R;
        const int hi = min(lo + R, c1);
        if (hok) {
            for (int ci = lo; ci < hi; ++ci) {
                int idx = selrow[ci];
                if (idx < 0) continue;
                const u8* srow = slot + (ci - lo) * rb;
                float kv[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
#pragma unroll 2
                for (int i = 0; i < per; ++i) kv[i] = sv_deq_k(srow, fmt, d0 + i, D);
                float s = 0.0f;
                for (int i = 0; i < per; ++i) s += qv[i] * kv[i];
                s = di_warp_reduce_sum(s) * sm_scale;
                if (s > m) {
                    float res = expf(m - s);
#pragma unroll
                    for (int i = 0; i < 16; ++i) ov[i] *= res;
                    l *= res;
                    m = s;
                }
                float p = expf(s - m);
                l += p;
                for (int i = 0; i < per; ++i) ov[i] += p * kv[i];
            }
        }
        __syncthreads();
        int nt = t + SV_PIPE_STAGES - 1;
        if (nt < ntiles) {
            sv_stage_tile(kv_pool, tbl, selrow, c0 + nt * R, c1,
                          sv_sm + (nt % SV_PIPE_STAGES) * R * rb, R, rb);
        }
    }
    if (hok) {
        const u64 hp = ((u64)row * H + h) * splits + split;
        float* porow = po + (hp * D);
        for (int i = 0; i < per; ++i) porow[d0 + i] = ov[i];
        if (lane == 0) {
            pm[hp] = m;
            pl[hp] = l;
        }
    }
}

// ---------------------------------------------------------------------------
// Dense-causal prefill over a paged window, PIPELINED. Same mapping as
// di_serve_prefill_dense_kernel with the tile loop driven by the CTA-wide
// key bound (warps whose query row ends early still march through the
// barrier pattern; their consume loop is simply empty).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_prefill_dense_pipe_kernel(const float* __restrict__ q,
                                   const void* __restrict__ kv_pool,
                                   const u64* __restrict__ tbl_ptrs,
                                   const int* __restrict__ cells,
                                   float* __restrict__ out,
                                   int H, int D, float sm_scale, int fmt,
                                   int sq, int B, int rows_per_stage) {
    int b = blockIdx.z;
    if (b >= B) return;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    int qrow = blockIdx.x * ATT_WARPS + (int)(threadIdx.x / DI_WARP);
    int h = blockIdx.y;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const int row = b * sq + qrow;
    const float* qv_ptr = q + ((u64)row * H + h) * D;
    const int* tbl = (const int*)tbl_ptrs[b];
    const int xlo = c[SV_XLO];
    const int xn = c[SV_XN];
    const int qoff = c[SV_XQOFF];
    const int rb = sv_row_bytes(fmt, D);
    extern __shared__ u8 sv_sm[]; // byte view of the dynamic smem
    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    // CTA-wide key bound (uniform so every warp runs the same barrier count).
    int qlast = blockIdx.x * ATT_WARPS + ATT_WARPS;
    if (qlast > sq) qlast = sq;
    int jmax = qoff + qlast;
    if (jmax > xn) jmax = xn;
    // Per-warp visibility: lazy qv load (a warp past sq never touches memory).
    if (qrow < sq) {
        for (int i = 0; i < per; ++i) qv[i] = qv_ptr[d0 + i];
    }
    float m = -INFINITY;
    float l = 0.0f;
    int jend = qoff + qrow + 1;
    if (jend > xn) jend = xn;
    const int R = rows_per_stage;
    const int ntiles = (jmax + R - 1) / R;
    for (int t = 0; t < SV_PIPE_STAGES - 1 && t < ntiles; ++t) {
        sv_stage_tile(kv_pool, tbl, (const int*)0, xlo + t * R, xlo + jmax,
                      sv_sm + t * R * rb, R, rb);
    }
    for (int t = 0; t < ntiles; ++t) {
        // The in-flight budget only holds when a NEWER group exists; the
        // final tile must drain completely (wait<0>).
        if (t + SV_PIPE_STAGES - 1 < ntiles) di_cp_wait<SV_PIPE_STAGES - 2>();
        else di_cp_wait<0>();
        __syncthreads();
        const u8* slot = sv_sm + (t % SV_PIPE_STAGES) * R * rb;
        const int lo = t * R;
        const int hi = min(lo + R, jmax);
        if (qrow < sq) {
            for (int j = lo; j < hi; ++j) {
                if (j >= jend) break;
                const u8* srow = slot + (j - lo) * rb;
                float kv[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
#pragma unroll 2
                for (int i = 0; i < per; ++i) kv[i] = sv_deq_k(srow, fmt, d0 + i, D);
                float s = 0.0f;
                for (int i = 0; i < per; ++i) s += qv[i] * kv[i];
                s = di_warp_reduce_sum(s) * sm_scale;
                if (s > m) {
                    float res = expf(m - s);
#pragma unroll
                    for (int i = 0; i < 16; ++i) ov[i] *= res;
                    l *= res;
                    m = s;
                }
                float p = expf(s - m);
                l += p;
                for (int i = 0; i < per; ++i) ov[i] += p * kv[i];
            }
        }
        __syncthreads();
        int nt = t + SV_PIPE_STAGES - 1;
        if (nt < ntiles) {
            sv_stage_tile(kv_pool, tbl, (const int*)0, xlo + nt * R, xlo + jmax,
                          sv_sm + (nt % SV_PIPE_STAGES) * R * rb, R, rb);
        }
    }
    if (qrow < sq) {
        float* orow = out + ((u64)row * H + h) * D;
        for (int i = 0; i < per; ++i) orow[d0 + i] = (l > 0.0f) ? (ov[i] / l) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Thunder logits, VECTORIZED: one warp per key row. The e4m3 row is read in
// aligned u32 packs (four dequants per load, fully coalesled across the
// warp); the f32 tail rows (not yet lightning-quantized) are read as
// float4s straight from the pool. Per-head dots stay d-ascending per lane
// and merge through the fixed warp tree; the head sum keeps h-ascending
// order. Deterministic; tolerance-golden vs the scalar kernel.
// grid = (ceil(s_kv / (ATT_WARPS)), B*sq); block = ATT_WARPS * DI_WARP.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_thunder_logits_vec_kernel(const float* __restrict__ q_f32,
                                   const u8* __restrict__ q_u8,
                                   const u8* __restrict__ q_sf,
                                   const void* __restrict__ k_norm_pool,
                                   const void* __restrict__ k_u8_pool,
                                   const u64* __restrict__ tbl_ptrs,
                                   const u64* __restrict__ sf_ptrs,
                                   const int* __restrict__ cells,
                                   const float* __restrict__ w,
                                   float* __restrict__ scores,
                                   int nh, int ihd, int qb, int q_nblocks,
                                   int max_cols, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    const int dim = nh * ihd;
    extern __shared__ float sq_f32[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        if (row / qb < q_nblocks) {
            float scale = di_e8m0_to_f32(q_sf[row / qb]);
            sq_f32[d] = di_e4m3_to_f32(q_u8[(u64)row * dim + d]) * scale;
        } else {
            sq_f32[d] = q_f32[(u64)row * dim + d];
        }
    }
    __syncthreads();
    const int s_kv = c[SV_TOTAL];
    const int knb = c[SV_KNB];
    const int* tbl = (const int*)tbl_ptrs[b];
    const u8* ksf = (const u8*)sf_ptrs[b];
    const float* wrow = w + (u64)row * nh;
    float* outrow = scores + (u64)row * max_cols;
    int warp = (int)(threadIdx.x / DI_WARP);
    int lane = di_lane_id();
    const int jstride = (int)(blockDim.x / DI_WARP); // warps per CTA
    for (int j = blockIdx.x * jstride + warp; j < s_kv; j += gridDim.x * jstride) {
        bool kq = (j / qb) < knb;
        float acc = 0.0f;
        if (kq) {
            float kscale = di_e8m0_to_f32(ksf[j / qb]);
            const u8* ku8 = (const u8*)sv_crow(k_u8_pool, tbl, j, dim);
            for (int h = 0; h < nh; ++h) {
                int base = h * ihd;
                float dot = 0.0f;
                if ((ihd & 3) == 0) {
                    // Packed u32 walk of this lane's slice, d-ascending
                    // (base+d stays 4-byte aligned: ihd % 4 == 0).
                    for (int d = lane * 4; d + 3 < ihd; d += DI_WARP * 4) {
                        u32 pack = *(const u32*)(ku8 + base + d);
                        dot += sq_f32[base + d] * (di_e4m3_to_f32((u8)(pack & 0xFFu)) * kscale);
                        dot += sq_f32[base + d + 1] * (di_e4m3_to_f32((u8)((pack >> 8) & 0xFFu)) * kscale);
                        dot += sq_f32[base + d + 2] * (di_e4m3_to_f32((u8)((pack >> 16) & 0xFFu)) * kscale);
                        dot += sq_f32[base + d + 3] * (di_e4m3_to_f32((u8)((pack >> 24) & 0xFFu)) * kscale);
                    }
                } else {
                    for (int d = lane; d < ihd; d += DI_WARP) {
                        dot += sq_f32[base + d] * (di_e4m3_to_f32(ku8[base + d]) * kscale);
                    }
                }
                acc += (dot > 0.0f ? dot : 0.0f) * wrow[h];
            }
        } else {
            const float* kf32 = (const float*)sv_crow(k_norm_pool, tbl, j, dim * 4);
            for (int h = 0; h < nh; ++h) {
                int base = h * ihd;
                float dot = 0.0f;
                for (int d = lane; d < ihd; d += DI_WARP) {
                    dot += sq_f32[base + d] * kf32[base + d];
                }
                acc += (dot > 0.0f ? dot : 0.0f) * wrow[h];
            }
        }
        float tot = di_warp_reduce_sum(acc);
        if (lane == 0) outrow[j] = tot;
    }
}

// ---------------------------------------------------------------------------
// Conv-branch scores, VECTORIZED: one warp per feat row, float4 loads.
// Deterministic (lane-sliced d-ascending partials + fixed warp tree);
// tolerance-golden vs the scalar kernel.
// grid = (ceil(s_kv / (ATT_WARPS)), B*sq); block = ATT_WARPS * DI_WARP.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_serve_dot_scores_vec_kernel(const float* __restrict__ q,
                               const void* __restrict__ feat_pool,
                               const u64* __restrict__ tbl_ptrs,
                               const int* __restrict__ cells,
                               float* __restrict__ out,
                               int hidden, int max_cols, int sq, int B) {
    int row = blockIdx.y;
    if (row >= B * sq) return;
    int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    const int* tbl = (const int*)tbl_ptrs[b];
    const float* qi = q + (u64)row * hidden;
    float* outrow = out + (u64)row * max_cols;
    int warp = (int)(threadIdx.x / DI_WARP);
    int lane = di_lane_id();
    const int jstride = (int)(blockDim.x / DI_WARP);
    for (int j = blockIdx.x * jstride + warp; j < c[SV_TOTAL];
         j += gridDim.x * jstride) {
        const float* fj = (const float*)sv_crow(feat_pool, tbl, j, hidden * 4);
        float acc = 0.0f;
        if ((hidden & 3) == 0) {
            // float4 walk (row starts and lane offsets stay 16-byte
            // aligned: hidden % 4 == 0).
            for (int d = lane * 4; d + 3 < hidden; d += DI_WARP * 4) {
                float4 v = *(const float4*)(fj + d);
                float4 qq = *(const float4*)(qi + d);
                acc += qq.x * v.x;
                acc += qq.y * v.y;
                acc += qq.z * v.z;
                acc += qq.w * v.w;
            }
        } else {
            for (int d = lane; d < hidden; d += DI_WARP) {
                acc += qi[d] * fj[d];
            }
        }
        float tot = di_warp_reduce_sum(acc);
        if (lane == 0) outrow[j] = tot;
    }
}

// ===========================================================================
// P1-7: the persistent self-layer block megakernel (phase 1).
//
// ONE launch replaces the per-layer chain
//     rmsnorm -> GEMV(fused_qkv) -> copy_cols x2 -> latent-norm -> rope(kv)
//     -> paged KV encode -> GEMV(q_b) -> rope(q)
// (8 kernels + 4 global round-trips of the intermediate rows) with a
// single CTA per (row, layer) that keeps h/x/qa/kv resident in shared
// memory and streams the two weight matrices through a 2-deep cp.async
// tile pipeline. The BS <= rows regime is the target (SGLang's megakernel
// win is single-user decode); the unfused path stays the default and this
// kernel is selected by DEEPINFER_MEGAKERNEL=1.
//
// Weight dtypes (wdtype): 0 = f32 [n,k], 1 = bf16 [n,k], 2 = fp8
// (e4m3 [n,k] + ue8m0 [n/128, k/128] blocks — the DeepGEMM recipe, with
// the ACTIVATION quantized per-token-group in-kernel exactly like the
// reference path), 3 = nvfp4 (e2m1 nibbles [n, k/2] even-low + e4m3
// [n, k/16] scales, activation f32).
//
// Determinism: every reduction is a fixed block/warp tree; every GEMV
// accumulates k-ascending per output. Versus the tiled GEMM kernels the
// association differs, so outputs are tolerance-goldens (the
// tcgen05-vs-portable class), pinned by golden_run's mega-vs-graph entry.
//
// grid = (rows, n_layers); block = 256. Pointer tables are indexed by
// blockIdx.y (layer). Requires: hidden % 4 == 0, q_lora % 16 == 0 (fp8
// group math), latent % 2 == 0, hd % 32 == 0 — the launcher checks and
// falls back to the unfused chain otherwise.
// ===========================================================================

#define MEGA_KQ 128 // k-tile width (elements)

// Stage one [tile_n x MEGA_KQ] weight tile (native dtype bytes, +16 B row
// pad) into `dst` from W[n0.., k0..]. Grid-cooperative; masked rows/cols
// are skipped (the consumer masks identically).
__device__ __forceinline__ void mega_stage_w(
        const u8* __restrict__ w, long long w_row_bytes, int n0, int n_out,
        int k0, int k_len, u8* __restrict__ dst, int tile_n, int elt) {
    const int row_bytes = MEGA_KQ * elt + 16;
    const int chunks = (MEGA_KQ * elt) >> 4; // full 16B chunks per row
    const int total = tile_n * chunks;
    for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
        int r = i / chunks;
        int c = i - r * chunks;
        int n = n0 + r;
        int ke = k0 + MEGA_KQ;
        if (ke > k_len) ke = k_len;
        int valid = (ke - k0) * elt; // bytes of this row in-tile
        if (n >= n_out || (c << 4) >= valid) continue;
        const u8* src = w + (long long)n * w_row_bytes + (long long)k0 * elt
                        + (c << 4);
        di_cp16(dst + r * row_bytes + (c << 4), src);
    }
    di_cp_commit();
}

// Decode one staged weight element (row bytes at native dtype).
__device__ __forceinline__ float mega_w_dec(const u8* __restrict__ wrow,
                                            const u8* __restrict__ sfb,
                                            int i, int wdtype, int k0) {
    if (wdtype == 0) return ((const float*)wrow)[i];
    if (wdtype == 1) return di_bf16_to_f32(((const u16*)wrow)[i]);
    if (wdtype == 2) {
        // sf byte: [(n/128) * (k/128) + (k0+i)/128] — caller passes sfb
        // pre-offset to this row's n-block.
        return di_e4m3_to_f32(wrow[i]) * di_e8m0_to_f32(sfb[(k0 + i) >> 7]);
    }
    // nvfp4: even dim = LOW nibble; one e4m3 scale per 16 dims.
    u8 b = wrow[(k0 + i) >> 1];
    u8 nib = (((k0 + i) & 1) == 0) ? (u8)(b & 0x0Fu) : (u8)(b >> 4);
    return di_e2m1_to_f32(nib) * di_e4m3_to_f32(sfb[(k0 + i) >> 4]);
}

extern "C" __global__ void __launch_bounds__(256)
di_mega_self_block_kernel(const float* __restrict__ h,
                          const unsigned long long* __restrict__ norm_ptrs,
                          const unsigned long long* __restrict__ qkv_ptrs,
                          const unsigned long long* __restrict__ qkv_sf_ptrs,
                          const unsigned long long* __restrict__ kvn_ptrs,
                          const unsigned long long* __restrict__ qb_ptrs,
                          const unsigned long long* __restrict__ qb_sf_ptrs,
                          const unsigned long long* __restrict__ pool_ptrs,
                          const u64* __restrict__ tbl_ptrs,
                          const int* __restrict__ cells,
                          const float* __restrict__ rope_cs,
                          const u32* __restrict__ pos_kv,
                          const u32* __restrict__ pos_q,
                          float* __restrict__ q_out,
                          int hidden, int q_lora, int hd, int latent,
                          int half, int n_heads, int rows, int sq, int B,
                          int wdtype, int fmt, float eps, int tile_n) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int l = blockIdx.y;
    const int b = row / sq;
    const int* c = cells + (u64)b * SV_COUNT;
    if (!c[SV_ACTIVE]) return;
    const int i = row - b * sq;

    const float* gain = (const float*)norm_ptrs[l];
    const u8* w_qkv = (const u8*)qkv_ptrs[l];
    const u8* sf_qkv = (const u8*)qkv_sf_ptrs[l];
    const float* kvn = (const float*)kvn_ptrs[l];
    const u8* w_qb = (const u8*)qb_ptrs[l];
    const u8* sf_qb = (const u8*)qb_sf_ptrs[l];
    const u8* pool = (const u8*)pool_ptrs[l];
    const int* tbl = (const int*)tbl_ptrs[b];

    const int fused_n = q_lora + hd;
    const int elt = (wdtype == 0) ? 4 : ((wdtype == 1) ? 2 : 1);

    extern __shared__ u8 msm[]; // byte view (the SWA kernels own `smem`)
    float* fused_s = (float*)msm;                 // [q_lora + hd]
    float* x_tile = (float*)(msm + fused_n * 4);  // [128] A staging
    u8* xq = (u8*)x_tile;                         // fp8 A quant (aliases x_tile)
    u8* qa_u8 = msm + fused_n * 4 + 512;          // fp8 qa quant [q_lora]
    u8* wtile = msm + fused_n * 4 + 512
                + ((wdtype == 2)
                       ? (((q_lora + ((q_lora + 127) >> 7)) + 15) & ~15)
                       : 0);
    const int w_row_bytes = MEGA_KQ * elt + 16;
    const int stage_bytes = tile_n * w_row_bytes;

    __shared__ float red[258];
    __shared__ float s_inv;

    // ---- 1) RMSNorm statistics over h[row] (fixed 256 tree; the same
    //          shape as di_rmsnorm_kernel). x is normalized on the fly
    //          into the per-k-tile staging buffer — it never round-trips
    //          through global memory. ------------------------------------
    const float* hrow = h + (u64)row * hidden;
    {
        float part = 0.0f;
        for (int j = (int)threadIdx.x; j < hidden; j += 256) {
            part += hrow[j] * hrow[j];
        }
        red[threadIdx.x] = part;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            s_inv = rsqrtf(red[0] / (float)hidden + eps);
        }
        __syncthreads();
    }

    // ---- 2) GEMV1: fused[row][n] = sum_k x[k] * W_qkv[n, k].
    //          Tiles: [tile_n x 128] double-buffered; the A tile is
    //          (re)staged per k-tile. -------------------------------------
    for (int n0 = 0; n0 < fused_n; n0 += tile_n) {
        const int nrem = fused_n - n0 < tile_n ? fused_n - n0 : tile_n;
        // Per-thread accumulators: thread t owns output n0 + t.
        float acc = 0.0f;
        const int n_mine = n0 + (int)threadIdx.x;
        const bool have_n = (int)threadIdx.x < nrem;
        // Per-thread weight-scale row base (fp8: one ue8m0 row per
        // 128x128 block; nvfp4: one e4m3 row per output row). Clamped so
        // tail threads never form out-of-range pointers.
        int n_sf = n_mine < fused_n ? n_mine : (fused_n - 1);
        const u8* sf_row =
            (wdtype == 2)
                ? sf_qkv + (long long)(n_sf >> 7) * ((hidden + 127) >> 7)
                : ((wdtype == 3)
                       ? sf_qkv + (long long)n_sf * ((hidden + 15) >> 4)
                       : sf_qkv);
        int ntiles = (hidden + MEGA_KQ - 1) / MEGA_KQ;
        // Prologue: stage W tile 0.
        mega_stage_w(w_qkv, (long long)hidden * elt, n0, fused_n, 0, hidden,
                     wtile, tile_n, elt);
        for (int t = 0; t < ntiles; ++t) {
            // Stage W tile t+1 while consuming t.
            if (t + 1 < ntiles) {
                mega_stage_w(w_qkv, (long long)hidden * elt, n0, fused_n,
                             (t + 1) * MEGA_KQ, hidden,
                             wtile + ((t + 1) & 1) * stage_bytes, tile_n, elt);
            }
            const int k0 = t * MEGA_KQ;
            int krem = hidden - k0 < MEGA_KQ ? hidden - k0 : MEGA_KQ;
            // A staging: normalize (and for fp8, quantize) h[k0..).
            if (wdtype == 2) {
                // Per-token-group quant of the 128-slice (warp w owns
                // group w%... : one warp per 128-group == one warp per
                // k-tile; lanes stride the group like the reference).
                int gw = (int)(threadIdx.x / DI_WARP);
                if (gw == 0) {
                    float amax = 0.0f;
                    for (int j = di_lane_id(); j < krem; j += DI_WARP) {
                        float v = hrow[k0 + j] * s_inv * gain[k0 + j];
                        if (isnan(v)) v = 0.0f;
                        amax = fmaxf(amax, fabsf(v));
                    }
                    amax = di_warp_reduce_max(amax);
                    float ratio = amax / 448.0f;
                    u8 sfb = 0x00u;
                    if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
                    __shared__ u8 s_xsf;
                    if (di_lane_id() == 0) s_xsf = sfb;
                    __syncwarp();
                    float scale = di_e8m0_to_f32(s_xsf);
                    for (int j = di_lane_id(); j < krem; j += DI_WARP) {
                        float v = hrow[k0 + j] * s_inv * gain[k0 + j];
                        if (isnan(v)) v = 0.0f;
                        xq[j] = di_f32_to_e4m3(v / scale, true);
                    }
                    if (di_lane_id() == 0) xq[128] = sfb;
                }
            } else {
                for (int j = (int)threadIdx.x; j < krem; j += 256) {
                    x_tile[j] = hrow[k0 + j] * s_inv * gain[k0 + j];
                }
            }
            if (t + 1 < ntiles) di_cp_wait<1>();
            else di_cp_wait<0>();
            __syncthreads();
            const u8* wr = wtile + (t & 1) * stage_bytes
                           + (int)threadIdx.x * w_row_bytes;
            if (have_n) {
                if (wdtype == 2) {
                    float xs = di_e8m0_to_f32(xq[128]);
                    for (int j = 0; j < krem; ++j) {
                        acc += (di_e4m3_to_f32(xq[j]) * xs)
                               * (di_e4m3_to_f32(wr[j])
                                  * di_e8m0_to_f32(sf_row[(k0 + j) >> 7]));
                    }
                } else {
                    for (int j = 0; j < krem; ++j) {
                        acc += x_tile[j] * mega_w_dec(wr, sf_row, j, wdtype, k0);
                    }
                }
            }
            __syncthreads();
        }
        if (have_n) fused_s[n_mine] = acc;
        __syncthreads();
    }

    // ---- 3) Latent RMSNorm over fused_s[q_lora .. q_lora+latent). ------
    {
        float part = 0.0f;
        for (int j = (int)threadIdx.x; j < latent; j += 256) {
            float v = fused_s[q_lora + j];
            part += v * v;
        }
        red[threadIdx.x] = part;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        float inv = rsqrtf(red[0] / (float)latent + 1e-6f);
        for (int j = (int)threadIdx.x; j < latent; j += 256) {
            fused_s[q_lora + j] *= inv * kvn[j];
        }
        __syncthreads();
    }

    // ---- 4) RoPE the kv tail (pairs (latent+2j, +1), DeepSeek
    //          interleaved convention). -----------------------------------
    {
        u32 p = pos_kv[row];
        for (int j = (int)threadIdx.x; j < half; j += 256) {
            float cc = rope_cs[((u64)p * half + j) << 1];
            float ss = rope_cs[(((u64)p * half + j) << 1) + 1];
            int i0 = q_lora + latent + 2 * j;
            float a = fused_s[i0];
            float b2 = fused_s[i0 + 1];
            fused_s[i0] = a * cc - b2 * ss;
            fused_s[i0 + 1] = a * ss + b2 * cc;
        }
        __syncthreads();
    }

    // ---- 5) Paged KV encode (f32 / FP8 528 B / FP4 288 B; group amax is
    //          exact so the bytes match di_serve_kv_encode_kernel). ------
    {
        const int rb = (fmt == 0) ? hd * 4 : (fmt == 1 ? 528 : 288);
        u8* dst = sv_row((void*)pool, tbl, c[SV_ROW0] + i, rb);
        const float* xs = fused_s + q_lora;
        if (fmt == 0) {
            for (int j = (int)threadIdx.x; j < hd; j += 256) {
                ((float*)dst)[j] = xs[j];
            }
        } else {
            const int gsz = (fmt == 1) ? 32 : 16;
            const int groups = hd / gsz;
            for (int g = (int)threadIdx.x; g < groups; g += 256) {
                const int g0 = g * gsz;
                float amax = 0.0f;
                for (int t = 0; t < gsz; ++t) {
                    float v = xs[g0 + t];
                    if (isnan(v)) v = 0.0f;
                    amax = fmaxf(amax, fabsf(v));
                }
                if (fmt == 1) {
                    float ratio = amax / 448.0f;
                    u8 sfb = 0x00u;
                    if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
                    float scale = di_e8m0_to_f32(sfb);
                    for (int t = 0; t < gsz; ++t) {
                        float v = xs[g0 + t];
                        if (isnan(v)) v = 0.0f;
                        dst[g0 + t] = di_f32_to_e4m3(v / scale, true);
                    }
                    dst[hd + g] = sfb;
                } else {
                    float sv = amax / 6.0f;
                    if (sv < 1.953125e-3f) sv = 1.953125e-3f;
                    if (sv > 448.0f) sv = 448.0f;
                    u8 sb = di_f32_to_e4m3(sv, true);
                    float scale = di_e4m3_to_f32(sb);
                    for (int t = 0; t < gsz; ++t) {
                        float v = xs[g0 + t];
                        if (isnan(v)) v = 0.0f;
                        u8 nib = di_f32_to_e2m1(v / scale);
                        int idx = g0 + t;
                        if ((idx & 1) == 0) dst[idx >> 1] = nib;
                        else dst[idx >> 1] |= (u8)(nib << 4);
                    }
                    dst[256 + g] = sb;
                }
            }
        }
        __syncthreads();
    }

    // ---- 6) fp8: quantize qa per-token-group for GEMV2 (the reference
    //          path quantizes every GEMM's activation). -------------------
    if (wdtype == 2) {
        int groups = (q_lora + 127) / 128;
        for (int g = (int)(threadIdx.x / DI_WARP); g < groups;
             g += (int)(blockDim.x / DI_WARP)) {
            const int g0 = g * 128;
            const int gren = q_lora - g0 < 128 ? q_lora - g0 : 128;
            float amax = 0.0f;
            for (int j = di_lane_id(); j < gren; j += DI_WARP) {
                float v = fused_s[g0 + j];
                if (isnan(v)) v = 0.0f;
                amax = fmaxf(amax, fabsf(v));
            }
            amax = di_warp_reduce_max(amax);
            float ratio = amax / 448.0f;
            u8 sfb = 0x00u;
            if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
            float scale = di_e8m0_to_f32(sfb);
            for (int j = di_lane_id(); j < gren; j += DI_WARP) {
                float v = fused_s[g0 + j];
                if (isnan(v)) v = 0.0f;
                qa_u8[g0 + j] = di_f32_to_e4m3(v / scale, true);
            }
            if (di_lane_id() == 0) qa_u8[q_lora + g] = sfb;
        }
        __syncthreads();
    }

    // ---- 7) GEMV2 + RoPE: q[row][n] for n in [0, n_heads*hd); the rope
    //          pair (2m, 2m+1) lives in adjacent threads — one shuffle
    //          exchanges the partner value. -------------------------------
    {
        const int n_out = n_heads * hd;
        const int qb_kb = q_lora;
        for (int n0 = 0; n0 < n_out; n0 += tile_n) {
            const int nrem = n_out - n0 < tile_n ? n_out - n0 : tile_n;
            float acc = 0.0f;
            const int n_mine = n0 + (int)threadIdx.x;
            const bool have_n = (int)threadIdx.x < nrem;
            int n_sf = n_mine < n_out ? n_mine : (n_out - 1);
            const u8* sf_mine =
                (wdtype == 2)
                    ? sf_qb + (long long)(n_sf >> 7) * ((qb_kb + 127) >> 7)
                    : ((wdtype == 3)
                           ? sf_qb + (long long)n_sf * ((qb_kb + 15) >> 4)
                           : sf_qb);
            int ntiles = (qb_kb + MEGA_KQ - 1) / MEGA_KQ;
            mega_stage_w(w_qb, (long long)qb_kb * elt, n0, n_out, 0, qb_kb,
                         wtile, tile_n, elt);
            for (int t = 0; t < ntiles; ++t) {
                if (t + 1 < ntiles) {
                    mega_stage_w(w_qb, (long long)qb_kb * elt, n0, n_out,
                                 (t + 1) * MEGA_KQ, qb_kb,
                                 wtile + ((t + 1) & 1) * stage_bytes, tile_n,
                                 elt);
                }
                const int k0 = t * MEGA_KQ;
                int krem = qb_kb - k0 < MEGA_KQ ? qb_kb - k0 : MEGA_KQ;
                if (t + 1 < ntiles) di_cp_wait<1>();
                else di_cp_wait<0>();
                __syncthreads();
                const u8* wr = wtile + (t & 1) * stage_bytes
                               + (int)threadIdx.x * w_row_bytes;
                if (have_n) {
                    if (wdtype == 2) {
                        for (int j = 0; j < krem; ++j) {
                            int g = (k0 + j) >> 7;
                            acc += (di_e4m3_to_f32(qa_u8[k0 + j])
                                    * di_e8m0_to_f32(qa_u8[q_lora + g]))
                                   * (di_e4m3_to_f32(wr[j])
                                      * di_e8m0_to_f32(sf_mine[(k0 + j) >> 7]));
                        }
                    } else {
                        for (int j = 0; j < krem; ++j) {
                            acc += fused_s[k0 + j]
                                   * mega_w_dec(wr, sf_mine, j, wdtype, k0);
                        }
                    }
                }
                __syncthreads();
            }
            // EVERY lane joins the shuffle (a rope pair lives in adjacent
            // lanes; a non-participating partner would be UB).
            float partner = __shfl_xor_sync(DI_FULL_MASK, acc, 1);
            if (!have_n) {
                // tail lanes of the last tile: nothing to write
            } else {
                const int n = n_mine;
                const int in_head = n % hd;
                float* qrow = q_out + (u64)row * n_out;
                if (in_head >= latent && ((in_head - latent) & 1) == 0) {
                    // Even element of a rope pair: the odd partner's value
                    // arrived through the shuffle; write both elements.
                    int j = (in_head - latent) >> 1;
                    int head = n / hd;
                    u32 p = pos_q[(u64)row * n_heads + head];
                    float cc = rope_cs[((u64)p * half + j) << 1];
                    float ss = rope_cs[(((u64)p * half + j) << 1) + 1];
                    float a = acc;
                    float b2 = partner;
                    qrow[n] = a * cc - b2 * ss;
                    qrow[n + 1] = a * ss + b2 * cc;
                } else if (in_head < latent) {
                    qrow[n] = acc;
                }
                // Odd rope elements: the even partner wrote both.
            }
        }
    }
}
