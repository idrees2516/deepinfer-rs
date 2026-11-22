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
