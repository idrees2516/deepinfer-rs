// decode.cu — the graph-replayable decode kernel family.
//
// A CUDA/HIP graph bakes launch geometry AND scalar arguments at capture
// time, but a decode step's scalars change every token: the sequence grows,
// cache write rows advance, attention windows slide. The production answer
// (the same one FlashAttention-style kernels give for `seq_lens`) is to
// make every varying scalar live in DEVICE MEMORY and have the kernels
// read it. This file is that family:
//
//   * `di_step_params_kernel`   — one host-written int (the position being
//                                  decoded) fans out into the whole
//                                  [DecodeParams] cell block;
//   * `di_pos_fill_kernel`      — RoPE position rows for q/kv;
//   * `di_kv_encode_*_dev`      — append one quantized KV row at a
//                                  device-read row index;
//   * `di_append_row_kernel`    — append one f32 row (indexer k) at a
//                                  device-read row index;
//   * `di_append_norm_row_kernel` — append the per-head-RMSNormed copy of
//                                  one indexer k row (the thunder-side
//                                  cache);
//   * `di_lightning_*_dev`      — incremental lightning quant of the LAST
//                                  quant block only (no-ops until the block
//                                  completes; idempotent once it has — the
//                                  CPU recomputes whole blocks per call,
//                                  the device amortizes the same math);
//   * `di_thunder_logits_dev` / `di_dot_scores_dev` / `di_gate_blend_dev`
//                              — the DSA score chain over a device-read
//                                s_kv with fixed (max-sized) grids;
//   * `di_causal_window_clip_dev` — causal + extrapolation-window clip and
//                                score-tail sanitization in one pass;
//   * `di_block_preselect_dev`  — top-K preselect over device-read s_kv
//                                (the merge tree / scatter kernels are
//                                reused with fixed geometry);
//   * `di_conv_feat_dev` / `di_conv_gated_norm_dev` — the conv branch for
//                                the single new key row.
//
// Launch grids are derived from FIXED maxima (max_position_embeddings,
// config topk, expert capacity), never from the current lengths, so one
// captured graph serves the whole sequence. The eager decode path launches
// the exact same kernels, so graph replay is bit-identical to eager.



// ---------------------------------------------------------------------------
// Step-parameter fanout. Reads p_pos[0] (the position of the token being
// decoded, i.e. the old sequence length) and writes the params cell block.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64)
di_step_params_kernel(const int* __restrict__ p_pos /* [2]: {pos, start} */,
                      int* __restrict__ params,
                      int cross_window,
                      int swa_window,
                      int quant_block,
                      int topk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int pos = p_pos[0];
    int start = p_pos[1];
    int len = pos + 1;
    int cross_n = (len < cross_window) ? len : cross_window;
    int swa_n = (len < swa_window) ? len : swa_window;
    params[DP_POS] = pos;
    params[DP_START] = start;
    params[DP_LEN] = len;
    params[DP_ROW] = start + pos; // arena row of the new token
    params[DP_S_KV] = len;
    params[DP_K_NBLOCKS] = len / quant_block;
    params[DP_LAST_BLOCK] = pos / quant_block;
    params[DP_CROSS_LO] = start + len - cross_n; // first arena row of the window
    params[DP_CROSS_NKEYS] = cross_n;
    params[DP_CROSS_QOFF] = cross_n - 1;
    params[DP_SWA_N] = swa_n;
    params[DP_SWA_QOFF] = swa_n - 1;
    params[DP_TOPK] = topk;
    params[DP_SWA_START] = (len - swa_n) % swa_window; // oldest visible slot
}

// RoPE position rows: q has H rows all at the decoded position; kv has one.
extern "C" __global__ void __launch_bounds__(128)
di_pos_fill_kernel(u32* __restrict__ pos_q, u32* __restrict__ pos_kv,
                   const int* __restrict__ p_pos, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = (u32)*p_pos;
    if (i < H) pos_q[i] = pos;
    if (i == 0) pos_kv[0] = pos;
}

// ---------------------------------------------------------------------------
// Single-row KV-cache encode at a device-read row (graph decode append).
// Same numerics as the prefill encoders (one warp per 32/16-dim group).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64)
di_kv_encode_fp8_dev_kernel(const float* __restrict__ x, // [D] f32
                            u8* __restrict__ cache,
                            const int* __restrict__ p_row,
                            int D) {
    int gw = blockIdx.x * (int)(blockDim.x / DI_WARP) + di_warp_id();
    int groups = D / 32;
    if (gw >= groups) return;
    const float* xs = x + (u64)gw * 32;
    u8* row = cache + (u64)(*p_row) * (D + D / 32);
    float amax = 0.0f;
    for (int i = di_lane_id(); i < 32; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float ratio = amax / 448.0f;
    u8 sfb = 0x00u;
    if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
    float scale = di_e8m0_to_f32(sfb);
    for (int i = di_lane_id(); i < 32; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        row[gw * 32 + i] = di_f32_to_e4m3(v / scale, true);
    }
    if (di_lane_id() == 0) row[D + gw] = sfb;
}

extern "C" __global__ void __launch_bounds__(64)
di_kv_encode_fp4_dev_kernel(const float* __restrict__ x, // [D] f32
                            u8* __restrict__ cache,
                            const int* __restrict__ p_row,
                            int D) {
    int gw = blockIdx.x * (int)(blockDim.x / DI_WARP) + di_warp_id();
    int groups = D / 16;
    if (gw >= groups) return;
    const float* xs = x + (u64)gw * 16;
    u8* row = cache + (u64)(*p_row) * (D / 2 + D / 16);
    float amax = 0.0f;
    for (int i = di_lane_id(); i < 16; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float sv = amax / 6.0f;
    if (sv < 1.953125e-3f) sv = 1.953125e-3f;
    if (sv > 448.0f) sv = 448.0f;
    u8 sb = di_f32_to_e4m3(sv, true);
    float scale = di_e4m3_to_f32(sb);
    for (int i = di_lane_id(); i < 16; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        u8 nib = di_f32_to_e2m1(v / scale);
        int idx = gw * 16 + i;
        if ((idx & 1) == 0) {
            row[idx >> 1] = nib;
        } else {
            row[idx >> 1] |= (u8)(nib << 4);
        }
    }
    if (di_lane_id() == 0) row[D / 2 + gw] = sb;
}

// Append one f32 row at a device-read row index (indexer k cache).
extern "C" __global__ void __launch_bounds__(256)
di_append_row_kernel(const float* __restrict__ src,
                     float* __restrict__ dst,
                     const int* __restrict__ p_row,
                     int floats) {
    float* row = dst + (u64)(*p_row) * floats;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < floats;
         i += gridDim.x * blockDim.x) {
        row[i] = src[i];
    }
}

// Append the per-(token, head) RMSNormed copy of one indexer k row (the
// thunder-side k cache; the conv branch keeps reading the raw cache).
extern "C" __global__ void __launch_bounds__(256)
di_append_norm_row_kernel(const float* __restrict__ src,
                          float* __restrict__ dst,
                          const int* __restrict__ p_row,
                          int nh,
                          int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nh) return;
    const float* s = src + (u64)idx * hd;
    float* row = dst + (u64)(*p_row) * nh * hd;
    float* d = row + (u64)idx * hd;
    float ms = 0.0f;
    for (int i = 0; i < hd; ++i) ms += s[i] * s[i];
    ms /= (float)hd;
    float inv = rsqrtf(ms + 1e-6f);
    for (int i = 0; i < hd; ++i) d[i] = s[i] * inv;
}

// ---------------------------------------------------------------------------
// Incremental lightning quant of the LAST quant block. `p_block` selects
// the block; `p_len` guards completion (a partial block is skipped — the
// CPU contract quantizes complete blocks only). Re-quantizing a completed
// block is idempotent (same rows, same amax), so launching every step is
// exactly equivalent to the CPU's per-call whole-history quant.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_lightning_block_zero_dev_kernel(u32* __restrict__ sf_bits,
                                   const int* __restrict__ p_block) {
    if (threadIdx.x == 0 && blockIdx.x == 0) sf_bits[*p_block] = 0u;
}

extern "C" __global__ void __launch_bounds__(256)
di_lightning_amax_dev_kernel(const float* __restrict__ x_base,
                             u32* __restrict__ sf_bits,
                             const int* __restrict__ p_block,
                             const int* __restrict__ p_len,
                             int dim,
                             int quant_block,
                             long long chunk_elems) {
    int b = *p_block;
    if (*p_len < (b + 1) * quant_block) return; // incomplete: no-op
    int chunk = blockIdx.x;
    const float* xs = x_base + (u64)b * quant_block * dim + (u64)chunk * chunk_elems;
    long long remain = (long long)quant_block * dim - (long long)chunk * chunk_elems;
    if (remain > chunk_elems) remain = chunk_elems;
    if (remain <= 0) return;
    float amax = 0.0f;
    for (long long i = threadIdx.x; i < remain; i += blockDim.x) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float red[256];
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0 && red[0] > 0.0f) {
        atomicMax(&sf_bits[b], __float_as_uint(red[0]));
    }
}

extern "C" __global__ void __launch_bounds__(256)
di_lightning_finalize_dev_kernel(const u32* __restrict__ sf_bits,
                                 u8* __restrict__ sf,
                                 const int* __restrict__ p_block) {
    int b = *p_block;
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    float amax = __uint_as_float(sf_bits[b]);
    float ratio = amax / 448.0f;
    sf[b] = (ratio > 0.0f) ? di_f32_to_e8m0_ceil(ratio) : (u8)0x00;
}

extern "C" __global__ void __launch_bounds__(256)
di_lightning_quant_dev_kernel(const float* __restrict__ x_base,
                              u8* __restrict__ q_base,
                              const u8* __restrict__ sf,
                              const int* __restrict__ p_block,
                              const int* __restrict__ p_len,
                              int dim,
                              int quant_block) {
    int b = *p_block;
    if (*p_len < (b + 1) * quant_block) return; // incomplete: no-op
    long long per_block = (long long)quant_block * dim;
    const float* xs = x_base + (u64)b * per_block;
    u8* qs = q_base + (u64)b * per_block;
    float scale = di_e8m0_to_f32(sf[b]);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < per_block;
         idx += (long long)gridDim.x * blockDim.x) {
        qs[idx] = di_f32_to_e4m3(xs[idx] / scale, true);
    }
}

// ---------------------------------------------------------------------------
// Thunder logits over a device-read s_kv. grid.x covers max_cols (fixed);
// threads beyond *p_s_kv return. Identical per-element math and accumulation
// order to `di_thunder_logits_kernel`.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_thunder_logits_dev_kernel(const float* __restrict__ q_f32,
                             const u8* __restrict__ q_u8,
                             const u8* __restrict__ q_sf,
                             const float* __restrict__ k_f32,
                             const u8* __restrict__ k_u8,
                             const u8* __restrict__ k_sf,
                             const float* __restrict__ w,
                             float* __restrict__ scores,
                             const int* __restrict__ p_s_kv,
                             const int* __restrict__ p_k_nblocks,
                             int s_q,
                             int nh,
                             int hd,
                             int q_nblocks,
                             int quant_block,
                             int max_cols) {
    int i = blockIdx.y;
    int s_kv = *p_s_kv;
    int k_nblocks = *p_k_nblocks;
    int dim = nh * hd;
    extern __shared__ float sq_f32[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        if (i / quant_block < q_nblocks) {
            int b = i / quant_block;
            float scale = di_e8m0_to_f32(q_sf[b]);
            sq_f32[d] = di_e4m3_to_f32(q_u8[(u64)i * dim + d]) * scale;
        } else {
            sq_f32[d] = q_f32[(u64)i * dim + d];
        }
    }
    __syncthreads();
    const float* wrow = w + (u64)i * nh;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < s_kv;
         j += gridDim.x * blockDim.x) {
        const u8* ku8 = k_u8 + (u64)j * dim;
        const float* kf32 = k_f32 + (u64)j * dim;
        bool kq = (j / quant_block) < k_nblocks;
        float kscale = 1.0f;
        if (kq) kscale = di_e8m0_to_f32(k_sf[j / quant_block]);
        float acc = 0.0f;
        for (int h = 0; h < nh; ++h) {
            float dot = 0.0f;
            int base = h * hd;
            if (kq) {
                for (int d = 0; d < hd; ++d) {
                    dot += sq_f32[base + d] * (di_e4m3_to_f32(ku8[base + d]) * kscale);
                }
            } else {
                for (int d = 0; d < hd; ++d) {
                    dot += sq_f32[base + d] * kf32[base + d];
                }
            }
            acc += (dot > 0.0f ? dot : 0.0f) * wrow[h];
        }
        scores[(u64)i * max_cols + j] = acc;
    }
}

// Conv-branch scores over a device-read s_kv (fixed max_cols grid).
extern "C" __global__ void __launch_bounds__(256)
di_dot_scores_dev_kernel(const float* __restrict__ q,
                         const float* __restrict__ feat,
                         float* __restrict__ out,
                         const int* __restrict__ p_s_kv,
                         int s_q,
                         int hidden,
                         int max_cols) {
    int i = blockIdx.y;
    int s_kv = *p_s_kv;
    const float* qi = q + (u64)i * hidden;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < s_kv;
         j += gridDim.x * blockDim.x) {
        const float* fj = feat + (u64)j * hidden;
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) acc += qi[d] * fj[d];
        out[(u64)i * max_cols + j] = acc;
    }
}

// Gate blend over fixed max_cols; j >= *p_s_kv entries are sanitized to
// -inf (the score tail must read as masked to the fixed-geometry
// preselect, whose blocks cover the whole max_cols range).
extern "C" __global__ void __launch_bounds__(256)
di_gate_blend_dev_kernel(const float* __restrict__ gt,
                         const float* __restrict__ gc,
                         const float* __restrict__ conv,
                         float* __restrict__ thunder,
                         const int* __restrict__ p_s_kv,
                         int s_q,
                         int max_cols) {
    int s_kv = *p_s_kv;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < s_q * max_cols;
         idx += gridDim.x * blockDim.x) {
        int i = idx / max_cols;
        int j = idx - i * max_cols;
        if (j >= s_kv) {
            thunder[idx] = -INFINITY;
            continue;
        }
        float th = thunder[idx];
        float cv = conv[idx];
        if (th == -INFINITY || cv == -INFINITY) {
            thunder[idx] = -INFINITY;
        } else {
            float denom = gt[i] + gc[i];
            thunder[idx] = (denom > 0.0f) ? ((gt[i] * th + gc[i] * cv) / denom) : 0.0f;
        }
    }
}

// Causal + extrapolation-window clip with tail sanitization, over the fixed
// max_cols width. window == 0 disables the extrapolation restriction.
// Causality is the crate contract `j > i` (see docs/GPU.md — the reference
// quirk note): query i is the i-th row of the current chunk, key j is the
// j-th cached row.
extern "C" __global__ void __launch_bounds__(256)
di_causal_window_clip_dev_kernel(float* __restrict__ scores,
                                 const int* __restrict__ p_s_kv,
                                 int s_q,
                                 int max_cols,
                                 int window) {
    int s_kv = *p_s_kv;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < s_q * max_cols;
         idx += gridDim.x * blockDim.x) {
        int i = idx / max_cols;
        int j = idx - i * max_cols;
        if (j >= s_kv || j > i || (window > 0 && j < i + 1 - window)) {
            scores[idx] = -INFINITY;
        }
    }
}

// Block preselect over a device-read s_kv with a FIXED block count (the
// graph geometry). Blocks whose start is beyond *p_s_kv return; candidate
// padding stays zero (the launcher zero-fills cands each step).
extern "C" __global__ void __launch_bounds__(1024)
di_block_preselect_dev_kernel(const float* __restrict__ scores,
                              u64* __restrict__ cands,
                              const int* __restrict__ p_s_kv,
                              int s_q,
                              int block,
                              int per,
                              int max_cols) {
    extern __shared__ u64 keys[];
    int i = blockIdx.y;
    int b = blockIdx.x;
    int s_kv = *p_s_kv;
    int bs = b * block;
    if (bs >= s_kv) return;
    int be = bs + block;
    if (be > s_kv) be = s_kv;
    int n = be - bs;
    int pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    for (int t = threadIdx.x; t < pow2; t += blockDim.x) {
        keys[t] = (t < n) ? di_score_pos_key(scores[(u64)i * max_cols + bs + t], (u32)(bs + t))
                          : 0ull;
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, pow2);
    if (threadIdx.x < per) {
        u64 k = keys[pow2 - 1 - (int)threadIdx.x];
        if (k != 0ull) {
            cands[((u64)i * gridDim.x) * per + (u64)b * per + threadIdx.x] = k;
        }
    }
}

// ---------------------------------------------------------------------------
// Conv branch for the single new key row (device-read j0; j1 = j0 + 1).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_conv_feat_dev_kernel(const float* __restrict__ k,
                        float* __restrict__ feat,
                        const float* __restrict__ kw,
                        const int* __restrict__ kw_off,
                        int num_kernels,
                        const int* __restrict__ p_j0,
                        int hidden) {
    int j0 = *p_j0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hidden;
         idx += gridDim.x * blockDim.x) {
        int d = idx;
        int j = j0;
        float acc = 0.0f;
        for (int c = 0; c < num_kernels; ++c) {
            int len = kw_off[c + 1] - kw_off[c];
            for (int t = 0; t < len; ++t) {
                if (t > j) break;
                acc += kw[kw_off[c] + t] * k[(u64)(j - t) * hidden + d];
            }
        }
        feat[(u64)j * hidden + d] = acc;
    }
}

// Gated RMSNorm of the single key row at a device-read index.
extern "C" __global__ void __launch_bounds__(256)
di_conv_gated_norm_dev_kernel(float* __restrict__ feat,
                              const float* __restrict__ gain,
                              const int* __restrict__ p_row,
                              int hidden) {
    if (blockIdx.x != 0) return;
    float* row = feat + (u64)(*p_row) * hidden;
    float g = 0.0f, ms = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        g += row[d] * gain[d];
        ms += row[d] * row[d];
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
        row[d] *= sh[0] * sh[1];
    }
}
