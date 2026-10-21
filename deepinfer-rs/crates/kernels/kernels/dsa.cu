// dsa.cu — DeepSeek Sparse Attention: lightning indexer + top-k selection.
// Ports of `deepinfer-dsa` (CPU golden):
//   * thunder branch: per-(token,head) RMSNorm (eps 1e-6) then "lightning
//     quant" (e4m3 + one ue8m0 scale per COMPLETE quant_block tokens, ceil
//     amax/448); score(i,j) = sum_h relu(qtilde_i^h . ktilde_j^h) * w_i^h
//     with h-ascending, dim-ascending accumulation (same order as CPU).
//   * conv branch: cascade causal depthwise convs (cascade-major, then tap,
//     then dim order), gated RMSNorm (sigmoid(dot) gain, eps 1e-6), then
//     plain dots.
//   * gate: g_t = sigmoid(row0 . [q_i; mean_q]), g_c = sigmoid(row1 . ...),
//     combined = (g_t*th + g_c*cv)/(g_t+g_c); -inf survives the blend.
//   * selection: per-block top-`per_block` preselect, then a merge-tree
//     bitonic top-K over packed (score desc, pos asc) u64 keys — the CPU
//     tie-break contract — with the final positions sorted ascending and
//     -1-padded.

#define DSA_EPS 1e-6f

// In-place per-(token, head) RMSNorm over head_dim slices.
// x: [rows, nh*hd]; thread per (row, head).
extern "C" __global__ void __launch_bounds__(256)
di_indexer_rmsnorm_per_head_kernel(float* __restrict__ x, int rows, int nh, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nh) return;
    int r = idx / nh;
    int h = idx - r * nh;
    float* s = x + (u64)r * nh * hd + (u64)h * hd;
    float ms = 0.0f;
    for (int i = 0; i < hd; ++i) ms += s[i] * s[i];
    ms /= (float)hd;
    float inv = rsqrtf(ms + DSA_EPS);
    for (int i = 0; i < hd; ++i) s[i] *= inv;
}

// Lightning-quant partial-amax: grid (blocks * chunks); each CTA covers
// `chunk` elements of one quant block and atomicMax's the ordered bits of
// its local amax into sf_bits[b] (u32 bit order == float order for >= 0).
extern "C" __global__ void __launch_bounds__(256)
di_lightning_amax_kernel(const float* __restrict__ x, u32* __restrict__ sf_bits,
                         int dim, int quant_block, long long chunk_elems) {
    int b = blockIdx.x;
    int chunk = blockIdx.y;
    const float* xs = x + (u64)b * quant_block * dim + (u64)chunk * chunk_elems;
    long long remain = (long long)quant_block * dim - (long long)chunk * chunk_elems;
    if (remain > chunk_elems) remain = chunk_elems;
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

// Lightning-quant apply: q[i] = e4m3(x[i] / scale_b) for every element of
// the COMPLETE blocks (b < nblocks); sf[b] was finalized by the host from
// the amax buffer (e8m0 ceil of amax/448; zero-amax blocks -> 0x00).
extern "C" __global__ void __launch_bounds__(256)
di_lightning_quant_kernel(const float* __restrict__ x, u8* __restrict__ q,
                          const u8* __restrict__ sf, int nblocks, int dim,
                          int quant_block) {
    long long per_block = (long long)quant_block * dim;
    long long n = (long long)nblocks * per_block;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += (long long)gridDim.x * blockDim.x) {
        int b = (int)(idx / per_block);
        float scale = di_e8m0_to_f32(sf[b]);
        q[idx] = di_f32_to_e4m3(x[idx] / scale, true);
    }
}

// Thunder logits: score[i][j] = sum_h relu(qtilde_i^h . ktilde_j^h) * w_i^h.
// Complete quant blocks read the e4m3 payload; trailing rows (beyond
// nblocks*quant_block) use the f32 (already per-head-normed) arrays — the
// CPU's partial-chunk semantics.
// grid = (j-chunks, s_q); block = 256; dynamic smem stages qtilde for row i.
extern "C" __global__ void __launch_bounds__(256)
di_thunder_logits_kernel(const float* __restrict__ q_f32,
                         const u8* __restrict__ q_u8,
                         const u8* __restrict__ q_sf,
                         const float* __restrict__ k_f32,
                         const u8* __restrict__ k_u8,
                         const u8* __restrict__ k_sf,
                         const float* __restrict__ w,
                         float* __restrict__ scores,
                         int s_q, int s_kv, int nh, int hd,
                         int q_nblocks, int k_nblocks, int quant_block) {
    int i = blockIdx.y;
    int dim = nh * hd;
    extern __shared__ float sq_f32[]; // dim floats (dequantized-or-f32)
    // Stage qtilde_i (both representations; only one is meaningful).
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
        scores[(u64)i * s_kv + j] = acc;
    }
}

// Conv feature map: feat[j] = sum_c sum_{t<=j} kw_c[t] * k[j-t] over dims
// (cascade-major, then tap — the CPU accumulation order). Rows [j0, j1)
// only (incremental decode support).
extern "C" __global__ void __launch_bounds__(256)
di_conv_feat_kernel(const float* __restrict__ k, float* __restrict__ feat,
                    const float* __restrict__ kw, const int* __restrict__ kw_off,
                    int num_kernels, int j0, int j1, int s_kv, int hidden) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (j1 - j0) * hidden; idx += gridDim.x * blockDim.x) {
        int r = idx / hidden;
        int d = idx - r * hidden;
        int j = j0 + r;
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

// Gated RMSNorm per key row: feat *= sigmoid(feat . gain) / sqrt(ms+eps).
extern "C" __global__ void __launch_bounds__(256)
di_conv_gated_norm_kernel(float* __restrict__ feat, const float* __restrict__ gain,
                          int rows, int hidden) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rows) return;
    float* row = feat + (u64)j * hidden;
    float g = 0.0f, ms = 0.0f;
    for (int d = 0; d < hidden; ++d) {
        g += row[d] * gain[d];
        ms += row[d] * row[d];
    }
    g = di_sigmoid(g);
    ms /= (float)hidden;
    float inv = g * rsqrtf(ms + DSA_EPS);
    for (int d = 0; d < hidden; ++d) row[d] *= inv;
}

// Conv-branch scores: out[i][j] = q_i . feat_j (serial dim-ascending dot).
extern "C" __global__ void __launch_bounds__(256)
di_dot_scores_kernel(const float* __restrict__ q, const float* __restrict__ feat,
                     float* __restrict__ out, int s_q, int s_kv, int hidden) {
    int i = blockIdx.y;
    const float* qi = q + (u64)i * hidden;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < s_kv;
         j += gridDim.x * blockDim.x) {
        const float* fj = feat + (u64)j * hidden;
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) acc += qi[d] * fj[d];
        out[(u64)i * s_kv + j] = acc;
    }
}

// Mean of q over rows (i-ascending accumulation per dim — CPU order).
extern "C" __global__ void __launch_bounds__(256)
di_qmean_kernel(const float* __restrict__ q, float* __restrict__ mean,
                int s_q, int hidden) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden;
         d += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < s_q; ++i) acc += q[(u64)i * hidden + d];
        mean[d] = acc / (float)s_q;
    }
}

// Gate logits per query: gt/gc = gate_proj rows . [q_i; mean_q].
extern "C" __global__ void __launch_bounds__(256)
di_gate_logits_kernel(const float* __restrict__ q, const float* __restrict__ mean,
                      const float* __restrict__ gate_proj, float* __restrict__ gt,
                      float* __restrict__ gc, int s_q, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s_q) return;
    const float* qi = q + (u64)i * hidden;
    float rt = 0.0f, rc = 0.0f;
    for (int u = 0; u < 2 * hidden; ++u) {
        float x = (u < hidden) ? qi[u] : mean[u - hidden];
        rt += gate_proj[u] * x;
        rc += gate_proj[2 * hidden + u] * x;
    }
    gt[i] = di_sigmoid(rt);
    gc[i] = di_sigmoid(rc);
}

// Blend: thunder[i][j] = (gt*th + gc*cv)/(gt+gc), -inf survives.
extern "C" __global__ void __launch_bounds__(256)
di_gate_blend_kernel(const float* __restrict__ gt, const float* __restrict__ gc,
                     const float* __restrict__ conv, float* __restrict__ thunder,
                     int s_q, int cols) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < s_q * cols;
         idx += gridDim.x * blockDim.x) {
        int i = idx / cols;
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

// Causal clip: scores[i][j] = -inf for j > i.
extern "C" __global__ void __launch_bounds__(256)
di_causal_clip_kernel(float* __restrict__ scores, int s_q, int s_kv) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < s_q * s_kv;
         idx += gridDim.x * blockDim.x) {
        int i = idx / s_kv;
        int j = idx - i * s_kv;
        if (j > i) scores[idx] = -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Top-k selection.
// ---------------------------------------------------------------------------

// Per-block preselect: top `per` keys of each block of `block` scores.
// grid = (num_blocks, s_q); block threads = `block` (<= 1024).
extern "C" __global__ void __launch_bounds__(1024)
di_block_preselect_kernel(const float* __restrict__ scores, u64* __restrict__ cands,
                          int s_q, int s_kv, int block, int per) {
    extern __shared__ u64 keys[]; // block entries
    int i = blockIdx.y;
    int b = blockIdx.x;
    int bs = b * block;
    int be = bs + block;
    if (be > s_kv) be = s_kv;
    int n = be - bs;
    int pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    for (int t = threadIdx.x; t < pow2; t += blockDim.x) {
        keys[t] = (t < n) ? di_score_pos_key(scores[(u64)i * s_kv + bs + t], (u32)(bs + t))
                          : 0ull; // 0 = smallest key (padding)
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, pow2); // ascending
    // Top-`per` = the LAST per entries; write them (any order; merge sorts).
    if (threadIdx.x < per) {
        u64 k = keys[pow2 - 1 - (int)threadIdx.x];
        if (k != 0ull) {
            // cands capacity per row: gridDim.x * per (host-allocated).
            cands[((u64)i * gridDim.x) * per + (u64)b * per + threadIdx.x] = k;
        }
    }
}

// Merge round: sort `chunk` (power of two <= 2048) keys per CTA, keep the
// top `keep` (largest), write contiguously. grid = (ceil(n_in/chunk), s_q).
extern "C" __global__ void __launch_bounds__(1024)
di_topk_merge_kernel(const u64* __restrict__ in, u64* __restrict__ out,
                     int n_in, int keep, int chunk) {
    extern __shared__ u64 keys[];
    int i = blockIdx.y;
    int c = blockIdx.x;
    int base = c * chunk;
    int n = n_in - base;
    if (n > chunk) n = chunk;
    if (n <= 0) return;
    for (int t = threadIdx.x; t < chunk; t += blockDim.x) {
        keys[t] = (t < n) ? in[(u64)i * n_in + base + t] : 0ull;
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, chunk); // ascending
    int kept = keep < n ? keep : n;
    for (int t = threadIdx.x; t < kept; t += blockDim.x) {
        out[(u64)i * keep + t] = keys[chunk - 1 - t];
    }
}

// Final: from `n_in` keys (top-K of the whole row, unsorted by position),
// emit K positions in ascending order with -1 padding.
// grid = (1, s_q); block = 256; smem holds the (up to 2048) positions.
extern "C" __global__ void __launch_bounds__(256)
di_topk_scatter_kernel(const u64* __restrict__ in, int* __restrict__ out,
                       int n_in, int k_out) {
    extern __shared__ u32 pos[]; // k_out entries (power-of-two sortable)
    int i = blockIdx.y;
    (void)blockIdx.x;
    int n = n_in < k_out ? n_in : k_out;
    for (int t = threadIdx.x; t < k_out; t += blockDim.x) {
        // in[i*k_in + t] for t < n holds the selected keys.
        pos[t] = (t < n) ? di_key_pos(in[(u64)i * n_in + t]) : 0xffffffffu;
    }
    __syncthreads();
    // Sort positions ascending (u32 bitonic over k_out, power of two).
    for (int size = 2; size <= k_out; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int t = threadIdx.x; t < k_out; t += blockDim.x) {
                int p = t ^ stride;
                if (p > t && p < k_out) {
                    if (((t & size) == 0) == (pos[t] > pos[p])) {
                        u32 tmp = pos[t];
                        pos[t] = pos[p];
                        pos[p] = tmp;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < k_out; t += blockDim.x) {
        out[(u64)i * k_out + t] = (pos[t] == 0xffffffffu) ? -1 : (int)pos[t];
    }
}

// Lightning-quant finalize: amax bits -> ue8m0 scale bytes (ceil policy,
// zero-amax -> 0x00). The f64 log2/ceil path matches the CPU's
// f32_to_e8m0(Ceil) within its 1e-9 exact-power-of-two guard.
extern "C" __global__ void __launch_bounds__(256)
di_lightning_finalize_kernel(const u32* __restrict__ sf_bits, u8* __restrict__ sf,
                             int nblocks) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    float amax = __uint_as_float(sf_bits[b]);
    float ratio = amax / 448.0f;
    sf[b] = (ratio > 0.0f) ? di_f32_to_e8m0_ceil(ratio) : (u8)0x00;
}
