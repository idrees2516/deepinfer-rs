// attention.cu — FlashMLA-equivalent attention for DeepSeek-V4.1.
//
// Port of `deepinfer-mla` (the CPU golden):
//   * decode: split-KV online softmax (flash-decoding). Partial states
//     (m, l, acc) per (query, head, split) are merged by a combine kernel
//     in split-ascending order — the same merge order as the CPU's chunk
//     loop, so numerics match within tolerance.
//   * attn_sink epilogue: out *= 1 / (1 + exp(sink - lse)); no-valid-index
//     queries produce out = 0 and lse = +inf (FlashMLA edge case).
//   * prefill: dense causal and sparse (index-gathered) flash attention with
//     optional per-query topk_length.
//
// Kernel structure is warp-size-parametric (DI_WARP): one warp owns one
// (head, query) pair; each lane holds D/DI_WARP accumulator dimensions; the
// score dot is a warp reduction. Works unchanged on NVIDIA (32) and
// AMD CDNA (64).
//
// KV cache formats (FlashMLA V4.1):
//   fmt 0: f32 rows  [rows, 512]
//   fmt 1: FP8 rows  [rows, 528]  (512 e4m3 + 16 ue8m0 per-32 scales)
//   fmt 2: FP4 rows  [rows, 288]  (256B packed e2m1 even-low + 32 e4m3
//                                  per-16 scales)

#define ATT_WARPS 8 // heads (decode) / query rows (prefill) per CTA

__device__ __forceinline__ float att_load_k(const void* __restrict__ cache,
                                            int fmt, int idx, int d, int D) {
    if (fmt == 0) {
        return ((const float*)cache)[(u64)idx * D + d];
    }
    if (fmt == 1) {
        const u8* row = (const u8*)cache + (u64)idx * 528;
        return di_e4m3_to_f32(row[d]) * di_e8m0_to_f32(row[D + (d >> 5)]);
    }
    const u8* row = (const u8*)cache + (u64)idx * 288;
    u8 nib = ((d & 1) == 0) ? (u8)(row[d >> 1] & 0x0Fu) : (u8)(row[d >> 1] >> 4);
    return di_e2m1_to_f32(nib) * di_e4m3_to_f32(row[256 + (d >> 4)]);
}

// ---------------------------------------------------------------------------
// Decode, split-KV partial pass.
// grid = (splits, ceil(H / ATT_WARPS), b * s_q); block = ATT_WARPS*DI_WARP.
// Partials: pm/pl [b*s_q*H, splits]; po [b*s_q*H, splits, D].
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_mla_decode_partial_kernel(const float* __restrict__ q,
                             const void* __restrict__ cache,
                             const int* __restrict__ indices,
                             float* __restrict__ pm,
                             float* __restrict__ pl,
                             float* __restrict__ po,
                             int H, int D, int topk, int chunk,
                             float sm_scale, int fmt, int cache_rows) {
    int split = blockIdx.x;
    int splits = gridDim.x;
    int hbase = blockIdx.y * ATT_WARPS;
    int bqi = blockIdx.z;
    int warp = (int)(threadIdx.x / DI_WARP);
    int h = hbase + warp;
    int lane = di_lane_id();
    if (h >= H) return;
    int per = D / DI_WARP; // requires D % DI_WARP == 0 (512 on both vendors)
    int d0 = lane * per;
    const float* qrow = q + ((u64)bqi * H + h) * D;
    const int* idxs = indices + (u64)bqi * topk;

    float qv[16];
    float ov[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        qv[i] = 0.0f;
        ov[i] = 0.0f;
    }
    for (int i = 0; i < per; ++i) qv[i] = qrow[d0 + i];

    float m = -INFINITY;
    float l = 0.0f;

    int cs = split * chunk;
    int ce = cs + chunk;
    if (ce > topk) ce = topk;
    for (int c = cs; c < ce; ++c) {
        int idx = idxs[c];
        if (idx < 0 || idx >= cache_rows) continue;
        float kv[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
        for (int i = 0; i < per; ++i) kv[i] = att_load_k(cache, fmt, idx, d0 + i, D);
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

    u64 pbase = ((u64)bqi * H + h) * splits + split;
    if (lane == 0) {
        pm[pbase] = m;
        pl[pbase] = l;
    }
    float* porow = po + pbase * D;
    for (int i = 0; i < per; ++i) porow[d0 + i] = ov[i];
}

// ---------------------------------------------------------------------------
// Decode, combine pass: merge split partials (split-ascending — the CPU
// chunk-merge order), normalize, fill lse, apply attn_sink.
// grid = (ceil(D / 256), b * s_q * H); block = 256.
// lse layout: lse[(bi * H + hi) * s_q + qi] (CPU contract).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_mla_decode_combine_kernel(const float* __restrict__ pm,
                             const float* __restrict__ pl,
                             const float* __restrict__ po,
                             const float* __restrict__ sink, // [H] or NULL
                             float* __restrict__ out,
                             float* __restrict__ lse,
                             int s_q, int H, int D, int splits) {
    int d = blockIdx.x * 256 + (int)threadIdx.x;
    int bqh = blockIdx.y; // = bqi * H + h
    if (d >= D) return;
    int bqi = bqh / H;
    int h = bqh - bqi * H;
    int qi = bqi % s_q;

    // Sequential split-ascending merge (identical structure to the CPU).
    float o = 0.0f, l = 0.0f, m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        u64 pbase = ((u64)bqh * splits) + s;
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
        v *= 1.0f / (1.0f + expf(sink[h] - lse_v));
    }
    out[((u64)bqh) * D + d] = v;
    if (threadIdx.x == 0) lse[((u64)bqi * H + h) * s_q + qi] = lse_v;
}

// ---------------------------------------------------------------------------
// Prefill: dense causal or sparse (index-gathered) attention.
// grid = (ceil(s_q / ATT_WARPS), H); block = ATT_WARPS * DI_WARP.
//   dense:  key j attends iff j <= q_offset + i   (q_offset = query block
//           start position in the kv sequence).
//   sparse: indices [s_q, topk] (+ optional topk_length [s_q]); rows are
//           gathered, no causal mask (upstream preselects valid tokens).
// Outputs: out [s_q, H, D]; optional max_logits / lse [s_q, H] (NULL skip).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(ATT_WARPS * 64)
di_mla_prefill_kernel(const float* __restrict__ q,
                      const void* __restrict__ cache,
                      const int* __restrict__ indices, // NULL for dense
                      const int* __restrict__ topk_length,
                      float* __restrict__ out,
                      float* __restrict__ max_logits,
                      float* __restrict__ lse,
                      int s_q, int H, int D, int s_kv, int q_offset,
                      int topk, float sm_scale, int fmt, int cache_rows) {
    int qrow = blockIdx.x * ATT_WARPS + (int)(threadIdx.x / DI_WARP);
    int h = blockIdx.y;
    if (qrow >= s_q) return;
    int lane = di_lane_id();
    int per = D / DI_WARP;
    int d0 = lane * per;
    const float* qv_ptr = q + ((u64)qrow * H + h) * D;

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

    if (indices) {
        int lim = topk;
        if (topk_length) lim = topk_length[qrow];
        const int* idxs = indices + (u64)qrow * topk;
        for (int c = 0; c < lim; ++c) {
            int idx = idxs[c];
            if (idx < 0 || idx >= cache_rows) continue;
            float kv[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
            for (int i = 0; i < per; ++i) kv[i] = att_load_k(cache, fmt, idx, d0 + i, D);
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
    } else {
        int jend = q_offset + qrow + 1;
        if (jend > s_kv) jend = s_kv;
        for (int j = 0; j < jend; ++j) {
            float kv[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) kv[i] = 0.0f;
            for (int i = 0; i < per; ++i) kv[i] = att_load_k(cache, fmt, j, d0 + i, D);
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

    float lse_v = (l > 0.0f) ? (m + logf(l)) : INFINITY;
    float* orow = out + ((u64)qrow * H + h) * D;
    for (int i = 0; i < per; ++i) orow[d0 + i] = (l > 0.0f) ? (ov[i] / l) : 0.0f;
    if (lane == 0) {
        if (max_logits) max_logits[(u64)qrow * H + h] = m;
        if (lse) lse[(u64)qrow * H + h] = lse_v;
    }
}
