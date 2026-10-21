// moe.cu — DeepSeekMoE routing, dispatch (permute) and combine.
// Ports of `deepinfer-moe` / `deepinfer-ep` semantics (CPU golden):
//   * routing: sigmoid scores; e_score_correction_bias affects SELECTION
//     only (noaux-tc); top-k with (score desc, expert asc) ties; weights =
//     bias-free sigmoids renormalized to sum 1 (uniform fallback on total
//     underflow). Grouped-expert mode is handled host-side (v1: GPU path
//     requires group_size == 1 — V4.1 default).
//   * dispatch: per-expert counts (atomics — the SET is deterministic, and
//     per-token outputs are independent of the permuted row order), aligned
//     segment offsets (multiples of the DeepGEMM contiguous alignment),
//     row scatter with the mapping recorded for combine, zero padding.
//   * combine: y[t] = sum_k w_k * expert_out[map(t,k)] with fixed
//     k-ascending accumulation — deterministic.

// Route one token per block: needs E <= blockDim (callers use 256 threads
// for E=256). Threads beyond E idle.
extern "C" __global__ void __launch_bounds__(1024)
di_moe_route_kernel(const float* __restrict__ logits, // [tokens, E]
                    const float* __restrict__ bias,    // [E] or NULL
                    float* __restrict__ topk_w,        // [tokens, K]
                    int* __restrict__ topk_idx,        // [tokens, K]
                    int tokens, int E, int K, int use_bias) {
    extern __shared__ u8 smem[];
    u64* keys = (u64*)smem;          // E entries
    float* nob = (float*)(smem + ((E + 1) / 2) * 8); // E floats after keys
    int t = blockIdx.x;
    if (t >= tokens) return;
    const float* z = logits + (u64)t * E;
    int pow2 = 1;
    while (pow2 < E) pow2 <<= 1;
    for (int e = threadIdx.x; e < pow2; e += blockDim.x) {
        float s = 0.0f, nb = 0.0f;
        if (e < E) {
            s = di_sigmoid(z[e]);
            nb = s;
            if (use_bias && bias) s += bias[e];
        }
        if (e < E) nob[e] = nb;
        keys[e] = (e < E) ? di_score_pos_key(s, (u32)e) : 0ull;
    }
    __syncthreads();
    di_bitonic_sort_u64(keys, pow2); // ascending; top-K = tail
    // Selected experts: K highest keys, in (score desc, expert asc) order —
    // matching top_k_desc's output order. Sum in a fixed tree (thread order)
    // to stay deterministic.
    __shared__ float red[1024];
    red[threadIdx.x] = (threadIdx.x < K)
                           ? nob[(int)di_key_pos(keys[pow2 - 1 - threadIdx.x])]
                           : 0.0f;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float* tot = red;
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int e = (int)di_key_pos(keys[pow2 - 1 - k]);
        topk_idx[(u64)t * K + k] = e;
        topk_w[(u64)t * K + k] = (tot[0] > 0.0f && isfinite(tot[0]))
                                        ? (nob[e] / tot[0])
                                        : (1.0f / (float)K);
    }
}

// Per-expert counts. topk_idx entries < 0 are skipped (padding tokens).
extern "C" __global__ void __launch_bounds__(256)
di_moe_count_kernel(const int* __restrict__ topk_idx, int* __restrict__ counts,
                    int tokens, int K, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tokens * K) return;
    int e = topk_idx[idx];
    if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
}

// Segment offsets with DeepGEMM contiguous alignment:
// offsets[0] = 0; offsets[g+1] = align_up(offsets[g] + counts[g], align).
// Single CTA, serial scan by thread 0 (E <= 512) — deterministic.
extern "C" __global__ void __launch_bounds__(256)
di_moe_prefix_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                     int E, int align) {
    if (threadIdx.x == 0) {
        int acc = 0;
        offsets[0] = 0;
        for (int g = 0; g < E; ++g) {
            int c = counts[g];
            acc += c;
            // round up to a multiple of `align`
            int rem = acc % align;
            if (rem != 0) acc += align - rem;
            offsets[g + 1] = acc;
        }
    }
}

// Scatter rows: one CTA per (token, k); copies the token's hidden row into
// its expert segment and records the permuted row index. Rows in the
// alignment padding are zeroed by di_fill_zero_kernel (host zeroes the whole
// perm buffer first).
extern "C" __global__ void __launch_bounds__(256)
di_moe_scatter_kernel(const float* __restrict__ x,
                      const int* __restrict__ topk_idx,
                      const int* __restrict__ offsets,
                      int* __restrict__ rank_counters,
                      float* __restrict__ perm,
                      int* __restrict__ map, // [tokens, K]
                      int tokens, int hidden, int K) {
    int tk = blockIdx.x;
    if (tk >= tokens * K) return;
    int t = tk / K;
    int k = tk - t * K;
    int e = topk_idx[tk];
    if (e < 0) {
        if (threadIdx.x == 0) map[tk] = -1;
        return;
    }
    int rank = 0;
    if (threadIdx.x == 0) rank = atomicAdd(&rank_counters[e], 1);
    __shared__ int srank;
    if (threadIdx.x == 0) srank = rank;
    __syncthreads();
    int row = offsets[e] + srank;
    if (threadIdx.x == 0) map[tk] = row;
    const float* src = x + (u64)t * hidden;
    float* dst = perm + (u64)row * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) dst[d] = src[d];
}

// Combine: y[t] = sum_k w_k * expert_out[map(t,k)], k-ascending.
extern "C" __global__ void __launch_bounds__(256)
di_moe_combine_kernel(const float* __restrict__ perm_out,
                      const int* __restrict__ map,
                      const float* __restrict__ topk_w,
                      float* __restrict__ y,
                      int tokens, int hidden, int K) {
    int t = blockIdx.x;
    if (t >= tokens) return;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            int row = map[(u64)t * K + k];
            if (row < 0) continue;
            acc += topk_w[(u64)t * K + k] * perm_out[(u64)row * hidden + d];
        }
        y[(u64)t * hidden + d] = acc;
    }
}

// Generic byte fill (used to zero the padded perm rows and workspaces).
extern "C" __global__ void __launch_bounds__(256)
di_fill_zero_kernel(void* __restrict__ p, u64 n) {
    u8* b = (u8*)p;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (u64)gridDim.x * blockDim.x) {
        b[i] = 0;
    }
}

// Sigmoid applied elementwise to router logits (feeds di_moe_route_kernel
// when the router GEMM ran on GPU).
extern "C" __global__ void __launch_bounds__(256)
di_sigmoid_kernel(const float* __restrict__ z, float* __restrict__ s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        s[i] = di_sigmoid(z[i]);
    }
}
