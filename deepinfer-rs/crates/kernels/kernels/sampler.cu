// sampler.cu — device-side sampling kernels for the serve path.
//
// Two families:
//
//  * `di_sample_topp_kernel` — one CTA per logits row (grid = (B,),
//    block = 256), reproducing the `deepinfer-sampler` pipeline on device
//    so a decode step never reads the `[B, vocab]` logits back to the host:
//
//      cap-mask -> sanitize -> temperature
//      -> [greedy: argmax, ties -> lowest id]
//      -> top-k (k-th value, ties -> lowest id) -> softmax
//      -> top-p (smallest descending prefix with cum >= p, ties -> lowest id)
//      -> min-p (prob >= min_p * max_prob) -> renormalize
//      -> categorical(u) in token-id order
//
//    Per-row parameters `{temperature, top_p, min_p, u, cap, top_k}` are
//    uploaded as a `[B, 6]` f32 block; the uniform `u` is pre-drawn on the
//    host from the request's Philox stream (bit-exact RNG stays on the
//    host; the kernel only consumes it). Grammar/structured-output masks
//    stay on the host path — the engine falls back to the logits readback
//    for those requests (the same split vLLM V1 and SGLang use).
//
//    Determinism: every reduction is a fixed warp-shuffle + shared-memory
//    tree, every selection pass a fixed-order bit descent, so replays are
//    bit-identical. Greedy rows are bit-identical to the CPU sampler (pure
//    comparisons over the same f32 values). Stochastic rows match the CPU
//    sampler up to `expf` rounding (documented tolerance: exact-tie masses
//    may flip a boundary token; golden tests assert distributional
//    equality and replay determinism).
//
//    Threshold selection without sorting: `di_mono_key` maps f32 to a u32
//    key monotone in value, so "the k-th largest value" and "the smallest
//    descending prefix with mass >= p" are found by bit-descent over the
//    key space (32 count/mass passes) and the tie boundaries (which ids
//    inside a tied value group survive) by bit-descent over the id space.
//    The `-inf` sentinel keys below every finite value and contributes
//    zero mass — exactly like the CPU's `-inf` fill.
//
//  * `di_draft_pick_part/final_kernel` — the DSpark fused draft-sample
//    step ("MarkovSampleStep"): logits row j = dot(hidden, embed[j]) over
//    the shared embedding table (the tied MTP head), reduced on the fly to
//    (argmax, top-1 softmax confidence) with an online-rescaled exp-sum.
//    No `[vocab]` logits are ever materialized.

// ---------------------------------------------------------------------------
// Monotone f32 -> u32 key (order-preserving; -inf maps into the low range).
// ---------------------------------------------------------------------------
__device__ __forceinline__ u32 di_mono_key(float x) {
    u32 b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// One row's sampling block geometry.
#define SMP_T 256
#define SMP_W (SMP_T / DI_WARP)

// Block-wide (val, id) argmax reduce: higher val wins, ties -> lower id.
__device__ __forceinline__ void smp_reduce_argmax(float* val, int* id) {
#pragma unroll
    for (int off = DI_WARP / 2; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(DI_FULL_MASK, *val, off);
        int oi = __shfl_xor_sync(DI_FULL_MASK, *id, off);
        if (ov > *val || (ov == *val && oi < *id)) {
            *val = ov;
            *id = oi;
        }
    }
    __shared__ float wv[SMP_W];
    __shared__ int wi[SMP_W];
    int w = di_warp_id();
    if (di_lane_id() == 0) {
        wv[w] = *val;
        wi[w] = *id;
    }
    __syncthreads();
    if (w == 0) {
        *val = (di_lane_id() < SMP_W) ? wv[di_lane_id()] : -1.0e30f;
        *id = (di_lane_id() < SMP_W) ? wi[di_lane_id()] : 0x7fffffff;
#pragma unroll
        for (int off = DI_WARP / 2; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(DI_FULL_MASK, *val, off);
            int oi = __shfl_xor_sync(DI_FULL_MASK, *id, off);
            if (ov > *val || (ov == *val && oi < *id)) {
                *val = ov;
                *id = oi;
            }
        }
        if (di_lane_id() == 0) {
            wv[0] = *val;
            wi[0] = *id;
        }
    }
    __syncthreads();
    *val = wv[0];
    *id = wi[0];
}

// Block-wide f32 sum reduce (fixed tree; deterministic).
__device__ __forceinline__ float smp_reduce_sum(float v) {
    v = di_warp_reduce_sum(v);
    __shared__ float ws[SMP_W];
    int w = di_warp_id();
    if (di_lane_id() == 0) ws[w] = v;
    __syncthreads();
    if (w == 0) {
        float x = (di_lane_id() < SMP_W) ? ws[di_lane_id()] : 0.0f;
        x = di_warp_reduce_sum(x);
        if (di_lane_id() == 0) ws[0] = x;
    }
    __syncthreads();
    return ws[0];
}

// ---------------------------------------------------------------------------
// di_sample_topp_kernel — one CTA per row.
//   logits  [B, V] f32 (row-major, stride V)
//   params  [B, 6] f32: {temperature, top_p, min_p, u, cap, top_k}
//           (cap/top_k carried as f32 — exact for ids < 2^24)
//   out_tok [B] u32, out_status [B] i32
// status: 0 ok, 1 no finite logit, 2 no positive mass after filtering,
//         3 invalid uniform.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(SMP_T)
di_sample_topp_kernel(const float* __restrict__ logits,
                      const float* __restrict__ params,
                      unsigned int* __restrict__ out_tok,
                      int* __restrict__ out_status,
                      int V) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= gridDim.x) return;
    const float* lg = logits + (u64)row * (u64)V;
    const float* pr = params + (u64)row * 6ull;
    const float T = pr[0];
    const float P = pr[1];
    const float MP = pr[2];
    const float U = pr[3];
    const int cap = (int)pr[4];
    const int K = (int)pr[5];
    const int Veff = (cap > 0 && cap < V) ? cap : V;
    if (tid == 0) out_status[row] = 0;

    // Pass 1 — sanitized raw max/argmax (ties -> lowest id) + finite count.
    float bval = -1.0e30f;
    int bid = 0x7fffffff;
    int finite = 0;
    for (int i = tid; i < V; i += SMP_T) {
        float raw = lg[i];
        bool ok = (i < Veff) && isfinite(raw);
        float v = ok ? raw : -1.0e30f;
        if (ok) finite++;
        if (v > bval || (v == bval && i < bid)) {
            bval = v;
            bid = i;
        }
    }
    smp_reduce_argmax(&bval, &bid);
    finite = (int)smp_reduce_sum((float)finite);
    if (finite == 0) {
        if (tid == 0) out_status[row] = 1;
        return;
    }

    // Greedy short-circuit (CPU rule: t <= 0, |t| < 1e-8 or NaN -> argmax
    // of the RAW masked row; consumes no RNG).
    const bool greedy = !(T > 0.0f) || !(T >= 1e-8f);
    if (greedy) {
        if (tid == 0) out_tok[row] = (u32)bid;
        return;
    }

    // Stochastic path: x(i) = lg[i] / T (finite, unmasked); m = max(x) —
    // correctly-rounded division by a positive constant is monotone, so
    // the scaled maximum is the scaled raw maximum.
    const float m = bval / T;
#define SMP_OK(i) ((i) < Veff)
#define SMP_X(i) (isfinite(lg[i]) ? (lg[i] / T) : -1.0e30f)
#define SMP_E(i) (isfinite(lg[i]) ? expf(SMP_X(i) - m) : 0.0f)
#define SMP_KEY(i) di_mono_key(SMP_X(i))

    // Pass 2 — z = sum e(i) (deterministic tree; -inf contributes 0).
    float zsum = 0.0f;
    for (int i = tid; i < V; i += SMP_T) zsum += SMP_E(i);
    const float z = smp_reduce_sum(zsum);

    // ---- top-k threshold: kappa_k + tie boundary -------------------------
    // kept_k(i) = key(x_i) > kappa_k || (key == kappa_k && i <= bnd_k)
    u32 kappa_k = 0u;
    int bnd_k = 0x7fffffff;
    const bool use_k = (K > 0) && (K < Veff);
    if (use_k) {
        u32 kappa = 0u;
        for (int b = 31; b >= 0; --b) {
            u32 trial = kappa | (1u << b);
            int cnt = 0;
            for (int i = tid; i < V; i += SMP_T) {
                if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) >= trial) cnt++;
            }
            if ((int)smp_reduce_sum((float)cnt) >= K) kappa = trial;
        }
        kappa_k = kappa;
        int below = 0;
        for (int i = tid; i < V; i += SMP_T) {
            if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) > kappa) below++;
        }
        below = (int)smp_reduce_sum((float)below);
        // The (K - below)-th smallest id inside the tied group.
        const int j = K - below;
        int t = Veff; // invariant: ties(<= t) >= j (t = Veff keeps all)
        for (int b = 24; b >= 0; --b) {
            int cand = t - (1 << b);
            if (cand < 0) continue;
            int cnt = 0;
            for (int i = tid; i < V; i += SMP_T) {
                if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) == kappa && i <= cand)
                    cnt++;
            }
            if ((int)smp_reduce_sum((float)cnt) >= j) t = cand;
        }
        bnd_k = t;
    }

    // ---- top-p threshold: kappa_p + tie boundary --------------------------
    // kept_p(i) = key > kappa_p || (key == kappa_p && i <= bnd_p)
    u32 kappa_p = 0u;
    int bnd_p = 0x7fffffff;
    const bool use_p = (P < 1.0f);
    if (use_p) {
        const float pz = P * z;
        u32 kappa = 0u;
        for (int b = 31; b >= 0; --b) {
            u32 trial = kappa | (1u << b);
            float mass = 0.0f;
            for (int i = tid; i < V; i += SMP_T) {
                if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) >= trial)
                    mass += SMP_E(i);
            }
            if (smp_reduce_sum(mass) >= pz) kappa = trial;
        }
        kappa_p = kappa;
        float below = 0.0f;
        for (int i = tid; i < V; i += SMP_T) {
            if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) > kappa) below += SMP_E(i);
        }
        below = smp_reduce_sum(below);
        // Smallest id t whose tie-mass closes the gap to P*z.
        int t = Veff;
        for (int b = 24; b >= 0; --b) {
            int cand = t - (1 << b);
            if (cand < 0) continue;
            float mass = 0.0f;
            for (int i = tid; i < V; i += SMP_T) {
                if (SMP_OK(i) && isfinite(lg[i]) && SMP_KEY(i) == kappa && i <= cand)
                    mass += SMP_E(i);
            }
            if (below + smp_reduce_sum(mass) >= pz) t = cand;
        }
        bnd_p = t;
    }

    // ---- min-p: prob(i) >= MP * max_prob, max_prob = fl(1/z) --------------
    const bool use_mp = (MP > 0.0f);
    const float mp_thr = MP * (1.0f / z);

    // kept(i): top-k AND top-p AND min-p (each filter's exact CPU rule).
#define SMP_KEPT(i)                                                          \
    (SMP_OK(i) && isfinite(lg[i]) &&                                         \
     (!use_k || SMP_KEY(i) > kappa_k || (SMP_KEY(i) == kappa_k && (i) <= bnd_k)) && \
     (!use_p || SMP_KEY(i) > kappa_p || (SMP_KEY(i) == kappa_p && (i) <= bnd_p)) && \
     (!use_mp || (SMP_E(i) / z) >= mp_thr))

    // Pass 5 — renormalized probabilities + total + last kept id.
    float tot1 = 0.0f;
    int last_kept = -1;
    for (int i = tid; i < V; i += SMP_T) {
        if (SMP_KEPT(i)) {
            tot1 += SMP_E(i) / z;
            if (i > last_kept) last_kept = i;
        }
    }
    const float total = smp_reduce_sum(tot1);
    // Block max of last_kept.
    {
#pragma unroll
        for (int off = DI_WARP / 2; off > 0; off >>= 1) {
            int ov = __shfl_xor_sync(DI_FULL_MASK, last_kept, off);
            if (ov > last_kept) last_kept = ov;
        }
        __shared__ int wl[SMP_W];
        int w = di_warp_id();
        if (di_lane_id() == 0) wl[w] = last_kept;
        __syncthreads();
        if (w == 0 && di_lane_id() < SMP_W) {
            int v = wl[di_lane_id()];
#pragma unroll
            for (int off = DI_WARP / 2; off > 0; off >>= 1) {
                int ov = __shfl_xor_sync(DI_FULL_MASK, v, off);
                if (ov > v) v = ov;
            }
            if (di_lane_id() == 0) wl[0] = v;
        }
        __syncthreads();
        last_kept = wl[0];
    }
    if (!(total > 0.0f) || isnan(total)) {
        if (tid == 0) out_status[row] = 2;
        return;
    }
    if (!(U >= 0.0f && U < 1.0f)) {
        if (tid == 0) out_status[row] = 3;
        return;
    }

    // Pass 6 — categorical walk in token-id order (the CPU contract:
    // smallest id whose cumulative renormalized mass strictly exceeds
    // u * total; the round-off fallback is the last positive id).
    __shared__ float slice[SMP_T];
    __shared__ int hit[SMP_W];
    // Each thread owns the contiguous slice [tid*S, (tid+1)*S).
    const int S = (V + SMP_T - 1) / SMP_T;
    const int lo = tid * S;
    const int hi = (lo + S < V) ? (lo + S) : V;
    float ssum = 0.0f;
    for (int i = lo; i < hi; ++i) {
        if (SMP_KEPT(i)) ssum += SMP_E(i) / z;
    }
    slice[tid] = ssum;
    __syncthreads();
    // Sequential prefix over the slice sums (thread 0; deterministic).
    if (tid == 0) {
        float run = 0.0f;
        for (int i = 0; i < SMP_T; ++i) {
            float v = slice[i];
            slice[i] = run;
            run += v;
        }
    }
    __syncthreads();
    const float target = U * total;
    int my_hit = 0x7fffffff;
    float cum = slice[tid];
    for (int i = lo; i < hi; ++i) {
        float p = SMP_KEPT(i) ? (SMP_E(i) / z) : 0.0f;
        if (p > 0.0f) {
            cum += p;
            if (cum > target) {
                my_hit = i;
                break;
            }
        }
    }
    // Warp + block min over the hits.
    {
#pragma unroll
        for (int off = DI_WARP / 2; off > 0; off >>= 1) {
            int ov = __shfl_xor_sync(DI_FULL_MASK, my_hit, off);
            if (ov < my_hit) my_hit = ov;
        }
        int w = di_warp_id();
        if (di_lane_id() == 0) hit[w] = my_hit;
        __syncthreads();
        if (w == 0 && di_lane_id() < SMP_W) {
            int v = hit[di_lane_id()];
#pragma unroll
            for (int off = DI_WARP / 2; off > 0; off >>= 1) {
                int ov = __shfl_xor_sync(DI_FULL_MASK, v, off);
                if (ov < v) v = ov;
            }
            if (di_lane_id() == 0) hit[0] = v;
        }
        __syncthreads();
        my_hit = hit[0];
    }
    if (tid == 0) {
        out_tok[row] = (my_hit != 0x7fffffff) ? (u32)my_hit : (u32)last_kept;
    }
#undef SMP_KEPT
#undef SMP_KEY
#undef SMP_E
#undef SMP_X
#undef SMP_OK
}

// ---------------------------------------------------------------------------
// di_draft_pick_part/final — the fused DSpark draft step. Part: each CTA
// owns a slab of token rows; per thread an online-rescaled (max, exp-sum)
// pair (no overflow: exp arguments are always <= 0), then a block fold to
// (slab max, slab argmax, exp-sum relative to the slab max). Final: merge
// the slab partials into the global argmax + softmax confidence.
//   hidden [H] f32, embed [V, H] f32 (row j = token j — the tied head)
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_draft_pick_part_kernel(const float* __restrict__ hidden,
                          const float* __restrict__ embed,
                          float* __restrict__ part_max,
                          int* __restrict__ part_idx,
                          float* __restrict__ part_sum,
                          int V, int H, int slab) {
    const int b = blockIdx.x;
    const int lo = b * slab;
    const int hi = (lo + slab < V) ? (lo + slab) : V;
    float bv = -1.0e30f;   // thread argmax value
    int bi = -1;           // thread argmax id
    float tm = -1.0e30f;   // thread running max
    float ts = 0.0f;       // exp-sum relative to tm
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        const float* row = embed + (u64)j * (u64)H;
        float acc = 0.0f;
        for (int d = 0; d < H; ++d) acc += hidden[d] * row[d];
        if (acc > bv || (acc == bv && (bi < 0 || j < bi))) {
            bv = acc;
            bi = j;
        }
        if (acc > tm) {
            ts = ts * expf(tm - acc) + 1.0f;
            tm = acc;
        } else {
            ts += expf(acc - tm);
        }
    }
    // Block argmax (ties -> lowest id).
    smp_reduce_argmax(&bv, &bi);
    // Block (max, sum) fold: warp partials into shared, thread 0 folds in
    // warp order (deterministic; rescale each pair to the block max).
    __shared__ float sm[SMP_W];
    __shared__ float ss[SMP_W];
    int w = di_warp_id();
#pragma unroll
    for (int off = DI_WARP / 2; off > 0; off >>= 1) {
        float om = __shfl_xor_sync(DI_FULL_MASK, tm, off);
        float os = __shfl_xor_sync(DI_FULL_MASK, ts, off);
        if (om > tm) {
            ts = os + ts * expf(tm - om);
            tm = om;
        } else {
            ts = ts + os * expf(om - tm);
        }
    }
    if (di_lane_id() == 0) {
        sm[w] = tm;
        ss[w] = ts;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float gm = sm[0];
        float gs = ss[0];
        for (int i = 1; i < SMP_W; ++i) {
            if (sm[i] > gm) {
                gs = gs * expf(gm - sm[i]) + ss[i];
                gm = sm[i];
            } else {
                gs = gs + ss[i] * expf(sm[i] - gm);
            }
        }
        // Rescale to the block argmax value (bv >= gm always).
        part_max[b] = bv;
        part_idx[b] = bi;
        part_sum[b] = (bi >= 0) ? gs * expf(gm - bv) : 0.0f;
    }
}

// Final merge of the draft partials: global argmax + softmax confidence.
extern "C" __global__ void __launch_bounds__(256)
di_draft_pick_final_kernel(const float* __restrict__ part_max,
                           const int* __restrict__ part_idx,
                           const float* __restrict__ part_sum,
                           unsigned int* __restrict__ out_tok,
                           float* __restrict__ out_conf,
                           int G) {
    if (blockIdx.x != 0) return;
    float bv = -1.0e30f;
    int bi = -1;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        float v = part_max[g];
        if (v > bv || (v == bv && part_idx[g] >= 0 && (bi < 0 || part_idx[g] < bi))) {
            bv = v;
            bi = part_idx[g];
        }
    }
    smp_reduce_argmax(&bv, &bi);
    float z = 0.0f;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        if (part_idx[g] >= 0) z += part_sum[g] * expf(part_max[g] - bv);
    }
    z = smp_reduce_sum(z);
    if (threadIdx.x == 0) {
        *out_tok = (u32)bi;
        *out_conf = 1.0f / z;
    }
}
