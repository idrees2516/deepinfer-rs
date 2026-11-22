// peer_ll.cu — NVSHMEM-style symmetric-buffer expert parallelism.
//
// The low-latency DeepEP communication pattern rebuilt on raw driver
// primitives (peer-access pointers / IPC-imported symmetric heaps) instead
// of the NVSHMEM library: every rank allocates one symmetric heap with the
// SAME layout, kernels write peers' heaps directly over NVLink / Infinity
// Fabric, and progress is tracked with system-scope acquire/release atomics
// in the heaps themselves — no host round-trip, no per-step readback, and
// no library beyond the driver we already load.
//
// Heap layout of rank r (identical on every rank; offsets in bytes):
//   [0, 4K)            counts[K]      slot reservations (senders atomically
//                                       claim channel rows on the receiver)
//   [4K, 8K)           produced[K]    release-incremented after a payload
//                                       row is fully written
//   [8K, 12K)          expected[K]    incremented once per sent row
//   [12K]              step_done      senders' completion barrier
//   [12K+4]            combine_done   combine-row arrivals for this home
//   [12K+8]            valid_pairs    this rank's own valid (token, k) pairs
//   [CH, CH+K*CAP*RB)  chan rows      dispatch channels [k][CAP] rows of
//                                     [hidden e4m3 | hidden/128 ue8m0 |
//                                      meta {src, token, k, expert} i32 x4]
//   [CB, ...)          combine rows   [tokens][K][hidden] f32
// where RB = align16(hidden + hidden/128 + 16), CH = align16(12K + 16) and
// CB = CH + K * CAP * RB. The same formulas are mirrored in
// deepinfer-kernels::peer_ll.
//
// Step protocol (all device-side; ranks may be enqueued in any order):
//   1. reset:       every rank's counters zeroed (the host orders resets
//                   before any dispatch through the event mesh).
//   2. dispatch:    per (token, k): reserve a channel slot on the owner
//                   rank, copy payload + meta, release-increment produced
//                   and expected on the owner; count own valid pairs.
//   3. dispatch_done (1 block per rank, after that rank's dispatch on its
//                   stream): release-increment step_done on EVERY heap.
//   4. group:       receiver spins step_done == world (acquire), then
//                   processes the (now final) channel rows into the masked
//                   per-expert layout [e*cap + rank] with local atomics,
//                   carrying (src, token, k) through meta2 for combine.
//   5. [expert masked GEMMs — the caller's own kernels]
//   6. combine:     owner writes each valid y row into the home rank's
//                   combine[token][k] slot (remote store) and
//                   release-increments the home's combine_done.
//   7. reduce:      home spins combine_done == valid_pairs (acquire), then
//                   out[t] = sum_k w[t,k] * combine[t][k] in fixed k order.
//
// Determinism: channel row order varies run-to-run (reservation races),
// but every row is computed independently and combined in fixed
// (token, k-ascending) order, so per-token outputs are deterministic —
// the same argument as the masked MoE path.

// ---- system-scope PTX helpers (the NVLink visibility primitives) ---------

__device__ __forceinline__ unsigned di_atom_add_sys(unsigned long long addr,
                                                    unsigned v) {
    unsigned r;
    asm volatile("atom.sys.global.add.u32 %0, [%1], %2;"
                 : "=r"(r) : "l"(addr), "r"(v) : "memory");
    return r;
}

__device__ __forceinline__ void di_red_add_sys_release(unsigned long long addr,
                                                       unsigned v) {
    asm volatile("red.release.sys.global.add.u32 [%0], %1;"
                 :: "l"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned di_ld_sys_acquire(unsigned long long addr) {
    unsigned r;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(r) : "l"(addr) : "memory");
    return r;
}

__device__ __forceinline__ void di_spin_until_ge(unsigned long long addr,
                                                 unsigned target) {
    while (di_ld_sys_acquire(addr) < target) {
#if !defined(__HIP__)
        __nanosleep(64);
#endif
    }
}

// Heap offset helpers (shared by every kernel below; the host mirrors them).
__device__ __forceinline__ long long ep_chan_off(int K) {
    return (((3 * (long long)K + 6) * 4) + 15) & ~15ll;
}
__device__ __forceinline__ int ep_row_bytes(int hidden) {
    return (hidden + (hidden >> 7) + 16 + 15) & ~15;
}
__device__ __forceinline__ long long ep_combine_off(int K, int cap, int hidden) {
    return ep_chan_off(K) + (long long)K * cap * ep_row_bytes(hidden);
}

// ---------------------------------------------------------------------------
// Step 1: reset the symmetric counters of ONE rank's heap.
// `cells` = 3*K + 6 i32 words (counts, produced, expected, step_done,
// combine_done, valid_pairs, pad). Data regions are not cleared (invalid
// slots are never read).
// ---------------------------------------------------------------------------
extern "C" __global__ void di_ep_reset_ll_kernel(unsigned long long heap,
                                                 int cells) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < cells) ((volatile int*)heap)[i] = 0;
}

// ---------------------------------------------------------------------------
// Step 2: dispatch. grid = (tokens, K); block = 128 threads per (token, k).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128)
di_ep_dispatch_ll_kernel(const unsigned char* __restrict__ xq,
                         const unsigned char* __restrict__ xsf,
                         const int* __restrict__ topk_idx,
                         const unsigned long long* __restrict__ peer_heaps,
                         const int* __restrict__ rank_map,
                         int my_rank, int world, int tokens, int hidden,
                         int K, int cap, int num_experts) {
    const int t = blockIdx.x;
    const int kk = blockIdx.y;
    if (t >= tokens || kk >= K) return;
    const int e = topk_idx[t * K + kk];
    if (e < 0) return; // padding slot
    // EPLB placement when a rank map is present; otherwise the balanced
    // contiguous-range map (the DeepEP default).
    const int dst = rank_map ? rank_map[e]
                             : (int)(((long long)e * world) / num_experts);
    const unsigned long long hd = peer_heaps[dst];
    const int sf_row = hidden >> 7;
    const int rb = ep_row_bytes(hidden);
    // Reserve a slot in the destination channel kk.
    const unsigned slot = di_atom_add_sys(hd + 4ull * (unsigned)kk, 1u);
    if (slot >= (unsigned)cap) return; // capacity drop (caller sizes cap)
    // Row layout: [hidden e4m3][sf_row ue8m0][meta 16].
    unsigned char* row = (unsigned char*)(hd + ep_chan_off(K)
        + ((long long)kk * cap + slot) * rb);
    const unsigned char* q = xq + (u64)t * hidden;
    const unsigned char* s = xsf + (u64)t * sf_row;
    for (int i = (int)threadIdx.x; i < hidden; i += 128) row[i] = q[i];
    for (int i = (int)threadIdx.x; i < sf_row; i += 128) row[hidden + i] = s[i];
    if (threadIdx.x < 4) {
        int* meta = (int*)(row + hidden + sf_row);
        meta[0] = my_rank;
        meta[1] = t;
        meta[2] = kk;
        meta[3] = e;
    }
    __syncthreads(); // join all payload writes before the release publish
    if (threadIdx.x == 0) {
        di_red_add_sys_release(hd + 4ull * (K + kk), 1u);      // produced
        di_red_add_sys_release(hd + 4ull * (2 * K + kk), 1u);  // expected
        // Own-token pair bookkeeping for the reduce step (plain atomic:
        // own heap, own stream; every valid pair of THIS rank comes back
        // to this rank's combine buffer exactly once).
        atomicAdd((int*)(peer_heaps[my_rank] + 4ull * (3 * K + 2)), 1);
    }
}

// ---------------------------------------------------------------------------
// Step 3: dispatch-done barrier. One block per rank (launched after that
// rank's dispatch on its stream): bumps step_done on every heap.
// ---------------------------------------------------------------------------
extern "C" __global__ void di_ep_dispatch_done_ll_kernel(
        const unsigned long long* __restrict__ peer_heaps, int world, int K) {
    if (threadIdx.x == 0) {
        for (int d = 0; d < world; ++d) {
            di_red_add_sys_release(peer_heaps[d] + 4ull * (3 * K), 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4: group. Receiver-side scatter of channel rows into the masked
// per-expert layout. grid = (K * ceil(cap / GR),); block = GR threads.
// After the step_done barrier the channel contents are final; each row is
// claimed for expert e with a local atomic rank and written to
// aq[e*cap + rank], asf[...], with meta2 = {src, token, k} kept for combine.
// ---------------------------------------------------------------------------
#define EP_LL_GR 128

extern "C" __global__ void __launch_bounds__(EP_LL_GR)
di_ep_group_ll_kernel(const unsigned long long* __restrict__ peer_heaps,
                      int my_rank, int world, int K, int cap, int hidden,
                      int* __restrict__ counters,
                      unsigned char* __restrict__ aq,
                      unsigned char* __restrict__ asf,
                      int* __restrict__ meta2, int* __restrict__ drops) {
    __shared__ int s_rank;
    // Every block waits for the full step barrier (cheap after the first).
    if (threadIdx.x == 0) {
        const unsigned long long my0 = peer_heaps[my_rank];
        di_spin_until_ge(my0 + 4ull * (3 * K), (unsigned)world);
    }
    __syncthreads();
    const int sf_row = hidden >> 7;
    const int rb = ep_row_bytes(hidden);
    const long long choff = ep_chan_off(K);
    const int tiles = (cap + EP_LL_GR - 1) / EP_LL_GR;
    const int kk = blockIdx.x / tiles;
    const int r0 = (blockIdx.x - kk * tiles) * EP_LL_GR;
    const unsigned long long my = peer_heaps[my_rank];
    const unsigned n = (unsigned)di_ld_sys_acquire(my + 4ull * (K + kk));
    const unsigned lim = min(n, (unsigned)cap);
    const unsigned char* base = (const unsigned char*)(my + choff + (long long)kk * cap * rb);
    for (int r = r0; r < (int)lim; ++r) {
        const unsigned char* row = base + (long long)r * rb;
        const int* meta = (const int*)(row + hidden + sf_row);
        const int src = meta[0];
        const int tok = meta[1];
        const int e = meta[3];
        if (threadIdx.x == 0) s_rank = atomicAdd(&counters[e], 1);
        __syncthreads();
        if (s_rank < cap) {
            unsigned char* dq = aq + ((long long)e * cap + s_rank) * hidden;
            unsigned char* ds = asf + ((long long)e * cap + s_rank) * sf_row;
            for (int i = (int)threadIdx.x; i < hidden; i += EP_LL_GR) dq[i] = row[i];
            for (int i = (int)threadIdx.x; i < sf_row; i += EP_LL_GR) ds[i] = row[hidden + i];
            if (threadIdx.x == 0) {
                int* m2 = meta2 + ((long long)e * cap + s_rank) * 4;
                m2[0] = src; m2[1] = tok; m2[2] = kk; m2[3] = e;
            }
        } else if (threadIdx.x == 0) {
            atomicAdd(drops, 1);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Step 6: combine. grid = (ceil(cap / GR), num_experts): each valid masked
// row y[e*cap + rank] is written to the home rank's combine[tok][kk] slot
// and the home's combine_done is release-incremented.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(EP_LL_GR)
di_ep_combine_ll_kernel(const float* __restrict__ y,
                        const int* __restrict__ meta2,
                        const int* __restrict__ counters,
                        const unsigned long long* __restrict__ peer_heaps,
                        int K, int cap, int hidden, int num_experts) {
    const int e = blockIdx.y;
    if (e >= num_experts) return;
    const int r0 = blockIdx.x * EP_LL_GR;
    const int m = min(counters[e], cap);
    const long long coff = ep_combine_off(K, cap, hidden);
    for (int r = r0; r < m; ++r) {
        const int* m2 = meta2 + ((long long)e * cap + r) * 4;
        const int src = m2[0];
        const int tok = m2[1];
        const int kk = m2[2];
        const unsigned long long hs = peer_heaps[src];
        float* dst = (float*)(hs + coff + (((long long)tok * K + kk) * hidden) * 4);
        const float* srcrow = y + ((long long)e * cap + r) * hidden;
        for (int i = (int)threadIdx.x; i < hidden; i += EP_LL_GR) {
            dst[i] = srcrow[i];
        }
        __syncthreads(); // join the row write before the release publish
        if (threadIdx.x == 0) {
            di_red_add_sys_release(hs + 4ull * (3 * K + 1), 1u);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Step 7: reduce. out[t] = sum_k w[t,k] * combine[t][k] (fixed k-ascending
// order; invalid experts contribute exactly zero). Spins on the home's
// combine_done reaching valid_pairs (final: same-stream dispatch wrote it).
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
di_ep_reduce_ll_kernel(const unsigned long long* __restrict__ peer_heaps,
                       int my_rank, int K, int cap, int hidden, int tokens,
                       const float* __restrict__ topk_w,
                       const int* __restrict__ topk_idx,
                       float* __restrict__ out) {
    const unsigned long long my = peer_heaps[my_rank];
    if (threadIdx.x == 0) {
        const unsigned want = (unsigned)di_ld_sys_acquire(my + 4ull * (3 * K + 2));
        di_spin_until_ge(my + 4ull * (3 * K + 1), want);
    }
    __syncthreads();
    const float* comb = (const float*)(my + ep_combine_off(K, cap, hidden));
    const long long tot = (long long)tokens * hidden;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < tot;
         i += (long long)gridDim.x * blockDim.x) {
        const int t = (int)(i / hidden);
        const int h = (int)(i - (long long)t * hidden);
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            if (topk_idx[t * K + kk] < 0) continue;
            acc += topk_w[t * K + kk] * comb[((long long)t * K + kk) * hidden + h];
        }
        out[i] = acc;
    }
}

// ===========================================================================
// P1-9: the FUSED expert path (the MegaMoE / DeepEP-v2 shape).
//
// The masked per-expert workspace (aq/asf/meta2/counters/y) moves INTO the
// symmetric heap, so:
//   * `di_ep_group_fused_ll_kernel` scatters channel rows straight into the
//     heap's masked region using a LOCAL expert slot map (the EPLB hook:
//     logical -> (rank, slot) is a device-resident table, not a formula);
//   * the grouped masked GEMM runs directly over the heap layout (the
//     caller's `gemm_grouped_masked_nt` over heap slices — no staging
//     copies between communication and compute);
//   * `di_ep_publish_ll_kernel` replaces the row-copying combine: the
//     expert rank writes one 8-byte completion entry
//     `(my_rank << 48) | y_row_byte_offset` into the home's completion
//     table (a tiny remote write instead of a hidden*f32 row) and
//     release-increments the home's combine_done;
//   * `di_ep_reduce_remote_ll_kernel` reads the y rows DIRECTLY from the
//     peers' heaps through the completion table (peer reads over the same
//     symmetric pointers the dispatch already uses).
// Per-(token,k) traffic drops from write+read of a full f32 row to one
// 64-bit entry — the staging copy between expert compute and the home
// reduction is gone.
//
// Heap layout v2 (appended after the legacy combine region, 16B aligned):
//   [MB, MB + E*cap*hidden)          aq      masked e4m3 activations
//   [MS, MS + E*cap*(hidden/128))    asf     ue8m0 scales
//   [MM, MM + E*cap*16)              meta2   {src, token, k, slot} i32 x4
//   [MC, MC + E*4)                   counters (masked_m)
//   [MD, MD + 4)                     drops
//   [MY, MY + E*cap*hidden*4)        y       expert outputs (f32)
//   [MT, MT + tokens_max*K*8)        completion table (u64 entries)
// ===========================================================================

__device__ __forceinline__ long long ep_masked_off(int K, int cap, int tokens,
                                                   int hidden) {
    return (ep_combine_off(K, cap, hidden)
            + (long long)tokens * K * hidden * 4 + 15) & ~15ll;
}

// The fused-path heap sub-pointers of one rank (formulas mirrored on the
// host side in deepinfer-kernels::peer_ll).
struct EpFusedPtrs {
    unsigned char* aq;
    unsigned char* asf;
    int* meta2;
    int* counters;
    int* drops;
    float* y;
    unsigned long long* ctab;
    long long y_off;    // byte offset of y inside the heap (completion entries)
    long long ctab_off;
};

__device__ __forceinline__ EpFusedPtrs ep_fused_ptrs(unsigned long long heap,
                                                     int K, int cap, int tokens,
                                                     int hidden, int E) {
    EpFusedPtrs p;
    unsigned char* base = (unsigned char*)heap;
    long long mb = ep_masked_off(K, cap, tokens, hidden);
    p.aq = base + mb;
    p.asf = p.aq + (long long)E * cap * hidden;
    p.meta2 = (int*)(p.asf + (long long)E * cap * (hidden >> 7));
    p.counters = p.meta2 + (long long)E * cap * 4;
    p.drops = p.counters + E;
    p.y = (float*)(((unsigned long long)(p.drops + 1) + 15) & ~15ull);
    p.ctab = (unsigned long long*)((unsigned char*)p.y
                                    + (long long)E * cap * hidden * 4);
    p.y_off = (unsigned char*)p.y - base;
    p.ctab_off = (unsigned char*)p.ctab - base;
    return p;
}

// Fused group: channel rows -> the heap's masked region, local slot map.
// grid = (K * ceil(cap / GR),); block = GR. `slot_map[e]` is the local
// expert slot of logical expert e ON THIS RANK (the dispatch rank map and
// the slot map are two views of the same EPLB placement).
extern "C" __global__ void __launch_bounds__(EP_LL_GR)
di_ep_group_fused_ll_kernel(const unsigned long long* peer_heaps,
                            int my_rank, int world, int K, int cap, int hidden,
                            int E, int tokens,
                            const int* __restrict__ slot_map) {
    __shared__ int s_rank;
    if (threadIdx.x == 0) {
        const unsigned long long my0 = peer_heaps[my_rank];
        di_spin_until_ge(my0 + 4ull * (3 * K), (unsigned)world);
    }
    __syncthreads();
    const int sf_row = hidden >> 7;
    const int rb = ep_row_bytes(hidden);
    const long long choff = ep_chan_off(K);
    const int tiles = (cap + EP_LL_GR - 1) / EP_LL_GR;
    const int kk = blockIdx.x / tiles;
    const int r0 = (blockIdx.x - kk * tiles) * EP_LL_GR;
    const unsigned long long my = peer_heaps[my_rank];
    EpFusedPtrs p = ep_fused_ptrs(my, K, cap, tokens, hidden, E);
    const unsigned n = (unsigned)di_ld_sys_acquire(my + 4ull * (K + kk));
    const unsigned lim = min(n, (unsigned)cap);
    const unsigned char* base =
        (const unsigned char*)(my + choff + (long long)kk * cap * rb);
    for (int r = r0; r < (int)lim; ++r) {
        const unsigned char* row = base + (long long)r * rb;
        const int* meta = (const int*)(row + hidden + sf_row);
        const int src = meta[0];
        const int tok = meta[1];
        const int e = meta[3];
        const int slot = slot_map[e];
        if (threadIdx.x == 0) s_rank = atomicAdd(&p.counters[slot], 1);
        __syncthreads();
        if (s_rank < cap) {
            unsigned char* dq = p.aq + ((long long)slot * cap + s_rank) * hidden;
            unsigned char* ds = p.asf + ((long long)slot * cap + s_rank) * sf_row;
            for (int i = (int)threadIdx.x; i < hidden; i += EP_LL_GR) dq[i] = row[i];
            for (int i = (int)threadIdx.x; i < sf_row; i += EP_LL_GR) ds[i] = row[hidden + i];
            if (threadIdx.x == 0) {
                int* m2 = p.meta2 + ((long long)slot * cap + s_rank) * 4;
                m2[0] = src; m2[1] = tok; m2[2] = kk; m2[3] = slot;
            }
        } else if (threadIdx.x == 0) {
            atomicAdd(p.drops, 1);
        }
        __syncthreads();
    }
}

// Fused publish: per valid (slot, r) row, write the completion entry into
// the home rank's table and release-increment its combine_done. Entry:
// (my_rank << 48) | byte offset of y[(slot*cap + r)] inside MY heap.
// grid = (ceil(cap / GR), E); block = GR.
extern "C" __global__ void __launch_bounds__(EP_LL_GR)
di_ep_publish_ll_kernel(const unsigned long long* peer_heaps,
                        int my_rank, int K, int cap, int hidden, int E,
                        int tokens) {
    const int slot = blockIdx.y;
    if (slot >= E) return;
    const unsigned long long my = peer_heaps[my_rank];
    EpFusedPtrs p = ep_fused_ptrs(my, K, cap, tokens, hidden, E);
    const int m = min(p.counters[slot], cap);
    const int r0 = blockIdx.x * EP_LL_GR;
    for (int r = r0 + (int)threadIdx.x; r < m; r += EP_LL_GR) {
        const int* m2 = p.meta2 + ((long long)slot * cap + r) * 4;
        const int src = m2[0];
        const int tok = m2[1];
        const int kk = m2[2];
        unsigned long long entry = ((unsigned long long)(unsigned)my_rank << 48)
                                 | (unsigned long long)(p.y_off
                                      + (((long long)slot * cap + r) * hidden) * 4);
        unsigned long long* dst = (unsigned long long*)(
            peer_heaps[src] + p.ctab_off + ((long long)tok * K + kk) * 8);
        *dst = entry; // exactly one writer per (tok, kk)
        di_red_add_sys_release(peer_heaps[src] + 4ull * (3 * K + 1), 1u);
    }
}

// Fused reduce: read y rows directly from the peers' heaps through the
// completion table (fixed k-ascending order; the acquire on combine_done
// orders every expert's y writes before this read).
extern "C" __global__ void __launch_bounds__(256)
di_ep_reduce_remote_ll_kernel(const unsigned long long* peer_heaps,
                              int my_rank, int K, int cap, int hidden, int E,
                              int tokens,
                              const float* __restrict__ topk_w,
                              const int* __restrict__ topk_idx,
                              float* __restrict__ out) {
    const unsigned long long my = peer_heaps[my_rank];
    if (threadIdx.x == 0) {
        const unsigned want = (unsigned)di_ld_sys_acquire(my + 4ull * (3 * K + 2));
        di_spin_until_ge(my + 4ull * (3 * K + 1), want);
    }
    __syncthreads();
    EpFusedPtrs p = ep_fused_ptrs(my, K, cap, tokens, hidden, E);
    const long long tot = (long long)tokens * hidden;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < tot;
         i += (long long)gridDim.x * blockDim.x) {
        const int t = (int)(i / hidden);
        const int h = (int)(i - (long long)t * hidden);
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            if (topk_idx[t * K + kk] < 0) continue;
            unsigned long long entry = p.ctab[(long long)t * K + kk];
            int peer = (int)(entry >> 48);
            const float* yrow = (const float*)(
                peer_heaps[peer] + (entry & 0x0000FFFFFFFFFFFFull));
            acc += topk_w[t * K + kk] * yrow[h];
        }
        out[i] = acc;
    }
}

// EPLB feedback: per-expert histogram of one routing decision
// (`topk_idx [tokens, K]`, -1 slots skipped). Integer atomics: the result
// is order-independent, hence deterministic.
extern "C" __global__ void __launch_bounds__(256)
di_ep_topk_histogram_ll_kernel(const int* __restrict__ topk_idx,
                               int* __restrict__ hist,
                               int tokens, int K, int num_experts) {
    const long long tot = (long long)tokens * K;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < tot;
         i += (long long)gridDim.x * blockDim.x) {
        int e = topk_idx[i];
        if (e >= 0 && e < num_experts) atomicAdd(&hist[e], 1);
    }
}
