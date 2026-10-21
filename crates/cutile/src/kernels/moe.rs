//! MoE kernels: noaux-tc sigmoid routing, expert dispatch (m_indices
//! layout for the contiguous grouped GEMM), and weighted combine.
//!
//! Dispatch assigns each routed token an expert-local rank via device
//! atomics; the intermediate row order within an expert is therefore
//! scheduling-dependent, but the final output is invariant (each GEMM row
//! is independent and combine gathers through the reverse map) — the same
//! argument documented for the C++ kernels crate.
//!
//! The dispatch kernels write through raw pointers (Tile-IR waiver) and
//! take no partitioned output; hosts launch them with an explicit
//! `.grid(...)`, mirroring NVIDIA's nvfp4_moe routing kernels.

/// Sigmoid routing with optional `e_score_correction_bias` (noaux-tc):
/// selection by `sigmoid(logit) + bias`, weights are the bias-free
/// sigmoids renormalized to sum 1. Ties: `(score desc, expert asc)`.
#[cutile::module]
pub mod route_sigmoid_topk_module {
    use cutile::core::*;

    /// Per CTA (token): `topk_ids[t, k]`, `topk_weights[t, k]` over
    /// `logits: [tokens, E]` F32, `bias: [E]` F32 (ignored when
    /// `has_bias == 0`).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn route_sigmoid_topk<const E: i32, const K: i32>(
        topk_ids: &mut Tensor<i32, { [1, K] }>,
        topk_weights: &mut Tensor<f32, { [1, K] }>,
        logits: &Tensor<f32, { [-1, E] }>,
        bias: &Tensor<f32, { [E] }>,
        has_bias: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t = pid.0;

        let lg: Tile<f32, { [1, E] }> = logits.partition(const_shape![1, E]).load([t, 0i32]);
        let b: Tile<f32, { [E] }> = bias.partition(const_shape![E]).load([0i32]);
        let b2: Tile<f32, { [1, E] }> = b.reshape(const_shape![1, E]);

        let one: Tile<f32, { [1, E] }> = constant(1.0f32, const_shape![1, E]);
        let zero: Tile<f32, { [1, E] }> = constant(0.0f32, const_shape![1, E]);
        let sig: Tile<f32, { [1, E] }> = true_div(one, one + exp(zero - lg));
        let sel: Tile<f32, { [1, E] }> = if has_bias != 0 { sig + b2 } else { sig };

        // Iterative top-K extraction, ties -> lowest expert id.
        let neg_big: Tile<f32, { [1, E] }> =
            constant(0.0f32, const_shape![1, E]) - constant(1.0e30f32, const_shape![1, E]);
        let ids: Tile<i32, { [1, E] }> = iota(const_shape![E]).reshape(const_shape![1, E]);
        let mut masked: Tile<f32, { [1, E] }> = sel;
        let mut out_ids: Tile<i32, { [K] }> = broadcast_scalar(0i32, const_shape![K]);
        let mut out_w: Tile<f32, { [K] }> = broadcast_scalar(0.0f32, const_shape![K]);
        let mut w_sum: f32 = 0.0f32;
        for j in 0i32..K {
            let m: Tile<f32, { [1] }> = reduce_max(masked, 1i32);
            let m_s: f32 = tile_to_scalar(m.reshape(const_shape![]));
            let is_top: Tile<bool, { [1, E] }> =
                eq_tile(masked, broadcast_scalar(m_s, const_shape![1, E]));
            let int_max: Tile<i32, { [1, E] }> =
                broadcast_scalar(2147483647i32, const_shape![1, E]);
            let cand: Tile<i32, { [1, E] }> = select(is_top, ids, int_max);
            let idv: Tile<i32, { [1] }> = reduce_min(cand, 1i32);
            let id_s: i32 = tile_to_scalar(idv.reshape(const_shape![]));
            let keep: Tile<bool, { [1, E] }> =
                eq_tile(ids, broadcast_scalar(id_s, const_shape![1, E]));
            let wv: Tile<f32, { [1] }> = reduce_sum(select(keep, sig, zero), 1i32);
            let w_s: f32 = tile_to_scalar(wv.reshape(const_shape![]));
            masked = select(keep, neg_big, masked);

            let slot: Tile<i32, { [K] }> = broadcast_scalar(j, const_shape![K]);
            let one_hot: Tile<bool, { [K] }> = eq_tile(iota(const_shape![K]), slot);
            out_ids = select(one_hot, broadcast_scalar(id_s, const_shape![K]), out_ids);
            out_w = select(one_hot, broadcast_scalar(w_s, const_shape![K]), out_w);
            w_sum = w_sum + w_s;
        }

        // Renormalize the bias-free sigmoid weights.
        let w_safe: f32 = if w_sum > 0.0f32 { w_sum } else { 1.0f32 };
        topk_ids.store(out_ids.reshape(const_shape![1, K]));
        topk_weights.store(true_div(out_w, broadcast_scalar(w_safe, const_shape![K]))
            .reshape(const_shape![1, K]));
    }
}

pub use route_sigmoid_topk_module::route_sigmoid_topk;

/// Dispatch: per-expert histogram of routed tokens (atomic).
#[cutile::module]
pub mod dispatch_count_module {
    use cutile::core::*;

    /// `counts[e] += #{(t, k): ids[t*K + k] == e}` over `total = T*K`
    /// routed entries, via relaxed device atomics. `counts` must be
    /// zeroed. Launch grid: `(ceil(total / RT), 1, 1)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn dispatch_count<const RT: i32>(
        counts: &Tensor<i32, { [-1] }>,
        ids: &Tensor<i32, { [-1] }>,
        num_routed: i32,
        total: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile = pid.0;
        let id: Tile<i32, { [RT] }> =
            ids.partition(const_shape![RT]).load([tile]).reshape(const_shape![RT]);
        let lane: Tile<i32, { [RT] }> =
            iota(const_shape![RT]) + broadcast_scalar(tile * RT, const_shape![RT]);
        let ok: Tile<bool, { [RT] }> =
            lt_tile(lane, broadcast_scalar(total, const_shape![RT]))
                & lt_tile(id, broadcast_scalar(num_routed, const_shape![RT]));
        let zero: Tile<i32, { [RT] }> = broadcast_scalar(0i32, const_shape![RT]);
        let safe: Tile<i32, { [RT] }> = select(ok, id, zero);
        let one: Tile<i32, { [RT] }> = broadcast_scalar(1i32, const_shape![RT]);
        let inc: Tile<i32, { [RT] }> = select(ok, one, zero);

        let cb: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(counts.as_ptr()));
        let cb_b: PointerTile<*mut i32, { [RT] }> =
            cb.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let _: (Tile<i32, { [RT] }>, Token) = unsafe {
            atomic_rmw_tko(
                cb_b.offset_tile(safe),
                inc,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                Some(ok),
                None,
            )
        };
    }
}

pub use dispatch_count_module::dispatch_count;

/// Dispatch layout builder: BM-aligned expert segments plus the
/// permutation and reverse maps for gather/combine.
#[cutile::module]
pub mod dispatch_permute_module {
    use cutile::core::*;

    /// Inputs: `ids: [T*K]`, `offsets: [E]` (padded exclusive prefix),
    /// `rank_counters: [E]` (zeroed). Outputs (written via pointers):
    /// `m_indices: [M_pad]`, `src_rows: [M_pad]`, `rev: [T*K]`, plus the
    /// incremented rank counters. Launch grid `(ceil(total / RT), 1, 1)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn dispatch_permute<const RT: i32>(
        m_indices: &Tensor<i32, { [-1] }>,
        src_rows: &Tensor<i32, { [-1] }>,
        rev: &Tensor<i32, { [-1] }>,
        rank_counters: &Tensor<i32, { [-1] }>,
        ids: &Tensor<i32, { [-1] }>,
        offsets: &Tensor<i32, { [-1] }>,
        total: i32,
        k: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile = pid.0;
        let id: Tile<i32, { [RT] }> =
            ids.partition(const_shape![RT]).load([tile]).reshape(const_shape![RT]);
        let lane: Tile<i32, { [RT] }> =
            iota(const_shape![RT]) + broadcast_scalar(tile * RT, const_shape![RT]);
        let ok: Tile<bool, { [RT] }> = lt_tile(lane, broadcast_scalar(total, const_shape![RT]));
        let tok: Tile<i32, { [RT] }> = lane / broadcast_scalar(k, const_shape![RT]);

        // offsets[e] gathered per lane via pointers.
        let ob: PointerTile<*const i32, { [] }> = pointer_to_tile(offsets.as_ptr());
        let ob_b: PointerTile<*const i32, { [RT] }> =
            ob.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let zero: Tile<i32, { [RT] }> = broadcast_scalar(0i32, const_shape![RT]);
        let safe_id: Tile<i32, { [RT] }> = select(ok, id, zero);
        let (off, _): (Tile<i32, { [RT] }>, Token) = unsafe {
            load_ptr_tko(
                ob_b.offset_tile(safe_id),
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };

        // Claim a rank within each expert via atomic add.
        let one: Tile<i32, { [RT] }> = broadcast_scalar(1i32, const_shape![RT]);
        let inc: Tile<i32, { [RT] }> = select(ok, one, zero);
        let rc: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(rank_counters.as_ptr()));
        let rc_b: PointerTile<*mut i32, { [RT] }> =
            rc.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let (rank, _): (Tile<i32, { [RT] }>, Token) = unsafe {
            atomic_rmw_tko(
                rc_b.offset_tile(safe_id),
                inc,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                Some(ok),
                None,
            )
        };

        // dest = offsets[e] + rank
        let off_safe: Tile<i32, { [RT] }> = select(ok, off, zero);
        let dest: Tile<i32, { [RT] }> = off_safe + rank;

        // Scatter m_indices[dest] = e, src_rows[dest] = t, rev[lane] = dest.
        let mi: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(m_indices.as_ptr()));
        let sr: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(src_rows.as_ptr()));
        let rv: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(rev.as_ptr()));
        let mi_b: PointerTile<*mut i32, { [RT] }> =
            mi.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let sr_b: PointerTile<*mut i32, { [RT] }> =
            sr.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let rv_b: PointerTile<*mut i32, { [RT] }> =
            rv.reshape(const_shape![1]).broadcast(const_shape![RT]);
        let _: Token = unsafe {
            store_ptr_tko(
                mi_b.offset_tile(dest),
                id,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ok),
                None::<Token>,
                Latency::<0>,
            )
        };
        let _: Token = unsafe {
            store_ptr_tko(
                sr_b.offset_tile(dest),
                tok,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ok),
                None::<Token>,
                Latency::<0>,
            )
        };
        let _: Token = unsafe {
            store_ptr_tko(
                rv_b.offset_tile(lane),
                dest,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ok),
                None::<Token>,
                Latency::<0>,
            )
        };
    }
}

pub use dispatch_permute_module::dispatch_permute;

/// Exclusive prefix sum of a small i32 tensor (single CTA).
#[cutile::module]
pub mod exclusive_scan_module {
    use cutile::core::*;

    /// `out[i] = Σ_{j < i} in[j]` over one `[N]` tile.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn exclusive_scan<const N: i32>(
        out: &mut Tensor<i32, { [N] }>,
        input: &Tensor<i32, { [-1] }>,
    ) {
        let x: Tile<i32, { [N] }> = input.partition(const_shape![N]).load([0i32]);
        let incl: Tile<i32, { [N] }> = scan_sum(x, 0i32, reverse::Forward, 0i32);
        out.store(incl - x);
    }
}

pub use exclusive_scan_module::exclusive_scan;

/// Combine: `out[t] = Σ_k w[t, k] · y[rev[t*K + k]]`.
#[cutile::module]
pub mod combine_gather_module {
    use cutile::core::*;

    /// Weighted gather of expert outputs back into token order over
    /// `y: [M_pad, H]` BF16, `rev: [T*K]`, `w: [T, K]` F32.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn combine_gather<const BH: i32>(
        out: &mut Tensor<bf16, { [1, BH] }>,
        y: &Tensor<bf16, { [-1, -1] }>,
        rev: &Tensor<i32, { [-1] }>,
        w: &Tensor<f32, { [-1, -1] }>,
        k: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t = pid.0;
        let c = pid.1;
        let y_part: Partition<bf16, { [1, BH] }> = y.partition(const_shape![1, BH]);
        let mut acc: Tile<f32, { [1, BH] }> = constant(0.0f32, const_shape![1, BH]);
        for j in 0i32..k {
            let dest: i32 = tile_to_scalar(
                rev.partition(const_shape![1])
                    .load([t * k + j])
                    .reshape(const_shape![]),
            );
            let wv: f32 = tile_to_scalar(
                w.partition(const_shape![1, 1]).load([t, j]).reshape(const_shape![]),
            );
            let row: Tile<bf16, { [1, BH] }> = y_part.load([dest, c]);
            let row_f: Tile<f32, { [1, BH] }> = convert_tile(row);
            acc = acc + row_f * broadcast_scalar(wv, const_shape![1, BH]);
        }
        out.store(convert_tile(acc));
    }
}

pub use combine_gather_module::combine_gather;
