//! DSA (DeepSeek Sparse Attention) kernels: the "thunder" weighted-ReLU
//! MQA indexer and the exact two-stage top-K selection.
//!
//! Selection order is `(score desc, position asc)` — bit-identical to the
//! CPU reference — and the output positions are **ascending**, padded with
//! `-1` to exactly `K = min(topk_tokens, s_kv)`:
//!
//! * Stage 1 ([`dsa_block_preselect`]) keeps the top `P` entries per 64-key
//!   block, written *position-sorted* within each block's `P` slots, so the
//!   whole survivor stream is position-ascending.
//! * Stage 2 ([`dsa_topk_select`]) finds the k-th largest score by
//!   bit-constructing the threshold over the order-preserving u32 key map
//!   (`positive: bits | 1<<31`, `negative: ~bits`), then compacts survivors
//!   scanning left-to-right — which yields ascending positions with exact
//!   tie handling for free.

/// Thunder indexer: `out[s, n] = Σ_h w[s, h] · relu(q[s, h, :] · k[n, h, :])`.
#[cutile::module]
pub mod thunder_logits_module {
    use cutile::core::*;

    /// Weighted-ReLU MQA scoring over `q_flat: [s, H*D]`,
    /// `k_flat: [n, H*D]`, `w: [s, H]` (BF16 data, F32 weights).
    /// Accumulates in F32; `out: [s, n]` F32.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn thunder_logits<const BM: i32, const BN: i32, const D: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        q_flat: &Tensor<bf16, { [-1, -1] }>,
        k_flat: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<f32, { [-1, -1] }>,
        heads: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_q = q_flat.partition(const_shape![BM, D]);
        let part_k = k_flat.partition(const_shape![BN, D]);
        let part_w = w.partition(const_shape![BM, 1]);

        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for h in 0i32..heads {
            let q: Tile<bf16, { [BM, D] }> = part_q.load([pid.0, h]);
            let k: Tile<bf16, { [BN, D] }> = part_k.load([pid.1, h]);
            let mut dot: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            dot = mma(q, k.transpose(), dot);
            let relu: Tile<f32, { [BM, BN] }> = max_tile(dot, zero);
            let wv: Tile<f32, { [BM, 1] }> = part_w.load([pid.0, h]);
            let wt: Tile<f32, { [BM, BN] }> = wv.broadcast(const_shape![BM, BN]);
            acc = acc + relu * wt;
        }
        out.store(acc);
    }
}

pub use thunder_logits_module::thunder_logits;

/// Stage 1: per-64-key-block top-P preselection, written position-sorted.
#[cutile::module]
pub mod dsa_block_preselect_module {
    use cutile::core::*;

    /// Per CTA `(query, block)`: keeps the top `P` scores of the block by
    /// `(score desc, pos asc)`, then writes them to
    /// `surv_s[q, block*P .. block*P+P]` / `surv_i[...]` in ascending
    /// position order (missing entries: score `-inf`, index `-1`).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn dsa_block_preselect<const P: i32>(
        surv_s: &mut Tensor<f32, { [1, P] }>,
        surv_i: &mut Tensor<i32, { [1, P] }>,
        scores: &Tensor<f32, { [-1, -1] }>,
        s_kv: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q = pid.0;
        let block = pid.1;

        let sc: Tile<f32, { [1, 64] }> = scores.partition(const_shape![1, 64]).load([q, block]);
        let pos: Tile<i32, { [1, 64] }> =
            iota(const_shape![64]).reshape(const_shape![1, 64])
                + broadcast_scalar(block * 64, const_shape![1, 64]);
        // Out-of-window lanes score -inf.
        let neg_inf: Tile<f32, { [1, 64] }> =
            constant(0.0f32, const_shape![1, 64]) - constant(1.0e30f32, const_shape![1, 64]);
        let lane_ok: Tile<bool, { [1, 64] }> =
            lt_tile(pos, broadcast_scalar(s_kv, const_shape![1, 64]));
        let sc: Tile<f32, { [1, 64] }> = select(lane_ok, sc, neg_inf);

        // Mark the top-P lanes (ties: lowest position wins — the eq-mark
        // takes every lane equal to the running max, and positions ascend).
        let mut taken: Tile<bool, { [1, 64] }> = constant(false, const_shape![1, 64]);
        for _p in 0i32..P {
            let masked: Tile<f32, { [1, 64] }> = select(taken, neg_inf, sc);
            let m: Tile<f32, { [1] }> = reduce_max(masked, 1i32);
            let m_s: f32 = tile_to_scalar(m.reshape(const_shape![]));
            let is_top: Tile<bool, { [1, 64] }> =
                eq_tile(masked, broadcast_scalar(m_s, const_shape![1, 64]))
                    & eq_tile(taken, constant(false, const_shape![1, 64]));
            taken = taken | is_top;
        }

        // Extract the P survivors in ascending position order (P passes of
        // min-position extraction among the marked lanes).
        let false_t: Tile<bool, { [1, 64] }> = constant(false, const_shape![1, 64]);
        let int_max: Tile<i32, { [1, 64] }> =
            broadcast_scalar(2147483647i32, const_shape![1, 64]);
        let mut written: Tile<bool, { [1, 64] }> = constant(false, const_shape![1, 64]);
        let ss_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(cast_mut(surv_s.as_ptr()));
        let si_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(surv_i.as_ptr()));
        let p_idx: Tile<i32, { [P] }> = iota(const_shape![P]);
        let ss_ptrs: PointerTile<*mut f32, { [P] }> = ss_base
            .reshape(const_shape![1])
            .broadcast(const_shape![P])
            .offset_tile(p_idx);
        let si_ptrs: PointerTile<*mut i32, { [P] }> = si_base
            .reshape(const_shape![1])
            .broadcast(const_shape![P])
            .offset_tile(p_idx);

        let mut sel_pos: Tile<i32, { [P] }> = broadcast_scalar(-1i32, const_shape![P]);
        let mut sel_sc: Tile<f32, { [P] }> = broadcast_scalar(-1.0e30f32, const_shape![P]);
        for j in 0i32..P {
            let cand: Tile<i32, { [1, 64] }> =
                select(taken & eq_tile(written, false_t), pos, int_max);
            let mp: Tile<i32, { [1] }> = reduce_min(cand, 1i32);
            let mp_s: i32 = tile_to_scalar(mp.reshape(const_shape![]));
            let is_min: Tile<bool, { [1, 64] }> =
                eq_tile(cand, broadcast_scalar(mp_s, const_shape![1, 64]));
            written = written | is_min;
            let sc_here: Tile<f32, { [1, 64] }> = select(is_min, sc, neg_inf);
            let sv: Tile<f32, { [1] }> = reduce_max(sc_here, 1i32);
            let sv_s: f32 = tile_to_scalar(sv.reshape(const_shape![]));

            let slot: Tile<i32, { [P] }> = broadcast_scalar(j, const_shape![P]);
            let one_hot: Tile<bool, { [P] }> = eq_tile(iota(const_shape![P]), slot);
            sel_pos = select(one_hot, broadcast_scalar(mp_s, const_shape![P]), sel_pos);
            sel_sc = select(one_hot, broadcast_scalar(sv_s, const_shape![P]), sel_sc);
        }
        let _: Token = unsafe {
            store_ptr_tko(
                ss_ptrs,
                sel_sc,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
        let _: Token = unsafe {
            store_ptr_tko(
                si_ptrs,
                sel_pos,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None::<Token>,
                Latency::<0>,
            )
        };
    }
}

pub use dsa_block_preselect_module::dsa_block_preselect;

/// Stage 2: exact top-K over the survivor stream (bit-constructed
/// threshold), compacted in stream order (= ascending positions).
#[cutile::module]
pub mod dsa_topk_select_module {
    use cutile::core::*;

    /// Order-preserving u32 key: positive floats map above all negatives.
    fn key_of<const ST: i32>(sc: Tile<f32, { [1, ST] }>) -> Tile<u32, { [1, ST] }> {
        let bits: Tile<u32, { [1, ST] }> = bitcast(sc);
        let sign_hi: Tile<u32, { [1, ST] }> =
            broadcast_scalar(2147483648u32, const_shape![1, ST]);
        let neg: Tile<bool, { [1, ST] }> =
            cmpi(bits, sign_hi, predicate::GreaterThanOrEqual);
        let flipped: Tile<u32, { [1, ST] }> =
            xori(bits, broadcast_scalar(4294967295u32, const_shape![1, ST]));
        let positive: Tile<u32, { [1, ST] }> = ori(bits, sign_hi);
        select(neg, flipped, positive)
    }

    /// Per CTA (query): finds the k-th largest survivor score, then writes
    /// the selected positions to `out[q, 0..K]` in ascending order
    /// (`-1` padding when fewer than `K` keys exist; `out` is pre-filled
    /// with `-1` by the caller). Grid `(queries,)` via the `[1, K]` output
    /// partition.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn dsa_topk_select<const ST: i32, const K: i32>(
        out: &mut Tensor<i32, { [1, K] }>,
        surv_s: &Tensor<f32, { [-1, -1] }>,
        surv_i: &Tensor<i32, { [-1, -1] }>,
        nslots: i32,
        k: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q = pid.0;
        let n_tiles: i32 = ceil_div(nslots, ST);

        let one_i: Tile<i32, { [1, ST] }> = broadcast_scalar(1i32, const_shape![1, ST]);
        let zero_i: Tile<i32, { [1, ST] }> = broadcast_scalar(0i32, const_shape![1, ST]);

        // Pass A: bit-construct the k-th largest key.
        let mut thresh: u32 = 0u32;
        for i in 0i32..32 {
            let bit: i32 = 31 - i;
            let one_t: Tile<u32, { [] }> = scalar_to_tile(1u32);
            let bit_t: Tile<u32, { [] }> = scalar_to_tile(bit as u32);
            let add: Tile<u32, { [] }> = shli(one_t, bit_t, overflow::None);
            let add_s: u32 = tile_to_scalar(add);
            let cand: u32 = thresh | add_s;
            let mut count: i32 = 0i32;
            for t in 0i32..n_tiles {
                let sc: Tile<f32, { [1, ST] }> =
                    surv_s.partition(const_shape![1, ST]).load([q, t]);
                let key: Tile<u32, { [1, ST] }> = key_of::<ST>(sc);
                let cand_t: Tile<u32, { [1, ST] }> =
                    broadcast_scalar(cand, const_shape![1, ST]);
                let ge: Tile<bool, { [1, ST] }> =
                    cmpi(key, cand_t, predicate::GreaterThanOrEqual);
                let c: Tile<i32, { [1] }> = reduce_sum(select(ge, one_i, zero_i), 1i32);
                let c_s: i32 = tile_to_scalar(c.reshape(const_shape![]));
                count = count + c_s;
            }
            if count >= k {
                thresh = cand;
            }
        }

        // Pass B: strictly-greater count and the tie budget.
        let mut count_gt: i32 = 0i32;
        for t in 0i32..n_tiles {
            let sc: Tile<f32, { [1, ST] }> = surv_s.partition(const_shape![1, ST]).load([q, t]);
            let key: Tile<u32, { [1, ST] }> = key_of::<ST>(sc);
            let th_t: Tile<u32, { [1, ST] }> = broadcast_scalar(thresh, const_shape![1, ST]);
            let gt: Tile<bool, { [1, ST] }> = cmpi(key, th_t, predicate::GreaterThan);
            let c: Tile<i32, { [1] }> = reduce_sum(select(gt, one_i, zero_i), 1i32);
            let c_s: i32 = tile_to_scalar(c.reshape(const_shape![]));
            count_gt = count_gt + c_s;
        }
        let need_eq: i32 = k - count_gt;

        // Pass C: compact in stream order; ties take the first `need_eq`.
        let ob: PointerTile<*mut i32, { [] }> = pointer_to_tile(cast_mut(out.as_ptr()));
        let mut base: i32 = 0i32;
        let mut eq_taken: i32 = 0i32;
        for t in 0i32..n_tiles {
            let sc: Tile<f32, { [1, ST] }> = surv_s.partition(const_shape![1, ST]).load([q, t]);
            let pos: Tile<i32, { [1, ST] }> = surv_i.partition(const_shape![1, ST]).load([q, t]);
            let lane: Tile<i32, { [1, ST] }> =
                iota(const_shape![ST]).reshape(const_shape![1, ST])
                    + broadcast_scalar(t * ST, const_shape![1, ST]);
            let lane_ok: Tile<bool, { [1, ST] }> =
                lt_tile(lane, broadcast_scalar(nslots, const_shape![1, ST]));

            // Tail lanes: key 0 (below everything real) and position -1,
            // so they only ever fill padding slots.
            let key: Tile<u32, { [1, ST] }> = key_of::<ST>(sc);
            let zero_u: Tile<u32, { [1, ST] }> = broadcast_scalar(0u32, const_shape![1, ST]);
            let key: Tile<u32, { [1, ST] }> = select(lane_ok, key, zero_u);
            let pos: Tile<i32, { [1, ST] }> =
                select(lane_ok, pos, broadcast_scalar(-1i32, const_shape![1, ST]));

            let th_t: Tile<u32, { [1, ST] }> = broadcast_scalar(thresh, const_shape![1, ST]);
            let gt: Tile<bool, { [1, ST] }> = cmpi(key, th_t, predicate::GreaterThan);
            let eq: Tile<bool, { [1, ST] }> = eq_tile(key, th_t) & lane_ok;
            let eq_rank: Tile<i32, { [1, ST] }> =
                scan_sum(select(eq, one_i, zero_i), 1i32, reverse::Forward, 0i32);
            let eq_ok: Tile<bool, { [1, ST] }> = eq & lt_tile(
                eq_rank,
                broadcast_scalar(need_eq - eq_taken + 1, const_shape![1, ST]),
            );
            let sel: Tile<bool, { [1, ST] }> = gt | eq_ok;

            let rank: Tile<i32, { [1, ST] }> =
                scan_sum(select(sel, one_i, zero_i), 1i32, reverse::Forward, 0i32);
            let dst: Tile<i32, { [1, ST] }> =
                rank + broadcast_scalar(base - 1, const_shape![1, ST]);
            let out_ptrs: PointerTile<*mut i32, { [1, ST] }> = ob
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, ST])
                .offset_tile(dst);
            let _: Token = unsafe {
                store_ptr_tko(
                    out_ptrs,
                    pos,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(sel),
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let cnt: Tile<i32, { [1] }> = reduce_sum(select(sel, one_i, zero_i), 1i32);
            let cnt_s: i32 = tile_to_scalar(cnt.reshape(const_shape![]));
            base = base + cnt_s;
            let eq_cnt: Tile<i32, { [1] }> = reduce_sum(select(eq_ok, one_i, zero_i), 1i32);
            let eq_s: i32 = tile_to_scalar(eq_cnt.reshape(const_shape![]));
            eq_taken = eq_taken + eq_s;
        }
    }
}

pub use dsa_topk_select_module::dsa_topk_select;
