//! MLA attention kernels (FlashMLA V4.1 semantics): latent-512 + rope-64
//! absorbed attention with f32 online softmax.
//!
//! * [`mla_prefill_flash`] — dense causal prefill over contiguous KV
//!   (also the second stage of sparse prefill after [`gather_kv_rows`],
//!   with `causal = 0`).
//! * [`mla_decode_splitkv_bf16`] / [`mla_decode_splitkv_fp8`] — decode over
//!   a gathered/paged KV window, split across KV for SM occupancy, writing
//!   per-split partials + LSE; [`mla_split_merge`] combines them and applies
//!   the FlashMLA `attn_sink` epilogue.
//!
//! Numerics: scores accumulate in f32; the online softmax runs in the log2
//! domain (`exp2`); the final LSE is converted to natural log. Empty
//! selections produce zero outputs and `lse = +inf`, matching the CPU
//! reference. `sm_scale` is passed in the *natural-log* domain (callers
//! fold any DSA `mscale` into it).

/// Dense / gathered causal-or-unmasked prefill attention.
#[cutile::module]
pub mod mla_prefill_flash_module {
    use cutile::core::*;

    /// `out[bh, m, :] = softmax(scores) @ [ckv | kpe]` where
    /// `scores = (q_nope·ckv + q_pe·kpe) * sm_scale` and the attention
    /// window is causal when `causal != 0` (query row `m` attends keys
    /// `<= q_offset + m`, clipped to `kv_len`), otherwise the full
    /// gathered set. Tensors are laid out `q_*: [b*h, s, D]`,
    /// `ckv/kpe: [b, n, D]`, `out: [b*h, s, 576]`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 16,),
                      sm_103 = (max_divisibility = 16,),
                    ))]
    unsafe fn mla_prefill_flash<
        const BM: i32,
        const BN: i32,
        const DNOPE: i32,
        const DPE: i32,
        const D: i32,
        const HHEADS: i32,
    >(
        out: &mut Tensor<bf16, { [1, BM, D] }>,
        q_nope: &Tensor<bf16, { [-1, -1, DNOPE] }>,
        q_pe: &Tensor<bf16, { [-1, -1, DPE] }>,
        ckv: &Tensor<bf16, { [-1, -1, DNOPE] }>,
        kpe: &Tensor<bf16, { [-1, -1, DPE] }>,
        sm_scale: f32,
        q_offset: i32,
        kv_len: i32,
        causal: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let bh: i32 = pid.0;
        let m_tile: i32 = pid.1;
        let batch: i32 = bh / HHEADS;

        // sm_scale in log2 domain.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let ln2: f32 = tile_to_scalar(log(two));
        let scale2: Tile<f32, { [BM, BN] }> =
            broadcast_scalar(sm_scale / ln2, const_shape![BM, BN]);

        let qn_part = q_nope.partition(const_shape![1, BM, DNOPE]);
        let qp_part = q_pe.partition(const_shape![1, BM, DPE]);
        let qn: Tile<bf16, { [BM, DNOPE] }> =
            qn_part.load([bh, m_tile, 0i32]).reshape(const_shape![BM, DNOPE]);
        let qp: Tile<bf16, { [BM, DPE] }> =
            qp_part.load([bh, m_tile, 0i32]).reshape(const_shape![BM, DPE]);

        let ckv_part = ckv.partition(const_shape![1, BN, DNOPE]);
        let kpe_part = kpe.partition(const_shape![1, BN, DPE]);

        let neg_big: Tile<f32, { [BM, BN] }> =
            constant(0.0f32, const_shape![BM, BN]) - constant(1.0e30f32, const_shape![BM, BN]);
        let mut m_i: Tile<f32, { [BM, 1] }> =
            constant(0.0f32, const_shape![BM, 1]) - constant(1.0e30f32, const_shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc_n: Tile<f32, { [BM, DNOPE] }> = constant(0.0f32, const_shape![BM, DNOPE]);
        let mut acc_p: Tile<f32, { [BM, DPE] }> = constant(0.0f32, const_shape![BM, DPE]);

        // Causal upper bound of this query tile.
        let hi: i32 = min(q_offset + (m_tile + 1) * BM, kv_len);
        let n_tiles: i32 = ceil_div(if causal != 0 { hi } else { kv_len }, BN);

        let offs_m: Tile<i32, { [BM] }> =
            iota(const_shape![BM]) + broadcast_scalar(q_offset + m_tile * BM, const_shape![BM]);
        let offs_m2: Tile<i32, { [BM, 1] }> = offs_m.reshape(const_shape![BM, 1]);
        let offs_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n2: Tile<i32, { [1, BN] }> = offs_n.reshape(const_shape![1, BN]);

        for j in 0i32..n_tiles {
            let kv_n: Tile<bf16, { [BN, DNOPE] }> = ckv_part
                .load([batch, j, 0i32])
                .reshape(const_shape![BN, DNOPE]);
            let kv_p: Tile<bf16, { [BN, DPE] }> =
                kpe_part.load([batch, j, 0i32]).reshape(const_shape![BN, DPE]);

            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            qk = mma(qn, kv_n.transpose(), qk);
            qk = mma(qp, kv_p.transpose(), qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * scale2;

            // Validity: key index < kv_len always; causality when requested.
            let j_base: Tile<i32, { [1, BN] }> =
                broadcast_scalar(j * BN, const_shape![1, BN]);
            let key_idx: Tile<i32, { [BM, BN] }> =
                (j_base + offs_n2).broadcast(const_shape![BM, BN]);
            let kv_ok: Tile<bool, { [BM, BN] }> =
                lt_tile(key_idx, broadcast_scalar(kv_len, const_shape![BM, BN]));
            let valid: Tile<bool, { [BM, BN] }> = if causal != 0 {
                kv_ok & ge_tile(offs_m2.broadcast(const_shape![BM, BN]), key_idx)
            } else {
                kv_ok
            };
            let zero_scores: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, zero_scores + neg_big);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk_shift: Tile<f32, { [BM, BN] }> =
                qk - m_ij.broadcast(const_shape![BM, BN]);
            let p: Tile<f32, { [BM, BN] }> = exp2(qk_shift, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            let alpha_n: Tile<f32, { [BM, DNOPE] }> = alpha.broadcast(const_shape![BM, DNOPE]);
            let alpha_p: Tile<f32, { [BM, DPE] }> = alpha.broadcast(const_shape![BM, DPE]);
            acc_n = acc_n * alpha_n;
            acc_p = acc_p * alpha_p;

            let p_bf: Tile<bf16, { [BM, BN] }> = convert_tile(p);
            acc_n = mma(p_bf, kv_n, acc_n);
            acc_p = mma(p_bf, kv_p, acc_p);
            m_i = m_ij;
        }

        // Normalize (guard all-masked rows).
        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_safe: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        let out_n: Tile<bf16, { [BM, DNOPE] }> =
            convert_tile(true_div(acc_n, l_safe.broadcast(const_shape![BM, DNOPE])));
        let out_p: Tile<bf16, { [BM, DPE] }> =
            convert_tile(true_div(acc_p, l_safe.broadcast(const_shape![BM, DPE])));
        let both: Tile<bf16, { [BM, D] }> = cat(out_n, out_p, 1i32);
        out.store(both.reshape(const_shape![1, BM, D]));
    }
}

pub use mla_prefill_flash_module::mla_prefill_flash;

/// Gathers selected KV rows (DSA sparse prefill stage 1).
#[cutile::module]
pub mod gather_kv_rows_module {
    use cutile::core::*;

    /// `out_*[b, t, :] = ckv/kpe[indices[b, t], :]` — one CTA per gathered
    /// row; invalid (negative) indices gather row 0 (the caller masks them
    /// out via the attention validity rules).
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (max_divisibility = 8,),
                      sm_103 = (max_divisibility = 8,),
                    ))]
    unsafe fn gather_kv_rows<const DNOPE: i32, const DPE: i32>(
        out_nope: &mut Tensor<bf16, { [1, 1, DNOPE] }>,
        out_pe: &mut Tensor<bf16, { [1, 1, DPE] }>,
        ckv: &Tensor<bf16, { [-1, DNOPE] }>,
        kpe: &Tensor<bf16, { [-1, DPE] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b = pid.0;
        let t = pid.1;
        let idx: i32 = tile_to_scalar(
            indices
                .partition(const_shape![1, 1])
                .load([b, t])
                .reshape(const_shape![]),
        );
        let safe: i32 = if idx >= 0 { idx } else { 0i32 };
        let ckv_part = ckv.partition(const_shape![1, DNOPE]);
        let kpe_part = kpe.partition(const_shape![1, DPE]);
        let row_n: Tile<bf16, { [1, DNOPE] }> = ckv_part.load([safe, 0i32]);
        let row_p: Tile<bf16, { [1, DPE] }> = kpe_part.load([safe, 0i32]);
        out_nope.store(row_n.reshape(const_shape![1, 1, DNOPE]));
        out_pe.store(row_p.reshape(const_shape![1, 1, DPE]));
    }
}

pub use gather_kv_rows_module::gather_kv_rows;

/// Split-KV decode attention over a gathered KV window, BF16 cache.
#[cutile::module]
pub mod mla_decode_splitkv_bf16_module {
    use cutile::core::*;

    /// Per CTA `(b, kv-group, split)`: processes its share of the
    /// `indices[b, 0..topk]` physical KV rows in `BN`-row tiles, writing
    /// normalized per-split partials (`att_*_out[b, g, split*GROUP.., :]`)
    /// and log2-domain LSE. Grid `(b, n_groups, NSPLITS)`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (occupancy = 1, max_divisibility = 16,),
                      sm_103 = (occupancy = 1, max_divisibility = 16,),
                    ))]
    unsafe fn mla_decode_splitkv_bf16<
        const GROUP: i32,
        const BN: i32,
        const DNOPE: i32,
        const DPE: i32,
        const NSPLITS: i32,
    >(
        att_nope_out: &mut Tensor<bf16, { [1, GROUP, DNOPE] }>,
        att_pe_out: &mut Tensor<bf16, { [1, GROUP, DPE] }>,
        lse_out: &mut Tensor<f32, { [1, GROUP] }>,
        q_nope: &Tensor<bf16, { [-1, GROUP, DNOPE] }>,
        q_pe: &Tensor<bf16, { [-1, GROUP, DPE] }>,
        ckv: &Tensor<bf16, { [-1, DNOPE] }>,
        kpe: &Tensor<bf16, { [-1, DPE] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        ngroups: i32,
        topk: i32,
        sm_scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row: i32 = pid.0;
        let split: i32 = pid.1;
        let b: i32 = row / ngroups;

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let ln2: f32 = tile_to_scalar(log(two));
        let scale2: Tile<f32, { [GROUP, BN] }> =
            broadcast_scalar(sm_scale / ln2, const_shape![GROUP, BN]);

        let qn: Tile<bf16, { [GROUP, DNOPE] }> = q_nope
            .partition(const_shape![1, GROUP, DNOPE])
            .load([row, 0i32, 0i32])
            .reshape(const_shape![GROUP, DNOPE]);
        let qp: Tile<bf16, { [GROUP, DPE] }> = q_pe
            .partition(const_shape![1, GROUP, DPE])
            .load([row, 0i32, 0i32])
            .reshape(const_shape![GROUP, DPE]);

        // Split range over BN-row tiles of the gathered window.
        let total_tiles: i32 = ceil_div(topk, BN);
        let tiles_per_split: i32 = ceil_div(total_tiles, NSPLITS);
        let start_tile: i32 = split * tiles_per_split;
        let mut end_tile: i32 = start_tile + tiles_per_split;
        end_tile = min(end_tile, total_tiles);

        let neg_big_g: Tile<f32, { [GROUP, BN] }> =
            constant(0.0f32, const_shape![GROUP, BN]) - constant(1.0e30f32, const_shape![GROUP, BN]);
        let mut m_i: Tile<f32, { [GROUP, 1] }> =
            constant(0.0f32, const_shape![GROUP, 1]) - constant(1.0e30f32, const_shape![GROUP, 1]);
        let mut l_i: Tile<f32, { [GROUP, 1] }> = constant(0.0f32, const_shape![GROUP, 1]);
        let mut acc_n: Tile<f32, { [GROUP, DNOPE] }> = constant(0.0f32, const_shape![GROUP, DNOPE]);
        let mut acc_p: Tile<f32, { [GROUP, DPE] }> = constant(0.0f32, const_shape![GROUP, DPE]);

        // Gathered row pointers.
        let ckv_base: PointerTile<*const bf16, { [] }> = pointer_to_tile(ckv.as_ptr());
        let kpe_base: PointerTile<*const bf16, { [] }> = pointer_to_tile(kpe.as_ptr());
        let idx_part = indices.partition(const_shape![1, BN]);
        let cols_n: Tile<i32, { [1, DNOPE] }> =
            iota(const_shape![DNOPE]).reshape(const_shape![1, DNOPE]);
        let cols_p: Tile<i32, { [1, DPE] }> = iota(const_shape![DPE]).reshape(const_shape![1, DPE]);
        let lane_n: Tile<i32, { [BN, DNOPE] }> =
            iota(const_shape![BN]).reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DNOPE]);
        let offs_n: Tile<i32, { [BN, DNOPE] }> =
            lane_n * broadcast_scalar(DNOPE, const_shape![BN, DNOPE])
                + cols_n.broadcast(const_shape![BN, DNOPE]);
        let offs_p: Tile<i32, { [BN, DPE] }> =
            iota(const_shape![BN]).reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DPE])
                * broadcast_scalar(DPE, const_shape![BN, DPE])
                + cols_p.broadcast(const_shape![BN, DPE]);

        let ckv_b: PointerTile<*const bf16, { [BN, DNOPE] }> = ckv_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, DNOPE]);
        let kpe_b: PointerTile<*const bf16, { [BN, DPE] }> = kpe_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, DPE]);

        for t in start_tile..end_tile {
            let rows: Tile<i32, { [BN] }> = idx_part.load([b, t]).reshape(const_shape![BN]);
            // -1 (invalid) and out-of-window lanes gather row 0; their scores
            // are masked to -1e30 below, so their (finite) data contributes 0.
            let safe_rows: Tile<i32, { [BN] }> = select(
                ge_tile(rows, broadcast_scalar(0i32, const_shape![BN])),
                rows,
                broadcast_scalar(0i32, const_shape![BN]),
            );
            let lane_ok: Tile<bool, { [BN] }> =
                lt_tile(iota(const_shape![BN]) + broadcast_scalar(t * BN, const_shape![BN]),
                        broadcast_scalar(topk, const_shape![BN]))
                    & ge_tile(rows, broadcast_scalar(0i32, const_shape![BN]));
            let row_off_n: Tile<i32, { [BN, DNOPE] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DNOPE])
                    * broadcast_scalar(DNOPE, const_shape![BN, DNOPE]);
            let row_off_p: Tile<i32, { [BN, DPE] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DPE])
                    * broadcast_scalar(DPE, const_shape![BN, DPE]);

            let (kv_n, tok_n): (Tile<bf16, { [BN, DNOPE] }>, Token) = unsafe {
                load_ptr_tko(
                    ckv_b.offset_tile(offs_n + row_off_n),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let (kv_p, _): (Tile<bf16, { [BN, DPE] }>, Token) = unsafe {
                load_ptr_tko(
                    kpe_b.offset_tile(offs_p + row_off_p),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };

            let mut qk: Tile<f32, { [GROUP, BN] }> = constant(0.0f32, const_shape![GROUP, BN]);
            qk = mma(qn, kv_n.transpose(), qk);
            qk = mma(qp, kv_p.transpose(), qk);
            let qk: Tile<f32, { [GROUP, BN] }> = qk * scale2;
            let zero_s: Tile<f32, { [GROUP, BN] }> = constant(0.0f32, const_shape![GROUP, BN]);
            let ok2: Tile<bool, { [GROUP, BN] }> =
                lane_ok.reshape(const_shape![1, BN]).broadcast(const_shape![GROUP, BN]);
            let qk: Tile<f32, { [GROUP, BN] }> = select(ok2, qk, zero_s + neg_big_g);

            let qk_max: Tile<f32, { [GROUP] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [GROUP, 1] }> = qk_max.reshape(const_shape![GROUP, 1]);
            let m_ij: Tile<f32, { [GROUP, 1] }> = max_tile(m_i, qk_max);
            let qk_shift: Tile<f32, { [GROUP, BN] }> =
                qk - m_ij.broadcast(const_shape![GROUP, BN]);
            let p: Tile<f32, { [GROUP, BN] }> = exp2(qk_shift, ftz::Disabled);
            let l_ij: Tile<f32, { [GROUP] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [GROUP, 1] }> = l_ij.reshape(const_shape![GROUP, 1]);
            let alpha: Tile<f32, { [GROUP, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc_n = acc_n * alpha.broadcast(const_shape![GROUP, DNOPE]);
            acc_p = acc_p * alpha.broadcast(const_shape![GROUP, DPE]);

            let p_bf: Tile<bf16, { [GROUP, BN] }> = convert_tile(p);
            acc_n = mma(p_bf, kv_n, acc_n);
            acc_p = mma(p_bf, kv_p, acc_p);
            m_i = m_ij;
            let _ = tok_n;
        }

        // Finalize this split (normalized partial + log2 LSE).
        let eps: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let l_safe: Tile<f32, { [GROUP, 1] }> = max_tile(l_i, eps);
        let out_n: Tile<bf16, { [GROUP, DNOPE] }> =
            convert_tile(true_div(acc_n, l_safe.broadcast(const_shape![GROUP, DNOPE])));
        let out_p: Tile<bf16, { [GROUP, DPE] }> =
            convert_tile(true_div(acc_p, l_safe.broadcast(const_shape![GROUP, DPE])));
        att_nope_out.store(out_n.reshape(const_shape![1, GROUP, DNOPE]));
        att_pe_out.store(out_p.reshape(const_shape![1, GROUP, DPE]));

        let lse: Tile<f32, { [GROUP, 1] }> = m_i + log2(l_safe);
        lse_out.store(lse.reshape(const_shape![1, GROUP]));
    }
}

pub use mla_decode_splitkv_bf16_module::mla_decode_splitkv_bf16;

/// Split-KV decode attention over the V4.1 FP8 KV cache (E4M3 data,
/// per-32 UE8M0 scales), dequantized in-register before the MMA.
#[cutile::module]
pub mod mla_decode_splitkv_fp8_module {
    use cutile::core::*;

    /// Same contract as [`mla_decode_splitkv_bf16`] but with
    /// `ckv: [rows, 512]` E4M3 + `ckv_scales: [rows, 16]` UE8M0 and
    /// `kpe: [rows, 64]` E4M3 + `kpe_scales: [rows, 2]` UE8M0.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (occupancy = 1, max_divisibility = 16,),
                      sm_103 = (occupancy = 1, max_divisibility = 16,),
                    ))]
    unsafe fn mla_decode_splitkv_fp8<
        const GROUP: i32,
        const BN: i32,
        const DNOPE: i32,
        const DPE: i32,
        const NSPLITS: i32,
    >(
        att_nope_out: &mut Tensor<bf16, { [1, GROUP, DNOPE] }>,
        att_pe_out: &mut Tensor<bf16, { [1, GROUP, DPE] }>,
        lse_out: &mut Tensor<f32, { [1, GROUP] }>,
        q_nope: &Tensor<bf16, { [-1, GROUP, DNOPE] }>,
        q_pe: &Tensor<bf16, { [-1, GROUP, DPE] }>,
        ckv: &Tensor<f8e4m3fn, { [-1, DNOPE] }>,
        ckv_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        kpe: &Tensor<f8e4m3fn, { [-1, DPE] }>,
        kpe_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        indices: &Tensor<i32, { [-1, -1] }>,
        ngroups: i32,
        topk: i32,
        sm_scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row: i32 = pid.0;
        let split: i32 = pid.1;
        let b: i32 = row / ngroups;

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let ln2: f32 = tile_to_scalar(log(two));
        let scale2: Tile<f32, { [GROUP, BN] }> =
            broadcast_scalar(sm_scale / ln2, const_shape![GROUP, BN]);

        let qn: Tile<bf16, { [GROUP, DNOPE] }> = q_nope
            .partition(const_shape![1, GROUP, DNOPE])
            .load([row, 0i32, 0i32])
            .reshape(const_shape![GROUP, DNOPE]);
        let qp: Tile<bf16, { [GROUP, DPE] }> = q_pe
            .partition(const_shape![1, GROUP, DPE])
            .load([row, 0i32, 0i32])
            .reshape(const_shape![GROUP, DPE]);

        let total_tiles: i32 = ceil_div(topk, BN);
        let tiles_per_split: i32 = ceil_div(total_tiles, NSPLITS);
        let start_tile: i32 = split * tiles_per_split;
        let mut end_tile: i32 = start_tile + tiles_per_split;
        end_tile = min(end_tile, total_tiles);

        let neg_big_g: Tile<f32, { [GROUP, BN] }> =
            constant(0.0f32, const_shape![GROUP, BN]) - constant(1.0e30f32, const_shape![GROUP, BN]);
        let mut m_i: Tile<f32, { [GROUP, 1] }> =
            constant(0.0f32, const_shape![GROUP, 1]) - constant(1.0e30f32, const_shape![GROUP, 1]);
        let mut l_i: Tile<f32, { [GROUP, 1] }> = constant(0.0f32, const_shape![GROUP, 1]);
        let mut acc_n: Tile<f32, { [GROUP, DNOPE] }> = constant(0.0f32, const_shape![GROUP, DNOPE]);
        let mut acc_p: Tile<f32, { [GROUP, DPE] }> = constant(0.0f32, const_shape![GROUP, DPE]);

        // Scale bytes are expanded inline below (no closures in the DSL).

        let ckv_base: PointerTile<*const f8e4m3fn, { [] }> = pointer_to_tile(ckv.as_ptr());
        let ckvs_base: PointerTile<*const f8e8m0fnu, { [] }> = pointer_to_tile(ckv_scales.as_ptr());
        let kpe_base: PointerTile<*const f8e4m3fn, { [] }> = pointer_to_tile(kpe.as_ptr());
        let kpes_base: PointerTile<*const f8e8m0fnu, { [] }> = pointer_to_tile(kpe_scales.as_ptr());
        let idx_part = indices.partition(const_shape![1, BN]);

        let cols_n: Tile<i32, { [1, DNOPE] }> =
            iota(const_shape![DNOPE]).reshape(const_shape![1, DNOPE]);
        let cols_p: Tile<i32, { [1, DPE] }> = iota(const_shape![DPE]).reshape(const_shape![1, DPE]);
        let cols_ns: Tile<i32, { [1, 16] }> = iota(const_shape![16]).reshape(const_shape![1, 16]);
        let cols_ps: Tile<i32, { [1, 2] }> = iota(const_shape![2]).reshape(const_shape![1, 2]);
        let _lane_n: Tile<i32, { [BN, 1] }> = iota(const_shape![BN]).reshape(const_shape![BN, 1]);

        let ckv_b: PointerTile<*const f8e4m3fn, { [BN, DNOPE] }> = ckv_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, DNOPE]);
        let ckvs_b: PointerTile<*const f8e8m0fnu, { [BN, 16] }> = ckvs_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, 16]);
        let kpe_b: PointerTile<*const f8e4m3fn, { [BN, DPE] }> = kpe_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, DPE]);
        let kpes_b: PointerTile<*const f8e8m0fnu, { [BN, 2] }> = kpes_base
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![BN, 2]);

        for t in start_tile..end_tile {
            let rows: Tile<i32, { [BN] }> = idx_part.load([b, t]).reshape(const_shape![BN]);
            let safe_rows: Tile<i32, { [BN] }> = select(
                ge_tile(rows, broadcast_scalar(0i32, const_shape![BN])),
                rows,
                broadcast_scalar(0i32, const_shape![BN]),
            );
            let lane_ok: Tile<bool, { [BN] }> =
                lt_tile(iota(const_shape![BN]) + broadcast_scalar(t * BN, const_shape![BN]),
                        broadcast_scalar(topk, const_shape![BN]))
                    & ge_tile(rows, broadcast_scalar(0i32, const_shape![BN]));

            let rows_n: Tile<i32, { [BN, DNOPE] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DNOPE]);
            let rows_p: Tile<i32, { [BN, DPE] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, DPE]);
            let rows_ns: Tile<i32, { [BN, 16] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 16]);
            let rows_ps: Tile<i32, { [BN, 2] }> =
                safe_rows.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 2]);

            let offs_n: Tile<i32, { [BN, DNOPE] }> =
                rows_n * broadcast_scalar(DNOPE, const_shape![BN, DNOPE])
                    + cols_n.broadcast(const_shape![BN, DNOPE]);
            let offs_p: Tile<i32, { [BN, DPE] }> =
                rows_p * broadcast_scalar(DPE, const_shape![BN, DPE])
                    + cols_p.broadcast(const_shape![BN, DPE]);
            let offs_ns: Tile<i32, { [BN, 16] }> =
                rows_ns * broadcast_scalar(16, const_shape![BN, 16])
                    + cols_ns.broadcast(const_shape![BN, 16]);
            let offs_ps: Tile<i32, { [BN, 2] }> =
                rows_ps * broadcast_scalar(2, const_shape![BN, 2])
                    + cols_ps.broadcast(const_shape![BN, 2]);

            let (kv_n8, _): (Tile<f8e4m3fn, { [BN, DNOPE] }>, Token) = unsafe {
                load_ptr_tko(
                    ckv_b.offset_tile(offs_n),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let (kv_ns8, _): (Tile<f8e8m0fnu, { [BN, 16] }>, Token) = unsafe {
                load_ptr_tko(
                    ckvs_b.offset_tile(offs_ns),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let (kv_p8, _): (Tile<f8e4m3fn, { [BN, DPE] }>, Token) = unsafe {
                load_ptr_tko(
                    kpe_b.offset_tile(offs_p),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let (kv_ps8, _): (Tile<f8e8m0fnu, { [BN, 2] }>, Token) = unsafe {
                load_ptr_tko(
                    kpes_b.offset_tile(offs_ps),
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };

            // Dequantize: value = e4m3 * 2^(byte-127) per 32-lane group.
            let ns_u8: Tile<u8, { [BN, 16] }> = bitcast(kv_ns8);
            let ns_i: Tile<i32, { [BN, 16] }> = exti(ns_u8);
            let ns_f: Tile<f32, { [BN, 16] }> = convert_tile(ns_i);
            let ns_e: Tile<f32, { [BN, 16] }> =
                exp2(ns_f - broadcast_scalar(127.0f32, const_shape![BN, 16]), ftz::Disabled);
            let ns_g: Tile<f32, { [BN, 16, 32] }> = ns_e
                .reshape(const_shape![BN, 16, 1])
                .broadcast(const_shape![BN, 16, 32]);
            let ns_flat: Tile<f32, { [BN, DNOPE] }> = ns_g.reshape(const_shape![BN, DNOPE]);
            let kv_n_f: Tile<f32, { [BN, DNOPE] }> = convert_tile(kv_n8);
            let kv_n: Tile<bf16, { [BN, DNOPE] }> = convert_tile(kv_n_f * ns_flat);

            let ps_u8: Tile<u8, { [BN, 2] }> = bitcast(kv_ps8);
            let ps_i: Tile<i32, { [BN, 2] }> = exti(ps_u8);
            let ps_f: Tile<f32, { [BN, 2] }> = convert_tile(ps_i);
            let ps_e: Tile<f32, { [BN, 2] }> =
                exp2(ps_f - broadcast_scalar(127.0f32, const_shape![BN, 2]), ftz::Disabled);
            let ps_g: Tile<f32, { [BN, 2, 32] }> = ps_e
                .reshape(const_shape![BN, 2, 1])
                .broadcast(const_shape![BN, 2, 32]);
            let ps_flat: Tile<f32, { [BN, DPE] }> = ps_g.reshape(const_shape![BN, DPE]);
            let kv_p_f: Tile<f32, { [BN, DPE] }> = convert_tile(kv_p8);
            let kv_p: Tile<bf16, { [BN, DPE] }> = convert_tile(kv_p_f * ps_flat);

            let mut qk: Tile<f32, { [GROUP, BN] }> = constant(0.0f32, const_shape![GROUP, BN]);
            qk = mma(qn, kv_n.transpose(), qk);
            qk = mma(qp, kv_p.transpose(), qk);
            let qk: Tile<f32, { [GROUP, BN] }> = qk * scale2;
            let zero_s: Tile<f32, { [GROUP, BN] }> = constant(0.0f32, const_shape![GROUP, BN]);
            let ok2: Tile<bool, { [GROUP, BN] }> =
                lane_ok.reshape(const_shape![1, BN]).broadcast(const_shape![GROUP, BN]);
            let qk: Tile<f32, { [GROUP, BN] }> = select(ok2, qk, zero_s + neg_big_g);

            let qk_max: Tile<f32, { [GROUP] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [GROUP, 1] }> = qk_max.reshape(const_shape![GROUP, 1]);
            let m_ij: Tile<f32, { [GROUP, 1] }> = max_tile(m_i, qk_max);
            let qk_shift: Tile<f32, { [GROUP, BN] }> =
                qk - m_ij.broadcast(const_shape![GROUP, BN]);
            let p: Tile<f32, { [GROUP, BN] }> = exp2(qk_shift, ftz::Disabled);
            let l_ij: Tile<f32, { [GROUP] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [GROUP, 1] }> = l_ij.reshape(const_shape![GROUP, 1]);
            let alpha: Tile<f32, { [GROUP, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc_n = acc_n * alpha.broadcast(const_shape![GROUP, DNOPE]);
            acc_p = acc_p * alpha.broadcast(const_shape![GROUP, DPE]);

            let p_bf: Tile<bf16, { [GROUP, BN] }> = convert_tile(p);
            acc_n = mma(p_bf, kv_n, acc_n);
            acc_p = mma(p_bf, kv_p, acc_p);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let l_safe: Tile<f32, { [GROUP, 1] }> = max_tile(l_i, eps);
        let out_n: Tile<bf16, { [GROUP, DNOPE] }> =
            convert_tile(true_div(acc_n, l_safe.broadcast(const_shape![GROUP, DNOPE])));
        let out_p: Tile<bf16, { [GROUP, DPE] }> =
            convert_tile(true_div(acc_p, l_safe.broadcast(const_shape![GROUP, DPE])));
        att_nope_out.store(out_n.reshape(const_shape![1, GROUP, DNOPE]));
        att_pe_out.store(out_p.reshape(const_shape![1, GROUP, DPE]));

        let lse: Tile<f32, { [GROUP, 1] }> = m_i + log2(l_safe);
        lse_out.store(lse.reshape(const_shape![1, GROUP]));
    }
}

pub use mla_decode_splitkv_fp8_module::mla_decode_splitkv_fp8;

/// Merges split-KV partials into the final decode output, applies the
/// FlashMLA `attn_sink` epilogue, and emits natural-log LSE.
#[cutile::module]
pub mod mla_split_merge_module {
    use cutile::core::*;

    /// Combines `NSPLITS` per-split partials (split-major layout
    /// `[b, g, split*GROUP + gi, D]`) into `out[b, g, gi, 576]` +
    /// `lse[b, g*GROUP + gi]` (natural log). Empty windows produce zero
    /// outputs and `lse = +inf`. `sink` is the per-head attention sink
    /// (`out *= 1 / (1 + exp(sink - lse))`), ignored when `has_sink == 0`.
    #[cutile::entry(print_ir = false,
                    unchecked_accesses = true,
                    optimization_hints = (
                      sm_100 = (occupancy = 4, max_divisibility = 16,),
                      sm_103 = (occupancy = 4, max_divisibility = 16,),
                    ))]
    unsafe fn mla_split_merge<
        const GROUP: i32,
        const DNOPE: i32,
        const DPE: i32,
        const D: i32,
        const NSPLITS: i32,
        const NSG: i32,
    >(
        out: &mut Tensor<bf16, { [1, GROUP, D] }>,
        lse: &mut Tensor<f32, { [1, GROUP] }>,
        att_nope_partial: &Tensor<bf16, { [-1, -1, DNOPE] }>,
        att_pe_partial: &Tensor<bf16, { [-1, -1, DPE] }>,
        lse_partial: &Tensor<f32, { [-1, -1] }>,
        sinks: &Tensor<f32, { [-1] }>,
        ngroups: i32,
        has_sink: i32,
        empty_lse: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row: i32 = pid.0;
        let g: i32 = row % ngroups;

        // LSE partials: [NSPLITS, GROUP] → [GROUP, NSPLITS].
        let lse_t: Tile<f32, { [1, NSG] }> = lse_partial
            .partition(const_shape![1, NSG])
            .load([row, 0i32]);
        let lse_ns: Tile<f32, { [NSPLITS, GROUP] }> =
            lse_t.reshape(const_shape![NSPLITS, GROUP]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> { dims: &[1i32, 0i32] };
        let lse_gn: Tile<f32, { [GROUP, NSPLITS] }> = permute(lse_ns, transpose_2d);

        // Per-split weights; all-empty windows collapse to zeros.
        let lse_max: Tile<f32, { [GROUP] }> = reduce_max(lse_gn, 1i32);
        let lse_max_c: Tile<f32, { [GROUP, 1] }> = lse_max.reshape(const_shape![GROUP, 1]);
        let shifted: Tile<f32, { [GROUP, NSPLITS] }> =
            lse_gn - lse_max_c.broadcast(const_shape![GROUP, NSPLITS]);
        let w_raw: Tile<f32, { [GROUP, NSPLITS] }> = exp2(shifted, ftz::Disabled);
        let w_sum: Tile<f32, { [GROUP] }> = reduce_sum(w_raw, 1i32);
        let w_sum_c: Tile<f32, { [GROUP, 1] }> = w_sum.reshape(const_shape![GROUP, 1]);
        let eps: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let w_safe: Tile<f32, { [GROUP, 1] }> = max_tile(w_sum_c, eps);
        let weights: Tile<f32, { [GROUP, NSPLITS] }> =
            true_div(w_raw, w_safe.broadcast(const_shape![GROUP, NSPLITS]));

        // Weighted combine of the partials.
        let transpose_3d: Array<{ [1, 0, 2] }> = Array::<{ [1, 0, 2] }> { dims: &[1i32, 0i32, 2i32] };
        let att_n_ns: Tile<bf16, { [1, NSG, DNOPE] }> = att_nope_partial
            .partition(const_shape![1, NSG, DNOPE])
            .load([row, 0i32, 0i32]);
        let att_n_sgd: Tile<bf16, { [NSPLITS, GROUP, DNOPE] }> =
            att_n_ns.reshape(const_shape![NSPLITS, GROUP, DNOPE]);
        let att_n_gsd: Tile<bf16, { [GROUP, NSPLITS, DNOPE] }> = permute(att_n_sgd, transpose_3d);
        let att_n_f: Tile<f32, { [GROUP, NSPLITS, DNOPE] }> = convert_tile(att_n_gsd);
        let w_n: Tile<f32, { [GROUP, NSPLITS, DNOPE] }> = weights
            .reshape(const_shape![GROUP, NSPLITS, 1])
            .broadcast(const_shape![GROUP, NSPLITS, DNOPE]);
        let out_n: Tile<f32, { [GROUP, DNOPE] }> =
            reduce_sum(att_n_f * w_n, 1i32);

        let att_p_ns: Tile<bf16, { [1, NSG, DPE] }> = att_pe_partial
            .partition(const_shape![1, NSG, DPE])
            .load([row, 0i32, 0i32]);
        let att_p_sgd: Tile<bf16, { [NSPLITS, GROUP, DPE] }> =
            att_p_ns.reshape(const_shape![NSPLITS, GROUP, DPE]);
        let att_p_gsd: Tile<bf16, { [GROUP, NSPLITS, DPE] }> = permute(att_p_sgd, transpose_3d);
        let att_p_f: Tile<f32, { [GROUP, NSPLITS, DPE] }> = convert_tile(att_p_gsd);
        let w_p: Tile<f32, { [GROUP, NSPLITS, DPE] }> = weights
            .reshape(const_shape![GROUP, NSPLITS, 1])
            .broadcast(const_shape![GROUP, NSPLITS, DPE]);
        let out_p: Tile<f32, { [GROUP, DPE] }> = reduce_sum(att_p_f * w_p, 1i32);

        // Natural-log LSE (log2 → ln); +inf for empty windows.
        let lse_log2: Tile<f32, { [GROUP, 1] }> = lse_max_c + log2(w_safe);
        let lse_log2: Tile<f32, { [GROUP] }> = lse_log2.reshape(const_shape![GROUP]);
        let empty: Tile<bool, { [GROUP] }> =
            lt_tile(lse_max, broadcast_scalar(-0.5e30f32, const_shape![GROUP]));
        let lse_val: Tile<f32, { [GROUP] }> = select(
            empty,
            broadcast_scalar(empty_lse, const_shape![GROUP]),
            lse_log2 * broadcast_scalar(0.6931471805599453f32, const_shape![GROUP]),
        );

        // Sink epilogue: out *= 1 / (1 + exp(sink - lse)), per head.
        if has_sink != 0 {
            // sinks are indexed by absolute head id = g * GROUP + i.
            let sink_base: PointerTile<*const f32, { [] }> = pointer_to_tile(sinks.as_ptr());
            let sink_off: Tile<i32, { [GROUP] }> =
                iota(const_shape![GROUP]) + broadcast_scalar(g * GROUP, const_shape![GROUP]);
            let sink_ptrs: PointerTile<*const f32, { [GROUP] }> = sink_base
                .reshape(const_shape![1])
                .broadcast(const_shape![GROUP])
                .offset_tile(sink_off);
            let (sink_t, _): (Tile<f32, { [GROUP] }>, Token) = unsafe {
                load_ptr_tko(
                    sink_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None::<Token>,
                    Latency::<0>,
                )
            };
            let one_g: Tile<f32, { [GROUP] }> = constant(1.0f32, const_shape![GROUP]);
            let gate: Tile<f32, { [GROUP] }> = true_div(one_g, one_g + exp(sink_t - lse_val));
            let gate_n: Tile<f32, { [GROUP, DNOPE] }> =
                gate.reshape(const_shape![GROUP, 1]).broadcast(const_shape![GROUP, DNOPE]);
            let gate_p: Tile<f32, { [GROUP, DPE] }> =
                gate.reshape(const_shape![GROUP, 1]).broadcast(const_shape![GROUP, DPE]);
            let gated_n: Tile<bf16, { [GROUP, DNOPE] }> = convert_tile(out_n * gate_n);
            let gated_p: Tile<bf16, { [GROUP, DPE] }> = convert_tile(out_p * gate_p);
            let both: Tile<bf16, { [GROUP, D] }> = cat(gated_n, gated_p, 1i32);
            out.store(both.reshape(const_shape![1, GROUP, D]));
        } else {
            let both: Tile<bf16, { [GROUP, D] }> = cat(convert_tile(out_n), convert_tile(out_p), 1i32);
            out.store(both.reshape(const_shape![1, GROUP, D]));
        }
        lse.store(lse_val.reshape(const_shape![1, GROUP]));
    }
}

pub use mla_split_merge_module::mla_split_merge;
