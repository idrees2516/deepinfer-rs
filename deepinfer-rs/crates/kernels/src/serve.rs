//! Paged-cache multi-sequence serving launchers (see `kernels/serve.cu`).
//!
//! The kernels address every cached row through a per-sequence page table
//! (`phys = tbl[r / 64] * 64 + r % 64`), so a fixed pool serves an
//! unbounded, dynamically changing set of sequences. All per-step scalars
//! (positions, lengths, window bounds) live in the device-resident cell
//! block built by [`Kernels::serve_fanout`]; launches use fixed maxima, so
//! a serve step is capturable exactly like the single-sequence decode
//! family.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::elementwise::CacheFormat;
use crate::Kernels;

/// Page size of the paged caches (matches the DSA sparse-index page
/// granularity `page_idx * 64 + offset`).
pub const SERVE_PAGE: usize = 64;

/// Host-written per-slot input cell (the first five fields the fanout
/// kernel expands into the full 16-field device block).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServeCellIn {
    /// 1 = the slot participates in this step.
    pub active: i32,
    /// Absolute position of the first new row.
    pub pos: i32,
    /// Cached rows before this step (sequence-local).
    pub len: i32,
    /// Rows appended this step (1 for decode, the chunk length for
    /// prefill; uniform across slots per call).
    pub sq: i32,
    /// Lightning-quant blocks already complete (decode: `len / qb`;
    /// prefill re-quantizes the whole history: 0).
    pub sqnb: i32,
}

impl ServeCellIn {
    /// The five i32 words, ready for the cell upload.
    pub fn words(&self) -> [i32; 5] {
        [self.active, self.pos, self.len, self.sq, self.sqnb]
    }
}

/// Device cache-format id (mirrors `att_load_k`'s `fmt`).
fn fmt_id(f: CacheFormat) -> i32 {
    match f {
        CacheFormat::F32 => 0,
        CacheFormat::Fp8Group32 => 1,
        CacheFormat::Fp4Group16 => 2,
    }
}

/// Next power of two >= n (for bitonic shared-memory sizing).
fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Number of cell fields (ABI with serve.cu's SV_COUNT).
#[allow(dead_code)] // consumed by the runtime serve engine
pub const SERVE_CELL_FIELDS: usize = 16;

#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_serve_fanout_kernel",
    "di_serve_pos_fill_kernel",
    "di_serve_append_rows_kernel",
    "di_serve_append_norm_rows_kernel",
    "di_serve_kv_encode_kernel",
    "di_serve_conv_feat_kernel",
    "di_serve_conv_gated_norm_kernel",
    "di_serve_lightning_quant_kernel",
    "di_serve_thunder_logits_kernel",
    "di_serve_dot_scores_kernel",
    "di_serve_gate_blend_kernel",
    "di_serve_causal_clip_kernel",
    "di_serve_block_preselect_kernel",
    "di_serve_topk_kernel",
    "di_serve_mla_decode_partial_kernel",
    "di_serve_mla_decode_combine_kernel",
    "di_serve_prefill_sparse_kernel",
    "di_serve_prefill_dense_kernel",
    "di_serve_ring_append_kernel",
    "di_serve_swa_attention_kernel",
];

impl Kernels {
    /// Expand the host cells `{active, pos, len, sq, sqnb}` `[B, 5]` i32
    /// into the full 16-field device block `[B, 16]` (`cells`).
    pub fn serve_fanout(
        &self,
        in_cells: &Buffer,
        cells: &Buffer,
        b: usize,
        cross_win: usize,
        swa_win: usize,
        qb: usize,
        topk: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(b > 0, InvalidArgument, "serve needs B > 0");
        let f = self.f("di_serve_fanout_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (b as u32, 1, 1),
                block: (32, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(in_cells)
                .ptr(cells)
                .i32(b as i32)
                .i32(cross_win as i32)
                .i32(swa_win as i32)
                .i32(qb as i32)
                .i32(topk as i32),
            stream,
        )
    }

    /// Fill RoPE position buffers for the whole batch: `pos_kv [B*sq]`,
    /// `pos_q [B*sq*H]`.
    pub fn serve_pos_fill(
        &self,
        cells: &Buffer,
        pos_kv: &Buffer,
        pos_q: &Buffer,
        b: usize,
        sq: usize,
        h: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_serve_pos_fill_kernel",
            b * sq,
            128,
            0,
            KernelArgs::new()
                .ptr(cells)
                .ptr(pos_kv)
                .ptr(pos_q)
                .i32(b as i32)
                .i32(sq as i32)
                .i32(h as i32),
            stream,
        )
    }

    /// Append `src [B*sq, elems]` f32 rows into the paged `pool` through
    /// the per-slot page tables (`tbl_ptrs [B]` u64).
    pub fn serve_append_rows(
        &self,
        src: &Buffer,
        pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        elems: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_append_rows_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (elems.div_ceil(256).max(1) as u32, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(src)
                .ptr(pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .i32(elems as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Per-head RMSNorm + paged append (`src [B*sq, nh*ihd]`).
    pub fn serve_append_norm_rows(
        &self,
        src: &Buffer,
        pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        nh: usize,
        ihd: usize,
        eps: f32,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_append_norm_rows_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (1, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(src)
                .ptr(pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .i32(nh as i32)
                .i32(ihd as i32)
                .f32(eps)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// KV-cache encode into the paged pool (`fmt` selects f32 / FP8 528 B
    /// / FP4 288 B rows; `src [B*sq, hd]` roped latent rows).
    pub fn serve_kv_encode(
        &self,
        src: &Buffer,
        pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        hd: usize,
        fmt: CacheFormat,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_kv_encode_kernel")?;
        let groups = if fmt_id(fmt) == 2 { hd / 16 } else { hd / 32 };
        f.launch(
            self.device(),
            Launch {
                grid: (groups.div_ceil(8).max(1) as u32, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(src)
                .ptr(pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .i32(hd as i32)
                .i32(fmt_id(fmt))
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Cascade conv features over the paged `k_pool`, written to the paged
    /// `feat_pool`.
    #[allow(clippy::too_many_arguments)]
    pub fn serve_conv_feat(
        &self,
        k_pool: &Buffer,
        feat_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        kw: &Buffer,
        kw_off: &Buffer,
        num_kernels: usize,
        hidden: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_conv_feat_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (hidden.div_ceil(256).max(1) as u32, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(k_pool)
                .ptr(feat_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(kw)
                .ptr(kw_off)
                .i32(num_kernels as i32)
                .i32(hidden as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Gated RMSNorm of the paged feat rows (in place).
    pub fn serve_conv_gated_norm(
        &self,
        feat_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        gain: &Buffer,
        hidden: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_serve_conv_gated_norm_kernel",
            b * sq,
            256,
            0,
            KernelArgs::new()
                .ptr(feat_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(gain)
                .i32(hidden as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Lightning block quant over the paged k_norm pool (blocks
    /// `[sqnb, knb)` per slot); `sf_ptrs [B]` u64 point at per-sequence
    /// `[max_blocks]` u8 scale buffers.
    pub fn serve_lightning_quant(
        &self,
        k_norm_pool: &Buffer,
        k_u8_pool: &Buffer,
        tbl_ptrs: &Buffer,
        sf_ptrs: &Buffer,
        cells: &Buffer,
        idx_dim: usize,
        qb: usize,
        b: usize,
        max_blocks: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_lightning_quant_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (max_blocks as u32, b as u32, 1),
                block: (256, 1, 1),
                shared: 256 * 4,
            },
            &KernelArgs::new()
                .ptr(k_norm_pool)
                .ptr(k_u8_pool)
                .ptr(tbl_ptrs)
                .ptr(sf_ptrs)
                .ptr(cells)
                .i32(idx_dim as i32)
                .i32(qb as i32)
                .i32(b as i32)
                .i32(max_blocks as i32),
            stream,
        )
    }

    /// Thunder logits over the paged k caches; `scores [B*sq, max_cols]`.
    #[allow(clippy::too_many_arguments)]
    pub fn serve_thunder_logits(
        &self,
        q_f32: &Buffer,
        q_u8: &Buffer,
        q_sf: &Buffer,
        k_norm_pool: &Buffer,
        k_u8_pool: &Buffer,
        tbl_ptrs: &Buffer,
        sf_ptrs: &Buffer,
        cells: &Buffer,
        w: &Buffer,
        scores: &Buffer,
        nh: usize,
        ihd: usize,
        qb: usize,
        q_nblocks: usize,
        max_cols: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_thunder_logits_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (32, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: ((nh * ihd * 4).max(4)) as u32,
            },
            &KernelArgs::new()
                .ptr(q_f32)
                .ptr(q_u8)
                .ptr(q_sf)
                .ptr(k_norm_pool)
                .ptr(k_u8_pool)
                .ptr(tbl_ptrs)
                .ptr(sf_ptrs)
                .ptr(cells)
                .ptr(w)
                .ptr(scores)
                .i32(nh as i32)
                .i32(ihd as i32)
                .i32(qb as i32)
                .i32(q_nblocks as i32)
                .i32(max_cols as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Conv-branch scores over the paged feat pool; `out [B*sq, max_cols]`.
    pub fn serve_dot_scores(
        &self,
        q: &Buffer,
        feat_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        out: &Buffer,
        hidden: usize,
        max_cols: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_dot_scores_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (32, (b * sq) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(feat_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(out)
                .i32(hidden as i32)
                .i32(max_cols as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Gate blend (`gt`/`gc [B*sq]`, matrices `[B*sq, max_cols]`).
    pub fn serve_gate_blend(
        &self,
        gt: &Buffer,
        gc: &Buffer,
        conv: &Buffer,
        thunder: &Buffer,
        cells: &Buffer,
        max_cols: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_serve_gate_blend_kernel",
            b * sq * max_cols,
            256,
            0,
            KernelArgs::new()
                .ptr(gt)
                .ptr(gc)
                .ptr(conv)
                .ptr(thunder)
                .ptr(cells)
                .i32(max_cols as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Causal + extrapolation-window clip.
    pub fn serve_causal_clip(
        &self,
        scores: &Buffer,
        cells: &Buffer,
        max_cols: usize,
        window: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_serve_causal_clip_kernel",
            b * sq * max_cols,
            256,
            0,
            KernelArgs::new()
                .ptr(scores)
                .ptr(cells)
                .i32(max_cols as i32)
                .i32(window as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Block preselect (fixed `num_blocks` x `per` geometry);
    /// `cands [B*sq, num_blocks*per]` u64.
    pub fn serve_block_preselect(
        &self,
        scores: &Buffer,
        cands: &Buffer,
        cells: &Buffer,
        block: usize,
        per: usize,
        num_blocks: usize,
        max_cols: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_block_preselect_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (num_blocks as u32, (b * sq) as u32, 1),
                block: (1024, 1, 1),
                shared: (next_pow2(block) * 8) as u32,
            },
            &KernelArgs::new()
                .ptr(scores)
                .ptr(cands)
                .ptr(cells)
                .i32(block as i32)
                .i32(per as i32)
                .i32(max_cols as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Top-K merge + scatter; `sel [B*sq, topk_max]` i32.
    pub fn serve_topk(
        &self,
        cands: &Buffer,
        sel: &Buffer,
        cells: &Buffer,
        num_blocks: usize,
        per: usize,
        topk_max: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_topk_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: ((b * sq) as u32, 1, 1),
                block: (1024, 1, 1),
                shared: (next_pow2(num_blocks * per) * 8) as u32,
            },
            &KernelArgs::new()
                .ptr(cands)
                .ptr(sel)
                .ptr(cells)
                .i32(num_blocks as i32)
                .i32(per as i32)
                .i32(topk_max as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// MLA decode, split-KV partial over the paged KV pool.
    #[allow(clippy::too_many_arguments)]
    pub fn serve_mla_decode_partial(
        &self,
        q: &Buffer,
        kv_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        sel: &Buffer,
        pm: &Buffer,
        pl: &Buffer,
        po: &Buffer,
        h: usize,
        d: usize,
        topk_max: usize,
        chunk: usize,
        sm_scale: f32,
        fmt: CacheFormat,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_mla_decode_partial_kernel")?;
        let splits = topk_max.div_ceil(chunk).max(1);
        f.launch(
            self.device(),
            Launch {
                grid: (
                    splits as u32,
                    h.div_ceil(8).max(1) as u32,
                    (b * sq) as u32,
                ),
                block: (8 * 32, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(kv_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(sel)
                .ptr(pm)
                .ptr(pl)
                .ptr(po)
                .i32(h as i32)
                .i32(d as i32)
                .i32(topk_max as i32)
                .i32(chunk as i32)
                .f32(sm_scale)
                .i32(fmt_id(fmt))
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// MLA decode combine (identical math to the arena path).
    #[allow(clippy::too_many_arguments)]
    pub fn serve_mla_decode_combine(
        &self,
        pm: &Buffer,
        pl: &Buffer,
        po: &Buffer,
        sink: Option<&Buffer>,
        out: &Buffer,
        lse: Option<&Buffer>,
        cells: &Buffer,
        h: usize,
        d: usize,
        splits: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_mla_decode_combine_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (d.div_ceil(256).max(1) as u32, (b * sq * h) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(pm)
                .ptr(pl)
                .ptr(po)
                .u64(sink.map(|s| s.ptr()).unwrap_or(0))
                .ptr(out)
                .u64(lse.map(|s| s.ptr()).unwrap_or(0))
                .ptr(cells)
                .i32(h as i32)
                .i32(d as i32)
                .i32(splits as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Sparse (DSA-selected) prefill over the paged KV pool.
    #[allow(clippy::too_many_arguments)]
    pub fn serve_prefill_sparse(
        &self,
        q: &Buffer,
        kv_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        sel: &Buffer,
        out: &Buffer,
        lse: Option<&Buffer>,
        h: usize,
        d: usize,
        topk_max: usize,
        sm_scale: f32,
        fmt: CacheFormat,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_prefill_sparse_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (
                    sq.div_ceil(8).max(1) as u32,
                    h as u32,
                    b as u32,
                ),
                block: (8 * 32, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(kv_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(sel)
                .ptr(out)
                .u64(lse.map(|s| s.ptr()).unwrap_or(0))
                .i32(h as i32)
                .i32(d as i32)
                .i32(topk_max as i32)
                .f32(sm_scale)
                .i32(fmt_id(fmt))
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Dense-causal prefill over the paged cross window.
    #[allow(clippy::too_many_arguments)]
    pub fn serve_prefill_dense(
        &self,
        q: &Buffer,
        kv_pool: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        out: &Buffer,
        h: usize,
        d: usize,
        sm_scale: f32,
        fmt: CacheFormat,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_prefill_dense_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (
                    sq.div_ceil(8).max(1) as u32,
                    h as u32,
                    b as u32,
                ),
                block: (8 * 32, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(kv_pool)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(out)
                .i32(h as i32)
                .i32(d as i32)
                .f32(sm_scale)
                .i32(fmt_id(fmt))
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// SWA ring append (`x [B*sq, hidden]` into the per-slot rings;
    /// `ring_ptrs [B * n_cross]` u64).
    pub fn serve_ring_append(
        &self,
        x: &Buffer,
        ring_ptrs: &Buffer,
        cells: &Buffer,
        hidden: usize,
        win: usize,
        n_cross: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_ring_append_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: ((b * sq) as u32, n_cross as u32, 1),
                block: (128, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(ring_ptrs)
                .ptr(cells)
                .i32(hidden as i32)
                .i32(win as i32)
                .i32(n_cross as i32)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }

    /// Sliding-window attention over the per-slot rings
    /// (`out [B*sq, hidden]`).
    pub fn serve_swa_attention(
        &self,
        x: &Buffer,
        ring_ptrs: &Buffer,
        cells: &Buffer,
        out: &Buffer,
        hidden: usize,
        win: usize,
        sm_scale: f32,
        n_cross: usize,
        sq: usize,
        b: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_serve_swa_attention_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: ((b * sq) as u32, n_cross as u32, 1),
                block: (256, 1, 1),
                shared: (hidden * 4 + 2 * 256 * 4) as u32,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(ring_ptrs)
                .ptr(cells)
                .ptr(out)
                .i32(hidden as i32)
                .i32(win as i32)
                .f32(sm_scale)
                .i32(sq as i32)
                .i32(b as i32),
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_matches_dsa_contract() {
        // The DSA sparse index encodes page_idx * 64 + offset; the paged
        // cache must use the same granularity.
        assert_eq!(SERVE_PAGE, 64);
        assert_eq!(SERVE_PAGE.trailing_zeros(), 6);
    }

    #[test]
    fn cell_words_roundtrip() {
        let c = ServeCellIn {
            active: 1,
            pos: 17,
            len: 17,
            sq: 1,
            sqnb: 0,
        };
        assert_eq!(c.words(), [1, 17, 17, 1, 0]);
    }

    #[test]
    fn next_pow2_sizes() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(64), 64);
        assert_eq!(next_pow2(65), 128);
    }

    #[test]
    fn format_ids_match_att_kernel_contract() {
        assert_eq!(fmt_id(CacheFormat::F32), 0);
        assert_eq!(fmt_id(CacheFormat::Fp8Group32), 1);
        assert_eq!(fmt_id(CacheFormat::Fp4Group16), 2);
    }
}
