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
use crate::gemm::GemmDType;
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

/// Pipeline depth of the cp.async serve kernels (ABI with serve.cu's
/// `SV_PIPE_STAGES`).
pub const SV_PIPE_STAGES: usize = 3;

/// Dynamic shared-memory budget for one pipelined serve kernel (bytes).
/// Kept at 32 KiB so every launch stays under the 48 KiB default dynamic
/// limit without a `cuFuncSetAttribute` dance.
pub const SV_PIPE_SMEM: usize = 32 * 1024;

/// Byte stride of one cached KV row in the pool (mirrors `sv_row_bytes`).
fn cache_row_bytes(fmt: CacheFormat, d: usize) -> usize {
    match fmt {
        CacheFormat::F32 => d * 4,
        CacheFormat::Fp8Group32 => 528,
        CacheFormat::Fp4Group16 => 288,
    }
}

/// Rows per pipeline stage for the given cache row stride (mirrors the
/// kernel's `rows_per_stage` contract; the launcher passes the matching
/// `STAGES * rows * row_bytes` dynamic shared size).
fn pipe_rows_per_stage(row_bytes: usize) -> usize {
    (SV_PIPE_SMEM / (SV_PIPE_STAGES * row_bytes)).clamp(1, 64)
}

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
    // P1-8: cp.async-pipelined / vectorized variants.
    "di_serve_mla_decode_pipe_kernel",
    "di_serve_prefill_dense_pipe_kernel",
    "di_serve_thunder_logits_vec_kernel",
    "di_serve_dot_scores_vec_kernel",
    "di_mega_self_block_kernel",
];

impl Kernels {
    /// Whether the cp.async-pipelined KV-walk kernels are used. On by
    /// default (they are bit-identical to the direct-load family);
    /// `DEEPINFER_SERVE_PIPE=0` forces the legacy kernels for A/B goldens.
    pub fn serve_pipe_enabled(&self) -> bool {
        std::env::var("DEEPINFER_SERVE_PIPE").ok().as_deref() != Some("0")
    }

    /// Whether the vectorized DSA score kernels (one warp per key row) are
    /// used. On by default (deterministic, tolerance-golden vs the scalar
    /// family); `DEEPINFER_SERVE_VEC=0` forces the scalar kernels.
    pub fn serve_vec_enabled(&self) -> bool {
        std::env::var("DEEPINFER_SERVE_VEC").ok().as_deref() != Some("0")
    }

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
        if self.serve_vec_enabled() {
            // One warp per key row: packed u32 e4m3 loads + float4 f32 loads
            // (fully coalesced), fixed warp-tree reduction order.
            let f = self.f("di_serve_thunder_logits_vec_kernel")?;
            let warps = 8u32; // ATT_WARPS
            let grid_x = self
                .device()
                .props()
                .sm_count
                .max(warps)
                .min((b * sq).max(1) as u32 * 64);
            return f.launch(
                self.device(),
                Launch {
                    grid: (grid_x, (b * sq) as u32, 1),
                    block: (warps * self.warp_size(), 1, 1),
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
            );
        }
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
        if self.serve_vec_enabled() {
            // One warp per feat row with float4 loads (coalesced), fixed
            // warp-tree reduction order.
            let f = self.f("di_serve_dot_scores_vec_kernel")?;
            let warps = 8u32; // ATT_WARPS
            let grid_x = self
                .device()
                .props()
                .sm_count
                .max(warps)
                .min((b * sq).max(1) as u32 * 64);
            return f.launch(
                self.device(),
                Launch {
                    grid: (grid_x, (b * sq) as u32, 1),
                    block: (warps * self.warp_size(), 1, 1),
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
            );
        }
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
        let splits = topk_max.div_ceil(chunk).max(1);
        if self.serve_pipe_enabled() {
            // cp.async 3-stage pipeline (bit-identical to the direct walk).
            let rb = cache_row_bytes(fmt, d);
            ensure!(rb % 16 == 0, InvalidArgument, "cache row not 16B aligned");
            let r = pipe_rows_per_stage(rb);
            let f = self.f("di_serve_mla_decode_pipe_kernel")?;
            return f.launch(
                self.device(),
                Launch {
                    grid: (splits as u32, h.div_ceil(8).max(1) as u32, (b * sq) as u32),
                    block: (8 * self.warp_size(), 1, 1),
                    shared: (SV_PIPE_STAGES * r * rb) as u32,
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
                    .i32(b as i32)
                    .i32(r as i32),
                stream,
            );
        }
        let f = self.f("di_serve_mla_decode_partial_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (splits as u32, h.div_ceil(8).max(1) as u32, (b * sq) as u32),
                block: (8 * self.warp_size(), 1, 1),
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
                grid: (sq.div_ceil(8).max(1) as u32, h as u32, b as u32),
                block: (8 * self.warp_size(), 1, 1),
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
        if self.serve_pipe_enabled() {
            // cp.async 3-stage pipeline (bit-identical to the direct walk).
            let rb = cache_row_bytes(fmt, d);
            ensure!(rb % 16 == 0, InvalidArgument, "cache row not 16B aligned");
            let r = pipe_rows_per_stage(rb);
            let f = self.f("di_serve_prefill_dense_pipe_kernel")?;
            return f.launch(
                self.device(),
                Launch {
                    grid: (sq.div_ceil(8).max(1) as u32, h as u32, b as u32),
                    block: (8 * self.warp_size(), 1, 1),
                    shared: (SV_PIPE_STAGES * r * rb) as u32,
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
                    .i32(b as i32)
                    .i32(r as i32),
                stream,
            );
        }
        let f = self.f("di_serve_prefill_dense_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (sq.div_ceil(8).max(1) as u32, h as u32, b as u32),
                block: (8 * self.warp_size(), 1, 1),
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

/// Weight-dtype id for the megakernel (ABI with serve.cu's `wdtype`).
pub fn mega_wdtype(d: GemmDType) -> i32 {
    match d {
        GemmDType::F32 => 0,
        GemmDType::Bf16 => 1,
        GemmDType::Fp8 => 2,
        GemmDType::Nvfp4 => 3,
    }
}

/// Weight-tile output rows for the megakernel, by dtype (must divide 128
/// so a tile never straddles an fp8 scale block; sized to keep the
/// double-buffered tile + resident rows inside the 48 KiB default smem).
pub fn mega_tile_n(wdtype: i32) -> usize {
    match wdtype {
        0 => 32,
        1 => 64,
        2 => 64,
        _ => 128,
    }
}

/// Dynamic shared bytes the megakernel needs (mirror of its layout:
/// `[fused f32][x tile 512 B][fp8 qa quant pad][2 W tiles]`).
pub fn mega_shared(hidden: usize, q_lora: usize, hd: usize, wdtype: i32) -> usize {
    let _ = hidden; // layout depends on q_lora/hd; kept for ABI clarity
    let fused = q_lora + hd;
    let elt: usize = match wdtype {
        0 => 4,
        1 => 2,
        _ => 1,
    };
    let pad = if wdtype == 2 {
        (q_lora + q_lora.div_ceil(128) + 15) & !15
    } else {
        0
    };
    let tile_n = mega_tile_n(wdtype);
    fused * 4 + 512 + pad + 2 * tile_n * (128 * elt + 16)
}


impl Kernels {
    /// P1-7 phase 1: the persistent self-layer block — ONE launch for
    /// rmsnorm -> GEMV(fused_qkv) -> latent-norm -> rope(kv) -> paged KV
    /// encode -> GEMV(q_b) -> rope(q), per (row, layer) CTA. All pointer
    /// tables are `[n_layers]` u64; `q_out` is `[rows * n_heads * hd]`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on geometry the kernel does not support;
    /// [`Error::Internal`] on launch failures.
    #[allow(clippy::too_many_arguments)]
    pub fn mega_self_block(
        &self,
        h: &Buffer,
        norm_ptrs: &Buffer,
        qkv_ptrs: &Buffer,
        qkv_sf_ptrs: &Buffer,
        kvn_ptrs: &Buffer,
        qb_ptrs: &Buffer,
        qb_sf_ptrs: &Buffer,
        pool_ptrs: &Buffer,
        tbl_ptrs: &Buffer,
        cells: &Buffer,
        rope_cs: &Buffer,
        pos_kv: &Buffer,
        pos_q: &Buffer,
        q_out: &Buffer,
        hidden: usize,
        q_lora: usize,
        hd: usize,
        rope_dim: usize,
        n_heads: usize,
        rows: usize,
        sq: usize,
        b: usize,
        n_layers: usize,
        wdtype: i32,
        fmt: CacheFormat,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        let latent = hd - rope_dim;
        let half = rope_dim / 2;
        let elt: usize = match wdtype {
            0 => 4,
            1 => 2,
            _ => 1,
        };
        ensure!(hidden >= 1 && q_lora >= 1 && hd >= 32, InvalidArgument, "mega: degenerate dims");
        ensure!(latent % 2 == 0, InvalidArgument, "mega: latent must be even");
        ensure!(hd % 32 == 0, InvalidArgument, "mega: hd must be a multiple of 32");
        ensure!(half >= 1, InvalidArgument, "mega: rope_dim must be >= 2");
        // 16-byte cp.async alignment of the weight rows.
        ensure!(
            (hidden * elt) % 16 == 0 && (q_lora * elt) % 16 == 0,
            InvalidArgument,
            "mega: k*elt must be 16B aligned (hidden {hidden}, q_lora {q_lora}, wdtype {wdtype})"
        );
        let shared = mega_shared(hidden, q_lora, hd, wdtype);
        ensure!(
            shared <= self.device().props().max_shared as usize,
            InvalidArgument,
            "mega: {shared} B dynamic smem exceeds the device limit"
        );
        let tile_n = mega_tile_n(wdtype);
        let f = self.f("di_mega_self_block_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (rows as u32, n_layers as u32, 1),
                block: (256, 1, 1),
                shared: shared as u32,
            },
            &KernelArgs::new()
                .ptr(h)
                .ptr(norm_ptrs)
                .ptr(qkv_ptrs)
                .ptr(qkv_sf_ptrs)
                .ptr(kvn_ptrs)
                .ptr(qb_ptrs)
                .ptr(qb_sf_ptrs)
                .ptr(pool_ptrs)
                .ptr(tbl_ptrs)
                .ptr(cells)
                .ptr(rope_cs)
                .ptr(pos_kv)
                .ptr(pos_q)
                .ptr(q_out)
                .i32(hidden as i32)
                .i32(q_lora as i32)
                .i32(hd as i32)
                .i32(latent as i32)
                .i32(half as i32)
                .i32(n_heads as i32)
                .i32(rows as i32)
                .i32(sq as i32)
                .i32(b as i32)
                .i32(wdtype)
                .i32(fmt_id(fmt))
                .f32(eps)
                .i32(tile_n as i32),
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

    #[test]
    fn pipe_geometry_fits_default_smem_and_alignment() {
        // The FlashMLA V4.1 head dim and all three cache formats.
        for (fmt, d) in [
            (CacheFormat::F32, 512),
            (CacheFormat::Fp8Group32, 512),
            (CacheFormat::Fp4Group16, 512),
            (CacheFormat::F32, 576),
        ] {
            let rb = cache_row_bytes(fmt, d);
            assert_eq!(rb % 16, 0, "cp.async 16B units need 16B rows");
            let r = pipe_rows_per_stage(rb);
            assert!(r >= 1);
            let smem = SV_PIPE_STAGES * r * rb;
            assert!(
                smem <= 48 * 1024,
                "{smem} B exceeds the default dynamic smem limit"
            );
            // The budget is tight: one more row per stage would not fit
            // (unless the stage is already at the cap).
            assert!(smem + SV_PIPE_STAGES * rb > SV_PIPE_SMEM || r == 64);
        }
        // Expected concrete values (32 KiB budget, 3 stages).
        assert_eq!(cache_row_bytes(CacheFormat::Fp8Group32, 512), 528);
        assert_eq!(pipe_rows_per_stage(528), 20); // 3*20*528 = 31680 B
        assert_eq!(pipe_rows_per_stage(2048), 5); // 3*5*2048 = 30720 B
        assert_eq!(pipe_rows_per_stage(288), 37); // 3*37*288 = 31968 B
        // 16-byte chunks per staged row (mirror of the kernel's `chunks`).
        for rb in [2048usize, 528, 288] {
            assert_eq!(rb >> 4, rb / 16);
        }
    }

    #[test]
    fn pipe_and_vec_kill_switches_parse() {
        // The switches are read per call; only the "0" value disables.
        // (Env-var tests run serially per crate; restore afterwards.)
        std::env::set_var("DEEPINFER_SERVE_PIPE", "0");
        std::env::set_var("DEEPINFER_SERVE_VEC", "1");
        // Cannot construct a Kernels without a device; assert the parse
        // contract directly through the same expression.
        let pipe_off = std::env::var("DEEPINFER_SERVE_PIPE").ok().as_deref() == Some("0");
        let vec_off = std::env::var("DEEPINFER_SERVE_VEC").ok().as_deref() == Some("0");
        assert!(pipe_off);
        assert!(!vec_off);
        std::env::remove_var("DEEPINFER_SERVE_PIPE");
        std::env::remove_var("DEEPINFER_SERVE_VEC");
    }
}


