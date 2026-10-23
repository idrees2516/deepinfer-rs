//! Launchers for the graph-replayable `_dev` decode kernel family
//! (see `kernels/decode.cu`).
//!
//! A CUDA/HIP graph bakes launch geometry and scalar arguments at capture
//! time, but a decode step's scalars change every token. Every launcher in
//! this module therefore:
//!
//! * reads the varying scalars from the device-resident [`DecodeParams`]
//!   block (written once per step by [`Kernels::step_params`], which itself
//!   only needs one host-uploaded int — the position being decoded);
//! * derives its grid from **fixed maxima** ([`StepGeometry`]), never from
//!   the current sequence length, so one captured graph serves the whole
//!   sequence.
//!
//! The eager decode path launches the exact same kernels, so graph replay
//! is bit-identical to eager execution.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::dsamoe::MERGE_CHUNK;
use crate::gemm::ceil_div;
use crate::Kernels;

/// Kernel entry points defined in `kernels/decode.cu` (used by the
/// source-name sanity test).
#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_step_params_kernel",
    "di_pos_fill_kernel",
    "di_kv_encode_fp8_dev_kernel",
    "di_kv_encode_fp4_dev_kernel",
    "di_append_row_kernel",
    "di_append_norm_row_kernel",
    "di_lightning_block_zero_dev_kernel",
    "di_lightning_amax_dev_kernel",
    "di_lightning_finalize_dev_kernel",
    "di_lightning_quant_dev_kernel",
    "di_thunder_logits_dev_kernel",
    "di_dot_scores_dev_kernel",
    "di_gate_blend_dev_kernel",
    "di_causal_window_clip_dev_kernel",
    "di_block_preselect_dev_kernel",
    "di_conv_feat_dev_kernel",
    "di_conv_gated_norm_dev_kernel",
];

/// Field offsets of the device-resident decode parameter block — the Rust
/// mirror of `kernels/decode_params.h` (ABI: both sides must agree; the
/// source-name test pins the header's contents).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeParams;

impl DecodeParams {
    /// Seq-local position of the token being decoded (= old length).
    pub const POS: usize = 0;
    /// Arena start row of this sequence (host-written input).
    pub const START: usize = 1;
    /// New total sequence length (seq-local).
    pub const LEN: usize = 2;
    /// Arena row of the new token (`START + POS`).
    pub const ROW: usize = 3;
    /// DSA / attention key count (= new total).
    pub const S_KV: usize = 4;
    /// Complete lightning-quant blocks (seq-local).
    pub const K_NBLOCKS: usize = 5;
    /// Last (possibly partial) block index (seq-local).
    pub const LAST_BLOCK: usize = 6;
    /// Cross-attention window: first **arena** row.
    pub const CROSS_LO: usize = 7;
    /// Cross-attention window: key count.
    pub const CROSS_NKEYS: usize = 8;
    /// Cross-attention dense-causal q_offset.
    pub const CROSS_QOFF: usize = 9;
    /// SWA ring: key count.
    pub const SWA_N: usize = 10;
    /// SWA dense-causal q_offset.
    pub const SWA_QOFF: usize = 11;
    /// Fixed selection width (config topk_tokens).
    pub const TOPK: usize = 12;
    /// SWA ring: slot of the oldest visible row.
    pub const SWA_START: usize = 13;
    /// Number of fields.
    pub const COUNT: usize = 14;
}

/// Host-side mirror of `di_step_params_kernel` (pure function; unit-tested
/// against the device kernel's contract).
pub fn fanout_params(
    pos: i32,
    start: i32,
    cross_window: usize,
    swa_window: usize,
    quant_block: usize,
    topk: usize,
) -> [i32; DecodeParams::COUNT] {
    let len = pos + 1;
    let cross_n = (len as usize).min(cross_window) as i32;
    let swa_n = (len as usize).min(swa_window) as i32;
    let mut p = [0i32; DecodeParams::COUNT];
    p[DecodeParams::POS] = pos;
    p[DecodeParams::START] = start;
    p[DecodeParams::LEN] = len;
    p[DecodeParams::ROW] = start + pos;
    p[DecodeParams::S_KV] = len;
    p[DecodeParams::K_NBLOCKS] = len / quant_block as i32;
    p[DecodeParams::LAST_BLOCK] = pos / quant_block as i32;
    p[DecodeParams::CROSS_LO] = start + len - cross_n;
    p[DecodeParams::CROSS_NKEYS] = cross_n;
    p[DecodeParams::CROSS_QOFF] = cross_n - 1;
    p[DecodeParams::SWA_N] = swa_n;
    p[DecodeParams::SWA_QOFF] = swa_n - 1;
    p[DecodeParams::TOPK] = topk as i32;
    p[DecodeParams::SWA_START] = (len - swa_n) % (swa_window as i32).max(1);
    p
}

/// Fixed launch geometry of the decode step (all maxima; computed once at
/// model construction and shared by eager and captured launches).
#[derive(Debug, Clone)]
pub struct StepGeometry {
    /// Maximum key count (= max_position_embeddings).
    pub max_s_kv: usize,
    /// Selection width (config topk_tokens).
    pub topk: usize,
    /// DSA preselect block size.
    pub block: usize,
    /// DSA preselect survivors per block.
    pub per: usize,
    /// Number of preselect blocks at `max_s_kv` (fixed).
    pub num_blocks: usize,
    /// Candidates per query row after preselect (fixed).
    pub cands_cap: usize,
}

impl StepGeometry {
    /// Geometry for a config. `block`/`per` are the DSA preselect
    /// parameters; `topk` the (fixed) selection width.
    pub fn new(max_s_kv: usize, topk: usize, block: usize, per: usize) -> Result<StepGeometry> {
        ensure!(max_s_kv >= 1, InvalidArgument, "max_s_kv must be >= 1");
        ensure!(topk >= 1, InvalidArgument, "topk must be >= 1");
        ensure!((1..=1024).contains(&block), InvalidArgument, "preselect block in [1, 1024]");
        ensure!(per >= 1, InvalidArgument, "per must be >= 1");
        let num_blocks = ceil_div(max_s_kv, block);
        let cands_cap = num_blocks
            .checked_mul(per)
            .ok_or_else(|| deepinfer_core::Error::CapacityExceeded("cands_cap overflows".into()))?;
        Ok(StepGeometry {
            max_s_kv,
            topk,
            block,
            per,
            num_blocks,
            cands_cap,
        })
    }
}

/// Byte offset of field `f` inside a `DecodeParams` block (for raw-pointer
/// arguments into the device cell).
pub(crate) fn p_at(params: &Buffer, f: usize) -> u64 {
    params.ptr() + (f * std::mem::size_of::<i32>()) as u64
}

impl Kernels {
    /// Fan the host-written `{pos, start}` pair (in `p_pos[0..2]`) out
    /// into the `params` cell block `[DecodeParams::COUNT]` i32. Must run
    /// once per decode step, before every other `_dev` kernel.
    pub fn step_params(
        &self,
        p_pos: &Buffer,
        params: &Buffer,
        cross_window: usize,
        swa_window: usize,
        quant_block: usize,
        topk: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(params.len() >= DecodeParams::COUNT * 4, InvalidArgument, "params cell too small");
        self.launch1d(
            "di_step_params_kernel",
            1,
            64,
            0,
            KernelArgs::new()
                .ptr(p_pos)
                .ptr(params)
                .i32(cross_window as i32)
                .i32(swa_window as i32)
                .i32(quant_block as i32)
                .i32(topk as i32),
            stream,
        )
    }

    /// Fill the RoPE position rows for one decode token: `pos_q [H]` all
    /// equal to the decoded position, `pos_kv [1]` likewise.
    pub fn pos_fill(
        &self,
        pos_q: &Buffer,
        pos_kv: &Buffer,
        p_pos: &Buffer,
        h: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(h >= 1, InvalidArgument, "pos_fill needs H >= 1");
        self.launch1d(
            "di_pos_fill_kernel",
            h.max(1),
            128,
            0,
            KernelArgs::new().ptr(pos_q).ptr(pos_kv).ptr(p_pos).i32(h as i32),
            stream,
        )
    }

    /// Append one FP8 KV-cache row (`x`: `[D]` f32) at the device-read
    /// `params[KV_WRITE]` row of `cache`.
    pub fn kv_encode_fp8_dev(
        &self,
        x: &Buffer,
        cache: &Buffer,
        params: &Buffer,
        d: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.kv_encode_dev("di_kv_encode_fp8_dev_kernel", x, cache, params, d, stream)
    }

    /// Append one FP4 KV-cache row (`x`: `[D]` f32) at the device-read
    /// `params[KV_WRITE]` row of `cache`.
    pub fn kv_encode_fp4_dev(
        &self,
        x: &Buffer,
        cache: &Buffer,
        params: &Buffer,
        d: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.kv_encode_dev("di_kv_encode_fp4_dev_kernel", x, cache, params, d, stream)
    }

    fn kv_encode_dev(
        &self,
        kernel: &'static str,
        x: &Buffer,
        cache: &Buffer,
        params: &Buffer,
        d: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(d >= 32, InvalidArgument, "kv encode needs D >= 32");
        let group = if kernel.contains("fp8") { 32 } else { 16 };
        ensure!(d % group == 0, InvalidArgument, "D must be a multiple of the group size");
        let warps_per_block = 64 / self.warp_size() as usize;
        let grid = (d / group).div_ceil(warps_per_block).max(1);
        let f = self.f(kernel)?;
        f.launch(
            self.device(),
            Launch {
                grid: (grid as u32, 1, 1),
                block: (64, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x)
                .ptr(cache)
                .raw_ptr(p_at(params, DecodeParams::ROW))
                .i32(d as i32),
            stream,
        )
    }

    /// Append one f32 row (`src`: `floats` values) at the device-read
    /// `params[ROW]` row of `dst` (row stride `floats`).
    pub fn append_row(
        &self,
        src: &Buffer,
        dst: &Buffer,
        params: &Buffer,
        floats: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(floats > 0, InvalidArgument, "append_row needs floats > 0");
        self.launch1d(
            "di_append_row_kernel",
            floats,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .raw_ptr(p_at(params, DecodeParams::ROW))
                .i32(floats as i32),
            stream,
        )
    }

    /// Append the per-(token, head) RMSNormed copy of one indexer k row at
    /// the device-read `params[ROW]` row (the thunder-side cache).
    pub fn append_norm_row(
        &self,
        src: &Buffer,
        dst: &Buffer,
        params: &Buffer,
        nh: usize,
        hd: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(nh >= 1 && hd >= 1, InvalidArgument, "append_norm_row needs nh/hd >= 1");
        self.launch1d(
            "di_append_norm_row_kernel",
            nh,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .raw_ptr(p_at(params, DecodeParams::ROW))
                .i32(nh as i32)
                .i32(hd as i32),
            stream,
        )
    }

    /// Incremental lightning quant of the LAST quant block (device-read
    /// `params[LAST_BLOCK]`, completion-guarded by `params[LEN]`).
    /// No-ops until the block completes; idempotent once it has.
    /// `sf_bits`: `[max_blocks]` u32 scratch; `sf`: `[max_blocks]` u8;
    /// `q_base`: `[max_rows * dim]` u8 payload.
    pub fn lightning_block_quant_dev(
        &self,
        x_base: &Buffer,
        q_base: &Buffer,
        sf: &Buffer,
        sf_bits: &Buffer,
        params: &Buffer,
        dim: usize,
        quant_block: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(dim >= 1 && quant_block >= 1, InvalidArgument, "lightning needs dim/quant_block >= 1");
        let per_block = quant_block
            .checked_mul(dim)
            .ok_or_else(|| deepinfer_core::Error::CapacityExceeded("block elems overflow".into()))?;
        // 1) zero the block's atomic amax cell
        self.launch1d(
            "di_lightning_block_zero_dev_kernel",
            1,
            256,
            0,
            KernelArgs::new()
                .ptr(sf_bits)
                .raw_ptr(p_at(params, DecodeParams::LAST_BLOCK)),
            stream,
        )?;
        // 2) partial amax (fixed chunk grid over one block)
        let chunk_elems = 1usize << 15;
        let chunks = ceil_div(per_block, chunk_elems).max(1);
        let f = self.f("di_lightning_amax_dev_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (chunks as u32, 1, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(x_base)
                .ptr(sf_bits)
                .raw_ptr(p_at(params, DecodeParams::LAST_BLOCK))
                .raw_ptr(p_at(params, DecodeParams::LEN))
                .i32(dim as i32)
                .i32(quant_block as i32)
                .u64(chunk_elems as u64),
            stream,
        )?;
        // 3) amax bits -> ue8m0 scale byte
        self.launch1d(
            "di_lightning_finalize_dev_kernel",
            1,
            256,
            0,
            KernelArgs::new()
                .ptr(sf_bits)
                .ptr(sf)
                .raw_ptr(p_at(params, DecodeParams::LAST_BLOCK)),
            stream,
        )?;
        // 4) quantize the block payload
        self.launch1d(
            "di_lightning_quant_dev_kernel",
            per_block,
            256,
            0,
            KernelArgs::new()
                .ptr(x_base)
                .ptr(q_base)
                .ptr(sf)
                .raw_ptr(p_at(params, DecodeParams::LAST_BLOCK))
                .raw_ptr(p_at(params, DecodeParams::LEN))
                .i32(dim as i32)
                .i32(quant_block as i32),
            stream,
        )
    }

    /// Thunder logits over a device-read `s_kv` with a fixed max-cols grid.
    /// `scores`: `[s_q, max_cols]` f32 (only `[., s_kv]` written).
    #[allow(clippy::too_many_arguments)]
    pub fn thunder_logits_dev(
        &self,
        q_f32: &Buffer,
        q_u8: &Buffer,
        q_sf: &Buffer,
        k_f32: &Buffer,
        k_u8: &Buffer,
        k_sf: &Buffer,
        w: &Buffer,
        scores: &Buffer,
        params: &Buffer,
        s_q: usize,
        nh: usize,
        hd: usize,
        q_nblocks: usize,
        quant_block: usize,
        max_cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        let dim = nh * hd;
        let f = self.f("di_thunder_logits_dev_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(max_cols, 256) as u32, s_q as u32, 1),
                block: (256, 1, 1),
                shared: (dim * 4) as u32,
            },
            &KernelArgs::new()
                .ptr(q_f32)
                .ptr(q_u8)
                .ptr(q_sf)
                .ptr(k_f32)
                .ptr(k_u8)
                .ptr(k_sf)
                .ptr(w)
                .ptr(scores)
                .raw_ptr(p_at(params, DecodeParams::S_KV))
                .raw_ptr(p_at(params, DecodeParams::K_NBLOCKS))
                .i32(s_q as i32)
                .i32(nh as i32)
                .i32(hd as i32)
                .i32(q_nblocks as i32)
                .i32(quant_block as i32)
                .i32(max_cols as i32),
            stream,
        )
    }

    /// Conv-branch scores over a device-read `s_kv` (fixed max-cols grid).
    pub fn dot_scores_dev(
        &self,
        q: &Buffer,
        feat: &Buffer,
        out: &Buffer,
        params: &Buffer,
        s_q: usize,
        hidden: usize,
        max_cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_dot_scores_dev_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(max_cols, 256) as u32, s_q as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(feat)
                .ptr(out)
                .raw_ptr(p_at(params, DecodeParams::S_KV))
                .i32(s_q as i32)
                .i32(hidden as i32)
                .i32(max_cols as i32),
            stream,
        )
    }

    /// Gate blend over the fixed `max_cols` width; entries `j >= s_kv` are
    /// sanitized to `-inf` (tail masking for the fixed-geometry preselect).
    pub fn gate_blend_dev(
        &self,
        gt: &Buffer,
        gc: &Buffer,
        conv: &Buffer,
        thunder: &Buffer,
        params: &Buffer,
        s_q: usize,
        max_cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_gate_blend_dev_kernel",
            s_q * max_cols,
            256,
            0,
            KernelArgs::new()
                .ptr(gt)
                .ptr(gc)
                .ptr(conv)
                .ptr(thunder)
                .raw_ptr(p_at(params, DecodeParams::S_KV))
                .i32(s_q as i32)
                .i32(max_cols as i32),
            stream,
        )
    }

    /// Causal (`j > i`) + extrapolation-window clip and tail sanitization
    /// over the fixed `max_cols` width. `window == 0` disables the window.
    pub fn causal_window_clip_dev(
        &self,
        scores: &Buffer,
        params: &Buffer,
        s_q: usize,
        max_cols: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_causal_window_clip_dev_kernel",
            s_q * max_cols,
            256,
            0,
            KernelArgs::new()
                .ptr(scores)
                .raw_ptr(p_at(params, DecodeParams::S_KV))
                .i32(s_q as i32)
                .i32(max_cols as i32)
                .i32(window as i32),
            stream,
        )
    }

    /// Full fixed-geometry top-K selection over a device-read `s_kv`:
    /// preselect (device-scalar) + the fixed merge tree + scatter. `out`:
    /// `[s_q, k]` i32 ascending with `-1` padding; `cands_a`/`cands_b`:
    /// `[s_q, geo.cands_cap]` u64 scratch (zeroed here).
    #[allow(clippy::too_many_arguments)]
    pub fn select_top_k_dev(
        &self,
        scores: &Buffer,
        cands_a: &Buffer,
        cands_b: &Buffer,
        out: &Buffer,
        params: &Buffer,
        s_q: usize,
        geo: &StepGeometry,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(s_q > 0, InvalidArgument, "select needs s_q > 0");
        ensure!(geo.block <= 1024, InvalidArgument, "preselect block must be <= 1024");
        // 1) zero the candidate buffer (padding keys must be 0)
        self.fill_zero(cands_a, s_q * geo.cands_cap * 8, stream)?;
        // 2) per-block preselect over the device-read s_kv, fixed grid
        let f = self.f("di_block_preselect_dev_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (geo.num_blocks as u32, s_q as u32, 1),
                block: (geo.block as u32, 1, 1),
                shared: (geo.block * 8) as u32,
            },
            &KernelArgs::new()
                .ptr(scores)
                .ptr(cands_a)
                .raw_ptr(p_at(params, DecodeParams::S_KV))
                .i32(s_q as i32)
                .i32(geo.block as i32)
                .i32(geo.per as i32)
                .i32(geo.max_s_kv as i32),
            stream,
        )?;
        // 3) fixed merge rounds down to MERGE_CHUNK, then the final top-K
        let mut cur = cands_a;
        let mut nxt = cands_b;
        let mut n = geo.cands_cap;
        while n > MERGE_CHUNK {
            let keep = n / 2;
            let f = self.f("di_topk_merge_kernel")?;
            f.launch(
                self.device(),
                Launch {
                    grid: (ceil_div(n, MERGE_CHUNK) as u32, s_q as u32, 1),
                    block: (1024, 1, 1),
                    shared: (MERGE_CHUNK * 8) as u32,
                },
                &KernelArgs::new()
                    .ptr(cur)
                    .ptr(nxt)
                    .i32(n as i32)
                    .i32(keep as i32)
                    .i32(MERGE_CHUNK as i32),
                stream,
            )?;
            std::mem::swap(&mut cur, &mut nxt);
            n = keep;
        }
        {
            let f = self.f("di_topk_merge_kernel")?;
            f.launch(
                self.device(),
                Launch {
                    grid: (1, s_q as u32, 1),
                    block: (1024, 1, 1),
                    shared: (MERGE_CHUNK * 8) as u32,
                },
                &KernelArgs::new()
                    .ptr(cur)
                    .ptr(nxt)
                    .i32(n as i32)
                    .i32(geo.topk as i32)
                    .i32(MERGE_CHUNK as i32),
                stream,
            )?;
        }
        // 4) scatter to ascending positions with -1 padding
        let k_pow2 = geo.topk.next_power_of_two();
        let f = self.f("di_topk_scatter_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (1, s_q as u32, 1),
                block: (256, 1, 1),
                shared: (k_pow2 * 4) as u32,
            },
            &KernelArgs::new()
                .ptr(nxt)
                .ptr(out)
                .i32(geo.topk as i32)
                .i32(geo.topk as i32),
            stream,
        )?;
        let _ = window; // windowing was applied by causal_window_clip_dev
        Ok(())
    }

    /// Conv features for the single new key row (device-read row index).
    pub fn conv_feat_dev(
        &self,
        k: &Buffer,
        feat: &Buffer,
        kw: &Buffer,
        kw_off: &Buffer,
        num_kernels: usize,
        params: &Buffer,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(hidden > 0, InvalidArgument, "conv needs hidden > 0");
        self.launch1d(
            "di_conv_feat_dev_kernel",
            hidden,
            256,
            0,
            KernelArgs::new()
                .ptr(k)
                .ptr(feat)
                .ptr(kw)
                .ptr(kw_off)
                .i32(num_kernels as i32)
                .raw_ptr(p_at(params, DecodeParams::ROW))
                .i32(hidden as i32),
            stream,
        )
    }

    /// Gated RMSNorm of the single key row at the device-read index.
    pub fn conv_gated_norm_dev(
        &self,
        feat: &Buffer,
        gain: &Buffer,
        params: &Buffer,
        hidden: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(hidden > 0, InvalidArgument, "gated norm needs hidden > 0");
        self.launch1d(
            "di_conv_gated_norm_dev_kernel",
            1,
            256,
            0,
            KernelArgs::new()
                .ptr(feat)
                .ptr(gain)
                .raw_ptr(p_at(params, DecodeParams::ROW))
                .i32(hidden as i32),
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fanout_matches_the_documented_semantics() {
        // pos = 5, start = 10, windows 4 / 3, quant_block 4, topk 7.
        let p = fanout_params(5, 10, 4, 3, 4, 7);
        assert_eq!(p[DecodeParams::POS], 5);
        assert_eq!(p[DecodeParams::START], 10);
        assert_eq!(p[DecodeParams::LEN], 6);
        assert_eq!(p[DecodeParams::ROW], 15);
        assert_eq!(p[DecodeParams::S_KV], 6);
        assert_eq!(p[DecodeParams::K_NBLOCKS], 1); // 6 / 4
        assert_eq!(p[DecodeParams::LAST_BLOCK], 1); // 5 / 4
        assert_eq!(p[DecodeParams::CROSS_LO], 12); // 10 + 6 - min(6, 4)
        assert_eq!(p[DecodeParams::CROSS_NKEYS], 4);
        assert_eq!(p[DecodeParams::CROSS_QOFF], 3);
        assert_eq!(p[DecodeParams::SWA_N], 3);
        assert_eq!(p[DecodeParams::SWA_QOFF], 2);
        assert_eq!(p[DecodeParams::TOPK], 7);
        assert_eq!(p[DecodeParams::SWA_START], 0); // (6 - 3) % 3
        // Window larger than len clamps.
        let q = fanout_params(0, 0, 512, 128, 2048, 2048);
        assert_eq!(q[DecodeParams::LEN], 1);
        assert_eq!(q[DecodeParams::ROW], 0);
        assert_eq!(q[DecodeParams::CROSS_NKEYS], 1);
        assert_eq!(q[DecodeParams::CROSS_QOFF], 0);
        assert_eq!(q[DecodeParams::SWA_N], 1);
        assert_eq!(q[DecodeParams::SWA_START], 0);
        assert_eq!(q[DecodeParams::K_NBLOCKS], 0);
        assert_eq!(q[DecodeParams::LAST_BLOCK], 0);
        // Block completion boundary: len == quant_block.
        let r = fanout_params(7, 3, 8, 8, 8, 8);
        assert_eq!(r[DecodeParams::LEN], 8);
        assert_eq!(r[DecodeParams::ROW], 10);
        assert_eq!(r[DecodeParams::K_NBLOCKS], 1);
        assert_eq!(r[DecodeParams::LAST_BLOCK], 0); // block 0 just completed
    }

    #[test]
    fn decode_params_header_matches_rust_constants() {
        let h = crate::source::DECODE_PARAMS;
        for (i, name) in [
            "DP_POS",
            "DP_START",
            "DP_LEN",
            "DP_ROW",
            "DP_S_KV",
            "DP_K_NBLOCKS",
            "DP_LAST_BLOCK",
            "DP_CROSS_LO",
            "DP_CROSS_NKEYS",
            "DP_CROSS_QOFF",
            "DP_SWA_N",
            "DP_SWA_QOFF",
            "DP_TOPK",
            "DP_SWA_START",
        ]
        .iter()
        .enumerate()
        {
            assert!(h.contains(&format!("#define {name} {i}")), "{name} != {i}");
        }
        assert!(h.contains("#define DP_COUNT 14"));
    }

    #[test]
    fn step_geometry_is_fixed_and_consistent() {
        let g = StepGeometry::new(1024, 8, 64, 2).unwrap();
        assert_eq!(g.num_blocks, 16);
        assert_eq!(g.cands_cap, 32);
        assert!(StepGeometry::new(0, 8, 64, 2).is_err());
        assert!(StepGeometry::new(1024, 0, 64, 2).is_err());
        assert!(StepGeometry::new(1024, 8, 2048, 2).is_err());
    }
}
