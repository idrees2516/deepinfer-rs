//! Attention launchers (FlashMLA decode split-KV + combine; prefill).

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::elementwise::CacheFormat;
use crate::gemm::ceil_div;
use crate::Kernels;

/// KV rows per decode split chunk (flash-decoding granularity).
pub const DECODE_CHUNK: usize = 256;

impl Kernels {
    /// Sparse decode, partial pass (one call = whole batch). Workspace
    /// layout contract:
    /// * `pm` / `pl`: `[b * s_q * H, splits]` f32
    /// * `po`: `[b * s_q * H, splits, D]` f32
    /// Returns the number of splits actually launched.
    pub fn mla_decode_partial(
        &self,
        q: &Buffer,
        cache: &Buffer,
        indices: &Buffer,
        pm: &Buffer,
        pl: &Buffer,
        po: &Buffer,
        b: usize,
        s_q: usize,
        h: usize,
        d: usize,
        topk: usize,
        sm_scale: f32,
        fmt: CacheFormat,
        cache_rows: usize,
        stream: &Stream,
    ) -> Result<usize> {
        ensure!(h % 8 == 0, InvalidArgument, "mla decode requires H % 8 == 0 (warp grouping)");
        ensure!(
            d % self.warp_size() as usize == 0,
            InvalidArgument,
            "head dim must be a multiple of the warp size"
        );
        ensure!(topk > 0 || (b * s_q) == 0, InvalidArgument, "topk must be > 0");
        let splits = ceil_div(topk, DECODE_CHUNK).max(1);
        let block = 8 * self.warp_size();
        let f = self.f("di_mla_decode_partial_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (splits as u32, ceil_div(h, 8) as u32, (b * s_q).max(1) as u32),
                block: (block, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(cache)
                .ptr(indices)
                .ptr(pm)
                .ptr(pl)
                .ptr(po)
                .i32(h as i32)
                .i32(d as i32)
                .i32(topk as i32)
                .i32(DECODE_CHUNK as i32)
                .f32(sm_scale)
                .i32(fmt.id())
                .i32(cache_rows as i32),
            stream,
        )?;
        Ok(splits)
    }

    /// Sparse decode, combine pass: merge partials, normalize, fill `lse`
    /// (`[b, H, s_q]`, CPU layout), apply optional `sink` (`[H]`).
    pub fn mla_decode_combine(
        &self,
        pm: &Buffer,
        pl: &Buffer,
        po: &Buffer,
        sink: Option<&Buffer>,
        out: &Buffer,
        lse: &Buffer,
        b: usize,
        s_q: usize,
        h: usize,
        d: usize,
        splits: usize,
        stream: &Stream,
    ) -> Result<()> {
        let f = self.f("di_mla_decode_combine_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(d, 256) as u32, (b * s_q * h).max(1) as u32, 1),
                block: (256, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(pm)
                .ptr(pl)
                .ptr(po)
                .u64(sink.map(|s| s.ptr()).unwrap_or(0))
                .ptr(out)
                .ptr(lse)
                .i32(s_q as i32)
                .i32(h as i32)
                .i32(d as i32)
                .i32(splits as i32),
            stream,
        )
    }

    /// Prefill attention (dense causal when `indices` is None; sparse
    /// gather otherwise). `out`: `[s_q, H, D]`; `max_logits`/`lse`:
    /// optional `[s_q, H]`.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_prefill(
        &self,
        q: &Buffer,
        cache: &Buffer,
        indices: Option<&Buffer>,
        topk_length: Option<&Buffer>,
        out: &Buffer,
        max_logits: Option<&Buffer>,
        lse: Option<&Buffer>,
        s_q: usize,
        h: usize,
        d: usize,
        s_kv: usize,
        q_offset: usize,
        topk: usize,
        sm_scale: f32,
        fmt: CacheFormat,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(h >= 1 && d >= 1 && s_q >= 1, InvalidArgument, "prefill needs positive dims");
        ensure!(
            d % self.warp_size() as usize == 0,
            InvalidArgument,
            "head dim must be a multiple of the warp size"
        );
        let block = 8 * self.warp_size();
        let f = self.f("di_mla_prefill_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (ceil_div(s_q, 8) as u32, h as u32, 1),
                block: (block, 1, 1),
                shared: 0,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(cache)
                .u64(indices.map(|x| x.ptr()).unwrap_or(0))
                .u64(topk_length.map(|x| x.ptr()).unwrap_or(0))
                .ptr(out)
                .u64(max_logits.map(|x| x.ptr()).unwrap_or(0))
                .u64(lse.map(|x| x.ptr()).unwrap_or(0))
                .i32(s_q as i32)
                .i32(h as i32)
                .i32(d as i32)
                .i32(s_kv as i32)
                .i32(q_offset as i32)
                .i32(topk as i32)
                .f32(sm_scale)
                .i32(fmt.id())
                .i32(s_kv as i32), // cache_rows (dense rows == s_kv)
            stream,
        )
    }
}
