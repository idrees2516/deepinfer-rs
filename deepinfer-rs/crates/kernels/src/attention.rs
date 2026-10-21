//! Attention launchers (FlashMLA decode split-KV + combine; prefill).

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::elementwise::CacheFormat;
use crate::gemm::ceil_div;
use crate::Kernels;

/// KV rows per decode split chunk (flash-decoding granularity).
pub const DECODE_CHUNK: usize = 256;

/// Kernel entry points defined in `kernels/attention.cu` (used by the
/// source-name sanity test).
#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_mla_decode_partial_kernel",
    "di_mla_decode_combine_kernel",
    "di_mla_prefill_kernel",
    "di_mla_decode_fp4_partial_kernel",
    "di_mla_prefill_dense_dev_kernel",
    "di_gather_rows_kernel",
    "di_swa_attention_dev_kernel",
];

impl Kernels {
    /// Sparse decode, partial pass (one call = whole batch). Workspace
    /// layout contract:
    /// * `pm` / `pl`: `[b * s_q * H, splits]` f32
    /// * `po`: `[b * s_q * H, splits, D]` f32
    ///
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
        self.mla_decode_partial_impl(
            "di_mla_decode_partial_kernel",
            q,
            cache,
            indices,
            pm,
            pl,
            po,
            b,
            s_q,
            h,
            d,
            topk,
            sm_scale,
            fmt,
            cache_rows,
            stream,
        )
    }

    /// **Dedicated FP4-cache decode partial** — same split-KV / online
    /// softmax semantics, accumulation order and outputs as
    /// [`Kernels::mla_decode_partial`] with `fmt = Fp4Group16` (bit-identical
    /// results), with a specialized 288 B-row unpacker: one e4m3 scale
    /// decode per 16-dim group per row (vs per element), one byte load per
    /// two dims with unrolled nibble extraction, and ILP=2 over KV rows.
    /// Requires `d == 512` (the FlashMLA V4.1 FP4 row width).
    pub fn mla_decode_fp4_partial(
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
        cache_rows: usize,
        stream: &Stream,
    ) -> Result<usize> {
        ensure!(d == 512, InvalidArgument, "fp4 decode requires D == 512 (FlashMLA V4.1 rows)");
        self.mla_decode_partial_impl(
            "di_mla_decode_fp4_partial_kernel",
            q,
            cache,
            indices,
            pm,
            pl,
            po,
            b,
            s_q,
            h,
            d,
            topk,
            sm_scale,
            CacheFormat::Fp4Group16,
            cache_rows,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_decode_partial_impl(
        &self,
        kernel: &'static str,
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
        ensure!(h >= 1, InvalidArgument, "mla decode requires H >= 1");
        ensure!(
            d % self.warp_size() as usize == 0,
            InvalidArgument,
            "head dim must be a multiple of the warp size"
        );
        if fmt == CacheFormat::Fp4Group16 {
            ensure!(d == 512, InvalidArgument, "fp4 rows require D == 512");
        }
        ensure!(topk > 0 || (b * s_q) == 0, InvalidArgument, "topk must be > 0");
        let splits = ceil_div(topk, DECODE_CHUNK).max(1);
        let block = 8 * self.warp_size();
        let f = self.f(kernel)?;
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

    /// Dense-causal prefill with the query offset and valid key count read
    /// from **device memory** (the graph-replayable decode path for the
    /// cross-attention window and the SWA ring, whose starts move every
    /// step). Key `j` is visible to query `i` iff `j <= *p_q_offset + i`
    /// and `j < *p_nkeys`. `out`: `[s_q, H, D]`; `lse`: optional `[s_q, H]`.
    pub fn mla_prefill_dense_dev(
        &self,
        q: &Buffer,
        cache: &Buffer,
        p_q_offset: &Buffer,
        p_nkeys: &Buffer,
        out: &Buffer,
        lse: Option<&Buffer>,
        s_q: usize,
        h: usize,
        d: usize,
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
        let f = self.f("di_mla_prefill_dense_dev_kernel")?;
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
                .ptr(p_q_offset)
                .ptr(p_nkeys)
                .ptr(out)
                .u64(lse.map(|x| x.ptr()).unwrap_or(0))
                .i32(s_q as i32)
                .i32(h as i32)
                .i32(d as i32)
                .i32(fmt.id()),
            stream,
        )
    }

    /// Row gather `dst[r] = src[(start + r) mod capacity]` for `r < n`,
    /// with `start` / `n` read from **device memory** (graph-replayable
    /// window staging; `max_rows` is the fixed capacity of `dst`).
    /// `row_bytes` is the row stride of both buffers.
    pub fn gather_rows(
        &self,
        src: &Buffer,
        dst: &Buffer,
        p_start: &Buffer,
        p_n: &Buffer,
        capacity: usize,
        row_bytes: usize,
        max_rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(capacity > 0 && row_bytes > 0 && max_rows > 0, InvalidArgument, "gather needs positive geometry");
        self.launch1d(
            "di_gather_rows_kernel",
            max_rows * row_bytes,
            256,
            0,
            KernelArgs::new()
                .ptr(src)
                .ptr(dst)
                .ptr(p_start)
                .ptr(p_n)
                .i32(capacity as i32)
                .i32(row_bytes as i32)
                .i32(max_rows as i32),
            stream,
        )
    }
}

impl Kernels {
    /// Sliding-window self-attention over staged ring rows with the query
    /// offset and key count read from device memory (shared by the eager
    /// prefill — cells uploaded per call — and the captured decode graph).
    /// `q`: `[s_q, D]`; `kv`: rows `[0, *p_nkeys)` valid; `out`: `[s_q, D]`.
    /// `D` may exceed the 16-per-lane budget of the MLA kernels (it is the
    /// model hidden size here). The output accumulator is dynamic shared
    /// memory of `D * 4 + 2 * block * 4` bytes.
    pub fn swa_attention_dev(
        &self,
        q: &Buffer,
        kv: &Buffer,
        p_q_offset: &Buffer,
        p_nkeys: &Buffer,
        out: &Buffer,
        s_q: usize,
        d: usize,
        sm_scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(s_q >= 1 && d >= 1, InvalidArgument, "swa needs s_q/d >= 1");
        let block = 256usize;
        let shared = d
            .checked_mul(4)
            .and_then(|v| v.checked_add(2 * block * 4))
            .ok_or_else(|| deepinfer_core::Error::CapacityExceeded("swa smem size overflows".into()))?;
        // 32 KB ov + 2 KB red is the realistic budget; larger hidden needs
        // a multi-CTA variant (not needed for the reference configs).
        ensure!(shared <= 64 * 1024, CapacityExceeded, "swa smem {shared} B exceeds 64 KB");
        let f = self.f("di_swa_attention_dev_kernel")?;
        f.launch(
            self.device(),
            Launch {
                grid: (s_q as u32, 1, 1),
                block: (block as u32, 1, 1),
                shared: shared as u32,
            },
            &KernelArgs::new()
                .ptr(q)
                .ptr(kv)
                .ptr(p_q_offset)
                .ptr(p_nkeys)
                .ptr(out)
                .i32(s_q as i32)
                .i32(d as i32)
                .f32(sm_scale),
            stream,
        )
    }
}
