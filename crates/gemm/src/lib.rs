//! # deepinfer-gemm
//!
//! DeepGEMM-equivalent GEMM kernels for **DeepInfer-rs** (CPU reference
//! port). All GEMMs are NT layout: `D = C + A @ B^T` with `A: [m, k]` and
//! `B: [n, k]` (DeepGEMM README: "NT layout"). Kernels:
//!
//! * [`fp8_gemm_nt`] — dense FP8 NT GEMM with fine-grained scaling
//!   (activations `1 x 128`, weights `128 x 128`, UE8M0 scales, f32
//!   accumulation).
//! * [`m_grouped_fp8_gemm_nt_contiguous`] — MoE-prefill grouped GEMM with
//!   expert-contiguous rows (`m_indices`) and M-alignment padding.
//! * [`m_grouped_fp8_gemm_nt_masked`] — MoE-decode / CUDA-graph-style
//!   grouped GEMM with fixed per-group capacity (`masked_m`).
//! * [`fp8_einsum_bhr_hdr_bhd`] — the `"bhr,hdr->bhd"` einsum used by the
//!   FlashMLA fused-kernel epilogue (recipe `(1, 1, 32)`, e8m0 `q` scales,
//!   e4m3 `Wv` scales).
//! * [`bf16_gemm_nt`] — BF16 NT GEMM with f32 accumulation (reference
//!   path), [`f32_gemm_nt`] — naive sequential f32 GEMM (the golden
//!   reference).
//!
//! Upstream references: `deepseek-ai/DeepGEMM` (README: scaling recipes,
//! grouped layouts, `get_mk_alignment_for_contiguous_layout`),
//! `deepseek-ai/FlashMLA` (fused kernel FP8 epilogue einsum).
//!
//! ## Determinism & parallelism
//!
//! Every parallel kernel splits work over **disjoint output rows** via
//! `deepinfer_core::sync::global_pool().par_for`; within one output
//! element the K axis is accumulated sequentially in fixed (ascending)
//! order, so the same input always produces bit-identical output.
//! Per-k-block scaling follows DeepGEMM's promotion strategy: the inner
//! products of dequantized E4M3 values are summed in f32 per 128-wide
//! K block and the block product is then multiplied by
//! `sf_a(row, block) * sf_b(n-block, block)` and added to the f32
//! accumulator (documented consistently across all FP8 kernels).
//! [`f32_gemm_nt`] is fully sequential by design (golden reference).
//!
//! ## Validation
//!
//! Every public function validates shapes with checked arithmetic (DoS
//! guard) and returns [`deepinfer_core::Result`]; no input can cause a
//! panic. `k` must be a multiple of the 128-wide scale group for the FP8
//! GEMMs; `b` scale grids use ceil semantics over `n`
//! (`ceil(n/128)` block rows) matching
//! [`deepinfer_fp8::quantize_weight_per_block`].

#![forbid(unsafe_code)]

mod dense;
mod einsum;
mod grouped;

pub use dense::{bf16_gemm_nt, f32_gemm_nt, fp8_gemm_nt};
pub use einsum::fp8_einsum_bhr_hdr_bhd;
pub use grouped::{
    get_mk_alignment_for_contiguous_layout, m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked, pad_to_alignment,
};

/// Shared internal helpers (validated parallel row sweep, checked products,
/// the scaled inner-product kernel shared by every FP8 GEMM).
pub(crate) mod util {
    use deepinfer_core::format::e4m3_to_f32;
    use deepinfer_core::sync::global_pool;
    use deepinfer_core::{ensure, Error, Result};
    use std::sync::{Arc, Mutex};

    /// K granularity of the FP8 scale groups (DeepGEMM: 128).
    pub(crate) const SF_GROUP_K: usize = 128;

    /// Minimum number of output rows per parallel chunk. Chunks are always
    /// disjoint, so the split has no effect on results.
    pub(crate) const PAR_MIN_ROWS: usize = 16;

    /// Checked `usize` product with a capacity error (DoS guard).
    pub(crate) fn prod(a: usize, b: usize, what: &str) -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| Error::CapacityExceeded(format!("{what} overflows: {a} * {b}")))
    }

    /// Validates that `k` is a multiple of the FP8 scale group and returns
    /// the number of K blocks.
    pub(crate) fn k_blocks(k: usize) -> Result<usize> {
        ensure!(
            k % SF_GROUP_K == 0,
            InvalidArgument,
            "k = {k} must be a multiple of the FP8 scale group {SF_GROUP_K}"
        );
        Ok(k / SF_GROUP_K)
    }

    /// Dequantizes an E4M3 byte buffer to f32 (exact, order-preserving).
    pub(crate) fn dequant_e4m3(buf: &[u8]) -> Vec<f32> {
        buf.iter().map(|&b| e4m3_to_f32(b)).collect()
    }

    /// Runs `row_fn(row, out_row)` for every output row of length
    /// `row_len` on the global thread pool. Rows are handed out in disjoint
    /// chunks (deterministic: each row is computed exactly once,
    /// sequentially inside itself); results are assembled in row order.
    ///
    /// `total` must be the pre-validated product `rows * row_len`
    /// (callers check it with [`prod`] — multiplying here unchecked could
    /// overflow and mis-size the output buffer).
    pub(crate) fn par_rows<F>(rows: usize, row_len: usize, total: usize, row_fn: F) -> Vec<f32>
    where
        F: Fn(usize, &mut [f32]) + Send + Sync + 'static,
    {
        let out = Arc::new(Mutex::new(vec![0.0f32; total]));
        if rows > 0 {
            let sink = Arc::clone(&out);
            global_pool().par_for(rows, PAR_MIN_ROWS, move |s, e| {
                let mut local = vec![0.0f32; (e - s) * row_len];
                for r in s..e {
                    let base = (r - s) * row_len;
                    row_fn(r, &mut local[base..base + row_len]);
                }
                // Lock recovery instead of unwrap: a poisoned lock can only
                // originate from an external panic, and disjoint ranges make
                // silent recovery safe here.
                let mut g = sink.lock().unwrap_or_else(|p| p.into_inner());
                g[s * row_len..e * row_len].copy_from_slice(&local);
            });
        }
        take_shared(out)
    }

    /// Unwraps the shared output buffer without panicking (pool closures
    /// are all dropped once `par_for` returns, so this normally takes the
    /// fast path).
    fn take_shared(arc: Arc<Mutex<Vec<f32>>>) -> Vec<f32> {
        match Arc::try_unwrap(arc) {
            Ok(m) => match m.into_inner() {
                Ok(v) => v,
                Err(p) => p.into_inner(),
            },
            Err(a) => match a.lock() {
                Ok(g) => g.clone(),
                Err(p) => p.into_inner().clone(),
            },
        }
    }

    /// Scaled inner product of two dequantized K-rows (the shared FP8 GEMM
    /// inner loop). For each 128-wide K block the dequantized products are
    /// summed in f32, multiplied by `sf_a[block] * sf_b[block]`, and added
    /// to the accumulator in ascending block order (DeepGEMM promotion
    /// strategy, fixed for determinism).
    #[inline]
    pub(crate) fn scaled_row_dot(
        a_row: &[f32],
        a_sf: &[f32],
        b_row: &[f32],
        b_sf: &[f32],
        k_blocks: usize,
    ) -> f32 {
        let mut acc = 0.0f32;
        for kb in 0..k_blocks {
            let mut partial = 0.0f32;
            let k0 = kb * SF_GROUP_K;
            for kk in k0..k0 + SF_GROUP_K {
                partial += a_row[kk] * b_row[kk];
            }
            acc += a_sf[kb] * b_sf[kb] * partial;
        }
        acc
    }
}
