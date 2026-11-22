//! Device-side sampling launchers (see `kernels/sampler.cu`).
//!
//! [`Kernels::sample_topp`] is the serve engine's device sampler: it
//! consumes the `[B, vocab]` logits in place and produces `[B]` token ids,
//! never moving the logits to the host (vLLM-V1 / SGLang's on-device
//! sampling). [`Kernels::draft_pick`] is the fused DSpark draft step
//! ("MarkovSampleStep"): one embedding-dot + argmax + confidence pass with
//! no `[vocab]` logits materialized.

use deepinfer_core::{ensure, Result};
use deepinfer_gpu::{Buffer, KernelArgs, Launch, Stream};

use crate::Kernels;

#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_sample_topp_kernel",
    "di_draft_pick_part_kernel",
    "di_draft_pick_final_kernel",
];

/// Per-row sampling parameters uploaded as one `[rows, 6]` f32 block.
///
/// Field order (ABI with `di_sample_topp_kernel`):
/// `temperature, top_p, min_p, u, cap, top_k` — `cap` and `top_k` are
/// carried as f32 (exact for ids < 2^24) so the block is one contiguous
/// upload.
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleParams {
    /// Temperature (`<= 0`, `|t| < 1e-8` or NaN → greedy argmax).
    pub temperature: f32,
    /// Top-p nucleus (`< 1.0` enables the prefix cut).
    pub top_p: f32,
    /// Min-p relative floor (`> 0` enables).
    pub min_p: f32,
    /// The pre-drawn uniform (`[0, 1)`; unused on the greedy path).
    pub u: f32,
    /// Vocabulary cap (0 = none; ids `>= cap` are masked to `-inf`).
    pub cap: usize,
    /// Top-k (0 = disabled).
    pub top_k: usize,
}

impl SampleParams {
    /// The six-field row, ready for the `[rows, 6]` upload.
    pub fn words(&self) -> [f32; 6] {
        [
            self.temperature,
            self.top_p,
            self.min_p,
            self.u,
            self.cap as f32,
            self.top_k as f32,
        ]
    }

    /// Greedy rows consume no uniform (the host must not draw one).
    /// The CPU rule: `t <= 0`, `|t| < 1e-8` or NaN (NaN compares false
    /// everywhere, so any NaN temperature selects the argmax path).
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature < 1e-8
    }
}

/// Sample `rows` logit rows on device.
///
/// * `logits` — `[rows, vocab]` f32 (the serve workspace logits buffer).
/// * `params` — `[rows, 6]` f32 staged device block ([`SampleParams`]).
/// * `out_tok` — `[rows]` u32 output tokens.
/// * `out_status` — `[rows]` i32 statuses (0 ok; 1 no finite logit;
///   2 no positive mass; 3 invalid uniform).
///
/// The caller stages `params`/`out_*` (persistent workspace buffers) and
/// synchronizes; the launch itself adds no sync.
///
/// # Errors
/// [`deepinfer_core::Error::InvalidArgument`] on degenerate shapes.
pub fn sample_topp(
    k: &Kernels,
    logits: &Buffer,
    params: &Buffer,
    out_tok: &Buffer,
    out_status: &Buffer,
    rows: usize,
    vocab: usize,
    stream: &Stream,
) -> Result<()> {
    ensure!(rows > 0, InvalidArgument, "sample_topp needs rows > 0");
    ensure!(vocab > 0, InvalidArgument, "sample_topp needs vocab > 0");
    ensure!(
        logits.len() >= rows * vocab * 4,
        InvalidArgument,
        "sample_topp: logits buffer {} too small for {rows}x{vocab} f32",
        logits.len()
    );
    ensure!(
        params.len() >= rows * 6 * 4,
        InvalidArgument,
        "sample_topp: params buffer {} too small for {rows}x6 f32",
        params.len()
    );
    ensure!(
        out_tok.len() >= rows * 4 && out_status.len() >= rows * 4,
        InvalidArgument,
        "sample_topp: outputs too small for {rows} rows"
    );
    let f = k.f("di_sample_topp_kernel")?;
    f.launch(
        k.device(),
        Launch {
            grid: (rows as u32, 1, 1),
            block: (256, 1, 1),
            shared: 0,
        },
        &KernelArgs::new()
            .ptr(logits)
            .ptr(params)
            .ptr(out_tok)
            .ptr(out_status)
            .i32(vocab as i32),
        stream,
    )
}

/// One fused DSpark draft pick: `argmax` and top-1 softmax confidence of
/// `hidden · embedᵀ` (the tied MTP head), without materializing logits.
///
/// * `hidden` — `[H]` f32.
/// * `embed` — `[vocab, H]` f32 (the shared embedding table).
/// * `part_*` — `[slabs]` scratch (max, idx, sum).
/// * `out_tok` — `[1]` u32; `out_conf` — `[1]` f32.
///
/// Two launches (part + final), no logits round trip.
///
/// # Errors
/// [`deepinfer_core::Error::InvalidArgument`] on degenerate shapes.
pub fn draft_pick(
    k: &Kernels,
    hidden: &Buffer,
    embed: &Buffer,
    part_max: &Buffer,
    part_idx: &Buffer,
    part_sum: &Buffer,
    out_tok: &Buffer,
    out_conf: &Buffer,
    vocab: usize,
    hidden_dim: usize,
    stream: &Stream,
) -> Result<()> {
    ensure!(vocab > 0, InvalidArgument, "draft_pick needs vocab > 0");
    ensure!(
        hidden_dim > 0,
        InvalidArgument,
        "draft_pick needs hidden > 0"
    );
    ensure!(
        hidden.len() >= hidden_dim * 4,
        InvalidArgument,
        "draft_pick: hidden buffer too small"
    );
    ensure!(
        embed.len() >= vocab * hidden_dim * 4,
        InvalidArgument,
        "draft_pick: embed buffer too small for {vocab}x{hidden_dim}"
    );
    // Slab sizing: 256-token slabs keep the grid ~vocab/256 CTAs (the
    // dot work per CTA is 256 * H MACs, a good occupancy unit).
    const SLAB: usize = 256;
    let slabs = vocab.div_ceil(SLAB).max(1);
    ensure!(
        part_max.len() >= slabs * 4
            && part_idx.len() >= slabs * 4
            && part_sum.len() >= slabs * 4,
        InvalidArgument,
        "draft_pick: partial buffers too small for {slabs} slabs"
    );
    let f = k.f("di_draft_pick_part_kernel")?;
    f.launch(
        k.device(),
        Launch {
            grid: (slabs as u32, 1, 1),
            block: (256, 1, 1),
            shared: 0,
        },
        &KernelArgs::new()
            .ptr(hidden)
            .ptr(embed)
            .ptr(part_max)
            .ptr(part_idx)
            .ptr(part_sum)
            .i32(vocab as i32)
            .i32(hidden_dim as i32)
            .i32(SLAB as i32),
        stream,
    )?;
    let f = k.f("di_draft_pick_final_kernel")?;
    f.launch(
        k.device(),
        Launch {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared: 0,
        },
        &KernelArgs::new()
            .ptr(part_max)
            .ptr(part_idx)
            .ptr(part_sum)
            .ptr(out_tok)
            .ptr(out_conf)
            .i32(slabs as i32),
        stream,
    )
}
