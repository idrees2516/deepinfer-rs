//! The GPU-resident DeepSeek-V4.1-Flash model — the `DeviceModel`.
//!
//! This is the wiring of the `deepinfer-kernels` operator layer into the
//! runtime that `docs/GPU.md`'s "Model-loop integration blueprint"
//! specifies: weights are quantized on the host with `deepinfer-fp8`
//! (128×128 blocks) and uploaded once; every forward pass then runs
//! entirely on the device (one stream, no implicit synchronization), and
//! only the final logits row crosses back to the host for sampling
//! (`deepinfer-sampler` stays host-side — it is latency-trivial and
//! batch-invariant by design).
//!
//! # Structure
//!
//! * [`Precision`] selects per-GEMM numerics: **F32** (the
//!   reference-precision mode used by the CPU-golden tests) or **Auto**
//!   (FP8 with the DeepGEMM 1×128/128×128 recipe whenever `k % 128 == 0`,
//!   BF16 otherwise — at V4.1 dims every GEMM is FP8).
//! * Per self-decoder layer the device keeps the append-only caches as
//!   flat arenas: the MLA KV rows (f32 / 528 B FP8 / 288 B FP4 per
//!   token), the DSA indexer's raw k rows (the conv branch), the
//!   per-head-normed k rows (the thunder branch) with their block-quantized
//!   FP8 copy, and the cascade conv features. Sequences carve contiguous
//!   row ranges out of the arenas with a bump allocator — the same
//!   capacity contract as the CPU model's paged caches (one max-length
//!   sequence worth of pages in total).
//! * The decode path is the **uniform `_dev` kernel family**: every
//!   step-varying scalar (position, cache write rows, attention window
//!   starts) is read from a device parameter cell that one tiny kernel
//!   fans out from a single host-written `{pos, start}` pair. The eager
//!   decode launches exactly these kernels, so a captured
//!   [`CUDA/HIP graph`](deepinfer_gpu::GraphExec) replays bit-identically
//!   (see [`DeviceModel::enable_decode_graph`]).
//!
//! # Documented deviations from the CPU reference
//!
//! * The DSA indexer's k projection and conv features are cached
//!   incrementally (the CPU re-projects the whole history per call —
//!   identical per-row math, so results match within reduction tolerance).
//! * Decode MoE uses the **masked capacity layout** (DeepEP low-latency
//!   semantics) with capacity = 1 row per expert: a single token's top-k
//!   experts are distinct, so nothing is ever dropped; prefill MoE uses
//!   the aligned contiguous layout (DeepGEMM `m_grouped_contiguous`) with
//!   per-chunk host readback of the segment table, chunked by
//!   [`MOE_CHUNK_TOKENS`] to bound the permuted-row buffers.
//! * The causal clip of the DSA score matrix follows the reference
//!   crate's `j > i` convention (query = chunk row, key = absolute cached
//!   row) rather than upstream's `q_offset`-relative mask — mirroring the
//!   golden exactly; flagged for a coordinated CPU+GPU fix upstream of
//!   both.
//! * MTP (NEXTN) drafting stays on the host ([`DeviceModel::mtp`]):
//!   the draft module is one layer, and verification re-enters the device
//!   through the multi-token prefill path.

use std::collections::BTreeMap;
use std::path::Path;

use deepinfer_core::format::f32_to_bf16_rne;
use deepinfer_core::tokenizer::Tokenizer;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::{Backend, Buffer, Device, Gpu, Stream};
use deepinfer_kernels::{
    CacheFormat as DevCacheFormat, DecodeParams, GemmDType, Kernels, StepGeometry,
};
use deepinfer_mla::CacheFormat;
use deepinfer_mtp::MtpWeights;

use crate::config::ModelConfig;
use crate::loader;
use crate::weights::ModelWeights;

/// Maximum tokens one device forward call may process (the CPU reference
/// bound is 8192; the device keeps a tighter bound so the activation
/// staging buffers stay bounded — callers chunk prompts, which the
/// scheduler crate does for continuous batching anyway).
pub const DEVICE_MAX_PREFILL_TOKENS: usize = 2048;

/// MoE prefill chunk (tokens per dispatch/combine round). Bounding the
/// routed-row buffers matters because the aligned contiguous layout pads
/// every non-empty expert segment to 128 rows.
pub const MOE_CHUNK_TOKENS: usize = 512;

/// DeepGEMM contiguous-layout alignment (rows per expert segment).
const MOE_ALIGN: usize = 128;

/// Per-GEMM numeric mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// f32 operands everywhere, scalar-core accumulation (the golden-test
    /// mode; matches the CPU model within reduction-order tolerance).
    F32,
    /// FP8 (DeepGEMM recipe) whenever `k % 128 == 0`, BF16 otherwise.
    /// At V4.1-Flash dims (hidden 7162-multiples, q-LoRA 1536, head 512,
    /// indexer 128·heads, MoE intermediate 2048) every GEMM runs FP8.
    Auto,
}

/// One NT GEMM weight, device-resident in the precision-selected layout.
struct GemmW {
    dtype: GemmDType,
    /// `[n, k]` u8 FP8 payload (FP8 mode).
    q: Option<Buffer>,
    /// `[ceil(n/128), k/128]` u8 ue8m0 scales (FP8 mode).
    sf: Option<Buffer>,
    /// `[n, k]` f32 or bf16-bits payload (F32/BF16 modes).
    raw: Option<Buffer>,
    n: usize,
    k: usize,
}

impl GemmW {
    /// Rows (`n`) of the weight.
    fn n(&self) -> usize {
        self.n
    }
}

/// Per-GEMM dtype selection.
fn dtype_for(prec: Precision, k: usize) -> GemmDType {
    match prec {
        Precision::F32 => GemmDType::F32,
        Precision::Auto if k % 128 == 0 => GemmDType::Fp8,
        Precision::Auto => GemmDType::Bf16,
    }
}

/// Uploads one `[n, k]` row-major weight matrix in the selected precision.
fn upload_gemm(dev: &Device, w: &[f32], n: usize, k: usize, prec: Precision) -> Result<GemmW> {
    ensure!(
        w.len() == n * k,
        DimensionMismatch,
        "weight buffer has {} floats, expected {n}x{k}",
        w.len()
    );
    match dtype_for(prec, k) {
        GemmDType::Fp8 => {
            let (q, sf) = deepinfer_fp8::quantize_weight_per_block(w, n, k, 128, 128)?;
            Ok(GemmW {
                dtype: GemmDType::Fp8,
                q: Some(dev.alloc_from(&q)?),
                sf: Some(dev.alloc_from(&sf)?),
                raw: None,
                n,
                k,
            })
        }
        GemmDType::Bf16 => {
            let bits: Vec<u16> = w.iter().map(|&v| f32_to_bf16_rne(v)).collect();
            Ok(GemmW {
                dtype: GemmDType::Bf16,
                q: None,
                sf: None,
                raw: Some(dev.alloc_from(&bits)?),
                n,
                k,
            })
        }
        GemmDType::F32 => Ok(GemmW {
            dtype: GemmDType::F32,
            q: None,
            sf: None,
            raw: Some(dev.alloc_from(w)?),
            n,
            k,
        }),
    }
}

/// Uploads one `[g, n, k]` grouped weight (routed experts) — FP8 scales
/// are computed per group (per expert), matching the grouped GEMM's
/// `b_sf [g, ceil(n/128), k/128]` layout.
fn upload_grouped_gemm(
    dev: &Device,
    w: &[f32],
    g: usize,
    n: usize,
    k: usize,
    prec: Precision,
) -> Result<GemmW> {
    ensure!(
        w.len() == g * n * k,
        DimensionMismatch,
        "grouped weight has {} floats, expected {g}x{n}x{k}",
        w.len()
    );
    match dtype_for(prec, k) {
        GemmDType::Fp8 => {
            let nk = n * k;
            let mut q_all = Vec::with_capacity(g * nk);
            let mut sf_all = Vec::new();
            for e in 0..g {
                let (q, sf) =
                    deepinfer_fp8::quantize_weight_per_block(&w[e * nk..(e + 1) * nk], n, k, 128, 128)?;
                q_all.extend_from_slice(&q);
                sf_all.extend_from_slice(&sf);
            }
            Ok(GemmW {
                dtype: GemmDType::Fp8,
                q: Some(dev.alloc_from(&q_all)?),
                sf: Some(dev.alloc_from(&sf_all)?),
                raw: None,
                n,
                k,
            })
        }
        dtype => {
            let payload: Vec<u16> = if dtype == GemmDType::Bf16 {
                w.iter().map(|&v| f32_to_bf16_rne(v)).collect()
            } else {
                // f32: reinterpreted below without conversion
                Vec::new()
            };
            let raw = if dtype == GemmDType::Bf16 {
                dev.alloc_from(&payload)?
            } else {
                dev.alloc_from(w)?
            };
            Ok(GemmW {
                dtype,
                q: None,
                sf: None,
                raw: Some(raw),
                n,
                k,
            })
        }
    }
}

/// MoE weights of one layer (router + routed grouped + shared dense).
struct MoeW {
    router: GemmW,
    router_bias: Option<Buffer>,
    routed_gate_up: GemmW,
    routed_down: GemmW,
    shared: Vec<(GemmW, GemmW)>,
}

/// Self-decoder layer weights (MLA + DSA indexer + MoE).
struct SelfLayerW {
    fused_qkv: GemmW,
    q_b: GemmW,
    wv: Vec<GemmW>,
    wo: GemmW,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    kv_norm: Buffer,
    sink: Option<Buffer>,
    wq: GemmW,
    wk: GemmW,
    weights_proj: GemmW,
    gate_proj: Buffer,
    conv_w: Buffer,
    conv_off: Buffer,
    conv_gain: Buffer,
    conv_kernels: usize,
    moe: MoeW,
}

/// Cross-decoder layer weights (YOCO cross-attention + MoE).
struct CrossLayerW {
    q_proj: GemmW,
    o_proj: GemmW,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    moe: MoeW,
}

/// Root weights (embedding, final norm, LM head, RoPE table).
struct RootW {
    embed: Buffer,
    final_norm: Buffer,
    lm_head: GemmW,
    rope_cs: Buffer,
}

/// Per-self-layer append-only device caches (flat arenas; sequences carve
/// contiguous row ranges).
struct SelfLayerState {
    /// `[max_rows, kv_row_bytes]` — MLA KV rows in the config's cache
    /// format (f32 / 528 B FP8 / 288 B FP4).
    kv: Buffer,
    /// `[max_rows, idx_dim]` f32 — raw projected indexer k (conv branch).
    k_raw: Buffer,
    /// `[max_rows, idx_dim]` f32 — per-head-normed indexer k (thunder
    /// branch; the norm is applied at append time, so the buffer is a
    /// pure function of the appended rows).
    k_norm: Buffer,
    /// `[max_rows, idx_dim]` u8 — block-quantized copy of the normed k
    /// rows (written when a quant block completes).
    k_u8: Buffer,
    /// `[max_rows, idx_dim]` f32 — cascade conv features (normed in place
    /// at append time).
    feat: Buffer,
}

/// Per-sequence device state: the logical length, the arena start row and
/// the small per-sequence cells (lightning scale buffers — block indices
/// are sequence-local — and the per-cross-layer SWA rings).
struct DevSeq {
    len: usize,
    start: usize,
    /// `[max_rows / quant_block + 1]` u8 — lightning scales of this
    /// sequence's k blocks.
    k_sf: Buffer,
    /// Same shape, u32 — atomic amax scratch.
    k_sf_bits: Buffer,
    /// Per cross layer: `[swa_window, hidden]` f32 ring. Position `p`
    /// lives at slot `p % swa_window`; rollback only truncates the
    /// logical length, and surviving positions keep their slots, so the
    /// rings need no rebuild.
    rings: Vec<Buffer>,
}

/// Reusable activation staging + decode workspaces (fixed addresses for
/// the whole model lifetime — the captured graph bakes them).
struct Workspace {
    h: Buffer,
    x: Buffer,
    qa: Buffer,
    kv_rows: Buffer,
    q: Buffer,
    attn: Buffer,
    wv_lora: Buffer,
    proj: Buffer,
    q_idx: Buffer,
    q_idx_n: Buffer,
    q_idx_u8: Buffer,
    q_idx_sf: Buffer,
    k_new: Buffer,
    hw: Buffer,
    scores: Buffer,
    conv: Buffer,
    mean: Buffer,
    gt: Buffer,
    gc: Buffer,
    sel: Buffer,
    pos_q: Buffer,
    pos_kv: Buffer,
    logits: Buffer,
    // decode attention split-KV workspace
    pm: Buffer,
    pl: Buffer,
    po: Buffer,
    lse: Buffer,
    // decode DSA scratch
    cands_a: Buffer,
    cands_b: Buffer,
    // step parameter cells
    p_pos: Buffer,
    params: Buffer,
    // prefill window cells (uploaded per call)
    cell2: Buffer,
    // decode MoE (masked layout, fixed)
    moe_counters: Buffer,
    moe_drops: Buffer,
    moe_perm: Buffer,
    moe_map: Buffer,
    moe_aq: Option<Buffer>,
    moe_asf: Option<Buffer>,
    moe_act: Buffer,
    moe_silu: Buffer,
    moe_aq2: Option<Buffer>,
    moe_asf2: Option<Buffer>,
    moe_a16: Option<Buffer>,
    moe_down: Buffer,
    moe_y: Buffer,
    topk_w: Buffer,
    topk_idx: Buffer,
    // eager prefill MoE (aligned layout, fixed cells)
    moe_counts: Buffer,
    moe_offsets: Buffer,
    moe_rank: Buffer,
    moe_map_p: Buffer,
    router_logits: Buffer,
    // activation staging for gemms
    act_q: Option<Buffer>,
    act_sf: Option<Buffer>,
    act_bf16: Option<Buffer>,
    // lightning amax scratch shared by the prefill full-quant
    lightning_bits: Buffer,
    // cross-attention window staging + swa staging
    cross_stage: Buffer,
    swa_stage: Buffer,
    swa_out: Buffer,
    // fused q_a|kv_a output rows (prefill: [chunk, q_lora + hd]; decode
    // uses the first row's slices)
    fused: Buffer,
    /// Token id staging for the embedding gather ([chunk] u32).
    tokens: Buffer,
}

/// A captured decode graph bound to one sequence.
struct DecodeGraph {
    seq: u32,
    exec: deepinfer_gpu::GraphExec,
}

/// The GPU-resident DeepSeek-V4.1-Flash model.
pub struct DeviceModel {
    cfg: ModelConfig,
    dev: Device,
    stream: Stream,
    k: Kernels,
    prec: Precision,
    root: RootW,
    self_layers: Vec<(SelfLayerW, SelfLayerState)>,
    cross_layers: Vec<CrossLayerW>,
    mtp: Option<MtpWeights>,
    seqs: BTreeMap<u32, DevSeq>,
    max_rows: usize,
    kv_row_bytes: usize,
    fmt: DevCacheFormat,
    ws: Workspace,
    geo: StepGeometry,
    graph_enabled: bool,
    graph: Option<DecodeGraph>,
    vocab_bytes: Option<Vec<Vec<u8>>>,
}

/// Converts the config cache format to the device kernel format id.
fn dev_cache_format(f: CacheFormat) -> DevCacheFormat {
    match f {
        CacheFormat::Bf16 => DevCacheFormat::F32, // exact f32 rows on device
        CacheFormat::Fp8Group32 => DevCacheFormat::Fp8Group32,
        CacheFormat::Fp4Group16 => DevCacheFormat::Fp4Group16,
    }
}

/// Bytes per cached KV row for `fmt` and `hd` dims.
fn kv_row_bytes(fmt: DevCacheFormat, hd: usize) -> usize {
    match fmt {
        DevCacheFormat::F32 => hd * 4,
        DevCacheFormat::Fp8Group32 => hd + hd / 32,
        DevCacheFormat::Fp4Group16 => hd / 2 + hd / 16,
    }
}

/// Uploads one MoE layer's weights.
fn upload_moe(
    dev: &Device,
    w: &deepinfer_moe::MoeWeights,
    cfg: &deepinfer_moe::MoeConfig,
    prec: Precision,
) -> Result<MoeW> {
    let hidden = cfg.hidden;
    let inter = cfg.intermediate;
    let e = cfg.num_routed;
    let mut gu_all = Vec::with_capacity(e * 2 * inter * hidden);
    let mut down_all = Vec::with_capacity(e * hidden * inter);
    for ex in &w.routed {
        gu_all.extend_from_slice(&ex.w_gate_up);
        down_all.extend_from_slice(&ex.w_down);
    }
    Ok(MoeW {
        router: upload_gemm(dev, &w.gate, e, hidden, prec)?,
        router_bias: if cfg.use_noaux_tc_bias {
            Some(dev.alloc_from(&w.gate_bias)?)
        } else {
            None
        },
        routed_gate_up: upload_grouped_gemm(dev, &gu_all, e, 2 * inter, hidden, prec)?,
        routed_down: upload_grouped_gemm(dev, &down_all, e, hidden, inter, prec)?,
        shared: w
            .shared
            .iter()
            .map(|ex| {
                Ok((
                    upload_gemm(dev, &ex.w_gate_up, 2 * inter, hidden, prec)?,
                    upload_gemm(dev, &ex.w_down, hidden, inter, prec)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

impl DeviceModel {
    /// Builds the model from an already-loaded host weight set, uploading
    /// everything to `dev` in the selected precision.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on unsupported configurations (grouped
    /// MoE experts, dimensions the kernels cannot serve);
    /// [`Error::CapacityExceeded`] when the caches do not fit;
    /// JIT/driver errors from [`Kernels::new`].
    pub(crate) fn on_device(
        dev: &Device,
        cfg: &ModelConfig,
        mut weights: ModelWeights,
        prec: Precision,
    ) -> Result<DeviceModel> {
        cfg.validate()?;
        ensure!(
            cfg.moe.group_size == 1,
            InvalidArgument,
            "the GPU MoE path requires group_size == 1 (V4.1 default)"
        );
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let idx_dim = cfg.dsa.num_heads * cfg.dsa.head_dim;
        let max_rows = cfg.max_position_embeddings.max(1);
        ensure!(
            cfg.max_position_embeddings <= i32::MAX as usize,
            CapacityExceeded,
            "positions beyond i32 are not addressable by the device kernels"
        );
        let fmt = dev_cache_format(cfg.cache);
        let kv_bytes = kv_row_bytes(fmt, hd);
        ensure!(
            hd % 128 == 0 || fmt == DevCacheFormat::F32 || hd % 32 == 0,
            InvalidArgument,
            "head_dim {hd} cannot be served by the quantized cache formats"
        );
        if fmt != DevCacheFormat::F32 {
            ensure!(hd == 512, InvalidArgument, "FlashMLA V4.1 quantized cache rows require head_dim 512");
        }
        let stream = dev.stream()?;
        let k = Kernels::new(dev)?;
        let warp = k.warp_size() as usize;
        ensure!(
            hd % warp == 0,
            InvalidArgument,
            "head_dim {hd} must be a multiple of the warp size {warp}"
        );

        // ---- root weights --------------------------------------------------
        let root = RootW {
            embed: dev.alloc_from(&weights.embed)?,
            final_norm: dev.alloc_from(&weights.final_norm)?,
            lm_head: upload_gemm(dev, &weights.lm_head, cfg.vocab_size, hidden, prec)?,
            rope_cs: dev.alloc_from(weights.rope.cos_sin_table())?,
        };

        // ---- per-layer weights ---------------------------------------------
        let mut self_layers = Vec::with_capacity(cfg.n_self_layers);
        for l in 0..cfg.n_self_layers {
            let sw = &weights.self_layers[l];
            let mla = &sw.mla;
            let idx = &sw.indexer;
            // Fused [q_a ; kv_a] rows: [q_lora + hd, hidden].
            let mut fused = Vec::with_capacity((cfg.q_lora_rank + hd) * hidden);
            fused.extend_from_slice(&mla.q_a_proj);
            fused.extend_from_slice(&mla.kv_a_proj);
            let mut conv_flat = Vec::new();
            let mut conv_off = vec![0i32];
            for taps in &idx.conv_weights {
                conv_flat.extend_from_slice(taps);
                conv_off.push(conv_flat.len() as i32);
            }
            let conv_gain =
                deepinfer_dsa::weights::conv_norm_gain_from_wk(&idx.wk, idx_dim, hidden);
            let w = SelfLayerW {
                fused_qkv: upload_gemm(dev, &fused, cfg.q_lora_rank + hd, hidden, prec)?,
                q_b: upload_gemm(dev, &mla.q_b_proj, cfg.n_heads * hd, cfg.q_lora_rank, prec)?,
                wv: (0..cfg.o_groups)
                    .map(|g| {
                        let r = (cfg.n_heads / cfg.o_groups) * hd;
                        upload_gemm(
                            dev,
                            &mla.wv_proj[g * cfg.o_lora_rank * r..(g + 1) * cfg.o_lora_rank * r],
                            cfg.o_lora_rank,
                            r,
                            prec,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                wo: upload_gemm(
                    dev,
                    &mla.wo_proj,
                    hidden,
                    cfg.o_groups * cfg.o_lora_rank,
                    prec,
                )?,
                attn_norm: dev.alloc_from(&sw.attn_norm)?,
                ffn_norm: dev.alloc_from(&sw.ffn_norm)?,
                kv_norm: dev.alloc_from(&mla.kv_norm)?,
                sink: mla.attn_sink.as_ref().map(|s| dev.alloc_from(s.as_slice())).transpose()?,
                wq: upload_gemm(dev, &idx.wq, idx_dim, hidden, prec)?,
                wk: upload_gemm(dev, &idx.wk, idx_dim, hidden, prec)?,
                weights_proj: upload_gemm(dev, &idx.weights_proj, cfg.dsa.num_heads, idx_dim, prec)?,
                gate_proj: dev.alloc_from(&idx.gate_proj)?,
                conv_w: dev.alloc_from(&conv_flat)?,
                conv_off: dev.alloc_from(&conv_off)?,
                conv_gain: dev.alloc_from(&conv_gain)?,
                conv_kernels: idx.conv_weights.len(),
                moe: upload_moe(dev, &sw.moe, &cfg.moe, prec)?,
            };
            let state = SelfLayerState {
                kv: dev.alloc(max_rows.max(1) * kv_bytes)?,
                k_raw: dev.alloc(max_rows.max(1) * idx_dim * 4)?,
                k_norm: dev.alloc(max_rows.max(1) * idx_dim * 4)?,
                k_u8: dev.alloc(max_rows.max(1) * idx_dim)?,
                feat: dev.alloc(max_rows.max(1) * idx_dim * 4)?,
            };
            self_layers.push((w, state));
        }
        let mut cross_layers = Vec::with_capacity(cfg.n_cross_layers);
        for l in 0..cfg.n_cross_layers {
            let cw = &weights.cross_layers[l];
            cross_layers.push(CrossLayerW {
                q_proj: upload_gemm(dev, &cw.q_proj, cfg.n_heads * hd, hidden, prec)?,
                o_proj: upload_gemm(dev, &cw.o_proj, hidden, cfg.n_heads * hd, prec)?,
                attn_norm: dev.alloc_from(&cw.attn_norm)?,
                ffn_norm: dev.alloc_from(&cw.ffn_norm)?,
                moe: upload_moe(dev, &cw.moe, &cfg.moe, prec)?,
            });
        }

        // ---- workspace -----------------------------------------------------
        let chunk = DEVICE_MAX_PREFILL_TOKENS;
        let nblocks_sf = (chunk / cfg.dsa.quant_block).max(1);
        let splits = cfg
            .dsa
            .topk_tokens
            .div_ceil(deepinfer_kernels::DECODE_CHUNK)
            .max(1);
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        let inter = cfg.moe.intermediate;
        let cap = 1usize; // one decode token's experts are distinct
        let max_k_fp8 = [
            hidden,
            cfg.q_lora_rank,
            cfg.n_heads * hd,
            cfg.o_groups * cfg.o_lora_rank,
            inter,
            idx_dim,
            cfg.vocab_size.min(1), // lm_head k == hidden already listed
        ]
        .into_iter()
        .filter(|&v| v % 128 == 0)
        .max()
        .unwrap_or(0);
        let max_k_bf16 = [hidden, cfg.q_lora_rank, cfg.n_heads * hd, cfg.o_lora_rank, inter, idx_dim]
            .into_iter()
            .filter(|&v| v % 128 != 0)
            .max()
            .unwrap_or(0);
        let ws = Workspace {
            h: dev.alloc(chunk * hidden * 4)?,
            x: dev.alloc(chunk * hidden * 4)?,
            qa: dev.alloc(chunk * cfg.q_lora_rank * 4)?,
            kv_rows: dev.alloc(chunk * hd * 4)?,
            q: dev.alloc(chunk * cfg.n_heads * hd * 4)?,
            attn: dev.alloc(chunk * cfg.n_heads * hd * 4)?,
            wv_lora: dev.alloc(chunk * cfg.o_groups * cfg.o_lora_rank * 4)?,
            proj: dev.alloc(chunk * hidden * 4)?,
            q_idx: dev.alloc(chunk * idx_dim * 4)?,
            q_idx_n: dev.alloc(chunk * idx_dim * 4)?,
            q_idx_u8: dev.alloc(chunk * idx_dim)?,
            q_idx_sf: dev.alloc(nblocks_sf)?,
            k_new: dev.alloc(chunk * idx_dim * 4)?,
            hw: dev.alloc(chunk * cfg.dsa.num_heads * 4)?,
            scores: dev.alloc(max_rows * 4)?, // decode: [1, max_rows]
            conv: dev.alloc(max_rows * 4)?,
            mean: dev.alloc(hidden * 4)?,
            gt: dev.alloc(chunk * 4)?,
            gc: dev.alloc(chunk * 4)?,
            sel: dev.alloc(chunk * cfg.dsa.topk_tokens.max(1) * 4)?,
            pos_q: dev.alloc(chunk * cfg.n_heads * 4)?,
            pos_kv: dev.alloc(chunk * 4)?,
            logits: dev.alloc(cfg.vocab_size * 4)?,
            pm: dev.alloc(cfg.n_heads * splits * 4)?,
            pl: dev.alloc(cfg.n_heads * splits * 4)?,
            po: dev.alloc(cfg.n_heads * splits * hd * 4)?,
            lse: dev.alloc(cfg.n_heads * 4)?,
            cands_a: {
                let g = StepGeometry::new(
                    max_rows,
                    cfg.dsa.topk_tokens,
                    cfg.dsa.block_size,
                    cfg.dsa.preselect_per_block,
                )?;
                dev.alloc(g.cands_cap * 8)?
            },
            cands_b: {
                let g = StepGeometry::new(
                    max_rows,
                    cfg.dsa.topk_tokens,
                    cfg.dsa.block_size,
                    cfg.dsa.preselect_per_block,
                )?;
                dev.alloc(g.cands_cap * 8)?
            },
            p_pos: dev.alloc(8)?,
            params: dev.alloc(DecodeParams::COUNT * 4)?,
            cell2: dev.alloc(8)?,
            moe_counters: dev.alloc(e * 4)?,
            moe_drops: dev.alloc(4)?,
            moe_perm: dev.alloc(e * cap * hidden * 4)?,
            moe_map: dev.alloc(ki * 4)?,
            moe_aq: (hidden % 128 == 0).then(|| dev.alloc(e * cap * hidden)).transpose()?,
            moe_asf: (hidden % 128 == 0).then(|| dev.alloc(e * cap * (hidden / 128))).transpose()?,
            moe_act: dev.alloc(e * cap * 2 * inter * 4)?,
            moe_silu: dev.alloc(e * cap * inter * 4)?,
            moe_aq2: (inter % 128 == 0).then(|| dev.alloc(e * cap * inter)).transpose()?,
            moe_asf2: (inter % 128 == 0).then(|| dev.alloc(e * cap * (inter / 128))).transpose()?,
            moe_a16: (hidden % 128 != 0 && inter % 128 != 0)
                .then(|| dev.alloc(e * cap * hidden.max(inter) * 2))
                .transpose()?,
            moe_down: dev.alloc(e * cap * hidden * 4)?,
            moe_y: dev.alloc(hidden * 4)?,
            topk_w: dev.alloc(chunk * ki * 4)?,
            topk_idx: dev.alloc(chunk * ki * 4)?,
            moe_counts: dev.alloc(e * 4)?,
            moe_offsets: dev.alloc((e + 1) * 4)?,
            moe_rank: dev.alloc(e * 4)?,
            moe_map_p: dev.alloc(chunk * ki * 4)?,
            router_logits: dev.alloc(chunk * e * 4)?,
            act_q: (prec == Precision::Auto && max_k_fp8 > 0)
                .then(|| dev.alloc(chunk * max_k_fp8))
                .transpose()?,
            act_sf: (prec == Precision::Auto && max_k_fp8 > 0)
                .then(|| dev.alloc(chunk * (max_k_fp8 / 128)))
                .transpose()?,
            act_bf16: (prec == Precision::Auto && max_k_bf16 > 0)
                .then(|| dev.alloc(chunk * max_k_bf16 * 2))
                .transpose()?,
            lightning_bits: dev.alloc((max_rows / cfg.dsa.quant_block + 1).max(1) * 4)?,
            cross_stage: dev.alloc(cfg.cross_attn_window.max(1) * kv_bytes)?,
            swa_stage: dev.alloc(cfg.swa_window.max(1) * hidden * 4)?,
            swa_out: dev.alloc(chunk * hidden * 4)?,
            fused: dev.alloc(chunk * (cfg.q_lora_rank + hd) * 4)?,
            tokens: dev.alloc(chunk * 4)?,
        };
        let geo = StepGeometry::new(
            max_rows,
            cfg.dsa.topk_tokens,
            cfg.dsa.block_size,
            cfg.dsa.preselect_per_block,
        )?;
        let mtp = weights.mtp.take();
        Ok(DeviceModel {
            cfg: cfg.clone(),
            dev: dev.clone(),
            stream,
            k,
            prec,
            root,
            self_layers,
            cross_layers,
            mtp,
            seqs: BTreeMap::new(),
            max_rows,
            kv_row_bytes: kv_bytes,
            fmt,
            ws,
            geo,
            graph_enabled: false,
            graph: None,
            vocab_bytes: None,
        })
    }

    /// Loads a checkpoint onto `dev` (safetensors shards, unchanged
    /// loader).
    ///
    /// # Errors
    /// As [`DeviceModel::on_device`] plus [`loader::load_weights`].
    pub fn load_checkpoint(
        dir: &Path,
        cfg: &ModelConfig,
        dev: &Device,
        prec: Precision,
    ) -> Result<DeviceModel> {
        let cp = deepinfer_core::safetensors::CheckpointDir::open(dir)?;
        let weights = loader::load_weights(&cp, cfg)?;
        DeviceModel::on_device(dev, cfg, weights, prec)
    }

    /// Deterministic random initialization (self-test path; see
    /// [`crate::weights::random_init`]).
    ///
    /// # Errors
    /// As [`DeviceModel::on_device`] plus the random-init bounds.
    pub fn random_init(
        cfg: &ModelConfig,
        seed: u64,
        dev: &Device,
        prec: Precision,
    ) -> Result<DeviceModel> {
        let weights = crate::weights::random_init(cfg, seed)?;
        DeviceModel::on_device(dev, cfg, weights, prec)
    }

    /// Opens the first available device (CUDA preferred, then HIP).
    ///
    /// # Errors
    /// [`Error::Internal`] when no GPU stack is usable.
    pub fn probe_device() -> Result<Device> {
        for b in [Backend::Cuda, Backend::Hip] {
            if let Ok(gpu) = Gpu::new(b) {
                if let Ok(n) = gpu.device_count() {
                    if n > 0 {
                        return gpu.device(0);
                    }
                }
            }
        }
        Err(Error::internal(
            "no usable GPU (tried CUDA then HIP); the DeviceModel needs a device",
        ))
    }

    /// The model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// The device this model runs on.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The execution stream (work is enqueued here; callers may fork).
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// The kernel library (for callers layering extra ops, e.g. the EP
    /// peer collectives).
    pub fn kernels(&self) -> &Kernels {
        &self.k
    }

    /// The MTP (NEXTN) draft weights (host-side drafting; see the module
    /// docs).
    pub fn mtp(&self) -> Option<&MtpWeights> {
        self.mtp.as_ref()
    }

    /// Whether MTP draft weights are present.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Tokens cached for `seq` (0 for unknown sequences).
    pub fn seq_len(&self, seq: u32) -> usize {
        self.seqs.get(&seq).map(|s| s.len).unwrap_or(0)
    }

    /// Whether decode-graph capture is enabled.
    pub fn decode_graph_enabled(&self) -> bool {
        self.graph_enabled
    }

    /// The active per-GEMM precision mode.
    pub fn precision(&self) -> Precision {
        self.prec
    }

    /// Enables or disables CUDA/HIP-graph decode replay. When enabled and
    /// supported by the driver, the first decode step of each sequence is
    /// captured once and replayed — bit-identical to the eager launches,
    /// with one launch instead of hundreds.
    ///
    /// # Errors
    /// [`Error::Internal`] when enabling on a driver without graph
    /// support.
    pub fn enable_decode_graph(&mut self, on: bool) -> Result<()> {
        if on && !deepinfer_gpu::graph_supported(&self.dev) {
            return Err(Error::internal(
                "decode graphs requested but the driver does not expose CUDA/HIP graph capture",
            ));
        }
        self.graph_enabled = on;
        if !on {
            self.graph = None;
        }
        Ok(())
    }
}

impl DeviceModel {
    // ------------------------------------------------------------------
    // GEMM dispatch (activation prep + precision-selected kernel)
    // ------------------------------------------------------------------

    /// `d = a @ w^T` with `a` f32 `[m, w.k]` and the weight's uploaded
    /// precision. FP8 mode stages the activation through the 1x128
    /// quantizer; BF16 mode rounds to bf16 first.
    fn gemm(&self, w: &GemmW, a_f32: &Buffer, d: &Buffer, m: usize) -> Result<()> {
        ensure!(m >= 1, InvalidArgument, "gemm needs m >= 1");
        let (n, kk) = (w.n(), w.k);
        match w.dtype {
            GemmDType::Fp8 => {
                let aq = self
                    .ws
                    .act_q
                    .as_ref()
                    .ok_or_else(|| Error::internal("FP8 activation staging missing"))?;
                let asf = self
                    .ws
                    .act_sf
                    .as_ref()
                    .ok_or_else(|| Error::internal("FP8 scale staging missing"))?;
                self.k
                    .quant_per_token_group(a_f32, aq, asf, m, kk, 128, &self.stream)?;
                self.k.gemm_fp8_nt(
                    aq,
                    asf,
                    w.q.as_ref().ok_or_else(|| Error::internal("fp8 payload missing"))?,
                    w.sf.as_ref().ok_or_else(|| Error::internal("fp8 scales missing"))?,
                    d,
                    m,
                    n,
                    kk,
                    &self.stream,
                )
            }
            GemmDType::Bf16 => {
                let a16 = self
                    .ws
                    .act_bf16
                    .as_ref()
                    .ok_or_else(|| Error::internal("BF16 activation staging missing"))?;
                self.k.f32_to_bf16(a_f32, a16, m * kk, &self.stream)?;
                self.k.gemm_bf16_nt(
                    a16,
                    w.raw.as_ref().ok_or_else(|| Error::internal("weight payload missing"))?,
                    None,
                    d,
                    m,
                    n,
                    kk,
                    &self.stream,
                )
            }
            GemmDType::F32 => self.k.gemm_f32_nt(
                a_f32,
                w.raw.as_ref().ok_or_else(|| Error::internal("weight payload missing"))?,
                None,
                d,
                m,
                n,
                kk,
                &self.stream,
            ),
        }
    }

    // ------------------------------------------------------------------
    // Sequence state helpers
    // ------------------------------------------------------------------

    /// Creates (or fetches) the per-sequence device state.
    ///
    /// Capacity contract: the flat arenas hold one max-length sequence —
    /// exactly the total page budget of the CPU model's paged caches — so
    /// the device model serves **one sequence at a time** (call
    /// [`DeviceModel::reset`] between sequences; a paged device cache is
    /// the documented multi-sequence upgrade). The sequence's rows start
    /// at arena row 0.
    fn seq_state(&mut self, seq: u32) -> Result<&mut DevSeq> {
        if !self.seqs.contains_key(&seq) {
            ensure!(
                self.seqs.is_empty(),
                CapacityExceeded,
                "the device model serves one sequence at a time ({} active); call reset() first",
                self.seqs.len()
            );
            let cfg = &self.cfg;
            let nblocks = self.max_rows / cfg.dsa.quant_block + 1;
            let k_sf = self.dev.alloc(nblocks.max(1))?;
            let k_sf_bits = self.dev.alloc(nblocks.max(1) * 4)?;
            let rings = (0..cfg.n_cross_layers)
                .map(|_| self.dev.alloc(cfg.swa_window.max(1) * cfg.hidden_size * 4))
                .collect::<Result<Vec<_>>>()?;
            self.seqs.insert(
                seq,
                DevSeq {
                    len: 0,
                    start: 0,
                    k_sf,
                    k_sf_bits,
                    rings,
                },
            );
        }
        Ok(self.seqs.get_mut(&seq).expect("just inserted"))
    }

    /// A view of the `field`-th i32 of the decode parameter cell.
    fn pcell(&self, field: usize) -> Result<deepinfer_gpu::Buffer> {
        self.ws.params.slice(field * 4, 4)
    }

    // ------------------------------------------------------------------
    // Prefill (eager, baked-scalar kernels)
    // ------------------------------------------------------------------

    /// Shared forward over `s >= 1` new tokens — the prefill path
    /// (multi-token, eager kernels; the graph path is decode-only).
    fn prefill_forward(&mut self, seq: u32, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = self.cfg.clone();
        let s = tokens.len();
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let latent = hd - cfg.rope_dim;
        let half = cfg.rope_dim / 2;
        let nh = cfg.dsa.num_heads;
        let ihd = cfg.dsa.head_dim;
        let idx_dim = nh * ihd;
        let qb = cfg.dsa.quant_block;
        let window = cfg.dsa.extrapolation_window.unwrap_or(0);
        ensure!(
            (1..=DEVICE_MAX_PREFILL_TOKENS).contains(&s),
            CapacityExceeded,
            "device forward chunk of {s} tokens outside [1, {DEVICE_MAX_PREFILL_TOKENS}]"
        );
        for &t in tokens {
            ensure!(
                (t as usize) < cfg.vocab_size,
                OutOfRange,
                "token {t} outside the vocabulary ({})",
                cfg.vocab_size
            );
        }
        self.seq_state(seq)?;
        let (pos0, start) = {
            let st = &self.seqs[&seq];
            (st.len, st.start)
        };
        let total = pos0 + s;
        ensure!(
            start + total <= self.max_rows,
            CapacityExceeded,
            "position {} would exceed the cache capacity ({})",
            start + total,
            self.max_rows
        );
        ensure!(
            total <= cfg.max_position_embeddings,
            CapacityExceeded,
            "position {total} exceeds max_position_embeddings ({})",
            cfg.max_position_embeddings
        );

        // Embedding lookup: token ids -> device gather from the table.
        self.stream
            .write_async(&self.dev, &self.ws.tokens, tokens)?;
        self.k.gather_indexed(
            &self.root.embed,
            &self.ws.h,
            &self.ws.tokens,
            s,
            hidden * 4,
            &self.stream,
        )?;

        // RoPE position rows.
        let pos_kv: Vec<u32> = (pos0..pos0 + s).map(|p| p as u32).collect();
        let pos_q: Vec<u32> = (0..s * cfg.n_heads)
            .map(|r| (pos0 + r / cfg.n_heads) as u32)
            .collect();
        self.stream.write_async(&self.dev, &self.ws.pos_kv, &pos_kv)?;
        self.stream.write_async(&self.dev, &self.ws.pos_q, &pos_q)?;

        let sm_eff = cfg.sm_scale() * deepinfer_dsa::mscale(cfg.dsa.topk_tokens);

        // Self-decoder layers.
        for l in 0..cfg.n_self_layers {
            let (w, state) = &self.self_layers[l];
            self.k
                .rmsnorm(&self.ws.h, &w.attn_norm, &self.ws.x, s, hidden, cfg.rms_eps, &self.stream)?;
            // Fused q_a|kv_a projection, then split the strided columns
            // (the kv latent norm needs contiguous rows; with one decode
            // row the slices are contiguous and no split is needed).
            self.gemm(&w.fused_qkv, &self.ws.x, &self.ws.fused, s)?;
            self.k.copy_cols(
                &self.ws.fused,
                &self.ws.qa,
                s,
                cfg.q_lora_rank + hd,
                cfg.q_lora_rank,
                0,
                0,
                cfg.q_lora_rank,
                &self.stream,
            )?;
            self.k.copy_cols(
                &self.ws.fused,
                &self.ws.kv_rows,
                s,
                cfg.q_lora_rank + hd,
                hd,
                cfg.q_lora_rank,
                0,
                hd,
                &self.stream,
            )?;
            self.k.rmsnorm_latent(
                &self.ws.kv_rows,
                &w.kv_norm,
                s,
                hd,
                latent,
                crate::weights::V41_RMS_EPS,
                &self.stream,
            )?;
            self.k.rope(
                &self.root.rope_cs,
                &self.ws.pos_kv,
                &self.ws.kv_rows,
                s,
                hd,
                latent,
                half,
                false,
                &self.stream,
            )?;
            self.gemm(&w.q_b, &self.ws.qa, &self.ws.q, s)?;
            self.k.rope(
                &self.root.rope_cs,
                &self.ws.pos_q,
                &self.ws.q,
                s * cfg.n_heads,
                hd,
                latent,
                half,
                false,
                &self.stream,
            )?;
            // KV cache append.
            match self.fmt {
                DevCacheFormat::F32 => self.k.append_rows(
                    &self.ws.kv_rows,
                    &state.kv,
                    start + pos0,
                    s,
                    hd,
                    &self.stream,
                )?,
                DevCacheFormat::Fp8Group32 => {
                    let dst = state.kv.slice((start + pos0) * self.kv_row_bytes, s * self.kv_row_bytes)?;
                    self.k.kv_encode_fp8(&self.ws.kv_rows, &dst, s, hd, &self.stream)?;
                }
                DevCacheFormat::Fp4Group16 => {
                    let dst = state.kv.slice((start + pos0) * self.kv_row_bytes, s * self.kv_row_bytes)?;
                    self.k.kv_encode_fp4(&self.ws.kv_rows, &dst, s, hd, &self.stream)?;
                }
            }
            // DSA indexer: projections from the RAW residual rows.
            self.gemm(&w.wq, &self.ws.h, &self.ws.q_idx, s)?;
            self.gemm(&w.wk, &self.ws.h, &self.ws.k_new, s)?;
            self.k
                .append_rows(&self.ws.k_new, &state.k_raw, start + pos0, s, idx_dim, &self.stream)?;
            self.k.append_norm_rows(
                &self.ws.k_new,
                &state.k_norm,
                start + pos0,
                s,
                nh,
                ihd,
                &self.stream,
            )?;
            self.k.conv_feat(
                &state.k_raw,
                &state.feat,
                &w.conv_w,
                &w.conv_off,
                w.conv_kernels,
                start + pos0,
                start + total,
                total,
                idx_dim,
                &self.stream,
            )?;
            {
                let feat_slice = state
                    .feat
                    .slice((start + pos0) * idx_dim * 4, s * idx_dim * 4)?;
                self.k.conv_gated_norm(&feat_slice, &w.conv_gain, s, idx_dim, &self.stream)?;
            }
            // Per-head normed q copy + lightning quant (whole history, the
            // CPU's per-call semantics).
            {
                let bytes = s * idx_dim * 4;
                let dst = self.ws.q_idx_n.slice(0, bytes)?;
                let src = self.ws.q_idx.slice(0, bytes)?;
                self.dev.memcpy_dtod_async(&dst, &src, &self.stream)?;
            }
            self.k
                .indexer_rmsnorm_per_head(&self.ws.q_idx_n, s, nh, ihd, &self.stream)?;
            {
                let k_norm_slice = state.k_norm.slice(start * idx_dim * 4, total * idx_dim * 4)?;
                let k_u8_slice = state.k_u8.slice(start * idx_dim, total * idx_dim)?;
                let k_sf_view = self.seqs[&seq].k_sf.slice(0, (total / qb).max(1))?;
                self.k.lightning_quant(
                    &k_norm_slice,
                    &k_u8_slice,
                    &k_sf_view,
                    &self.ws.lightning_bits,
                    total,
                    idx_dim,
                    qb,
                    &self.stream,
                )?;
                self.k.lightning_quant(
                    &self.ws.q_idx_n,
                    &self.ws.q_idx_u8,
                    &self.ws.q_idx_sf,
                    &self.ws.lightning_bits,
                    s,
                    idx_dim,
                    qb,
                    &self.stream,
                )?;
                // Scores.
                let scores = self
                    .dev
                    .alloc(s * total * 4)
                    .map_err(|e| Error::CapacityExceeded(format!("score matrix: {e}")))?;
                let conv = self
                    .dev
                    .alloc(s * total * 4)
                    .map_err(|e| Error::CapacityExceeded(format!("conv matrix: {e}")))?;
                self.gemm(&w.weights_proj, &self.ws.q_idx, &self.ws.hw, s)?;
                self.k.thunder_logits(
                    &self.ws.q_idx_n,
                    &self.ws.q_idx_u8,
                    &self.ws.q_idx_sf,
                    &k_norm_slice,
                    &k_u8_slice,
                    &k_sf_view,
                    &self.ws.hw,
                    &scores,
                    s,
                    total,
                    nh,
                    ihd,
                    s / qb,
                    total / qb,
                    qb,
                    &self.stream,
                )?;
                self.k.dot_scores(
                    &self.ws.q_idx,
                    &state.feat.slice(start * idx_dim * 4, total * idx_dim * 4)?,
                    &conv,
                    s,
                    total,
                    idx_dim,
                    &self.stream,
                )?;
                self.k
                    .qmean(&self.ws.h, &self.ws.mean, s, hidden, &self.stream)?;
                self.k.gate_logits(
                    &self.ws.h,
                    &self.ws.mean,
                    &w.gate_proj,
                    &self.ws.gt,
                    &self.ws.gc,
                    s,
                    hidden,
                    &self.stream,
                )?;
                self.k
                    .gate_blend(&self.ws.gt, &self.ws.gc, &conv, &scores, s, total, &self.stream)?;
                self.k
                    .causal_window_clip(&scores, s, total, window, &self.stream)?;
                let k_eff = cfg.dsa.topk_tokens.min(total);
                let cands_cap =
                    total.div_ceil(cfg.dsa.block_size) * cfg.dsa.preselect_per_block;
                let cands_a = self.dev.alloc(s * cands_cap * 8)?;
                let cands_b = self.dev.alloc(s * cands_cap * 8)?;
                let sel = self.ws.sel.slice(0, s * k_eff * 4)?;
                self.k.select_top_k(
                    &scores,
                    &cands_a,
                    &cands_b,
                    cands_cap,
                    &sel,
                    s,
                    total,
                    cfg.dsa.block_size,
                    cfg.dsa.preselect_per_block,
                    k_eff,
                    &self.stream,
                )?;
                // Sparse attention over the sequence's cache rows.
                let kv_seq = state.kv.slice(start * self.kv_row_bytes, total * self.kv_row_bytes)?;
                self.k.mla_prefill(
                    &self.ws.q,
                    &kv_seq,
                    Some(&sel),
                    None,
                    &self.ws.attn,
                    None,
                    None,
                    s,
                    cfg.n_heads,
                    hd,
                    total,
                    0,
                    k_eff,
                    sm_eff,
                    self.fmt,
                    &self.stream,
                )?;
            }
            // O-RoPE conjugate + output projections + residual.
            self.k.rope(
                &self.root.rope_cs,
                &self.ws.pos_q,
                &self.ws.attn,
                s * cfg.n_heads,
                hd,
                latent,
                half,
                true,
                &self.stream,
            )?;
            let r = (cfg.n_heads / cfg.o_groups) * hd;
            for g in 0..cfg.o_groups {
                if s == 1 {
                    let a = self.ws.attn.slice(g * r * 4, r * 4)?;
                    let d = self.ws.wv_lora.slice(g * cfg.o_lora_rank * 4, cfg.o_lora_rank * 4)?;
                    self.gemm(&w.wv[g], &a, &d, 1)?;
                } else {
                    let tmp = self.ws.swa_out.slice(0, s * r * 4)?; // reuse as [s, r]
                    self.k.copy_cols(
                        &self.ws.attn,
                        &tmp,
                        s,
                        cfg.n_heads * hd,
                        r,
                        g * r,
                        0,
                        r,
                        &self.stream,
                    )?;
                    let tmp2 = self.ws.qa.slice(0, s * cfg.o_lora_rank * 4)?; // reuse as [s, o_lora]
                    self.gemm(&w.wv[g], &tmp, &tmp2, s)?;
                    self.k.copy_cols(
                        &tmp2,
                        &self.ws.wv_lora,
                        s,
                        cfg.o_lora_rank,
                        cfg.o_groups * cfg.o_lora_rank,
                        0,
                        g * cfg.o_lora_rank,
                        cfg.o_lora_rank,
                        &self.stream,
                    )?;
                }
            }
            self.gemm(&w.wo, &self.ws.wv_lora, &self.ws.proj, s)?;
            self.k.add(&self.ws.proj, &self.ws.h, s * hidden, &self.stream)?;
            // FFN.
            self.k
                .rmsnorm(&self.ws.h, &w.ffn_norm, &self.ws.x, s, hidden, cfg.rms_eps, &self.stream)?;
            self.moe_prefill(&w.moe, s)?;
        }

        // Cross-decoder layers (attend the LAST self layer's cache).
        if cfg.n_cross_layers > 0 {
            let last = cfg.n_self_layers - 1;
            let n = total.min(cfg.swa_window);
            let ring_start = (total - n) % cfg.swa_window.max(1);
            let swa_qoff = pos0 - (total - n);
            let cell: Vec<i32> = vec![ring_start as i32, n as i32, swa_qoff as i32, n as i32];
            self.stream.write_async(&self.dev, &self.ws.cell2, &cell)?;
            let p_start = self.ws.cell2.slice(0, 4)?;
            let p_n = self.ws.cell2.slice(4, 4)?;
            let p_qoff = self.ws.cell2.slice(8, 4)?;
            let nkeys = self.ws.cell2.slice(12, 4)?;
            let lo = total - total.min(cfg.cross_attn_window);
            let nkeys_c = total - lo;
            let qoff_c = pos0 - lo;
            for l in 0..cfg.n_cross_layers {
                let w = &self.cross_layers[l];
                self.k.rmsnorm(
                    &self.ws.h,
                    &w.attn_norm,
                    &self.ws.x,
                    s,
                    hidden,
                    cfg.rms_eps,
                    &self.stream,
                )?;
                // Ring append (wrapped host-split writes) + gather + SWA.
                let slot0 = pos0 % cfg.swa_window;
                if slot0 + s <= cfg.swa_window {
                    let dst = self.seqs[&seq].rings[l]
                        .slice(slot0 * hidden * 4, s * hidden * 4)?;
                    self.k
                        .append_rows(&self.ws.x, &dst, 0, s, hidden, &self.stream)?;
                } else {
                    let head = cfg.swa_window - slot0;
                    let d1 = self.seqs[&seq].rings[l].slice(slot0 * hidden * 4, head * hidden * 4)?;
                    self.k.append_rows(&self.ws.x, &d1, 0, head, hidden, &self.stream)?;
                    let d2 = self.seqs[&seq].rings[l].slice(0, (s - head) * hidden * 4)?;
                    let src2 = self.ws.x.slice(head * hidden * 4, (s - head) * hidden * 4)?;
                    self.k.append_rows(&src2, &d2, 0, s - head, hidden, &self.stream)?;
                }
                self.k.gather_rows(
                    &self.seqs[&seq].rings[l],
                    &self.ws.swa_stage,
                    &p_start,
                    &p_n,
                    cfg.swa_window,
                    hidden * 4,
                    cfg.swa_window,
                    &self.stream,
                )?;
                self.k.swa_attention_dev(
                    &self.ws.x,
                    &self.ws.swa_stage,
                    &p_qoff,
                    &nkeys,
                    &self.ws.swa_out,
                    s,
                    hidden,
                    1.0 / (hidden as f32).sqrt(),
                    &self.stream,
                )?;
                // Cross-attention into the last self layer's KV window.
                self.gemm(&w.q_proj, &self.ws.x, &self.ws.q, s)?;
                self.k.rope(
                    &self.root.rope_cs,
                    &self.ws.pos_q,
                    &self.ws.q,
                    s * cfg.n_heads,
                    hd,
                    latent,
                    half,
                    false,
                    &self.stream,
                )?;
                let kv_win = self.self_layers[last].1.kv.slice(
                    (start + lo) * self.kv_row_bytes,
                    nkeys_c * self.kv_row_bytes,
                )?;
                self.k.mla_prefill(
                    &self.ws.q,
                    &kv_win,
                    None,
                    None,
                    &self.ws.attn,
                    None,
                    None,
                    s,
                    cfg.n_heads,
                    hd,
                    nkeys_c,
                    qoff_c,
                    0,
                    cfg.sm_scale(),
                    self.fmt,
                    &self.stream,
                )?;
                self.k.rope(
                    &self.root.rope_cs,
                    &self.ws.pos_q,
                    &self.ws.attn,
                    s * cfg.n_heads,
                    hd,
                    latent,
                    half,
                    true,
                    &self.stream,
                )?;
                self.gemm(&w.o_proj, &self.ws.attn, &self.ws.proj, s)?;
                self.k.add(&self.ws.proj, &self.ws.h, s * hidden, &self.stream)?;
                self.k.add(&self.ws.swa_out, &self.ws.h, s * hidden, &self.stream)?;
                self.k.rmsnorm(
                    &self.ws.h,
                    &w.ffn_norm,
                    &self.ws.x,
                    s,
                    hidden,
                    cfg.rms_eps,
                    &self.stream,
                )?;
                self.moe_prefill(&w.moe, s)?;
            }
        }

        // Final norm + LM head over the last position.
        {
            let last_row = self.ws.h.slice((s - 1) * hidden * 4, hidden * 4)?;
            self.k.rmsnorm(
                &last_row,
                &self.root.final_norm,
                &self.ws.x,
                1,
                hidden,
                cfg.rms_eps,
                &self.stream,
            )?;
            let x1 = self.ws.x.slice(0, hidden * 4)?;
            self.gemm(&self.root.lm_head, &x1, &self.ws.logits, 1)?;
        }
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.dev.read(&self.ws.logits, &mut logits)?;
        let mut residual = vec![0.0f32; hidden];
        let last_row = self.ws.h.slice((s - 1) * hidden * 4, hidden * 4)?;
        self.dev.read(&last_row, &mut residual)?;
        self.seqs.get_mut(&seq).expect("seq exists").len = total;
        Ok((logits, residual))
    }

}

impl DeviceModel {
    // ------------------------------------------------------------------
    // MoE — prefill (aligned contiguous) and decode (masked capacity)
    // ------------------------------------------------------------------

    /// Prefill MoE over `s` tokens (input in `ws.x`), adding into `ws.h`.
    /// Routed experts run the aligned-contiguous layout in
    /// [`MOE_CHUNK_TOKENS`] chunks; shared experts run dense.
    fn moe_prefill(&self, m: &MoeW, s: usize) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.moe.hidden;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        for c0 in (0..s).step_by(MOE_CHUNK_TOKENS.max(1)) {
            let cs = MOE_CHUNK_TOKENS.min(s - c0);
            let x = self.ws.x.slice(c0 * hidden * 4, cs * hidden * 4)?;
            let topk_w = self.ws.topk_w.slice(c0 * ki * 4, cs * ki * 4)?;
            let topk_idx = self.ws.topk_idx.slice(c0 * ki * 4, cs * ki * 4)?;
            let map = self.ws.moe_map_p.slice(c0 * ki * 4, cs * ki * 4)?;
            let rlogits = self.ws.router_logits.slice(0, cs * e * 4)?;
            // Router.
            self.gemm(&m.router, &x, &rlogits, cs)?;
            self.k.moe_route(
                &rlogits,
                m.router_bias.as_ref(),
                &topk_w,
                &topk_idx,
                cs,
                e,
                ki,
                &self.stream,
            )?;
            // Aligned dispatch (reads back the segment table).
            let bound = cs
                .checked_mul(ki)
                .and_then(|v| v.checked_add(127 * v.min(e)))
                .ok_or_else(|| Error::CapacityExceeded("moe row bound overflows".into()))?
                .next_multiple_of(MOE_ALIGN);
            let perm = self.dev.alloc(bound.max(1) * hidden * 4)?;
            let m_total = self.k.moe_dispatch(
                &x,
                &topk_idx,
                &self.ws.moe_counts,
                &self.ws.moe_offsets,
                &self.ws.moe_rank,
                &perm,
                &map,
                cs,
                hidden,
                ki,
                e,
                MOE_ALIGN,
                &self.stream,
            )?;
            self.stream.synchronize()?;
            let mut counts = vec![0i32; e];
            self.dev.read(&self.ws.moe_counts, &mut counts)?;
            let act = self.dev.alloc(m_total.max(1) * 2 * inter * 4)?;
            let silu = self.dev.alloc(m_total.max(1) * inter * 4)?;
            let down = self.dev.alloc(m_total.max(1) * hidden * 4)?;
            match m.routed_gate_up.dtype {
                GemmDType::Fp8 => {
                    let aq = self.dev.alloc(m_total.max(1) * hidden)?;
                    let asf = self.dev.alloc(m_total.max(1) * (hidden / 128))?;
                    self.k.quant_per_token_group(
                        &perm,
                        &aq,
                        &asf,
                        m_total,
                        hidden,
                        128,
                        &self.stream,
                    )?;
                    self.k.gemm_fp8_grouped_nt(
                        &aq,
                        &asf,
                        m.routed_gate_up.q.as_ref().unwrap(),
                        m.routed_gate_up.sf.as_ref().unwrap(),
                        &self.ws.moe_offsets,
                        &act,
                        m_total,
                        2 * inter,
                        hidden,
                        e,
                        &self.stream,
                    )?;
                }
                dtype => {
                    // Per-expert dense GEMMs over the actual (unaligned)
                    // row counts; outputs land in the aligned layout.
                    let mut off = 0usize;
                    for ex in 0..e {
                        let cnt = counts[ex].max(0) as usize;
                        if cnt > 0 {
                            let a = perm.slice(off * hidden * 4, cnt * hidden * 4)?;
                            let d = act.slice(off * 2 * inter * 4, cnt * 2 * inter * 4)?;
                            let (gw, _dtype) = (&m.routed_gate_up, dtype);
                            match gw.dtype {
                                GemmDType::Fp8 => unreachable!(),
                                GemmDType::Bf16 => {
                                    let a16 = self.dev.alloc(cnt * hidden * 2)?;
                                    self.k.f32_to_bf16(&a, &a16, cnt * hidden, &self.stream)?;
                                    self.k.gemm_bf16_nt(
                                        &a16,
                                        gw.raw.as_ref().unwrap(),
                                        None,
                                        &d,
                                        cnt,
                                        2 * inter,
                                        hidden,
                                        &self.stream,
                                    )?;
                                }
                                GemmDType::F32 => {
                                    self.k.gemm_f32_nt(
                                        &a,
                                        gw.raw.as_ref().unwrap(),
                                        None,
                                        &d,
                                        cnt,
                                        2 * inter,
                                        hidden,
                                        &self.stream,
                                    )?;
                                }
                            }
                        }
                        off += cnt.next_multiple_of(MOE_ALIGN);
                    }
                }
            }
            self.k
                .silu_mul_fused(&act, &silu, m_total, inter, &self.stream)?;
            match m.routed_down.dtype {
                GemmDType::Fp8 => {
                    let aq = self.dev.alloc(m_total.max(1) * inter)?;
                    let asf = self.dev.alloc(m_total.max(1) * (inter / 128))?;
                    self.k.quant_per_token_group(
                        &silu,
                        &aq,
                        &asf,
                        m_total,
                        inter,
                        128,
                        &self.stream,
                    )?;
                    self.k.gemm_fp8_grouped_nt(
                        &aq,
                        &asf,
                        m.routed_down.q.as_ref().unwrap(),
                        m.routed_down.sf.as_ref().unwrap(),
                        &self.ws.moe_offsets,
                        &down,
                        m_total,
                        hidden,
                        inter,
                        e,
                        &self.stream,
                    )?;
                }
                dtype => {
                    let mut off = 0usize;
                    for ex in 0..e {
                        let cnt = counts[ex].max(0) as usize;
                        if cnt > 0 {
                            let a = silu.slice(off * inter * 4, cnt * inter * 4)?;
                            let d = down.slice(off * hidden * 4, cnt * hidden * 4)?;
                            let dw = &m.routed_down;
                            match dtype {
                                GemmDType::Fp8 => unreachable!(),
                                GemmDType::Bf16 => {
                                    let a16 = self.dev.alloc(cnt * inter * 2)?;
                                    self.k.f32_to_bf16(&a, &a16, cnt * inter, &self.stream)?;
                                    self.k.gemm_bf16_nt(
                                        &a16,
                                        dw.raw.as_ref().unwrap(),
                                        None,
                                        &d,
                                        cnt,
                                        hidden,
                                        inter,
                                        &self.stream,
                                    )?;
                                }
                                GemmDType::F32 => {
                                    self.k.gemm_f32_nt(
                                        &a,
                                        dw.raw.as_ref().unwrap(),
                                        None,
                                        &d,
                                        cnt,
                                        hidden,
                                        inter,
                                        &self.stream,
                                    )?;
                                }
                            }
                        }
                        off += cnt.next_multiple_of(MOE_ALIGN);
                    }
                }
            }
            let y = self.ws.h.slice(c0 * hidden * 4, cs * hidden * 4)?;
            // combine writes y; then the residual add happens per chunk.
            let y_buf = self.ws.swa_out.slice(0, cs * hidden * 4)?;
            self.k.moe_combine(
                &down,
                &map,
                &topk_w,
                &y_buf,
                cs,
                hidden,
                ki,
                &self.stream,
            )?;
            self.k.add(&y_buf, &y, cs * hidden, &self.stream)?;
            // Shared experts (unweighted sum, dense; transient staging —
            // the masked decode buffers are capacity-sized, not chunk-sized).
            for (gu, dn) in &m.shared {
                let su = self.dev.alloc(cs * 2 * inter * 4)?;
                let so = self.dev.alloc(cs * inter * 4)?;
                let sd = self.dev.alloc(cs * hidden * 4)?;
                self.gemm(gu, &x, &su, cs)?;
                self.k.silu_mul_fused(&su, &so, cs, inter, &self.stream)?;
                self.gemm(dn, &so, &sd, cs)?;
                self.k.add(&sd, &y, cs * hidden, &self.stream)?;
            }
        }
        Ok(())
    }

    /// Decode MoE over one token (input `ws.x`, output `ws.moe_y`) — the
    /// uniform masked-capacity path shared by eager and captured graphs.
    /// Capacity 1 per expert never drops (a token's top-k experts are
    /// distinct).
    fn moe_decode(&self, m: &MoeW) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.moe.hidden;
        let inter = cfg.moe.intermediate;
        let e = cfg.moe.num_routed;
        let ki = cfg.moe.num_activated;
        let cap = 1usize;
        let rows = e * cap;
        self.gemm(&m.router, &self.ws.x, &self.ws.router_logits, 1)?;
        self.k.moe_route(
            &self.ws.router_logits,
            m.router_bias.as_ref(),
            &self.ws.topk_w,
            &self.ws.topk_idx,
            1,
            e,
            ki,
            &self.stream,
        )?;
        self.k.moe_dispatch_masked(
            &self.ws.x,
            &self.ws.topk_idx,
            &self.ws.moe_counters,
            &self.ws.moe_perm,
            &self.ws.moe_map,
            &self.ws.moe_drops,
            1,
            hidden,
            ki,
            e,
            cap,
            &self.stream,
        )?;
        // gate_up
        match m.routed_gate_up.dtype {
            GemmDType::Fp8 => {
                let aq = self.ws.moe_aq.as_ref().unwrap();
                let asf = self.ws.moe_asf.as_ref().unwrap();
                self.k
                    .quant_per_token_group(&self.ws.moe_perm, aq, asf, rows, hidden, 128, &self.stream)?;
                self.k.gemm_grouped_masked_nt(
                    aq,
                    Some(asf),
                    m.routed_gate_up.q.as_ref().unwrap(),
                    m.routed_gate_up.sf.as_ref(),
                    &self.ws.moe_counters,
                    &self.ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Fp8,
                    &self.stream,
                )?;
            }
            GemmDType::Bf16 => {
                let a16 = self.ws.moe_a16.as_ref().unwrap();
                self.k
                    .f32_to_bf16(&self.ws.moe_perm, a16, rows * hidden, &self.stream)?;
                self.k.gemm_grouped_masked_nt(
                    a16,
                    None,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &self.ws.moe_counters,
                    &self.ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::Bf16,
                    &self.stream,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_masked_nt(
                    &self.ws.moe_perm,
                    None,
                    m.routed_gate_up.raw.as_ref().unwrap(),
                    None,
                    &self.ws.moe_counters,
                    &self.ws.moe_act,
                    cap,
                    2 * inter,
                    hidden,
                    e,
                    GemmDType::F32,
                    &self.stream,
                )?;
            }
        }
        self.k.silu_mul_masked(
            &self.ws.moe_act,
            &self.ws.moe_silu,
            &self.ws.moe_counters,
            cap,
            e,
            inter,
            &self.stream,
        )?;
        // down
        match m.routed_down.dtype {
            GemmDType::Fp8 => {
                let aq = self.ws.moe_aq2.as_ref().unwrap();
                let asf = self.ws.moe_asf2.as_ref().unwrap();
                self.k.quant_per_token_group(
                    &self.ws.moe_silu,
                    aq,
                    asf,
                    rows,
                    inter,
                    128,
                    &self.stream,
                )?;
                self.k.gemm_grouped_masked_nt(
                    aq,
                    Some(asf),
                    m.routed_down.q.as_ref().unwrap(),
                    m.routed_down.sf.as_ref(),
                    &self.ws.moe_counters,
                    &self.ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::Fp8,
                    &self.stream,
                )?;
            }
            GemmDType::Bf16 => {
                let a16_full = self
                    .ws
                    .moe_a16
                    .as_ref()
                    .ok_or_else(|| Error::internal("BF16 staging missing for masked down"))?;
                let a16 = a16_full.slice(0, rows * inter * 2)?;
                self.k
                    .f32_to_bf16(&self.ws.moe_silu, &a16, rows * inter, &self.stream)?;
                self.k.gemm_grouped_masked_nt(
                    &a16,
                    None,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &self.ws.moe_counters,
                    &self.ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::Bf16,
                    &self.stream,
                )?;
            }
            GemmDType::F32 => {
                self.k.gemm_grouped_masked_nt(
                    &self.ws.moe_silu,
                    None,
                    m.routed_down.raw.as_ref().unwrap(),
                    None,
                    &self.ws.moe_counters,
                    &self.ws.moe_down,
                    cap,
                    hidden,
                    inter,
                    e,
                    GemmDType::F32,
                    &self.stream,
                )?;
            }
        }
        self.k.moe_combine(
            &self.ws.moe_down,
            &self.ws.moe_map,
            &self.ws.topk_w,
            &self.ws.moe_y,
            1,
            hidden,
            ki,
            &self.stream,
        )?;
        // Shared experts (dense, m=1).
        for (gu, dn) in &m.shared {
            let su = self.ws.moe_act.slice(0, 2 * inter * 4)?;
            let so = self.ws.moe_silu.slice(0, inter * 4)?;
            let sd = self.ws.moe_down.slice(0, hidden * 4)?;
            self.gemm(gu, &self.ws.x, &su, 1)?;
            self.k.silu_mul_fused(&su, &so, 1, inter, &self.stream)?;
            self.gemm(dn, &so, &sd, 1)?;
            self.k.add(&sd, &self.ws.moe_y, hidden, &self.stream)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Decode — the uniform _dev kernel path (eager == captured graph)
    // ------------------------------------------------------------------

    /// The decode step's kernel sequence — **only kernel launches**, no
    /// host transfers or synchronization, so the identical function serves
    /// eager execution and CUDA/HIP-graph capture. All varying scalars
    /// come from `ws.params` (written by `di_step_params_kernel` from the
    /// host-uploaded `{pos, start}` pair).
    fn decode_step_kernels(&self, seq: u32) -> Result<()> {
        let cfg = &self.cfg;
        let s = &self.stream;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let latent = hd - cfg.rope_dim;
        let half = cfg.rope_dim / 2;
        let nh = cfg.dsa.num_heads;
        let ihd = cfg.dsa.head_dim;
        let idx_dim = nh * ihd;
        let qb = cfg.dsa.quant_block;
        let window = cfg.dsa.extrapolation_window.unwrap_or(0);
        let topk = self.geo.topk;
        let sm_eff = cfg.sm_scale() * deepinfer_dsa::mscale(cfg.dsa.topk_tokens);
        let splits = topk
            .div_ceil(deepinfer_kernels::DECODE_CHUNK)
            .max(1);
        let st_seq = self.seqs.get(&seq).ok_or_else(|| Error::internal("seq missing"))?;
        let start = st_seq.start;

        self.k.step_params(
            &self.ws.p_pos,
            &self.ws.params,
            cfg.cross_attn_window,
            cfg.swa_window,
            qb,
            topk,
            s,
        )?;
        self.k
            .pos_fill(&self.ws.pos_q, &self.ws.pos_kv, &self.ws.p_pos, cfg.n_heads, s)?;

        for l in 0..cfg.n_self_layers {
            let (w, state) = &self.self_layers[l];
            self.k
                .rmsnorm(&self.ws.h, &w.attn_norm, &self.ws.x, 1, hidden, cfg.rms_eps, s)?;
            self.gemm(&w.fused_qkv, &self.ws.x, &self.ws.fused, 1)?;
            let kv_slice = self.ws.fused.slice(cfg.q_lora_rank * 4, hd * 4)?;
            self.k
                .rmsnorm_latent(&kv_slice, &w.kv_norm, 1, hd, latent, crate::weights::V41_RMS_EPS, s)?;
            let qa_slice = self.ws.fused.slice(0, cfg.q_lora_rank * 4)?;
            self.gemm(&w.q_b, &qa_slice, &self.ws.q, 1)?;
            self.k
                .rope(&self.root.rope_cs, &self.ws.pos_q, &self.ws.q, cfg.n_heads, hd, latent, half, false, s)?;
            self.k
                .rope(&self.root.rope_cs, &self.ws.pos_kv, &kv_slice, 1, hd, latent, half, false, s)?;
            match self.fmt {
                DevCacheFormat::Fp8Group32 => {
                    self.k.kv_encode_fp8_dev(&kv_slice, &state.kv, &self.ws.params, hd, s)?
                }
                DevCacheFormat::Fp4Group16 => {
                    self.k.kv_encode_fp4_dev(&kv_slice, &state.kv, &self.ws.params, hd, s)?
                }
                DevCacheFormat::F32 => {
                    self.k.append_row(&kv_slice, &state.kv, &self.ws.params, hd, s)?
                }
            }
            // DSA indexer (projections consume the RAW residual row).
            self.gemm(&w.wq, &self.ws.h, &self.ws.q_idx, 1)?;
            self.gemm(&w.wk, &self.ws.h, &self.ws.k_new, 1)?;
            self.k.append_row(&self.ws.k_new, &state.k_raw, &self.ws.params, idx_dim, s)?;
            self.k
                .append_norm_row(&self.ws.k_new, &state.k_norm, &self.ws.params, nh, ihd, s)?;
            self.k.conv_feat_dev(
                &state.k_raw,
                &state.feat,
                &w.conv_w,
                &w.conv_off,
                w.conv_kernels,
                &self.ws.params,
                idx_dim,
                s,
            )?;
            self.k
                .conv_gated_norm_dev(&state.feat, &w.conv_gain, &self.ws.params, idx_dim, s)?;
            {
                let k_norm_seq = state.k_norm.slice(start * idx_dim * 4, (self.max_rows - start) * idx_dim * 4)?;
                let k_u8_seq = state.k_u8.slice(start * idx_dim, (self.max_rows - start) * idx_dim)?;
                self.k.lightning_block_quant_dev(
                    &k_norm_seq,
                    &k_u8_seq,
                    &st_seq.k_sf,
                    &st_seq.k_sf_bits,
                    &self.ws.params,
                    idx_dim,
                    qb,
                    s,
                )?;
            }
            self.gemm(&w.weights_proj, &self.ws.q_idx, &self.ws.hw, 1)?;
            {
                let dst = self.ws.q_idx_n.slice(0, idx_dim * 4)?;
                let src = self.ws.q_idx.slice(0, idx_dim * 4)?;
                self.dev.memcpy_dtod_async(&dst, &src, s)?;
            }
            self.k
                .indexer_rmsnorm_per_head(&self.ws.q_idx_n, 1, nh, ihd, s)?;
            self.k.lightning_quant(
                &self.ws.q_idx_n,
                &self.ws.q_idx_u8,
                &self.ws.q_idx_sf,
                &self.ws.lightning_bits,
                1,
                idx_dim,
                qb,
                s,
            )?;
            {
                let k_norm_seq = state.k_norm.slice(start * idx_dim * 4, (self.max_rows - start) * idx_dim * 4)?;
                let k_u8_seq = state.k_u8.slice(start * idx_dim, (self.max_rows - start) * idx_dim)?;
                self.k.thunder_logits_dev(
                    &self.ws.q_idx_n,
                    &self.ws.q_idx_u8,
                    &self.ws.q_idx_sf,
                    &k_norm_seq,
                    &k_u8_seq,
                    &st_seq.k_sf,
                    &self.ws.hw,
                    &self.ws.scores,
                    &self.ws.params,
                    1,
                    nh,
                    ihd,
                    1 / qb,
                    qb,
                    self.max_rows,
                    s,
                )?;
                let feat_seq = state.feat.slice(start * idx_dim * 4, (self.max_rows - start) * idx_dim * 4)?;
                self.k.dot_scores_dev(
                    &self.ws.q_idx,
                    &feat_seq,
                    &self.ws.conv,
                    &self.ws.params,
                    1,
                    idx_dim,
                    self.max_rows,
                    s,
                )?;
            }
            self.k.qmean(&self.ws.h, &self.ws.mean, 1, hidden, s)?;
            self.k.gate_logits(
                &self.ws.h,
                &self.ws.mean,
                &w.gate_proj,
                &self.ws.gt,
                &self.ws.gc,
                1,
                hidden,
                s,
            )?;
            self.k.gate_blend_dev(
                &self.ws.gt,
                &self.ws.gc,
                &self.ws.conv,
                &self.ws.scores,
                &self.ws.params,
                1,
                self.max_rows,
                s,
            )?;
            self.k.causal_window_clip_dev(
                &self.ws.scores,
                &self.ws.params,
                1,
                self.max_rows,
                window,
                s,
            )?;
            self.k.select_top_k_dev(
                &self.ws.scores,
                &self.ws.cands_a,
                &self.ws.cands_b,
                &self.ws.sel,
                &self.ws.params,
                1,
                &self.geo,
                window,
                s,
            )?;
            {
                let kv_seq = state.kv.slice(start * self.kv_row_bytes, (self.max_rows - start) * self.kv_row_bytes)?;
                if self.fmt == DevCacheFormat::Fp4Group16 {
                    self.k.mla_decode_fp4_partial(
                        &self.ws.q,
                        &kv_seq,
                        &self.ws.sel,
                        &self.ws.pm,
                        &self.ws.pl,
                        &self.ws.po,
                        1,
                        1,
                        cfg.n_heads,
                        hd,
                        topk,
                        sm_eff,
                        self.max_rows,
                        s,
                    )?;
                } else {
                    self.k.mla_decode_partial(
                        &self.ws.q,
                        &kv_seq,
                        &self.ws.sel,
                        &self.ws.pm,
                        &self.ws.pl,
                        &self.ws.po,
                        1,
                        1,
                        cfg.n_heads,
                        hd,
                        topk,
                        sm_eff,
                        self.fmt,
                        self.max_rows,
                        s,
                    )?;
                }
                self.k.mla_decode_combine(
                    &self.ws.pm,
                    &self.ws.pl,
                    &self.ws.po,
                    w.sink.as_ref(),
                    &self.ws.attn,
                    &self.ws.lse,
                    1,
                    1,
                    cfg.n_heads,
                    hd,
                    splits,
                    s,
                )?;
            }
            self.k.rope(
                &self.root.rope_cs,
                &self.ws.pos_q,
                &self.ws.attn,
                cfg.n_heads,
                hd,
                latent,
                half,
                true,
                s,
            )?;
            let r = (cfg.n_heads / cfg.o_groups) * hd;
            for g in 0..cfg.o_groups {
                let a = self.ws.attn.slice(g * r * 4, r * 4)?;
                let d = self
                    .ws
                    .wv_lora
                    .slice(g * cfg.o_lora_rank * 4, cfg.o_lora_rank * 4)?;
                self.gemm(&w.wv[g], &a, &d, 1)?;
            }
            self.gemm(&w.wo, &self.ws.wv_lora, &self.ws.proj, 1)?;
            self.k.add(&self.ws.proj, &self.ws.h, hidden, s)?;
            self.k
                .rmsnorm(&self.ws.h, &w.ffn_norm, &self.ws.x, 1, hidden, cfg.rms_eps, s)?;
            self.moe_decode(&w.moe)?;
            self.k.add(&self.ws.moe_y, &self.ws.h, hidden, s)?;
        }

        // Cross-decoder layers.
        if cfg.n_cross_layers > 0 {
            let last = cfg.n_self_layers - 1;
            let p_swa_slot = self.pcell(DecodeParams::SWA_START)?;
            let p_swa_n = self.pcell(DecodeParams::SWA_N)?;
            let p_swa_qoff = self.pcell(DecodeParams::SWA_QOFF)?;
            let p_cross_lo = self.pcell(DecodeParams::CROSS_LO)?;
            let p_cross_n = self.pcell(DecodeParams::CROSS_NKEYS)?;
            let p_cross_qoff = self.pcell(DecodeParams::CROSS_QOFF)?;
            for l in 0..cfg.n_cross_layers {
                let w = &self.cross_layers[l];
                self.k
                    .rmsnorm(&self.ws.h, &w.attn_norm, &self.ws.x, 1, hidden, cfg.rms_eps, s)?;
                // Ring append at the device-read slot, then gather + SWA.
                let ring = &self.seqs[&seq].rings[l];
                self.k.append_row(&self.ws.x, ring, &p_swa_slot, hidden, s)?;
                self.k.gather_rows(
                    ring,
                    &self.ws.swa_stage,
                    &p_swa_slot,
                    &p_swa_n,
                    cfg.swa_window,
                    hidden * 4,
                    cfg.swa_window,
                    s,
                )?;
                self.k.swa_attention_dev(
                    &self.ws.x,
                    &self.ws.swa_stage,
                    &p_swa_qoff,
                    &p_swa_n,
                    &self.ws.swa_out,
                    1,
                    hidden,
                    1.0 / (hidden as f32).sqrt(),
                    s,
                )?;
                self.gemm(&w.q_proj, &self.ws.x, &self.ws.q, 1)?;
                self.k.rope(
                    &self.root.rope_cs,
                    &self.ws.pos_q,
                    &self.ws.q,
                    cfg.n_heads,
                    hd,
                    latent,
                    half,
                    false,
                    s,
                )?;
                // Cross window: gather into the fixed staging buffer.
                let kv_arena = &self.self_layers[last].1.kv;
                self.k.gather_rows(
                    kv_arena,
                    &self.ws.cross_stage,
                    &p_cross_lo,
                    &p_cross_n,
                    self.max_rows,
                    self.kv_row_bytes,
                    cfg.cross_attn_window,
                    s,
                )?;
                self.k.mla_prefill_dense_dev(
                    &self.ws.q,
                    &self.ws.cross_stage,
                    &p_cross_qoff,
                    &p_cross_n,
                    &self.ws.attn,
                    None,
                    1,
                    cfg.n_heads,
                    hd,
                    self.fmt,
                    s,
                )?;
                self.k.rope(
                    &self.root.rope_cs,
                    &self.ws.pos_q,
                    &self.ws.attn,
                    cfg.n_heads,
                    hd,
                    latent,
                    half,
                    true,
                    s,
                )?;
                self.gemm(&w.o_proj, &self.ws.attn, &self.ws.proj, 1)?;
                self.k.add(&self.ws.proj, &self.ws.h, hidden, s)?;
                self.k.add(&self.ws.swa_out, &self.ws.h, hidden, s)?;
                self.k
                    .rmsnorm(&self.ws.h, &w.ffn_norm, &self.ws.x, 1, hidden, cfg.rms_eps, s)?;
                self.moe_decode(&w.moe)?;
                self.k.add(&self.ws.moe_y, &self.ws.h, hidden, s)?;
            }
        }

        // Final norm + LM head.
        self.k.rmsnorm(
            &self.ws.h,
            &self.root.final_norm,
            &self.ws.x,
            1,
            hidden,
            cfg.rms_eps,
            s,
        )?;
        let x1 = self.ws.x.slice(0, hidden * 4)?;
        self.gemm(&self.root.lm_head, &x1, &self.ws.logits, 1)?;
        Ok(())
    }

    /// Host-side decode inputs: the embedded row and the `{pos, start}`
    /// parameter pair. Everything else is computed on the device.
    fn upload_decode_inputs(&self, token: u32, pos: usize, start: usize) -> Result<()> {
        let cfg = &self.cfg;
        ensure!((token as usize) < cfg.vocab_size, OutOfRange, "token {token} outside the vocabulary");
        self.stream
            .write_async(&self.dev, &self.ws.tokens, &[token])?;
        self.k.gather_indexed(
            &self.root.embed,
            &self.ws.h,
            &self.ws.tokens,
            1,
            cfg.hidden_size * 4,
            &self.stream,
        )?;
        let cell: [i32; 2] = [pos as i32, start as i32];
        self.stream.write_async(&self.dev, &self.ws.p_pos, &cell)?;
        Ok(())
    }

    /// Reads the logits + residual after a decode step and advances the
    /// sequence length.
    fn finish_decode(&mut self, seq: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.cfg;
        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.dev.read(&self.ws.logits, &mut logits)?;
        let mut residual = vec![0.0f32; cfg.hidden_size];
        let row = self.ws.h.slice(0, cfg.hidden_size * 4)?;
        self.dev.read(&row, &mut residual)?;
        self.seqs.get_mut(&seq).expect("seq exists").len += 1;
        Ok((logits, residual))
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Prefills `tokens` for `seq` (appending to any existing state) and
    /// returns the next-token logits `[vocab]` for the final position.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on an empty prompt;
    /// [`Error::OutOfRange`] on out-of-vocabulary tokens;
    /// [`Error::CapacityExceeded`] over the position / chunk bounds.
    pub fn prefill(&mut self, seq: u32, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty(),
            InvalidArgument,
            "prefill requires at least one prompt token"
        );
        let (logits, _) = self.prefill_forward(seq, tokens)?;
        Ok(logits)
    }

    /// Decodes one `token` for `seq` (appends to the caches) and returns
    /// the next-token logits `[vocab]`. When decode graphs are enabled
    /// (see [`DeviceModel::enable_decode_graph`]) the per-sequence step is
    /// captured once and replayed.
    ///
    /// # Errors
    /// [`Error::OutOfRange`] on out-of-vocabulary tokens;
    /// [`Error::CapacityExceeded`] over the position bound.
    pub fn decode_step(&mut self, seq: u32, token: u32) -> Result<Vec<f32>> {
        ensure!(
            (token as usize) < self.cfg.vocab_size,
            OutOfRange,
            "token {token} outside the vocabulary ({})",
            self.cfg.vocab_size
        );
        {
            self.seq_state(seq)?;
            let (len, start) = {
                let st = &self.seqs[&seq];
                (st.len, st.start)
            };
            ensure!(
                start + len < self.max_rows,
                CapacityExceeded,
                "cache capacity exceeded ({})",
                self.max_rows
            );
            ensure!(
                len < self.cfg.max_position_embeddings,
                CapacityExceeded,
                "position {} exceeds max_position_embeddings ({})",
                len + 1,
                self.cfg.max_position_embeddings
            );
        }
        let (pos, start) = {
            let st = &self.seqs[&seq];
            (st.len, st.start)
        };
        self.upload_decode_inputs(token, pos, start)?;
        if self.graph_enabled {
            if let Some(g) = &self.graph {
                if g.seq == seq {
                    g.exec.launch(&self.stream)?;
                    return self.finish_decode(seq).map(|(l, _)| l);
                }
            }
            // Capture this sequence's step once; replay forever after.
            let guard = self.dev.begin_capture(&self.stream)?;
            self.decode_step_kernels(seq)?;
            let exec = guard.end()?;
            exec.launch(&self.stream)?;
            self.graph = Some(DecodeGraph { seq, exec });
        } else {
            self.decode_step_kernels(seq)?;
        }
        self.finish_decode(seq).map(|(l, _)| l)
    }

    /// Rolls back the newest `n` tokens of `seq` (MTP rejection). Device
    /// caches are append-only with position-keyed rows, so truncating the
    /// logical length is exact: surviving rows keep their slots, stale
    /// rows are unreachable, and re-appended rows overwrite them. A
    /// captured decode graph stays valid (it reads the position from the
    /// device cell every replay).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when `n` exceeds the stored length.
    pub fn rollback_seq(&mut self, seq: u32, n: usize) -> Result<()> {
        let st = self
            .seqs
            .get_mut(&seq)
            .ok_or_else(|| Error::invalid("rollback on unknown sequence {seq}"))?;
        ensure!(
            n <= st.len,
            InvalidArgument,
            "cannot roll back {n} tokens: sequence {seq} only holds {}",
            st.len
        );
        st.len -= n;
        Ok(())
    }

    /// Clears all sequence state (arena bump pointer, rings, graphs).
    /// Weights stay resident.
    pub fn reset(&mut self) {
        self.seqs.clear();
        self.graph = None;
    }

    /// Total cached tokens (all sequences; every layer stays in lockstep).
    pub fn total_cached_tokens(&self) -> usize {
        self.seqs.values().map(|s| s.len).sum()
    }

    /// Attaches a vocabulary byte table for structured-output masking
    /// (same contract as the CPU model; consumed by host-side samplers).
    pub fn set_vocab_bytes(&mut self, vocab_bytes: Vec<Vec<u8>>) {
        self.vocab_bytes = Some(vocab_bytes);
    }

    /// The attached vocabulary byte table, if any.
    pub fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        self.vocab_bytes.as_deref()
    }

    /// Derives a vocabulary byte table from a tokenizer (CPU-model
    /// semantics; see `DeepSeekModel::vocab_bytes_from_tokenizer`).
    ///
    /// # Errors
    /// [`Error::OutOfRange`] on a bad `eos_id`.
    pub fn vocab_bytes_from_tokenizer(
        tok: &dyn Tokenizer,
        eos_id: u32,
    ) -> Result<Vec<Vec<u8>>> {
        let vocab = tok.vocab_size();
        ensure!(
            vocab >= 1 && (eos_id as usize) < vocab,
            OutOfRange,
            "eos_id {eos_id} outside vocab [0, {vocab})"
        );
        let mut out = Vec::with_capacity(vocab);
        if vocab == 256 {
            for t in 0..256u32 {
                out.push(vec![t as u8]);
            }
        } else {
            for t in 0..vocab as u32 {
                let s = tok.decode(&[t]).map_err(|e| {
                    Error::Checkpoint(format!("decoding token {t} for vocab bytes: {e}"))
                })?;
                out.push(s.into_bytes());
            }
        }
        out[eos_id as usize] = Vec::new();
        Ok(out)
    }

    /// Greedy / sampled generation loop (host-side sampling via
    /// `deepinfer-sampler` — batch-invariant by construction). Returns the
    /// generated token ids (prompt excluded). EOS terminates when the
    /// sampler picks it.
    ///
    /// # Errors
    /// Propagates [`DeviceModel::prefill`] / [`DeviceModel::decode_step`]
    /// and sampler validation.
    pub fn generate(
        &mut self,
        seq: u32,
        prompt: &[u32],
        max_tokens: usize,
        params: &deepinfer_sampler::SamplingParams,
        seed: Option<u64>,
        eos: Option<u32>,
    ) -> Result<Vec<u32>> {
        params.validate(self.cfg.vocab_size)?;
        ensure!(
            !prompt.is_empty(),
            InvalidArgument,
            "generate requires a non-empty prompt"
        );
        ensure!(max_tokens >= 1, InvalidArgument, "max_tokens must be >= 1");
        let sampler = deepinfer_sampler::Sampler::new(params.clone());
        let mut rng = deepinfer_sampler::Sampler::seq_rng(seed, seq as u64);
        let mut out = Vec::new();
        let mut logits = self.prefill(seq, prompt)?;
        for _ in 0..max_tokens {
            let tok = sampler.sample(&logits, None, &mut rng)?;
            if tok as usize >= self.cfg.vocab_size {
                return Err(Error::internal("sampler produced an out-of-vocabulary token"));
            }
            out.push(tok);
            if Some(tok) == eos {
                break;
            }
            logits = self.decode_step(seq, tok)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny config whose dimensions are all 128-multiples (exercises the
    /// FP8 GEMM path end-to-end) and whose quant blocks are small (exercises
    /// the lightning quant both sides).
    fn fp8_tiny(cache: CacheFormat) -> ModelConfig {
        ModelConfig {
            hidden_size: 128,
            n_layers: 2,
            n_self_layers: 1,
            n_cross_layers: 1,
            n_heads: 4,
            head_dim: 512,
            q_lora_rank: 128,
            rope_dim: 64,
            rope_base: 10_000.0,
            rms_eps: 1e-4,
            vocab_size: 256,
            max_position_embeddings: 256,
            o_groups: 1,
            o_lora_rank: 128,
            swa_window: 16,
            cross_attn_window: 32,
            tie_word_embeddings: false,
            cache,
            dsa: deepinfer_dsa::LightningIndexerConfig {
                hidden_size: 128,
                num_heads: 4,
                head_dim: 32,
                block_size: 4,
                topk_tokens: 8,
                preselect_per_block: 2,
                quant_block: 8,
                conv_kernels: vec![3],
                extrapolation_window: None,
            },
            moe: deepinfer_moe::MoeConfig {
                hidden: 128,
                intermediate: 128,
                num_routed: 8,
                num_activated: 2,
                num_shared: 1,
                group_size: 1,
                use_noaux_tc_bias: true,
            },
            mtp: deepinfer_mtp::MtpConfig {
                hidden: 128,
                vocab: 256,
                draft_len: 2,
            },
        }
    }

    fn gpu_device() -> Option<Device> {
        for b in [Backend::Cuda, Backend::Hip] {
            if let Ok(gpu) = Gpu::new(b) {
                if let Ok(n) = gpu.device_count() {
                    if n > 0 {
                        if let Ok(dev) = gpu.device(0) {
                            return Some(dev);
                        }
                    }
                }
            }
        }
        eprintln!("note: no GPU present; DeviceModel integration tests skipped");
        None
    }

    #[test]
    fn cache_format_mapping() {
        assert_eq!(dev_cache_format(CacheFormat::Bf16), DevCacheFormat::F32);
        assert_eq!(
            dev_cache_format(CacheFormat::Fp8Group32),
            DevCacheFormat::Fp8Group32
        );
        assert_eq!(
            dev_cache_format(CacheFormat::Fp4Group16),
            DevCacheFormat::Fp4Group16
        );
        assert_eq!(kv_row_bytes(DevCacheFormat::F32, 512), 2048);
        assert_eq!(kv_row_bytes(DevCacheFormat::Fp8Group32, 512), 528);
        assert_eq!(kv_row_bytes(DevCacheFormat::Fp4Group16, 512), 288);
    }

    #[test]
    fn dtype_selection() {
        assert_eq!(dtype_for(Precision::F32, 7168), GemmDType::F32);
        assert_eq!(dtype_for(Precision::F32, 64), GemmDType::F32);
        assert_eq!(dtype_for(Precision::Auto, 7168), GemmDType::Fp8);
        assert_eq!(dtype_for(Precision::Auto, 128), GemmDType::Fp8);
        assert_eq!(dtype_for(Precision::Auto, 64), GemmDType::Bf16);
    }

    #[test]
    fn step_geometry_for_tiny_configs() {
        let g = StepGeometry::new(256, 8, 4, 2).unwrap();
        assert_eq!(g.num_blocks, 64);
        assert_eq!(g.cands_cap, 128);
        assert!(g.cands_cap > g.topk);
    }

    /// Compile-time sanity pins for the public tuning constants.
    const _: () = {
        assert!(DEVICE_MAX_PREFILL_TOKENS >= 1 && DEVICE_MAX_PREFILL_TOKENS <= 8192);
        assert!(MOE_CHUNK_TOKENS >= 1 && MOE_CHUNK_TOKENS <= DEVICE_MAX_PREFILL_TOKENS);
    };

    #[test]
    fn probe_device_is_callable_without_gpu() {
        let _ = DeviceModel::probe_device();
    }

    #[test]
    fn fp8_tiny_config_validates() {
        fp8_tiny(CacheFormat::Bf16).validate().unwrap();
        fp8_tiny(CacheFormat::Fp8Group32).validate().unwrap();
        fp8_tiny(CacheFormat::Fp4Group16).validate().unwrap();
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut worst = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            let m = x.abs().max(y.abs()).max(1e-3);
            worst = worst.max(d / m);
        }
        worst
    }

    /// F32-precision DeviceModel vs the CPU golden model on the tiny
    /// config: prefill + decode steps.
    #[test]
    fn device_f32_matches_cpu_tiny() {
        let Some(dev) = gpu_device() else { return };
        let cfg = ModelConfig::v41_tiny();
        let mut cpu = crate::model::DeepSeekModel::random_init(&cfg, 42).unwrap();
        let mut gpu =
            DeviceModel::random_init(&cfg, 42, &dev, Precision::F32).unwrap();
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 61) % 250).collect();
        let la = cpu.prefill(7, &prompt).unwrap();
        let lb = gpu.prefill(7, &prompt).unwrap();
        assert_eq!(lb.len(), cfg.vocab_size);
        assert!(lb.iter().all(|v| v.is_finite()));
        let e = rel_err(&la, &lb);
        assert!(e < 5e-2, "prefill logits rel err {e}");
        // Decode steps.
        let mut t = 5u32;
        for _ in 0..4 {
            let xa = cpu.decode_step(7, t).unwrap();
            let xb = gpu.decode_step(7, t).unwrap();
            let e = rel_err(&xa, &xb);
            assert!(e < 5e-2, "decode logits rel err {e}");
            t = deepinfer_core::math::argmax(&xb).unwrap() as u32;
        }
        assert_eq!(gpu.seq_len(7), 10);
        assert_eq!(gpu.total_cached_tokens(), 10);
    }

    /// Auto precision (FP8 everywhere at 128-multiple dims) with the FP8
    /// KV cache: finite outputs and loose agreement with the CPU.
    #[test]
    fn device_fp8_auto_smoke() {
        let Some(dev) = gpu_device() else { return };
        let cfg = fp8_tiny(CacheFormat::Fp8Group32);
        let mut cpu = crate::model::DeepSeekModel::random_init(&cfg, 42).unwrap();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::Auto).unwrap();
        let prompt: Vec<u32> = (0..8u32).map(|t| (t * 31) % 250).collect();
        let la = cpu.prefill(3, &prompt).unwrap();
        let lb = gpu.prefill(3, &prompt).unwrap();
        assert!(lb.iter().all(|v| v.is_finite()));
        let e = rel_err(&la, &lb);
        assert!(e < 0.25, "fp8 prefill rel err {e}");
        let xa = cpu.decode_step(3, 100).unwrap();
        let xb = gpu.decode_step(3, 100).unwrap();
        assert!(xb.iter().all(|v| v.is_finite()));
        let e = rel_err(&xa, &xb);
        assert!(e < 0.25, "fp8 decode rel err {e}");
    }

    /// The FP4 KV cache + dedicated FP4 decode kernel.
    #[test]
    fn device_fp4_cache_decode() {
        let Some(dev) = gpu_device() else { return };
        let cfg = fp8_tiny(CacheFormat::Fp4Group16);
        let mut cpu = crate::model::DeepSeekModel::random_init(&cfg, 42).unwrap();
        let mut gpu = DeviceModel::random_init(&cfg, 42, &dev, Precision::Auto).unwrap();
        let prompt: Vec<u32> = (0..8u32).map(|t| (t * 17) % 250).collect();
        let la = cpu.prefill(3, &prompt).unwrap();
        let lb = gpu.prefill(3, &prompt).unwrap();
        assert!(lb.iter().all(|v| v.is_finite()));
        let e = rel_err(&la, &lb);
        assert!(e < 0.3, "fp4 prefill rel err {e}");
        let xa = cpu.decode_step(3, 100).unwrap();
        let xb = gpu.decode_step(3, 100).unwrap();
        assert!(xb.iter().all(|v| v.is_finite()));
        let e = rel_err(&xa, &xb);
        assert!(e < 0.3, "fp4 decode rel err {e}");
    }

    /// Graph decode replay is bit-identical to the eager decode path, and
    /// the graph survives sequence growth (all varying scalars are
    /// device-read).
    #[test]
    fn decode_graph_replay_matches_eager() {
        let Some(dev) = gpu_device() else { return };
        let cfg = fp8_tiny(CacheFormat::Fp8Group32);
        let prompt: Vec<u32> = (0..6u32).map(|t| (t * 23) % 250).collect();
        // Eager run.
        let mut eager = DeviceModel::random_init(&cfg, 7, &dev, Precision::F32).unwrap();
        let _ = eager.prefill(1, &prompt).unwrap();
        let mut eager_logits = Vec::new();
        let mut t = 9u32;
        for _ in 0..6 {
            let l = eager.decode_step(1, t).unwrap();
            t = deepinfer_core::math::argmax(&l).unwrap() as u32;
            eager_logits.push(l);
        }
        // Graph run.
        let mut graphed = DeviceModel::random_init(&cfg, 7, &dev, Precision::F32).unwrap();
        let _ = graphed.prefill(1, &prompt).unwrap();
        assert!(graphed.enable_decode_graph(true).is_ok());
        assert!(graphed.decode_graph_enabled());
        let mut t = 9u32;
        for step in 0..6 {
            let l = graphed.decode_step(1, t).unwrap();
            assert_eq!(
                l, eager_logits[step],
                "graph replay diverged from eager at step {step}"
            );
            t = deepinfer_core::math::argmax(&l).unwrap() as u32;
        }
        assert_eq!(graphed.seq_len(1), prompt.len() + 6);
    }

    /// Rollback on the device: append, roll back, re-append — the state
    /// must reproduce the no-rollback logits bit-for-bit (append-only
    /// caches with position-keyed rows).
    #[test]
    fn rollback_restores_exact_state() {
        let Some(dev) = gpu_device() else { return };
        let cfg = fp8_tiny(CacheFormat::Fp8Group32);
        let prompt: Vec<u32> = (0..5u32).map(|t| (t * 13) % 250).collect();
        let mut m = DeviceModel::random_init(&cfg, 3, &dev, Precision::F32).unwrap();
        let _ = m.prefill(0, &prompt).unwrap();
        let l1 = m.decode_step(0, 11).unwrap();
        let l2 = m.decode_step(0, 12).unwrap();
        m.rollback_seq(0, 1).unwrap();
        let after = m.decode_step(0, 12).unwrap();
        assert_eq!(after, l2, "rollback must restore the exact state");
        let _ = l1;
        assert!(m.rollback_seq(0, 100).is_err());
    }

    /// Reset frees the single-sequence slot.
    #[test]
    fn reset_clears_state() {
        let Some(dev) = gpu_device() else { return };
        let cfg = ModelConfig::v41_tiny();
        let mut m = DeviceModel::random_init(&cfg, 1, &dev, Precision::F32).unwrap();
        let _ = m.prefill(0, &[1, 2, 3]).unwrap();
        assert_eq!(m.seq_len(0), 3);
        m.reset();
        assert_eq!(m.seq_len(0), 0);
        let _ = m.prefill(9, &[1, 2, 3]).unwrap();
        assert_eq!(m.seq_len(9), 3);
    }

    /// The uniform decode path is deterministic run-to-run.
    #[test]
    fn decode_is_deterministic() {
        let Some(dev) = gpu_device() else { return };
        let cfg = fp8_tiny(CacheFormat::Fp4Group16);
        let prompt: Vec<u32> = vec![3, 41, 17, 90, 12];
        let mut a = DeviceModel::random_init(&cfg, 11, &dev, Precision::Auto).unwrap();
        let mut b = DeviceModel::random_init(&cfg, 11, &dev, Precision::Auto).unwrap();
        let la = a.prefill(0, &prompt).unwrap();
        let lb = b.prefill(0, &prompt).unwrap();
        assert_eq!(la, lb);
        for i in 0..3 {
            let xa = a.decode_step(0, 40 + i).unwrap();
            let xb = b.decode_step(0, 40 + i).unwrap();
            assert_eq!(xa, xb);
        }
    }
}
