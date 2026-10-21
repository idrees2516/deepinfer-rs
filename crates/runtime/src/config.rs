//! DeepSeek-V4.1-Flash model configuration, parsed from HF-style
//! `config.json` documents.
//!
//! Ground truth: the FlashMLA README (github.com/deepseek-ai/FlashMLA)
//! pins the V4.1-Flash attention geometry — absorbed per-head width 512
//! (448 latent + 64 interleaved rope dims), q-LoRA 1536, 8 output groups
//! of 512 — and the DeepInfer upstream README pins the YOCO self/cross
//! decoder split with SWA bounded replay (see the `deepinfer-swa` crate).
//! Every key accepted by [`ModelConfig::from_json`] carries a documented
//! default, so partial configs (and the empty object) parse; unknown keys
//! are ignored, and the parsed configuration is always run through
//! [`ModelConfig::validate`] before use.

use deepinfer_core::json::Json;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_dsa::LightningIndexerConfig;
use deepinfer_mla::CacheFormat;
use deepinfer_moe::MoeConfig;
use deepinfer_mtp::MtpConfig;

/// FlashMLA page size (tokens per KV-cache page) used by the runtime.
pub(crate) const PAGE_SIZE: usize = 64;

/// Default cross-attention key window — the documented CPU-tractability
/// cap on how many recent self-decoder KV rows a cross-decoder layer
/// attends over densely (see [`ModelConfig::cross_attn_window`]).
pub const DEFAULT_CROSS_ATTN_WINDOW: usize = 512;

/// Default sliding-window size of the cross-decoder's local self-attention.
pub const DEFAULT_SWA_WINDOW: usize = 128;

/// Upper bound on `hidden_size` accepted by [`ModelConfig::validate`].
pub const MAX_HIDDEN_SIZE: usize = 8192;
/// Upper bound on `vocab_size` accepted by [`ModelConfig::validate`]
/// (matches `deepinfer_mtp::config::MAX_VOCAB`).
pub const MAX_VOCAB_SIZE: usize = 1 << 21;
/// Upper bound on the layer count.
pub const MAX_LAYERS: usize = 128;
/// Upper bound on the head count.
pub const MAX_HEADS: usize = 128;
/// Upper bound on the absorbed per-head width.
pub const MAX_HEAD_DIM: usize = 1024;
/// Upper bound on the query-LoRA rank.
pub const MAX_Q_LORA_RANK: usize = 8192;
/// Upper bound on `max_position_embeddings`.
pub const MAX_POSITIONS: usize = 1 << 20;
/// Upper bound on any attention window field.
pub const MAX_WINDOW: usize = 1 << 16;

/// Full configuration of the DeepSeek-V4.1-Flash runtime model: the YOCO
/// layer split, MLA geometry, DSA indexer, MoE and MTP sub-configurations.
///
/// The first `n_self_layers` layers form the self-decoder (MLA attention
/// with DSA sparse selection + MoE); the remaining `n_cross_layers` form
/// the cross-decoder (cross-attention into the self-decoder KV + SWA local
/// self-attention + MoE) — the "You Only Cache Once" layout (Sun et al.,
/// arXiv:2405.05254).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Model hidden width.
    pub hidden_size: usize,
    /// Total layer count (`n_self_layers + n_cross_layers`).
    pub n_layers: usize,
    /// Self-decoder layers (MLA + DSA + MoE).
    pub n_self_layers: usize,
    /// Cross-decoder layers (cross-attn + SWA + MoE).
    pub n_cross_layers: usize,
    /// Attention heads per layer.
    pub n_heads: usize,
    /// Absorbed per-head attention width (latent + rope; V4.1: 512).
    pub head_dim: usize,
    /// Query-LoRA rank (V4.1: 1536).
    pub q_lora_rank: usize,
    /// Rotated dims per head row (V4.1: 64, interleaved).
    pub rope_dim: usize,
    /// RoPE base (`theta`; V4.1: 10000).
    pub rope_base: f64,
    /// RMSNorm epsilon (V4.1: 1e-4).
    pub rms_eps: f32,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum supported absolute token position.
    pub max_position_embeddings: usize,
    /// MLA output-LoRA groups (`n_wv_group`; V4.1: 8).
    pub o_groups: usize,
    /// MLA output-LoRA rank (V4.1: 512).
    pub o_lora_rank: usize,
    /// SWA window of the cross-decoder's local self-attention.
    pub swa_window: usize,
    /// Cross-attention key cap: each cross-decoder layer densely attends
    /// the most recent `cross_attn_window` rows of the self-decoder KV
    /// cache (a documented CPU-tractability simplification of the YOCO
    /// cross-attention, which reads the full encoder cache).
    pub cross_attn_window: usize,
    /// Whether `lm_head` is tied to the token embedding.
    pub tie_word_embeddings: bool,
    /// KV-cache row format (`bf16` reference, or the V4.1 FP8/FP4 rows).
    pub cache: CacheFormat,
    /// DSA lightning-indexer configuration (hidden pinned to
    /// `hidden_size`).
    pub dsa: LightningIndexerConfig,
    /// MoE configuration (hidden pinned to `hidden_size`).
    pub moe: MoeConfig,
    /// MTP (NEXTN) draft-module shape configuration. The *effective*
    /// draft length is always read from the environment
    /// (`DEEPINFER_USE_NEXTN_DECODE`) at generation time; `draft_len`
    /// here only records the default (5, matching `Env::default`).
    pub mtp: MtpConfig,
}

impl ModelConfig {
    /// Parses an HF-style `config.json` document. Missing keys fall back
    /// to V4.1-Flash-ish defaults; the result is validated
    /// ([`ModelConfig::validate`]) before being returned.
    ///
    /// # Errors
    /// [`Error::Checkpoint`] on unreadable/malformed values (the
    /// document describes an impossible model), [`Error::OutOfRange`] /
    /// [`Error::InvalidArgument`] / [`Error::CapacityExceeded`] bubbling
    /// from validation.
    pub fn from_json(json: &Json) -> Result<Self> {
        let cfg = Self::parse_relaxed(json)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Small self-test configuration exercising every architectural
    /// element (MLA + DSA + SWA + MoE + MTP) at toy scale: hidden 64,
    /// 2 self + 2 cross layers, 2 heads over the full V4.1 absorbed
    /// head width 512 (latent 448 + rope 64), byte-level vocab 256,
    /// context 256, 4 routed experts of which 2 are active.
    pub fn v41_tiny() -> Self {
        ModelConfig {
            hidden_size: 64,
            n_layers: 4,
            n_self_layers: 2,
            n_cross_layers: 2,
            n_heads: 2,
            head_dim: 512,
            q_lora_rank: 32,
            rope_dim: 64,
            rope_base: 10_000.0,
            rms_eps: 1e-4,
            vocab_size: 256,
            max_position_embeddings: 256,
            o_groups: 1,
            o_lora_rank: 32,
            swa_window: 16,
            cross_attn_window: DEFAULT_CROSS_ATTN_WINDOW,
            tie_word_embeddings: false,
            cache: CacheFormat::Bf16,
            dsa: LightningIndexerConfig {
                hidden_size: 64,
                num_heads: 2,
                head_dim: 8,
                block_size: 4,
                topk_tokens: 8,
                preselect_per_block: 2,
                quant_block: 2048,
                conv_kernels: vec![3],
                extrapolation_window: None,
            },
            moe: MoeConfig {
                hidden: 64,
                intermediate: 16,
                num_routed: 4,
                num_activated: 2,
                num_shared: 1,
                group_size: 1,
                use_noaux_tc_bias: true,
            },
            mtp: MtpConfig {
                hidden: 64,
                vocab: 256,
                draft_len: 5,
            },
        }
    }

    /// Latent (KV-LoRA) rank: `head_dim - rope_dim` (V4.1 absorbed
    /// geometry: the cached KV row is `[latent | rope]`).
    pub fn kv_lora_rank(&self) -> usize {
        self.head_dim - self.rope_dim
    }

    /// Base softmax scale of the MLA attention paths: `1/sqrt(head_dim)`.
    /// (The DSA mscale is folded in on top of this by the self-decoder
    /// layers; see `deepinfer_dsa::mscale`.)
    pub fn sm_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Pages allocatable per paged KV cache: enough for one
    /// `max_position_embeddings`-token sequence at the FlashMLA page
    /// size of 64.
    pub fn max_pages(&self) -> usize {
        self.max_position_embeddings.div_ceil(PAGE_SIZE).max(1)
    }

    /// Validates every field and cross-field invariant.
    ///
    /// Checks: positive/bounded dimensions; `rope_dim` even and strictly
    /// inside `head_dim`; `n_self_layers + n_cross_layers == n_layers`
    /// with at least one self layer (the cross layers read its KV);
    /// `o_groups` divides `n_heads`; sub-config hidden/vocab fields
    /// agree with the model; MoE divisibility; RoPE base > 1; finite
    /// epsilon.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] / [`Error::OutOfRange`] /
    /// [`Error::CapacityExceeded`] with a message naming the offending
    /// field.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size >= 1 && self.hidden_size <= MAX_HIDDEN_SIZE,
            OutOfRange,
            "hidden_size must be in [1, {MAX_HIDDEN_SIZE}] (got {})",
            self.hidden_size
        );
        ensure!(
            self.vocab_size >= 1 && self.vocab_size <= MAX_VOCAB_SIZE,
            OutOfRange,
            "vocab_size must be in [1, {MAX_VOCAB_SIZE}] (got {})",
            self.vocab_size
        );
        ensure!(
            self.n_layers >= 1 && self.n_layers <= MAX_LAYERS,
            OutOfRange,
            "n_layers must be in [1, {MAX_LAYERS}] (got {})",
            self.n_layers
        );
        ensure!(
            self.n_self_layers + self.n_cross_layers == self.n_layers,
            InvalidArgument,
            "yoco split inconsistent: {} self + {} cross != {} layers",
            self.n_self_layers,
            self.n_cross_layers,
            self.n_layers
        );
        ensure!(
            self.n_self_layers >= 1,
            InvalidArgument,
            "at least one self-decoder layer is required (cross layers read its KV cache)"
        );
        ensure!(
            self.n_heads >= 1 && self.n_heads <= MAX_HEADS,
            OutOfRange,
            "n_heads must be in [1, {MAX_HEADS}] (got {})",
            self.n_heads
        );
        ensure!(
            self.head_dim >= 2 && self.head_dim <= MAX_HEAD_DIM,
            OutOfRange,
            "head_dim must be in [2, {MAX_HEAD_DIM}] (got {})",
            self.head_dim
        );
        ensure!(
            self.rope_dim >= 2 && self.rope_dim % 2 == 0 && self.rope_dim < self.head_dim,
            InvalidArgument,
            "rope_dim must be even and strictly below head_dim (got rope {} / head {})",
            self.rope_dim,
            self.head_dim
        );
        ensure!(
            self.q_lora_rank >= 1 && self.q_lora_rank <= MAX_Q_LORA_RANK,
            OutOfRange,
            "q_lora_rank must be in [1, {MAX_Q_LORA_RANK}] (got {})",
            self.q_lora_rank
        );
        ensure!(
            self.max_position_embeddings >= 1
                && self.max_position_embeddings <= MAX_POSITIONS,
            OutOfRange,
            "max_position_embeddings must be in [1, {MAX_POSITIONS}] (got {})",
            self.max_position_embeddings
        );
        ensure!(
            self.o_groups >= 1 && self.n_heads % self.o_groups == 0,
            InvalidArgument,
            "o_groups ({}) must divide n_heads ({})",
            self.o_groups,
            self.n_heads
        );
        ensure!(
            self.o_lora_rank >= 1 && self.o_lora_rank <= MAX_Q_LORA_RANK,
            OutOfRange,
            "o_lora_rank must be in [1, {MAX_Q_LORA_RANK}] (got {})",
            self.o_lora_rank
        );
        ensure!(
            self.swa_window >= 1 && self.swa_window <= MAX_WINDOW,
            OutOfRange,
            "swa_window must be in [1, {MAX_WINDOW}] (got {})",
            self.swa_window
        );
        ensure!(
            self.cross_attn_window >= 1 && self.cross_attn_window <= MAX_WINDOW,
            OutOfRange,
            "cross_attn_window must be in [1, {MAX_WINDOW}] (got {})",
            self.cross_attn_window
        );
        ensure!(
            self.rope_base.is_finite() && self.rope_base > 1.0,
            InvalidArgument,
            "rope_base must be finite and > 1 (got {})",
            self.rope_base
        );
        ensure!(
            self.rms_eps.is_finite() && self.rms_eps > 0.0,
            InvalidArgument,
            "rms_eps must be finite and positive (got {})",
            self.rms_eps
        );
        // Sub-configuration consistency.
        ensure!(
            self.dsa.hidden_size == self.hidden_size,
            InvalidArgument,
            "dsa.hidden_size ({}) must equal hidden_size ({})",
            self.dsa.hidden_size,
            self.hidden_size
        );
        for (name, v) in [
            ("num_heads", self.dsa.num_heads),
            ("head_dim", self.dsa.head_dim),
            ("block_size", self.dsa.block_size),
            ("topk_tokens", self.dsa.topk_tokens),
            ("preselect_per_block", self.dsa.preselect_per_block),
            ("quant_block", self.dsa.quant_block),
        ] {
            ensure!(
                (1..=MAX_WINDOW).contains(&v),
                OutOfRange,
                "dsa.{name} must be in [1, {MAX_WINDOW}] (got {v})"
            );
        }
        ensure!(
            !self.dsa.conv_kernels.is_empty()
                && self
                    .dsa
                    .conv_kernels
                    .iter()
                    .all(|k| *k >= 1 && *k <= MAX_WINDOW),
            OutOfRange,
            "dsa.conv_kernels entries must be in [1, {MAX_WINDOW}]"
        );
        if let Some(xw) = self.dsa.extrapolation_window {
            ensure!(
                (1..=MAX_WINDOW).contains(&xw),
                OutOfRange,
                "dsa.extrapolation_window must be in [1, {MAX_WINDOW}] (got {xw})"
            );
        }
        let m = &self.moe;
        ensure!(m.hidden == self.hidden_size, InvalidArgument,
            "moe.hidden ({}) must equal hidden_size ({})", m.hidden, self.hidden_size);
        ensure!(m.intermediate >= 1 && m.intermediate <= MAX_HIDDEN_SIZE,
            OutOfRange, "moe.intermediate must be in [1, {MAX_HIDDEN_SIZE}] (got {})", m.intermediate);
        ensure!(m.num_routed >= 1 && m.num_routed <= MAX_WINDOW,
            OutOfRange, "moe.num_routed must be in [1, {MAX_WINDOW}] (got {})", m.num_routed);
        ensure!(m.num_activated >= 1 && m.num_activated <= m.num_routed,
            InvalidArgument, "moe.num_activated ({}) must be in [1, num_routed {}]", m.num_activated, m.num_routed);
        ensure!(m.num_shared <= MAX_WINDOW, OutOfRange, "moe.num_shared must be <= {MAX_WINDOW} (got {})", m.num_shared);
        ensure!(m.group_size >= 1, InvalidArgument, "moe.group_size must be >= 1 (got {})", m.group_size);
        ensure!(m.num_routed % m.group_size == 0 && m.num_activated % m.group_size == 0,
            InvalidArgument, "moe.group_size ({}) must divide num_routed ({}) and num_activated ({})",
            m.group_size, m.num_routed, m.num_activated);
        self.mtp.validate()?;
        ensure!(
            self.mtp.hidden == self.hidden_size && self.mtp.vocab == self.vocab_size,
            InvalidArgument,
            "mtp shape ({}, {}) must match the model ({}, {})",
            self.mtp.hidden,
            self.mtp.vocab,
            self.hidden_size,
            self.vocab_size
        );
        Ok(())
    }

    /// Field-by-field parse with defaults; no validation.
    fn parse_relaxed(json: &Json) -> Result<Self> {
        let usize_of = |v: u64, what: &str| -> Result<usize> {
            ensure!(
                v <= usize::MAX as u64,
                Checkpoint,
                "{what} value {v} exceeds usize"
            );
            Ok(v as usize)
        };
        let field = |keys: &[&str]| -> Option<u64> {
            for k in keys {
                if let Some(v) = json.u64_field(k) {
                    return Some(v);
                }
            }
            None
        };

        let hidden_size = match field(&["hidden_size"]) {
            Some(v) => usize_of(v, "hidden_size")?,
            None => 4096,
        };
        let n_layers = match field(&["n_layers", "num_hidden_layers"]) {
            Some(v) => usize_of(v, "n_layers")?,
            None => 48,
        };
        let n_self_layers = match field(&["n_self_layers", "num_self_decoder_layers"]) {
            Some(v) => usize_of(v, "n_self_layers")?,
            None => n_layers.div_ceil(2),
        };
        let n_cross_layers = match field(&["n_cross_layers", "num_cross_decoder_layers"]) {
            Some(v) => usize_of(v, "n_cross_layers")?,
            None => n_layers - n_self_layers,
        };
        let n_heads = match field(&["num_attention_heads", "n_heads"]) {
            Some(v) => usize_of(v, "num_attention_heads")?,
            None => 64,
        };
        let head_dim = match field(&["head_dim", "qk_head_dim"]) {
            Some(v) => usize_of(v, "head_dim")?,
            None => 512,
        };
        let rope_dim = match field(&["rope_dim", "qk_rope_head_dim"]) {
            Some(v) => usize_of(v, "rope_dim")?,
            None => 64,
        };
        let q_lora_rank = match field(&["q_lora_rank"]) {
            Some(v) => usize_of(v, "q_lora_rank")?,
            None => 1536,
        };
        if let Some(kv_lora) = field(&["kv_lora_rank"]) {
            let kv_lora = usize_of(kv_lora, "kv_lora_rank")?;
            ensure!(
                kv_lora + rope_dim == head_dim,
                Checkpoint,
                "absorbed V4.1 geometry violated: kv_lora_rank ({kv_lora}) + rope_dim ({rope_dim}) != head_dim ({head_dim})"
            );
        }
        let rope_base = json.f64_field("rope_theta").unwrap_or(10_000.0);
        let rms_eps = json.f64_field("rms_norm_eps").unwrap_or(1e-4) as f32;
        let vocab_size = match field(&["vocab_size"]) {
            Some(v) => usize_of(v, "vocab_size")?,
            None => 129_280,
        };
        let max_position_embeddings =
            match field(&["max_position_embeddings"]) {
                Some(v) => usize_of(v, "max_position_embeddings")?,
                None => 163_840,
            };
        let o_groups = match field(&["o_groups", "n_wv_group"]) {
            Some(v) => usize_of(v, "o_groups")?,
            None => 8,
        };
        let o_lora_rank = match field(&["o_lora_rank"]) {
            Some(v) => usize_of(v, "o_lora_rank")?,
            None => 512,
        };
        let swa_window = match field(&["swa_window", "sliding_window"]) {
            Some(v) => usize_of(v, "swa_window")?,
            None => DEFAULT_SWA_WINDOW,
        };
        let cross_attn_window = match field(&["cross_attn_window"]) {
            Some(v) => usize_of(v, "cross_attn_window")?,
            None => DEFAULT_CROSS_ATTN_WINDOW,
        };
        let tie_word_embeddings = json.bool_field("tie_word_embeddings").unwrap_or(false);
        let cache = match json
            .str_field("kv_cache_dtype")
            .or_else(|| json.str_field("cache_format"))
            .unwrap_or("auto")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "auto" | "bf16" | "f16" | "fp16" => CacheFormat::Bf16,
            "fp8" | "e4m3" => CacheFormat::Fp8Group32,
            "fp4" | "e2m1" => CacheFormat::Fp4Group16,
            other => {
                return Err(Error::Checkpoint(format!(
                    "unsupported kv cache dtype {other:?} (expected auto/bf16/fp8/fp4)"
                )))
            }
        };

        let dsa = Self::parse_dsa(json, hidden_size)?;
        let moe = Self::parse_moe(json, hidden_size)?;

        Ok(ModelConfig {
            hidden_size,
            n_layers,
            n_self_layers,
            n_cross_layers,
            n_heads,
            head_dim,
            q_lora_rank,
            rope_dim,
            rope_base,
            rms_eps,
            vocab_size,
            max_position_embeddings,
            o_groups,
            o_lora_rank,
            swa_window,
            cross_attn_window,
            tie_word_embeddings,
            cache,
            dsa,
            moe,
            mtp: MtpConfig {
                hidden: hidden_size,
                vocab: vocab_size,
                // Placeholder only — the effective draft length always
                // comes from DEEPINFER_USE_NEXTN_DECODE at generation time.
                draft_len: 5,
            },
        })
    }

    /// DSA lightning-indexer section: nested `dsa`/`indexer` object with
    /// V3.2 defaults, hidden pinned to the model.
    fn parse_dsa(json: &Json, hidden: usize) -> Result<LightningIndexerConfig> {
        let default = LightningIndexerConfig::v32_default(hidden);
        let Some(node) = json.get("dsa").or_else(|| json.get("indexer")) else {
            return Ok(default);
        };
        let get = |keys: &[&str]| -> Option<u64> {
            for k in keys {
                if let Some(v) = node.u64_field(k) {
                    return Some(v);
                }
            }
            None
        };
        let usize_of = |v: u64, what: &str| -> Result<usize> {
            ensure!(v <= usize::MAX as u64, Checkpoint, "{what} value {v} exceeds usize");
            Ok(v as usize)
        };
        let conv_kernels = match node.get("conv_kernels").and_then(Json::as_arr) {
            Some(arr) => {
                let mut ks = Vec::with_capacity(arr.len());
                for v in arr {
                    let raw = v.as_u64().ok_or_else(|| {
                        Error::Checkpoint("dsa.conv_kernels entries must be integers".into())
                    })?;
                    ks.push(usize_of(raw, "conv_kernels")?);
                }
                ks
            }
            None => default.conv_kernels,
        };
        Ok(LightningIndexerConfig {
            hidden_size: hidden,
            num_heads: match get(&["num_heads"]) {
                Some(v) => usize_of(v, "dsa.num_heads")?,
                None => default.num_heads,
            },
            head_dim: match get(&["head_dim"]) {
                Some(v) => usize_of(v, "dsa.head_dim")?,
                None => default.head_dim,
            },
            block_size: match get(&["block_size"]) {
                Some(v) => usize_of(v, "dsa.block_size")?,
                None => default.block_size,
            },
            topk_tokens: match get(&["topk_tokens"]) {
                Some(v) => usize_of(v, "dsa.topk_tokens")?,
                None => default.topk_tokens,
            },
            preselect_per_block: match get(&["preselect_per_block"]) {
                Some(v) => usize_of(v, "dsa.preselect_per_block")?,
                None => default.preselect_per_block,
            },
            quant_block: match get(&["quant_block"]) {
                Some(v) => usize_of(v, "dsa.quant_block")?,
                None => default.quant_block,
            },
            conv_kernels,
            extrapolation_window: match node.u64_field("extrapolation_window") {
                Some(v) => Some(usize_of(v, "dsa.extrapolation_window")?),
                None => None,
            },
        })
    }

    /// MoE section: nested `moe` object (preferred) with top-level
    /// DeepSeek-style fallbacks, hidden pinned to the model.
    fn parse_moe(json: &Json, hidden: usize) -> Result<MoeConfig> {
        let default = MoeConfig::v41_default(hidden);
        let node = json.get("moe").or_else(|| json.get("mlp"));
        // Reads through the nested node when present, else top level.
        let get = |keys: &[&str]| -> Option<u64> {
            for k in keys {
                if let Some(n) = node {
                    if let Some(v) = n.u64_field(k) {
                        return Some(v);
                    }
                }
                if let Some(v) = json.u64_field(k) {
                    return Some(v);
                }
            }
            None
        };
        let get_bool = |keys: &[&str]| -> Option<bool> {
            for k in keys {
                if let Some(n) = node {
                    if let Some(v) = n.bool_field(k) {
                        return Some(v);
                    }
                }
                if let Some(v) = json.bool_field(k) {
                    return Some(v);
                }
            }
            None
        };
        let usize_of = |v: u64, what: &str| -> Result<usize> {
            ensure!(v <= usize::MAX as u64, Checkpoint, "{what} value {v} exceeds usize");
            Ok(v as usize)
        };
        Ok(MoeConfig {
            hidden,
            intermediate: match get(&["intermediate", "moe_intermediate_size"]) {
                Some(v) => usize_of(v, "moe.intermediate")?,
                None => default.intermediate,
            },
            num_routed: match get(&["num_routed", "n_routed_experts"]) {
                Some(v) => usize_of(v, "moe.num_routed")?,
                None => default.num_routed,
            },
            num_activated: match get(&["num_activated", "num_experts_per_tok", "n_activated_experts"]) {
                Some(v) => usize_of(v, "moe.num_activated")?,
                None => default.num_activated,
            },
            num_shared: match get(&["num_shared", "n_shared_experts"]) {
                Some(v) => usize_of(v, "moe.num_shared")?,
                None => default.num_shared,
            },
            group_size: match get(&["group_size", "expert_group_size"]) {
                Some(v) => usize_of(v, "moe.group_size")?,
                None => default.group_size,
            },
            use_noaux_tc_bias: get_bool(&["use_noaux_tc_bias", "noaux_tc"]).unwrap_or(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_parses_realistic_v41_flash_config() {
        let text = r#"{
            "architectures": ["DeepseekV41ForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 61,
            "num_self_decoder_layers": 32,
            "num_cross_decoder_layers": 29,
            "num_attention_heads": 64,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 448,
            "q_lora_rank": 1536,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-4,
            "vocab_size": 129280,
            "max_position_embeddings": 163840,
            "n_wv_group": 8,
            "o_lora_rank": 512,
            "sliding_window": 128,
            "cross_attn_window": 512,
            "tie_word_embeddings": false,
            "kv_cache_dtype": "fp8",
            "dsa": {
                "num_heads": 64,
                "head_dim": 112,
                "block_size": 64,
                "topk_tokens": 2048,
                "preselect_per_block": 8,
                "quant_block": 2048,
                "conv_kernels": [3, 7, 15],
                "extrapolation_window": 8192
            },
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_intermediate_size": 1024,
            "expert_group_size": 1
        }"#;
        let json = Json::parse(text).unwrap();
        let cfg = ModelConfig::from_json(&json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.n_layers, 61);
        assert_eq!(cfg.n_self_layers, 32);
        assert_eq!(cfg.n_cross_layers, 29);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.kv_lora_rank(), 448);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.rope_base, 10_000.0);
        assert!((cfg.rms_eps - 1e-4).abs() < 1e-12);
        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.max_position_embeddings, 163_840);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 512);
        assert_eq!(cfg.swa_window, 128);
        assert_eq!(cfg.cross_attn_window, 512);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.cache, CacheFormat::Fp8Group32);
        assert_eq!(cfg.dsa.hidden_size, 4096);
        assert_eq!(cfg.dsa.num_heads, 64);
        assert_eq!(cfg.dsa.head_dim, 112);
        assert_eq!(cfg.dsa.topk_tokens, 2048);
        assert_eq!(cfg.dsa.conv_kernels, vec![3, 7, 15]);
        assert_eq!(cfg.dsa.extrapolation_window, Some(8192));
        assert_eq!(cfg.moe.hidden, 4096);
        assert_eq!(cfg.moe.num_routed, 256);
        assert_eq!(cfg.moe.num_activated, 8);
        assert_eq!(cfg.moe.num_shared, 1);
        assert_eq!(cfg.moe.intermediate, 1024);
        assert_eq!(cfg.mtp.hidden, 4096);
        assert_eq!(cfg.mtp.vocab, 129_280);
        assert!((cfg.sm_scale() - 1.0 / 512.0f32.sqrt()).abs() < 1e-12);
        assert_eq!(cfg.max_pages(), 163_840usize.div_ceil(PAGE_SIZE));
    }

    #[test]
    fn from_json_defaults_on_empty_object() {
        let cfg = ModelConfig::from_json(&Json::parse("{}").unwrap()).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.n_layers, 48);
        assert_eq!(cfg.n_self_layers, 24);
        assert_eq!(cfg.n_cross_layers, 24);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 512);
        assert_eq!(cfg.swa_window, DEFAULT_SWA_WINDOW);
        assert_eq!(cfg.cross_attn_window, DEFAULT_CROSS_ATTN_WINDOW);
        assert_eq!(cfg.cache, CacheFormat::Bf16);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.dsa.topk_tokens, 2048);
        assert_eq!(cfg.dsa.conv_kernels.len(), 8);
        assert_eq!(cfg.moe.num_routed, 256);
        assert_eq!(cfg.moe.num_activated, 8);
        assert_eq!(cfg.moe.intermediate, 1024); // hidden / 4
    }

    #[test]
    fn from_json_rejects_inconsistent_yoco_split() {
        let json = Json::parse(
            r#"{"n_layers": 4, "n_self_layers": 3, "n_cross_layers": 3}"#,
        )
        .unwrap();
        assert!(ModelConfig::from_json(&json).is_err());
    }

    #[test]
    fn from_json_rejects_absorbed_dim_mismatch() {
        let json = Json::parse(
            r#"{"head_dim": 512, "rope_dim": 64, "kv_lora_rank": 384}"#,
        )
        .unwrap();
        let err = ModelConfig::from_json(&json).unwrap_err();
        assert!(err.to_string().contains("kv_lora_rank"), "{err}");
    }

    #[test]
    fn from_json_rejects_bad_cache_dtype() {
        let json = Json::parse(r#"{"kv_cache_dtype": "int4"}"#).unwrap();
        assert!(ModelConfig::from_json(&json).is_err());
    }

    #[test]
    fn v41_tiny_fields() {
        let cfg = ModelConfig::v41_tiny();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.n_layers, 4);
        assert_eq!(cfg.n_self_layers, 2);
        assert_eq!(cfg.n_cross_layers, 2);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.kv_lora_rank(), 448);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.max_position_embeddings, 256);
        assert_eq!(cfg.dsa.topk_tokens, 8);
        assert_eq!(cfg.dsa.block_size, 4);
        assert_eq!(cfg.moe.num_routed, 4);
        assert_eq!(cfg.moe.num_activated, 2);
        assert_eq!(cfg.mtp.vocab, 256);
        assert_eq!(cfg.max_pages(), 4);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_rejects_bad_configs() {
        let bad = |mutate: fn(&mut ModelConfig)| {
            let mut cfg = ModelConfig::v41_tiny();
            mutate(&mut cfg);
            assert!(cfg.validate().is_err(), "{cfg:?}");
        };
        bad(|c| c.hidden_size = 0);
        bad(|c| c.vocab_size = 0);
        bad(|c| c.n_cross_layers = 3); // split mismatch
        bad(|c| {
            c.n_self_layers = 0;
            c.n_cross_layers = 4;
        });
        bad(|c| c.rope_dim = 63); // odd
        bad(|c| c.rope_dim = 512); // >= head_dim
        bad(|c| c.o_groups = 3); // does not divide 2 heads
        bad(|c| c.swa_window = 0);
        bad(|c| c.cross_attn_window = 0);
        bad(|c| c.rope_base = 1.0);
        bad(|c| c.rms_eps = -1.0);
        bad(|c| c.moe.num_activated = 5);
        bad(|c| c.moe.group_size = 3);
        bad(|c| c.dsa.topk_tokens = 0);
        bad(|c| c.dsa.hidden_size = 32);
        bad(|c| c.mtp.vocab = 128);
        bad(|c| c.max_position_embeddings = 0);
    }

    #[test]
    fn validate_accepts_no_cross_layers() {
        // A pure self-decoder model (n_cross == 0) is legal.
        let mut cfg = ModelConfig::v41_tiny();
        cfg.n_layers = 2;
        cfg.n_self_layers = 2;
        cfg.n_cross_layers = 0;
        cfg.validate().unwrap();
    }
}
