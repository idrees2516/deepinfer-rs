//! MLA layer configuration and KV-cache format selection for
//! DeepSeek-V4.1-Flash.
//!
//! Ground truth: FlashMLA README (github.com/deepseek-ai/FlashMLA) —
//! h_q = 64, head_dim = 512 = 448 latent + 64 rope dims, q_lora_rank =
//! 1536, n_wv_group = 8, wv_group_size = 8, o_lora_rank = 512,
//! interleaved ("non-neox") RoPE with base 10000, page size 64.

use deepinfer_core::rope::YarnConfig;

/// Quantization format of one KV-cache token row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheFormat {
    /// Two BF16 bytes per dim (`head_dim * 2` bytes per token) — the
    /// exact reference format.
    Bf16,
    /// V4.1 FP8 cache: one E4M3 value per dim plus one UE8M0 scale per 32
    /// dims (`head_dim + head_dim / 32` bytes per token; 528 for the
    /// canonical 512-dim row).
    Fp8Group32,
    /// V4.1 FP4 cache: two E2M1 values per byte (the even-index value in
    /// the LOW nibble) plus one E4M3 scale per 16 dims
    /// (`head_dim / 2 + head_dim / 16` bytes per token; 288 for the
    /// canonical 512-dim row).
    Fp4Group16,
}

/// FlashMLA page size: tokens per KV-cache page.
pub(crate) const PAGE_SIZE: usize = 64;

/// Configuration of one V4.1-Flash MLA attention layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaConfig {
    /// Model hidden width (V4.1: 4096).
    pub hidden_size: usize,
    /// Query-LoRA rank (V4.1: 1536).
    pub q_lora_rank: usize,
    /// Number of query heads (V4.1: 64).
    pub n_heads: usize,
    /// Absorbed per-head query / KV-cache width (V4.1: 512 = 448 latent
    /// + 64 rope dims; `Q_b` folds `W_UK` in, so queries and cached KV
    /// share this width).
    pub head_dim: usize,
    /// Rotated dims per head row (V4.1: 64, interleaved style, occupying
    /// the LAST `rope_dim` dims of each row).
    pub rope_dim: usize,
    /// Output-LoRA groups (`n_wv_group`, V4.1: 8).
    pub o_groups: usize,
    /// Output-LoRA rank (V4.1: 512).
    pub o_lora_rank: usize,
    /// RoPE base (`theta`, V4.1: 10000).
    pub rope_base: f64,
    /// Optional YaRN long-context scaling.
    pub yarn: Option<YarnConfig>,
    /// RMSNorm epsilon (V4.1: 1e-4).
    pub rms_eps: f32,
    /// Softmax scale; the default is `1/sqrt(head_dim)`. Callers fold the
    /// DSA mscale into this value when running sparse attention.
    pub sm_scale: f32,
    /// KV-cache row format.
    pub cache: CacheFormat,
    /// Maximum number of pages allocatable by [`crate::MlaPagedCache`].
    pub max_pages: usize,
}

impl MlaConfig {
    /// DeepSeek-V4.1-Flash defaults for a context of `max_position`
    /// tokens: hidden 4096, q_lora 1536, 64 heads x 512 dims
    /// (448 latent + 64 rope), 8 output groups x 512 lora rank,
    /// interleaved RoPE base 10000 (no YaRN; callers opt in via the
    /// `yarn` field), `sm_scale = 1/sqrt(512)`, and a paged cache sized
    /// so that one `max_position`-token sequence fits
    /// (`ceil(max_position / 64)` pages, at least one).
    pub fn v41_default(max_position: usize) -> MlaConfig {
        MlaConfig {
            hidden_size: 4096,
            q_lora_rank: 1536,
            n_heads: 64,
            head_dim: 512,
            rope_dim: 64,
            o_groups: 8,
            o_lora_rank: 512,
            rope_base: 10_000.0,
            yarn: None,
            rms_eps: 1e-4,
            sm_scale: 1.0 / (512.0f32).sqrt(),
            cache: CacheFormat::Bf16,
            max_pages: max_position.div_ceil(PAGE_SIZE).max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v41_defaults_match_flashmla_ground_truth() {
        let cfg = MlaConfig::v41_default(8192);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.n_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.o_groups, 8);
        assert_eq!(cfg.o_lora_rank, 512);
        assert_eq!(cfg.rope_base, 10_000.0);
        assert!(cfg.yarn.is_none());
        assert!((cfg.rms_eps - 1e-4).abs() < 1e-12);
        assert!((cfg.sm_scale - 1.0 / 512.0f32.sqrt()).abs() < 1e-12);
        assert_eq!(cfg.cache, CacheFormat::Bf16);
        // 8192 tokens / 64 per page = 128 pages.
        assert_eq!(cfg.max_pages, 128);
        // Non-multiple rounds up; zero still allocates one page.
        assert_eq!(MlaConfig::v41_default(65).max_pages, 2);
        assert_eq!(MlaConfig::v41_default(0).max_pages, 1);
    }
}
