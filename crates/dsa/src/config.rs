//! Configuration for the DSA lightning indexer.
//!
//! DeepSeek Sparse Attention (DSA, arXiv:2509.07152) replaces dense
//! attention with a lightweight *lightning indexer* that scores every
//! (query, key) pair and keeps only the top-`topk_tokens` keys per query.
//! The defaults here follow the DeepSeek-V3.2 configuration: 64 scoring
//! heads of head_dim 112 (index dim 7168), a 64-key selection block with 8
//! pre-selected survivors per block, `topk_tokens = 2048`, a lightning-quant
//! block of 2048 tokens, and a cascade of causal depthwise convolutions
//! with kernel sizes 3..511.

/// Configuration of the DSA lightning indexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightningIndexerConfig {
    /// Model hidden size the indexer projects from (`wq`/`wk` input dim).
    pub hidden_size: usize,
    /// Number of scoring heads for the MQA-form logits (V3.2: 64).
    pub num_heads: usize,
    /// Per-head index dimension (V3.2: 112, i.e. index dim
    /// `num_heads * head_dim = 7168`).
    pub head_dim: usize,
    /// Key-side block size for top-k pre-selection (V3.2: 64).
    pub block_size: usize,
    /// Number of keys kept per query after the global top-k (V3.2: 2048).
    pub topk_tokens: usize,
    /// Survivors kept per selection block before the global top-k (V3.2: 8).
    pub preselect_per_block: usize,
    /// Lightning-quant block size, counted in tokens along the sequence axis
    /// (V3.2: 2048).
    pub quant_block: usize,
    /// Cascade depthwise-conv kernel sizes
    /// (V3.2 default: 3, 7, 15, 31, 63, 127, 255, 511).
    pub conv_kernels: Vec<usize>,
    /// Context-extrapolation window: `Some(w)` restricts the candidate keys
    /// of query `i` to the last `w` positions ending at `i` (inclusive);
    /// `None` disables extrapolation.
    pub extrapolation_window: Option<usize>,
}

impl LightningIndexerConfig {
    /// DeepSeek-V3.2 defaults for a model with the given hidden size.
    ///
    /// * `num_heads = 64`, `head_dim = 112` (index dim 7168)
    /// * `block_size = 64`, `preselect_per_block = 8`, `topk_tokens = 2048`
    /// * `quant_block = 2048`
    /// * `conv_kernels = [3, 7, 15, 31, 63, 127, 255, 511]`
    /// * `extrapolation_window = None`
    pub fn v32_default(hidden_size: usize) -> Self {
        LightningIndexerConfig {
            hidden_size,
            num_heads: 64,
            head_dim: 112,
            block_size: 64,
            topk_tokens: 2048,
            preselect_per_block: 8,
            quant_block: 2048,
            conv_kernels: vec![3, 7, 15, 31, 63, 127, 255, 511],
            extrapolation_window: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v32_default_fields() {
        let cfg = LightningIndexerConfig::v32_default(7168);
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.head_dim, 112);
        assert_eq!(cfg.num_heads * cfg.head_dim, 7168);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.topk_tokens, 2048);
        assert_eq!(cfg.preselect_per_block, 8);
        assert_eq!(cfg.quant_block, 2048);
        assert_eq!(cfg.conv_kernels, vec![3, 7, 15, 31, 63, 127, 255, 511]);
        assert_eq!(cfg.extrapolation_window, None);
    }
}
