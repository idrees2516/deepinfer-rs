//! Lightning-indexer weight storage and deterministic random initialization.
//!
//! The layouts mirror the DeepSeek-V3.2 indexer checkpoint tensors
//! (arXiv:2509.07152, "Lightning Indexer"): a query/key projection into the
//! index dim, per-query head weights consumed by the MQA-logits scoring
//! kernel, a coefficient-gating projection, and one depthwise kernel per
//! cascade entry.

use crate::config::LightningIndexerConfig;
use deepinfer_core::norm::rms_mean_square;
use deepinfer_core::rng::Philox;

/// Weights of the DSA lightning indexer (all buffers row-major).
#[derive(Debug, Clone)]
pub struct LightningIndexerWeights {
    /// Query projection `[num_heads*head_dim, hidden]`.
    pub wq: Vec<f32>,
    /// Key projection `[num_heads*head_dim, hidden]`.
    pub wk: Vec<f32>,
    /// Per-query head weights `[num_heads, num_heads*head_dim]`:
    /// `w_i = weights_proj @ q_i` yields one scalar per head per query.
    pub weights_proj: Vec<f32>,
    /// Coefficient-gating projection `[2, 2*hidden]`; after `sigmoid`, row 0
    /// gates the thunder branch and row 1 the conv branch.
    pub gate_proj: Vec<f32>,
    /// Cascade depthwise-conv kernels: one tap-weight vector per cascade
    /// entry, `conv_weights[c].len() == config.conv_kernels[c]`.
    pub conv_weights: Vec<Vec<f32>>,
}

impl LightningIndexerWeights {
    /// Deterministic random initialization from `rng`.
    ///
    /// Each buffer is filled with uniform values in `[-b, b]` with
    /// `b = 1/sqrt(fan_in)` — fan-in `hidden` for `wq`/`wk`, the index dim
    /// for `weights_proj`, `2*hidden` for `gate_proj`, and the kernel size
    /// for each conv cascade. The generator is consumed in the fixed order
    /// `wq`, `wk`, `weights_proj`, `gate_proj`, `conv_weights`, so identical
    /// `(seed, stream)` pairs always produce bit-identical weights.
    ///
    /// Degenerate configs (zero dims, or buffer sizes that overflow
    /// `usize`) yield empty buffers instead of panicking;
    /// [`crate::indexer_forward`] then rejects them with a
    /// dimension-mismatch error.
    pub fn from_config(cfg: &LightningIndexerConfig, rng: &mut Philox) -> Self {
        let degenerate = || LightningIndexerWeights {
            wq: Vec::new(),
            wk: Vec::new(),
            weights_proj: Vec::new(),
            gate_proj: Vec::new(),
            conv_weights: Vec::new(),
        };
        if cfg.hidden_size == 0 || cfg.num_heads == 0 || cfg.head_dim == 0 {
            return degenerate();
        }
        let Some(index_dim) = cfg.num_heads.checked_mul(cfg.head_dim) else {
            return degenerate();
        };
        let Some(proj_len) = index_dim.checked_mul(cfg.hidden_size) else {
            return degenerate();
        };
        let Some(weights_len) = cfg.num_heads.checked_mul(index_dim) else {
            return degenerate();
        };
        let Some(gate_len) = cfg.hidden_size.checked_mul(4) else {
            return degenerate();
        };
        let mut w = LightningIndexerWeights {
            wq: vec![0.0; proj_len],
            wk: vec![0.0; proj_len],
            weights_proj: vec![0.0; weights_len],
            gate_proj: vec![0.0; gate_len],
            conv_weights: cfg.conv_kernels.iter().map(|&ks| vec![0.0; ks]).collect(),
        };
        let b_proj = 1.0 / (cfg.hidden_size as f32).sqrt();
        rng.fill_uniform(&mut w.wq, -b_proj, b_proj);
        rng.fill_uniform(&mut w.wk, -b_proj, b_proj);
        let b_w = 1.0 / (index_dim as f32).sqrt();
        rng.fill_uniform(&mut w.weights_proj, -b_w, b_w);
        let b_g = 1.0 / (2.0 * cfg.hidden_size as f32).sqrt();
        rng.fill_uniform(&mut w.gate_proj, -b_g, b_g);
        for (c, &ks) in cfg.conv_kernels.iter().enumerate() {
            if ks == 0 {
                continue; // empty kernel; rejected later by `conv_scores`
            }
            let b = 1.0 / (ks as f32).sqrt();
            rng.fill_uniform(&mut w.conv_weights[c], -b, b);
        }
        w
    }
}

/// Derives the per-dim gated-RMSNorm gain used by the conv branch from the
/// key projection: `gain[r] = ||wk[r, :]||_2 / sqrt(hidden)` (the row RMS).
///
/// Modeling choice (documented): upstream DSA carries a dedicated learnable
/// per-dim `conv_norm_gain` vector for the gated RMSNorm of the cascade-conv
/// features. The SPEC-mandated [`LightningIndexerWeights`] struct has no such
/// field, so this port derives a deterministic per-dim gain from the key
/// projection rows — the very features the conv branch normalizes.
pub(crate) fn conv_norm_gain_from_wk(wk: &[f32], index_dim: usize, hidden: usize) -> Vec<f32> {
    debug_assert_eq!(wk.len(), index_dim * hidden);
    let mut gain = vec![0.0f32; index_dim];
    for r in 0..index_dim {
        let row = &wk[r * hidden..r * hidden + hidden];
        gain[r] = rms_mean_square(row).sqrt();
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_shapes_and_determinism() {
        let cfg = LightningIndexerConfig::v32_default(32);
        let mut a = Philox::new(9, 9);
        let mut b = Philox::new(9, 9);
        let wa = LightningIndexerWeights::from_config(&cfg, &mut a);
        let wb = LightningIndexerWeights::from_config(&cfg, &mut b);
        assert_eq!(wa.wq, wb.wq);
        assert_eq!(wa.wk, wb.wk);
        assert_eq!(wa.weights_proj, wb.weights_proj);
        assert_eq!(wa.gate_proj, wb.gate_proj);
        assert_eq!(wa.conv_weights, wb.conv_weights);

        let d = cfg.num_heads * cfg.head_dim;
        assert_eq!(wa.wq.len(), d * cfg.hidden_size);
        assert_eq!(wa.wk.len(), d * cfg.hidden_size);
        assert_eq!(wa.weights_proj.len(), cfg.num_heads * d);
        assert_eq!(wa.gate_proj.len(), 2 * 2 * cfg.hidden_size);
        assert_eq!(wa.conv_weights.len(), cfg.conv_kernels.len());
        for (c, &ks) in cfg.conv_kernels.iter().enumerate() {
            assert_eq!(wa.conv_weights[c].len(), ks);
        }

        // A different stream produces different weights.
        let mut c = Philox::new(9, 10);
        let wc = LightningIndexerWeights::from_config(&cfg, &mut c);
        assert_ne!(wa.wq, wc.wq);
    }

    #[test]
    fn from_config_degenerate_configs_do_not_panic() {
        let bad = LightningIndexerConfig {
            hidden_size: 0,
            ..LightningIndexerConfig::v32_default(8)
        };
        let w = LightningIndexerWeights::from_config(&bad, &mut Philox::new(1, 1));
        assert!(w.wq.is_empty() && w.wk.is_empty() && w.weights_proj.is_empty());
        let huge = LightningIndexerConfig {
            head_dim: usize::MAX,
            ..LightningIndexerConfig::v32_default(8)
        };
        let w = LightningIndexerWeights::from_config(&huge, &mut Philox::new(1, 1));
        assert!(w.wq.is_empty());
    }

    #[test]
    fn conv_norm_gain_is_row_rms() {
        // wk rows: [3, 4] -> RMS sqrt(12.5); [1, 0] -> RMS sqrt(0.5).
        let wk = [3.0f32, 4.0, 1.0, 0.0];
        let gain = conv_norm_gain_from_wk(&wk, 2, 2);
        assert!((gain[0] - 12.5f32.sqrt()).abs() < 1e-6, "gain[0] = {}", gain[0]);
        assert!((gain[1] - 0.5f32.sqrt()).abs() < 1e-6, "gain[1] = {}", gain[1]);
    }
}
