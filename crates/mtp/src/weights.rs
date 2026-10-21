//! MTP module weights: the DeepSeek-V3 Multi-Token-Prediction module at
//! depth 1 (the V4.1 "NEXTN" layer) plus the shared transformer block it
//! contains.
//!
//! Reference: DeepSeek-V3 technical report §2.3 (Multi-Token Prediction).
//! The depth-`d` module concatenates the main model's hidden state `h`
//! with the embedding of the next token, projects through `eh_proj`,
//! applies a shared RMSNorm, feeds one transformer block (shared across
//! depths in the paper; with depth 1 there is exactly one), and an output
//! head predicts the token at the following position. V4.1-Flash ships
//! only the first MTP depth, exposed as the NEXTN decode layer (DeepInfer
//! env `DEEPINFER_USE_NEXTN_DECODE`).
//!
//! Fidelity note (CPU reference): the draft context of the depth-1 module
//! is the single hidden vector of the last accepted position, so the
//! shared block runs on a sequence of length 1, where single-head
//! `softmax(QKᵀ/√d)` attention degenerates exactly to the value path.
//! The full pre-norm block structure (attention + SwiGLU MLP) is
//! implemented so the module stays structurally complete.

use deepinfer_core::math::{axpy, dot, softmax_inplace};
use deepinfer_core::norm::rmsnorm;
use deepinfer_core::prop::random_matrix;
use deepinfer_core::rng::Philox;
use deepinfer_core::{ensure, Result};

use crate::config::MtpConfig;

/// RMSNorm epsilon used throughout the MTP module (V4.1 `rms_eps`).
pub const MTP_RMS_EPS: f32 = 1e-4;
/// Hard bound on `hidden` accepted by the random initializer (DoS guard:
/// the block + `eh_proj` hold ~9·`hidden`² floats).
pub const MAX_INIT_HIDDEN: usize = 1024;
/// Hard bound on `vocab` accepted by the random initializer (DoS guard).
pub const MAX_INIT_VOCAB: usize = 1 << 21;
/// Hard bound on the number of floats of any single randomly initialized
/// matrix (DoS guard for `eh_proj` / `head` / the embedding table).
pub const MAX_INIT_MATRIX_FLOATS: usize = 1 << 22;

/// Weights of the single shared transformer block inside the MTP module.
///
/// Layout — row-major, every matrix multiplies a column vector
/// (`out = W @ x`, i.e. `out[r] = Σ_j W[r][j] · x[j]`):
///
/// | field | shape | role |
/// |---|---|---|
/// | `attn_norm` | `[hidden]` | pre-attention RMSNorm gain |
/// | `wq`/`wk`/`wv`/`wo` | `[hidden, hidden]` | single-head Q/K/V/O projections |
/// | `mlp_norm` | `[hidden]` | pre-MLP RMSNorm gain |
/// | `w_gate_up` | `[2·intermediate, hidden]` | SwiGLU gate (rows `0..intermediate`) and up (rows `intermediate..`) |
/// | `w_down` | `[hidden, intermediate]` | MLP down projection |
///
/// `intermediate` is the SwiGLU inner width; [`MtpWeights::from_config`]
/// sets it to `hidden` (a reference choice — the production checkpoint
/// uses the main model's FFN width).
pub struct MtpBlockWeights {
    /// Pre-attention RMSNorm gain `[hidden]`.
    pub attn_norm: Vec<f32>,
    /// Single-head query projection `[hidden, hidden]`.
    pub wq: Vec<f32>,
    /// Single-head key projection `[hidden, hidden]`.
    pub wk: Vec<f32>,
    /// Single-head value projection `[hidden, hidden]`.
    pub wv: Vec<f32>,
    /// Attention output projection `[hidden, hidden]`.
    pub wo: Vec<f32>,
    /// Pre-MLP RMSNorm gain `[hidden]`.
    pub mlp_norm: Vec<f32>,
    /// SwiGLU gate+up projection `[2 * intermediate, hidden]`.
    pub w_gate_up: Vec<f32>,
    /// MLP down projection `[hidden, intermediate]`.
    pub w_down: Vec<f32>,
}

impl MtpBlockWeights {
    /// Random initialization for tests/reference runs (matrix entries
    /// uniform in `[-0.02, 0.02]`, unit norm gains).
    pub(crate) fn random(hidden: usize, rng: &mut Philox) -> MtpBlockWeights {
        MtpBlockWeights {
            attn_norm: vec![1.0; hidden],
            wq: random_matrix(rng, hidden, hidden, 0.02),
            wk: random_matrix(rng, hidden, hidden, 0.02),
            wv: random_matrix(rng, hidden, hidden, 0.02),
            wo: random_matrix(rng, hidden, hidden, 0.02),
            mlp_norm: vec![1.0; hidden],
            w_gate_up: random_matrix(rng, 2 * hidden, hidden, 0.02),
            w_down: random_matrix(rng, hidden, hidden, 0.02),
        }
    }

    /// Validates field shapes against a hidden width of `h` and returns
    /// the SwiGLU intermediate width implied by `w_down`.
    pub(crate) fn validate_shapes(&self, h: usize) -> Result<usize> {
        ensure!(h >= 1, InvalidArgument, "block: hidden must be >= 1");
        ensure!(
            self.attn_norm.len() == h,
            DimensionMismatch,
            "block: attn_norm length {} != hidden {h}",
            self.attn_norm.len()
        );
        ensure!(
            self.mlp_norm.len() == h,
            DimensionMismatch,
            "block: mlp_norm length {} != hidden {h}",
            self.mlp_norm.len()
        );
        for (name, m) in [
            ("wq", &self.wq),
            ("wk", &self.wk),
            ("wv", &self.wv),
            ("wo", &self.wo),
        ] {
            ensure!(
                (m.len() as u64) == h as u64 * h as u64,
                DimensionMismatch,
                "block: {name} has {} floats, expected {h}x{h}",
                m.len()
            );
        }
        ensure!(
            !self.w_down.is_empty() && self.w_down.len() % h == 0,
            DimensionMismatch,
            "block: w_down has {} floats, not a positive multiple of hidden {h}",
            self.w_down.len()
        );
        let gi = self.w_down.len() / h;
        ensure!(
            (self.w_gate_up.len() as u64) == 2u64 * gi as u64 * h as u64,
            DimensionMismatch,
            "block: w_gate_up has {} floats, expected {}x{h}",
            self.w_gate_up.len(),
            2 * gi
        );
        Ok(gi)
    }

    /// Pre-norm transformer block forward over the length-1 draft context
    /// `x` (`[hidden]`), returning the residual stream (`[hidden]`).
    ///
    /// `x = x + Wo(softmax(QKᵀ/√d) · V)` then `x = x + Wd(silu(Wg x) ⊙
    /// (Wu x))`. Single head over the draft context: on this CPU reference
    /// the context is the single hidden vector (sequence length 1), so
    /// `QKᵀ/√d` is a 1×1 score matrix and its softmax is exactly `[1.0]` —
    /// attention degenerates to the value path. The score/softmax path is
    /// spelled out so the block remains a structurally complete pre-norm
    /// transformer block.
    pub(crate) fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let h = x.len();
        let gi = self.validate_shapes(h)?;

        // Attention sub-block (pre-norm, single head).
        let mut normed = x.to_vec();
        rmsnorm(&mut normed, &self.attn_norm, MTP_RMS_EPS);
        let q = matvec(&self.wq, &normed, h, h)?;
        let k = matvec(&self.wk, &normed, h, h)?;
        let v = matvec(&self.wv, &normed, h, h)?;
        let mut scores = vec![dot(&q, &k) / (h as f32).sqrt()];
        softmax_inplace(&mut scores);
        let mut attn = vec![0.0f32; h];
        axpy(scores[0], &v, &mut attn);
        let attn_proj = matvec(&self.wo, &attn, h, h)?;
        let mut y = x.to_vec();
        axpy(1.0, &attn_proj, &mut y);

        // MLP sub-block (pre-norm SwiGLU).
        let mut normed2 = y.clone();
        rmsnorm(&mut normed2, &self.mlp_norm, MTP_RMS_EPS);
        let gate_up = matvec(&self.w_gate_up, &normed2, 2 * gi, h)?;
        let mut act = vec![0.0f32; gi];
        for i in 0..gi {
            act[i] = silu(gate_up[i]) * gate_up[gi + i];
        }
        let mlp_proj = matvec(&self.w_down, &act, h, gi)?;
        axpy(1.0, &mlp_proj, &mut y);
        Ok(y)
    }
}

/// V3-paper MTP module weights (depth 1 = V4.1 NEXTN).
///
/// Forward semantics (V3 technical report §2.3):
///
/// ```text
/// h'     = rmsnorm_shared(eh_proj @ [h ; emb(t_next)])
/// x      = block(h')                        // one shared pre-norm block
/// logits = head @ rmsnorm_depth(x)
/// ```
pub struct MtpWeights {
    /// Concatenation projection `[hidden, 2 * hidden]` acting on
    /// `[h ; emb(t_next)]` (hidden state first, embedding second).
    pub eh_proj: Vec<f32>,
    /// Shared RMSNorm gain applied after `eh_proj` `[hidden]`.
    pub shared_norm: Vec<f32>,
    /// Depth-output RMSNorm gain applied before the head `[hidden]`.
    pub depth_norm: Vec<f32>,
    /// The single shared transformer block.
    pub block: MtpBlockWeights,
    /// Output head `[vocab, hidden]`.
    pub head: Vec<f32>,
}

impl MtpWeights {
    /// Random initialization for tests and small reference models.
    ///
    /// `cfg` is clamped into the bounds documented on
    /// [`MAX_INIT_HIDDEN`] / [`MAX_INIT_VOCAB`] /
    /// [`MAX_INIT_MATRIX_FLOATS`] before any allocation, so this
    /// constructor is safe on untrusted configurations. Production-sized
    /// modules (hidden 7168, vocab ~129k ⇒ tens of GB of random weights)
    /// must come from a checkpoint instead. The caller should validate
    /// `cfg` first ([`MtpConfig::validate`]); clamping only guards
    /// allocation — a clamped module refuses mismatched inputs at
    /// [`MtpWeights::forward_logits`] with a shape error.
    pub fn from_config(cfg: &MtpConfig, rng: &mut Philox) -> MtpWeights {
        let (hidden, vocab) = clamped_dims(cfg);
        MtpWeights {
            eh_proj: random_matrix(rng, hidden, 2 * hidden, 0.02),
            shared_norm: vec![1.0; hidden],
            depth_norm: vec![1.0; hidden],
            block: MtpBlockWeights::random(hidden, rng),
            head: random_matrix(rng, vocab, hidden, 0.02),
        }
    }

    /// Runs the MTP module. `hidden_state` is the main model's hidden
    /// vector for the last accepted position; `token_emb` is the embedding
    /// of the next (draft) token. Returns the `[vocab]` logits for the
    /// position after that token.
    ///
    /// All shapes are validated against `hidden_state.len()`, which
    /// defines the hidden width the weights are interpreted with.
    pub fn forward_logits(&self, hidden_state: &[f32], token_emb: &[f32]) -> Result<Vec<f32>> {
        let h = hidden_state.len();
        ensure!(
            h >= 1,
            InvalidArgument,
            "forward_logits: hidden_state must be non-empty"
        );
        ensure!(
            token_emb.len() == h,
            DimensionMismatch,
            "forward_logits: token embedding length {} does not match hidden length {h}",
            token_emb.len()
        );
        let eh_len = 2u64 * h as u64 * h as u64;
        ensure!(
            self.eh_proj.len() as u64 == eh_len,
            DimensionMismatch,
            "forward_logits: eh_proj has {} floats, expected [hidden, 2*{h}] = {eh_len}",
            self.eh_proj.len()
        );
        ensure!(
            self.shared_norm.len() == h,
            DimensionMismatch,
            "forward_logits: shared_norm length {} != hidden {h}",
            self.shared_norm.len()
        );
        ensure!(
            self.depth_norm.len() == h,
            DimensionMismatch,
            "forward_logits: depth_norm length {} != hidden {h}",
            self.depth_norm.len()
        );
        ensure!(
            !self.head.is_empty() && self.head.len() % h == 0,
            DimensionMismatch,
            "forward_logits: head has {} floats, not a positive multiple of hidden {h}",
            self.head.len()
        );
        let vocab = self.head.len() / h;
        ensure!(
            vocab <= u32::MAX as usize,
            CapacityExceeded,
            "forward_logits: vocabulary {vocab} exceeds the u32 token-id space"
        );

        // [h ; emb]
        let mut concat = Vec::with_capacity(2 * h);
        concat.extend_from_slice(hidden_state);
        concat.extend_from_slice(token_emb);
        let mut hprime = matvec(&self.eh_proj, &concat, h, 2 * h)?;
        rmsnorm(&mut hprime, &self.shared_norm, MTP_RMS_EPS);
        let x = self.block.forward(&hprime)?;
        let mut hdepth = x;
        rmsnorm(&mut hdepth, &self.depth_norm, MTP_RMS_EPS);
        matvec(&self.head, &hdepth, vocab, h)
    }
}

/// Dimensions used by the random initializer after clamping (DoS guard).
pub(crate) fn clamped_dims(cfg: &MtpConfig) -> (usize, usize) {
    let hidden = cfg.hidden.clamp(1, MAX_INIT_HIDDEN);
    let vocab_by_matrix = (MAX_INIT_MATRIX_FLOATS / hidden).max(1);
    let vocab = cfg.vocab.clamp(1, MAX_INIT_VOCAB).min(vocab_by_matrix);
    (hidden, vocab)
}

/// Row-major matrix–vector product: `out[r] = Σ_j w[r * cols + j] · x[j]`
/// with sequential `f32` accumulation (fixed order ⇒ deterministic).
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    ensure!(
        (w.len() as u64) == rows as u64 * cols as u64,
        DimensionMismatch,
        "matvec: weight buffer has {} floats, expected {rows}x{cols}",
        w.len()
    );
    ensure!(
        x.len() == cols,
        DimensionMismatch,
        "matvec: input length {} does not match column count {cols}",
        x.len()
    );
    let mut out = vec![0.0f32; rows];
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
    Ok(out)
}

/// SiLU activation (`x · sigmoid(x)`), used by the block's SwiGLU MLP.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_init_shapes() {
        let cfg = MtpConfig {
            hidden: 8,
            vocab: 16,
            draft_len: 5,
        };
        let w = MtpWeights::from_config(&cfg, &mut Philox::new(1, 2));
        assert_eq!(w.eh_proj.len(), 2 * 8 * 8);
        assert_eq!(w.shared_norm.len(), 8);
        assert_eq!(w.depth_norm.len(), 8);
        assert_eq!(w.head.len(), 16 * 8);
        let b = &w.block;
        assert_eq!(b.attn_norm.len(), 8);
        assert_eq!(b.wq.len(), 64);
        assert_eq!(b.wk.len(), 64);
        assert_eq!(b.wv.len(), 64);
        assert_eq!(b.wo.len(), 64);
        assert_eq!(b.mlp_norm.len(), 8);
        assert_eq!(b.w_gate_up.len(), 2 * 8 * 8);
        assert_eq!(b.w_down.len(), 8 * 8);
        assert!(w.shared_norm.iter().all(|&v| v == 1.0));
        assert!(w.depth_norm.iter().all(|&v| v == 1.0));
        assert!(b.attn_norm.iter().all(|&v| v == 1.0));
        assert!(b.mlp_norm.iter().all(|&v| v == 1.0));
        // intermediate = hidden in the reference initializer
        assert_eq!(b.validate_shapes(8).unwrap(), 8);
    }

    #[test]
    fn from_config_is_deterministic() {
        let cfg = MtpConfig {
            hidden: 6,
            vocab: 9,
            draft_len: 4,
        };
        let w1 = MtpWeights::from_config(&cfg, &mut Philox::new(77, 1));
        let w2 = MtpWeights::from_config(&cfg, &mut Philox::new(77, 1));
        assert_eq!(w1.eh_proj, w2.eh_proj);
        assert_eq!(w1.shared_norm, w2.shared_norm);
        assert_eq!(w1.head, w2.head);
        assert_eq!(w1.block.wq, w2.block.wq);
        assert_eq!(w1.block.w_gate_up, w2.block.w_gate_up);
        assert_eq!(w1.block.w_down, w2.block.w_down);
    }

    #[test]
    fn forward_logits_shapes_finite_deterministic() {
        let cfg = MtpConfig {
            hidden: 8,
            vocab: 12,
            draft_len: 5,
        };
        let w = MtpWeights::from_config(&cfg, &mut Philox::new(3, 4));
        let h: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.75).collect();
        let e: Vec<f32> = (0..8).map(|i| ((i * 7 + 3) as f32) * 0.11 - 0.4).collect();

        let l1 = w.forward_logits(&h, &e).unwrap();
        let l2 = w.forward_logits(&h, &e).unwrap();
        assert_eq!(l1, l2, "forward_logits must be bit-deterministic");
        assert_eq!(l1.len(), 12);
        assert!(l1.iter().all(|v| v.is_finite()));

        assert!(w.forward_logits(&h[..7], &e).is_err(), "hidden mismatch");
        assert!(w.forward_logits(&h, &e[..7]).is_err(), "embedding mismatch");
        assert!(w.forward_logits(&[], &[]).is_err(), "empty inputs");
    }

    #[test]
    fn zero_eh_proj_gives_zero_logits() {
        // rmsnorm(0) = 0 and the block is residual, so a zero eh_proj must
        // produce an exactly-zero logit vector.
        let cfg = MtpConfig {
            hidden: 4,
            vocab: 6,
            draft_len: 2,
        };
        let mut w = MtpWeights::from_config(&cfg, &mut Philox::new(5, 5));
        for v in w.eh_proj.iter_mut() {
            *v = 0.0;
        }
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let e = vec![0.5f32, -0.5, 0.25, -0.25];
        let logits = w.forward_logits(&h, &e).unwrap();
        assert!(logits.iter().all(|v| *v == 0.0), "{logits:?}");
    }

    #[test]
    fn forward_composition_identity_reference() {
        // eh_proj = [I | 0] extracts the hidden state from [h ; emb];
        // zero attention/MLP matrices make the block the identity (pure
        // residuals); unit norms leave plain RMSNorm. The head rows then
        // read h'' directly — an independent check of the concat order,
        // the two norms, the block residual structure and the head.
        let hidden = 2usize;
        let vocab = 3usize;
        let cfg = MtpConfig {
            hidden,
            vocab,
            draft_len: 2,
        };
        let mut w = MtpWeights::from_config(&cfg, &mut Philox::new(9, 9));
        w.eh_proj = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        w.shared_norm = vec![1.0; hidden];
        w.depth_norm = vec![1.0; hidden];
        w.block.attn_norm = vec![1.0; hidden];
        w.block.mlp_norm = vec![1.0; hidden];
        for m in [
            &mut w.block.wq,
            &mut w.block.wk,
            &mut w.block.wv,
            &mut w.block.wo,
            &mut w.block.w_gate_up,
            &mut w.block.w_down,
        ] {
            for v in m.iter_mut() {
                *v = 0.0;
            }
        }
        w.head = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let h = vec![3.0f32, 4.0];
        let emb = vec![5.0f32, 6.0];
        let logits = w.forward_logits(&h, &emb).unwrap();

        let ones = vec![1.0f32; hidden];
        let mut expect = h.clone();
        rmsnorm(&mut expect, &ones, MTP_RMS_EPS);
        rmsnorm(&mut expect, &ones, MTP_RMS_EPS);
        assert_eq!(logits, vec![expect[0], expect[1], expect[0] + expect[1]]);
    }

    #[test]
    fn from_config_clamps_untrusted_dimensions() {
        // A huge vocabulary is clamped so the head stays bounded (DoS guard).
        let cfg = MtpConfig {
            hidden: 4,
            vocab: u32::MAX as usize,
            draft_len: 2,
        };
        let w = MtpWeights::from_config(&cfg, &mut Philox::new(2, 2));
        let expect_vocab = MAX_INIT_MATRIX_FLOATS / 4;
        assert_eq!(w.head.len(), expect_vocab * 4);
        assert!(w.head.len() <= MAX_INIT_MATRIX_FLOATS);

        // Degenerate dimensions clamp to 1 — no panic, consistent shapes.
        let cfg = MtpConfig {
            hidden: 0,
            vocab: 3,
            draft_len: 1,
        };
        let w = MtpWeights::from_config(&cfg, &mut Philox::new(3, 3));
        assert_eq!(w.eh_proj.len(), 2); // [1, 2]
        assert_eq!(w.head.len(), 3); // [3, 1]
        let logits = w.forward_logits(&[0.5], &[0.25]).unwrap();
        assert_eq!(logits.len(), 3);
    }
}
