//! Draft tokens, the [`DraftModel`] trait, and the concrete NEXTN draft
//! network used by DSpark.
//!
//! The V3-paper MTP module at depth 1 (V4.1 NEXTN) proposes the token
//! *after* the last accepted position from the main model's hidden state
//! and the embedding of the accepted token (V3 technical report §2.3).
//! Drafting is greedy: the proposed token is the argmax of the MTP head
//! (ties → lowest id) and its confidence is the softmax probability of
//! that token — the signal DSpark's confidence scheduler thresholds on.

use deepinfer_core::math::{argmax, softmax_inplace};
use deepinfer_core::prop::random_matrix;
use deepinfer_core::rng::Philox;
use deepinfer_core::{ensure, Error, Result};

use crate::config::{MtpConfig, MAX_DRAFT_LEN};
use crate::weights::{clamped_dims, MtpWeights};

/// One proposed token from the MTP draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    /// Proposed token id (argmax of the MTP head, ties → lowest id).
    pub token: u32,
    /// Softmax probability of `token` under the draft logits — DSpark's
    /// confidence signal (in `(0, 1]`).
    pub confidence: f32,
    /// Full draft logits for the position (`[vocab]`).
    pub logits: Vec<f32>,
}

/// A depth-1 MTP (NEXTN) draft model.
pub trait DraftModel {
    /// One draft step: propose the token following `(last_token, hidden)`,
    /// where `hidden` is the main model's hidden state at `last_token`.
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken>;
    /// The draft module's configuration.
    fn config(&self) -> &MtpConfig;
}

/// Concrete NEXTN draft network: [`MtpWeights`] plus the shared token
/// embedding table.
pub struct NextNetDraft {
    config: MtpConfig,
    /// MTP module weights (`eh_proj`, shared block, depth norm, head).
    pub weights: MtpWeights,
    /// Shared embedding table `[vocab, hidden]` (row `t` embeds token `t`).
    pub embedding: Vec<f32>,
}

impl NextNetDraft {
    /// Random initialization for tests/reference runs. Dimensions are
    /// clamped exactly like [`MtpWeights::from_config`] and the stored
    /// [`MtpConfig`] (see [`DraftModel::config`]) reflects the clamped —
    /// i.e. actual — dimensions.
    pub fn from_config(cfg: &MtpConfig, rng: &mut Philox) -> NextNetDraft {
        let (hidden, vocab) = clamped_dims(cfg);
        NextNetDraft {
            config: MtpConfig {
                hidden,
                vocab,
                draft_len: cfg.draft_len.clamp(1, MAX_DRAFT_LEN),
            },
            weights: MtpWeights::from_config(cfg, rng),
            embedding: random_matrix(rng, vocab, hidden, 0.02),
        }
    }
}

impl DraftModel for NextNetDraft {
    fn draft_step(&mut self, last_token: u32, hidden: &[f32]) -> Result<DraftToken> {
        let cfg = self.config();
        ensure!(
            hidden.len() == cfg.hidden,
            DimensionMismatch,
            "draft_step: hidden state length {} does not match the draft config ({})",
            hidden.len(),
            cfg.hidden
        );
        ensure!(
            (last_token as usize) < cfg.vocab,
            OutOfRange,
            "draft_step: token {last_token} is outside the draft vocabulary ({})",
            cfg.vocab
        );
        let product = cfg.vocab.checked_mul(cfg.hidden).ok_or_else(|| {
            Error::DimensionMismatch("draft_step: vocab*hidden overflows usize".into())
        })?;
        ensure!(
            self.embedding.len() == product,
            DimensionMismatch,
            "draft_step: embedding has {} floats, expected vocab*hidden = {product}",
            self.embedding.len()
        );

        let start = last_token as usize * cfg.hidden;
        let emb = &self.embedding[start..start + cfg.hidden];
        let logits = self.weights.forward_logits(hidden, emb)?;
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            Numerical,
            "draft_step: MTP head produced non-finite logits"
        );
        let idx = argmax(&logits)
            .ok_or_else(|| Error::Internal("draft_step: MTP head produced empty logits".into()))?;
        let mut probs = logits.clone();
        softmax_inplace(&mut probs);
        Ok(DraftToken {
            token: idx as u32,
            confidence: probs[idx],
            logits,
        })
    }

    fn config(&self) -> &MtpConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> MtpConfig {
        MtpConfig {
            hidden: 8,
            vocab: 12,
            draft_len: 4,
        }
    }

    #[test]
    fn draft_step_deterministic_and_valid() {
        let mut draft = NextNetDraft::from_config(&tiny_cfg(), &mut Philox::new(0xD2A2, 7));
        let h: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 - 1.05).collect();

        let a = draft.draft_step(3, &h).unwrap();
        let b = draft.draft_step(3, &h).unwrap();
        assert_eq!(a.token, b.token);
        assert_eq!(a.confidence, b.confidence);
        assert_eq!(a.logits, b.logits, "draft logits must be bit-deterministic");

        assert_eq!(a.logits.len(), 12);
        assert!((a.token as usize) < 12);
        assert!(a.logits.iter().all(|v| v.is_finite()));
        assert!(a.confidence.is_finite() && a.confidence > 0.0 && a.confidence <= 1.0 + 1e-5);

        // token == argmax (ties -> lowest); confidence == its softmax mass.
        let am = argmax(&a.logits).unwrap();
        assert_eq!(a.token as usize, am);
        let mut p = a.logits.clone();
        softmax_inplace(&mut p);
        assert!((p[am] - a.confidence).abs() <= 1e-6);

        // A chained draft loop stays valid.
        let mut last = a.token;
        for _ in 0..4 {
            let d = draft.draft_step(last, &h).unwrap();
            assert!((d.token as usize) < 12);
            assert!(d.confidence > 0.0);
            last = d.token;
        }
    }

    #[test]
    fn draft_step_validates_inputs() {
        let mut draft = NextNetDraft::from_config(&tiny_cfg(), &mut Philox::new(1, 1));
        let h = vec![0.1f32; 8];
        assert!(draft.draft_step(12, &h).is_err(), "token == vocab");
        assert!(draft.draft_step(u32::MAX, &h).is_err(), "token overflow");
        assert!(draft.draft_step(0, &h[..7]).is_err(), "hidden too short");
        assert!(draft.draft_step(0, &[]).is_err(), "empty hidden");
        assert_eq!(draft.config().hidden, 8);
        assert_eq!(draft.config().vocab, 12);
        assert_eq!(draft.config().draft_len, 4);
    }

    #[test]
    fn from_config_clamps_and_stores_actual_dims() {
        let cfg = MtpConfig {
            hidden: 0,
            vocab: 5,
            draft_len: 0,
        };
        let mut d = NextNetDraft::from_config(&cfg, &mut Philox::new(4, 4));
        assert_eq!(d.config().hidden, 1);
        assert_eq!(d.config().vocab, 5);
        assert_eq!(d.config().draft_len, 1);
        assert_eq!(d.embedding.len(), 5);
        let tok = d.draft_step(2, &[0.5]).unwrap();
        assert!((tok.token as usize) < 5);
    }

    #[test]
    fn prop_draft_step_invariants() {
        use deepinfer_core::prop::prop;
        prop("mtp-draft-invariants", 0xD2A7, 120, |i, rng| {
            let hidden = 4 + (i as usize % 5);
            let vocab = 4 + (i as usize % 7);
            let cfg = MtpConfig {
                hidden,
                vocab,
                draft_len: 3,
            };
            let mut draft = NextNetDraft::from_config(&cfg, rng);
            let h: Vec<f32> = (0..hidden).map(|_| rng.normal_f32()).collect();
            let tok = rng.next_u32() % vocab as u32;

            let d1 = draft.draft_step(tok, &h)?;
            let d2 = draft.draft_step(tok, &h)?;
            ensure!(
                d1.token == d2.token && d1.confidence == d2.confidence,
                Internal,
                "draft_step not deterministic"
            );
            ensure!(d1.logits == d2.logits, Internal, "draft logits not deterministic");
            ensure!(d1.logits.len() == vocab, Internal, "logits length");
            ensure!((d1.token as usize) < vocab, Internal, "token outside vocab");
            ensure!(
                d1.logits.iter().all(|v| v.is_finite()),
                Internal,
                "non-finite logits"
            );
            ensure!(
                d1.confidence.is_finite() && d1.confidence > 0.0 && d1.confidence <= 1.0 + 1e-5,
                Internal,
                "confidence {} out of (0, 1]",
                d1.confidence
            );
            let am = argmax(&d1.logits).unwrap_or(0);
            ensure!(d1.token as usize == am, Internal, "token != argmax");
            let mut p = d1.logits.clone();
            softmax_inplace(&mut p);
            ensure!(
                (p[am] - d1.confidence).abs() <= 1e-6,
                Internal,
                "confidence != softmax mass"
            );
            Ok(())
        });
    }
}
