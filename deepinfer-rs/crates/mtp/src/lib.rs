//! # deepinfer-mtp
//!
//! DSpark speculative decoding for DeepInfer-rs: the DeepSeek-V3
//! Multi-Token-Prediction (MTP) draft module — V4.1-Flash ships exactly
//! the first depth as its "NEXTN" decode layer (V3 technical report §2.3;
//! DeepInfer env `DEEPINFER_USE_NEXTN_DECODE`, default 5) — plus
//! DeepInfer's DSpark confidence-scheduled verification and the
//! backtracking bookkeeping the runtime needs to roll rejected draft
//! positions out of the KV/SWA caches.
//!
//! Components:
//! * [`MtpConfig`] — env-derived configuration.
//! * [`MtpWeights`] / [`MtpBlockWeights`] — the MTP module:
//!   `h' = rmsnorm_shared(eh_proj @ [h ; emb(t_next)])`, one shared
//!   pre-norm transformer block, `logits = head @ rmsnorm_depth(x)`.
//! * [`DraftModel`] / [`NextNetDraft`] / [`DraftToken`] — greedy drafting
//!   with per-token confidence (softmax mass of the argmax token).
//! * [`DsparkScheduler`] — when to verify: draft budget exhausted,
//!   uncertain draft (any confidence < `conf_lo`), or confidently-done
//!   short chain (mean confidence >= `conf_hi`).
//! * [`verify_greedy`] / [`verify_stochastic`] / [`backtrack_steps`] —
//!   prefix acceptance against the target model and rollback math.
//!
//! Fidelity notes (CPU reference): the draft context is the single hidden
//! vector of the last accepted position, so the shared block runs on a
//! sequence of length 1 (single-head attention degenerates to the value
//! path — see [`weights`]); stochastic verification resamples each
//! position from the temperature-scaled target distribution with **no**
//! nucleus filtering, per the DSpark verification contract.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod draft;
pub mod verify;
pub mod weights;

pub use config::MtpConfig;
pub use draft::{DraftModel, DraftToken, NextNetDraft};
pub use verify::{
    backtrack_steps, verify_greedy, verify_stochastic, Decision, DsparkScheduler,
    VerificationOutcome,
};
pub use weights::{MtpBlockWeights, MtpWeights};

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::rng::Philox;

    /// One full DSpark round: draft under the confidence scheduler, score
    /// a synthetic target, verify greedily, check the rollback math.
    #[test]
    fn dspark_round_trip() {
        let cfg = MtpConfig {
            hidden: 8,
            vocab: 12,
            draft_len: 4,
        };
        let mut rng = Philox::new(0xD5A2C, 1);
        let mut draft = NextNetDraft::from_config(&cfg, &mut rng);
        let sched = DsparkScheduler::new(cfg.draft_len);
        let hidden: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25 - 1.0).collect();

        // Draft loop: continue until the scheduler says verify.
        let mut drafts: Vec<DraftToken> = Vec::new();
        let mut last = 3u32;
        loop {
            if sched.should_verify(&drafts) == Decision::VerifyNow {
                break;
            }
            assert!(
                drafts.len() < cfg.draft_len,
                "scheduler never asked to verify"
            );
            let d = draft.draft_step(last, &hidden).unwrap();
            last = d.token;
            drafts.push(d);
        }
        assert!(!drafts.is_empty());
        assert_eq!(sched.verify_positions(&drafts), drafts.len() + 1);
        for d in &drafts {
            assert!((d.token as usize) < cfg.vocab);
            assert!(d.confidence > 0.0 && d.confidence <= 1.0 + 1e-5);
        }

        // Synthetic target that accepts everything and continues with 7.
        let mut rows: Vec<Vec<f32>> = drafts.iter().map(|d| spike(d.token, cfg.vocab)).collect();
        rows.push(spike(7, cfg.vocab));
        let out = verify_greedy(&drafts, &rows).unwrap();
        assert_eq!(out.accepted, drafts.len());
        assert_eq!(out.next_token, 7);
        assert_eq!(backtrack_steps(&drafts, &out), 0);

        // Synthetic target that rejects at the first position.
        let mut rows: Vec<Vec<f32>> = drafts.iter().map(|_| spike(0, cfg.vocab)).collect();
        rows.push(spike(7, cfg.vocab));
        let out = verify_greedy(&drafts, &rows).unwrap();
        assert_eq!(out.accepted, 0);
        assert_eq!(out.next_token, 0);
        assert_eq!(backtrack_steps(&drafts, &out), drafts.len());
    }

    fn spike(tok: u32, vocab: usize) -> Vec<f32> {
        let mut r = vec![0.0f32; vocab];
        r[tok as usize] = 40.0;
        r
    }
}
