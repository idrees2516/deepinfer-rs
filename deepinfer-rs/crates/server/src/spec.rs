//! DSpark speculative-decoding rounds for the **batched serve path**.
//!
//! The offline `generate` loop drafts with the MTP (NEXTN) head, verifies
//! the chain against the target and rolls rejected positions back out of
//! the caches — one sequence at a time. This module lifts the same
//! algorithm into the continuous-batching engine:
//!
//! * Each decoding step, every spec-eligible request drafts a chain of
//!   up to `spec_len` tokens with the DSpark confidence scheduler
//!   ([`deepinfer_mtp::DsparkScheduler`]) from its *anchor* — the last
//!   committed token with the target logits and residual hidden state at
//!   that position ([`SpecAnchor`]).
//! * The chain is verified against the target in one
//!   [`crate::engine::TokenEngine::verify`] call (the CPU golden engine
//!   runs the parity-faithful sequential forwards; the GPU engine makes
//!   it a single chunked multi-row forward).
//! * The accepted prefix plus the bonus token from the verification row
//!   are emitted as this step's tokens; the **rejected tail is rolled
//!   back** out of the request's cache rows
//!   ([`crate::engine::TokenEngine::rollback`]) — under batching, one
//!   request's rollback never touches another request's rows (the paged
//!   caches are per-sequence).
//! * The bonus token's decode (which refreshes the anchor) is queued and
//!   executed in the step's single **batched decode** together with the
//!   plain-decode requests — on the GPU engine that is one
//!   `serve_decode` launch for the whole batch.
//!
//! Grammar-aware drafting and verification (beyond the offline path,
//! which disables speculation under a grammar): the draft head's argmax
//! is masked with the DFA mask at each draft position, the target rows
//! are masked with the same states, and the DFA state is snapshotted per
//! position so acceptance advances it and rejection restores it. The
//! masked-argmax verification rule preserves the plain-path invariant:
//! **greedy spec output is token-identical to greedy plain output under
//! a grammar** (an invalid draft can never equal the masked target
//! argmax, so it is always corrected).
//!
//! Stochastic requests verify with `verify_stochastic` (temperature-only
//! categorical resampling, no nucleus filtering) — the documented DSpark
//! verification contract, same as the offline path.

use deepinfer_core::mask::TokenMask;
use deepinfer_core::math::softmax_inplace;
use deepinfer_grammar::{Grammar, TokenMaskComputer};

/// The speculative anchor of one request: the last committed token with
/// the target's next-token logits and residual hidden state at that
/// position. Established when the prompt finishes prefilling and
/// refreshed by every round's bonus decode.
#[derive(Debug, Clone)]
pub(crate) struct SpecAnchor {
    /// The last committed (emitted) token.
    pub tok: u32,
    /// Target logits **at** `tok` (predict the token after it) — the
    /// verification row 0.
    pub logits: Vec<f32>,
    /// Residual hidden state at `tok` — the MTP draft input.
    pub hidden: Vec<f32>,
}

/// Grammar mask at `state`, with the EOS-clearing rule the plain sampler
/// applies (`EOS` is only legal at accepting states).
pub(crate) fn state_mask(
    masker: &mut TokenMaskComputer,
    grammar: &Grammar,
    state: u32,
    eos: u32,
) -> TokenMask {
    let mut mk = masker.compute_mask(grammar, state);
    if !grammar.is_accepting(state) {
        mk.clear(eos);
    }
    mk
}

/// Grammar-masked greedy pick over `logits`: the argmax among allowed
/// tokens (ties to the lowest id — `iter_ones` ascends), with the
/// confidence defined as the token's softmax mass over the **full**
/// distribution — DSpark's "how sure is the draft head" signal, unchanged
/// by the external grammar constraint. `None` when no token is allowed
/// (a dead-end state) or every allowed logit is non-finite.
pub(crate) fn masked_pick(logits: &[f32], mask: &TokenMask) -> Option<(u32, f32)> {
    let mut best_tok: Option<u32> = None;
    let mut best_val = f32::NEG_INFINITY;
    for t in mask.iter_ones() {
        let v = logits.get(t as usize).copied().unwrap_or(f32::NEG_INFINITY);
        let v = if v.is_finite() { v } else { f32::NEG_INFINITY };
        if best_tok.is_none() || v > best_val {
            best_tok = Some(t);
            best_val = v;
        }
    }
    let tok = best_tok?;
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let idx = tok as usize;
    let conf = probs.get(idx).copied().unwrap_or(0.0);
    Some((tok, conf))
}

/// Applies `mask` to one verification row in place: disallowed logits
/// become `-inf` (the verifier's `greedy_pick`/sampler treat non-finite
/// entries as never-winning). Only called for masks with at least one
/// allowed token.
pub(crate) fn mask_row(row: &mut [f32], mask: &TokenMask) {
    for (t, v) in row.iter_mut().enumerate() {
        if !mask.test(t as u32) {
            *v = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(pairs: &[&str]) -> Vec<Vec<u8>> {
        pairs.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    #[test]
    fn masked_pick_argmax_ties_and_confidence() {
        let mask = {
            let mut m = TokenMask::zeros(6);
            for t in [0u32, 2, 3, 5] {
                m.set(t);
            }
            m
        };
        // Logits: token 1 highest overall but masked out; 2 and 3 tie
        // among allowed (lowest id wins); 5 low.
        let logits = vec![0.1, 9.0, 4.0, 4.0, 8.0, -1.0];
        let (tok, conf) = masked_pick(&logits, &mask).unwrap();
        assert_eq!(tok, 2, "ties resolve to the lowest allowed id");
        let mut p = logits.clone();
        softmax_inplace(&mut p);
        assert!((conf - p[2]).abs() < 1e-6, "confidence = full-softmax mass");

        // Non-finite allowed logits never win.
        let bad = vec![f32::NAN, 9.0, f32::INFINITY, 1.0, 0.0, 0.5];
        let (tok, _) = masked_pick(&bad, &mask).unwrap();
        assert_eq!(tok, 3);

        // Empty mask -> None.
        assert!(masked_pick(&logits, &TokenMask::zeros(6)).is_none());
    }

    #[test]
    fn mask_row_blanks_disallowed() {
        let mut mask = TokenMask::zeros(4);
        mask.set(1);
        mask.set(3);
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        mask_row(&mut row, &mask);
        assert_eq!(row, vec![f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 4.0]);
    }

    #[test]
    fn state_mask_clears_eos_at_non_accepting_states() {
        // Grammar: one or more digits.
        let g = Grammar::compile_regex("[0-9]+").unwrap();
        let mut mc = TokenMaskComputer::new(vocab(&["1", ""]));
        let s0 = g.initial_state();
        let m0 = state_mask(&mut mc, &g, s0, 1);
        assert!(!m0.test(1), "EOS cleared before an accepting state");
        let s1 = mc.advance_token(&g, s0, 0).unwrap();
        let m1 = state_mask(&mut mc, &g, s1, 1);
        assert!(m1.test(1), "EOS allowed at an accepting state");
        assert!(m1.test(0));
    }
}
