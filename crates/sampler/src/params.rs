//! Sampling parameters — the DeepInfer `SamplingParams` contract.
//!
//! Mirrors the parameter set of DeepInfer's deterministic sampling layer
//! (upstream README: "deterministic sampling" / "batch-invariant
//! inference"): vLLM-style `temperature` / `top_p` / `top_k` / `min_p`
//! filters plus a per-sequence `seq_seed` that seeds a Philox stream keyed
//! by the sequence id (see [`crate::pipeline::Sampler::seq_rng`]).
//! `so_args` carries a structured-output constraint; the grammar itself
//! lives in `deepinfer-grammar`, which defines its own copy of
//! [`SchemaType`] / [`StructuredOutputArgs`] — the runtime converts
//! between the two so the crates stay decoupled (no dependency cycle).

use deepinfer_core::{ensure, Result};

/// The grammar formalism of a structured-output constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaType {
    /// Regular expression (full-match semantics).
    Regex,
    /// JSON Schema document.
    Json,
    /// xgrammar-style EBNF grammar.
    Ebnf,
}

/// A structured-output (grammar) constraint passed alongside the sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutputArgs {
    /// Which formalism `schema` is written in.
    pub schema_type: SchemaType,
    /// The grammar source (regex pattern, JSON Schema text, or EBNF text).
    pub schema: String,
}

/// Deterministic sampling parameters for one request.
///
/// Defaults: `temperature` 1.0, `top_p` 1.0 (off), `top_k` 0 (off),
/// `min_p` 0.0 (off), `seq_seed` `None`, `so_args` `None`,
/// `max_tokens` 64 — i.e. plain temperature-1 categorical sampling.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. `0` — or any magnitude below
    /// [`crate::pipeline::GREEDY_TEMPERATURE_EPS`] — selects greedy
    /// argmax decoding (ties → lowest token id, no RNG consumed). Must be
    /// finite and `>= 0`.
    pub temperature: f32,
    /// Nucleus (top-p) cutoff: keep the smallest descending-probability
    /// prefix whose cumulative mass is `>= top_p`. `1` disables the
    /// filter. Must lie in `(0, 1]`.
    pub top_p: f32,
    /// Top-k cutoff: keep the `k` highest-probability tokens (ties →
    /// lower token id). `0` disables the filter. Must not exceed the
    /// vocabulary size.
    pub top_k: usize,
    /// Min-p cutoff: drop tokens whose probability is below
    /// `min_p * max_prob`. `0` disables the filter. Must lie in `[0, 1]`.
    pub min_p: f32,
    /// Per-sequence seed for the deterministic Philox stream. `None`
    /// derives a stable seed from the sequence id (see
    /// [`crate::pipeline::Sampler::seq_rng`]).
    pub seq_seed: Option<u64>,
    /// Optional structured-output (grammar) constraint; the sampler only
    /// validates that it is well-formed, the runtime turns it into a
    /// [`deepinfer_core::mask::TokenMask`].
    pub so_args: Option<StructuredOutputArgs>,
    /// Maximum number of tokens to generate for the request (`>= 1`).
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seq_seed: None,
            so_args: None,
            max_tokens: 64,
        }
    }
}

impl SamplingParams {
    /// Validates parameter ranges against a vocabulary of size `vocab`.
    ///
    /// Rejects: non-positive vocab; non-finite or negative `temperature`;
    /// `top_p` outside `(0, 1]`; `top_k > vocab`; `min_p` outside `[0, 1]`;
    /// `max_tokens < 1`; an empty (whitespace-only) structured-output
    /// schema.
    pub fn validate(&self, vocab: usize) -> Result<()> {
        ensure!(
            vocab >= 1,
            InvalidArgument,
            "validate: vocabulary size must be >= 1 (got {vocab})"
        );
        let t = self.temperature;
        ensure!(
            t.is_finite(),
            OutOfRange,
            "validate: temperature must be finite (got {t})"
        );
        ensure!(
            t >= 0.0,
            OutOfRange,
            "validate: temperature must be >= 0 (got {t}); 0 selects greedy decoding"
        );
        let p = self.top_p;
        ensure!(
            p.is_finite() && p > 0.0 && p <= 1.0,
            OutOfRange,
            "validate: top_p must lie in (0, 1] (got {p})"
        );
        ensure!(
            self.top_k <= vocab,
            OutOfRange,
            "validate: top_k ({}) must not exceed the vocabulary size ({vocab})",
            self.top_k
        );
        let m = self.min_p;
        ensure!(
            m.is_finite() && m >= 0.0 && m <= 1.0,
            OutOfRange,
            "validate: min_p must lie in [0, 1] (got {m})"
        );
        ensure!(
            self.max_tokens >= 1,
            OutOfRange,
            "validate: max_tokens must be >= 1 (got {})",
            self.max_tokens
        );
        if let Some(so) = &self.so_args {
            ensure!(
                !so.schema.trim().is_empty(),
                InvalidArgument,
                "validate: structured-output schema must be a non-empty string"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_spec() {
        let p = SamplingParams::default();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.min_p, 0.0);
        assert_eq!(p.seq_seed, None);
        assert_eq!(p.so_args, None);
        assert_eq!(p.max_tokens, 64);
        assert!(p.validate(128).is_ok());
    }

    #[test]
    fn validate_accepts_boundary_values() {
        // temperature 0 (greedy), top_p exactly 1 (off), top_k == vocab,
        // min_p 0 and 1 are all legal.
        let p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 128,
            min_p: 1.0,
            max_tokens: 1,
            ..SamplingParams::default()
        };
        assert!(p.validate(128).is_ok());
        let p = SamplingParams {
            min_p: 0.0,
            top_p: f32::MIN_POSITIVE,
            ..SamplingParams::default()
        };
        assert!(p.validate(128).is_ok());
    }

    #[test]
    fn validate_rejects_bad_params() {
        let base = SamplingParams::default();
        assert!(base.validate(0).is_err(), "empty vocabulary");

        for bad_temp in [-0.5f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let p = SamplingParams {
                temperature: bad_temp,
                ..base.clone()
            };
            assert!(p.validate(128).is_err(), "temperature {bad_temp}");
        }
        for bad_p in [0.0f32, -0.1, 1.5, f32::NAN] {
            let p = SamplingParams {
                top_p: bad_p,
                ..base.clone()
            };
            assert!(p.validate(128).is_err(), "top_p {bad_p}");
        }
        let p = SamplingParams {
            top_k: 129,
            ..base.clone()
        };
        assert!(p.validate(128).is_err(), "top_k > vocab");
        for bad_m in [-0.1f32, 1.5, f32::NAN] {
            let p = SamplingParams {
                min_p: bad_m,
                ..base.clone()
            };
            assert!(p.validate(128).is_err(), "min_p {bad_m}");
        }
        let p = SamplingParams {
            max_tokens: 0,
            ..base.clone()
        };
        assert!(p.validate(128).is_err(), "max_tokens 0");

        let p = SamplingParams {
            so_args: Some(StructuredOutputArgs {
                schema_type: SchemaType::Regex,
                schema: "   ".to_string(),
            }),
            ..base.clone()
        };
        assert!(p.validate(128).is_err(), "blank schema");
        let p = SamplingParams {
            so_args: Some(StructuredOutputArgs {
                schema_type: SchemaType::Ebnf,
                schema: "root ::= \"ok\";".to_string(),
            }),
            ..base
        };
        assert!(p.validate(128).is_ok(), "valid schema");
    }
}
