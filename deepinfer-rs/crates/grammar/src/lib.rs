//! # deepinfer-grammar
//!
//! XGrammar-equivalent structured outputs for DeepInfer-rs: EBNF, JSON
//! Schema and regular-expression front-ends compiled to a **byte-level
//! DFA**, plus adaptive token-mask computation over a vocabulary — the
//! engine behind DeepInfer's `so_args = {"schema_type": ..., "schema": ...}`
//! structured-output contract.
//!
//! Pipeline (Thompson-style, mirroring mlc-ai/xgrammar):
//! parse → epsilon-NFA ([`nfa`]) → subset construction with a hard state
//! cap → liveness pruning ([`dfa`]) → [`Grammar`]. Every stage is
//! allocation-bounded by [`GrammarLimits`]: untrusted grammar input can
//! only ever produce `Error::CapacityExceeded`, never unbounded memory
//! growth or a panic. This crate is a security boundary.
//!
//! Determinism: identical source text compiles to an identical automaton
//! (fixed construction order, no hash-order iteration in outputs).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dfa;
pub mod ebnf;
mod json_schema;
mod mask;
pub mod nfa;
pub mod regex;

pub use json_schema::compile_json_schema_nfa;
pub use mask::TokenMaskComputer;

use deepinfer_core::{Error, Result};

/// Compilation budgets — the denial-of-service guard for untrusted
/// grammar input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarLimits {
    /// Maximum number of NFA nodes (`200_000` by default).
    pub max_nfa_nodes: usize,
    /// Maximum number of DFA states after subset construction (`65_536`).
    pub max_dfa_states: usize,
    /// Maximum recursion / rule-reference depth (`256`).
    pub max_recursion_depth: u32,
    /// Maximum bounded-repetition count (`{m,n}` with `n` beyond this is
    /// rejected; `{m,}` means `{m, min(m + 2048, max_count)}`).
    pub max_count: u32,
}

impl Default for GrammarLimits {
    fn default() -> GrammarLimits {
        GrammarLimits {
            max_nfa_nodes: nfa::DEFAULT_MAX_NFA_NODES,
            max_dfa_states: 1 << 16,
            max_recursion_depth: 256,
            max_count: 4096,
        }
    }
}

/// A compiled, pruned, byte-level deterministic grammar.
///
/// `advance(state, byte) == None` unambiguously means "no valid
/// completion exists through this byte" — dead states are pruned away,
/// which is what makes [`TokenMaskComputer`] exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    dfa: dfa::Dfa,
    limits: GrammarLimits,
}

impl Grammar {
    /// Compiles an EBNF grammar (xgrammar-style subset; see [`ebnf`]).
    pub fn compile_ebnf(text: &str) -> Result<Grammar> {
        Self::compile_ebnf_with_limits(text, &GrammarLimits::default())
    }

    /// Compiles an EBNF grammar under explicit limits.
    pub fn compile_ebnf_with_limits(text: &str, limits: &GrammarLimits) -> Result<Grammar> {
        let nfa = ebnf::compile_ebnf_nfa(text, limits)?;
        Self::from_nfa(nfa, limits)
    }

    /// Compiles a JSON Schema (supported subset; see [`json_schema`]).
    pub fn compile_json_schema(schema: &deepinfer_core::json::Json) -> Result<Grammar> {
        Self::compile_json_schema_with_limits(schema, &GrammarLimits::default())
    }

    /// Compiles a JSON Schema under explicit limits.
    pub fn compile_json_schema_with_limits(
        schema: &deepinfer_core::json::Json,
        limits: &GrammarLimits,
    ) -> Result<Grammar> {
        let nfa = json_schema::compile_json_schema_nfa(schema, limits)?;
        Self::from_nfa(nfa, limits)
    }

    /// Compiles a regular expression with **full-match** semantics (the
    /// whole byte string must match; see [`regex`]).
    pub fn compile_regex(pattern: &str) -> Result<Grammar> {
        Self::compile_regex_with_limits(pattern, &GrammarLimits::default())
    }

    /// Compiles a regular expression under explicit limits.
    pub fn compile_regex_with_limits(pattern: &str, limits: &GrammarLimits) -> Result<Grammar> {
        let nfa = regex::compile_regex_nfa(pattern, limits)?;
        Self::from_nfa(nfa, limits)
    }

    /// Dispatches DeepInfer `so_args`: `schema_type` is `"regex"`,
    /// `"json"` or `"ebnf"`; `schema` is the corresponding source text
    /// (JSON schemas are parsed first).
    pub fn from_so_args(schema_type: &str, schema: &str) -> Result<Grammar> {
        match schema_type.trim().to_ascii_lowercase().as_str() {
            "regex" | "re" => Grammar::compile_regex(schema),
            "json" | "json_schema" | "jsonschema" => {
                let parsed = deepinfer_core::json::Json::parse(schema).map_err(|e| {
                    Error::Grammar(format!("so_args json schema does not parse: {e}"))
                })?;
                Grammar::compile_json_schema(&parsed)
            }
            "ebnf" | "bnf" => Grammar::compile_ebnf(schema),
            other => Err(Error::Grammar(format!(
                "unsupported so_args schema_type {other:?} (expected regex/json/ebnf)"
            ))),
        }
    }

    fn from_nfa(nfa: nfa::Nfa, limits: &GrammarLimits) -> Result<Grammar> {
        let dfa = dfa::determinize(&nfa, limits.max_dfa_states)?;
        let dfa = dfa::prune_live(dfa);
        Ok(Grammar { dfa, limits: *limits })
    }

    /// The initial state.
    pub fn initial_state(&self) -> u32 {
        0
    }

    /// Whether `state` is accepting (the generation may stop here).
    pub fn is_accepting(&self, state: u32) -> bool {
        self.dfa
            .accepting
            .get(state as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Advances one byte. `None` means the byte kills every completion
    /// (dead state).
    pub fn advance(&self, state: u32, byte: u8) -> Option<u32> {
        let row = self.dfa.trans.get(state as usize)?;
        row[byte as usize]
    }

    /// Number of DFA states.
    pub fn num_states(&self) -> usize {
        self.dfa.len()
    }



    /// Whether the full byte string `bytes` is accepted (test/diagnostic
    /// helper).
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut s = self.initial_state();
        for &b in bytes {
            match self.advance(s, b) {
                Some(n) => s = n,
                None => return false,
            }
        }
        self.is_accepting(s)
    }


}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_full_match_basics() {
        let g = Grammar::compile_regex("a*b").unwrap();
        assert!(g.accepts(b"b"));
        assert!(g.accepts(b"aaab"));
        assert!(!g.accepts(b"aac"));
        assert!(!g.accepts(b"ab\n"));

        let g = Grammar::compile_regex("(a|b)+").unwrap();
        assert!(g.accepts(b"abab"));
        assert!(!g.accepts(b""));
        assert!(!g.accepts(b"abc"));

        let g = Grammar::compile_regex("[a-c]{2,4}").unwrap();
        assert!(g.accepts(b"ab"));
        assert!(g.accepts(b"abcc"));
        assert!(!g.accepts(b"a"));
        assert!(!g.accepts(b"abcde"));
        assert!(!g.accepts(b"abcd"), "'d' is outside [a-c]");

        let g = Grammar::compile_regex("").unwrap();
        assert!(g.accepts(b""));
        assert!(!g.accepts(b"a"));
    }

    #[test]
    fn regex_rejects_unsupported() {
        assert!(Grammar::compile_regex(r"(a)\1").is_err());
        assert!(Grammar::compile_regex(r"\bfoo\b").is_err());
        assert!(Grammar::compile_regex(r"a(?=b)").is_err());
    }

    #[test]
    fn regex_state_cap_is_a_dos_guard() {
        // Exponential subset construction: "an 'a' 20-from-last" needs
        // 2^20 DFA states (the classic blowup example).
        let pat = "[ab]*a[ab]{20}";
        assert!(matches!(
            Grammar::compile_regex(&pat),
            Err(Error::CapacityExceeded { .. })
        ));
        // Huge counted repetition is rejected by the count cap.
        assert!(matches!(
            Grammar::compile_regex("a{1000000}"),
            Err(Error::CapacityExceeded { .. }) | Err(Error::Grammar(_))
        ));
    }

    #[test]
    fn ebnf_compile_and_accepts() {
        let g = Grammar::compile_ebnf(
            "root ::= \"(\" root \")\" | \"x\" ;",
        )
        .unwrap();
        assert!(g.accepts(b"x"));
        assert!(g.accepts(b"(((x)))"));
        assert!(!g.accepts(b"((x)"));
        assert!(!g.accepts(b""));
    }

    #[test]
    fn ebnf_rejects_garbage() {
        assert!(Grammar::compile_ebnf("root ::= ").is_err());
        assert!(Grammar::compile_ebnf("root ::= [a-").is_err());
        // Non-terminating recursion.
        assert!(Grammar::compile_ebnf("root ::= root ;").is_err());
    }

    #[test]
    fn from_so_args_dispatch() {
        let g = Grammar::from_so_args("regex", "ab+c").unwrap();
        assert!(g.accepts(b"abbbc"));
        let g = Grammar::from_so_args("ebnf", "root ::= \"hi\" ;").unwrap();
        assert!(g.accepts(b"hi"));
        let g = Grammar::from_so_args(
            "json",
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}"#,
        )
        .unwrap();
        assert!(g.accepts(br#"{"a":7}"#));
        assert!(!g.accepts(br#"{"a":"x"}"#));
        assert!(Grammar::from_so_args("yaml", "...").is_err());
        // Invalid JSON text under the "json" type is a grammar error.
        assert!(Grammar::from_so_args("json", "{not json").is_err());
    }

    #[test]
    fn determinism() {
        let a = Grammar::compile_regex("[0-9]+(\\.[0-9]+)?").unwrap();
        let b = Grammar::compile_regex("[0-9]+(\\.[0-9]+)?").unwrap();
        assert_eq!(a, b);
    }
}
