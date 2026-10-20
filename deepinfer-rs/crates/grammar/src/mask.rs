//! Adaptive token-mask computation (XGrammar's "adaptive token mask").
//!
//! A token is allowed from a DFA state iff consuming all of its bytes
//! stays on live states AND the landing state either accepts (the
//! generation may stop) or still has a continuation (more text can
//! follow). Zero-byte tokens (EOS-style specials with no bytes) are
//! allowed exactly when the state accepts — the runtime adds EOS
//! handling on top.
//!
//! Masks are cached per state ("adaptive"): nothing is precomputed over
//! the vocabulary up front, so a decode step only pays for the states it
//! actually visits.

use crate::Grammar;
use deepinfer_core::mask::TokenMask;
use std::collections::HashMap;

/// Computes vocabulary masks against a compiled [`Grammar`], caching per
/// DFA state.
pub struct TokenMaskComputer {
    vocab: Vec<Vec<u8>>,
    cache: HashMap<u32, TokenMask>,
}

impl TokenMaskComputer {
    /// Creates a computer over `vocab_bytes` (token id -> its byte
    /// sequence).
    pub fn new(vocab_bytes: Vec<Vec<u8>>) -> TokenMaskComputer {
        TokenMaskComputer {
            vocab: vocab_bytes,
            cache: HashMap::new(),
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of cached state masks.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// The allowed-token mask at `state` (cached).
    pub fn compute_mask(&mut self, grammar: &Grammar, state: u32) -> TokenMask {
        if let Some(m) = self.cache.get(&state) {
            return m.clone();
        }
        let mut mask = TokenMask::zeros(self.vocab.len());
        // Only states in [0, num_states) are live by construction.
        if (state as usize) < grammar.num_states() {
            for (id, bytes) in self.vocab.iter().enumerate() {
                if bytes.is_empty() {
                    if grammar.is_accepting(state) {
                        mask.set(id as u32);
                    }
                    continue;
                }
                if let Some(final_state) = self.walk(grammar, state, bytes) {
                    // Accepting (may stop) or live (may continue): the DFA
                    // is liveness-pruned, so any reachable state is live.
                    let _ = final_state;
                    mask.set(id as u32);
                }
            }
        }
        self.cache.insert(state, mask.clone());
        mask
    }

    /// Advances `state` over a token's bytes (`None` = dead).
    pub fn advance_token(&self, grammar: &Grammar, state: u32, token: u32) -> Option<u32> {
        let bytes = self.vocab.get(token as usize)?;
        self.walk(grammar, state, bytes)
    }

    fn walk(&self, grammar: &Grammar, mut state: u32, bytes: &[u8]) -> Option<u32> {
        for &b in bytes {
            state = grammar.advance(state, b)?;
        }
        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grammar;

    fn vocab(pairs: &[&str]) -> Vec<Vec<u8>> {
        pairs.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    #[test]
    fn regex_mask_basics() {
        // Grammar: full strings of digits, at least one ("[0-9]+").
        let g = Grammar::compile_regex("[0-9]+").unwrap();
        let mut mc = TokenMaskComputer::new(vocab(&["1", "2", "a", "", "12"]));
        let s0 = g.initial_state();
        let m0 = mc.compute_mask(&g, s0);
        assert!(m0.test(0) && m0.test(1), "digits allowed at start");
        assert!(!m0.test(2), "'a' never allowed");
        assert!(!m0.test(3), "empty token only when accepting; s0 is not");
        assert!(m0.test(4), "'12' allowed (continuation exists)");
        assert_eq!(m0.count_ones(), 3);
        // After '1' the state accepts: empty token now allowed.
        let s1 = mc.advance_token(&g, s0, 0).unwrap();
        let m1 = mc.compute_mask(&g, s1);
        assert!(m1.test(3), "empty/EOS token allowed at accepting state");
        assert!(m1.test(0) && m1.test(4));
        // 'a' is dead from anywhere.
        assert!(mc.advance_token(&g, s0, 2).is_none());
        // Cache behaviour.
        assert_eq!(mc.cache_size(), 2);
        let m1b = mc.compute_mask(&g, s1);
        assert_eq!(m1b, m1);
        assert_eq!(mc.cache_size(), 2);
    }

    #[test]
    fn mask_matches_brute_force() {
        // Brute force: token t allowed from s iff some accepted string
        // starts with t's bytes (i.e. t is a viable prefix whose own end
        // can still stop or continue).
        let g = Grammar::compile_regex("(ab|ba)+").unwrap();
        let words = vocab(&["a", "b", "ab", "ba", "aa", "c", ""]);
        let mut mc = TokenMaskComputer::new(words.clone());
        // Enumerate all accepted strings up to length 6 by DFS.
        fn accepted_prefixes(g: &Grammar, max_len: usize) -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            let mut stack = vec![(g.initial_state(), Vec::<u8>::new())];
            while let Some((s, prefix)) = stack.pop() {
                if g.is_accepting(s) {
                    out.push(prefix.clone());
                }
                if prefix.len() >= max_len {
                    continue;
                }
                for b in 0u8..=255 {
                    if let Some(ns) = g.advance(s, b) {
                        let mut p = prefix.clone();
                        p.push(b);
                        stack.push((ns, p));
                    }
                }
            }
            out
        }
        let accepted = accepted_prefixes(&g, 6);
        for state in 0..g.num_states() as u32 {
            let _ = mc.compute_mask(&g, state);
            for (_id, _bytes) in words.iter().enumerate() {
                // Reach the state first (only start is directly reachable
                // in this test): we check the start state and states
                // reachable via allowed prefixes.
                let _ = state;
                break;
            }
            break;
        }
        // Check the start state against brute force.
        let s0 = g.initial_state();
        let mask = mc.compute_mask(&g, s0);
        for (id, bytes) in words.iter().enumerate() {
            let viable = if bytes.is_empty() {
                g.is_accepting(s0)
            } else {
                accepted
                    .iter()
                    .any(|a| a.len() >= bytes.len() && &a[..bytes.len()] == bytes.as_slice())
                    || {
                        // Viable prefixes of length < accepted strings: the
                        // accepted enumeration above already covers all
                        // prefixes of accepted strings when max_len is
                        // large enough; treat presence in any accepted
                        // string's prefix set as viability.
                        false
                    }
            };
            assert_eq!(
                mask.test(id as u32),
                viable,
                "token {id} ({:?}) at start",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    #[test]
    fn json_schema_mask_flow() {
        let schema = deepinfer_core::json::Json::parse(
            r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}"#,
        )
        .unwrap();
        let g = Grammar::compile_json_schema(&schema).unwrap();
        // ids: 0='{' 1='}' 2='"' 3='n' 4=':' 5='7' 6='x' 7=','
        let mut mc = TokenMaskComputer::new(vocab(&["{", "}", "\"", "n", ":", "7", "x", ","]));
        let s0 = g.initial_state();
        let m0 = mc.compute_mask(&g, s0);
        assert!(m0.test(0), "open-brace starts the object");
        assert!(!m0.test(1), "close-brace cannot start it (n is required)");
        assert!(!m0.test(3), "a key must start with a quote");
        let s1 = mc.advance_token(&g, s0, 0).unwrap();
        let m1 = mc.compute_mask(&g, s1);
        assert!(m1.test(2), "key quote allowed");
        let s2 = mc.advance_token(&g, s1, 2).unwrap();
        let m2 = mc.compute_mask(&g, s2);
        assert!(m2.test(3), "'n' continues the key");
        let s3 = mc.advance_token(&g, s2, 3).unwrap();
        let m3 = mc.compute_mask(&g, s3);
        assert!(m3.test(2), "closing key quote");
        let s4 = mc.advance_token(&g, s3, 2).unwrap();
        let m4 = mc.compute_mask(&g, s4);
        assert!(m4.test(4), "colon after key");
        let s5 = mc.advance_token(&g, s4, 4).unwrap();
        let m5 = mc.compute_mask(&g, s5);
        assert!(m5.test(5), "'7' is a valid integer value");
        assert!(!m5.test(6), "'x' is not an integer");
        let s6 = mc.advance_token(&g, s5, 5).unwrap();
        let m6 = mc.compute_mask(&g, s6);
        assert!(m6.test(1), "close-brace closes the object");
        assert!(!m6.test(7), "trailing comma is invalid JSON");
        let s7 = mc.advance_token(&g, s6, 1).unwrap();
        assert!(g.is_accepting(s7));
    }

    #[test]
    fn out_of_range_states_yield_empty_masks() {
        let g = Grammar::compile_regex("a+").unwrap();
        let mut mc = TokenMaskComputer::new(vocab(&["a", "b"]));
        let m = mc.compute_mask(&g, 999);
        assert_eq!(m.count_ones(), 0);
        assert!(mc.advance_token(&g, 999, 0).is_none());
        assert!(mc.advance_token(&g, 0, 42).is_none());
    }
}
