//! Determinization, liveness pruning and DFA intersection.
//!
//! Subset construction (Rabin–Scott / Hopcroft–Ullman) converts the
//! Thompson epsilon-NFA into a deterministic byte-level automaton; the
//! state count is capped by `GrammarLimits::max_dfa_states` so that
//! grammars whose DFA explodes (e.g. `(a|aa){40}`) fail fast with
//! `Error::CapacityExceeded` instead of exhausting memory — the same DoS
//! posture XGrammar takes with its compiled-grammar size limits.
//!
//! After determinization, states that cannot reach an accepting state
//! ("dead" states) are pruned via reverse reachability: every transition
//! into a dead state becomes `None`. This is what makes token masking
//! correct: `advance() == None` unambiguously means "no valid completion
//! exists through this byte".
//!
//! `intersect` is the synchronous product construction used to implement
//! JSON Schema `allOf` (and `$ref` siblings); `embed_dfa` splices a
//! finished DFA back into a larger NFA under construction by routing
//! epsilon edges from a fresh start to the DFA initial state and from
//! every DFA accepting state to a fresh fragment end. This is exact: the
//! embedded fragment accepts exactly the DFA's language.

use crate::nfa::{Fragment, Nfa, NfaBuilder};
use deepinfer_core::{ensure, Error, Result};
use std::collections::HashMap;

/// A deterministic byte-level automaton. `trans[s][b] == None` means the
/// transition dies (no accepting continuation). State `0` is always the
/// start state, and every state is both reachable and co-reachable
/// ("live"): non-live states are pruned away.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dfa {
    /// `trans[s][b]`: target state on byte `b`, or `None` if dead.
    pub(crate) trans: Vec<[Option<u32>; 256]>,
    /// `accepting[s]`: whether `s` is a final state.
    pub(crate) accepting: Vec<bool>,
}

impl Dfa {
    /// Number of states.
    pub fn len(&self) -> usize {
        self.trans.len()
    }

    /// `true` when the automaton accepts the empty language (the canonical
    /// one-state dead form produced by pruning).
    pub fn is_dead(&self) -> bool {
        self.trans.len() == 1 && !self.accepting[0]
    }

    /// The single-state representation of the empty language.
    pub fn dead() -> Dfa {
        Dfa {
            trans: vec![[None; 256]],
            accepting: vec![false],
        }
    }
}

/// Workspace for epsilon-closure computation with O(1) reset.
struct Closure {
    in_set: Vec<bool>,
    touched: Vec<u32>,
    stack: Vec<u32>,
}

impl Closure {
    fn new(n: usize) -> Closure {
        Closure {
            in_set: vec![false; n],
            touched: Vec::new(),
            stack: Vec::new(),
        }
    }

    /// Epsilon closure of `seeds`, returned as a sorted, deduplicated set.
    fn close(&mut self, nfa: &Nfa, seeds: &[u32]) -> Vec<u32> {
        for &s in seeds {
            let s = s as usize;
            if !self.in_set[s] {
                self.in_set[s] = true;
                self.touched.push(s as u32);
                self.stack.push(s as u32);
            }
        }
        while let Some(n) = self.stack.pop() {
            for &e in &nfa.nodes[n as usize].eps {
                let e = e as usize;
                if !self.in_set[e] {
                    self.in_set[e] = true;
                    self.touched.push(e as u32);
                    self.stack.push(e as u32);
                }
            }
        }
        let mut out = std::mem::take(&mut self.touched);
        out.sort_unstable();
        // touched is now in `out`; clear membership flags.
        for t in out.iter() {
            self.in_set[*t as usize] = false;
        }
        out
    }
}

/// Subset construction + liveness pruning. The result only contains states
/// that are both reachable from the start and co-reachable to an accepting
/// state (dead states are pruned); the empty language is represented by
/// [`Dfa::dead`].
pub fn determinize(nfa: &Nfa, max_states: usize) -> Result<Dfa> {
    // Guard on total subset size to bound memory (DoS): each DFA state is
    // an NFA-node set; cap the sum of set sizes at 64 nodes per DFA state.
    let max_total_elems = max_steps_guard(max_states);
    let mut closure = Closure::new(nfa.nodes.len());
    let mut ids: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut sets: Vec<Vec<u32>> = Vec::new();
    let mut trans: Vec<[Option<u32>; 256]> = Vec::new();
    let mut acc: Vec<bool> = Vec::new();
    let mut total_elems: usize = 0;

    let start_set = closure.close(nfa, &[nfa.start]);
    ids.insert(start_set.clone(), 0);
    sets.push(start_set.clone());
    total_elems += start_set.len();

    let mut moves: Vec<(u8, u32)> = Vec::new();
    let mut i = 0;
    while i < sets.len() {
        ensure!(
            sets.len() <= max_states,
            CapacityExceeded,
            "grammar DFA exceeds the state cap of {} states (exponential subset blowup); increase GrammarLimits::max_dfa_states if you trust the grammar",
            max_states
        );
        let set = sets[i].clone();
        let is_acc = set.iter().any(|&n| nfa.nodes[n as usize].accept);
        let mut row: [Option<u32>; 256] = [None; 256];
        moves.clear();
        for &n in &set {
            for &(b, t) in &nfa.nodes[n as usize].edges {
                moves.push((b, t));
            }
        }
        moves.sort_unstable_by_key(|&(b, _)| b);
        let mut k = 0;
        while k < moves.len() {
            let byte = moves[k].0;
            let mut j = k;
            while j < moves.len() && moves[j].0 == byte {
                j += 1;
            }
            let targets: Vec<u32> = moves[k..j].iter().map(|&(_, t)| t).collect();
            let closed = closure.close(nfa, &targets);
            let id = match ids.get(&closed) {
                Some(&id) => id,
                None => {
                    ensure!(
                        sets.len() < max_states,
                        CapacityExceeded,
                        "grammar DFA exceeds the state cap of {} states (exponential subset blowup); increase GrammarLimits::max_dfa_states if you trust the grammar",
                        max_states
                    );
                    total_elems = total_elems
                        .checked_add(closed.len())
                        .ok_or_else(|| Error::CapacityExceeded("grammar DFA size overflow".into()))?;
                    ensure!(
                        total_elems <= max_total_elems,
                        CapacityExceeded,
                        "grammar DFA subset sets exceed the internal size guard of {} elements",
                        max_total_elems
                    );
                    let id = sets.len() as u32;
                    ids.insert(closed.clone(), id);
                    sets.push(closed);
                    id
                }
            };
            row[byte as usize] = Some(id);
            k = j;
        }
        trans.push(row);
        acc.push(is_acc);
        i += 1;
    }

    Ok(prune_live(Dfa { trans, accepting: acc }))
}

fn max_steps_guard(max_states: usize) -> usize {
    max_states.saturating_mul(64)
}

/// Removes states that cannot reach an accepting state; renumbers the
/// survivors in their original order (state 0 stays 0 whenever the
/// language is non-empty). Transitions into dead states become `None`.
pub fn prune_live(dfa: Dfa) -> Dfa {
    let n = dfa.trans.len();
    if n == 0 {
        return Dfa::dead();
    }
    // Reverse adjacency.
    let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (s, row) in dfa.trans.iter().enumerate() {
        for t in row.iter().flatten() {
            rev[*t as usize].push(s as u32);
        }
    }
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for (s, &a) in dfa.accepting.iter().enumerate() {
        if a && !live[s] {
            live[s] = true;
            stack.push(s);
        }
    }
    while let Some(s) = stack.pop() {
        for &p in &rev[s] {
            let p = p as usize;
            if !live[p] {
                live[p] = true;
                stack.push(p);
            }
        }
    }
    if !live[0] {
        // Empty language.
        return Dfa::dead();
    }
    let mut new_id = vec![u32::MAX; n];
    let mut next: u32 = 0;
    for s in 0..n {
        if live[s] {
            new_id[s] = next;
            next += 1;
        }
    }
    let count = next as usize;
    let mut trans = vec![[None; 256]; count];
    let mut accepting = vec![false; count];
    for s in 0..n {
        if !live[s] {
            continue;
        }
        let ns = new_id[s];
        accepting[ns as usize] = dfa.accepting[s];
        for b in 0..256 {
            trans[ns as usize][b] = dfa.trans[s][b].and_then(|t| {
                let nt = new_id[t as usize];
                if nt == u32::MAX {
                    None
                } else {
                    Some(nt)
                }
            });
        }
    }
    Dfa { trans, accepting }
}

/// Synchronous product of two DFAs (language intersection), followed by
/// liveness pruning. Used for `allOf` and `$ref`-sibling merging.
pub fn intersect(a: &Dfa, b: &Dfa, max_states: usize) -> Result<Dfa> {
    // Fast path: empty language on either side.
    if !a.accepting.iter().any(|&x| x) || !b.accepting.iter().any(|&x| x) {
        return Ok(Dfa::dead());
    }
    let mut ids: HashMap<(u32, u32), u32> = HashMap::new();
    let mut pairs: Vec<(u32, u32)> = vec![(0, 0)];
    ids.insert((0, 0), 0);
    let mut trans: Vec<[Option<u32>; 256]> = Vec::new();
    let mut acc: Vec<bool> = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        ensure!(
            pairs.len() <= max_states,
            CapacityExceeded,
            "grammar DFA intersection exceeds the state cap of {} states",
            max_states
        );
        let (sa, sb) = pairs[i];
        let mut row: [Option<u32>; 256] = [None; 256];
        for byte in 0..256 {
            if let (Some(ta), Some(tb)) = (a.trans[sa as usize][byte], b.trans[sb as usize][byte]) {
                let key = (ta, tb);
                let id = match ids.get(&key) {
                    Some(&id) => id,
                    None => {
                        ensure!(
                            pairs.len() < max_states,
                            CapacityExceeded,
                            "grammar DFA intersection exceeds the state cap of {} states",
                            max_states
                        );
                        let id = pairs.len() as u32;
                        ids.insert(key, id);
                        pairs.push(key);
                        id
                    }
                };
                row[byte] = Some(id);
            }
        }
        trans.push(row);
        acc.push(a.accepting[sa as usize] && b.accepting[sb as usize]);
        i += 1;
    }
    Ok(prune_live(Dfa {
        trans,
        accepting: acc,
    }))
}

/// Folds a list of DFAs down to their intersection (identity for one).
pub fn intersect_all(dfas: Vec<Dfa>, max_states: usize) -> Result<Dfa> {
    let mut it = dfas.into_iter();
    let Some(mut acc) = it.next() else {
        return Ok(Dfa::dead());
    };
    for d in it {
        acc = intersect(&acc, &d, max_states)?;
    }
    Ok(acc)
}

/// Splices a finished DFA into an NFA under construction as an exact
/// fragment for its language.
pub fn embed_dfa(builder: &mut NfaBuilder, dfa: &Dfa) -> Result<Fragment> {
    let s = builder.node()?;
    let e = builder.node()?;
    let mut map = vec![u32::MAX; dfa.len()];
    for slot in map.iter_mut() {
        *slot = builder.node()?;
    }
    for (i, row) in dfa.trans.iter().enumerate() {
        for (b, t) in row.iter().enumerate() {
            if let Some(t) = t {
                builder.edge_byte(map[i], b as u8, map[*t as usize]);
            }
        }
    }
    builder.edge_eps(s, map[0]);
    for (i, &a) in dfa.accepting.iter().enumerate() {
        if a {
            builder.edge_eps(map[i], e);
        }
    }
    Ok(Fragment { start: s, end: e })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nfa::ClassSet;

    fn lit_nfa(bytes: &[u8]) -> Nfa {
        let mut b = NfaBuilder::new(100);
        let f = b.frag_literal(bytes).unwrap();
        b.finish(Some(f)).unwrap()
    }

    #[test]
    fn determinize_literal() {
        let dfa = determinize(&lit_nfa(b"ab"), 1000).unwrap();
        assert_eq!(dfa.len(), 3);
        assert!(!dfa.accepting[0]);
        assert!(!dfa.accepting[1]);
        assert!(dfa.accepting[2]);
        let t = dfa.trans[0][b'a' as usize].unwrap();
        assert_eq!(dfa.trans[t as usize][b'b' as usize].unwrap(), 2);
        assert!(dfa.trans[0][b'b' as usize].is_none());
    }

    #[test]
    fn dead_transitions_pruned() {
        // "a" followed by a class matching nothing is dead overall, but a
        // simpler case: (ab | a) — after "a" both continuing and stopping
        // are possible; "ac" must die.
        let mut b = NfaBuilder::new(100);
        let ab = b.frag_literal(b"ab").unwrap();
        let a = b.frag_literal(b"a").unwrap();
        let alt = b.alt(vec![Some(ab), Some(a)]).unwrap().unwrap();
        let nfa = b.finish(Some(alt)).unwrap();
        let dfa = determinize(&nfa, 1000).unwrap();
        let s1 = dfa.trans[0][b'a' as usize].unwrap();
        assert!(dfa.accepting[s1 as usize]);
        assert!(dfa.trans[s1 as usize][b'c' as usize].is_none());
        let s2 = dfa.trans[s1 as usize][b'b' as usize].unwrap();
        assert!(dfa.accepting[s2 as usize]);
    }

    #[test]
    fn empty_language_is_single_dead_state() {
        let mut b = NfaBuilder::new(100);
        let none = b.frag_class(&ClassSet::empty()).unwrap();
        let nfa = b.finish(none).unwrap();
        let dfa = determinize(&nfa, 1000).unwrap();
        assert_eq!(dfa.len(), 1);
        assert!(!dfa.accepting[0]);
        assert!(dfa.trans[0].iter().all(|t| t.is_none()));
    }

    #[test]
    fn state_cap_enforced() {
        // "an 'a' 20-from-last" requires 2^20 DFA states (classic subset
        // blowup); the small NFA determinizes past any modest cap.
        let nfa = crate::regex::compile_regex_nfa(
            "[ab]*a[ab]{20}",
            &crate::GrammarLimits::default(),
        )
        .unwrap();
        assert!(matches!(
            determinize(&nfa, 1000),
            Err(Error::CapacityExceeded(_))
        ));
    }

    #[test]
    fn intersection_exact() {
        // L1 = strings over {a,b} ending in "ab"; L2 = length 3.
        let cls = {
            let mut c = ClassSet::empty();
            c.add(b'a');
            c.add(b'b');
            c
        };
        let mut builder = NfaBuilder::new(200);
        let fc = builder.frag_class(&cls).unwrap().unwrap();
        let any = builder.star(fc).unwrap();
        let ab = builder.frag_literal(b"ab").unwrap();
        let f1 = builder.seq(vec![Some(any), Some(ab)]).unwrap().unwrap();
        let nfa1 = builder.finish(Some(f1)).unwrap();
        let d1 = determinize(&nfa1, 1000).unwrap();

        let mut builder2 = NfaBuilder::new(200);
        let c1 = builder2.frag_class(&cls).unwrap().unwrap();
        let c2 = builder2.frag_class(&cls).unwrap().unwrap();
        let c3 = builder2.frag_class(&cls).unwrap().unwrap();
        let f2 = builder2.seq(vec![Some(c1), Some(c2), Some(c3)]).unwrap().unwrap();
        let nfa2 = builder2.finish(Some(f2)).unwrap();
        let d2 = determinize(&nfa2, 1000).unwrap();

        let prod = intersect(&d1, &d2, 1000).unwrap();
        let dfa = prod;
        fn accepted(dfa: &Dfa, s: &str) -> bool {
            let mut st = 0u32;
            for &b in s.as_bytes() {
                match dfa.trans[st as usize][b as usize] {
                    Some(t) => st = t,
                    None => return false,
                }
            }
            dfa.accepting[st as usize]
        }
        assert!(accepted(&dfa, "aab"));
        assert!(accepted(&dfa, "bab"));
        assert!(!accepted(&dfa, "ab"));
        assert!(!accepted(&dfa, "aabb"));
        assert!(!accepted(&dfa, "aba"));
    }

    #[test]
    fn embed_roundtrip() {
        let dfa = determinize(&lit_nfa(b"xyz"), 100).unwrap();
        let mut b = NfaBuilder::new(100);
        let f = embed_dfa(&mut b, &dfa).unwrap();
        let pre = b.frag_literal(b"p").unwrap();
        let seq = b.seq(vec![Some(pre), Some(f)]).unwrap().unwrap();
        let nfa = b.finish(Some(seq)).unwrap();
        let d2 = determinize(&nfa, 100).unwrap();
        let mut st = 0u32;
        for &bb in b"pxyz" {
            st = d2.trans[st as usize][bb as usize].unwrap();
        }
        assert!(d2.accepting[st as usize]);
    }
}
