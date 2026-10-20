//! Byte-level epsilon-NFA (Thompson construction) shared by every grammar
//! front-end of `deepinfer-grammar`.
//!
//! This follows the classical Thompson NFA pipeline used by XGrammar
//! (mlc-ai/xgrammar, `xgrammar/ebnf_script/` + `grammar.cc`): a parsed
//! grammar tree is lowered to an NFA whose states are `u32` indices and
//! whose transitions are either epsilon edges or single-byte edges
//! (`0..=255`). Every construction path is allocation-bounded by
//! [`GrammarLimits::max_nfa_nodes`] so that untrusted grammar input can
//! never trigger unbounded memory growth — the builder returns
//! `Error::CapacityExceeded` instead.
//!
//! Fragments are `(start, end)` node pairs with the invariant that the
//! `end` node has no outgoing edges *at the time the fragment is
//! returned*; combinators may then wire `end` onwards. `None` (of the
//! `Option<Fragment>` returned by the lowerers) denotes the empty
//! language (a dead branch), which is distinct from the empty *string*
//! (a fragment whose start equals its end).

use deepinfer_core::{ensure, Result};

/// Hard default for the NFA node budget (DoS guard).
pub const DEFAULT_MAX_NFA_NODES: usize = 200_000;

/// A 256-bit byte-set used for character classes and byte ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassSet {
    bits: [u64; 4],
}

impl ClassSet {
    /// The empty set.
    pub fn empty() -> ClassSet {
        ClassSet { bits: [0; 4] }
    }

    /// All 256 bytes.
    pub fn full() -> ClassSet {
        ClassSet { bits: [u64::MAX; 4] }
    }

    /// Adds a single byte.
    pub fn add(&mut self, b: u8) {
        self.bits[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    /// Adds an inclusive byte range (no-op if `lo > hi`).
    pub fn add_range(&mut self, lo: u8, hi: u8) {
        if lo > hi {
            return;
        }
        for b in lo..=hi {
            self.add(b);
        }
    }

    /// Complements over all 256 bytes.
    pub fn negate(&mut self) {
        for w in &mut self.bits {
            *w = !*w;
        }
    }

    /// Removes a single byte.
    pub fn remove(&mut self, b: u8) {
        self.bits[(b >> 6) as usize] &= !(1u64 << (b & 63));
    }

    /// Membership test.
    pub fn contains(&self, b: u8) -> bool {
        (self.bits[(b >> 6) as usize] >> (b & 63)) & 1 == 1
    }

    /// `true` if no byte is in the set.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|w| *w == 0)
    }

    /// Intersects with `other`.
    pub fn intersect_with(&mut self, other: &ClassSet) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a &= *b;
        }
    }

    /// Iterates member bytes in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u16).filter(move |&b| self.contains(b as u8)).map(|b| b as u8)
    }
}

/// One NFA node: epsilon edges, byte edges and an accepting flag.
#[derive(Debug, Clone, Default)]
pub struct NfaNode {
    /// Epsilon (empty-string) successors, in insertion order.
    pub(crate) eps: Vec<u32>,
    /// Byte-labelled successors, in insertion order.
    pub(crate) edges: Vec<(u8, u32)>,
    /// Whether this node is an accepting (final) state.
    pub(crate) accept: bool,
}

/// A byte-level epsilon-NFA with a single start state.
#[derive(Debug, Clone, Default)]
pub struct Nfa {
    /// All nodes; indices are the state ids.
    pub(crate) nodes: Vec<NfaNode>,
    /// Start node id.
    pub(crate) start: u32,
}

/// A `(start, end)` handle to a piece of NFA (Thompson fragment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fragment {
    /// Entry node.
    pub(crate) start: u32,
    /// Exit node (no outgoing edges while the fragment is "free").
    pub(crate) end: u32,
}

/// Allocation-bounded NFA builder (Thompson construction).
pub struct NfaBuilder {
    /// The NFA under construction.
    pub(crate) nfa: Nfa,
    max_nodes: usize,
}

impl NfaBuilder {
    /// Creates a builder bounded by `max_nodes`.
    pub fn new(max_nodes: usize) -> NfaBuilder {
        NfaBuilder {
            nfa: Nfa::default(),
            max_nodes,
        }
    }

    /// Allocates a fresh node, enforcing the node budget.
    pub fn node(&mut self) -> Result<u32> {
        ensure!(
            self.nfa.nodes.len() < self.max_nodes,
            CapacityExceeded,
            "grammar NFA exceeds the node budget of {} states; the grammar is too large to compile",
            self.max_nodes
        );
        self.nfa.nodes.push(NfaNode::default());
        Ok((self.nfa.nodes.len() - 1) as u32)
    }

    /// Adds an epsilon edge.
    pub fn edge_eps(&mut self, from: u32, to: u32) {
        self.nfa.nodes[from as usize].eps.push(to);
    }

    /// Adds a byte edge.
    pub fn edge_byte(&mut self, from: u32, byte: u8, to: u32) {
        self.nfa.nodes[from as usize].edges.push((byte, to));
    }

    /// Adds one edge per byte of `class` (expands to at most 256 edges).
    pub fn edge_class(&mut self, from: u32, class: &ClassSet, to: u32) {
        for b in class.iter() {
            self.edge_byte(from, b, to);
        }
    }

    /// A fragment matching the empty string.
    pub fn frag_empty(&mut self) -> Result<Fragment> {
        let n = self.node()?;
        Ok(Fragment { start: n, end: n })
    }

    /// A fragment matching exactly one byte of `class`
    /// (`None` if the class is empty = empty language).
    pub fn frag_class(&mut self, class: &ClassSet) -> Result<Option<Fragment>> {
        if class.is_empty() {
            return Ok(None);
        }
        let s = self.node()?;
        let e = self.node()?;
        self.edge_class(s, class, e);
        Ok(Some(Fragment { start: s, end: e }))
    }

    /// A fragment matching the exact byte string `bytes` (empty = empty string).
    pub fn frag_literal(&mut self, bytes: &[u8]) -> Result<Fragment> {
        if bytes.is_empty() {
            return self.frag_empty();
        }
        let mut prev = self.node()?;
        let start = prev;
        for &b in bytes {
            let n = self.node()?;
            self.edge_byte(prev, b, n);
            prev = n;
        }
        Ok(Fragment { start, end: prev })
    }

    /// Concatenation. Any dead (`None`) part makes the whole sequence dead.
    ///
    /// The returned fragment keeps the FIRST part's start and the LAST
    /// part's end — the entry point of a concatenation is its first
    /// element, not its last.
    pub fn seq(&mut self, parts: Vec<Option<Fragment>>) -> Result<Option<Fragment>> {
        let mut it = parts.into_iter();
        let Some(first) = it.next() else {
            return self.frag_empty().map(Some);
        };
        let Some(mut acc) = first else {
            return Ok(None);
        };
        let start = acc.start;
        for p in it {
            match p {
                Some(next) => {
                    self.edge_eps(acc.end, next.start);
                    acc = next;
                }
                None => return Ok(None),
            }
        }
        Ok(Some(Fragment {
            start,
            end: acc.end,
        }))
    }

    /// Alternation. `None` if every branch is dead; a single live branch is
    /// passed through unchanged.
    pub fn alt(&mut self, branches: Vec<Option<Fragment>>) -> Result<Option<Fragment>> {
        let live: Vec<Fragment> = branches.into_iter().flatten().collect();
        match live.len() {
            0 => Ok(None),
            1 => Ok(Some(live[0])),
            _ => {
                let s = self.node()?;
                let e = self.node()?;
                for f in live {
                    self.edge_eps(s, f.start);
                    self.edge_eps(f.end, e);
                }
                Ok(Some(Fragment { start: s, end: e }))
            }
        }
    }

    /// Kleene star of a live fragment (matches zero or more).
    pub fn star(&mut self, inner: Fragment) -> Result<Fragment> {
        let s = self.node()?;
        let e = self.node()?;
        self.edge_eps(s, inner.start);
        self.edge_eps(s, e);
        self.edge_eps(inner.end, inner.start);
        self.edge_eps(inner.end, e);
        Ok(Fragment { start: s, end: e })
    }

    /// Optional (zero or one) of a live fragment.
    pub fn optional(&mut self, inner: Fragment) -> Result<Fragment> {
        let s = self.node()?;
        let e = self.node()?;
        self.edge_eps(s, inner.start);
        self.edge_eps(s, e);
        self.edge_eps(inner.end, e);
        Ok(Fragment { start: s, end: e })
    }

    /// Finishes the NFA: marks the fragment's end accepting. `None`
    /// produces a valid NFA for the empty language (one dead start node).
    pub fn finish(mut self, frag: Option<Fragment>) -> Result<Nfa> {
        match frag {
            Some(f) => {
                self.nfa.nodes[f.end as usize].accept = true;
                self.nfa.start = f.start;
            }
            None => {
                let n = self.node()?;
                self.nfa.start = n;
            }
        }
        Ok(self.nfa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(nfa: &Nfa, bytes: &[u8]) -> bool {
        // Small exponential closure walk (test-only).
        fn close(nfa: &Nfa, set: &mut Vec<u32>) {
            let mut i = 0;
            while i < set.len() {
                let n = set[i];
                for &e in &nfa.nodes[n as usize].eps {
                    if !set.contains(&e) {
                        set.push(e);
                    }
                }
                i += 1;
            }
        }
        let mut cur = vec![nfa.start];
        close(nfa, &mut cur);
        for &b in bytes {
            let mut next = Vec::new();
            for &n in &cur {
                for &(eb, t) in &nfa.nodes[n as usize].edges {
                    if eb == b && !next.contains(&t) {
                        next.push(t);
                    }
                }
            }
            close(nfa, &mut next);
            cur = next;
            if cur.is_empty() {
                return false;
            }
        }
        cur.iter().any(|&n| nfa.nodes[n as usize].accept)
    }

    #[test]
    fn literal_and_seq() {
        let mut b = NfaBuilder::new(1000);
        let l1 = b.frag_literal(b"ab").unwrap();
        let l2 = b.frag_literal(b"c").unwrap();
        let seq = b.seq(vec![Some(l1), Some(l2)]).unwrap().unwrap();
        let nfa = b.finish(Some(seq)).unwrap();
        assert!(walk(&nfa, b"abc"));
        assert!(!walk(&nfa, b"ab"));
        assert!(!walk(&nfa, b"abcd"));
    }

    #[test]
    fn alt_and_star() {
        let mut b = NfaBuilder::new(1000);
        let a = b.frag_literal(b"a").unwrap();
        let cls = {
            let mut s = ClassSet::empty();
            s.add_range(b'c', b'd');
            s
        };
        let c = b.frag_class(&cls)
        .unwrap()
        .unwrap();
        let alt = b.alt(vec![Some(a), Some(c)]).unwrap().unwrap();
        let st = b.star(alt).unwrap();
        let nfa = b.finish(Some(st)).unwrap();
        assert!(walk(&nfa, b""));
        assert!(walk(&nfa, b"acdadcc"));
        assert!(!walk(&nfa, b"b"));
    }

    #[test]
    fn dead_propagates() {
        let mut b = NfaBuilder::new(1000);
        let empty_class = b.frag_class(&ClassSet::empty()).unwrap();
        assert!(empty_class.is_none());
        let lit = b.frag_literal(b"x").unwrap();
        let seq = b.seq(vec![Some(lit), empty_class]).unwrap();
        assert!(seq.is_none());
        let alt = b.alt(vec![empty_class]).unwrap();
        assert!(alt.is_none());
    }

    #[test]
    fn node_budget_enforced() {
        let mut b = NfaBuilder::new(4);
        assert!(b.frag_literal(b"abcde").is_err()); // needs 5 nodes, budget 4
    }
}
