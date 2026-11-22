//! The prompt prefix cache: a radix tree over token ids mapping shared
//! prompt prefixes to reusable engine state snapshots.
//!
//! This is the CPU engine's answer to SGLang's RadixAttention and
//! vLLM's automatic prefix caching: it snapshots the exact f32 KV
//! shadow rows, DSA indexer rows and SWA rings of a completed prompt
//! ([`deepinfer_runtime::PrefixState`]) and *re-appends* them on a
//! later request sharing the prefix — a memory-bound copy replacing a
//! compute-bound forward, bit-identical at the same chunk boundary
//! (see the runtime `prefix_snapshot_restore_is_bit_identical` test).
//! The GPU paged-serve engine does **not** use this index — its
//! prefix cache is engine-internal block-granular page sharing over
//! the paged pools (`deepinfer_runtime`'s `serve_prefix_*` surface,
//! reported through [`crate::engine::TokenEngine::prefix_stats`]).
//!
//! Workloads that win: multi-turn conversations (every turn extends
//! the previous conversation), few-shot prompts with a shared system
//! preamble, and agentic loops re-issuing near-identical prompts.
//!
//! Concurrency: the index is shared between the coordinator thread
//! (lookup at admission) and the engine thread (insert after prompt
//! completion, restore before the first chunk) through a
//! `Arc<Mutex<PrefixIndex>>`; the critical sections are O(prompt)
//! tree walks with no model work inside.
//!
//! Eviction: LRU over snapshot-bearing nodes, bounded by a total
//! cached-token budget (`ServerConfig::prefix_cache_tokens`, 0
//! disables the feature). Tree nodes without snapshots are pruned
//! bottom-up after eviction so a long-dead branch costs nothing.

use std::collections::HashMap;
use std::sync::Arc;

use deepinfer_runtime::PrefixState;

/// One radix-tree node: the token on the edge from its parent, an
/// optional snapshot for the prefix ending here, and child edges.
#[derive(Debug, Default)]
struct Node {
    /// Child edges keyed by the next prompt token.
    children: HashMap<u32, Node>,
    /// Snapshot for the prefix ending at this node, if any.
    snap: Option<Arc<PrefixState>>,
    /// LRU stamp of the last lookup/insert touching this node's
    /// snapshot (0 = never).
    last_used: u64,
}

/// The shared prefix index.
#[derive(Debug)]
pub struct PrefixIndex {
    root: Node,
    /// Total tokens covered by live snapshots (sum of `snap.len`).
    cached_tokens: usize,
    /// Budget in tokens; 0 disables both insert and lookup.
    max_tokens: usize,
    /// Monotonic clock for LRU stamps.
    tick: u64,
    /// Diagnostics: total lookups that returned > 0.
    hits: u64,
    /// Diagnostics: total tokens returned by lookups.
    reused_tokens: u64,
    /// Diagnostics: total snapshots inserted.
    inserts: u64,
    /// Diagnostics: total snapshots evicted.
    evictions: u64,
}

impl PrefixIndex {
    /// An index holding at most `max_tokens` prompt tokens of
    /// snapshots (0 = the feature is off: lookups always miss).
    pub fn new(max_tokens: usize) -> PrefixIndex {
        PrefixIndex {
            root: Node::default(),
            cached_tokens: 0,
            max_tokens,
            tick: 0,
            hits: 0,
            evictions: 0,
            inserts: 0,
            reused_tokens: 0,
        }
    }

    /// Whether the feature is enabled (budget > 0).
    pub fn enabled(&self) -> bool {
        self.max_tokens > 0
    }

    /// Live snapshot count.
    pub fn entries(&self) -> usize {
        Self::count_snaps(&self.root)
    }

    /// Total cached tokens (the memory budget gauge).
    pub fn tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Lookups that returned a non-zero prefix (counter).
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Tokens returned by lookups over the index lifetime (counter).
    pub fn reused_tokens(&self) -> u64 {
        self.reused_tokens
    }

    /// Snapshots inserted over the index lifetime (counter).
    pub fn inserts(&self) -> u64 {
        self.inserts
    }

    /// Snapshots evicted over the index lifetime (counter).
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    fn count_snaps(n: &Node) -> usize {
        n.children.values().map(Self::count_snaps).sum::<usize>() + n.snap.is_some() as usize
    }

    /// Longest cached prefix of `prompt`: walks the prompt through the
    /// tree, touching LRU stamps, and returns the length of the deepest
    /// prefix with a live snapshot (0 on a miss). Never returns
    /// `prompt.len()`-or-more callers cannot act on: the *caller*
    /// clamps the result below the prompt length before scheduling
    /// reuse (the engine must always prefill at least one token).
    pub fn lookup(&mut self, prompt: &[u32]) -> usize {
        if !self.enabled() {
            return 0;
        }
        self.tick += 1;
        let now = self.tick;
        let mut best = 0usize;
        let mut node = &mut self.root;
        for (i, &t) in prompt.iter().enumerate() {
            let Some(child) = node.children.get_mut(&t) else {
                break;
            };
            if child.snap.is_some() {
                child.last_used = now;
                best = i + 1;
            }
            node = child;
        }
        if best > 0 {
            self.hits += 1;
            self.reused_tokens += best as u64;
        }
        best
    }

    /// The deepest snapshot of `prompt[..at_most]` still live, for the
    /// engine's restore: returns `(depth, snapshot)` with
    /// `depth <= at_most`, or `(0, None)` on a miss (the entry was
    /// evicted between the coordinator's admission lookup and this
    /// step's execution — the caller falls back to a cold prefill of
    /// the merged slice). Does NOT touch LRU stamps (the admission
    /// lookup already did).
    pub fn take_at_most(
        &self,
        prompt: &[u32],
        at_most: usize,
    ) -> (usize, Option<Arc<PrefixState>>) {
        let limit = at_most.min(prompt.len());
        let mut best: Option<(usize, Arc<PrefixState>)> = None;
        let mut node = &self.root;
        for (i, &t) in prompt[..limit].iter().enumerate() {
            let Some(child) = node.children.get(&t) else {
                break;
            };
            if let Some(s) = &child.snap {
                best = Some((i + 1, Arc::clone(s)));
            }
            node = child;
        }
        match best {
            Some((d, s)) => (d, Some(s)),
            None => (0, None),
        }
    }

    /// Inserts (or replaces) the snapshot for `prompt`. Evicts LRU
    /// entries until the token budget holds, then prunes snapshot-less
    /// leaves. A zero-length prompt or a disabled index is a no-op.
    pub fn insert(&mut self, prompt: &[u32], snap: Arc<PrefixState>) {
        if !self.enabled() || prompt.is_empty() || snap.len != prompt.len() {
            return;
        }
        self.tick += 1;
        let now = self.tick;
        // Walk (creating edges) to the prompt's node.
        let mut node = &mut self.root;
        for &t in prompt {
            node = node.children.entry(t).or_default();
        }
        if let Some(old) = node.snap.replace(snap) {
            self.cached_tokens = self.cached_tokens.saturating_sub(old.len);
        } else {
            self.inserts += 1;
        }
        node.last_used = now;
        self.cached_tokens += prompt.len();
        self.evict();
        Self::prune_node(&mut self.root);
    }

    /// LRU eviction down to the token budget: repeatedly drops the
    /// least-recently-used snapshot anywhere in the tree.
    fn evict(&mut self) {
        while self.cached_tokens > self.max_tokens {
            let (path, len) = match Self::lru_path(&self.root, &mut Vec::new()) {
                Some((_, p, l)) => (p, l),
                None => break,
            };
            let mut node = &mut self.root;
            for &t in &path {
                node = node.children.get_mut(&t).expect("lru path exists");
            }
            if node.snap.take().is_some() {
                self.cached_tokens = self.cached_tokens.saturating_sub(len);
                self.evictions += 1;
            }
        }
    }

    /// Finds the LRU snapshot-bearing node: returns `(last_used stamp,
    /// edge path from the root, snapshot length)` for the node with the
    /// smallest stamp anywhere in the subtree.
    fn lru_path(node: &Node, path: &mut Vec<u32>) -> Option<(u64, Vec<u32>, usize)> {
        let mut best: Option<(u64, Vec<u32>, usize)> = node
            .snap
            .as_ref()
            .map(|s| (node.last_used, path.clone(), s.len));
        for (&t, child) in &node.children {
            path.push(t);
            if let Some(found) = Self::lru_path(child, path) {
                if best.as_ref().map(|b| found.0 < b.0).unwrap_or(true) {
                    best = Some(found);
                }
            }
            path.pop();
        }
        best
    }

    /// Bottom-up pruning of snapshot-less, childless nodes (called by
    /// [`PrefixIndex::insert`] after eviction).
    fn prune_node(node: &mut Node) {
        let keys: Vec<u32> = node.children.keys().copied().collect();
        for k in keys {
            if let Some(child) = node.children.get_mut(&k) {
                Self::prune_node(child);
                if child.snap.is_none() && child.children.is_empty() {
                    node.children.remove(&k);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal stand-in snapshot with the given length.
    fn snap(len: usize) -> Arc<PrefixState> {
        Arc::new(PrefixState {
            len,
            kv_shadow: Vec::new(),
            indexer_k: Vec::new(),
            rings: Vec::new(),
        })
    }

    #[test]
    fn lookup_exact_partial_and_miss() {
        let mut ix = PrefixIndex::new(1024);
        assert_eq!(ix.lookup(&[1, 2, 3]), 0, "empty index misses");
        ix.insert(&[1, 2, 3, 4, 5], snap(5));
        assert_eq!(ix.lookup(&[1, 2, 3, 4, 5, 6]), 5, "longer prompt reuses 5");
        // A shorter prompt cannot reuse a longer cached prefix (the
        // snapshot lives at depth 5, beyond its length).
        assert_eq!(ix.lookup(&[1, 2, 3, 4]), 0);
        // Inserting the shorter prefix makes it reusable.
        ix.insert(&[1, 2], snap(2));
        assert_eq!(ix.lookup(&[1, 2, 3, 4]), 2);
        // A prompt diverging after a cached prefix still reuses it.
        assert_eq!(ix.lookup(&[1, 2, 9]), 2);
        assert_eq!(ix.lookup(&[9, 1, 2]), 0, "no shared leading tokens");
        assert_eq!(ix.hits(), 3);
        assert_eq!(ix.reused_tokens(), 9);
        assert_eq!(ix.entries(), 2);
        assert_eq!(ix.tokens(), 7);
    }

    #[test]
    fn take_at_most_respects_limit_and_survives_eviction() {
        let mut ix = PrefixIndex::new(1024);
        ix.insert(&[1, 2, 3, 4], snap(4));
        let (d, s) = ix.take_at_most(&[1, 2, 3, 4, 5], 4);
        assert_eq!((d, s.is_some()), (4, true));
        // A shorter limit falls back to the deepest snapshot within it.
        let (d, s) = ix.take_at_most(&[1, 2, 3, 4, 5], 2);
        assert_eq!((d, s.is_none()), (0, true), "no snapshot within 2 tokens");
        // After replacing with a shorter prefix at [1,2], limit 4 uses it.
        ix.insert(&[1, 2], snap(2));
        let (d, _) = ix.take_at_most(&[1, 2, 3, 4, 5], 4);
        assert_eq!(d, 4, "the deeper snapshot at 4 still wins");
        // Evict everything: the only entry cannot fit a 1-token budget.
        ix = PrefixIndex::new(1);
        ix.insert(&[1, 2, 3], snap(3));
        assert_eq!(
            ix.tokens(),
            0,
            "a 3-token snapshot cannot fit a 1-token budget"
        );
        assert_eq!(ix.entries(), 0);
    }

    #[test]
    fn lru_eviction_order_and_pruning() {
        let mut ix = PrefixIndex::new(10);
        ix.insert(&[1, 1], snap(2)); // A
        ix.insert(&[2, 2], snap(2)); // B
        ix.insert(&[3, 3], snap(2)); // C
        assert_eq!(ix.tokens(), 6);
        // Touch A and B (lookups refresh their stamps).
        assert_eq!(ix.lookup(&[1, 1, 9]), 2);
        assert_eq!(ix.lookup(&[2, 2, 9]), 2);
        // Inserting 6 more tokens evicts C (the untouched LRU) first.
        ix.insert(&[4, 4, 4, 4, 4, 4], snap(6));
        assert_eq!(ix.tokens(), 10);
        assert_eq!(ix.lookup(&[3, 3]), 0, "C evicted");
        assert_eq!(ix.lookup(&[1, 1, 9]), 2, "A survives (touched)");
        assert_eq!(ix.lookup(&[2, 2, 9]), 2, "B survives (touched)");
        assert_eq!(ix.evictions(), 1);
        // The evicted branch was pruned (no snapshot, no children).
        assert_eq!(ix.entries(), 3);
        // Budget of 0 disables everything.
        let mut off = PrefixIndex::new(0);
        assert!(!off.enabled());
        off.insert(&[1], snap(1));
        assert_eq!(off.lookup(&[1]), 0);
        assert_eq!(off.entries(), 0);
    }

    #[test]
    fn replace_updates_accounting() {
        let mut ix = PrefixIndex::new(1024);
        ix.insert(&[5, 6, 7], snap(3));
        ix.insert(&[5, 6, 7], snap(3)); // replace, not add
        assert_eq!(ix.entries(), 1);
        assert_eq!(ix.tokens(), 3);
        assert_eq!(ix.inserts(), 1);
        // A mismatched length is rejected (caller bug guard).
        ix.insert(&[9], snap(3));
        assert_eq!(ix.entries(), 1);
        ix.insert(&[], snap(0));
        assert_eq!(ix.entries(), 1);
    }
}
