//! The GPU prefix cache: a block-granular radix tree over the paged
//! serve pools.
//!
//! This is the paged-serve engine's answer to vLLM's automatic prefix
//! caching and SGLang's RadixAttention, adapted to this engine's
//! architecture. Instead of snapshotting cache *contents* (the CPU
//! engine's [`crate::model::PrefixState`] approach), the GPU engine
//! shares **physical pages**: the five per-layer pools are addressed
//! through per-sequence page tables, so a cached prefix is just a list
//! of page ids that a new sequence can point at. A page referenced by a
//! live sequence *and* the radix tree is returned to the free list only
//! when the last lease is released — reference-counted page sharing
//! inside the pools (the layout the shared free list was designed to
//! grow into).
//!
//! Structure: one tree node per [`deepinfer_kernels::SERVE_PAGE`]-token
//! block, keyed by a chained FNV-1a hash of the token blocks (the
//! vLLM hash-chain scheme) and verified against the exact block tokens
//! (a collision cannot alias two different prompts silently). Each node
//! holds:
//!
//! * the physical page id of its block (leased; the content is frozen
//!   — copy-on-write protects it, see [`crate::serve`]),
//! * the lightning-scale (`k_sf`) bytes that complete inside the block
//!   (a scale byte belongs to the block in which its `quant_block`-row
//!   window closes, so concatenating a path's segments reproduces the
//!   prefix's scale prefix exactly),
//! * a device snapshot of the SWA rings at the block's end — the only
//!   state the paged pools do not hold. A node with rings is a **match
//!   endpoint**: restoring a prefix needs pages, scales *and* rings, so
//!   matches stop at the deepest ring-bearing node.
//!
//! Eviction is LRU over leaves, bounded by a budget in page-units
//! (a node costs its page plus its ring snapshot converted at
//! `ring_bytes / page_bytes` per unit). The tree is host-side metadata
//! only — no kernel launches — so it is fully unit-testable without a
//! device; the device copies (ring snapshots, page-table uploads) live
//! in [`crate::serve`].
//!
//! Workloads that win: multi-turn conversations (every turn extends
//! the previous one), few-shot prompts with a shared system preamble,
//! and agentic loops re-issuing near-identical prompts.

use std::collections::HashMap;

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::Buffer;

/// Tokens per radix block (kept in lockstep with the serve page size).
const PAGE: usize = deepinfer_kernels::SERVE_PAGE;

/// FNV-1a 64-bit constants (the block hash; fast, dependency-free, and
/// collision-guarded by exact token comparison on every walk).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Mixes one 64-token block into the running chained hash.
fn chain_block(h: u64, block: &[u32]) -> u64 {
    let mut x = h;
    for &t in block {
        x ^= u64::from(t);
        x = x.wrapping_mul(FNV_PRIME);
    }
    // A final tag mix: block boundaries must not be prefix-truncatable.
    x ^= 0x5a5a_5a5a_5a5a_5a5a;
    x.wrapping_mul(FNV_PRIME)
}

/// The `k_sf` byte range owned by block `b`: byte `i` covers rows
/// `[i*qb, (i+1)*qb)` and belongs to the block in which that window
/// **completes** (integer division on both ends implements exactly
/// that: a byte appears in the segment of the block whose end row
/// first reaches `(i+1)*qb`).
///
/// Concatenating the segments of blocks `0..n` yields the scale bytes
/// `0..floor(n*PAGE/qb)` — precisely the bytes whose windows lie fully
/// inside the prefix.
pub(crate) fn ksf_seg_range(b: usize, qb: usize) -> (usize, usize) {
    (b * PAGE / qb, (b + 1) * PAGE / qb)
}

/// One radix node: the cache state of a single 64-token block.
struct Node {
    /// The block's tokens (exact-match guard against hash collisions).
    block: Vec<u32>,
    /// Chained hash through this block (the child key of the parent).
    hash: u64,
    /// The leased physical page id (frozen content).
    page: u32,
    /// Scale bytes completing inside this block (see [`ksf_seg_range`]).
    k_sf: Vec<u8>,
    /// SWA-ring device snapshot at the block's end; `Some` makes the
    /// node a match endpoint. An empty `Vec` is a valid snapshot for
    /// zero cross-layer configs.
    rings: Option<Vec<Buffer>>,
    /// Child edges keyed by the child's chained hash.
    children: HashMap<u64, usize>,
    /// Arena index of the parent (0 = the root).
    parent: usize,
    /// LRU stamp of the last walk through this node.
    last_used: u64,
    /// Budget cost: 1 page-unit plus the ring snapshot's page-units.
    units: usize,
    /// Arena slot in use.
    alive: bool,
}

/// Counters and gauges mirroring the CPU engine's prefix index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Match lookups that returned a non-zero prefix.
    pub hits: u64,
    /// Tokens returned by lookups over the tree lifetime.
    pub reused_tokens: u64,
    /// Block nodes inserted (created, not re-touched).
    pub inserts: u64,
    /// Block nodes evicted.
    pub evictions: u64,
    /// Live block nodes.
    pub nodes: usize,
    /// Cached prefix tokens (nodes x page size).
    pub cached_tokens: usize,
    /// The token budget (0 = disabled).
    pub budget_tokens: usize,
}

/// The radix tree (host metadata; device payloads ride in the nodes).
pub(crate) struct GpuPrefixTree {
    /// Node arena; index 0 is the root (no block of its own).
    nodes: Vec<Node>,
    /// Recycled arena slots.
    free_slots: Vec<usize>,
    /// Budget in page-units; 0 disables insert and lookup.
    budget_units: usize,
    /// Ring-snapshot cost of one node in page-units.
    ring_units: usize,
    /// Live cost in page-units.
    cached_units: usize,
    /// Monotonic clock for LRU stamps.
    tick: u64,
    /// Diagnostics (see [`PrefixCacheStats`]).
    hits: u64,
    reused_tokens: u64,
    inserts: u64,
    evictions: u64,
}

impl GpuPrefixTree {
    /// An empty tree with the given budget (in page-units; 0 = off) and
    /// ring-snapshot cost (in page-units per ring-bearing node).
    pub(crate) fn new(budget_units: usize, ring_units: usize) -> GpuPrefixTree {
        GpuPrefixTree {
            nodes: vec![Node {
                block: Vec::new(),
                hash: 0,
                page: 0,
                k_sf: Vec::new(),
                rings: None,
                children: HashMap::new(),
                parent: usize::MAX,
                last_used: 0,
                units: 0,
                alive: true,
            }],
            free_slots: Vec::new(),
            budget_units,
            ring_units,
            cached_units: 0,
            tick: 0,
            hits: 0,
            reused_tokens: 0,
            inserts: 0,
            evictions: 0,
        }
    }

    /// Whether the cache is enabled (budget > 0).
    pub(crate) fn enabled(&self) -> bool {
        self.budget_units > 0
    }

    /// Retarget the budget (page-units) and ring cost; callers evict to
    /// the new budget afterwards.
    pub(crate) fn set_budget(&mut self, budget_units: usize, ring_units: usize) {
        self.budget_units = budget_units;
        self.ring_units = ring_units;
    }

    /// Live block-node count.
    pub(crate) fn nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.alive).count() - 1 // minus the root
    }

    /// Whether any node holds a page lease.
    pub(crate) fn holds_pages(&self) -> bool {
        self.nodes.iter().any(|n| n.alive && !n.block.is_empty())
    }

    /// Stats snapshot (budget reported back in tokens).
    pub(crate) fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hits: self.hits,
            reused_tokens: self.reused_tokens,
            inserts: self.inserts,
            evictions: self.evictions,
            nodes: self.nodes(),
            cached_tokens: self.nodes() * PAGE,
            budget_tokens: self.budget_units * PAGE,
        }
    }

    /// Advance the LRU clock one step.
    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Longest cached prefix of `tokens` that can be *restored* — the
    /// deepest chain of matching blocks whose final node carries a ring
    /// snapshot — in **blocks** (0 on a miss or when disabled). Walks
    /// whole blocks only; the caller guarantees at least one prompt
    /// token remains beyond the returned prefix.
    pub(crate) fn match_blocks(&mut self, tokens: &[u32]) -> usize {
        if !self.enabled() {
            return 0;
        }
        let now = self.touch();
        let mut best = 0usize;
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        let full = tokens.len() / PAGE;
        for b in 0..full {
            let block = &tokens[b * PAGE..(b + 1) * PAGE];
            h = chain_block(h, block);
            let Some(&child) = self.nodes[idx].children.get(&h) else {
                break;
            };
            let node = &self.nodes[child];
            if node.block != block {
                break; // hash collision: not the same tokens
            }
            if node.rings.is_some() {
                best = b + 1;
            }
            idx = child;
        }
        // Touch every walked node (they were all just validated).
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        for b in 0..full {
            let block = &tokens[b * PAGE..(b + 1) * PAGE];
            h = chain_block(h, block);
            let Some(&child) = self.nodes[idx].children.get(&h) else {
                break;
            };
            if self.nodes[child].block != block {
                break;
            }
            self.nodes[child].last_used = now;
            idx = child;
        }
        if best > 0 {
            self.hits += 1;
            self.reused_tokens += (best * PAGE) as u64;
        }
        best
    }

    /// Arena path (root-exclusive) of the first `blocks` blocks of
    /// `tokens`: `Some` only when every block matches a live node.
    fn path_idx(&self, tokens: &[u32], blocks: usize) -> Option<Vec<usize>> {
        let mut path = Vec::with_capacity(blocks);
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        for b in 0..blocks {
            let block = &tokens[b * PAGE..(b + 1) * PAGE];
            h = chain_block(h, block);
            let &child = self.nodes[idx].children.get(&h)?;
            let node = &self.nodes[child];
            if !node.alive || node.block != block {
                return None;
            }
            path.push(child);
            idx = child;
        }
        Some(path)
    }

    /// The restore payload for `blocks` matched blocks: the physical
    /// page ids and the concatenated `k_sf` bytes of the prefix.
    pub(crate) fn restore_payload(&self, tokens: &[u32], blocks: usize) -> (Vec<u32>, Vec<u8>) {
        let Some(path) = self.path_idx(tokens, blocks) else {
            return (Vec::new(), Vec::new());
        };
        let pages: Vec<u32> = path.iter().map(|&i| self.nodes[i].page).collect();
        let ksf: Vec<u8> = path
            .iter()
            .flat_map(|&i| self.nodes[i].k_sf.iter().copied())
            .collect();
        (pages, ksf)
    }

    /// The ring snapshot of the endpoint node at `blocks` matched
    /// blocks (the device source for the restore copies).
    pub(crate) fn endpoint_rings(&self, tokens: &[u32], blocks: usize) -> Option<&[Buffer]> {
        let path = self.path_idx(tokens, blocks)?;
        self.nodes[*path.last()?].rings.as_deref()
    }

    /// Leading blocks of `tokens` already cached (LRU-touched): the
    /// insert starting point.
    pub(crate) fn shared_blocks(&mut self, tokens: &[u32]) -> usize {
        let now = self.touch();
        let mut shared = 0usize;
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        let full = tokens.len() / PAGE;
        for b in 0..full {
            let block = &tokens[b * PAGE..(b + 1) * PAGE];
            h = chain_block(h, block);
            let Some(&child) = self.nodes[idx].children.get(&h) else {
                break;
            };
            let node = &self.nodes[child];
            if node.block != block {
                break;
            }
            self.nodes[child].last_used = now;
            idx = child;
            shared = b + 1;
        }
        shared
    }

    /// Leading blocks of `tokens` matching cached nodes exactly, **without
    /// touching LRU stamps** — the copy-on-write guard's identical-rewrite
    /// fast path (a scheduler rewind re-prefills the same prompt; blocks
    /// whose frozen content is reproduced byte-for-byte need no copy).
    pub(crate) fn matching_blocks(&self, tokens: &[u32]) -> usize {
        let mut matched = 0usize;
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        let full = tokens.len() / PAGE;
        for b in 0..full {
            let block = &tokens[b * PAGE..(b + 1) * PAGE];
            h = chain_block(h, block);
            let Some(&child) = self.nodes[idx].children.get(&h) else {
                break;
            };
            let node = &self.nodes[child];
            if !node.alive || node.block != block {
                break;
            }
            idx = child;
            matched = b + 1;
        }
        matched
    }

    /// Create the node for block `b` of `tokens`. Requires the parent
    /// path to exist (`b == 0` or block `b-1` cached — the
    /// [`Self::shared_blocks`] contract) and block `b` itself absent.
    /// The tree takes a lease on `page`; `rings` (when `Some`) makes
    /// the node a match endpoint.
    ///
    /// # Errors
    /// [`Error::Internal`] on an out-of-order insert or a hash
    /// collision with different tokens (both caller-bug guards).
    pub(crate) fn insert_block(
        &mut self,
        tokens: &[u32],
        b: usize,
        page: u32,
        k_sf: Vec<u8>,
        rings: Option<Vec<Buffer>>,
    ) -> Result<()> {
        ensure!(self.enabled(), Internal, "prefix cache disabled");
        ensure!(
            b * PAGE + PAGE <= tokens.len(),
            Internal,
            "insert past full blocks"
        );
        let block: Vec<u32> = tokens[b * PAGE..(b + 1) * PAGE].to_vec();
        // Walk to the parent (depth b).
        let mut idx = 0usize;
        let mut h = FNV_OFFSET;
        for i in 0..b {
            let blk = &tokens[i * PAGE..(i + 1) * PAGE];
            h = chain_block(h, blk);
            let &child = self.nodes[idx]
                .children
                .get(&h)
                .ok_or_else(|| Error::internal("prefix insert out of order"))?;
            ensure!(
                self.nodes[child].block == blk,
                Internal,
                "prefix insert path mismatch"
            );
            idx = child;
        }
        let hash = chain_block(h, &block);
        ensure!(
            !self.nodes[idx].children.contains_key(&hash),
            Internal,
            "prefix insert of an existing block"
        );
        let units = 1 + if rings.is_some() { self.ring_units } else { 0 };
        let node = Node {
            last_used: self.touch(),
            hash,
            block,
            page,
            k_sf,
            rings,
            children: HashMap::new(),
            parent: idx,
            units,
            alive: true,
        };
        let slot = match self.free_slots.pop() {
            Some(s) => {
                self.nodes[s] = node;
                s
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        };
        self.nodes[idx].children.insert(hash, slot);
        self.cached_units += units;
        self.inserts += 1;
        Ok(())
    }

    /// Drop the least-recently-used leaf node; returns its page (the
    /// ended tree lease) or `None` when no evictable leaf exists.
    /// Ring snapshots are freed with the node (device memory).
    pub(crate) fn evict_lru_leaf(&mut self) -> Option<u32> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i > 0 && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i)?;
        self.remove_node(victim)
    }

    /// Evict LRU leaves until the budget holds; returns the pages whose
    /// tree lease ended (the caller decrements refcounts).
    pub(crate) fn evict_to_budget(&mut self) -> Vec<u32> {
        let mut freed = Vec::new();
        while self.cached_units > self.budget_units {
            match self.evict_lru_leaf() {
                Some(p) => freed.push(p),
                None => break,
            }
        }
        freed
    }

    /// Drop everything; returns all leased pages.
    pub(crate) fn clear(&mut self) -> Vec<u32> {
        let mut freed = Vec::new();
        while self.holds_pages() {
            match self.evict_lru_leaf() {
                Some(p) => freed.push(p),
                None => break,
            }
        }
        freed
    }

    /// Detach `victim` from its parent, tombstone the slot and return
    /// the released page. Only leaves may be removed (the caller's
    /// invariant; children would otherwise be orphaned mid-walk).
    fn remove_node(&mut self, victim: usize) -> Option<u32> {
        let node = &mut self.nodes[victim];
        if !node.alive || !node.children.is_empty() || victim == 0 {
            return None;
        }
        let page = node.page;
        let units = node.units;
        let hash = node.hash;
        let parent = node.parent;
        *node = Node {
            block: Vec::new(),
            hash: 0,
            page: 0,
            k_sf: Vec::new(),
            rings: None,
            children: HashMap::new(),
            parent: usize::MAX,
            last_used: 0,
            units: 0,
            alive: false,
        };
        self.free_slots.push(victim);
        self.cached_units = self.cached_units.saturating_sub(units);
        self.evictions += 1;
        if parent != usize::MAX {
            self.nodes[parent].children.remove(&hash);
        }
        Some(page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic tokens: `t(i) = (i * a + c) % 251`.
    fn toks(n: usize, a: u32, c: u32) -> Vec<u32> {
        (0..n).map(|i| (i as u32 * a + c) % 251).collect()
    }

    /// Insert helper: fake pages 1000+b, one k_sf byte per block, and
    /// empty ring snapshots (`Some(vec![])` = a zero-layer endpoint).
    /// Skips blocks already cached (the caller's shared-blocks walk).
    fn ins(tree: &mut GpuPrefixTree, tokens: &[u32], upto_blocks: usize) {
        let start = tree.shared_blocks(tokens);
        for b in start..upto_blocks {
            tree.insert_block(tokens, b, 1000 + b as u32, vec![b as u8], Some(vec![]))
                .unwrap();
        }
    }

    #[test]
    fn ksf_segments_compose_the_prefix() {
        // qb == PAGE: every block owns exactly one byte.
        for b in 0..6 {
            assert_eq!(ksf_seg_range(b, 64), (b, b + 1));
        }
        // qb == 2*PAGE: odd blocks own the shared byte, even blocks none.
        assert_eq!(ksf_seg_range(0, 128), (0, 0));
        assert_eq!(ksf_seg_range(1, 128), (0, 1));
        assert_eq!(ksf_seg_range(2, 128), (1, 1));
        assert_eq!(ksf_seg_range(3, 128), (1, 2));
        // qb == PAGE/2: two bytes per block.
        assert_eq!(ksf_seg_range(0, 32), (0, 2));
        assert_eq!(ksf_seg_range(3, 32), (6, 8));
        // qb == 32*PAGE (the tiny config's 2048): only block 31 owns byte 0.
        assert_eq!(ksf_seg_range(30, 2048), (0, 0));
        assert_eq!(ksf_seg_range(31, 2048), (0, 1));
        assert_eq!(ksf_seg_range(32, 2048), (1, 1));
        // Concatenated segments of blocks 0..n == bytes 0..n*PAGE/qb.
        for &qb in &[16usize, 32, 64, 128, 256, 2048] {
            for n in 1..8usize {
                let total: usize = (0..n)
                    .map(|b| ksf_seg_range(b, qb).1 - ksf_seg_range(b, qb).0)
                    .sum();
                assert_eq!(total, n * PAGE / qb, "qb={qb} n={n}");
            }
        }
    }

    #[test]
    fn match_exact_partial_and_miss() {
        let mut t = GpuPrefixTree::new(1024, 0);
        let a = toks(5 * PAGE, 7, 1);
        assert_eq!(t.match_blocks(&a), 0, "empty tree misses");
        ins(&mut t, &a, 5);
        // A longer prompt reuses all 5 blocks.
        let mut longer = a.clone();
        longer.extend(toks(PAGE, 9, 9));
        assert_eq!(t.match_blocks(&longer), 5);
        // A shorter prompt reuses its own full blocks (every node is a
        // ring-bearing endpoint — block-granular sharing).
        assert_eq!(t.match_blocks(&a[..4 * PAGE]), 4);
        // A diverging tail still reuses the shared blocks.
        let mut div = a[..3 * PAGE].to_vec();
        div.extend(toks(2 * PAGE, 11, 3));
        assert_eq!(t.match_blocks(&div), 3);
        // No shared leading tokens at all.
        assert_eq!(t.match_blocks(&toks(3 * PAGE, 13, 5)), 0);
        let st = t.stats();
        assert_eq!(st.hits, 3);
        assert_eq!(st.reused_tokens as usize, (5 + 4 + 3) * PAGE);
        assert_eq!(st.nodes, 5);
        // Trailing partial blocks are ignored by the match.
        let mut partial = a[..4 * PAGE].to_vec();
        partial.push(77);
        assert_eq!(
            t.match_blocks(&partial),
            4,
            "the partial tail is not matched"
        );
    }

    #[test]
    fn restore_payload_and_endpoint_rings() {
        let mut t = GpuPrefixTree::new(1024, 0);
        let a = toks(4 * PAGE, 5, 2);
        ins(&mut t, &a, 4);
        let (pages, ksf) = t.restore_payload(&a, 3);
        assert_eq!(pages, vec![1000, 1001, 1002]);
        assert_eq!(ksf, vec![0, 1, 2]);
        assert!(t.endpoint_rings(&a, 3).is_some());
        assert!(t.endpoint_rings(&a, 4).is_some());
        // A mismatching prefix yields nothing.
        assert_eq!(t.restore_payload(&toks(3 * PAGE, 6, 2), 3).0.len(), 0);
        assert!(t.endpoint_rings(&toks(3 * PAGE, 6, 2), 3).is_none());
    }

    #[test]
    fn rings_gate_the_match_endpoint() {
        let mut t = GpuPrefixTree::new(1024, 0);
        let a = toks(3 * PAGE, 3, 4);
        // Insert blocks 0-1 without rings (walk-through only), block 2 with.
        for b in 0..2 {
            t.insert_block(&a, b, 10 + b as u32, Vec::new(), None)
                .unwrap();
        }
        t.insert_block(&a, 2, 12, vec![2], Some(vec![])).unwrap();
        assert_eq!(t.match_blocks(&a), 3);
        let mut shorter = a[..2 * PAGE].to_vec();
        shorter.push(1);
        assert_eq!(
            t.match_blocks(&shorter),
            0,
            "the depth-2 node has no rings: not an endpoint"
        );
        // ... but it still extends: a deeper cached prompt matches through it.
        let mut longer = a.clone();
        longer.extend(toks(PAGE, 8, 8));
        t.insert_block(&longer, 3, 13, vec![3], Some(vec![]))
            .unwrap();
        assert_eq!(t.match_blocks(&longer), 4);
    }

    #[test]
    fn shared_blocks_and_reinsert_is_refused() {
        let mut t = GpuPrefixTree::new(1024, 0);
        let a = toks(3 * PAGE, 3, 4);
        assert_eq!(t.shared_blocks(&a), 0);
        ins(&mut t, &a, 2);
        assert_eq!(t.shared_blocks(&a), 2);
        // Re-inserting an existing block is a caller bug.
        assert!(t.insert_block(&a, 1, 99, Vec::new(), Some(vec![])).is_err());
        // Out-of-order inserts are refused.
        assert!(t.insert_block(&a, 3, 99, Vec::new(), Some(vec![])).is_err());
        // Extending the path works.
        ins(&mut t, &a, 3);
        assert_eq!(t.shared_blocks(&a), 3);
    }

    #[test]
    fn lru_leaf_eviction_respects_budget() {
        // Budget of 4 units with free rings: 4 nodes fit.
        let mut t = GpuPrefixTree::new(4, 0);
        let a = toks(2 * PAGE, 3, 1);
        let b = toks(2 * PAGE, 5, 1);
        ins(&mut t, &a, 2);
        ins(&mut t, &b, 2);
        assert_eq!(t.nodes(), 4);
        // Touch path a (shared_blocks refreshes LRU stamps).
        assert_eq!(t.shared_blocks(&a), 2);
        // Inserting 2 more nodes (path c) exceeds the budget; the
        // caller's evict-to-budget then drops the LRU leaves of path b
        // (b1 first, then b0 once childless), keeping path a.
        let c = toks(2 * PAGE, 7, 1);
        ins(&mut t, &c, 2);
        let freed = t.evict_to_budget();
        assert_eq!(freed.len(), 2, "two leaves evicted back to budget");
        assert_eq!(t.nodes(), 4, "still within the budget");
        assert_eq!(t.match_blocks(&a), 2, "touched path survives");
        assert_eq!(t.match_blocks(&b), 0, "cold path evicted");
        assert_eq!(t.match_blocks(&c), 2);
        assert_eq!(t.stats().evictions, 2);
        // Retargeting the budget down evicts to the new bound.
        t.set_budget(2, 0);
        let freed = t.evict_to_budget();
        assert_eq!(freed.len(), 2);
        assert_eq!(t.nodes(), 2);
    }

    #[test]
    fn pressure_valve_and_clear() {
        let mut t = GpuPrefixTree::new(1024, 0);
        let a = toks(3 * PAGE, 3, 1);
        ins(&mut t, &a, 3);
        assert!(t.holds_pages());
        // The pressure valve evicts the LRU *leaf* — the deepest node
        // of the single path (a2, page 1002): interior nodes with
        // children are not evictable.
        assert_eq!(t.evict_lru_leaf(), Some(1002));
        assert_eq!(t.match_blocks(&a), 2, "depth 3 lost its endpoint");
        let freed = t.clear();
        assert_eq!(freed, vec![1001, 1000]);
        assert!(!t.holds_pages());
        assert_eq!(t.nodes(), 0);
        assert_eq!(t.match_blocks(&a), 0);
        assert_eq!(t.evict_lru_leaf(), None);
    }

    #[test]
    fn disabled_tree_is_inert() {
        let mut t = GpuPrefixTree::new(0, 0);
        let a = toks(2 * PAGE, 3, 1);
        assert!(!t.enabled());
        assert_eq!(t.match_blocks(&a), 0);
        assert_eq!(t.shared_blocks(&a), 0);
        assert!(t.insert_block(&a, 0, 1, Vec::new(), Some(vec![])).is_err());
        assert_eq!(t.stats().budget_tokens, 0);
    }

    #[test]
    fn hash_chain_distinguishes_depths_and_blocks() {
        let x = toks(PAGE, 1, 0);
        let y = toks(PAGE, 1, 1);
        let h0 = chain_block(FNV_OFFSET, &x);
        let h1 = chain_block(h0, &x);
        assert_ne!(h0, h1, "depth matters");
        assert_ne!(chain_block(FNV_OFFSET, &x), chain_block(FNV_OFFSET, &y));
        assert_ne!(
            chain_block(FNV_OFFSET, &x),
            chain_block(chain_block(FNV_OFFSET, &y), &x)
        );
    }
}
