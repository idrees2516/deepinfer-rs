//! Sliding-window KV row ring cache for YOCO decoder self-attention.
//!
//! YOCO ("You Only Cache Once", Sun et al., arXiv:2405.05254) replaces the
//! self-attention half of the decoder with layers whose keys/values are
//! only ever consumed inside a sliding window, so each such layer retains
//! just the last `window` rows per sequence (DeepInfer upstream README:
//! "YOCO SWA bounded replay"). MTP draft verification may reject suffix
//! tokens, so the cache can also drop its newest rows before replaying
//! from an earlier position (the `max_backtrack` headroom in
//! [`crate::SwaConfig`]).
//!
//! The store is a `VecDeque` ring of owned `[dim]` rows: pushes and
//! evictions are O(1), rows are allocated as they are pushed (never
//! `window * dim` up front, so constructing a cache cannot be used for a
//! denial-of-service), and the logical row order is fully deterministic.

use std::collections::VecDeque;

use deepinfer_core::{ensure, Result};

/// Ring buffer of the most recent decoder KV rows (one cache per
/// self-attention layer per sequence), supporting MTP rollback.
///
/// Rows are retained in logical order oldest..newest, at most `window` of
/// them (the `window` passed to [`SwaRingCache::new`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SwaRingCache {
    window: usize,
    dim: usize,
    rows: VecDeque<Vec<f32>>,
}

impl SwaRingCache {
    /// Creates a cache that retains up to `window` rows of width `dim`.
    ///
    /// No allocation happens here (rows are stored as they are pushed), so
    /// any `window` / `dim` — including `0` — is accepted: a
    /// `window == 0` cache drops every pushed row and stays empty forever.
    pub fn new(window: usize, dim: usize) -> SwaRingCache {
        SwaRingCache {
            window,
            dim,
            rows: VecDeque::new(),
        }
    }

    /// Number of retained rows (always `<= window`).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns `true` when no rows are retained.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Appends `row` as the newest row, validating its width; when the
    /// cache already holds `window` rows the oldest is evicted.
    ///
    /// # Errors
    /// [`deepinfer_core::Error::DimensionMismatch`] when
    /// `row.len() != dim`; the cache is left untouched.
    pub fn push(&mut self, row: &[f32]) -> Result<()> {
        ensure!(
            row.len() == self.dim,
            DimensionMismatch,
            "ring row has width {} but cache dim is {}",
            row.len(),
            self.dim
        );
        if self.window == 0 {
            // Zero capacity: the row is evicted the moment it arrives.
            return Ok(());
        }
        if self.rows.len() == self.window {
            self.rows.pop_front();
        }
        self.rows.push_back(row.to_vec());
        Ok(())
    }

    /// Drops the `n` newest rows (MTP rollback of rejected draft tokens),
    /// leaving older rows untouched.
    ///
    /// # Errors
    /// [`deepinfer_core::Error::InvalidArgument`] when `n > len`; the
    /// cache is left untouched.
    pub fn rollback(&mut self, n: usize) -> Result<()> {
        ensure!(
            n <= self.rows.len(),
            InvalidArgument,
            "cannot roll back {} rows: only {} retained",
            n,
            self.rows.len()
        );
        for _ in 0..n {
            self.rows.pop_back();
        }
        Ok(())
    }

    /// Returns the retained rows in logical order oldest..newest (at most
    /// `window` rows).
    pub fn window_rows(&self) -> Vec<Vec<f32>> {
        self.rows.iter().cloned().collect()
    }

    /// Absolute position of the oldest retained row, given `newest_pos` —
    /// the absolute position of the newest retained row:
    /// `newest_pos + 1 - len`.
    ///
    /// For an empty cache this is `newest_pos + 1` (the position the next
    /// row would occupy). The subtraction saturates at `0` for
    /// inconsistent inputs (`newest_pos + 1 < len`) instead of panicking.
    pub fn start_position(&self, newest_pos: usize) -> usize {
        newest_pos.saturating_add(1).saturating_sub(self.rows.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::prop;
    use deepinfer_core::Error;
    use std::collections::VecDeque;

    #[test]
    fn push_evicts_oldest_beyond_window() {
        let mut c = SwaRingCache::new(3, 1);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        for v in 1..=5usize {
            c.push(&[v as f32]).unwrap();
            assert_eq!(c.len(), v.min(3));
        }
        // Rows 1 and 2 were evicted; 3, 4, 5 remain, oldest first.
        assert_eq!(c.window_rows(), vec![vec![3.0], vec![4.0], vec![5.0]]);
    }

    #[test]
    fn window_rows_are_oldest_to_newest() {
        let mut c = SwaRingCache::new(4, 2);
        c.push(&[1.0, 2.0]).unwrap();
        c.push(&[3.0, 4.0]).unwrap();
        c.push(&[5.0, 6.0]).unwrap();
        assert_eq!(
            c.window_rows(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
    }

    #[test]
    fn rollback_after_eviction() {
        let mut c = SwaRingCache::new(3, 1);
        for v in 1..=5usize {
            c.push(&[v as f32]).unwrap();
        }
        // Retained: 3, 4, 5. Rolling back 2 leaves row 3 (an evicted-era
        // row must NOT reappear).
        c.rollback(2).unwrap();
        assert_eq!(c.window_rows(), vec![vec![3.0]]);
        // Refill after rollback re-applies eviction.
        for v in 6..=9usize {
            c.push(&[v as f32]).unwrap();
        }
        assert_eq!(c.window_rows(), vec![vec![7.0], vec![8.0], vec![9.0]]);
    }

    #[test]
    fn rollback_to_empty_and_reuse() {
        let mut c = SwaRingCache::new(4, 2);
        c.push(&[1.0, 2.0]).unwrap();
        c.push(&[3.0, 4.0]).unwrap();
        c.rollback(2).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert!(c.window_rows().is_empty());
        // rollback(0) is a documented no-op.
        c.rollback(0).unwrap();
        assert!(c.is_empty());
        // The cache is reusable after a full rollback.
        c.push(&[5.0, 6.0]).unwrap();
        assert_eq!(c.window_rows(), vec![vec![5.0, 6.0]]);
    }

    #[test]
    fn over_rollback_errors_and_preserves_state() {
        let mut c = SwaRingCache::new(8, 1);
        c.push(&[1.0]).unwrap();
        c.push(&[2.0]).unwrap();
        assert!(matches!(c.rollback(3), Err(Error::InvalidArgument(_))));
        assert_eq!(c.len(), 2);
        assert_eq!(c.window_rows(), vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn push_wrong_dim_errors_and_preserves_state() {
        let mut c = SwaRingCache::new(4, 3);
        assert!(matches!(
            c.push(&[1.0, 2.0]),
            Err(Error::DimensionMismatch(_))
        ));
        assert!(c.is_empty());
        assert!(c.push(&[1.0, 2.0, 3.0]).is_ok());
        // Zero-width row for a dim-3 cache is also a mismatch.
        assert!(matches!(c.push(&[]), Err(Error::DimensionMismatch(_))));
        assert_eq!(c.len(), 1);
        assert_eq!(c.window_rows(), vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn start_position_math() {
        let mut c = SwaRingCache::new(10, 1);
        // Empty cache: newest_pos + 1 (saturating on usize::MAX).
        assert_eq!(c.start_position(5), 6);
        assert_eq!(c.start_position(usize::MAX), usize::MAX);
        for v in 0..3usize {
            c.push(&[v as f32]).unwrap();
        }
        // len 3: 100 + 1 - 3 = 98.
        assert_eq!(c.start_position(100), 98);
        // Exactly at the origin: 2 + 1 - 3 = 0.
        assert_eq!(c.start_position(2), 0);
        // Inconsistent inputs saturate at 0 instead of panicking.
        assert_eq!(c.start_position(1), 0);
        assert_eq!(c.start_position(0), 0);
    }

    #[test]
    fn zero_window_drops_every_row() {
        let mut c = SwaRingCache::new(0, 2);
        assert!(c.push(&[1.0, 2.0]).is_ok());
        assert!(c.push(&[3.0, 4.0]).is_ok());
        assert!(c.is_empty());
        assert!(c.window_rows().is_empty());
        assert!(c.rollback(0).is_ok());
        assert!(matches!(c.rollback(1), Err(Error::InvalidArgument(_))));
        assert_eq!(c.start_position(9), 10);
    }

    #[test]
    fn property_matches_vecdeque_model() {
        prop("swa_ring_model", 0x512, 200, |_i, rng| {
            let window = rng.uniform_usize(1, 8);
            let dim = rng.uniform_usize(1, 4);
            let mut cache = SwaRingCache::new(window, dim);
            // Independent model: (position, row) pairs in a plain VecDeque
            // with the same eviction / rollback rules.
            let mut model: VecDeque<(usize, Vec<f32>)> = VecDeque::new();
            let mut next_pos = 0usize;
            for step in 0..80 {
                if rng.next_f32() < 0.6 {
                    let row: Vec<f32> =
                        (0..dim).map(|k| (next_pos * 16 + k) as f32).collect();
                    cache.push(&row)?;
                    model.push_back((next_pos, row));
                    next_pos += 1;
                    while model.len() > window {
                        model.pop_front();
                    }
                } else {
                    let n = rng.uniform_usize(0, model.len() + 1);
                    if n <= model.len() {
                        cache.rollback(n)?;
                        for _ in 0..n {
                            model.pop_back();
                        }
                    } else {
                        // Over-rollback must Err and preserve state.
                        ensure!(cache.rollback(n).is_err(), Internal, "over-rollback");
                    }
                }
                ensure!(cache.len() == model.len(), Internal, "len at step {step}");
                ensure!(cache.len() <= window, Internal, "capacity at step {step}");
                ensure!(
                    cache.is_empty() == model.is_empty(),
                    Internal,
                    "is_empty at step {step}"
                );
                let got = cache.window_rows();
                let want: Vec<Vec<f32>> = model.iter().map(|(_, r)| r.clone()).collect();
                ensure!(got == want, Internal, "rows at step {step}");
                // start_position formula, including edge inputs.
                for &p in &[0usize, 1, 7, 33] {
                    let expect = p.saturating_add(1).saturating_sub(model.len());
                    ensure!(
                        cache.start_position(p) == expect,
                        Internal,
                        "start_position({p}) at step {step}"
                    );
                }
            }
            Ok(())
        });
    }
}
