//! Sliding-window expert statistics (EPLB's `ExpertStats` counterpart: the
//! moving average of historical per-expert loads that feeds
//! `rebalance_experts` as the `weight` vector).

/// Moving average of per-expert batch counts over a fixed-size window.
///
/// `record` pushes one batch of per-expert counts into a ring buffer of at
/// most `window` entries (the oldest entry is evicted on overflow);
/// `average` returns the per-expert mean over the stored history (zeros
/// when nothing is recorded yet). The SPEC struct fields are:
/// `window` (capacity), `history` (stored batches, oldest first) and
/// `counts` (running per-expert sum over exactly the stored history, kept
/// incrementally so `average` is O(num_experts)).
///
/// Determinism: pure in-order `f64` accumulation (add on record, subtract
/// on eviction) — identical call sequences produce bit-identical averages.
///
/// Panics never: mismatched-length `record` calls are documented no-ops and
/// `window == 0` stores nothing.
#[derive(Debug, Clone)]
pub struct ExpertStats {
    /// Maximum number of stored batches.
    window: usize,
    /// Stored batches, oldest first (at most `window` entries).
    history: Vec<Vec<f64>>,
    /// Running per-expert sum over exactly the stored history.
    counts: Vec<f64>,
}

impl ExpertStats {
    /// Creates statistics for `num_experts` experts with a sliding window
    /// of `window` batches. `window == 0` disables history entirely (every
    /// `record` is dropped, `average` always returns zeros).
    pub fn new(num_experts: usize, window: usize) -> Self {
        ExpertStats {
            window,
            history: Vec::new(),
            counts: vec![0.0; num_experts],
        }
    }

    /// Records one batch of per-expert counts.
    ///
    /// The entry is appended to the history; when the history exceeds
    /// `window` entries the oldest is evicted and subtracted from the
    /// running sums. A call whose `batch_counts.len()` differs from
    /// `num_experts` is a documented **no-op** (the SPEC signature returns
    /// no error; silently ignoring malformed batches keeps the statistics
    /// well-formed without panicking).
    pub fn record(&mut self, batch_counts: &[u64]) {
        if batch_counts.len() != self.counts.len() || self.window == 0 {
            return;
        }
        let row: Vec<f64> = batch_counts.iter().map(|&c| c as f64).collect();
        for (acc, &v) in self.counts.iter_mut().zip(row.iter()) {
            *acc += v;
        }
        self.history.push(row);
        if self.history.len() > self.window {
            let evicted = self.history.remove(0);
            for (acc, &v) in self.counts.iter_mut().zip(evicted.iter()) {
                *acc -= v;
            }
        }
    }

    /// Per-expert mean over the stored history: `counts[e] / len(history)`.
    /// Returns zeros (one per expert) when no history is stored.
    pub fn average(&self) -> Vec<f64> {
        if self.history.is_empty() {
            return vec![0.0; self.counts.len()];
        }
        let n = self.history.len() as f64;
        self.counts.iter().map(|&c| c / n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::prop;

    /// Mirrors the incremental running-sum semantics exactly.
    fn reference_average(window: usize, batches: &[Vec<u64>]) -> Vec<f64> {
        let mut stats = ExpertStats::new(batches.first().map(|b| b.len()).unwrap_or(0), window);
        for b in batches {
            stats.record(b);
        }
        stats.average()
    }

    #[test]
    fn partial_history_average() {
        let mut s = ExpertStats::new(3, 4);
        assert_eq!(s.average(), vec![0.0, 0.0, 0.0]); // no history -> zeros
        s.record(&[1, 2, 3]);
        assert_eq!(s.average(), vec![1.0, 2.0, 3.0]); // single batch
        s.record(&[3, 4, 5]);
        assert_eq!(s.average(), vec![2.0, 3.0, 4.0]); // mean of two
    }

    #[test]
    fn window_eviction() {
        let mut s = ExpertStats::new(2, 2);
        s.record(&[10, 20]);
        s.record(&[30, 40]);
        s.record(&[50, 60]); // evicts [10, 20]
        assert_eq!(s.history.len(), 2);
        assert_eq!(s.history[0], vec![30.0, 40.0]);
        assert_eq!(s.history[1], vec![50.0, 60.0]);
        assert_eq!(s.counts, vec![80.0, 100.0]);
        let avg = s.average();
        assert!(
            (avg[0] - 40.0).abs() < 1e-12 && (avg[1] - 50.0).abs() < 1e-12,
            "average after eviction: {avg:?}"
        );
    }

    #[test]
    fn zero_window_stores_nothing() {
        let mut s = ExpertStats::new(3, 0);
        s.record(&[5, 5, 5]);
        s.record(&[1, 2, 3]);
        assert!(s.history.is_empty());
        assert_eq!(s.average(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn mismatched_record_is_noop() {
        let mut s = ExpertStats::new(3, 4);
        s.record(&[1, 2, 3]);
        s.record(&[9, 9]); // wrong length -> ignored
        s.record(&[]); // wrong length -> ignored
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.average(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn moving_average_property() {
        prop("eplb moving average", 0xE9_3A, 200, |_i, rng| {
            let num_experts = rng.uniform_usize(1, 6);
            let window = rng.uniform_usize(1, 5);
            let batches = rng.uniform_usize(1, 12);
            let mut s = ExpertStats::new(num_experts, window);
            let mut history: Vec<Vec<u64>> = Vec::new();
            for _ in 0..batches {
                let b: Vec<u64> = (0..num_experts)
                    .map(|_| rng.uniform_usize(0, 1000) as u64)
                    .collect();
                s.record(&b);
                history.push(b);
                if history.len() > window {
                    history.remove(0);
                }
                // Invariant: stored history matches the reference window.
                assert_eq!(s.history.len(), history.len());
                // Average over the stored history matches a mirrored
                // incremental computation...
                let avg = s.average();
                let mirrored = reference_average(window, &history);
                assert_eq!(avg.len(), num_experts);
                for e in 0..num_experts {
                    assert!(
                        (avg[e] - mirrored[e]).abs() < 1e-9,
                        "expert {e}: {avg:?} vs {mirrored:?}"
                    );
                    // ... and a direct recomputation (tolerance covers the
                    // incremental add/subtract rounding).
                    let direct: f64 =
                        history.iter().map(|h| h[e] as f64).sum::<f64>() / history.len() as f64;
                    assert!(
                        (avg[e] - direct).abs() < 1e-6,
                        "expert {e}: {avg:?} vs direct {direct}"
                    );
                    assert!(avg[e] >= 0.0 && avg[e].is_finite());
                }
            }
            Ok(())
        });
    }

    #[test]
    fn record_is_deterministic() {
        let mut a = ExpertStats::new(3, 3);
        let mut b = ExpertStats::new(3, 3);
        let batches: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![9, 8, 7], vec![4, 5, 6], vec![0, 1, 2]];
        for bth in &batches {
            a.record(bth);
            b.record(bth);
        }
        assert_eq!(a.average(), b.average());
        assert_eq!(a.history, b.history);
    }
}
