//! Shared numeric kernels: stable softmax, log-sum-exp, dot products,
//! deterministic top-k and categorical sampling.
//!
//! Determinism contract: every routine here is a pure function of its
//! inputs with fully specified tie-breaking, and accumulation order is
//! fixed (sequential `f32`). This is what allows
//! `DEEPINFER_BATCH_INVARIANT` style guarantees on the CPU reference.

/// Numerically stable in-place softmax over `x`.
///
/// `-inf` entries map to zero probability; an all-`-inf` input maps to an
/// all-zero vector (never NaN).
pub fn softmax_inplace(x: &mut [f32]) {
    let Some(&m) = x.iter().max_by(|a, b| a.total_cmp(b)) else {
        return;
    };
    if !m.is_finite() {
        // All -inf (or empty/NaN): produce zeros (NaN stays NaN-free).
        if m.is_nan() {
            return;
        }
        for v in x.iter_mut() {
            if v.is_finite() {
                *v = 0.0;
            } else {
                *v = 0.0;
            }
        }
        return;
    }
    let mut z = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - m).exp();
        *v = e;
        z += e;
    }
    if z > 0.0 {
        for v in x.iter_mut() {
            *v /= z;
        }
    }
}

/// Log-sum-exp of `x` (returns `-inf` for empty input, never NaN for
/// finite / `-inf` inputs).
pub fn logsumexp(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return m;
    }
    let s: f32 = x.iter().map(|v| (*v - m).exp()).sum();
    m + s.ln()
}

/// Sequential-accumulation dot product (fixed order ⇒ deterministic).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Index of the maximum element; ties resolve to the **lowest index**
/// (deterministic).
pub fn argmax(x: &[f32]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &v) in x.iter().enumerate() {
        match best {
            None => best = Some(i),
            Some(b) => {
                if v > x[b] {
                    best = Some(i);
                }
            }
        }
    }
    best
}

/// Result of a deterministic top-k selection.
#[derive(Debug, Clone, PartialEq)]
pub struct TopK {
    /// Selected indices, ordered by descending value; ties broken by
    /// ascending index.
    pub indices: Vec<u32>,
    /// Values corresponding to `indices`.
    pub values: Vec<f32>,
}

/// Selects the `k` largest entries of `x`.
///
/// Ordering is total and deterministic: `(value desc, index asc)`.
pub fn top_k_desc(x: &[f32], k: usize) -> TopK {
    let k = k.min(x.len());
    let mut idx: Vec<u32> = (0..x.len() as u32).collect();
    // Full sort with a total order — deterministic and simple; O(n log n).
    idx.sort_by(|&a, &b| {
        let (va, vb) = (x[a as usize], x[b as usize]);
        vb.total_cmp(&va).then(a.cmp(&b))
    });
    idx.truncate(k);
    TopK {
        values: idx.iter().map(|&i| x[i as usize]).collect(),
        indices: idx,
    }
}

/// Samples `index` from a (possibly unnormalized) probability vector using
/// a uniform draw `u ∈ [0, 1)`.
///
/// Deterministic: the smallest index whose cumulative mass strictly exceeds
/// `u * total` is chosen. Zero-mass prefixes are skipped exactly.
pub fn sample_categorical(probs: &[f32], u: f32) -> Option<usize> {
    let total: f32 = probs.iter().sum();
    if !(total > 0.0) || !u.is_finite() || !(0.0..1.0).contains(&u) {
        return None;
    }
    let target = u * total;
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        cum += p;
        if cum > target {
            return Some(i);
        }
    }
    // Floating point round-off: return the last positive index.
    probs.iter().rposition(|&p| p > 0.0)
}

/// In-place `y += alpha * x` (axpy).
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (yv, &xv) in y.iter_mut().zip(x.iter()) {
        *yv += alpha * xv;
    }
}

/// Row-wise scaling `x[i] *= s`.
pub fn scale(x: &mut [f32], s: f32) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Philox;

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, -4.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn softmax_handles_inf_and_empty() {
        let mut x = vec![f32::NEG_INFINITY; 4];
        x[1] = 0.0;
        softmax_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert_eq!(x[1], 1.0);
        let mut all_inf = vec![f32::NEG_INFINITY; 3];
        softmax_inplace(&mut all_inf);
        assert!(all_inf.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn lse_matches_naive() {
        let x = vec![0.5f32, 1.5, -2.0];
        let naive = x.iter().map(|v| v.exp()).sum::<f32>().ln();
        assert!((logsumexp(&x) - naive).abs() < 1e-5);
        assert_eq!(logsumexp(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn topk_order_and_ties() {
        let x = vec![1.0, 3.0, 3.0, 2.0, 0.0];
        let tk = top_k_desc(&x, 3);
        assert_eq!(tk.indices, vec![1, 2, 3]); // tie 3.0 -> lower index first
        assert_eq!(tk.values, vec![3.0, 3.0, 2.0]);
        let tk_all = top_k_desc(&x, 10);
        assert_eq!(tk_all.indices.len(), 5);
    }

    #[test]
    fn categorical_hits_positive_mass_only() {
        let p = vec![0.0, 0.7, 0.0, 0.3];
        for u in [0.0f32, 0.1, 0.5, 0.69, 0.999] {
            let i = sample_categorical(&p, u).unwrap();
            assert!(i == 1 || i == 3);
        }
        assert!(sample_categorical(&[0.0, 0.0], 0.5).is_none());
    }

    #[test]
    fn argmax_ties_lowest() {
        assert_eq!(argmax(&[1.0, 5.0, 5.0, 1.0]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn categorical_property_uniformity() {
        let mut rng = Philox::new(0xCAFE, 9);
        let p = vec![0.25f32, 0.5, 0.25];
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            let i = sample_categorical(&p, rng.next_f32()).unwrap();
            counts[i] += 1;
        }
        let n = 30_000.0f64;
        assert!((counts[0] as f64 / n - 0.25).abs() < 0.02);
        assert!((counts[1] as f64 / n - 0.50).abs() < 0.02);
        assert!((counts[2] as f64 / n - 0.25).abs() < 0.02);
    }
}
