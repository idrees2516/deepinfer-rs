//! Property-based testing harness (dependency-free replacement for
//! `proptest`-style checks, tuned for deterministic reproducibility).

use crate::error::Result;
use crate::rng::Philox;

/// Runs `iters` randomized checks of `prop`. Each iteration receives its
/// index and a deterministic generator derived from `(seed, iter)`, so any
/// failure can be reproduced exactly from the reported seed.
///
/// # Panics
/// Panics with a reproducible message when `prop` returns `Err`. Tests
/// only — never call this from production code paths.
pub fn prop<F>(label: &str, seed: u64, iters: u32, mut prop: F)
where
    F: FnMut(u32, &mut Philox) -> Result<()>,
{
    for i in 0..iters {
        let mut rng = Philox::new(seed, i as u64);
        if let Err(e) = prop(i, &mut rng) {
            panic!(
                "PROPERTY FAILED: '{label}' at iteration {i} (seed={seed:#x}, stream={i:#x}): {e}"
            );
        }
    }
}

/// Asserts `|a - b| <= tol` with a context label.
pub fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(
        d <= tol,
        "not close [{ctx}]: {a} vs {b} (|diff| = {d} > tol {tol})"
    );
}

/// Asserts element-wise closeness of two slices.
pub fn assert_close_slice(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "length mismatch [{ctx}]");
    let mut worst: Option<(usize, f32)> = None;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > tol && worst.map(|(_, w)| d > w).unwrap_or(true) {
            worst = Some((i, d));
        }
    }
    if let Some((i, d)) = worst {
        panic!("not close [{ctx}] at index {i}: {} vs {} (|diff| = {d} > tol {tol})", a[i], b[i]);
    }
}

/// Random dense matrix `rows × cols` in `[-mag, mag]`.
pub fn random_matrix(rng: &mut Philox, rows: usize, cols: usize, mag: f32) -> Vec<f32> {
    let mut v = vec![0.0; rows * cols];
    rng.fill_uniform(&mut v, -mag, mag);
    v
}

/// Random vector in `[-mag, mag]`.
pub fn random_vec(rng: &mut Philox, n: usize, mag: f32) -> Vec<f32> {
    let mut v = vec![0.0; n];
    rng.fill_uniform(&mut v, -mag, mag);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prop_passes_and_reproducers() {
        let mut ok = false;
        prop("always ok", 1, 10, |i, rng| {
            let _ = i;
            let _ = rng.next_f32();
            ok = true;
            Ok(())
        });
        assert!(ok);
    }

    #[test]
    #[should_panic(expected = "PROPERTY FAILED")]
    fn prop_fails_with_label() {
        prop("deliberate", 7, 5, |i, _| {
            if i >= 3 {
                Err(crate::Error::InvalidArgument("boom".into()))
            } else {
                Ok(())
            }
        });
    }

    #[test]
    fn closeness_helpers() {
        assert_close(1.0, 1.0 + 1e-7, 1e-6, "scalar");
        assert_close_slice(&[1.0, 2.0], &[1.0, 2.0 + 1e-7], 1e-6, "slice");
    }

    #[test]
    fn matrix_shapes() {
        let mut rng = Philox::new(3, 1);
        let m = random_matrix(&mut rng, 4, 5, 2.0);
        assert_eq!(m.len(), 20);
        assert!(m.iter().all(|v| v.abs() <= 2.0));
    }
}
