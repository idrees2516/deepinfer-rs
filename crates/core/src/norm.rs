//! RMSNorm — the normalization used throughout DeepSeek architectures
//! (attention latents, MLP inputs, final norm).

/// Applies RMSNorm over the `len`-element slice of `x` starting at
/// `offset`, writing to the corresponding slice of `out`.
///
/// `out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`
///
/// Deterministic single-pass mean with `f32` accumulation (matches GPU
/// kernels' warp-ordered reductions closely enough for a reference).
pub fn rmsnorm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32], offset: usize, len: usize) {
    debug_assert!(offset + len <= x.len());
    debug_assert!(offset + len <= out.len());
    debug_assert!(w.len() >= len);
    let mut sum_sq = 0.0f32;
    for i in 0..len {
        let v = x[offset + i];
        sum_sq += v * v;
    }
    let inv = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    for i in 0..len {
        out[offset + i] = x[offset + i] * inv * w[i];
    }
}

/// In-place RMSNorm of a full buffer.
pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let inv = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * inv * w[i];
    }
}

/// RMSNorm statistics (mean square) of a slice — exposed for tests and for
/// gated normalization variants implemented in `deepinfer-dsa`.
pub fn rms_mean_square(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x.iter() {
        s += v * v;
    }
    s / x.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_weights_scale_invariant() {
        let mut x = vec![3.0f32, 4.0];
        let w = vec![1.0f32, 1.0];
        rmsnorm(&mut x, &w, 1e-6);
        // Unit weights => output RMS == 1 (RMS is the invariant, not L2).
        let rms = (x[0] * x[0] + x[1] * x[1]) / 2.0;
        assert!((rms - 1.0).abs() < 1e-4, "rms {rms}");
        // Vector direction preserved.
        assert!((x[0] / 3.0 - x[1] / 4.0).abs() < 1e-5);
    }

    #[test]
    fn offset_variant_matches_slice_local() {
        let x: Vec<f32> = (0..20).map(|i| i as f32 * 0.3 - 3.0).collect();
        let w = vec![0.5f32; 12];
        let mut out = vec![0.0f32; 20];
        rmsnorm_into(&x, &w, 1e-6, &mut out, 8, 12);
        // Reference: normalize the extracted slice with its own weights.
        let mut slice = x[8..20].to_vec();
        rmsnorm(&mut slice, &w, 1e-6);
        for i in 0..12 {
            assert!((out[8 + i] - slice[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn weights_apply_elementwise() {
        let mut x = vec![2.0f32, 2.0, 2.0, 2.0];
        let w = vec![1.0f32, 2.0, 0.5, 0.0];
        rmsnorm(&mut x, &w, 1e-6);
        assert_eq!(x[3], 0.0);
        assert!((x[1] - 2.0 * x[0]).abs() < 1e-5);
    }
}
