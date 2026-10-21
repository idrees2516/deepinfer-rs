//! Rotary position embeddings with YaRN scaling.
//!
//! Matches the RoPE used by DeepSeek models (HF `transformers`
//! implementation): interleaved ("non-neox" / GPT-J style, used by
//! DeepSeek-V4.1) and half-split ("neox" / rotate-half) layouts, optional
//! YaRN frequency scaling with the standard `beta_fast`/`beta_slow`
//! correction range and `_get_mscale` attention factor, plus the
//! attention-conjugate rotation used by FlashMLA's fused kernel.
use crate::ensure;


use crate::error::Result;

/// Pairing layout of the RoPE dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Interleaved pairs `(x0,x1), (x2,x3), ...` — "non-neox", the
    /// DeepSeek-V4.1 / FlashMLA fused-kernel convention.
    Interleaved,
    /// Half-split `(x[i], x[i + dim/2])` — "neox" / rotate-half.
    HalfSplit,
}

/// YaRN long-context scaling parameters (DeepSeek-V3 style).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnConfig {
    /// Stretch factor.
    pub factor: f64,
    /// Context length the base frequencies were trained for.
    pub original_max_position_embeddings: usize,
    /// Fast correction boundary (rotations).
    pub beta_fast: f64,
    /// Slow correction boundary (rotations).
    pub beta_slow: f64,
    /// Attention mscale factor (see [`Rope::attention_factor`]).
    pub mscale: f64,
    /// Second mscale factor applied via `mscale_all_dim`.
    pub mscale_all_dim: f64,
}

impl Default for YarnConfig {
    fn default() -> YarnConfig {
        // DeepSeek-V3 values.
        YarnConfig {
            factor: 40.0,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
        }
    }
}

fn get_mscale(scale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * scale.ln() + 1.0
    }
}

fn yarn_correction_dim(num_rotations: f64, dim: usize, base: f64, max_pos: usize) -> f64 {
    (dim as f64 * (max_pos as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// A precomputed RoPE table.
#[derive(Debug, Clone)]
pub struct Rope {
    /// Number of rotated dimensions (must be even).
    pub dim: usize,
    /// Pairing layout.
    pub style: RopeStyle,
    /// Base frequency (`theta`).
    pub base: f64,
    /// Optional YaRN scaling.
    pub scaling: Option<YarnConfig>,
    /// Flattapped cache: `cos_sin[pos * (dim/2) * 2 + j*2 + {0,1}`].
    cos_sin: Vec<f32>,
    max_pos: usize,
}

impl Rope {
    /// Builds the cos/sin cache for positions `[0, max_position)`.
    pub fn new(
        dim: usize,
        style: RopeStyle,
        base: f64,
        scaling: Option<YarnConfig>,
        max_position: usize,
    ) -> Result<Rope> {
        ensure!(dim % 2 == 0 && dim > 0, InvalidArgument, "rope dim must be even and positive, got {dim}");
        ensure!(max_position > 0, InvalidArgument, "max_position must be positive");
        ensure!(base > 1.0, InvalidArgument, "rope base must exceed 1.0");
        let half = dim / 2;

        // Per-dimension inverse frequencies.
        let mut inv_freq = vec![0.0f64; half];
        for (j, f) in inv_freq.iter_mut().enumerate() {
            let e = (2.0 * j as f64) / dim as f64;
            *f = base.powf(-e);
        }
        if let Some(y) = &scaling {
            let low = yarn_correction_dim(y.beta_fast, dim, base, y.original_max_position_embeddings)
                .floor()
                .max(0.0)
                .min(half as f64);
            let high = yarn_correction_dim(y.beta_slow, dim, base, y.original_max_position_embeddings)
                .ceil()
                .max(low + 1e-3)
                .min(half as f64);
            let denom = (high - low).max(1e-9);
            for (j, f) in inv_freq.iter_mut().enumerate() {
                let jf = j as f64;
                // Linear ramp: 0 at/below `low` (keep original), 1 at/above
                // `high` (fully interpolated by the factor).
                let ramp = ((jf - low) / denom).clamp(0.0, 1.0);
                let interp = *f / y.factor;
                *f = interp * ramp + *f * (1.0 - ramp);
            }
        }

        let mut cos_sin = Vec::with_capacity(max_position * half * 2);
        for pos in 0..max_position {
            for f in inv_freq.iter().take(half) {
                let ang = pos as f64 * f;
                cos_sin.push(ang.cos() as f32);
                cos_sin.push(ang.sin() as f32);
            }
        }
        Ok(Rope {
            dim,
            style,
            base,
            scaling,
            cos_sin,
            max_pos: max_position,
        })
    }

    /// Maximum cached position (exclusive).
    pub fn max_position(&self) -> usize {
        self.max_pos
    }

    /// Read-only access to the flat cos/sin table
    /// `[max_pos][dim/2][2]` (cos first). Used by GPU launchers to upload
    /// the precomputed table to a device — the YaRN math itself never runs
    /// on the device.
    pub fn cos_sin_table(&self) -> &[f32] {
        &self.cos_sin
    }

    /// Attention-logit scale factor implied by YaRN (`mscale`), or 1.0.
    pub fn attention_factor(&self) -> f64 {
        match &self.scaling {
            None => 1.0,
            Some(y) => get_mscale(y.mscale) * get_mscale(y.mscale_all_dim.max(1.0)),
        }
    }

    #[inline]
    fn cos_sin_at(&self, pos: usize, j: usize) -> (f32, f32) {
        let idx = (pos * (self.dim / 2) + j) * 2;
        (self.cos_sin[idx], self.cos_sin[idx + 1])
    }

    /// Applies RoPE in-place to `x`, viewed as `rows` rows of length
    /// `row_len`, rotating the slice `[offset, offset + dim)` of each row,
    /// with per-row positions from `positions`.
    pub fn apply_at(
        &self,
        positions: &[u32],
        x: &mut [f32],
        row_len: usize,
        offset: usize,
    ) -> Result<()> {
        ensure!(
            positions.len().saturating_mul(row_len) <= x.len(),
            DimensionMismatch,
            "rope buffer too small: need {} rows x {row_len}, got {} elems",
            positions.len(),
            x.len()
        );
        ensure!(
            offset + self.dim <= row_len,
            InvalidArgument,
            "rope window [{offset}, {}) exceeds row length {row_len}",
            offset + self.dim
        );
        let half = self.dim / 2;
        for (r, &p) in positions.iter().enumerate() {
            ensure!(p < self.max_pos as u32, OutOfRange, "position {p} >= cached max {}", self.max_pos);
            let pos = p as usize;
            let row = &mut x[r * row_len..(r + 1) * row_len];
            match self.style {
                RopeStyle::Interleaved => {
                    for j in 0..half {
                        let (c, s) = self.cos_sin_at(pos, j);
                        let i0 = offset + 2 * j;
                        let i1 = i0 + 1;
                        let (a, b) = (row[i0], row[i1]);
                        row[i0] = a * c - b * s;
                        row[i1] = a * s + b * c;
                    }
                }
                RopeStyle::HalfSplit => {
                    for j in 0..half {
                        let (c, s) = self.cos_sin_at(pos, j);
                        let i0 = offset + j;
                        let i1 = offset + j + half;
                        let (a, b) = (row[i0], row[i1]);
                        row[i0] = a * c - b * s;
                        row[i1] = a * s + b * c;
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies RoPE to the first `dim` elements of each row.
    pub fn apply(&self, positions: &[u32], x: &mut [f32], row_len: usize) -> Result<()> {
        self.apply_at(positions, x, row_len, 0)
    }

    /// Applies the **conjugate** rotation (rotation by `-position`), used by
    /// FlashMLA's fused kernel on attention outputs (O-RoPE) so that the
    /// output latents live in the query's relative frame.
    pub fn apply_conjugate_at(
        &self,
        positions: &[u32],
        x: &mut [f32],
        row_len: usize,
        offset: usize,
    ) -> Result<()> {
        ensure!(
            offset + self.dim <= row_len,
            InvalidArgument,
            "rope window exceeds row length"
        );
        let half = self.dim / 2;
        for (r, &p) in positions.iter().enumerate() {
            ensure!(p < self.max_pos as u32, OutOfRange, "position {p} >= cached max {}", self.max_pos);
            let pos = p as usize;
            let row = &mut x[r * row_len..(r + 1) * row_len];
            match self.style {
                RopeStyle::Interleaved => {
                    for j in 0..half {
                        let (c, s) = self.cos_sin_at(pos, j);
                        let i0 = offset + 2 * j;
                        let i1 = i0 + 1;
                        let (a, b) = (row[i0], row[i1]);
                        row[i0] = a * c + b * s;
                        row[i1] = -a * s + b * c;
                    }
                }
                RopeStyle::HalfSplit => {
                    for j in 0..half {
                        let (c, s) = self.cos_sin_at(pos, j);
                        let i0 = offset + j;
                        let i1 = offset + j + half;
                        let (a, b) = (row[i0], row[i1]);
                        row[i0] = a * c + b * s;
                        row[i1] = -a * s + b * c;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn rotation_preserves_norm_both_styles() {
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let rope = Rope::new(8, style, 10000.0, None, 64).unwrap();
            let mut x = vec![0.0f32; 64 * 16];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i * 37) % 11) as f32 - 5.0;
            }
            let before = norm(&x[..16]);
            let positions: Vec<u32> = (0..64).map(|i| (i * 3) % 64).collect();
            rope.apply(&positions, &mut x, 16).unwrap();
            let after = norm(&x[..16]);
            assert!(
                (before - after).abs() < 1e-4,
                "style {style:?}: {before} vs {after}"
            );
        }
    }

    #[test]
    fn conjugate_undoes_rotation() {
        let rope = Rope::new(8, RopeStyle::Interleaved, 10000.0, None, 32).unwrap();
        let orig: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7) - 5.0).collect();
        let mut x = orig.clone();
        let pos = [7u32];
        rope.apply_at(&pos, &mut x, 16, 0).unwrap();
        rope.apply_conjugate_at(&pos, &mut x, 16, 0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn yarn_extends_periods() {
        let plain = Rope::new(64, RopeStyle::Interleaved, 10000.0, None, 128).unwrap();
        let yarn = Rope::new(
            64,
            RopeStyle::Interleaved,
            10000.0,
            Some(YarnConfig {
                factor: 40.0,
                original_max_position_embeddings: 64,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all_dim: 1.0,
            }),
            128,
        )
        .unwrap();
        // Excite a low-frequency pair (j = 31 of 32): YaRN interpolates it.
        let mut a = vec![0.0f32; 64];
        a[62] = 1.0;
        let mut b = a.clone();
        plain.apply(&[100], &mut a, 64).unwrap();
        yarn.apply(&[100], &mut b, 64).unwrap();
        // With YaRN the phase at position 100 must differ (slower rotation).
        assert_ne!(a[62], b[62], "yarn did not change low-frequency pair");
        assert!(yarn.attention_factor() >= 1.0);
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(Rope::new(7, RopeStyle::Interleaved, 10000.0, None, 8).is_err());
        assert!(Rope::new(8, RopeStyle::Interleaved, 1.0, None, 8).is_err());
    }

    #[test]
    fn position_bounds_checked() {
        let rope = Rope::new(4, RopeStyle::Interleaved, 100.0, None, 4).unwrap();
        let mut x = vec![0.0f32; 4];
        assert!(rope.apply(&[4], &mut x, 4).is_err());
        assert!(rope.apply(&[3], &mut x, 4).is_ok());
    }
}
