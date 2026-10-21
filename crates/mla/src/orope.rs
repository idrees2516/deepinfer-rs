//! O-RoPE: the conjugate rotation FlashMLA's fused kernel applies to the
//! attention output's rope slice, so the output latents live in the
//! query's relative frame (DeepSeek-V4.1-Flash "O-RoPE (conjugate)"
//! in the FlashMLA README; implemented by
//! `deepinfer_core::rope::Rope::apply_conjugate_at`).

use deepinfer_core::rope::Rope;
use deepinfer_core::{ensure, Error, Result};

/// Applies the O-RoPE conjugate to the rope slice of each head's output
/// row: `x` is viewed as `rows` rows of `row_len` and the slice
/// `[row_len - rope_dim, row_len)` of each row is rotated by the
/// CONJUGATE of the query position for that row (all other dims are
/// untouched).
///
/// For an attention output laid out `[s, n_heads, head_dim]` pass
/// `rows = s * n_heads`, `row_len = head_dim` and per-row positions
/// (each token's position repeated `n_heads` times).
///
/// # Errors
/// [`Error::DimensionMismatch`] / [`Error::InvalidArgument`] on shape
/// mismatches; [`Error::OutOfRange`] for positions beyond the rope table.
pub fn o_rope_conjugate(
    rope: &Rope,
    positions: &[u32],
    x: &mut [f32],
    rows: usize,
    row_len: usize,
) -> Result<()> {
    ensure!(
        positions.len() == rows,
        DimensionMismatch,
        "expected {rows} positions, got {}",
        positions.len()
    );
    let expect = rows
        .checked_mul(row_len)
        .ok_or_else(|| Error::DimensionMismatch("rows*row_len overflows".into()))?;
    ensure!(
        x.len() == expect,
        DimensionMismatch,
        "buffer has {} elems, expected rows*row_len = {expect}",
        x.len()
    );
    ensure!(
        row_len >= rope.dim,
        InvalidArgument,
        "row_len {row_len} is smaller than the rope dim {}",
        rope.dim
    );
    rope.apply_conjugate_at(positions, x, row_len, row_len - rope.dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::prop::assert_close_slice;
    use deepinfer_core::rope::RopeStyle;

    #[test]
    fn rotation_then_conjugate_is_identity_on_rope_slice() {
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 32).unwrap();
        let mut x = vec![0.0f32; 3 * 20];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 37) % 23) as f32 * 0.31 - 3.4;
        }
        let orig = x.clone();
        let positions = [3u32, 7, 11];
        rope.apply_at(&positions, &mut x, 20, 12).unwrap();
        o_rope_conjugate(&rope, &positions, &mut x, 3, 20).unwrap();
        assert_close_slice(&x, &orig, 1e-5, "rotate then conjugate");
    }

    #[test]
    fn conjugate_alone_rotates_only_rope_slice() {
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 32).unwrap();
        let mut x = vec![0.0f32; 2 * 20];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 5.0;
        }
        let orig = x.clone();
        let positions = [5u32, 9];
        o_rope_conjugate(&rope, &positions, &mut x, 2, 20).unwrap();
        for r in 0..2 {
            for j in 0..12 {
                assert_eq!(x[r * 20 + j], orig[r * 20 + j], "prefix untouched");
            }
            let mut changed = false;
            for j in 12..20 {
                changed |= x[r * 20 + j] != orig[r * 20 + j];
            }
            assert!(changed, "rope slice must change");
        }
    }

    #[test]
    fn validates_shapes() {
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 32).unwrap();
        let mut x = vec![0.0f32; 2 * 20];
        assert!(o_rope_conjugate(&rope, &[1u32], &mut x, 2, 20).is_err());
        assert!(o_rope_conjugate(&rope, &[1u32, 2], &mut x[..39], 2, 20).is_err());
        assert!(o_rope_conjugate(&rope, &[1u32, 2], &mut x, 2, 8).is_err()); // row_len < dim
        assert!(o_rope_conjugate(&rope, &[1u32, 2], &mut x, 2, 20).is_ok());
        // Position beyond the table.
        assert!(o_rope_conjugate(&rope, &[32u32, 0], &mut x, 2, 20).is_err());
    }
}
