//! Host-side byte repacking between DeepInfer's storage layouts and the
//! layouts cuTile's block-scaled MMA (`mmaf_scaled`) consumes.
//!
//! cuTile's `mmaf_scaled` takes E8M0 scale tensors at the hardware
//! granularity of **one scale byte per 32 elements**. DeepGEMM's DeepSeek
//! recipe quantizes activations per **1×128** block and weights per
//! **128×128** block (on SM100 the UE8M0 bytes are "packed ×4": one byte
//! covers four consecutive 32-element groups). This module performs exactly
//! that replication on the host, where weights are repacked once at load
//! time and activations are repacked by the device quantization kernel
//! ([`crate::kernels`] `quant_fp8_block128`) which writes each scale byte
//! into four consecutive 32-lane slots.
//!
//! All functions here are pure CPU code over byte slices and run in every
//! build configuration (no CUDA required).

/// Replicates each byte of a `[rows, k/128]` per-128 scale tensor four times
/// along K, producing the `[rows, k/32]` layout `mmaf_scaled` consumes.
///
/// `k` is the GEMM's K dimension and must be a multiple of 128.
pub fn replicate_scales_128_to_32(scales: &[u8], rows: usize, k: usize) -> Result<Vec<u8>, String> {
    if k % 128 != 0 {
        return Err(format!("K={k} is not a multiple of 128"));
    }
    let in_cols = k / 128;
    if scales.len() != rows * in_cols {
        return Err(format!(
            "scale buffer has {} bytes, expected rows*ceil(K/128) = {rows}*{in_cols}",
            scales.len()
        ));
    }
    let out_cols = k / 32;
    let mut out = vec![0u8; rows * out_cols];
    for r in 0..rows {
        for c in 0..in_cols {
            let b = scales[r * in_cols + c];
            for rep in 0..4 {
                out[r * out_cols + c * 4 + rep] = b;
            }
        }
    }
    Ok(out)
}

/// Replicates a `[n, k/128]` weight-scale tensor to `[n, k/32]` in place of a
/// fresh allocation — convenience wrapper over
/// [`replicate_scales_128_to_32`].
pub fn replicate_weight_scales(scales: &[u8], n: usize, k: usize) -> Result<Vec<u8>, String> {
    replicate_scales_128_to_32(scales, n, k)
}

/// Builds the E8M0 scale byte for one 128-element block from its amax,
/// matching the CPU reference (`f32_to_e8m0(amax/448, Ceil)` in
/// `deepinfer-mla`'s cache quantizer and DeepGEMM's `round_sf` Ceil policy):
/// `scale = 2^ceil(log2(amax / 448))` with exact powers of two *not* rounded
/// up, and byte `0` (2^-127) for an all-zero block.
pub fn fp8_block_scale_byte(amax: f32) -> u8 {
    if !(amax > 0.0) {
        // All-zero (or NaN/inf-guarded by callers) block: 2^-127.
        return 0;
    }
    let l2 = (amax as f64 / 448.0f64).log2();
    let mut e = l2.ceil();
    // Exact powers of two must not be rounded up by Ceil.
    if (e - l2).abs() < 1e-9 {
        e = l2.floor();
    }
    let e = e.clamp(-127.0, 127.0);
    (e as i32 + 127) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replication_shape_and_content() {
        let rows = 3;
        let k = 256; // 2 scale cols in, 8 out
        let scales: Vec<u8> = (0..rows * 2).map(|i| 120 + i as u8).collect();
        let out = replicate_scales_128_to_32(&scales, rows, k).unwrap();
        assert_eq!(out.len(), rows * 8);
        for r in 0..rows {
            for c in 0..2 {
                for rep in 0..4 {
                    assert_eq!(out[r * 8 + c * 4 + rep], scales[r * 2 + c]);
                }
            }
        }
    }

    #[test]
    fn replication_rejects_bad_shapes() {
        assert!(replicate_scales_128_to_32(&[0; 2], 1, 100).is_err());
        assert!(replicate_scales_128_to_32(&[0; 2], 1, 128).is_err());
    }

    #[test]
    fn scale_byte_matches_reference() {
        // amax = 448 -> q = 1 (exact power of two) -> e = 0 -> byte 127.
        assert_eq!(fp8_block_scale_byte(448.0), 127);
        // All-zero block -> byte 0 (2^-127), matching the CPU reference.
        assert_eq!(fp8_block_scale_byte(0.0), 0);
        // amax = 896 -> q = 2 (exact) -> e = 1 -> byte 128.
        assert_eq!(fp8_block_scale_byte(896.0), 128);
        // amax slightly above 448 must round the exponent UP (ceil policy).
        assert_eq!(fp8_block_scale_byte(449.0), 128);
        // amax slightly below 448 stays at exponent 0.
        assert_eq!(fp8_block_scale_byte(447.0), 127);
        // Exact power-of-two quotients are never rounded up.
        assert_eq!(fp8_block_scale_byte(224.0), 126);
        // Very small amax clamps at 2^-127.
        assert_eq!(fp8_block_scale_byte(1e-40), 0);
    }
}
