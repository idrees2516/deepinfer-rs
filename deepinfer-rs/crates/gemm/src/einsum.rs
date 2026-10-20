//! The FlashMLA fused-kernel epilogue einsum: `"bhr,hdr->bhd"`.
//!
//! `q` is `[b, h, r]` e4m3 with e8m0 scales (recipe `(1, 1, 32)`, produced
//! by `deepinfer_fp8::per_token_cast_row`); `w` is the permuted `Wv`
//! projection `[h, d, r]` e4m3 with **e4m3** scales per
//! `(h, d, r-group)` (FlashMLA's FP4-adjacent Wv recipe — scales are
//! dequantized with `e4m3_to_f32`, not `e8m0_to_f32`).

use crate::util::{dequant_e4m3, par_rows, prod};
use deepinfer_core::format::{e4m3_to_f32, e8m0_to_f32};
use deepinfer_core::{ensure, Result};
use std::sync::Arc;

/// FP8 einsum `"bhr,hdr->bhd"` (FlashMLA fused-kernel epilogue).
///
/// `q`: `[b, h, r]` e4m3 bytes with `q_sf`: `[b, h, ceil(r/group)]` e8m0
/// bytes; `w`: `[h, d, r]` e4m3 bytes with `w_sf`:
/// `[h, d, ceil(r/group)]` **e4m3** bytes. Returns `[b, h, d]` f32 with
///
/// `out[b][h][d] = sum over r-blocks of deq(q_sf) * e4m3_to_f32(w_sf) *
/// sum_{r in block} e4m3_to_f32(q) * e4m3_to_f32(w)`,
///
/// accumulated in f32 with fixed ascending order over blocks and `r`.
/// The final r-block may be partial (ceil semantics), matching the
/// `ceil(r/group)` scale layout. `(b, h)` pairs are computed as disjoint
/// output rows on the global thread pool (bit-deterministic).
///
/// # Errors
///
/// `group == 0`, or any buffer length mismatch
/// (`q = b*h*r`, `q_sf = b*h*ceil(r/group)`, `w = h*d*r`,
/// `w_sf = h*d*ceil(r/group)`), or an overflowing dimension product.
pub fn fp8_einsum_bhr_hdr_bhd(
    q: &[u8],
    q_sf: &[u8],
    w: &[u8],
    w_sf: &[u8],
    b: usize,
    h: usize,
    d: usize,
    r: usize,
    group: usize,
) -> Result<Vec<f32>> {
    ensure!(group > 0, InvalidArgument, "group must be > 0, got {group}");
    let rb = r.div_ceil(group);
    let bh = prod(b, h, "b*h")?;
    let q_len = prod(bh, r, "b*h*r")?;
    let qsf_len = prod(bh, rb, "b*h*r_blocks")?;
    let hd = prod(h, d, "h*d")?;
    let out_len = prod(bh, d, "b*h*d")?;
    let w_len = prod(hd, r, "h*d*r")?;
    let wsf_len = prod(hd, rb, "h*d*r_blocks")?;
    ensure!(q.len() == q_len, DimensionMismatch, "q has {} bytes, expected b*h*r = {q_len}", q.len());
    ensure!(
        q_sf.len() == qsf_len,
        DimensionMismatch,
        "q_sf has {} scales, expected b*h*ceil(r/group) = {qsf_len}",
        q_sf.len()
    );
    ensure!(w.len() == w_len, DimensionMismatch, "w has {} bytes, expected h*d*r = {w_len}", w.len());
    ensure!(
        w_sf.len() == wsf_len,
        DimensionMismatch,
        "w_sf has {} scales, expected h*d*ceil(r/group) = {wsf_len}",
        w_sf.len()
    );

    let q_f = Arc::new(dequant_e4m3(q));
    let qsf_f = Arc::new(q_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let w_f = Arc::new(dequant_e4m3(w));
    let wsf_f = Arc::new(w_sf.iter().map(|&s| e4m3_to_f32(s)).collect::<Vec<f32>>());

    // One output row per (b, h) pair, holding d elements.
    let out = par_rows(bh, d, out_len, move |bh_i, row| {
        let q_row = &q_f[bh_i * r..(bh_i + 1) * r];
        let qsf_row = &qsf_f[bh_i * rb..(bh_i + 1) * rb];
        let hi = bh_i % h;
        for (di, od) in row.iter_mut().enumerate() {
            let w_row = &w_f[(hi * d + di) * r..(hi * d + di + 1) * r];
            let wsf_row = &wsf_f[(hi * d + di) * rb..(hi * d + di + 1) * rb];
            let mut acc = 0.0f32;
            for rb_i in 0..rb {
                let g0 = rb_i * group;
                let g1 = (g0 + group).min(r);
                let mut partial = 0.0f32;
                for kk in g0..g1 {
                    partial += q_row[kk] * w_row[kk];
                }
                acc += qsf_row[rb_i] * wsf_row[rb_i] * partial;
            }
            *od = acc;
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::format::{e4m3_to_f32, e8m0_to_f32, f32_to_e4m3_rne, f32_to_e8m0, E8M0Round};
    use deepinfer_core::prop::{prop, random_vec};
    use deepinfer_core::rng::Philox;
    use deepinfer_core::Result;
    use deepinfer_fp8::per_token_cast_row;

    /// Per-token-group e4m3 + e8m0 quantization that tolerates a partial
    /// final group (ceil semantics), mirroring `per_token_cast_row` for
    /// full groups — used to build q rows whose length is not a multiple
    /// of the group.
    fn quantize_q_row_ceil(x: &[f32], group: usize) -> (Vec<u8>, Vec<u8>) {
        let rb = x.len().div_ceil(group);
        let mut values = vec![0u8; x.len()];
        let mut scales = vec![0u8; rb];
        for g in 0..rb {
            let seg = &x[g * group..x.len().min((g + 1) * group)];
            let mut amax = 0.0f32;
            for &v in seg {
                let a = if v.is_nan() { 0.0 } else { v.abs() };
                if a > amax {
                    amax = a;
                }
            }
            let amax = if amax.is_finite() { amax } else { f32::MAX };
            let sb = if amax <= 0.0 {
                0u8
            } else {
                f32_to_e8m0(amax / 448.0, E8M0Round::Ceil).unwrap_or(0)
            };
            let s = e8m0_to_f32(sb);
            scales[g] = sb;
            for (kk, &v) in seg.iter().enumerate() {
                values[g * group + kk] = f32_to_e4m3_rne(if v.is_nan() { 0.0 } else { v } / s, true);
            }
        }
        (values, scales)
    }

    /// Quantizes one w row with the FlashMLA Wv recipe: per-group e4m3
    /// scale `e4m3_rne(amax/6)` clamped to [2^-9, 448], e4m3 values.
    fn quantize_w_row(x: &[f32], group: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        let rb = x.len().div_ceil(group);
        let mut values = vec![0u8; x.len()];
        let mut scales = vec![0u8; rb];
        for g in 0..rb {
            let seg = &x[g * group..x.len().min((g + 1) * group)];
            let mut amax = 0.0f32;
            for &v in seg {
                let a = if v.is_nan() { 0.0 } else { v.abs() };
                if a > amax {
                    amax = a;
                }
            }
            let scale_val = (amax / 6.0).clamp(0.001_953_125, 448.0);
            let s_byte = f32_to_e4m3_rne(scale_val, true);
            let s = e4m3_to_f32(s_byte);
            scales[g] = s_byte;
            for (kk, &v) in seg.iter().enumerate() {
                values[g * group + kk] = f32_to_e4m3_rne(if v.is_nan() { 0.0 } else { v } / s, true);
            }
        }
        Ok((values, scales))
    }

    /// Elementwise-dequantized f32 reference (per-element scale application,
    /// plain dot over r — a different association than the kernel).
    fn einsum_reference(
        q: &[u8],
        q_sf: &[u8],
        w: &[u8],
        w_sf: &[u8],
        b: usize,
        h: usize,
        d: usize,
        r: usize,
        group: usize,
    ) -> Vec<f32> {
        let rb = r.div_ceil(group);
        let mut out = vec![0.0f32; b * h * d];
        for bi in 0..b {
            for hi in 0..h {
                for di in 0..d {
                    let q_row = &q[(bi * h + hi) * r..(bi * h + hi + 1) * r];
                    let qsf = &q_sf[(bi * h + hi) * rb..(bi * h + hi + 1) * rb];
                    let w_row = &w[(hi * d + di) * r..(hi * d + di + 1) * r];
                    let wsf = &w_sf[(hi * d + di) * rb..(hi * d + di + 1) * rb];
                    let mut acc = 0.0f32;
                    for kk in 0..r {
                        let sq = e8m0_to_f32(qsf[kk / group]);
                        let sw = e4m3_to_f32(wsf[kk / group]);
                        acc += (e4m3_to_f32(q_row[kk]) * sq) * (e4m3_to_f32(w_row[kk]) * sw);
                    }
                    out[(bi * h + hi) * d + di] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn einsum_matches_f32_reference() -> Result<()> {
        let (b, h, d, r, group) = (2usize, 3usize, 4usize, 64usize, 32usize);
        let mut rng = Philox::new(0xE150, 31);
        let mut q_bytes = Vec::new();
        let mut qsf_bytes = Vec::new();
        for _ in 0..b * h {
            let row = random_vec(&mut rng, r, 1.0);
            let (qv, qs) = per_token_cast_row(&row, group)?;
            q_bytes.extend_from_slice(&qv);
            qsf_bytes.extend_from_slice(&qs);
        }
        let mut w_bytes = Vec::new();
        let mut wsf_bytes = Vec::new();
        for _ in 0..h * d {
            let row = random_vec(&mut rng, r, 1.0);
            let (wv, ws) = quantize_w_row(&row, group)?;
            w_bytes.extend_from_slice(&wv);
            wsf_bytes.extend_from_slice(&ws);
        }
        let out = fp8_einsum_bhr_hdr_bhd(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group)?;
        assert_eq!(out.len(), b * h * d);
        let reference = einsum_reference(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group);
        for (idx, (&o, &rf)) in out.iter().zip(reference.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + rf.abs());
            assert!(
                (o - rf).abs() <= tol,
                "index {idx}: {o} vs reference {rf} (|diff| {} > tol {tol})",
                (o - rf).abs()
            );
        }
        // Determinism.
        let out2 = fp8_einsum_bhr_hdr_bhd(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group)?;
        assert_eq!(out, out2);
        Ok(())
    }

    #[test]
    fn einsum_partial_final_group() -> Result<()> {
        // r = 100 with group = 32 -> 4 r-blocks, the last one covers only
        // 4 elements (ceil semantics).
        let (b, h, d, r, group) = (1usize, 2usize, 3usize, 100usize, 32usize);
        let mut rng = Philox::new(0xE151, 32);
        let mut q_bytes = Vec::new();
        let mut qsf_bytes = Vec::new();
        for _ in 0..b * h {
            let row = random_vec(&mut rng, r, 1.0);
            // r is not a multiple of the group here, so use the ceil-aware
            // local quantizer (the public per-token cast requires alignment).
            let (qv, qs) = quantize_q_row_ceil(&row, group);
            q_bytes.extend_from_slice(&qv);
            qsf_bytes.extend_from_slice(&qs);
        }
        assert_eq!(qsf_bytes.len(), b * h * 4);
        let mut w_bytes = Vec::new();
        let mut wsf_bytes = Vec::new();
        for _ in 0..h * d {
            let row = random_vec(&mut rng, r, 1.0);
            let (wv, ws) = quantize_w_row(&row, group)?;
            w_bytes.extend_from_slice(&wv);
            wsf_bytes.extend_from_slice(&ws);
        }
        let out = fp8_einsum_bhr_hdr_bhd(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group)?;
        let reference = einsum_reference(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group);
        for (idx, (&o, &rf)) in out.iter().zip(reference.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + rf.abs());
            assert!((o - rf).abs() <= tol, "index {idx}: {o} vs {rf}");
        }
        Ok(())
    }

    #[test]
    fn einsum_validation_errors() -> Result<()> {
        let (b, h, d, r, group) = (1usize, 2usize, 2usize, 32usize, 32usize);
        let q = vec![0u8; b * h * r];
        let qsf = vec![127u8; b * h * (r / group)];
        let w = vec![0u8; h * d * r];
        let wsf = vec![56u8; h * d * (r / group)];
        assert!(fp8_einsum_bhr_hdr_bhd(&q, &qsf, &w, &wsf, b, h, d, r, group).is_ok());
        // group == 0.
        assert!(fp8_einsum_bhr_hdr_bhd(&q, &qsf, &w, &wsf, b, h, d, r, 0).is_err());
        // Length mismatches.
        assert!(fp8_einsum_bhr_hdr_bhd(&q[..r - 1], &qsf, &w, &wsf, b, h, d, r, group).is_err());
        assert!(fp8_einsum_bhr_hdr_bhd(&q, &qsf[..1], &w, &wsf, b, h, d, r, group).is_err());
        assert!(fp8_einsum_bhr_hdr_bhd(&q, &qsf, &w[..r - 1], &wsf, b, h, d, r, group).is_err());
        assert!(fp8_einsum_bhr_hdr_bhd(&q, &qsf, &w, &wsf[..1], b, h, d, r, group).is_err());
        // Zero-sized dims are fine (w stays [h, d, r] when b = 0).
        let b0 = fp8_einsum_bhr_hdr_bhd(&[], &[], &w, &wsf, 0, h, d, r, group)?;
        assert!(b0.is_empty());
        let r0 = fp8_einsum_bhr_hdr_bhd(&[], &[], &[], &[], b, h, d, 0, group)?;
        assert_eq!(r0.len(), b * h * d);
        assert!(r0.iter().all(|&v| v == 0.0), "r = 0 contributes zero");
        Ok(())
    }

    #[test]
    fn prop_einsum_matches_reference_and_determinism() {
        prop("fp8-einsum", 0xE152, 40, |_i, rng| {
            let b = rng.uniform_usize(1, 3);
            let h = rng.uniform_usize(1, 4);
            let d = rng.uniform_usize(1, 4);
            let group = [8usize, 16, 32][rng.uniform_usize(0, 3)];
            let r = group * rng.uniform_usize(1, 4);
            let mut q_bytes = Vec::new();
            let mut qsf_bytes = Vec::new();
            for _ in 0..b * h {
                let row = random_vec(rng, r, 1.0);
                let (qv, qs) = per_token_cast_row(&row, group)?;
                q_bytes.extend_from_slice(&qv);
                qsf_bytes.extend_from_slice(&qs);
            }
            let mut w_bytes = Vec::new();
            let mut wsf_bytes = Vec::new();
            for _ in 0..h * d {
                let row = random_vec(rng, r, 1.0);
                let (wv, ws) = quantize_w_row(&row, group)?;
                w_bytes.extend_from_slice(&wv);
                wsf_bytes.extend_from_slice(&ws);
            }
            let out = fp8_einsum_bhr_hdr_bhd(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group)?;
            let reference =
                einsum_reference(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group);
            for (idx, (&o, &rf)) in out.iter().zip(reference.iter()).enumerate() {
                let tol = 1e-3 * (1.0 + rf.abs());
                ensure!((o - rf).abs() <= tol, Internal, "idx {idx}: {o} vs {rf}");
            }
            let out2 = fp8_einsum_bhr_hdr_bhd(&q_bytes, &qsf_bytes, &w_bytes, &wsf_bytes, b, h, d, r, group)?;
            ensure!(out == out2, Internal, "nondeterministic");
            Ok(())
        });
    }
}
