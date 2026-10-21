//! Dense NT GEMMs: FP8 with fine-grained scaling, BF16, and the sequential
//! f32 golden reference.
//!
//! All follow `D = C + A @ B^T` with `A: [m, k]`, `B: [n, k]` row-major
//! (DeepGEMM NT layout). FP8 scaling semantics: `a_sf` is
//! `[m, k/128]` e8m0 (per-token-group recipe), `b_sf` is
//! `[ceil(n/128), k/128]` e8m0 (per-128x128-block weight recipe), and for
//! output `(i, j)` each 128-wide K block contributes
//! `sf_a(i, kb) * sf_b(j/128, kb) * sum_{kk in block} deq(a)*deq(b)`
//! accumulated in f32 (see [`crate::util::scaled_row_dot`]).

use crate::util::{dequant_e4m3, k_blocks, par_rows, prod, scaled_row_dot, SF_GROUP_K};
use deepinfer_core::format::{bf16_to_f32, e8m0_to_f32};
use deepinfer_core::{ensure, Result};
use std::sync::Arc;

/// Dense FP8 NT GEMM (DeepGEMM fine-grained scaling).
///
/// `a`: `[m, k]` e4m3 bytes; `a_sf`: `[m, k/128]` e8m0 bytes;
/// `b`: `[n, k]` e4m3 bytes; `b_sf`: `[ceil(n/128), k/128]` e8m0 bytes
/// (ceil semantics over `n` match `deepinfer_fp8::quantize_weight_per_block`);
/// `c`: optional `[m, n]` f32 addend. Returns the `[m, n]` f32 output.
///
/// `out\[i\]\[j\] = c\[i\]\[j\] + sum over k-blocks of sf_a(i,kb) * sf_b(j/128,kb)
/// * sum_{kk in block} deq(a)\[i,kk\] * deq(b)\[j,kk\]`,
/// accumulated in f32 with fixed ascending order. Rows are computed on
/// the global thread pool over disjoint output rows (bit-deterministic).
///
/// # Errors
///
/// - `k % 128 != 0` (the scale-group granularity).
/// - Any buffer length mismatch (`a = m*k`, `b = n*k`,
///   `a_sf = m*(k/128)`, `b_sf = ceil(n/128)*(k/128)`, `c = m*n`), or a
///   dimension product that overflows `usize`.
pub fn fp8_gemm_nt(
    a: &[u8],
    a_sf: &[u8],
    b: &[u8],
    b_sf: &[u8],
    m: usize,
    n: usize,
    k: usize,
    c: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let kb = k_blocks(k)?;
    let mk = prod(m, k, "m*k")?;
    let nk = prod(n, k, "n*k")?;
    let nb = n.div_ceil(SF_GROUP_K);
    let asf_len = prod(m, kb, "m*k_blocks")?;
    let bsf_len = prod(nb, kb, "n_blocks*k_blocks")?;
    let mn = prod(m, n, "m*n")?;
    ensure!(a.len() == mk, DimensionMismatch, "a has {} bytes, expected m*k = {mk}", a.len());
    ensure!(b.len() == nk, DimensionMismatch, "b has {} bytes, expected n*k = {nk}", b.len());
    ensure!(
        a_sf.len() == asf_len,
        DimensionMismatch,
        "a_sf has {} scales, expected m*(k/128) = {asf_len}",
        a_sf.len()
    );
    ensure!(
        b_sf.len() == bsf_len,
        DimensionMismatch,
        "b_sf has {} scales, expected ceil(n/128)*(k/128) = {bsf_len}",
        b_sf.len()
    );
    if let Some(cv) = c {
        ensure!(cv.len() == mn, DimensionMismatch, "c has {} elements, expected m*n = {mn}", cv.len());
    }

    // Pre-dequantize to f32 (exact, order-preserving): the parallel row
    // sweep then only does f32 arithmetic. The byte buffers are copied into
    // Arc handles because par_for closures must be 'static.
    let a_f = Arc::new(dequant_e4m3(a));
    let b_f = Arc::new(dequant_e4m3(b));
    let asf_f = Arc::new(a_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let bsf_f = Arc::new(b_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let c_f = c.map(|cv| Arc::new(cv.to_vec()));

    let out = par_rows(m, n, mn, move |i, row| {
        let a_row = &a_f[i * k..(i + 1) * k];
        let asf_row = &asf_f[i * kb..(i + 1) * kb];
        for (j, oj) in row.iter_mut().enumerate() {
            let b_row = &b_f[j * k..(j + 1) * k];
            let jb = j / SF_GROUP_K;
            let bsf_row = &bsf_f[jb * kb..(jb + 1) * kb];
            let mut acc = match &c_f {
                Some(cv) => cv[i * n + j],
                None => 0.0f32,
            };
            acc += scaled_row_dot(a_row, asf_row, b_row, bsf_row, kb);
            *oj = acc;
        }
    });
    Ok(out)
}

/// BF16 NT GEMM with f32 accumulation (reference path).
///
/// `a`/`b` are `[m, k]` / `[n, k]` BF16 bit patterns converted exactly via
/// `deepinfer_core::format::bf16_to_f32`; `c` is an optional `[m, n]` f32
/// addend. Output rows are computed on the global thread pool over disjoint
/// rows; each output element accumulates sequentially over `k` in f32.
///
/// # Errors
///
/// Length mismatches (`a = m*k`, `b = n*k`, `c = m*n`) or an overflowing
/// dimension product.
pub fn bf16_gemm_nt(
    a: &[u16],
    b: &[u16],
    m: usize,
    n: usize,
    k: usize,
    c: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let mk = prod(m, k, "m*k")?;
    let nk = prod(n, k, "n*k")?;
    let mn = prod(m, n, "m*n")?;
    ensure!(a.len() == mk, DimensionMismatch, "a has {} elements, expected m*k = {mk}", a.len());
    ensure!(b.len() == nk, DimensionMismatch, "b has {} elements, expected n*k = {nk}", b.len());
    if let Some(cv) = c {
        ensure!(cv.len() == mn, DimensionMismatch, "c has {} elements, expected m*n = {mn}", cv.len());
    }

    let a_f = Arc::new(a.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>());
    let b_f = Arc::new(b.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>());
    let c_f = c.map(|cv| Arc::new(cv.to_vec()));

    let out = par_rows(m, n, mn, move |i, row| {
        let a_row = &a_f[i * k..(i + 1) * k];
        for (j, oj) in row.iter_mut().enumerate() {
            let b_row = &b_f[j * k..(j + 1) * k];
            let mut acc = match &c_f {
                Some(cv) => cv[i * n + j],
                None => 0.0f32,
            };
            for kk in 0..k {
                acc += a_row[kk] * b_row[kk];
            }
            *oj = acc;
        }
    });
    Ok(out)
}

/// F32 NT GEMM: naive sequential triple loop — the golden reference.
///
/// `out[i][j] = sum_k a[i][k] * b[j][k]` accumulated in f32 in ascending
/// `k` order. Entirely single-threaded by design (determinism reference
/// for the parallel kernels).
///
/// # Errors
///
/// Length mismatches (`a = m*k`, `b = n*k`) or an overflowing dimension
/// product.
pub fn f32_gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
    let mk = prod(m, k, "m*k")?;
    let nk = prod(n, k, "n*k")?;
    let mn = prod(m, n, "m*n")?;
    ensure!(a.len() == mk, DimensionMismatch, "a has {} elements, expected m*k = {mk}", a.len());
    ensure!(b.len() == nk, DimensionMismatch, "b has {} elements, expected n*k = {nk}", b.len());
    let mut out = vec![0.0f32; mn];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let a_row = &a[i * k..(i + 1) * k];
            let b_row = &b[j * k..(j + 1) * k];
            for kk in 0..k {
                acc += a_row[kk] * b_row[kk];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepinfer_core::format::{bf16_to_f32, f32_to_bf16_rne};
    use deepinfer_core::prop::{prop, random_matrix, random_vec};
    use deepinfer_core::rng::Philox;
    use deepinfer_core::Result;
    use deepinfer_fp8::{
        dequantize_per_token_group, dequantize_weight_per_block, quantize_per_token_group,
        quantize_weight_per_block,
    };

    /// Checks `out` against `reference` with a relative tolerance.
    fn check_relative(out: &[f32], reference: &[f32], rtol: f32, ctx: &str) {
        assert_eq!(out.len(), reference.len(), "length mismatch [{ctx}]");
        for (idx, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
            let tol = rtol * (1.0 + r.abs());
            assert!(
                (o - r).abs() <= tol,
                "[{ctx}] index {idx}: {o} vs reference {r} (|diff| {} > tol {tol})",
                (o - r).abs()
            );
        }
    }

    #[test]
    fn f32_gemm_nt_matches_manual_loops() -> Result<()> {
        // Hand-computed 2x2 case: rows (1,2,3), (4,5,6); B rows (1,0,0), (0,1,1).
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]; // [2, 3]
        let out = f32_gemm_nt(&a, &b, 2, 2, 3)?;
        assert_eq!(out, vec![1.0, 5.0, 4.0, 11.0]);
        // Property: matches a naive scalar triple loop bit-for-bit.
        prop("f32-gemm-golden", 0x601D, 60, |_i, rng| {
            let m = rng.uniform_usize(1, 9);
            let n = rng.uniform_usize(1, 9);
            let k = rng.uniform_usize(1, 17);
            let a = random_vec(rng, m * k, 5.0);
            let b = random_vec(rng, n * k, 5.0);
            let out = f32_gemm_nt(&a, &b, m, n, k)?;
            let mut expect = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    for kk in 0..k {
                        expect[i * n + j] += a[i * k + kk] * b[j * k + kk];
                    }
                }
            }
            ensure!(out == expect, Internal, "f32 golden mismatch");
            Ok(())
        });
        // Validation.
        assert!(f32_gemm_nt(&a, &b, 2, 2, 4).is_err()); // len != m*k
        assert!(f32_gemm_nt(&a, &b, 3, 2, 3).is_err()); // len != m*k
        Ok(())
    }

    #[test]
    fn fp8_gemm_nt_matches_dequantized_reference() -> Result<()> {
        // m, n in {8, 16, 130} (130 exercises the ceil(n/128) b-scale path),
        // k a multiple of 128.
        let cases: [(usize, usize, usize); 5] =
            [(8, 8, 128), (16, 16, 128), (130, 16, 128), (16, 130, 256), (8, 16, 384)];
        for &(m, n, k) in &cases {
            let mut rng = Philox::new(0xC0FFEE, (m * 131 + n * 7 + k) as u64);
            let a = random_matrix(&mut rng, m, k, 1.0);
            let b = random_matrix(&mut rng, n, k, 1.0);
            let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
            let (bq, bsf) = quantize_weight_per_block(&b, n, k, 128, 128)?;
            let out = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
            // Reference on the dequantized same data (mathematically equal
            // up to f32 reassociation of the per-block promotion).
            let da = dequantize_per_token_group(&aq, &asf, k, 128)?;
            let db = dequantize_weight_per_block(&bq, &bsf, n, k, 128, 128)?;
            let reference = f32_gemm_nt(&da, &db, m, n, k)?;
            check_relative(&out, &reference, 0.03, "fp8 vs dequantized reference");
            // Secondary sanity check vs the ORIGINAL data: worst-case
            // elementwise FP8 error is 2 * 2^-4 of each factor, so the dot
            // error is bounded by ~0.13 * sum |da * db|.
            let orig = f32_gemm_nt(&a, &b, m, n, k)?;
            for i in 0..m {
                for j in 0..n {
                    let mut abs_terms = 0.0f32;
                    for kk in 0..k {
                        abs_terms += (da[i * k + kk] * db[j * k + kk]).abs();
                    }
                    let tol = 0.13 * abs_terms + 1e-2;
                    assert!(
                        (out[i * n + j] - orig[i * n + j]).abs() <= tol,
                        "m={m} n={n} k={k} ({i},{j}): {} vs {} (tol {tol})",
                        out[i * n + j],
                        orig[i * n + j]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn fp8_gemm_nt_c_addend_and_edge_shapes() -> Result<()> {
        let mut rng = Philox::new(0xADDE, 3);
        let (m, n, k) = (8usize, 16usize, 128usize);
        let a = random_matrix(&mut rng, m, k, 1.0);
        let b = random_matrix(&mut rng, n, k, 1.0);
        let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
        let (bq, bsf) = quantize_weight_per_block(&b, n, k, 128, 128)?;
        let c = random_vec(&mut rng, m * n, 3.0);
        let with_c = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, Some(&c))?;
        let without_c = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
        for idx in 0..m * n {
            let tol = 1e-3 * (1.0 + with_c[idx].abs());
            assert!(
                ((with_c[idx] - without_c[idx]) - c[idx]).abs() <= tol,
                "C addend wrong at {idx}"
            );
        }
        // Wrong c length.
        assert!(fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, Some(&c[..m * n - 1])).is_err());
        // k not a multiple of 128.
        assert!(fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, 127, None).is_err());
        // Buffer length mismatches.
        assert!(fp8_gemm_nt(&aq[..m * k - 1], &asf, &bq, &bsf, m, n, k, None).is_err());
        assert!(fp8_gemm_nt(&aq, &asf[..m - 1], &bq, &bsf, m, n, k, None).is_err());
        assert!(fp8_gemm_nt(&aq, &asf, &bq[..n * k - 1], &bsf, m, n, k, None).is_err());
        assert!(fp8_gemm_nt(&aq, &asf, &bq, &bsf[..bsf.len() - 1], m, n, k, None).is_err());
        // Zero-sized operands are well defined.
        let empty = fp8_gemm_nt(&[], &[], &bq, &bsf, 0, n, k, None)?; // m = 0
        assert!(empty.is_empty());
        let k0 = fp8_gemm_nt(&[], &[], &[], &[], 0, n, 0, None)?; // m = 0 and k = 0
        assert!(k0.is_empty());
        Ok(())
    }

    #[test]
    fn fp8_gemm_nt_deterministic_bitwise() -> Result<()> {
        let mut rng = Philox::new(0xDe7, 9);
        let (m, n, k) = (130usize, 16usize, 256usize);
        let a = random_matrix(&mut rng, m, k, 1.0);
        let b = random_matrix(&mut rng, n, k, 1.0);
        let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
        let (bq, bsf) = quantize_weight_per_block(&b, n, k, 128, 128)?;
        let o1 = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
        let o2 = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
        assert_eq!(o1, o2, "bit-identical across runs");
        Ok(())
    }

    #[test]
    fn prop_fp8_gemm_accuracy_and_determinism() {
        prop("fp8-gemm-accuracy", 0x6E11, 40, |_i, rng| {
            let m = rng.uniform_usize(1, 9);
            let n = rng.uniform_usize(1, 9);
            let k = 128 * rng.uniform_usize(1, 3);
            let a = random_vec(rng, m * k, 1.0);
            let b = random_vec(rng, n * k, 1.0);
            let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
            let (bq, bsf) = quantize_weight_per_block(&b, n, k, 128, 128)?;
            let out = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
            let da = dequantize_per_token_group(&aq, &asf, k, 128)?;
            let db = dequantize_weight_per_block(&bq, &bsf, n, k, 128, 128)?;
            let reference = f32_gemm_nt(&da, &db, m, n, k)?;
            for (idx, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
                let tol = 0.03 * (1.0 + r.abs());
                ensure!((o - r).abs() <= tol, Internal, "idx {idx}: {o} vs {r}");
            }
            let out2 = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, k, None)?;
            ensure!(out == out2, Internal, "nondeterministic");
            Ok(())
        });
    }

    #[test]
    fn bf16_gemm_nt_matches_f32_on_same_rounded_data() -> Result<()> {
        let mut rng = Philox::new(0xB16, 5);
        let (m, n, k) = (24usize, 16usize, 64usize);
        let a = random_matrix(&mut rng, m, k, 3.0);
        let b = random_matrix(&mut rng, n, k, 3.0);
        let a_bits: Vec<u16> = a.iter().map(|&v| f32_to_bf16_rne(v)).collect();
        let b_bits: Vec<u16> = b.iter().map(|&v| f32_to_bf16_rne(v)).collect();
        let out = bf16_gemm_nt(&a_bits, &b_bits, m, n, k, None)?;
        // The kernel dequantizes exactly and accumulates sequentially in
        // ascending k — bit-identical to f32_gemm_nt on the same values.
        let da: Vec<f32> = a_bits.iter().map(|&v| bf16_to_f32(v)).collect();
        let db: Vec<f32> = b_bits.iter().map(|&v| bf16_to_f32(v)).collect();
        let reference = f32_gemm_nt(&da, &db, m, n, k)?;
        assert_eq!(out, reference, "bit-identical (same ops, same order)");
        // C addend within reassociation tolerance.
        let c = random_vec(&mut rng, m * n, 2.0);
        let with_c = bf16_gemm_nt(&a_bits, &b_bits, m, n, k, Some(&c))?;
        for idx in 0..m * n {
            let tol = 1e-3 * (1.0 + with_c[idx].abs());
            assert!(((with_c[idx] - reference[idx]) - c[idx]).abs() <= tol);
        }
        // Validation.
        assert!(bf16_gemm_nt(&a_bits, &b_bits, m, n, k + 1, None).is_err());
        assert!(bf16_gemm_nt(&a_bits, &b_bits, m, n, k, Some(&c[..5])).is_err());
        Ok(())
    }
}
