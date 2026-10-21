//! Grouped FP8 NT GEMMs for MoE layers (DeepGEMM m-grouped layouts) plus
//! the contiguous-layout alignment helpers.
//!
//! * **Contiguous** ([`m_grouped_fp8_gemm_nt_contiguous`]) — MoE prefill:
//!   tokens of each expert are stacked along M; `m_indices[i]` names the
//!   expert of row `i`. Rows of one group must be contiguous and each
//!   group's row count a multiple of the M/K alignment (128) — the layout
//!   DeepGEMM's `get_mn_alignment_for_contiguous_layout`-style padding
//!   produces.
//! * **Masked** ([`m_grouped_fp8_gemm_nt_masked`]) — MoE decode /
//!   CUDA-graph style: fixed capacity `m_max` per group, `masked_m[g]`
//!   valid rows; rows beyond the mask are zeroed in the output.

use crate::util::{dequant_e4m3, k_blocks, par_rows, prod, scaled_row_dot, SF_GROUP_K};
use deepinfer_core::format::e8m0_to_f32;
use deepinfer_core::{ensure, Result};
use std::sync::Arc;

/// The M/K alignment required for the contiguous grouped layout
/// (DeepGEMM: 128). Each group's row count in
/// [`m_grouped_fp8_gemm_nt_contiguous`] must be a multiple of this value.
pub fn get_mk_alignment_for_contiguous_layout() -> usize {
    128
}

/// Pads a token count up to the contiguous-layout alignment:
/// `ceil(tokens / alignment) * alignment` (saturating on overflow).
///
/// `alignment <= 1` is a no-op (returns `tokens` unchanged); this keeps the
/// function total and panic-free for degenerate alignments.
pub fn pad_to_alignment(tokens: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        tokens
    } else {
        tokens.div_ceil(alignment).saturating_mul(alignment)
    }
}

/// Grouped-contiguous FP8 NT GEMM (MoE prefill, DeepGEMM layout).
///
/// `a`: `[m, k]` e4m3 with `a_sf`: `[m, k/128]` e8m0;
/// `m_indices[i]`: group (expert) of row `i` (`m_indices.len() == m`);
/// `b`: `[num_groups, n, k]` e4m3 with
/// `b_sf`: `[num_groups, ceil(n/128), k/128]` e8m0.
/// Row `i` is multiplied against `b[m_indices[i]]` with the same
/// fine-grained scaling as [`crate::fp8_gemm_nt`]. Returns `[m, n]` f32.
///
/// Group rows must be **contiguous** along M and each group's row count
/// must be a **multiple of** [`get_mk_alignment_for_contiguous_layout`]
/// (128) — DeepGEMM's contiguous-layout contract. Groups may appear in any
/// order along M and may be empty.
///
/// # Errors
///
/// - `k % 128 != 0`, any buffer length mismatch, `m_indices.len() != m`.
/// - An `m_indices` value `>= num_groups`.
/// - Non-contiguous rows for some group, or a group row count not a
///   multiple of 128.
pub fn m_grouped_fp8_gemm_nt_contiguous(
    a: &[u8],
    a_sf: &[u8],
    b: &[u8],
    b_sf: &[u8],
    m_indices: &[u32],
    m: usize,
    n: usize,
    k: usize,
    num_groups: usize,
) -> Result<Vec<f32>> {
    let kb = k_blocks(k)?;
    let mk = prod(m, k, "m*k")?;
    let asf_len = prod(m, kb, "m*k_blocks")?;
    let out_len = prod(m, n, "m*n")?;
    let nb = n.div_ceil(SF_GROUP_K);
    let b_len = prod(prod(num_groups, n, "num_groups*n")?, k, "num_groups*n*k")?;
    let bsf_len = prod(prod(num_groups, nb, "num_groups*n_blocks")?, kb, "group scale count")?;
    ensure!(a.len() == mk, DimensionMismatch, "a has {} bytes, expected m*k = {mk}", a.len());
    ensure!(
        a_sf.len() == asf_len,
        DimensionMismatch,
        "a_sf has {} scales, expected m*(k/128) = {asf_len}",
        a_sf.len()
    );
    ensure!(b.len() == b_len, DimensionMismatch, "b has {} bytes, expected num_groups*n*k = {b_len}", b.len());
    ensure!(
        b_sf.len() == bsf_len,
        DimensionMismatch,
        "b_sf has {} scales, expected num_groups*ceil(n/128)*(k/128) = {bsf_len}",
        b_sf.len()
    );
    ensure!(
        m_indices.len() == m,
        DimensionMismatch,
        "m_indices has {} entries, expected m = {m}",
        m_indices.len()
    );

    // Range, contiguity and alignment validation per group.
    let mut first = vec![usize::MAX; num_groups];
    let mut last = vec![0usize; num_groups];
    let mut counts = vec![0usize; num_groups];
    for (i, &g) in m_indices.iter().enumerate() {
        let g = g as usize;
        ensure!(
            g < num_groups,
            OutOfRange,
            "m_indices[{i}] = {g} is out of range (num_groups = {num_groups})"
        );
        counts[g] += 1;
        if first[g] == usize::MAX {
            first[g] = i;
        }
        last[g] = i;
    }
    let alignment = get_mk_alignment_for_contiguous_layout();
    for g in 0..num_groups {
        if counts[g] > 0 {
            ensure!(
                last[g] - first[g] + 1 == counts[g],
                InvalidArgument,
                "rows of group {g} are not contiguous (first = {}, last = {}, count = {})",
                first[g],
                last[g],
                counts[g]
            );
            ensure!(
                counts[g] % alignment == 0,
                InvalidArgument,
                "group {g} has {} rows; the contiguous layout requires a multiple of {alignment}",
                counts[g]
            );
        }
    }

    let a_f = Arc::new(dequant_e4m3(a));
    let asf_f = Arc::new(a_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let b_f = Arc::new(dequant_e4m3(b));
    let bsf_f = Arc::new(b_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let m_idx = Arc::new(m_indices.to_vec());
    // Saturating strides: whenever a group tile is actually indexed,
    // num_groups * n * k was already validated to fit, so these are exact;
    // with num_groups == 0 no row ever runs and the values are unused.
    let group_b = n.saturating_mul(k); // stride of one group's [n, k] tile
    let group_bs = nb.saturating_mul(kb);

    let out = par_rows(m, n, out_len, move |i, row| {
        let g = m_idx[i] as usize;
        let a_row = &a_f[i * k..(i + 1) * k];
        let asf_row = &asf_f[i * kb..(i + 1) * kb];
        let b_g = &b_f[g * group_b..(g + 1) * group_b];
        let bsf_g = &bsf_f[g * group_bs..(g + 1) * group_bs];
        for (j, oj) in row.iter_mut().enumerate() {
            let b_row = &b_g[j * k..(j + 1) * k];
            let jb = j / SF_GROUP_K;
            let bsf_row = &bsf_g[jb * kb..(jb + 1) * kb];
            *oj = scaled_row_dot(a_row, asf_row, b_row, bsf_row, kb);
        }
    });
    Ok(out)
}

/// Grouped-masked FP8 NT GEMM (MoE decode / CUDA-graph style, DeepGEMM
/// layout).
///
/// `a`: `[num_groups, m_max, k]` e4m3 with
/// `a_sf`: `[num_groups, m_max, k/128]` e8m0;
/// `b`: `[num_groups, n, k]` e4m3 with
/// `b_sf`: `[num_groups, ceil(n/128), k/128]` e8m0;
/// `masked_m[g] <= m_max` valid rows per group. Only rows `[0, masked_m[g])`
/// of each group are computed (invalid rows are ignored); the output is
/// `[num_groups, m_max, n]` f32 with rows `>= masked_m[g]` set to zero.
/// Scaling semantics match [`crate::fp8_gemm_nt`].
///
/// # Errors
///
/// - `k % 128 != 0`, `masked_m.len() != num_groups`, `masked_m[g] > m_max`,
///   or any buffer length mismatch.
pub fn m_grouped_fp8_gemm_nt_masked(
    a: &[u8],
    a_sf: &[u8],
    b: &[u8],
    b_sf: &[u8],
    masked_m: &[usize],
    m_max: usize,
    n: usize,
    k: usize,
    num_groups: usize,
) -> Result<Vec<f32>> {
    let kb = k_blocks(k)?;
    let nb = n.div_ceil(SF_GROUP_K);
    let gm = prod(num_groups, m_max, "num_groups*m_max")?;
    let out_len = prod(gm, n, "num_groups*m_max*n")?;
    let a_len = prod(gm, k, "a len")?;
    let asf_len = prod(gm, kb, "a_sf len")?;
    let b_len = prod(prod(num_groups, n, "num_groups*n")?, k, "b len")?;
    let bsf_len = prod(prod(num_groups, nb, "num_groups*n_blocks")?, kb, "b_sf len")?;
    ensure!(
        masked_m.len() == num_groups,
        DimensionMismatch,
        "masked_m has {} entries, expected num_groups = {num_groups}",
        masked_m.len()
    );
    for (g, &mm) in masked_m.iter().enumerate() {
        ensure!(mm <= m_max, OutOfRange, "masked_m[{g}] = {mm} exceeds m_max = {m_max}");
    }
    ensure!(a.len() == a_len, DimensionMismatch, "a has {} bytes, expected num_groups*m_max*k = {a_len}", a.len());
    ensure!(
        a_sf.len() == asf_len,
        DimensionMismatch,
        "a_sf has {} scales, expected num_groups*m_max*(k/128) = {asf_len}",
        a_sf.len()
    );
    ensure!(b.len() == b_len, DimensionMismatch, "b has {} bytes, expected num_groups*n*k = {b_len}", b.len());
    ensure!(
        b_sf.len() == bsf_len,
        DimensionMismatch,
        "b_sf has {} scales, expected num_groups*ceil(n/128)*(k/128) = {bsf_len}",
        b_sf.len()
    );

    let a_f = Arc::new(dequant_e4m3(a));
    let asf_f = Arc::new(a_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let b_f = Arc::new(dequant_e4m3(b));
    let bsf_f = Arc::new(b_sf.iter().map(|&s| e8m0_to_f32(s)).collect::<Vec<f32>>());
    let masks = Arc::new(masked_m.to_vec());
    // Saturating strides (exact whenever a tile is actually indexed; see
    // the contiguous variant above).
    let group_b = n.saturating_mul(k);
    let group_bs = nb.saturating_mul(kb);

    // One output "row" per (group, slot); disjoint across the pool.
    let out = par_rows(gm, n, out_len, move |r, row| {
        let g = r / m_max;
        let i = r % m_max;
        if i >= masks[g] {
            return; // stays zero
        }
        let a_row = &a_f[r * k..(r + 1) * k];
        let asf_row = &asf_f[r * kb..(r + 1) * kb];
        let b_g = &b_f[g * group_b..(g + 1) * group_b];
        let bsf_g = &bsf_f[g * group_bs..(g + 1) * group_bs];
        for (j, oj) in row.iter_mut().enumerate() {
            let b_row = &b_g[j * k..(j + 1) * k];
            let jb = j / SF_GROUP_K;
            let bsf_row = &bsf_g[jb * kb..(jb + 1) * kb];
            *oj = scaled_row_dot(a_row, asf_row, b_row, bsf_row, kb);
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8_gemm_nt;
    use deepinfer_core::prop::{prop, random_matrix};
    use deepinfer_core::rng::Philox;
    use deepinfer_core::Result;
    use deepinfer_fp8::{quantize_per_token_group, quantize_weight_per_block};

    #[test]
    fn alignment_helpers() {
        assert_eq!(get_mk_alignment_for_contiguous_layout(), 128);
        assert_eq!(pad_to_alignment(0, 128), 0);
        assert_eq!(pad_to_alignment(1, 128), 128);
        assert_eq!(pad_to_alignment(127, 128), 128);
        assert_eq!(pad_to_alignment(128, 128), 128);
        assert_eq!(pad_to_alignment(129, 128), 256);
        assert_eq!(pad_to_alignment(384, 128), 384);
        // Degenerate alignments must not panic.
        assert_eq!(pad_to_alignment(10, 0), 10);
        assert_eq!(pad_to_alignment(10, 1), 10);
        assert_eq!(pad_to_alignment(usize::MAX, 128), usize::MAX); // saturates, no panic
    }

    /// Builds grouped-contiguous inputs: `layout` gives the group id of each
    /// 128-row block along M; group ids must be distinct (contiguity
    /// contract), but may leave some groups empty.
    fn run_contiguous(
        rng: &mut Philox,
        layout: &[u32],
        n: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let num_groups = layout.iter().copied().max().map(|g| g as usize + 1).unwrap_or(0);
        let m = layout.len() * 128;
        let a = random_matrix(rng, m, k, 1.0);
        let mut bq_all = Vec::new();
        let mut bsf_all = Vec::new();
        for _ in 0..num_groups {
            let bw = random_matrix(rng, n, k, 1.0);
            let (bq, bsf) = quantize_weight_per_block(&bw, n, k, 128, 128)?;
            bq_all.extend_from_slice(&bq);
            bsf_all.extend_from_slice(&bsf);
        }
        let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
        let mut m_indices = Vec::with_capacity(m);
        for &g in layout {
            m_indices.extend(std::iter::repeat(g).take(128));
        }
        let out = m_grouped_fp8_gemm_nt_contiguous(
            &aq, &asf, &bq_all, &bsf_all, &m_indices, m, n, k, num_groups,
        )?;
        // Expected: per-group dense GEMMs on the group's row slice.
        let mut expected = Vec::new();
        for (bi, &g) in layout.iter().enumerate() {
            let g = g as usize;
            let rows = 128;
            let a_g = &aq[bi * rows * k..(bi + 1) * rows * k];
            let asf_g = &asf[bi * rows * (k / 128)..(bi + 1) * rows * (k / 128)];
            let nb = n.div_ceil(128);
            let b_g = &bq_all[g * n * k..(g + 1) * n * k];
            let bsf_g = &bsf_all[g * nb * (k / 128)..(g + 1) * nb * (k / 128)];
            let dense =
                fp8_gemm_nt(a_g, asf_g, b_g, bsf_g, rows, n, k, None)?;
            expected.push(dense);
        }
        Ok((out, expected))
    }

    #[test]
    fn contiguous_matches_per_group_dense() -> Result<()> {
        // Groups in arbitrary (permuted) order along M: rows 0..128 ->
        // group 2, 128..256 -> group 0, 256..384 -> group 1.
        let mut rng = Philox::new(0x90C, 11);
        let (out, expected) = run_contiguous(&mut rng, &[2, 0, 1], 16, 128)?;
        for (bi, dense) in expected.iter().enumerate() {
            let got = &out[bi * 128 * 16..(bi + 1) * 128 * 16];
            assert_eq!(got, dense, "block {bi} must match the per-group dense GEMM");
        }
        Ok(())
    }

    #[test]
    fn contiguous_zero_row_group_is_valid() -> Result<()> {
        // num_groups = 2 with only group 1 populated (group 0 empty).
        let mut rng = Philox::new(0xE0, 12);
        let (out, expected) = run_contiguous(&mut rng, &[1], 8, 128)?;
        assert_eq!(out.len(), 128 * 8);
        assert_eq!(&out[..], &expected[0][..]);
        Ok(())
    }

    #[test]
    fn contiguous_validation_errors() -> Result<()> {
        let mut rng = Philox::new(0xE44, 13);
        let (n, k, num_groups) = (8usize, 128usize, 2usize);
        // Per-group weights so b/b_sf lengths are exactly right.
        let mut bq = Vec::new();
        let mut bsf = Vec::new();
        for _ in 0..num_groups {
            let bw = random_matrix(&mut rng, n, k, 1.0);
            let (bq_g, bsf_g) = quantize_weight_per_block(&bw, n, k, 128, 128)?;
            bq.extend_from_slice(&bq_g);
            bsf.extend_from_slice(&bsf_g);
        }
        let mk = 256 * k;
        // 256-row activations.
        let a256 = random_matrix(&mut rng, 256, k, 1.0);
        let (aq, asf) = quantize_per_token_group(&a256, k, 128)?;
        // Non-contiguous rows: group 0, then 1, then 0, then 1.
        let mut mi = Vec::new();
        for &g in &[0u32, 1, 0, 1] {
            mi.extend(std::iter::repeat(g).take(128));
        }
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq, &asf, &bq, &bsf, &mi, 256, n, k, num_groups).is_err());
        // k not a multiple of 128.
        let mi4 = vec![0u32; 256];
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq, &asf, &bq, &bsf, &mi4, 256, n, 127, num_groups).is_err());
        // Buffer length mismatches.
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq[..mk - 1], &asf, &bq, &bsf, &mi4, 256, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq, &asf[..255], &bq, &bsf, &mi4, 256, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq, &asf, &bq[..num_groups * n * k - 1], &bsf, &mi4, 256, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq, &asf, &bq, &bsf[..bsf.len() - 1], &mi4, 256, n, k, num_groups)
            .is_err());
        // Unaligned group row count: 100 rows of group 0 (m = 100).
        let a100 = random_matrix(&mut rng, 100, k, 1.0);
        let (aq100, asf100) = quantize_per_token_group(&a100, k, 128)?;
        let mi100 = vec![0u32; 100];
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq100, &asf100, &bq, &bsf, &mi100, 100, n, k, num_groups)
            .is_err());
        // Index out of range and m_indices length mismatch (m = 128).
        let a128 = random_matrix(&mut rng, 128, k, 1.0);
        let (aq128, asf128) = quantize_per_token_group(&a128, k, 128)?;
        let mi3 = vec![5u32; 128];
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq128, &asf128, &bq, &bsf, &mi3, 128, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_contiguous(&aq128, &asf128, &bq, &bsf, &mi3[..127], 128, n, k, num_groups)
            .is_err());
        Ok(())
    }

    #[test]
    fn masked_matches_dense_prefix_and_zeros() -> Result<()> {
        let mut rng = Philox::new(0xA5C, 21);
        let (num_groups, m_max, n, k) = (3usize, 8usize, 16usize, 256usize);
        let masked_m = [3usize, 0, 8];
        let a = random_matrix(&mut rng, num_groups * m_max, k, 1.0); // [num_groups*m_max, k]
        let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
        let mut bq_all = Vec::new();
        let mut bsf_all = Vec::new();
        for _ in 0..num_groups {
            let bw = random_matrix(&mut rng, n, k, 1.0);
            let (bq, bsf) = quantize_weight_per_block(&bw, n, k, 128, 128)?;
            bq_all.extend_from_slice(&bq);
            bsf_all.extend_from_slice(&bsf);
        }
        let out = m_grouped_fp8_gemm_nt_masked(
            &aq, &asf, &bq_all, &bsf_all, &masked_m, m_max, n, k, num_groups,
        )?;
        assert_eq!(out.len(), num_groups * m_max * n);
        let nb = n.div_ceil(128);
        for g in 0..num_groups {
            for i in 0..m_max {
                let row = &out[(g * m_max + i) * n..(g * m_max + i + 1) * n];
                if i >= masked_m[g] {
                    assert!(row.iter().all(|&v| v == 0.0), "row ({g},{i}) must be zero");
                } else {
                    // Dense reference on the same valid rows.
                    let a_row = &aq[(g * m_max + i) * k..(g * m_max + i + 1) * k];
                    let asf_row =
                        &asf[(g * m_max + i) * (k / 128)..(g * m_max + i + 1) * (k / 128)];
                    let b_g = &bq_all[g * n * k..(g + 1) * n * k];
                    let bsf_g = &bsf_all[g * nb * (k / 128)..(g + 1) * nb * (k / 128)];
                    let dense = fp8_gemm_nt(a_row, asf_row, b_g, bsf_g, 1, n, k, None)?;
                    assert_eq!(row, &dense[..], "row ({g},{i}) must match dense");
                }
            }
        }
        // Determinism.
        let out2 = m_grouped_fp8_gemm_nt_masked(
            &aq, &asf, &bq_all, &bsf_all, &masked_m, m_max, n, k, num_groups,
        )?;
        assert_eq!(out, out2);
        Ok(())
    }

    #[test]
    fn masked_validation_errors() -> Result<()> {
        let mut rng = Philox::new(0xA5E, 22);
        let (num_groups, m_max, n, k) = (2usize, 4usize, 8usize, 128usize);
        let a = random_matrix(&mut rng, num_groups * m_max, k, 1.0);
        let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
        // Per-group weights so b/b_sf lengths are exactly right.
        let mut bq = Vec::new();
        let mut bsf = Vec::new();
        for _ in 0..num_groups {
            let bw = random_matrix(&mut rng, n, k, 1.0);
            let (bq_g, bsf_g) = quantize_weight_per_block(&bw, n, k, 128, 128)?;
            bq.extend_from_slice(&bq_g);
            bsf.extend_from_slice(&bsf_g);
        }
        let masked = [2usize, 4];
        let ok = m_grouped_fp8_gemm_nt_masked(&aq, &asf, &bq, &bsf, &masked, m_max, n, k, num_groups);
        assert!(ok.is_ok());
        assert!(m_grouped_fp8_gemm_nt_masked(&aq, &asf, &bq, &bsf, &masked[..1], m_max, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_masked(&aq, &asf, &bq, &bsf, &[5, 1], m_max, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_masked(&aq, &asf, &bq, &bsf, &masked, m_max, n, 127, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_masked(
            &aq[..aq.len() - 1], &asf, &bq, &bsf, &masked, m_max, n, k, num_groups
        )
        .is_err());
        assert!(m_grouped_fp8_gemm_nt_masked(&aq, &asf[..asf.len() - 1], &bq, &bsf, &masked, m_max, n, k, num_groups)
            .is_err());
        assert!(m_grouped_fp8_gemm_nt_masked(&aq, &asf, &bq, &bsf[..bsf.len() - 1], &masked, m_max, n, k, num_groups)
            .is_err());
        Ok(())
    }

    #[test]
    fn prop_contiguous_matches_dense_and_determinism() {
        prop("grouped-contiguous", 0xC0FE, 12, |_i, rng| {
            let n = rng.uniform_usize(2, 9);
            let k = 128 * rng.uniform_usize(1, 2);
            // Distinct group ids per 128-row block (contiguity contract);
            // optionally reversed to vary the order along M.
            let n_blocks = rng.uniform_usize(1, 4);
            let mut layout: Vec<u32> = (0..n_blocks as u32).collect();
            if rng.next_u32() & 1 == 1 {
                layout.reverse();
            }
            let (out, expected) = run_contiguous(rng, &layout, n, k)?;
            for (bi, dense) in expected.iter().enumerate() {
                let got = &out[bi * 128 * n..(bi + 1) * 128 * n];
                ensure!(got == dense, Internal, "block {bi} mismatch");
            }
            Ok(())
        });
    }

    #[test]
    fn prop_masked_matches_dense() {
        prop("grouped-masked", 0xA5D, 25, |_i, rng| {
            let num_groups = rng.uniform_usize(1, 3);
            let m_max = rng.uniform_usize(1, 6);
            let n = rng.uniform_usize(2, 7);
            let k = 128 * rng.uniform_usize(1, 2);
            let masked_m: Vec<usize> = (0..num_groups).map(|_| rng.uniform_usize(0, m_max + 1)).collect();
            let a = random_matrix(rng, num_groups * m_max, k, 1.0);
            let (aq, asf) = quantize_per_token_group(&a, k, 128)?;
            let mut bq_all = Vec::new();
            let mut bsf_all = Vec::new();
            for _ in 0..num_groups {
                let bw = random_matrix(rng, n, k, 1.0);
                let (bq, bsf) = quantize_weight_per_block(&bw, n, k, 128, 128)?;
                bq_all.extend_from_slice(&bq);
                bsf_all.extend_from_slice(&bsf);
            }
            let out = m_grouped_fp8_gemm_nt_masked(
                &aq, &asf, &bq_all, &bsf_all, &masked_m, m_max, n, k, num_groups,
            )?;
            let nb = n.div_ceil(128);
            for g in 0..num_groups {
                for i in 0..m_max {
                    let row = &out[(g * m_max + i) * n..(g * m_max + i + 1) * n];
                    if i >= masked_m[g] {
                        ensure!(row.iter().all(|&v| v == 0.0), Internal, "row ({g},{i}) not zero");
                    } else {
                        let a_row = &aq[(g * m_max + i) * k..(g * m_max + i + 1) * k];
                        let asf_row = &asf[(g * m_max + i) * (k / 128)..(g * m_max + i + 1) * (k / 128)];
                        let b_g = &bq_all[g * n * k..(g + 1) * n * k];
                        let bsf_g = &bsf_all[g * nb * (k / 128)..(g + 1) * nb * (k / 128)];
                        let dense = fp8_gemm_nt(a_row, asf_row, b_g, bsf_g, 1, n, k, None)?;
                        ensure!(row == &dense[..], Internal, "row ({g},{i}) mismatch");
                    }
                }
            }
            Ok(())
        });
    }
}
