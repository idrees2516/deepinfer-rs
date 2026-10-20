//! Sparse MLA decode with FlashMLA's exact semantics and split-K
//! structure.
//!
//! Ground truth: FlashMLA README — decode gathers the per-query selected
//! KV rows (physical indices, `-1` invalid), scores them against the
//! absorbed 512-dim query heads, and produces
//! `out [b, s_q, n_heads, head_dim]` plus `lse [b, n_heads, s_q]`
//! (natural log). Queries with no valid index yield a zero output row and
//! `lse = +inf`.
//!
//! Paper-fidelity structure: the selected rows are processed in CHUNKS OF
//! 64 (the FlashMLA page size) maintaining a running `(m, l, o)` state;
//! each chunk computes its own partial `(m_c, l_c, o_c)` and is merged
//! exactly like flash-decoding split-K:
//! `m_new = max(m, m_c)`, `o = o*exp(m-m_new) + o_c*exp(m_c-m_new)`,
//! `l = l*exp(m-m_new) + l_c*exp(m_c-m_new)`; finally `out = o / l` and
//! `lse = m_new + ln(l)`.

use crate::cache::MlaPagedCache;
use deepinfer_core::math::{axpy, dot, scale};
use deepinfer_core::{ensure, Error, Result};

/// KV rows processed per chunk (FlashMLA page size).
pub(crate) const DECODE_CHUNK: usize = 64;

/// Sparse decode: EXACT FlashMLA semantics.
///
/// * `q`: `[b][s_q][n_heads][head_dim]` with RoPE already applied by the
///   caller (rotate the last `rope_dim` dims of each head row) and the
///   DSA mscale (if any) folded into `sm_scale`.
/// * `indices`: `[b][s_q][topk]` physical indices (`page_id * page_size +
///   offset`); `-1` marks invalid entries. A non-negative index that
///   references an unallocated page or an unwritten row is treated as
///   invalid (mirrors the README's `indices >= s_kv` rule).
/// * `out`: `[b][s_q][n_heads][head_dim]` (attention over latents; O-RoPE
///   NOT applied here). `lse`: `[b][n_heads][s_q]` natural-log values.
/// * `attn_sink` (optional, `[n_heads]`): FlashMLA epilogue
///   `out *= exp(lse) / (exp(lse) + exp(attn_sink))` (computed in the
///   stable form `1 / (1 + exp(attn_sink - lse))`); the returned `lse` is
///   NOT modified.
///
/// `n_heads` is inferred from `q.len() / (b * s_q * head_dim)` where
/// `head_dim` is the cache's row width.
///
/// # Errors
/// [`Error::DimensionMismatch`] on shape mismatches, [`Error::Numerical`]
/// on non-finite inputs/scores, [`Error::InvalidArgument`] when decoding
/// against an empty cache with a non-empty query buffer.
pub fn flash_mla_decode(
    q: &[f32],
    cache: &MlaPagedCache,
    indices: &[i32],
    b: usize,
    s_q: usize,
    topk: usize,
    sm_scale: f32,
    attn_sink: Option<&[f32]>,
    out: &mut [f32],
    lse: &mut [f32],
) -> Result<()> {
    ensure!(
        sm_scale.is_finite(),
        Numerical,
        "sm_scale must be finite, got {sm_scale}"
    );
    let bs = b
        .checked_mul(s_q)
        .ok_or_else(|| Error::DimensionMismatch(format!("b*s_q overflows: {b} x {s_q}")))?;
    let idx_len = bs
        .checked_mul(topk)
        .ok_or_else(|| Error::DimensionMismatch(format!("b*s_q*topk overflows: {bs} x {topk}")))?;
    ensure!(
        indices.len() == idx_len,
        DimensionMismatch,
        "indices has {} elems, expected b*s_q*topk = {idx_len}",
        indices.len()
    );
    if let Some(sk) = attn_sink {
        ensure!(
            sk.iter().all(|v| v.is_finite()),
            Numerical,
            "attn_sink contains non-finite values"
        );
    }
    let d = match cache.head_dim() {
        Some(d) => d,
        None => {
            ensure!(
                q.is_empty(),
                InvalidArgument,
                "decoding against an empty cache requires an empty query buffer (head_dim unknown)"
            );
            ensure!(
                out.is_empty() && lse.is_empty(),
                DimensionMismatch,
                "decode with an empty q requires empty out/lse buffers"
            );
            return Ok(());
        }
    };
    ensure!(
        q.iter().all(|v| v.is_finite()),
        Numerical,
        "q contains non-finite values"
    );
    let bd = bs
        .checked_mul(d)
        .ok_or_else(|| Error::DimensionMismatch(format!("b*s_q*head_dim overflows: {bs} x {d}")))?;
    let h = if bs == 0 {
        ensure!(
            q.is_empty(),
            DimensionMismatch,
            "q must be empty when b*s_q == 0, got {} elems",
            q.len()
        );
        0
    } else {
        ensure!(
            q.len() % bd == 0,
            DimensionMismatch,
            "q length {} is not a multiple of b*s_q*head_dim ({bd})",
            q.len()
        );
        q.len() / bd
    };
    ensure!(
        out.len() == q.len(),
        DimensionMismatch,
        "out has {} elems, expected {}",
        out.len(),
        q.len()
    );
    let lse_len = b
        .checked_mul(h)
        .and_then(|v| v.checked_mul(s_q))
        .ok_or_else(|| Error::DimensionMismatch("b*n_heads*s_q overflows".into()))?;
    ensure!(
        lse.len() == lse_len,
        DimensionMismatch,
        "lse has {} elems, expected b*n_heads*s_q = {lse_len}",
        lse.len()
    );
    if let Some(sk) = attn_sink {
        ensure!(
            sk.len() == h,
            DimensionMismatch,
            "attn_sink has {} elems, expected n_heads = {h}",
            sk.len()
        );
    }

    // Per-query running state and per-chunk partials (hoisted, reset per
    // query / chunk). Sequential loops keep accumulation order fixed =>
    // bit-identical determinism.
    let hd = h.checked_mul(d).unwrap_or(0);
    let mut m = vec![f32::NEG_INFINITY; h];
    let mut l = vec![0.0f32; h];
    let mut o = vec![0.0f32; hd];
    let mut mc = vec![f32::NEG_INFINITY; h];
    let mut lc = vec![0.0f32; h];
    let mut oc = vec![0.0f32; hd];

    for bi in 0..b {
        for qi in 0..s_q {
            m.fill(f32::NEG_INFINITY);
            l.fill(0.0);
            o.fill(0.0);
            let qbase = (bi * s_q + qi) * h;
            let idxs = &indices[(bi * s_q + qi) * topk..][..topk];
            for cs in (0..topk).step_by(DECODE_CHUNK) {
                let ce = (cs + DECODE_CHUNK).min(topk);
                mc.fill(f32::NEG_INFINITY);
                lc.fill(0.0);
                oc.fill(0.0);
                for &idx in &idxs[cs..ce] {
                    if idx < 0 {
                        continue;
                    }
                    // Non-negative indices pointing outside the stored
                    // cache are invalid (score -inf), matching the
                    // README's `indices >= s_kv` rule.
                    let Ok(row) = cache.row_f32(idx) else { continue };
                    for hi in 0..h {
                        let q_row = &q[(qbase + hi) * d..][..d];
                        let s = dot(q_row, &row) * sm_scale;
                        ensure!(
                            s.is_finite(),
                            Numerical,
                            "non-finite attention score at (b={bi}, q={qi}, h={hi})"
                        );
                        if s > mc[hi] {
                            let res = (mc[hi] - s).exp();
                            scale(&mut oc[hi * d..(hi + 1) * d], res);
                            lc[hi] *= res;
                            mc[hi] = s;
                        }
                        let p = (s - mc[hi]).exp();
                        lc[hi] += p;
                        axpy(p, &row, &mut oc[hi * d..(hi + 1) * d]);
                    }
                }
                // Merge the chunk partial into the running state
                // (flash-decoding split-K merge).
                for hi in 0..h {
                    let m_new = m[hi].max(mc[hi]);
                    if m_new == f32::NEG_INFINITY {
                        continue; // both empty so far
                    }
                    let so = (m[hi] - m_new).exp();
                    let sc = (mc[hi] - m_new).exp();
                    for j in 0..d {
                        o[hi * d + j] = o[hi * d + j] * so + oc[hi * d + j] * sc;
                    }
                    l[hi] = l[hi] * so + lc[hi] * sc;
                    m[hi] = m_new;
                }
            }
            // Finalize: normalize, fill lse, apply the attn_sink epilogue.
            for hi in 0..h {
                let li = (bi * h + hi) * s_q + qi;
                let obase = (qbase + hi) * d;
                if l[hi] > 0.0 {
                    lse[li] = m[hi] + l[hi].ln();
                    for j in 0..d {
                        out[obase + j] = o[hi * d + j] / l[hi];
                    }
                } else {
                    // FlashMLA edge case: no valid index at all.
                    lse[li] = f32::INFINITY;
                    for j in 0..d {
                        out[obase + j] = 0.0;
                    }
                }
                if let Some(sk) = attn_sink {
                    // exp(lse)/(exp(lse)+exp(sink)) == 1/(1+exp(sink-lse))
                    // (stable; +inf lse gives factor 1).
                    let f = 1.0 / (1.0 + (sk[hi] - lse[li]).exp());
                    for j in 0..d {
                        out[obase + j] *= f;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheFormat;
    use crate::testutil::{build_cache, naive_decode, small_cfg};
    use deepinfer_core::prop::{assert_close_slice, prop, random_vec};
    use deepinfer_core::rng::Philox;

    fn make_indices(
        rng: &mut Philox,
        cache: &MlaPagedCache,
        s_kv: usize,
        count: usize,
    ) -> Vec<i32> {
        let mut idx = Vec::with_capacity(count);
        for _ in 0..count {
            match rng.uniform_usize(0, 10) {
                0 => idx.push(-1),
                1 => idx.push((s_kv as i64 + 5 + rng.uniform_usize(0, 100) as i64) as i32),
                2 => {
                    // Inside an allocated page but beyond the written
                    // prefix, or one past the last page.
                    let last = cache.physical_index(0, s_kv - 1).unwrap() as i64;
                    idx.push((last + rng.uniform_usize(1, 64) as i64) as i32);
                }
                _ => {
                    let pos = rng.uniform_usize(0, s_kv);
                    idx.push(cache.physical_index(0, pos).unwrap());
                }
            }
        }
        idx
    }

    #[test]
    fn decode_matches_naive_bf16() {
        let mut rng = Philox::new(0xDDEC, 1);
        let (b, s_q, h, d, s_kv, topk) = (2usize, 3, 4, 64, 150, 200);
        let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
        let cache = build_cache(CacheFormat::Bf16, &rows, 0, 8);
        let indices = make_indices(&mut rng, &cache, s_kv, b * s_q * topk);
        let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
        let sm = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; q.len()];
        let mut lse = vec![0.0f32; b * h * s_q];
        flash_mla_decode(&q, &cache, &indices, b, s_q, topk, sm, None, &mut out, &mut lse)
            .unwrap();
        let (nout, nlse) = naive_decode(&q, &cache, &indices, b, s_q, topk, h, d, sm, None);
        assert_close_slice(&out, &nout, 1e-4, "decode out (bf16, 4 chunks)");
        assert_close_slice(&lse, &nlse, 1e-4, "decode lse (bf16, 4 chunks)");
    }

    #[test]
    fn decode_matches_naive_at_chunk_boundaries() {
        // topk crossing every multiple of 64 must equal the single-pass
        // naive computation (chunked merge == single pass).
        let mut rng = Philox::new(0xCBC, 2);
        for topk in [0usize, 1, 63, 64, 65, 127, 128, 129, 200, 257] {
            let (b, s_q, h, d, s_kv) = (1usize, 2, 2, 32, 100);
            let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
            let cache = build_cache(CacheFormat::Bf16, &rows, 0, 8);
            let indices = make_indices(&mut rng, &cache, s_kv, b * s_q * topk);
            let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
            let sm = 0.25f32;
            let mut out = vec![0.0f32; q.len()];
            let mut lse = vec![0.0f32; b * h * s_q];
            flash_mla_decode(&q, &cache, &indices, b, s_q, topk, sm, None, &mut out, &mut lse)
                .unwrap();
            let (nout, nlse) = naive_decode(&q, &cache, &indices, b, s_q, topk, h, d, sm, None);
            assert_close_slice(&out, &nout, 1e-4, "chunked out (topk {topk})");
            assert_close_slice(&lse, &nlse, 1e-4, "chunked lse (topk {topk})");
        }
    }

    #[test]
    fn decode_matches_naive_quantized_formats() {
        let mut rng = Philox::new(0xF08, 3);
        for (format, out_tol, lse_tol) in [
            (CacheFormat::Fp8Group32, 5e-2, 5e-2),
            (CacheFormat::Fp4Group16, 1.5, 1.5),
        ] {
            let (b, s_q, h, d, s_kv, topk) = (1usize, 2, 2, 32, 60, 45);
            let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
            let cache = build_cache(format, &rows, 0, 4);
            let indices: Vec<i32> = (0..b * s_q * topk)
                .map(|_| cache.physical_index(0, rng.uniform_usize(0, s_kv)).unwrap())
                .collect();
            let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
            let sm = 1.0 / (d as f32).sqrt();
            let mut out = vec![0.0f32; q.len()];
            let mut lse = vec![0.0f32; b * h * s_q];
            flash_mla_decode(&q, &cache, &indices, b, s_q, topk, sm, None, &mut out, &mut lse)
                .unwrap();
            // (a) Algorithmic equality: naive over the same dequantized
            //     cache rows.
            let (nout, nlse) = naive_decode(&q, &cache, &indices, b, s_q, topk, h, d, sm, None);
            assert_close_slice(&out, &nout, 1e-4, "{format:?} out vs dequantized naive");
            assert_close_slice(&lse, &nlse, 1e-4, "{format:?} lse vs dequantized naive");
            // (b) Quantization accuracy: naive over the original f32 rows.
            let (mut oref, mut lref) = (vec![0.0f32; out.len()], vec![0.0f32; lse.len()]);
            naive_decode_rows(&q, &rows, &indices, b, s_q, topk, h, d, sm, None, &mut oref, &mut lref);
            assert_close_slice(&out, &oref, out_tol, "{format:?} out vs original rows");
            assert_close_slice(&lse, &lref, lse_tol, "{format:?} lse vs original rows");
        }
    }

    /// Naive decode over explicit f32 rows (no cache quantization).
    fn naive_decode_rows(
        q: &[f32],
        rows: &[Vec<f32>],
        indices: &[i32],
        b: usize,
        s_q: usize,
        topk: usize,
        h: usize,
        d: usize,
        sm_scale: f32,
        sink: Option<&[f32]>,
        out: &mut [f32],
        lse: &mut [f32],
    ) {
        for bi in 0..b {
            for qi in 0..s_q {
                let selected: Vec<&Vec<f32>> = indices[(bi * s_q + qi) * topk..][..topk]
                    .iter()
                    .filter(|&&i| i >= 0)
                    // Physical index == logical position: these caches are
                    // built by appending one fresh sequence, so pages are
                    // allocated in order.
                    .filter_map(|&i| rows.get(i as usize))
                    .collect();
                for hi in 0..h {
                    let obase = ((bi * s_q + qi) * h + hi) * d;
                    let li = (bi * h + hi) * s_q + qi;
                    if selected.is_empty() {
                        lse[li] = f32::INFINITY;
                        continue;
                    }
                    let q_row = &q[obase..obase + d];
                    let scores: Vec<f32> =
                        selected.iter().map(|r| dot(q_row, r) * sm_scale).collect();
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    for j in 0..d {
                        out[obase + j] =
                            selected.iter().zip(exps.iter()).map(|(r, &e)| e * r[j]).sum::<f32>()
                                / sum;
                    }
                    lse[li] = m + sum.ln();
                    if let Some(sk) = sink {
                        let f = 1.0 / (1.0 + (sk[hi] - lse[li]).exp());
                        for j in 0..d {
                            out[obase + j] *= f;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decode_canonical_512_dims() {
        let mut rng = Philox::new(0x512, 4);
        let (b, s_q, h, s_kv, topk) = (1usize, 2, 2, 100, 70);
        let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, 512, 1.0)).collect();
        let cache = build_cache(CacheFormat::Bf16, &rows, 0, 8);
        let indices: Vec<i32> = (0..b * s_q * topk)
            .map(|_| cache.physical_index(0, rng.uniform_usize(0, s_kv)).unwrap())
            .collect();
        let q = random_vec(&mut rng, b * s_q * h * 512, 1.0);
        let sm = 1.0 / (512.0f32).sqrt();
        let mut out = vec![0.0f32; q.len()];
        let mut lse = vec![0.0f32; b * h * s_q];
        flash_mla_decode(&q, &cache, &indices, b, s_q, topk, sm, None, &mut out, &mut lse)
            .unwrap();
        let (nout, nlse) = naive_decode(&q, &cache, &indices, b, s_q, topk, h, 512, sm, None);
        assert_close_slice(&out, &nout, 1e-4, "canonical 512-dim out");
        assert_close_slice(&lse, &nlse, 1e-4, "canonical 512-dim lse");
    }

    #[test]
    fn decode_no_valid_index_edge() {
        let mut rng = Philox::new(0xED9E, 5);
        let (b, s_q, h, d, s_kv) = (2usize, 2, 3, 32, 5);
        let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
        let cache = build_cache(CacheFormat::Bf16, &rows, 0, 2);
        let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
        // All invalid: -1 and out-of-range physical indices.
        let indices = vec![-1i32, 10_000, -1, -1, -1, -1, -1, -1];
        let mut out = vec![1.0f32; q.len()];
        let mut lse = vec![0.0f32; b * h * s_q];
        flash_mla_decode(&q, &cache, &indices, b, s_q, 2, 0.5, None, &mut out, &mut lse)
            .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(lse.iter().all(|&v| v == f32::INFINITY));
        // topk = 0 with valid cache: same edge.
        let mut out2 = vec![1.0f32; q.len()];
        let mut lse2 = vec![0.0f32; b * h * s_q];
        flash_mla_decode(&q, &cache, &[], b, s_q, 0, 0.5, None, &mut out2, &mut lse2).unwrap();
        assert!(out2.iter().all(|&v| v == 0.0));
        assert!(lse2.iter().all(|&v| v == f32::INFINITY));
        // Mixed: query 0 valid, query 1 all invalid.
        let mut mixed = Vec::new();
        for bi in 0..b {
            for qi in 0..s_q {
                for _ in 0..3 {
                    mixed.push(if (bi * s_q + qi) % s_q == 0 { 0 } else { -1 });
                }
            }
        }
        let mut out3 = vec![0.0f32; q.len()];
        let mut lse3 = vec![0.0f32; b * h * s_q];
        flash_mla_decode(&q, &cache, &mixed, b, s_q, 3, 0.5, None, &mut out3, &mut lse3).unwrap();
        for bi in 0..b {
            for hi in 0..h {
                assert!(lse3[(bi * h + hi) * s_q] != f32::INFINITY);
                assert!(lse3[(bi * h + hi) * s_q + 1] == f32::INFINITY);
                assert!(out3[((bi * s_q + 1) * h + hi) * d..][..d].iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn decode_attn_sink_hand_computed() {
        // q = [1, 0]; kv rows [1,0] and [0,1]; sm_scale = 1; sink = 0.5.
        let cache = build_cache(CacheFormat::Bf16, &[vec![1.0, 0.0], vec![0.0, 1.0]], 0, 2);
        let q = vec![1.0f32, 0.0];
        let indices = vec![0i32, 1];
        let mut out = vec![0.0f32; 2];
        let mut lse = vec![0.0f32; 1];
        flash_mla_decode(&q, &cache, &indices, 1, 1, 2, 1.0, Some(&[0.5]), &mut out, &mut lse)
            .unwrap();
        // scores [1, 0]; softmax [1, e^-1]/(1+e^-1); lse = 1 + ln(1+e^-1).
        let e1 = (-1.0f32).exp();
        let sum = 1.0 + e1;
        let lse_expect = 1.0f32 + sum.ln();
        let out_expect = [1.0 / sum, e1 / sum];
        // factor = exp(lse)/(exp(lse)+exp(0.5)) = 1/(1+exp(0.5-lse))
        let factor = 1.0f32 / (1.0 + (0.5f32 - lse_expect).exp());
        assert!((lse[0] - lse_expect).abs() < 1e-6, "{} vs {lse_expect}", lse[0]);
        for j in 0..2 {
            assert!((out[j] - out_expect[j] * factor).abs() < 1e-6, "{j}: {} vs {}", out[j], out_expect[j] * factor);
        }
        // The sink must not change the returned lse.
        let mut out2 = vec![0.0f32; 2];
        let mut lse2 = vec![0.0f32; 1];
        flash_mla_decode(&q, &cache, &indices, 1, 1, 2, 1.0, None, &mut out2, &mut lse2).unwrap();
        assert_eq!(lse[0], lse2[0]);
        // ... and with no valid index the sink leaves the zero output.
        let mut out3 = vec![7.0f32; 2];
        let mut lse3 = vec![0.0f32; 1];
        flash_mla_decode(&q, &cache, &[-1, -1], 1, 1, 2, 1.0, Some(&[0.5]), &mut out3, &mut lse3)
            .unwrap();
        assert!(out3.iter().all(|&v| v == 0.0));
        assert!(lse3[0] == f32::INFINITY);
    }

    #[test]
    fn decode_deterministic_bit_identical() {
        let mut rng = Philox::new(0xDE7E, 6);
        let (b, s_q, h, d, s_kv, topk) = (2usize, 2, 3, 32, 130, 100);
        let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
        let cache = build_cache(CacheFormat::Bf16, &rows, 0, 8);
        let indices = make_indices(&mut rng, &cache, s_kv, b * s_q * topk);
        let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
        let run = || {
            let mut out = vec![0.0f32; q.len()];
            let mut lse = vec![0.0f32; b * h * s_q];
            let sink = vec![0.1f32; h];
            flash_mla_decode(&q, &cache, &indices, b, s_q, topk, 0.2, Some(&sink), &mut out, &mut lse)
                .unwrap();
            (out, lse)
        };
        let (o1, l1) = run();
        let (o2, l2) = run();
        assert_eq!(o1, o2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn decode_validates_inputs() {
        let mut rng = Philox::new(0xBAD, 7);
        let (b, s_q, h, d, s_kv, topk) = (1usize, 2, 2, 32, 10, 4);
        let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(&mut rng, d, 1.0)).collect();
        let cache = build_cache(CacheFormat::Bf16, &rows, 0, 2);
        let q = random_vec(&mut rng, b * s_q * h * d, 1.0);
        let indices = vec![0i32; b * s_q * topk];
        let mut out = vec![0.0f32; q.len()];
        let mut lse = vec![0.0f32; b * h * s_q];
        // Wrong indices length.
        assert!(flash_mla_decode(&q, &cache, &indices[..3], b, s_q, topk, 1.0, None, &mut out, &mut lse).is_err());
        // q not a multiple of b*s_q*head_dim.
        assert!(flash_mla_decode(&q[..q.len() - 1], &cache, &indices, b, s_q, topk, 1.0, None, &mut out, &mut lse).is_err());
        // out/lse size mismatches.
        let out_short = vec![0.0f32; out.len() - 1];
        let mut out_short = out_short;
        assert!(flash_mla_decode(&q, &cache, &indices, b, s_q, topk, 1.0, None, &mut out_short, &mut lse).is_err());
        let mut lse_short = vec![0.0f32; lse.len() - 1];
        assert!(flash_mla_decode(&q, &cache, &indices, b, s_q, topk, 1.0, None, &mut out, &mut lse_short).is_err());
        // attn_sink length mismatch and non-finite sm_scale.
        let sink_long = vec![0.1f32; h + 1];
        assert!(flash_mla_decode(&q, &cache, &indices, b, s_q, topk, 1.0, Some(&sink_long), &mut out, &mut lse).is_err());
        assert!(flash_mla_decode(&q, &cache, &indices, b, s_q, topk, f32::NAN, None, &mut out, &mut lse).is_err());
        // Non-finite q.
        let mut bad_q = q.clone();
        bad_q[3] = f32::NAN;
        assert!(flash_mla_decode(&bad_q, &cache, &indices, b, s_q, topk, 1.0, None, &mut out, &mut lse).is_err());
        // Empty cache with non-empty q.
        let empty = MlaPagedCache::new(CacheFormat::Bf16, 64, 2);
        assert!(flash_mla_decode(&q, &empty, &indices, b, s_q, topk, 1.0, None, &mut out, &mut lse).is_err());
        // Empty cache with everything empty: vacuous Ok.
        assert!(flash_mla_decode(&[], &empty, &[], 0, 0, 0, 1.0, None, &mut [], &mut []).is_ok());
        // Vacuous b == 0 with empty buffers.
        assert!(flash_mla_decode(&[], &cache, &[], 0, 5, 0, 1.0, None, &mut [], &mut []).is_ok());
    }

    #[test]
    fn prop_decode_matches_naive() {
        prop("mla_decode_vs_naive", 0x51A1, 60, |_i, rng| {
            let b = rng.uniform_usize(1, 3);
            let s_q = rng.uniform_usize(1, 4);
            let h = rng.uniform_usize(1, 5);
            let d = 32 * rng.uniform_usize(1, 3);
            let s_kv = rng.uniform_usize(1, 160);
            let topk = rng.uniform_usize(0, 140);
            let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(rng, d, 1.0)).collect();
            let cache = build_cache(CacheFormat::Bf16, &rows, 0, s_kv.div_ceil(64).max(1) + 2);
            let indices = make_indices(rng, &cache, s_kv, b * s_q * topk);
            let q = random_vec(rng, b * s_q * h * d, 1.0);
            let sm = 1.0 / (d as f32).sqrt();
            let sink: Option<Vec<f32>> = if rng.next_f32() < 0.5 {
                Some(random_vec(rng, h, 0.5))
            } else {
                None
            };
            let mut out = vec![0.0f32; q.len()];
            let mut lse = vec![0.0f32; b * h * s_q];
            flash_mla_decode(
                &q,
                &cache,
                &indices,
                b,
                s_q,
                topk,
                sm,
                sink.as_deref(),
                &mut out,
                &mut lse,
            )?;
            let (nout, nlse) =
                naive_decode(&q, &cache, &indices, b, s_q, topk, h, d, sm, sink.as_deref());
            assert_close_slice(&out, &nout, 1e-4, "prop decode out");
            assert_close_slice(&lse, &nlse, 1e-4, "prop decode lse");
            Ok(())
        });
    }

    #[test]
    fn prop_decode_quantized_cache_accuracy() {
        prop("mla_decode_quant_accuracy", 0x51A2, 25, |_i, rng| {
            for (format, tol) in [
                (CacheFormat::Fp8Group32, 5e-2f32),
                (CacheFormat::Fp4Group16, 1.5f32),
            ] {
                let (b, s_q, h, d) = (1usize, rng.uniform_usize(1, 3), 2, 32);
                let s_kv = rng.uniform_usize(2, 80);
                let topk = rng.uniform_usize(1, s_kv.min(70));
                let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| random_vec(rng, d, 1.0)).collect();
                let cache = build_cache(format, &rows, 0, s_kv.div_ceil(64).max(1) + 1);
                let indices: Vec<i32> = (0..b * s_q * topk)
                    .map(|_| cache.physical_index(0, rng.uniform_usize(0, s_kv)).unwrap())
                    .collect();
                let q = random_vec(rng, b * s_q * h * d, 1.0);
                let sm = 1.0 / (d as f32).sqrt();
                let mut out = vec![0.0f32; q.len()];
                let mut lse = vec![0.0f32; b * h * s_q];
                flash_mla_decode(&q, &cache, &indices, b, s_q, topk, sm, None, &mut out, &mut lse)?;
                let mut oref = vec![0.0f32; out.len()];
                let mut lref = vec![0.0f32; lse.len()];
                naive_decode_rows(
                    &q, &rows, &indices, b, s_q, topk, h, d, sm, None, &mut oref, &mut lref,
                );
                assert_close_slice(&out, &oref, tol, "{format:?} out accuracy");
                assert_close_slice(&lse, &lref, tol, "{format:?} lse accuracy");
            }
            Ok(())
        });
    }

    #[test]
    fn decode_full_pipeline_smoke() {
        // project -> rope -> cache -> decode -> O-RoPE -> project_o.
        use crate::orope::o_rope_conjugate;
        let cfg = small_cfg(CacheFormat::Bf16, 16);
        let mut rng = Philox::new(0x91DE, 8);
        let w = crate::weights::MlaWeights::from_config(&cfg, &mut rng);
        let s = 5usize;
        let hidden = random_vec(&mut rng, s * cfg.hidden_size, 1.0);
        let mut q = vec![0.0f32; s * cfg.n_heads * cfg.head_dim];
        let mut kv = vec![0.0f32; s * cfg.head_dim];
        w.project_q(&hidden, s, &mut q).unwrap();
        w.project_kv(&hidden, s, &mut kv).unwrap();
        let positions: Vec<u32> = (0..s as u32).collect();
        let latent = cfg.head_dim - cfg.rope_dim;
        w.rope
            .apply_at(&positions, &mut kv, cfg.head_dim, latent)
            .unwrap();
        let mut qpos = Vec::with_capacity(s * cfg.n_heads);
        for &p in &positions {
            for _ in 0..cfg.n_heads {
                qpos.push(p);
            }
        }
        w.rope
            .apply_at(&qpos, &mut q, cfg.head_dim, latent)
            .unwrap();
        let mut cache = MlaPagedCache::new(CacheFormat::Bf16, 64, 8);
        for t in 0..s {
            cache.append(0, &kv[t * cfg.head_dim..(t + 1) * cfg.head_dim]).unwrap();
        }
        let idx: Vec<i32> = (0..s).map(|t| cache.physical_index(0, t).unwrap()).collect();
        let mut indices = Vec::new();
        for _ in 0..s {
            indices.extend_from_slice(&idx);
        }
        let mut out = vec![0.0f32; q.len()];
        let mut lse = vec![0.0f32; cfg.n_heads * s];
        flash_mla_decode(
            &q,
            &cache,
            &indices,
            1,
            s,
            s,
            cfg.sm_scale,
            w.attn_sink.as_deref(),
            &mut out,
            &mut lse,
        )
        .unwrap();
        o_rope_conjugate(&w.rope, &qpos, &mut out, s * cfg.n_heads, cfg.head_dim).unwrap();
        let mut proj = vec![0.0f32; s * cfg.hidden_size];
        w.project_o(&out, s, &mut proj).unwrap();
        assert!(proj.iter().all(|v| v.is_finite()));
        assert!(lse.iter().all(|v| v.is_finite()));
    }
}
