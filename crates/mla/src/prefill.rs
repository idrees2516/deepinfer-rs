//! Sparse (DSA) and dense causal MLA prefill.
//!
//! `flash_mla_sparse_prefill` is a faithful transcription of the FlashMLA
//! README pseudocode:
//!
//! ```text
//! kv2 = kv.squeeze(1)                      # [s_kv, d]
//! invalid = (indices < 0) | (indices >= s_kv)
//! indices = where(invalid, 0, indices)
//! focused_kv = kv2[indices]                # [s_q, topk, d]
//! P = (Q @ focused_kv^T) * sm_scale        # [s_q, h, topk]
//! P = where(invalid, -inf, P)
//! max_logits = P.max(-1)                   # [s_q, h]
//! lse = logsumexp(P, -1)
//! S = softmax(P, -1)
//! out = S @ focused_kv                     # [s_q, h, d]
//! # edge case: no valid index -> max_logits = -inf, lse = +inf, out = 0
//! # attn_sink: out *= exp(lse)/(exp(lse)+exp(sink)) if provided
//! ```
//!
//! plus `topk_length [s_q]`: query i only attends `indices[0..topk_length[i]]`.
//!
//! `flash_mla_dense_prefill` is the causal (non-sparse) variant used by
//! decoder layers without DSA: query i attends keys
//! `pos <= i + q_offset`, processed in 64-key blocks with the same
//! running-`(m, l, o)` online-softmax merge as the decode kernel.

use deepinfer_core::math::{axpy, dot};
use deepinfer_core::{ensure, Error, Result};

/// KV keys processed per block (FlashMLA page size).
const BLOCK: usize = 64;

/// Validates the shared shape contract of the prefill kernels.
fn validate_prefill_shapes(
    q: &[f32],
    kv: &[f32],
    s_q: usize,
    h: usize,
    d_qk: usize,
    s_kv: usize,
    sm_scale: f32,
    attn_sink: Option<&[f32]>,
) -> Result<()> {
    ensure!(
        sm_scale.is_finite(),
        Numerical,
        "sm_scale must be finite, got {sm_scale}"
    );
    let n_q = s_q
        .checked_mul(h)
        .and_then(|v| v.checked_mul(d_qk))
        .ok_or_else(|| Error::DimensionMismatch("s_q*h*d_qk overflows".into()))?;
    ensure!(
        q.len() == n_q,
        DimensionMismatch,
        "q has {} elems, expected s_q*h*d_qk = {n_q}",
        q.len()
    );
    let n_kv = s_kv
        .checked_mul(d_qk)
        .ok_or_else(|| Error::DimensionMismatch("s_kv*d_qk overflows".into()))?;
    ensure!(
        kv.len() == n_kv,
        DimensionMismatch,
        "kv has {} elems, expected s_kv*d_qk = {n_kv}",
        kv.len()
    );
    if let Some(sk) = attn_sink {
        ensure!(
            sk.len() == h,
            DimensionMismatch,
            "attn_sink has {} elems, expected h = {h}",
            sk.len()
        );
        ensure!(
            sk.iter().all(|v| v.is_finite()),
            Numerical,
            "attn_sink contains non-finite values"
        );
    }
    ensure!(
        q.iter().all(|v| v.is_finite()),
        Numerical,
        "q contains non-finite values"
    );
    ensure!(
        kv.iter().all(|v| v.is_finite()),
        Numerical,
        "kv contains non-finite values"
    );
    Ok(())
}

/// Sparse prefill: EXACT equivalent of the FlashMLA README pseudocode
/// (see the [module docs](self) for the transcription), including the
/// empty-selection edge case (`max_logits = -inf`, `lse = +inf`, zero
/// output) and the `attn_sink` epilogue
/// `out *= exp(lse) / (exp(lse) + exp(sink))` (stable form), which does
/// NOT modify the returned `max_logits`/`lse`.
///
/// Layouts: `q` `[s_q, h, d_qk]` flat; `kv` `[s_kv, d_qk]` flat (the MQA
/// cache row, `h_kv = 1`); `indices` `[s_q, topk]` int32 where `-1` or
/// `>= s_kv` marks invalid entries; `topk_length` (optional) caps the
/// valid prefix per query. Returns `(out [s_q*h*d_qk], max_logits
/// [s_q, h], lse [s_q, h])`.
///
/// # Errors
/// [`Error::DimensionMismatch`] on shape mismatches, [`Error::OutOfRange`]
/// for malformed `topk_length`, [`Error::Numerical`] for non-finite
/// inputs or scores.
pub fn flash_mla_sparse_prefill(
    q: &[f32],
    kv: &[f32],
    indices: &[i32],
    s_q: usize,
    h: usize,
    d_qk: usize,
    s_kv: usize,
    topk: usize,
    sm_scale: f32,
    attn_sink: Option<&[f32]>,
    topk_length: Option<&[i32]>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    validate_prefill_shapes(q, kv, s_q, h, d_qk, s_kv, sm_scale, attn_sink)?;
    let n_idx = s_q
        .checked_mul(topk)
        .ok_or_else(|| Error::DimensionMismatch("s_q*topk overflows".into()))?;
    ensure!(
        indices.len() == n_idx,
        DimensionMismatch,
        "indices has {} elems, expected s_q*topk = {n_idx}",
        indices.len()
    );
    if let Some(tl) = topk_length {
        ensure!(
            tl.len() == s_q,
            DimensionMismatch,
            "topk_length has {} elems, expected s_q = {s_q}",
            tl.len()
        );
        for (i, &v) in tl.iter().enumerate() {
            ensure!(
                v >= 0 && (v as usize) <= topk,
                OutOfRange,
                "topk_length[{i}] = {v} outside [0, {topk}]"
            );
        }
    }
    let n_out = q.len();
    let mut out = vec![0.0f32; n_out];
    let mut max_logits = vec![f32::NEG_INFINITY; s_q * h];
    let mut lse = vec![f32::INFINITY; s_q * h];
    // Per-query score matrix [h][topk] and softmax weights, hoisted.
    let mut scores = vec![0.0f32; h.checked_mul(topk).unwrap_or(0)];
    let mut exps = vec![0.0f32; topk];

    for i in 0..s_q {
        let valid_len = topk_length.map(|tl| tl[i] as usize).unwrap_or(topk);
        // P = (Q @ focused_kv^T) * sm_scale, invalid slots -> -inf.
        // (The pseudocode gathers kv[0] for invalid slots and masks the
        // scores to -inf; skipping the gather is identical because the
        // softmax weight of a -inf slot is exactly zero and kv is finite.)
        for t in 0..topk {
            let raw = indices[i * topk + t];
            let valid = raw >= 0 && (raw as usize) < s_kv && t < valid_len;
            if !valid {
                for hi in 0..h {
                    scores[hi * topk + t] = f32::NEG_INFINITY;
                }
                continue;
            }
            let krow = &kv[(raw as usize) * d_qk..(raw as usize + 1) * d_qk];
            for hi in 0..h {
                let qrow = &q[(i * h + hi) * d_qk..(i * h + hi + 1) * d_qk];
                let s = dot(qrow, krow) * sm_scale;
                ensure!(
                    s.is_finite(),
                    Numerical,
                    "non-finite attention score (query {i}, head {hi}, slot {t})"
                );
                scores[hi * topk + t] = s;
            }
        }
        for hi in 0..h {
            let row = &scores[hi * topk..(hi + 1) * topk];
            let mut m = f32::NEG_INFINITY;
            for &v in row {
                if v > m {
                    m = v;
                }
            }
            max_logits[i * h + hi] = m;
            if m == f32::NEG_INFINITY {
                // Edge case: no valid index for this query.
                continue;
            }
            let mut l = 0.0f32;
            for t in 0..topk {
                let e = (row[t] - m).exp();
                exps[t] = e;
                l += e;
            }
            let lse_v = m + l.ln();
            lse[i * h + hi] = lse_v;
            // out = S @ focused_kv with fixed slot order.
            let obase = (i * h + hi) * d_qk;
            for t in 0..topk {
                let w = exps[t] / l;
                if w == 0.0 {
                    continue;
                }
                let raw = indices[i * topk + t];
                if raw < 0 || (raw as usize) >= s_kv {
                    continue; // defensive: zero-weight anyway
                }
                let krow = &kv[(raw as usize) * d_qk..(raw as usize + 1) * d_qk];
                axpy(w, krow, &mut out[obase..obase + d_qk]);
            }
            if let Some(sk) = attn_sink {
                // exp(lse)/(exp(lse)+exp(sink)) == 1/(1+exp(sink-lse)).
                let f = 1.0 / (1.0 + (sk[hi] - lse_v).exp());
                for v in out[obase..obase + d_qk].iter_mut() {
                    *v *= f;
                }
            }
        }
    }
    Ok((out, max_logits, lse))
}

/// Dense causal prefill over the full KV — used by decoder layers without
/// DSA. Query `i` attends every key with `key_pos <= i + q_offset`
/// (`q_offset` = number of KV tokens already processed before this q
/// block). Keys are processed in 64-token blocks with a running
/// `(m, l, o)` online-softmax state merged per block (the same split-K
/// structure as [`crate::flash_mla_decode`]).
///
/// Layouts: `q` `[s_q, h, d_qk]`, `kv` `[s_kv, d_qk]`, `out`
/// `[s_q, h, d_qk]` (fully written; queries with no valid keys — only
/// possible when `s_kv == 0` — produce zero rows).
///
/// # Errors
/// [`Error::DimensionMismatch`] on shape mismatches, [`Error::Numerical`]
/// for non-finite inputs or scores.
pub fn flash_mla_dense_prefill(
    q: &[f32],
    kv: &[f32],
    s_q: usize,
    h: usize,
    d_qk: usize,
    s_kv: usize,
    q_offset: usize,
    sm_scale: f32,
    out: &mut [f32],
) -> Result<()> {
    validate_prefill_shapes(q, kv, s_q, h, d_qk, s_kv, sm_scale, None)?;
    ensure!(
        out.len() == q.len(),
        DimensionMismatch,
        "out has {} elems, expected {}",
        out.len(),
        q.len()
    );
    out.fill(0.0);
    let mut m = vec![f32::NEG_INFINITY; s_q * h];
    let mut l = vec![0.0f32; s_q * h];
    let mut mc = vec![f32::NEG_INFINITY; h];
    let mut lc = vec![0.0f32; h];
    let mut oc = vec![0.0f32; h.checked_mul(d_qk).unwrap_or(0)];

    for i in 0..s_q {
        // Causal limit: key pos <= i + q_offset (u64 math: q_offset may
        // be arbitrarily large).
        let limit = ((i as u64) + (q_offset as u64) + 1).min(s_kv as u64) as usize;
        if limit == 0 {
            continue;
        }
        for cs in (0..limit).step_by(BLOCK) {
            let ce = (cs + BLOCK).min(limit);
            mc.fill(f32::NEG_INFINITY);
            lc.fill(0.0);
            oc.fill(0.0);
            for k in cs..ce {
                let krow = &kv[k * d_qk..(k + 1) * d_qk];
                for hi in 0..h {
                    let qrow = &q[(i * h + hi) * d_qk..(i * h + hi + 1) * d_qk];
                    let s = dot(qrow, krow) * sm_scale;
                    ensure!(
                        s.is_finite(),
                        Numerical,
                        "non-finite attention score (query {i}, head {hi}, key {k})"
                    );
                    if s > mc[hi] {
                        let res = (mc[hi] - s).exp();
                        deepinfer_core::math::scale(&mut oc[hi * d_qk..(hi + 1) * d_qk], res);
                        lc[hi] *= res;
                        mc[hi] = s;
                    }
                    let p = (s - mc[hi]).exp();
                    lc[hi] += p;
                    axpy(p, krow, &mut oc[hi * d_qk..(hi + 1) * d_qk]);
                }
            }
            // Merge the block partial into the running state.
            for hi in 0..h {
                let m_new = m[i * h + hi].max(mc[hi]);
                if m_new == f32::NEG_INFINITY {
                    continue;
                }
                let so = (m[i * h + hi] - m_new).exp();
                let sc = (mc[hi] - m_new).exp();
                let obase = (i * h + hi) * d_qk;
                for j in 0..d_qk {
                    out[obase + j] = out[obase + j] * so + oc[hi * d_qk + j] * sc;
                }
                l[i * h + hi] = l[i * h + hi] * so + lc[hi] * sc;
                m[i * h + hi] = m_new;
            }
        }
        // Normalize.
        for hi in 0..h {
            if l[i * h + hi] > 0.0 {
                let obase = (i * h + hi) * d_qk;
                for j in 0..d_qk {
                    out[obase + j] /= l[i * h + hi];
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{naive_dense_prefill, naive_sparse_prefill};
    use deepinfer_core::prop::{assert_close, assert_close_slice, prop, random_vec};
    use deepinfer_core::rng::Philox;

    #[test]
    fn sparse_prefill_matches_pseudocode_transcription() {
        let mut rng = Philox::new(0x5F1, 1);
        let (s_q, h, d, s_kv, topk) = (5usize, 3, 32, 40, 17);
        let q = random_vec(&mut rng, s_q * h * d, 1.0);
        let kv = random_vec(&mut rng, s_kv * d, 1.0);
        let mut indices = Vec::with_capacity(s_q * topk);
        for _ in 0..s_q * topk {
            match rng.uniform_usize(0, 6) {
                0 => indices.push(-1),
                1 => indices.push((s_kv + rng.uniform_usize(0, 20)) as i32),
                _ => indices.push(rng.uniform_usize(0, s_kv) as i32),
            }
        }
        let sm = 1.0 / (d as f32).sqrt();
        let (out, max_l, lse) =
            flash_mla_sparse_prefill(&q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, None)
                .unwrap();
        let (nout, nmax, nlse) = naive_sparse_prefill(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, None,
        );
        assert_close_slice(&out, &nout, 1e-4, "sparse out");
        assert_close_slice(&max_l, &nmax, 1e-4, "sparse max_logits");
        assert_close_slice(&lse, &nlse, 1e-4, "sparse lse");
    }

    #[test]
    fn sparse_prefill_topk_length_semantics() {
        let mut rng = Philox::new(0x71A, 2);
        let (s_q, h, d, s_kv, topk) = (3usize, 2, 16, 10, 6);
        let q = random_vec(&mut rng, s_q * h * d, 1.0);
        let kv = random_vec(&mut rng, s_kv * d, 1.0);
        let indices: Vec<i32> = (0..s_q * topk).map(|_| rng.uniform_usize(0, s_kv) as i32).collect();
        let tl = [4i32, 0, 6];
        let sm = 0.3f32;
        let (out, max_l, lse) = flash_mla_sparse_prefill(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, Some(&tl),
        )
        .unwrap();
        let (nout, nmax, nlse) = naive_sparse_prefill(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, Some(&tl),
        );
        assert_close_slice(&out, &nout, 1e-4, "topk_length out");
        assert_close_slice(&max_l, &nmax, 1e-4, "topk_length max");
        assert_close_slice(&lse, &nlse, 1e-4, "topk_length lse");
        // tl[1] = 0: query 1 has no valid index -> edge values.
        for hi in 0..h {
            assert!(max_l[1 * h + hi] == f32::NEG_INFINITY);
            assert!(lse[1 * h + hi] == f32::INFINITY);
            assert!(out[(1 * h + hi) * d..][..d].iter().all(|&v| v == 0.0));
        }    }

    #[test]
    fn sparse_prefill_edge_cases() {
        let (s_q, h, d, s_kv) = (2usize, 2, 8, 5);
        let q = random_vec(&mut Philox::new(1, 1), s_q * h * d, 1.0);
        let kv = random_vec(&mut Philox::new(1, 2), s_kv * d, 1.0);
        // All invalid.
        let (out, max_l, lse) = flash_mla_sparse_prefill(
            &q, &kv, &[-1, -1, 99, -1], s_q, h, d, s_kv, 2, 1.0, None, None,
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(max_l.iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(lse.iter().all(|&v| v == f32::INFINITY));
        // topk = 0: every query selects nothing -> full-size edge values.
        let (out, max_l, lse) =
            flash_mla_sparse_prefill(&q, &kv, &[], s_q, h, d, s_kv, 0, 1.0, None, None).unwrap();
        assert_eq!(out.len(), s_q * h * d);
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(max_l.iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(lse.iter().all(|&v| v == f32::INFINITY));
        // s_kv = 0: every index invalid.
        let (out, max_l, lse) = flash_mla_sparse_prefill(
            &q, &[], &[0, 0], s_q, h, d, 0, 1, 1.0, None, None,
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(max_l.iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(lse.iter().all(|&v| v == f32::INFINITY));
        // s_q = 0: vacuous.
        let r = flash_mla_sparse_prefill(&[], &kv, &[], 0, h, d, s_kv, 3, 1.0, None, None);
        assert!(r.unwrap().0.is_empty());
    }

    #[test]
    fn sparse_prefill_attn_sink_hand_computed() {
        // q = [1, 0]; kv = [[1,0],[0,1]]; sm_scale = 1; sink = 0.5.
        let q = vec![1.0f32, 0.0];
        let kv = vec![1.0f32, 0.0, 0.0, 1.0];
        let (out, max_l, lse) = flash_mla_sparse_prefill(
            &q, &kv, &[0, 1], 1, 1, 2, 2, 2, 1.0, Some(&[0.5]), None,
        )
        .unwrap();
        let e1 = (-1.0f32).exp();
        let sum = 1.0 + e1;
        let lse_expect = 1.0f32 + sum.ln();
        let factor = 1.0f32 / (1.0 + (0.5f32 - lse_expect).exp());
        assert_close(max_l[0], 1.0, 1e-6, "max logit");
        assert_close(lse[0], lse_expect, 1e-6, "lse");
        assert_close(out[0], (1.0 / sum) * factor, 1e-6, "out[0]");
        assert_close(out[1], (e1 / sum) * factor, 1e-6, "out[1]");
        // Sink does not change max_logits / lse.
        let (_, max2, lse2) =
            flash_mla_sparse_prefill(&q, &kv, &[0, 1], 1, 1, 2, 2, 2, 1.0, None, None).unwrap();
        assert_eq!(max_l, max2);
        assert_eq!(lse, lse2);
    }

    #[test]
    fn sparse_prefill_validates_inputs() {
        let q = vec![0.1f32; 2 * 3 * 4];
        let kv = vec![0.1f32; 5 * 4];
        let indices = vec![0i32; 2 * 3];
        let r = |indices: &[i32], tl: Option<&[i32]>, sink: Option<&[f32]>| {
            flash_mla_sparse_prefill(q.as_slice(), &kv, indices, 2, 3, 4, 5, 3, 1.0, sink, tl)
        };
        assert!(r(&indices[..5], None, None).is_err()); // indices len
        assert!(r(&indices, Some(&[1]), None).is_err()); // topk_length len (1 != s_q=2)
        assert!(r(&indices, Some(&[3, -1]), None).is_err()); // negative
        assert!(r(&indices, Some(&[3, 4]), None).is_err()); // > topk
        assert!(r(&indices, None, Some(&[0.1])).is_err()); // sink len
        assert!(r(&indices, None, Some(&[f32::NAN, 0.0, 0.0])).is_err()); // sink NaN
        // Bad q / kv / sm_scale.
        assert!(flash_mla_sparse_prefill(&q[..q.len() - 1], &kv, &indices, 2, 3, 4, 5, 3, 1.0, None, None).is_err());
        assert!(flash_mla_sparse_prefill(&q, &kv[..kv.len() - 1], &indices, 2, 3, 4, 5, 3, 1.0, None, None).is_err());
        assert!(flash_mla_sparse_prefill(&q, &kv, &indices, 2, 3, 4, 5, 3, f32::INFINITY, None, None).is_err());
        let mut nq = q.clone();
        nq[0] = f32::NAN;
        assert!(flash_mla_sparse_prefill(&nq, &kv, &indices, 2, 3, 4, 5, 3, 1.0, None, None).is_err());
    }

    #[test]
    fn sparse_prefill_deterministic() {
        let mut rng = Philox::new(0xDE7, 3);
        let (s_q, h, d, s_kv, topk) = (4usize, 2, 16, 30, 12);
        let q = random_vec(&mut rng, s_q * h * d, 1.0);
        let kv = random_vec(&mut rng, s_kv * d, 1.0);
        let indices: Vec<i32> = (0..s_q * topk)
            .map(|_| if rng.next_f32() < 0.2 { -1 } else { rng.uniform_usize(0, s_kv) as i32 })
            .collect();
        let run = || {
            flash_mla_sparse_prefill(&q, &kv, &indices, s_q, h, d, s_kv, topk, 0.2, Some(&[0.3, -0.2]), None)
                .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    #[test]
    fn dense_prefill_matches_naive() {
        let mut rng = Philox::new(0xDE5, 4);
        for (s_q, h, d, s_kv, q_offset) in [
            (5usize, 2, 16, 30usize, 0usize),
            (3, 3, 8, 100, 5),   // spans two 64-blocks
            (70, 1, 8, 70, 0),   // queries span blocks
            (4, 2, 32, 200, 50), // many blocks + offset
        ] {
            let q = random_vec(&mut rng, s_q * h * d, 1.0);
            let kv = random_vec(&mut rng, s_kv * d, 1.0);
            let sm = 1.0 / (d as f32).sqrt();
            let mut out = vec![0.0f32; q.len()];
            flash_mla_dense_prefill(&q, &kv, s_q, h, d, s_kv, q_offset, sm, &mut out).unwrap();
            let nout = naive_dense_prefill(&q, &kv, s_q, h, d, s_kv, q_offset, sm);
            assert_close_slice(&out, &nout, 1e-4, "dense (s_q {s_q}, s_kv {s_kv}, off {q_offset})");
        }
    }

    #[test]
    fn dense_prefill_edges() {
        // s_kv = 0: zero output.
        let q = vec![0.5f32; 2 * 2 * 4];
        let mut out = vec![1.0f32; q.len()];
        flash_mla_dense_prefill(&q, &[], 2, 2, 4, 0, 0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        // Huge q_offset: full (non-causal) attention over all keys.
        let kv = vec![0.25f32; 3 * 4];
        let mut out2 = vec![0.0f32; q.len()];
        flash_mla_dense_prefill(&q, &kv, 2, 2, 4, 3, usize::MAX / 2, 1.0, &mut out2).unwrap();
        let full = naive_dense_prefill(&q, &kv, 2, 2, 4, 3, usize::MAX / 2, 1.0);
        assert_close_slice(&out2, &full, 1e-4, "huge offset");
        // Shape validation.
        let mut out3 = vec![0.0f32; out2.len() - 1];
        assert!(flash_mla_dense_prefill(&q, &kv, 2, 2, 4, 3, 0, 1.0, &mut out3).is_err());
        assert!(flash_mla_dense_prefill(&q[..q.len() - 1], &kv, 2, 2, 4, 3, 0, 1.0, &mut out2).is_err());
        assert!(flash_mla_dense_prefill(&q, &kv, 2, 2, 4, 3, 0, f32::NAN, &mut out2).is_err());
        // s_q = 0: vacuous.
        assert!(flash_mla_dense_prefill(&[], &kv, 0, 2, 4, 3, 0, 1.0, &mut []).is_ok());
    }

    #[test]
    fn prop_sparse_prefill_matches_naive() {
        prop("mla_sparse_prefill_vs_naive", 0x5F51, 60, |_i, rng| {
            let s_q = rng.uniform_usize(1, 6);
            let h = rng.uniform_usize(1, 4);
            let d = 8 * rng.uniform_usize(1, 5);
            let s_kv = rng.uniform_usize(1, 50);
            let topk = rng.uniform_usize(0, 25);
            let q = random_vec(rng, s_q * h * d, 1.0);
            let kv = random_vec(rng, s_kv * d, 1.0);
            let indices: Vec<i32> = (0..s_q * topk)
                .map(|_| match rng.uniform_usize(0, 8) {
                    0 => -1,
                    1 => (s_kv + rng.uniform_usize(0, 10)) as i32,
                    _ => rng.uniform_usize(0, s_kv) as i32,
                })
                .collect();
            let tl: Option<Vec<i32>> = if topk > 0 && rng.next_f32() < 0.5 {
                Some((0..s_q).map(|_| rng.uniform_usize(0, topk + 1) as i32).collect())
            } else {
                None
            };
            let sm = 1.0 / (d as f32).sqrt();
            let sink: Option<Vec<f32>> = if rng.next_f32() < 0.5 {
                Some(random_vec(rng, h, 0.5))
            } else {
                None
            };
            let (out, max_l, lse) = flash_mla_sparse_prefill(
                &q,
                &kv,
                &indices,
                s_q,
                h,
                d,
                s_kv,
                topk,
                sm,
                sink.as_deref(),
                tl.as_deref(),
            )?;
            let (nout, nmax, nlse) = naive_sparse_prefill(
                &q,
                &kv,
                &indices,
                s_q,
                h,
                d,
                s_kv,
                topk,
                sm,
                sink.as_deref(),
                tl.as_deref(),
            );
            assert_close_slice(&out, &nout, 1e-4, "prop sparse out");
            assert_close_slice(&max_l, &nmax, 1e-4, "prop sparse max");
            assert_close_slice(&lse, &nlse, 1e-4, "prop sparse lse");
            Ok(())
        });
    }

    #[test]
    fn prop_dense_prefill_matches_naive() {
        prop("mla_dense_prefill_vs_naive", 0x0E5E, 60, |_i, rng| {
            let s_q = rng.uniform_usize(1, 8);
            let h = rng.uniform_usize(1, 4);
            let d = 8 * rng.uniform_usize(1, 4);
            let s_kv = rng.uniform_usize(1, 140);
            let q_offset = rng.uniform_usize(0, 40);
            let q = random_vec(rng, s_q * h * d, 1.0);
            let kv = random_vec(rng, s_kv * d, 1.0);
            let sm = 1.0 / (d as f32).sqrt();
            let mut out = vec![0.0f32; q.len()];
            flash_mla_dense_prefill(&q, &kv, s_q, h, d, s_kv, q_offset, sm, &mut out)?;
            let nout = naive_dense_prefill(&q, &kv, s_q, h, d, s_kv, q_offset, sm);
            assert_close_slice(&out, &nout, 1e-4, "prop dense out");
            Ok(())
        });
    }
}
