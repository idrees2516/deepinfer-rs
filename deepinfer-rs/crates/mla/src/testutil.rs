//! Shared test-only reference implementations ("naive" transcriptions of
//! the FlashMLA semantics) and small-config helpers.

use crate::cache::MlaPagedCache;
use crate::config::{CacheFormat, MlaConfig};
use deepinfer_core::math::dot;

/// A small but fully-featured MLA config for tests (head_dim 64 = latent
/// 48 + rope 16; widths chosen so every format's divisibility holds).
pub(crate) fn small_cfg(cache: CacheFormat, max_pages: usize) -> MlaConfig {
    MlaConfig {
        hidden_size: 32,
        q_lora_rank: 16,
        n_heads: 4,
        head_dim: 64,
        rope_dim: 16,
        o_groups: 2,
        o_lora_rank: 24,
        rope_base: 10_000.0,
        yarn: None,
        rms_eps: 1e-4,
        sm_scale: 1.0 / 8.0,
        cache,
        max_pages,
    }
}

/// Builds a paged cache holding `rows` appended to one fresh sequence
/// (pages allocated in order, so physical index == logical position).
pub(crate) fn build_cache(format: CacheFormat, rows: &[Vec<f32>], seq: u32, max_pages: usize) -> MlaPagedCache {
    let mut cache = MlaPagedCache::new(format, 64, max_pages);
    for row in rows {
        cache.append(seq, row).unwrap();
    }
    cache
}

/// Naive single-pass decode reference: gathers the selected rows from the
/// cache (dequantized), computes full softmax per (batch, query, head)
/// and the weighted sum. Mirrors the FlashMLA decode contract including
/// the empty-selection edge case (`lse = +inf`, zero output) and the
/// `attn_sink` epilogue.
pub(crate) fn naive_decode(
    q: &[f32],
    cache: &MlaPagedCache,
    indices: &[i32],
    b: usize,
    s_q: usize,
    topk: usize,
    h: usize,
    d: usize,
    sm_scale: f32,
    sink: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; b * s_q * h * d];
    let mut lse = vec![0.0f32; b * h * s_q];
    for bi in 0..b {
        for qi in 0..s_q {
            let rows: Vec<Vec<f32>> = indices[(bi * s_q + qi) * topk..][..topk]
                .iter()
                .filter(|&&i| i >= 0)
                .filter_map(|&i| cache.row_f32(i).ok())
                .collect();
            for hi in 0..h {
                let obase = ((bi * s_q + qi) * h + hi) * d;
                let li = (bi * h + hi) * s_q + qi;
                if rows.is_empty() {
                    lse[li] = f32::INFINITY;
                    continue;
                }
                let q_row = &q[obase..obase + d];
                let scores: Vec<f32> = rows.iter().map(|r| dot(q_row, r) * sm_scale).collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for j in 0..d {
                    out[obase + j] = rows
                        .iter()
                        .zip(exps.iter())
                        .map(|(r, &e)| e * r[j])
                        .sum::<f32>()
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
    (out, lse)
}

/// Naive direct transcription of the FlashMLA README sparse-prefill
/// pseudocode (gather, mask, softmax, PV) including `topk_length`,
/// the empty-selection edge case and the `attn_sink` epilogue.
pub(crate) fn naive_sparse_prefill(
    q: &[f32],
    kv: &[f32],
    indices: &[i32],
    s_q: usize,
    h: usize,
    d: usize,
    s_kv: usize,
    topk: usize,
    sm_scale: f32,
    sink: Option<&[f32]>,
    topk_length: Option<&[i32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; s_q * h * d];
    let mut max_logits = vec![f32::NEG_INFINITY; s_q * h];
    let mut lse = vec![f32::INFINITY; s_q * h];
    for i in 0..s_q {
        let valid_len = topk_length.map(|t| t[i] as usize).unwrap_or(topk);
        // invalid = (indices < 0) | (indices >= s_kv) | t >= valid_len;
        // indices = where(invalid, 0, indices)
        let invalid: Vec<bool> = (0..topk)
            .map(|t| {
                let raw = indices[i * topk + t];
                raw < 0 || (raw as usize) >= s_kv || t >= valid_len
            })
            .collect();
        let remapped: Vec<usize> = (0..topk)
            .map(|t| if invalid[t] { 0 } else { indices[i * topk + t] as usize })
            .collect();
        // focused_kv = kv2[indices]
        let focused: Vec<&[f32]> = remapped.iter().map(|&j| &kv[j * d..(j + 1) * d]).collect();
        for hi in 0..h {
            let q_row = &q[(i * h + hi) * d..(i * h + hi + 1) * d];
            // P = (Q @ focused_kv^T) * sm_scale; P = where(invalid, -inf, P)
            let mut p: Vec<f32> = focused.iter().map(|f| dot(q_row, f) * sm_scale).collect();
            for (t, pv) in p.iter_mut().enumerate() {
                if invalid[t] {
                    *pv = f32::NEG_INFINITY;
                }
            }
            // max_logits = P.max(-1)
            let m = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            max_logits[i * h + hi] = m;
            if m == f32::NEG_INFINITY {
                continue; // edge: lse stays +inf, out stays 0
            }
            // lse = logsumexp(P, -1)
            let e: Vec<f32> = p.iter().map(|&s| (s - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            let l = m + sum.ln();
            lse[i * h + hi] = l;
            // out = softmax(P) @ focused_kv
            let obase = (i * h + hi) * d;
            for t in 0..topk {
                let w = e[t] / sum;
                for j in 0..d {
                    out[obase + j] += w * focused[t][j];
                }
            }
            // attn_sink epilogue
            if let Some(sk) = sink {
                let f = 1.0 / (1.0 + (sk[hi] - l).exp());
                for j in 0..d {
                    out[obase + j] *= f;
                }
            }
        }
    }
    (out, max_logits, lse)
}

/// Naive causal dense prefill: per (query, head) full softmax over the
/// causal key range `[0, min(i + q_offset + 1, s_kv))`.
pub(crate) fn naive_dense_prefill(
    q: &[f32],
    kv: &[f32],
    s_q: usize,
    h: usize,
    d: usize,
    s_kv: usize,
    q_offset: usize,
    sm_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; s_q * h * d];
    for i in 0..s_q {
        let limit = ((i as u64) + (q_offset as u64) + 1).min(s_kv as u64) as usize;
        for hi in 0..h {
            let obase = (i * h + hi) * d;
            if limit == 0 {
                continue;
            }
            let q_row = &q[obase..obase + d];
            let scores: Vec<f32> = (0..limit)
                .map(|k| dot(q_row, &kv[k * d..(k + 1) * d]) * sm_scale)
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..d {
                out[obase + j] = (0..limit)
                    .map(|k| exps[k] * kv[k * d + j])
                    .sum::<f32>()
                    / sum;
            }
        }
    }
    out
}
