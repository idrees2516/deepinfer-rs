//! Fused attention epilogue: sparse attention + optional O-RoPE
//! conjugate + per-32 e4m3/e8m0 output cast.
//!
//! Mirrors the FlashMLA V4.1 fused kernel composition (README: "Q-RoPE,
//! attention, O-RoPE conjugate, cast to FP8 per-32 e8m0 packed,
//! consumed by deep_gemm.fp8_einsum('bhr,hdr->bhd', recipe (1,1,32))"):
//! the (already Q-RoPE'd) query runs through
//! [`crate::flash_mla_sparse_prefill`], the output's rope slice is
//! optionally conjugated (O-RoPE), and every `(query, head)` row is cast
//! to E4M3 values plus one UE8M0 scale per 32 dims — directly consumable
//! as the `q`/`q_sf` operands of the gemm crate's
//! `fp8_einsum_bhr_hdr_bhd` with `b = s_q`, `h`, `r = d_qk`,
//! `group = 32`.

use crate::fp8cache;
use crate::orope::o_rope_conjugate;
use crate::prefill::flash_mla_sparse_prefill;
use deepinfer_core::rope::Rope;
use deepinfer_core::{ensure, Error, Result};

/// Group size of the output cast (FlashMLA `num_per_channels = 32`).
pub(crate) const OUT_CAST_GROUP: usize = 32;

/// Fused sparse-attention output cast (see the [module
/// docs](self)). Runs [`crate::flash_mla_sparse_prefill`] with the given
/// operands, optionally applies the O-RoPE conjugate (`o_rope` = rope
/// table + one position per query token; every head row of a token shares
/// the token's position), then casts each `[d_qk]` output row to E4M3 +
/// one UE8M0 scale per 32 dims.
///
/// Returns `(data, sf)`:
/// * `data`: `[s_q * h * d_qk]` E4M3 bytes, row-major over
///   `(query, head)` rows;
/// * `sf`: `[s_q * h * (d_qk / 32)]` UE8M0 bytes.
///
/// `d_qk` must be a multiple of 32.
///
/// # Errors
/// Propagates the prefill / O-RoPE validation errors; adds
/// [`Error::InvalidArgument`] when `d_qk % 32 != 0` or the O-RoPE
/// position count is wrong.
pub fn fused_attention_out_fp8(
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
    o_rope: Option<(&Rope, &[u32])>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (mut attn, _max_logits, _lse) = flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        s_q,
        h,
        d_qk,
        s_kv,
        topk,
        sm_scale,
        attn_sink,
        topk_length,
    )?;
    if let Some((rope, positions)) = o_rope {
        ensure!(
            positions.len() == s_q,
            DimensionMismatch,
            "o_rope needs one position per query ({}), got {}",
            s_q,
            positions.len()
        );
        let rows = s_q
            .checked_mul(h)
            .ok_or_else(|| Error::DimensionMismatch("s_q*h overflows".into()))?;
        let mut row_pos = Vec::with_capacity(rows);
        for &p in positions {
            for _ in 0..h {
                row_pos.push(p);
            }
        }
        o_rope_conjugate(rope, &row_pos, &mut attn, rows, d_qk)?;
    }
    ensure!(
        d_qk % OUT_CAST_GROUP == 0,
        InvalidArgument,
        "fused output cast needs d_qk divisible by {OUT_CAST_GROUP}, got {d_qk}"
    );
    let rows = s_q
        .checked_mul(h)
        .ok_or_else(|| Error::DimensionMismatch("s_q*h overflows".into()))?;
    let groups = d_qk / OUT_CAST_GROUP;
    let mut data = vec![0u8; attn.len()];
    let mut sf = vec![0u8; rows
        .checked_mul(groups)
        .ok_or_else(|| Error::DimensionMismatch("row*group count overflows".into()))?];
    for r in 0..rows {
        fp8cache::quantize_e4m3_e8m0_groups(
            &attn[r * d_qk..(r + 1) * d_qk],
            OUT_CAST_GROUP,
            &mut data[r * d_qk..(r + 1) * d_qk],
            &mut sf[r * groups..(r + 1) * groups],
        )?;
    }
    Ok((data, sf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::naive_sparse_prefill;
    use deepinfer_core::prop::{assert_close_slice, random_vec};
    use deepinfer_core::rng::Philox;
    use deepinfer_core::rope::RopeStyle;

    #[test]
    fn fused_matches_manual_composition() {
        let mut rng = Philox::new(0xF05, 1);
        let (s_q, h, d, s_kv, topk) = (3usize, 2, 32, 20, 9);
        let q = random_vec(&mut rng, s_q * h * d, 1.0);
        let kv = random_vec(&mut rng, s_kv * d, 1.0);
        let indices: Vec<i32> = (0..s_q * topk)
            .map(|_| if rng.next_f32() < 0.2 { -1 } else { rng.uniform_usize(0, s_kv) as i32 })
            .collect();
        let sm = 1.0 / (d as f32).sqrt();
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 64).unwrap();
        let positions: Vec<u32> = (0..s_q as u32).collect();
        let (data, sf) = fused_attention_out_fp8(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, Some(&[0.25, -0.1]), None,
            Some((&rope, &positions)),
        )
        .unwrap();
        // Manual composition.
        let (mut attn, _, _) =
            flash_mla_sparse_prefill(&q, &kv, &indices, s_q, h, d, s_kv, topk, sm, Some(&[0.25, -0.1]), None)
                .unwrap();
        let mut row_pos = Vec::new();
        for &p in &positions {
            for _ in 0..h {
                row_pos.push(p);
            }
        }
        o_rope_conjugate(&rope, &row_pos, &mut attn, s_q * h, d).unwrap();
        let mut mdata = vec![0u8; s_q * h * d];
        let mut msf = vec![0u8; s_q * h * (d / 32)];
        for r in 0..s_q * h {
            fp8cache::quantize_e4m3_e8m0_groups(
                &attn[r * d..(r + 1) * d],
                OUT_CAST_GROUP,
                &mut mdata[r * d..(r + 1) * d],
                &mut msf[r * (d / 32)..(r + 1) * (d / 32)],
            )
            .unwrap();
        }
        assert_eq!(data, mdata, "byte-identical cast");
        assert_eq!(sf, msf);
        assert_eq!(data.len(), s_q * h * d);
        assert_eq!(sf.len(), s_q * h * (d / 32));
        // Determinism.
        let (data2, sf2) = fused_attention_out_fp8(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, Some(&[0.25, -0.1]), None,
            Some((&rope, &positions)),
        )
        .unwrap();
        assert_eq!(data, data2);
        assert_eq!(sf, sf2);
    }

    #[test]
    fn fused_matches_reference_pipeline() {
        // End-to-end: the fused output must dequantize back to (a
        // quantized version of) the reference attention + O-RoPE output.
        let mut rng = Philox::new(0xF06, 2);
        let (s_q, h, d, s_kv, topk) = (2usize, 2, 64, 15, 7);
        let q = random_vec(&mut rng, s_q * h * d, 1.0);
        let kv = random_vec(&mut rng, s_kv * d, 1.0);
        let indices: Vec<i32> = (0..s_q * topk)
            .map(|_| rng.uniform_usize(0, s_kv) as i32)
            .collect();
        let sm = 1.0 / (d as f32).sqrt();
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 64).unwrap();
        let positions: Vec<u32> = (0..s_q as u32).collect();
        let (data, sf) = fused_attention_out_fp8(
            &q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, None, Some((&rope, &positions)),
        )
        .unwrap();
        let (mut ref_out, _, _) = naive_sparse_prefill(&q, &kv, &indices, s_q, h, d, s_kv, topk, sm, None, None);
        let mut row_pos = Vec::new();
        for &p in &positions {
            for _ in 0..h {
                row_pos.push(p);
            }
        }
        o_rope_conjugate(&rope, &row_pos, &mut ref_out, s_q * h, d).unwrap();
        for r in 0..s_q * h {
            for j in 0..d {
                let scale =
                    deepinfer_core::format::e8m0_to_f32(sf[r * (d / 32) + j / 32]);
                let deq = deepinfer_core::format::e4m3_to_f32(data[r * d + j]) * scale;
                assert_close_slice(&[deq], &[ref_out[r * d + j]], 0.05, "fused dequant");
            }
        }
    }

    #[test]
    fn fused_validates() {
        let q = vec![0.1f32; 2 * 2 * 32];
        let kv = vec![0.1f32; 5 * 32];
        // d_qk not divisible by 32.
        assert!(fused_attention_out_fp8(&q[..2 * 2 * 30], &kv[..5 * 30], &[0, 1], 2, 2, 30, 5, 1, 1.0, None, None, None).is_err());
        // Wrong o_rope position count.
        let rope = Rope::new(8, RopeStyle::Interleaved, 10_000.0, None, 64).unwrap();
        assert!(fused_attention_out_fp8(&q, &kv, &[0, 1], 2, 2, 32, 5, 1, 1.0, None, None, Some((&rope, &[0u32]))).is_err());
        assert!(fused_attention_out_fp8(&q, &kv, &[0, 1], 2, 2, 32, 5, 1, 1.0, None, None, Some((&rope, &[0u32, 1]))).is_ok());
    }
}
