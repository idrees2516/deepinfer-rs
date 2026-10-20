//! Dispatch payloads: the token-row buffers "sent" between ranks
//! (raw FP32 rows or per-token-group E4M3 + UE8M0 FP8 rows).

use crate::config::EpConfig;
use deepinfer_core::{ensure, Error, Result};
use deepinfer_fp8::{dequantize_per_token_group, quantize_per_token_group};

/// FP8 activation quantization granularity along the hidden axis:
/// per-token-group E4M3 values with one UE8M0 scale per `128` elements —
/// the DeepGEMM/DeepEP "act scaling 1x128" recipe used by DeepEP's FP8
/// dispatch. FP8 payloads therefore require `hidden % FP8_QUANT_GROUP == 0`.
pub const FP8_QUANT_GROUP: usize = 128;

/// The token-row buffer produced by [`dispatch`](crate::dispatch).
///
/// The payload carries the **whole sender batch** in row-major
/// `[tokens, hidden]` layout; which rows a receiver keeps is dictated by
/// the paired [`DispatchHandle`](crate::DispatchHandle)'s `topk_idx` (each
/// slot names the expert, hence the owner rank, of one token row).
///
/// * [`DispatchPayload::F32`] — raw `f32` rows (DeepEP's BF16/FP32 path is
///   represented exactly by FP32 on this CPU reference).
/// * [`DispatchPayload::Fp8`] — E4M3 `data` (`[tokens, hidden]` bytes) plus
///   UE8M0 `sf` scales (`[tokens, hidden / group]` bytes) produced by
///   `deepinfer_fp8::quantize_per_token_group`.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchPayload {
    /// Raw FP32 token rows, row-major `[tokens, hidden]`.
    F32(Vec<f32>),
    /// Per-token-group quantized token rows: E4M3 data with UE8M0 scales.
    Fp8 {
        /// E4M3-quantized values, row-major `[tokens, hidden]` bytes.
        data: Vec<u8>,
        /// UE8M0 scales, row-major `[tokens, hidden / group]` bytes.
        sf: Vec<u8>,
        /// Quantization group size along hidden (128).
        group: usize,
    },
}

/// Decodes a whole payload back into `f32` rows (`[rows, hidden]`,
/// row-major). F32 payloads are passed through; FP8 payloads are
/// dequantized with `deepinfer_fp8::dequantize_per_token_group`.
///
/// # Errors
///
/// * `DimensionMismatch` when the byte/element counts do not describe a
///   whole number of `hidden`-wide rows (F32), or when the scale-factor
///   count does not match `rows * (hidden / group)` (Fp8);
/// * `InvalidArgument` when `group == 0` or `hidden % group != 0`;
/// * `CapacityExceeded` on row-count overflow.
pub(crate) fn decode_payload(payload: &DispatchPayload, cfg: &EpConfig) -> Result<Vec<f32>> {
    let hidden = cfg.hidden;
    match payload {
        DispatchPayload::F32(v) => {
            ensure!(
                v.len() % hidden == 0,
                DimensionMismatch,
                "F32 payload has {} elements, not a multiple of hidden = {hidden}",
                v.len()
            );
            Ok(v.clone())
        }
        DispatchPayload::Fp8 { data, sf, group } => {
            ensure!(
                *group >= 1,
                InvalidArgument,
                "Fp8 payload group must be >= 1, got {group}"
            );
            ensure!(
                hidden % group == 0,
                InvalidArgument,
                "Fp8 payload: hidden ({hidden}) must be a multiple of group ({group})"
            );
            ensure!(
                data.len() % hidden == 0,
                DimensionMismatch,
                "Fp8 payload data has {} bytes, not a multiple of hidden = {hidden}",
                data.len()
            );
            let rows = data.len() / hidden;
            let sf_needed = rows
                .checked_mul(hidden / group)
                .ok_or_else(|| Error::CapacityExceeded("Fp8 payload scale count overflows".into()))?;
            ensure!(
                sf.len() == sf_needed,
                DimensionMismatch,
                "Fp8 payload sf has {} scales, expected rows * (hidden/group) = {sf_needed}",
                sf.len()
            );
            dequantize_per_token_group(data, sf, hidden, *group)
        }
    }
}

/// Builds the FP8 payload for a sender batch (whole-batch quantization).
pub(crate) fn quantize_payload(x: &[f32], cfg: &EpConfig) -> Result<DispatchPayload> {
    let (data, sf) = quantize_per_token_group(x, cfg.hidden, FP8_QUANT_GROUP)?;
    Ok(DispatchPayload::Fp8 {
        data,
        sf,
        group: FP8_QUANT_GROUP,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(hidden: usize) -> EpConfig {
        EpConfig {
            world_size: 1,
            num_experts: 2,
            num_max_tokens_per_rank: 8,
            hidden,
            expert_alignment: 1,
        }
    }

    #[test]
    fn f32_payload_passthrough() {
        let c = cfg(4);
        let p = DispatchPayload::F32(vec![1.0; 8]);
        assert_eq!(decode_payload(&p, &c).unwrap(), vec![1.0; 8]);
        let bad = DispatchPayload::F32(vec![1.0; 7]);
        assert!(decode_payload(&bad, &c).is_err());
    }

    #[test]
    fn fp8_payload_roundtrip_and_validation() {
        let c = cfg(FP8_QUANT_GROUP);
        let x: Vec<f32> = (0..2 * FP8_QUANT_GROUP)
            .map(|i| ((i as f32 * 0.37) % 2.0) - 1.0)
            .collect();
        let p = quantize_payload(&x, &c).unwrap();
        match &p {
            DispatchPayload::Fp8 { data, sf, group } => {
                assert_eq!(data.len(), x.len());
                assert_eq!(sf.len(), 2 * FP8_QUANT_GROUP / FP8_QUANT_GROUP);
                assert_eq!(*group, FP8_QUANT_GROUP);
            }
            _ => panic!("expected Fp8 payload"),
        }
        let decoded = decode_payload(&p, &c).unwrap();
        assert_eq!(decoded.len(), x.len());
        let err2: f64 = decoded
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64))
            .sum();
        let norm2: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(err2.sqrt() / norm2.sqrt() <= 0.15, "rel err too large");

        // Malformed payloads are rejected, never panic.
        let mut bad = p.clone();
        if let DispatchPayload::Fp8 { sf, .. } = &mut bad {
            sf.truncate(0);
        }
        assert!(decode_payload(&bad, &c).is_err());
        let bad = DispatchPayload::Fp8 {
            data: vec![0u8; 3],
            sf: vec![],
            group: FP8_QUANT_GROUP,
        };
        assert!(decode_payload(&bad, &c).is_err());
        let bad = DispatchPayload::Fp8 {
            data: vec![0u8; FP8_QUANT_GROUP],
            sf: vec![0u8; 1],
            group: 0,
        };
        assert!(decode_payload(&bad, &c).is_err());
        // hidden not divisible by the group.
        let small = cfg(100);
        assert!(decode_payload(&p, &small).is_err());
        // hidden smaller than the group cannot be quantized at all.
        assert!(quantize_payload(&[0.0; 100], &small).is_err());
    }
}
