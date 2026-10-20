//! # deepinfer-fp8
//!
//! Quantization recipes for **DeepInfer-rs**, mirroring DeepSeek's
//! DeepGEMM / tile-kernels FP8 casting utilities and the FlashMLA V4.1
//! KV-cache row formats.
//!
//! * **Activation recipe** — per-token-group E4M3 quantization with UE8M0
//!   (power-of-two) scales, granularity `1 x 128` along the K axis
//!   (DeepGEMM README: "act scaling 1x128"). See [`quantize_per_token_group`].
//! * **Weight recipe** — per-`128 x 128`-block E4M3 quantization with UE8M0
//!   scales (DeepGEMM README: "weight scaling 128x128"). See
//!   [`quantize_weight_per_block`].
//! * **SM100 packed-sf layout** — [`pack_ue8m0_x4`] packs four UE8M0 bytes
//!   into one `u32` exactly as DeepGEMM's SM100 kernels expect.
//! * **FlashMLA V4.1 KV-cache rows** — the exact 528-byte FP8
//!   (512 e4m3 values + 16 e8m0 per-32 scales) and 288-byte FP4
//!   (256 bytes of packed e2m1, even nibble low, + 32 e4m3 per-16 scales)
//!   token layouts from the FlashMLA README.
//!
//! Upstream references: `deepseek-ai/DeepGEMM` (README, scaling layouts,
//! `quant` utilities), `deepseek-ai/FlashMLA` (README, V4.1 KV formats),
//! `deepseek-ai/tile-kernels` (`quant.per_token_cast`).
//!
//! ## Fidelity notes
//!
//! * Scales are `2^ceil(log2(amax / 448))` clamped to the E8M0 range
//!   `[2^-127, 2^127]`; an all-zero group gets `2^-127`. This is the
//!   "round_sf" ceil policy used by DeepGEMM so the scaled amax stays
//!   inside the E4M3 representable range (max finite 448).
//! * No-panic policy for non-finite inputs (documented deviation): `NaN`
//!   entries are mapped to `+0` so quantized payloads never contain the
//!   E4M3 NaN encoding, and `±Inf` entries saturate to `±448` after
//!   scaling (a group containing `±Inf` takes the largest finite scale
//!   path). Every function returns [`deepinfer_core::Result`] instead of
//!   panicking.
//! * The FP4 KV scale is `e4m3_rne(amax / 6)` clamped to
//!   `[2^-9, 448]` (FlashMLA's FP4-adjacent recipe; `6` is the largest
//!   E2M1 magnitude, `2^-9` the smallest positive E4M3).

#![forbid(unsafe_code)]

pub mod recipes;

pub use recipes::*;
