//! GPU-gated golden tests: cuTile kernels vs the CPU reference crates.
//!
//! Every test skips (with a notice) when no CUDA device is present, so the
//! GPU-less plane stays green in CI; on a GPU box (GB200 / GB300 / any
//! sm_80+) run with:
//!
//! ```bash
//! source scripts/gpu-env.sh
//! cargo test -p deepinfer-cutile --features cuda --test golden
//! ```

#![cfg(feature = "cuda")]
// SAFETY: unsafe blocks call Tile-IR-waived kernel launchers after the
// same shape validation the ops layer performs.
#![allow(unsafe_code)]

use cutile::prelude::*;
use cutile::tensor::{ToHostVec, TryPartition};
use deepinfer_cutile::{device, ops};

/// Skips (returns true) when no GPU is available.
fn no_gpu() -> bool {
    if device::device_count() == 0 {
        eprintln!("skip: no CUDA device present");
        true
    } else {
        false
    }
}

fn ctx() -> ops::GpuCtx {
    ops::GpuCtx::new(0).expect("open device 0")
}

/// RMSNorm vs `deepinfer_core::norm::rmsnorm`.
#[test]
fn rms_norm_matches_cpu() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    let (rows, n) = (8usize, 512usize);
    let eps = 1e-4f32;

    let mut x_host: Vec<f32> = (0..rows * n)
        .map(|i| ((i as f32 * 0.37) % 1.7) - 0.85)
        .collect();
    let w_host: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();

    let mut reference = x_host.clone();
    deepinfer_core::norm::rmsnorm(&mut reference, &w_host, eps);

    let x = ops::upload_bf16(&ctx, &x_host, &[rows, n]).unwrap();
    let w = ops::upload_bf16(&ctx, &w_host, &[n]).unwrap();
    let out = ops::rms_norm(&ctx, &x, &w, rows, n, eps).unwrap();
    let got: Vec<cutile::half::bf16> = out.to_host_vec().sync_on(ctx.stream()).unwrap();
    let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();

    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() <= 2e-2 + 2e-2 * r.abs(),
            "rms_norm mismatch at {i}: gpu={g} cpu={r}"
        );
    }
}

/// Dense BF16 GEMM vs an f64 reference.
#[test]
fn bf16_gemm_matches_reference() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    let (m, n, k) = (32usize, 32usize, 64usize);

    let a: Vec<f32> = (0..m * k).map(|i| ((i * 37 % 23) as f32 / 23.0) - 0.5).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i * 41 % 19) as f32 / 19.0) - 0.5).collect();

    let a_t = ops::upload_bf16(&ctx, &a, &[m, k]).unwrap();
    let b_t = ops::upload_bf16(&ctx, &b, &[n, k]).unwrap();
    let d = ops::bf16_gemm_nt(&ctx, &a_t, &b_t, m, n, k).unwrap();
    let got: Vec<cutile::half::bf16> = d.to_host_vec().sync_on(ctx.stream()).unwrap();

    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f64;
            for ki in 0..k {
                acc += a[mi * k + ki] as f64 * b[ni * k + ki] as f64;
            }
            let g = got[mi * n + ni].to_f32() as f64;
            assert!(
                (g - acc).abs() <= 1e-2 + 1e-3 * acc.abs(),
                "gemm mismatch at ({mi},{ni}): gpu={g} ref={acc}"
            );
        }
    }
}

/// Quantize → dequantize round trip stays within FP8 E4M3 relative error.
#[test]
fn quant_fp8_block128_roundtrip() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    let (rows, k) = (4usize, 256usize);

    let x: Vec<f32> = (0..rows * k)
        .map(|i| {
            let v = ((i * 73 % 101) as f32 / 101.0) - 0.5;
            v * 8.0
        })
        .collect();
    let x_t = ops::upload_bf16(&ctx, &x, &[rows, k]).unwrap();
    let (q, _s) = ops::quant_fp8_block128(&ctx, &x_t, rows, k).unwrap();

    // Dequantize on the host with the exact CPU recipe and compare.
    let q_arc = std::sync::Arc::new(q);
    let q_bytes = q_arc
        .reinterpret::<u8>(&[rows, k])
        .unwrap();
    let q_host: Vec<u8> = q_bytes.to_host_vec().sync_on(ctx.stream()).unwrap();
    for (i, v) in x.iter().enumerate() {
        let dq = deepinfer_core::format::e4m3_to_f32(q_host[i]);
        assert!(
            (dq - v).abs() <= 0.35 * v.abs().max(1.0),
            "fp8 roundtrip too far at {i}: {dq} vs {v} (byte {:#x})",
            q_host[i]
        );
    }
}

/// Greedy argmax (ties → lowest index).
#[test]
fn argmax_ties_lowest() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    let len = 4096usize;
    let mut logits = vec![0.5f32; len];
    logits[100] = 2.0;
    logits[4000] = 2.0; // tie — lowest index must win

    let l = ops::upload_bf16(&ctx, &logits, &[len]).unwrap();
    let (m, idx) = ops::argmax_bf16(&ctx, &l, len).unwrap();
    assert_eq!(idx, 100, "argmax tie must resolve to the lowest index");
    assert!((m - 2.0).abs() < 1e-2, "argmax value {m}");
}

/// Two-stage DSA top-K selection vs `deepinfer_dsa::select::select_top_k`.
#[test]
fn dsa_select_matches_cpu() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    use deepinfer_dsa::config::LightningIndexerConfig;

    let s_q = 3usize;
    let s_kv = 512usize;
    let topk = 64usize;
    let cfg = LightningIndexerConfig {
        hidden_size: 64,
        num_heads: 2,
        head_dim: 32,
        block_size: 64,
        topk_tokens: topk,
        preselect_per_block: 8,
        quant_block: 2048,
        conv_kernels: vec![3],
        extrapolation_window: None,
    };

    // Deterministic pseudo-random scores, some exact ties.
    let scores: Vec<f32> = (0..s_q * s_kv)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2654435761);
            let v = (h >> 8) as f32 / 16777216.0;
            if h % 17 == 0 { (v * 10.0).round() / 10.0 } else { v }
        })
        .collect();

    let expect = deepinfer_dsa::select::select_top_k(&scores, s_q, s_kv, &cfg).unwrap();
    let k = expect.len() / s_q;

    // GPU: stage 1 + stage 2.
    use deepinfer_cutile::kernels;
    let nblocks = (s_kv + 63) / 64;
    let p = 8usize;
    let nslots = nblocks * p;
    let scores_t = ops::upload_f32(&ctx, &scores, &[s_q, s_kv]).unwrap();

    let surv_s = cutile::api::zeros::<f32>(&[s_q, nslots]).partition([1, p]);
    let surv_i = cutile::api::zeros::<i32>(&[s_q, nslots]).partition([1, p]);
    let (_a, _b, ..) = unsafe { kernels::dsa_block_preselect(surv_s, surv_i, scores_t.clone(), s_kv as i32) }
        .generics(vec![p.to_string()])
        .sync_on(ctx.stream())
        .unwrap();
    let surv_s = std::sync::Arc::new(_a.unpartition());
    let surv_i = std::sync::Arc::new(_b.unpartition());

    let out_host: Vec<i32> = vec![-1; s_q * k];
    let out_t = ops::upload_i32(&ctx, &out_host, &[s_q, k]).unwrap();
    let out_part = out_t.try_partition([1, k]).unwrap();
    let (out_part, ..) = unsafe { kernels::dsa_topk_select(out_part, surv_s, surv_i, nslots as i32, k as i32) }
        .generics(vec!["1024".to_string(), k.to_string()])
        .sync_on(ctx.stream())
        .unwrap();
    let got = ops::download_i32(&ctx, &std::sync::Arc::new(out_part.unpartition())).unwrap();

    assert_eq!(got.len(), expect.len());
    for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
        assert_eq!(g, e, "dsa select mismatch at {i}: gpu={g} cpu={e}");
    }
}

/// MLA decode (BF16 cache, gathered window) vs an f64 dense reference.
#[test]
fn mla_decode_bf16_matches_reference() {
    if no_gpu() {
        return;
    }
    let ctx = ctx();
    let b = 2usize;
    let ngroups = 1usize;
    let group = 8usize; // heads per kv group
    let topk = 96usize;
    let dn = 512usize;
    let dp = 64usize;
    let rows_total = 512usize;

    // Deterministic q / kv.
    let qn: Vec<f32> = (0..b * ngroups * group * dn)
        .map(|i| ((i as u32).wrapping_mul(1103515245) >> 16) as f32 / 65536.0 - 0.5)
        .collect();
    let qp: Vec<f32> = (0..b * ngroups * group * dp)
        .map(|i| ((i as u32).wrapping_mul(12345) >> 16) as f32 / 65536.0 - 0.5)
        .collect();
    let ckv: Vec<f32> = (0..rows_total * dn)
        .map(|i| ((i as u32).wrapping_mul(69069) >> 16) as f32 / 65536.0 - 0.5)
        .collect();
    let kpe: Vec<f32> = (0..rows_total * dp)
        .map(|i| ((i as u32).wrapping_mul(99991) >> 16) as f32 / 65536.0 - 0.5)
        .collect();

    // Gathered window: batch 0 keeps rows 0..topk, batch 1 keeps a
    // strided mix with two invalid slots.
    let mut indices: Vec<i32> = Vec::new();
    for t in 0..topk {
        indices.push(t as i32);
    }
    for t in 0..topk {
        indices.push(if t == 3 || t == 10 { -1 } else { (rows_total - 1 - t) as i32 });
    }

    let sm_scale = 1.0f32 / 576.0f32.sqrt();

    let qn_t = ops::upload_bf16(&ctx, &qn, &[b * ngroups, group, dn]).unwrap();
    let qp_t = ops::upload_bf16(&ctx, &qp, &[b * ngroups, group, dp]).unwrap();
    let ckv_t = ops::upload_bf16(&ctx, &ckv, &[rows_total, dn]).unwrap();
    let kpe_t = ops::upload_bf16(&ctx, &kpe, &[rows_total, dp]).unwrap();
    let idx_t = ops::upload_i32(&ctx, &indices, &[b, topk]).unwrap();

    let (out, lse) = ops::mla_decode_bf16(
        &ctx, &qn_t, &qp_t, &ckv_t, &kpe_t, &idx_t, None, b, ngroups, topk, sm_scale, 4,
    )
    .unwrap();

    let out_host: Vec<cutile::half::bf16> = out.to_host_vec().sync_on(ctx.stream()).unwrap();
    let lse_host: Vec<f32> = lse.to_host_vec().sync_on(ctx.stream()).unwrap();

    // f64 dense reference.
    for bi in 0..b {
        for h in 0..group {
            let row = bi * ngroups + 0;
            let mut scores: Vec<f64> = Vec::new();
            let mut valid: Vec<bool> = Vec::new();
            for t in 0..topk {
                let r = indices[bi * topk + t];
                if r < 0 {
                    scores.push(f64::NEG_INFINITY);
                    valid.push(false);
                    continue;
                }
                let r = r as usize;
                let mut dot = 0.0f64;
                for d in 0..dn {
                    dot += qn[(row * group + h) * dn + d] as f64 * ckv[r * dn + d] as f64;
                }
                for d in 0..dp {
                    dot += qp[(row * group + h) * dp + d] as f64 * kpe[r * dp + d] as f64;
                }
                scores.push(dot * sm_scale as f64);
                valid.push(true);
            }
            let m = scores.iter().filter(|s| s.is_finite()).cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = exps.iter().sum();
            for d in 0..dn {
                let mut acc = 0.0f64;
                for (t, e) in exps.iter().enumerate() {
                    if valid[t] {
                        acc += e * ckv[indices[bi * topk + t] as usize * dn + d] as f64;
                    }
                }
                let ref_v = acc / l;
                let gpu = out_host[(bi * ngroups * group + h) * (dn + dp) + d].to_f32() as f64;
                assert!(
                    (gpu - ref_v).abs() <= 5e-2 + 5e-2 * ref_v.abs(),
                    "decode out mismatch b={bi} h={h} d={d}: gpu={gpu} ref={ref_v}"
                );
            }
            let lse_ref = m + l.ln();
            let lse_gpu = lse_host[bi * ngroups * group + h] as f64;
            assert!(
                (lse_gpu - lse_ref).abs() <= 5e-2,
                "decode lse mismatch b={bi} h={h}: gpu={lse_gpu} ref={lse_ref}"
            );
        }
    }
}
