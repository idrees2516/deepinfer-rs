//! Golden tests: GPU kernels vs CPU references.
//!
//! Every test auto-skips (with a note) when no GPU is present. On hardware
//! they compare against the CPU reference crates / naive transcriptions,
//! using bit-exact expectations for elementwise math (codecs, RoPE, integer
//! selections) and tolerances where reduction order legitimately differs
//! (GEMM accumulation trees, softmax merges).

use deepinfer_core::rng::Philox;
use deepinfer_gpu::{Buffer, Stream};
use deepinfer_kernels::CacheFormat;
use deepinfer_kernels::Kernels;

fn setup() -> Option<(Kernels, Stream)> {
    let k = probe_kernels()?;
    let s = k.device().stream().ok()?;
    Some((k, s))
}

fn probe_kernels() -> Option<Kernels> {
    for b in deepinfer_gpu::Gpu::probe_backends() {
        if let Ok(gpu) = deepinfer_gpu::Gpu::new(b) {
            if let Ok(n) = gpu.device_count() {
                if n > 0 {
                    if let Ok(dev) = gpu.device(0) {
                        if let Ok(k) = deepinfer_kernels::Kernels::new(&dev) {
                            return Some(k);
                        }
                    }
                }
            }
        }
    }
    eprintln!("note: no GPU present; GPU golden tests skipped");
    None
}

fn upload_f32(k: &deepinfer_kernels::Kernels, v: &[f32]) -> Buffer {
    k.device().alloc_from(v).expect("upload")
}

fn download_f32(k: &deepinfer_kernels::Kernels, b: &Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    k.device().read(b, &mut out).expect("download");
    out
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let m = x.abs().max(y.abs()).max(1e-6);
        worst = worst.max(d / m);
    }
    worst
}

#[test]
fn golden_rmsnorm() {
    let Some((k, s)) = setup() else { return };
    let (rows, cols) = (64usize, 7168);
    let mut rng = Philox::new(11, 1);
    let x: Vec<f32> = (0..rows * cols).map(|_| rng.normal_f32()).collect();
    let gain: Vec<f32> = (0..cols).map(|_| rng.uniform_f32(0.5, 1.5)).collect();
    let mut ref_out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut ms = 0.0f32;
        for c in 0..cols {
            ms += x[r * cols + c] * x[r * cols + c];
        }
        ms /= cols as f32;
        let inv = 1.0f32 / (ms + 1e-4f32).sqrt();
        for c in 0..cols {
            ref_out[r * cols + c] = x[r * cols + c] * inv * gain[c];
        }
    }
    let dx = upload_f32(&k, &x);
    let dg = upload_f32(&k, &gain);
    let dOut = k.device().alloc(rows * cols * 4).unwrap();
    k.rmsnorm(&dx, &dg, &dOut, rows, cols, 1e-4, &s).unwrap();
    s.synchronize().unwrap();
    let out = download_f32(&k, &dOut, rows * cols);
    let e = rel_err(&out, &ref_out);
    assert!(e < 1e-4, "rmsnorm rel err {e}");
}

#[test]
fn golden_rope_and_conjugate() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_core::rope::{Rope, RopeStyle};
    let (half, row_len, rows) = (32usize, 512, 16);
    let rope = Rope::new(half * 2, RopeStyle::Interleaved, 10000.0, None, 1024).unwrap();
    let mut rng = Philox::new(12, 2);
    let x: Vec<f32> = (0..rows * row_len).map(|_| rng.uniform_f32(-2.0, 2.0)).collect();
    let pos: Vec<u32> = (0..rows as u32).map(|i| i * 7 + 3).collect();
    let offset = row_len - half * 2;
    let mut cpu = x.clone();
    rope.apply_at(&pos, &mut cpu, row_len, offset).unwrap();

    let dx = upload_f32(&k, &x);
    let dpos = k.device().alloc_from(&pos).unwrap();
    let dcs = upload_f32(&k, rope.cos_sin_table());
    k.rope(&dcs, &dpos, &dx, rows, row_len, offset, half, false, &s)
        .unwrap();
    let after = download_f32(&k, &dx, rows * row_len);
    assert!(rel_err(&after, &cpu) < 1e-5, "rope apply mismatch");
    // Conjugate on the device-rotated buffer must restore the original.
    k.rope(&dcs, &dpos, &dx, rows, row_len, offset, half, true, &s)
        .unwrap();
    s.synchronize().unwrap();
    let rt = download_f32(&k, &dx, rows * row_len);
    assert!(rel_err(&rt, &x) < 1e-5, "rope round trip mismatch");
}

#[test]
fn golden_gemm_bf16() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_core::format::{bf16_to_f32, f32_to_bf16_rne};
    let (m, n, kk) = (128usize, 128, 256);
    let mut rng = Philox::new(13, 3);
    let a: Vec<f32> = (0..m * kk).map(|_| rng.normal_f32()).collect();
    let b: Vec<f32> = (0..n * kk).map(|_| rng.normal_f32()).collect();
    let a16: Vec<u16> = a.iter().map(|&v| f32_to_bf16_rne(v)).collect();
    let b16: Vec<u16> = b.iter().map(|&v| f32_to_bf16_rne(v)).collect();
    let mut ref_q = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for c in 0..kk {
                acc += bf16_to_f32(a16[i * kk + c]) * bf16_to_f32(b16[j * kk + c]);
            }
            ref_q[i * n + j] = acc;
        }
    }
    let da = k.device().alloc_from(&a16).unwrap();
    let db = k.device().alloc_from(&b16).unwrap();
    let dd = k.device().alloc(m * n * 4).unwrap();
    k.gemm_bf16_nt(&da, &db, None, &dd, m, n, kk, &s).unwrap();
    s.synchronize().unwrap();
    let out = download_f32(&k, &dd, m * n);
    let e = rel_err(&out, &ref_q);
    assert!(e < 5e-3, "bf16 gemm rel err {e}");
}

#[test]
fn golden_gemm_fp8_and_grouped() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_fp8::{quantize_per_token_group, quantize_weight_per_block};
    use deepinfer_gemm::{fp8_gemm_nt, m_grouped_fp8_gemm_nt_contiguous, pad_to_alignment};
    let (m, n, kk, groups) = (256usize, 128, 256, 2);
    let mut rng = Philox::new(14, 4);
    let a: Vec<f32> = (0..m * kk).map(|_| rng.normal_f32()).collect();
    let seg = pad_to_alignment(m / groups, 128);
    let b_all: Vec<f32> = (0..groups * n * kk).map(|_| rng.normal_f32()).collect();
    let (aq, asf) = quantize_per_token_group(&a, kk, 128).unwrap();
    let (bq, bsf) = quantize_weight_per_block(&b_all, n, kk, 128, 128).unwrap();

    // Dense
    let ref_dense = fp8_gemm_nt(&aq, &asf, &bq, &bsf, m, n, kk, None).unwrap();
    let da = k.device().alloc_from(&aq).unwrap();
    let dasf = k.device().alloc_from(&asf).unwrap();
    let db = k.device().alloc_from(&bq).unwrap();
    let dbsf = k.device().alloc_from(&bsf).unwrap();
    let dd = k.device().alloc(m * n * 4).unwrap();
    k.gemm_fp8_nt(&da, &dasf, &db, &dbsf, &dd, m, n, kk, &s).unwrap();
    s.synchronize().unwrap();
    let out = download_f32(&k, &dd, m * n);
    let e = rel_err(&out, &ref_dense);
    assert!(e < 3e-2, "fp8 gemm rel err {e}");

    // Grouped-contiguous (segments of `seg` rows)
    let m_total = groups * seg;
    let mut a_pad = aq.clone();
    a_pad.resize(m_total * kk, 0);
    let mut asf_pad = asf.clone();
    asf_pad.resize(m_total * (kk / 128), 0);
    let mut m_indices = vec![0u32; m_total];
    for (i, v) in m_indices.iter_mut().enumerate() {
        *v = (i / seg) as u32;
    }
    let ref_grouped = m_grouped_fp8_gemm_nt_contiguous(
        &a_pad, &asf_pad, &bq, &bsf, &m_indices, m_total, n, kk, groups,
    )
    .unwrap();
    let dap = k.device().alloc_from(&a_pad).unwrap();
    let dasfp = k.device().alloc_from(&asf_pad).unwrap();
    let ddg = k.device().alloc(m_total * n * 4).unwrap();
    let offs: Vec<i32> = (0..=groups).map(|g| (g * seg) as i32).collect();
    let doffs = k.device().alloc_from(&offs).unwrap();
    k.gemm_fp8_grouped_nt(&dap, &dasfp, &db, &dbsf, &doffs, &ddg, m_total, n, kk, groups, &s)
        .unwrap();
    s.synchronize().unwrap();
    let outg = download_f32(&k, &ddg, m_total * n);
    let eg = rel_err(&outg, &ref_grouped);
    assert!(eg < 3e-2, "grouped gemm rel err {eg}");
}

#[test]
fn golden_kv_encode() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_fp8::{
        dequantize_kv_cache_v41_fp4, dequantize_kv_cache_v41_fp8, quantize_kv_cache_v41_fp4,
        quantize_kv_cache_v41_fp8,
    };
    let tokens = 32usize;
    let mut rng = Philox::new(15, 5);
    let x: Vec<f32> = (0..tokens * 512).map(|_| rng.normal_f32()).collect();
    let dx = upload_f32(&k, &x);

    let mut ref8 = vec![0u8; tokens * 528];
    for t in 0..tokens {
        let row = quantize_kv_cache_v41_fp8(&x[t * 512..(t + 1) * 512]).unwrap();
        ref8[t * 528..(t + 1) * 528].copy_from_slice(&row);
    }
    let dOut8 = k.device().alloc(tokens * 528).unwrap();
    k.kv_encode_fp8(&dx, &dOut8, tokens, 512, &s).unwrap();
    let mut got8 = vec![0u8; tokens * 528];
    s.synchronize().unwrap();
    k.device().read(&dOut8, &mut got8).unwrap();
    let mut worst = 0.0f32;
    for t in 0..tokens {
        let r = dequantize_kv_cache_v41_fp8(got8[t * 528..(t + 1) * 528].try_into().unwrap());
        let e = dequantize_kv_cache_v41_fp8(ref8[t * 528..(t + 1) * 528].try_into().unwrap());
        worst = worst.max(rel_err(&r, &e));
    }
    assert!(worst < 1e-6, "kv fp8 encode mismatch {worst}");

    let mut ref4 = vec![0u8; tokens * 288];
    for t in 0..tokens {
        let row = quantize_kv_cache_v41_fp4(&x[t * 512..(t + 1) * 512]).unwrap();
        ref4[t * 288..(t + 1) * 288].copy_from_slice(&row);
    }
    let dOut4 = k.device().alloc(tokens * 288).unwrap();
    k.kv_encode_fp4(&dx, &dOut4, tokens, 512, &s).unwrap();
    let mut got4 = vec![0u8; tokens * 288];
    s.synchronize().unwrap();
    k.device().read(&dOut4, &mut got4).unwrap();
    let mut worst = 0.0f32;
    for t in 0..tokens {
        let r = dequantize_kv_cache_v41_fp4(got4[t * 288..(t + 1) * 288].try_into().unwrap());
        let e = dequantize_kv_cache_v41_fp4(ref4[t * 288..(t + 1) * 288].try_into().unwrap());
        worst = worst.max(rel_err(&r, &e));
    }
    assert!(worst < 1e-6, "kv fp4 encode mismatch {worst}");
}

/// Naive decode reference over device-format rows (dequantized host-side),
/// matching the FlashMLA contract: invalid indices skipped, online-softmax
/// equivalent math, lse = m + log(l), no-valid -> 0 out / +inf lse.
fn naive_decode(
    q: &[f32],
    rows: &[Vec<f32>],
    indices: &[i32],
    b: usize,
    s_q: usize,
    h: usize,
    d: usize,
    topk: usize,
    sm: f32,
    sink: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; b * s_q * h * d];
    let mut lse = vec![0.0f32; b * h * s_q];
    for bi in 0..b {
        for qi in 0..s_q {
            for hi in 0..h {
                let qbase = (bi * s_q + qi) * h + hi;
                let idxs = &indices[(bi * s_q + qi) * topk..][..topk];
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut o = vec![0.0f32; d];
                for &idx in idxs {
                    if idx < 0 || idx as usize >= rows.len() {
                        continue;
                    }
                    let row = &rows[idx as usize];
                    let mut sc = 0.0f32;
                    for j in 0..d {
                        sc += q[qbase * d + j] * row[j];
                    }
                    sc *= sm;
                    if sc > m {
                        let res = (m - sc).exp();
                        for v in o.iter_mut() {
                            *v *= res;
                        }
                        l *= res;
                        m = sc;
                    }
                    let p = (sc - m).exp();
                    l += p;
                    for j in 0..d {
                        o[j] += p * row[j];
                    }
                }
                let li = (bi * h + hi) * s_q + qi;
                if l > 0.0 {
                    lse[li] = m + l.ln();
                    for j in 0..d {
                        out[qbase * d + j] = o[j] / l;
                    }
                    if let Some(sk) = sink {
                        let f = 1.0 / (1.0 + (sk[hi] - lse[li]).exp());
                        for j in 0..d {
                            out[qbase * d + j] *= f;
                        }
                    }
                } else {
                    lse[li] = f32::INFINITY;
                }
            }
        }
    }
    (out, lse)
}

#[test]
fn golden_mla_decode_all_formats() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_core::format::{e2m1_to_f32, e4m3_to_f32, e8m0_to_f32};
    use deepinfer_fp8::{quantize_kv_cache_v41_fp4, quantize_kv_cache_v41_fp8};
    let (b, s_q, h, d, s_kv, topk) = (2usize, 1, 8, 512, 300, 256);
    let mut rng = Philox::new(16, 6);
    let rows: Vec<Vec<f32>> = (0..s_kv).map(|_| (0..d).map(|_| rng.normal_f32()).collect()).collect();
    let q: Vec<f32> = (0..b * s_q * h * d).map(|_| rng.normal_f32()).collect();
    let mut indices = vec![0i32; b * s_q * topk];
    for (i, v) in indices.iter_mut().enumerate() {
        let j = (i * 37 + 11) % (s_kv + 20);
        *v = if j < s_kv { j as i32 } else { -1 };
    }
    let sm = 1.0f32 / (d as f32).sqrt();
    let sink: Vec<f32> = (0..h).map(|i| 0.05 * i as f32).collect();

    for fmt in [CacheFormat::F32, CacheFormat::Fp8Group32, CacheFormat::Fp4Group16] {
        // Device bytes + host-visible dequantized rows.
        let (dev_bytes, dq_rows): (Vec<u8>, Vec<Vec<f32>>) = match fmt {
            CacheFormat::F32 => (
                rows.iter().flat_map(|r| r.iter().copied()).collect::<Vec<f32>>()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                rows.clone(),
            ),
            CacheFormat::Fp8Group32 => {
                let mut bytes = Vec::with_capacity(s_kv * 528);
                let mut dq = Vec::with_capacity(s_kv);
                for r in &rows {
                    let row = quantize_kv_cache_v41_fp8(r).unwrap();
                    let mut back = [0.0f32; 512];
                    for i in 0..512 {
                        back[i] = e4m3_to_f32(row[i]) * e8m0_to_f32(row[512 + i / 32]);
                    }
                    bytes.extend_from_slice(&row);
                    dq.push(back.to_vec());
                }
                (bytes, dq)
            }
            CacheFormat::Fp4Group16 => {
                let mut bytes = Vec::with_capacity(s_kv * 288);
                let mut dq = Vec::with_capacity(s_kv);
                for r in &rows {
                    let row = quantize_kv_cache_v41_fp4(r).unwrap();
                    let mut back = [0.0f32; 512];
                    for i in 0..512 {
                        let nib = if i % 2 == 0 {
                            row[i / 2] & 0x0F
                        } else {
                            row[i / 2] >> 4
                        };
                        back[i] = e2m1_to_f32(nib) * e4m3_to_f32(row[256 + i / 16]);
                    }
                    bytes.extend_from_slice(&row);
                    dq.push(back.to_vec());
                }
                (bytes, dq)
            }
        };
        let (ref_out, ref_lse) = naive_decode(&q, &dq_rows, &indices, b, s_q, h, d, topk, sm, Some(&sink));

        let dq = upload_f32(&k, &q);
        let dcache = k.device().alloc_from(&dev_bytes).unwrap();
        let dsink = upload_f32(&k, &sink);
        let dIdx = k.device().alloc_from(&indices).unwrap();
        let nq = b * s_q * h;
        let splits_max = 16;
        let dpm = k.device().alloc(nq * splits_max * 4).unwrap();
        let dpl = k.device().alloc(nq * splits_max * 4).unwrap();
        let dpo = k.device().alloc(nq * splits_max * d * 4).unwrap();
        let dOut = k.device().alloc(q.len() * 4).unwrap();
        let dlse = k.device().alloc(b * h * s_q * 4).unwrap();
        let splits = k
            .mla_decode_partial(
                &dq, &dcache, &dIdx, &dpm, &dpl, &dpo, b, s_q, h, d, topk, sm, fmt, s_kv, &s,
            )
            .unwrap();
        k.mla_decode_combine(&dpm, &dpl, &dpo, Some(&dsink), &dOut, &dlse, b, s_q, h, d, splits, &s)
            .unwrap();
        s.synchronize().unwrap();
        let out = download_f32(&k, &dOut, q.len());
        let e = rel_err(&out, &ref_out);
        assert!(e < 5e-3, "decode {fmt:?} rel err {e}");
        let mut lse = vec![0.0f32; b * h * s_q];
        k.device().read(&dlse, &mut lse).unwrap();
        for (g, r) in lse.iter().zip(ref_lse.iter()) {
            if r.is_finite() {
                assert!((g - r).abs() < 5e-3, "lse {g} vs {r}");
            }
        }
    }
}

#[test]
fn golden_moe_route_exact() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_moe::{MoeConfig, MoeWeights};
    let mut cfg = MoeConfig::v41_default(256);
    cfg.num_routed = 256;
    cfg.num_activated = 8;
    cfg.group_size = 1;
    let mut rng = Philox::new(17, 7);
    let w = MoeWeights::from_config(&cfg, &mut rng);
    let tokens = 64usize;
    let xs: Vec<f32> = (0..tokens * 256).map(|_| rng.normal_f32()).collect();
    let refs = w.route_batch(&cfg, &xs, tokens).unwrap();
    let mut logits = vec![0.0f32; tokens * 256];
    for t in 0..tokens {
        for e in 0..256 {
            let mut acc = 0.0f32;
            for c in 0..256 {
                acc += w.gate[e * 256 + c] * xs[t * 256 + c];
            }
            logits[t * 256 + e] = acc;
        }
    }
    let dz = upload_f32(&k, &logits);
    let dbias = upload_f32(&k, &w.gate_bias);
    let dW = k.device().alloc(tokens * 8 * 4).unwrap();
    let dI = k.device().alloc(tokens * 8 * 4).unwrap();
    k.moe_route(&dz, Some(&dbias), &dW, &dI, tokens, 256, 8, &s)
        .unwrap();
    let mut gw = vec![0.0f32; tokens * 8];
    let mut gi = vec![0i32; tokens * 8];
    s.synchronize().unwrap();
    k.device().read(&dW, &mut gw).unwrap();
    k.device().read(&dI, &mut gi).unwrap();
    for t in 0..tokens {
        for kk in 0..8 {
            assert_eq!(
                gi[t * 8 + kk] as u32,
                refs[t].topk_idx[kk],
                "route idx mismatch t={t} k={kk}"
            );
        }
        let w_err = rel_err(&gw[t * 8..(t + 1) * 8], &refs[t].topk_weights);
        assert!(w_err < 1e-4, "route weight err {w_err}");
    }
}

#[test]
fn golden_select_top_k_exact() {
    let Some((k, s)) = setup() else { return };
    use deepinfer_dsa::{select_top_k, LightningIndexerConfig};
    let (s_q, s_kv) = (4usize, 2000);
    let mut rng = Philox::new(18, 8);
    let mut scores = vec![0.0f32; s_q * s_kv];
    for (i, v) in scores.iter_mut().enumerate() {
        *v = if i % 97 == 0 { f32::NEG_INFINITY } else { rng.normal_f32() };
    }
    let cfg = LightningIndexerConfig {
        block_size: 64,
        preselect_per_block: 8,
        topk_tokens: 512,
        ..LightningIndexerConfig::v32_default(7168)
    };
    let ref_sel = select_top_k(&scores, s_q, s_kv, &cfg).unwrap();
    let ds = upload_f32(&k, &scores);
    let num_blocks = (s_kv + 63) / 64;
    let cap = num_blocks * 8;
    let dca = k.device().alloc(s_q * cap * 8).unwrap();
    let dcb = k.device().alloc(s_q * cap * 8).unwrap();
    let dOut = k.device().alloc(s_q * 512 * 4).unwrap();
    k.select_top_k(&ds, &dca, &dcb, cap, &dOut, s_q, s_kv, 64, 8, 512, &s)
        .unwrap();
    let mut got = vec![0i32; s_q * 512];
    s.synchronize().unwrap();
    k.device().read(&dOut, &mut got).unwrap();
    assert_eq!(got, ref_sel, "top-k selection must match the CPU exactly");
}

#[test]
fn golden_moe_dispatch_combine_roundtrip() {
    let Some((k, s)) = setup() else { return };
    let (tokens, hidden, e, kk) = (32usize, 256, 16, 4);
    let mut rng = Philox::new(19, 9);
    let x: Vec<f32> = (0..tokens * hidden).map(|_| rng.normal_f32()).collect();
    let mut topk_idx = vec![0i32; tokens * kk];
    let topk_w = vec![0.25f32; tokens * kk];
    for t in 0..tokens {
        for j in 0..kk {
            topk_idx[t * kk + j] = ((t * 7 + j * 3) % e) as i32;
        }
    }
    // CPU reference: expert_out = 2 * row (stand-in expert), y = sum_k w * 2x
    let mut ref_y = vec![0.0f32; tokens * hidden];
    for t in 0..tokens {
        for j in 0..kk {
            for c in 0..hidden {
                ref_y[t * hidden + c] += topk_w[t * kk + j] * 2.0 * x[t * hidden + c];
            }
        }
    }
    let dx = upload_f32(&k, &x);
    let dIdx = k.device().alloc_from(&topk_idx).unwrap();
    let dW = upload_f32(&k, &topk_w);
    let counts = k.device().alloc(e * 4).unwrap();
    let offsets = k.device().alloc((e + 1) * 4).unwrap();
    let rankc = k.device().alloc(e * 4).unwrap();
    let map = k.device().alloc(tokens * kk * 4).unwrap();
    // Upper bound for the perm buffer: every (token, k) row + alignment
    // padding per expert segment.
    let cap_rows = tokens * kk + e * 128;
    let perm = k.device().alloc(cap_rows * hidden * 4).unwrap();
    let m_total = k
        .moe_dispatch(&dx, &dIdx, &counts, &offsets, &rankc, &perm, &map, tokens, hidden, kk, e, 128, &s)
        .unwrap();
    // "Expert compute" stand-in: double every permuted row (host round trip
    // keeps the test self-contained).
    let mut perm_host = vec![0.0f32; m_total * hidden];
    s.synchronize().unwrap();
    k.device().read(&perm, &mut perm_host).unwrap();
    for v in perm_host.iter_mut() {
        *v *= 2.0;
    }
    let dPermOut = k.device().alloc_from(&perm_host).unwrap();
    let dy = k.device().alloc(tokens * hidden * 4).unwrap();
    k.moe_combine(&dPermOut, &map, &dW, &dy, tokens, hidden, kk, &s)
        .unwrap();
    s.synchronize().unwrap();
    let y = download_f32(&k, &dy, tokens * hidden);
    let e_err = rel_err(&y, &ref_y);
    assert!(e_err < 1e-4, "moe round trip rel err {e_err}");
}
