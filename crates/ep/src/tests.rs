//! Integration tests for `deepinfer-ep` (dispatch / recv / combine /
//! low-latency dispatch, FP8 payloads, alignment, determinism).

use crate::{combine, dispatch, low_latency_dispatch, recv, DispatchPayload, EpConfig};
use deepinfer_core::prop::{assert_close_slice, prop};
use deepinfer_core::rng::Philox;

fn cfg(alignment: usize, hidden: usize) -> EpConfig {
    EpConfig {
        world_size: 2,
        num_experts: 4,
        num_max_tokens_per_rank: 1024,
        hidden,
        expert_alignment: alignment,
    }
}

fn row(x: &[f32], tokens: usize, t: usize) -> &[f32] {
    let h = x.len() / tokens;
    &x[t * h..(t + 1) * h]
}

/// Expert function used for round-trips: y = 2 * x + 1 (elementwise).
fn expert_fn(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| 2.0 * v + 1.0).collect()
}

#[test]
fn owner_rank_is_block_contiguous() {
    let c = cfg(1, 8);
    assert_eq!(c.num_local_experts(), 2);
    // Experts 0,1 -> rank 0; experts 2,3 -> rank 1.
    assert_eq!(c.owner_rank(0), 0);
    assert_eq!(c.owner_rank(1), 0);
    assert_eq!(c.owner_rank(2), 1);
    assert_eq!(c.owner_rank(3), 1);
    assert!(c.validate().is_ok());
}

#[test]
fn validate_rejects_bad_configs() {
    let mut c = cfg(1, 8);
    c.world_size = 0;
    assert!(c.validate().is_err());
    let mut c = cfg(1, 8);
    c.num_experts = 3; // does not divide by world_size 2
    assert!(c.validate().is_err());
    let mut c = cfg(1, 8);
    c.hidden = 0;
    assert!(c.validate().is_err());
}

#[test]
fn dispatch_validates_inputs() {
    let c = cfg(1, 8);
    let x = vec![0.1f32; 3 * 8];
    let idx = vec![0u32, 1, 2, 3, 0, 1];
    let w = vec![0.25f32; 6];
    // Rank out of range.
    assert!(dispatch(&c, 2, &x, &idx, &w, 3, 2, false).is_err());
    // Length mismatches.
    assert!(dispatch(&c, 0, &x, &idx[..5], &w, 3, 2, false).is_err());
    assert!(dispatch(&c, 0, &x, &idx, &w[..5], 3, 2, false).is_err());
    assert!(dispatch(&c, 0, &x[..23], &idx, &w, 3, 2, false).is_err());
    // k == 0.
    assert!(dispatch(&c, 0, &x, &[], &[], 3, 0, false).is_err());
    // Expert out of range.
    let bad_idx = vec![0u32, 4, 2, 3, 0, 1];
    assert!(dispatch(&c, 0, &x, &bad_idx, &w, 3, 2, false).is_err());
    // Non-finite weights.
    let mut bad_w = w.clone();
    bad_w[0] = f32::NAN;
    assert!(dispatch(&c, 0, &x, &idx, &bad_w, 3, 2, false).is_err());
    // Non-finite x.
    let mut bad_x = x.clone();
    bad_x[3] = f32::INFINITY;
    assert!(dispatch(&c, 0, &bad_x, &idx, &w, 3, 2, false).is_err());
    // Per-receiver capacity exceeded.
    let tight = EpConfig {
        world_size: 2,
        num_experts: 4,
        num_max_tokens_per_rank: 1,
        hidden: 8,
        expert_alignment: 1,
    };
    assert!(dispatch(&tight, 0, &x, &idx, &w, 3, 2, false).is_err());
}

#[test]
fn dispatch_handle_bookkeeping() {
    let c = cfg(1, 8);
    let x: Vec<f32> = (0..3 * 8).map(|i| i as f32 * 0.01).collect();
    let idx = vec![0u32, 2, 1, 3, 0, 0];
    let w = vec![0.5f32, 0.5, 0.25, 0.25, 1.0, 0.0];
    let (payload, handle) = dispatch(&c, 0, &x, &idx, &w, 3, 2, false).unwrap();
    match payload {
        DispatchPayload::F32(v) => assert_eq!(v, x),
        _ => panic!("expected F32 payload"),
    }
    assert_eq!(handle.src_rank, 0);
    assert_eq!(handle.token_slots.len(), 6);
    assert_eq!(handle.topk_idx, idx);
    assert_eq!(handle.topk_weights, w);
    // Counts per expert.
    let mut expect = vec![0usize; 4];
    for &e in &idx {
        expect[e as usize] += 1;
    }
    assert_eq!(handle.recv_counts_per_expert, expect);
    assert!(handle.hook.is_fulfilled());
    handle.hook.wait(); // Instant in this simulation.
}

#[test]
fn round_trip_f32_single_sender() {
    // alignment = 1 => no padding; combine round-trip is exact.
    let c = cfg(1, 8);
    let (tokens, k) = (5usize, 2);
    let mut rng = Philox::new(0xE9, 1);
    let x: Vec<f32> = (0..tokens * 8).map(|_| rng.uniform_f32(-1.0, 1.0)).collect();
    let idx: Vec<u32> = (0..tokens * k).map(|i| (i % 4) as u32).collect();
    let w: Vec<f32> = (0..tokens * k).map(|_| rng.uniform_f32(0.1, 0.9)).collect();

    let (payload, handle) = dispatch(&c, 0, &x, &idx, &w, tokens, k, false).unwrap();
    let pending = [(payload, handle)];
    let mut y_full = vec![vec![0.0f32; 8]; tokens];
    for rank in 0..c.world_size {
        let res = recv(&c, rank, &pending).unwrap();
        // Every received row must belong to a local expert of `rank`.
        for &e in res.recv_topk_idx.iter() {
            assert_eq!(c.owner_rank(e), rank);
        }
        // Expert rows are grouped contiguously and in ascending expert id.
        let mut last = 0u32;
        for &e in res.recv_topk_idx.iter() {
            assert!(e >= last);
            last = e;
        }
        // Compute expert outputs for the TRUE rows and combine.
        let true_rows = res.handle.token_slots.len();
        assert_eq!(true_rows, res.recv_x.len() / 8);
        let mut expert_out = Vec::with_capacity(true_rows * 8);
        for r in 0..true_rows {
            expert_out.extend_from_slice(&expert_fn(&res.recv_x[r * 8..(r + 1) * 8]));
        }
        let y = combine(&c, &res.handle, &expert_out).unwrap();
        for (t, yrow) in y.chunks(8).enumerate() {
            for d in 0..8 {
                y_full[t][d] += yrow[d];
            }
        }
    }
    // Expected: y[t] = sum_k w_k * expert_fn(x[t]).
    for t in 0..tokens {
        let mut expect = vec![0.0f32; 8];
        for ki in 0..k {
            let i = t * k + ki;
            let f = expert_fn(row(&x, tokens, t));
            for d in 0..8 {
                expect[d] += w[i] * f[d];
            }
        }
        assert_close_slice(&y_full[t], &expect, 1e-5, "round trip token {t}");
    }
}

#[test]
fn recv_orders_by_expert_sender_token_k() {
    // Two senders; rank 1's tokens have lower indices but the pending list
    // puts sender 1 second. Order must still be (expert, sender, token, k).
    let c = cfg(1, 8);
    let x0 = vec![1.0f32; 2 * 8];
    let x1 = vec![2.0f32; 2 * 8];
    let (p0, h0) = dispatch(&c, 0, &x0, &[0, 0, 2, 2], &[0.1; 4], 2, 2, false).unwrap();
    let (p1, h1) = dispatch(&c, 1, &x1, &[0, 0, 2, 2], &[0.2; 4], 2, 2, false).unwrap();
    // Sender values distinguish rows: x0 rows are 1.0, x1 rows are 2.0.
    let res = recv(&c, 0, &[(p1, h1), (p0, h0)]).unwrap();
    // Rank 0 owns experts 0,1: each sender contributes its two expert-0
    // pairs -> 4 rows, ordered (expert, sender, token, k).
    let got: Vec<f32> = (0..res.recv_x.len() / 8)
        .map(|r| res.recv_x[r * 8])
        .collect();
    assert_eq!(got, vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(res.recv_src_idx, vec![0, 0, 0, 0]);
    assert_eq!(res.handle.token_slots.len(), 4);
}

#[test]
fn alignment_padding_properties() {
    let c = cfg(4, 8); // pad each expert segment to a multiple of 4
    let (tokens, k) = (3usize, 2);
    let x = vec![0.5f32; tokens * 8];
    // Expert 0 gets 3 rows, expert 2 gets 3 rows (routed to rank 1).
    let idx = vec![0u32, 0, 0, 2, 2, 2];
    let w = vec![0.2f32; 6];
    let (payload, handle) = dispatch(&c, 0, &x, &idx, &w, tokens, k, false).unwrap();
    let res = recv(&c, 0, &[(payload, handle)]).unwrap();
    // Rank 0 owns experts 0 and 1: only expert 0 rows arrive (3 true).
    assert_eq!(res.handle.token_slots.len(), 3);
    // Padded to a multiple of 4: 4 rows total for the expert-0 segment.
    assert_eq!(res.recv_x.len() / 8, 4);
    // Padding row is zero; padding carries the segment expert id, weight 0.
    assert!(res.recv_x[3 * 8..4 * 8].iter().all(|&v| v == 0.0));
    assert_eq!(res.recv_topk_idx, vec![0u32, 0, 0, 0]);
    assert_eq!(res.recv_topk_weights[3], 0.0);
    // True counts preserved in the handle.
    assert_eq!(res.handle.recv_counts_per_expert[0], 3);
    // True rows survive verbatim.
    for r in 0..3 {
        assert!(res.recv_x[r * 8..(r + 1) * 8].iter().all(|&v| v == 0.5));
    }
}

#[test]
fn fp8_payload_round_trip_bounds() {
    let c = cfg(1, 128); // hidden must be a multiple of the 128 quant group
    let mut rng = Philox::new(0xE8, 7);
    let (tokens, k) = (4usize, 2);
    let x: Vec<f32> = (0..tokens * 128).map(|_| rng.uniform_f32(-2.0, 2.0)).collect();
    let idx = vec![0u32, 1, 2, 3, 0, 1, 2, 3];
    let w = vec![0.25f32; 8];
    let (p32, h32) = dispatch(&c, 0, &x, &idx, &w, tokens, k, false).unwrap();
    let (p8, h8) = dispatch(&c, 0, &x, &idx, &w, tokens, k, true).unwrap();
    assert!(matches!(p8, DispatchPayload::Fp8 { .. }));
    let r32 = recv(&c, 0, &[(p32, h32)]).unwrap();
    let r8 = recv(&c, 0, &[(p8, h8)]).unwrap();
    assert_eq!(r32.recv_x.len(), r8.recv_x.len());
    let n = r32.recv_x.len();
    let mut worst = 0.0f32;
    for i in 0..n {
        let a = r32.recv_x[i].abs().max(1e-3);
        worst = worst.max((r32.recv_x[i] - r8.recv_x[i]).abs() / a);
    }
    assert!(worst <= 0.2, "fp8 relative error {worst} exceeds 0.2");
    // Non-multiple-of-128 hidden must fail FP8 dispatch.
    let bad = cfg(1, 100);
    assert!(dispatch(&bad, 0, &vec![0.1; 2 * 100], &[0, 1], &[0.5, 0.5], 2, 1, true).is_err());
}

#[test]
fn low_latency_layout_and_drops() {
    let c = cfg(1, 8);
    // Rank 0 owns experts 0,1; capacity 2 per local expert.
    let (tokens, k, capacity) = (3usize, 2, 2);
    let x = vec![0.25f32; tokens * 8];
    // Expert 0 receives 3 pairs (capacity 2 -> 1 drop), expert 1: 2, expert 2: 1.
    let idx = vec![0u32, 0, 0, 0, 0, 2];
    let w = vec![0.5f32; 6];
    let res = low_latency_dispatch(&c, 0, &x, &idx, &w, tokens, k, capacity, false).unwrap();
    assert_eq!(res.masked_m, vec![2usize, 0]);
    assert_eq!(res.recv_x.len(), 2 * 2 * 8); // 2 local experts * capacity 2
    // Rows beyond counts are zero.
    assert!(res.recv_x[2 * 8..].iter().all(|&v| v == 0.0));
    // Dropped pairs reported deterministically ((token, k) order):
    // expert 0 receives 5 pairs, capacity 2 keeps (0,0),(0,1).
    assert_eq!(res.dropped, vec![(1u32, 0u32), (1u32, 1u32), (2u32, 0u32)]);
    // capacity 0 invalid.
    assert!(low_latency_dispatch(&c, 0, &x, &idx, &w, tokens, k, 0, false).is_err());
}

#[test]
fn combine_validates_inputs() {
    let c = cfg(1, 8);
    let x = vec![0.1f32; 2 * 8];
    let (payload, handle) = dispatch(&c, 0, &x, &[0, 1, 0, 1], &[0.5; 4], 2, 2, false).unwrap();
    let res = recv(&c, 0, &[(payload, handle)]).unwrap();
    let good = vec![1.0f32; res.handle.token_slots.len() * 8];
    assert!(combine(&c, &res.handle, &good).is_ok());
    // Row-count mismatch.
    assert!(combine(&c, &res.handle, &good[..good.len() - 8]).is_err());
    // Non-finite expert output.
    let mut bad = good.clone();
    bad[3] = f32::NAN;
    assert!(combine(&c, &res.handle, &bad).is_err());
    // Not a multiple of hidden.
    assert!(combine(&c, &res.handle, &good[..good.len() - 1]).is_err());
}

#[test]
fn determinism_property() {
    prop("ep round trip deterministic", 0xE7, 40, |_i, rng| {
        let c = cfg(1, 8);
        let tokens = rng.uniform_usize(1, 6);
        let k = rng.uniform_usize(1, 3);
        let x: Vec<f32> = (0..tokens * 8).map(|_| rng.uniform_f32(-1.0, 1.0)).collect();
        let idx: Vec<u32> = (0..tokens * k).map(|_| rng.uniform_usize(0, 4) as u32).collect();
        let w: Vec<f32> = (0..tokens * k).map(|_| rng.uniform_f32(0.05, 0.95)).collect();
        type Outs = Vec<(Vec<f32>, Vec<u32>, Vec<u32>)>;
        let run = || -> deepinfer_core::Result<Outs> {
            let mut outs: Outs = Vec::new();
            for rank in 0..c.world_size {
                let (p, h) = dispatch(&c, rank, &x, &idx, &w, tokens, k, false)?;
                let res = recv(&c, rank, &[(p, h)])?;
                outs.push((
                    res.recv_x.clone(),
                    res.recv_src_idx.clone(),
                    res.recv_topk_idx.clone(),
                ));
            }
            Ok(outs)
        };
        let a = run()?;
        let b = run()?;
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.0, rb.0, "recv_x must be bit-identical");
            assert_eq!(ra.1, rb.1);
            assert_eq!(ra.2, rb.2);
        }
        Ok(())
    });
}

#[test]
fn round_trip_property() {
    prop("ep dispatch/recv/combine == local moe", 0xE6, 40, |_i, rng| {
        let c = cfg(1, 8);
        let tokens = rng.uniform_usize(1, 5);
        let k = rng.uniform_usize(1, 3);
        let x: Vec<f32> = (0..tokens * 8).map(|_| rng.uniform_f32(-1.0, 1.0)).collect();
        let idx: Vec<u32> = (0..tokens * k).map(|_| rng.uniform_usize(0, 4) as u32).collect();
        let w: Vec<f32> = (0..tokens * k).map(|_| rng.uniform_f32(0.05, 0.95)).collect();

        let (payload, handle) = dispatch(&c, 0, &x, &idx, &w, tokens, k, false)?;
        let res = recv(&c, 0, &[(payload, handle)])?;
        let true_rows = res.handle.token_slots.len();
        let mut expert_out = Vec::with_capacity(true_rows * 8);
        for r in 0..true_rows {
            expert_out.extend_from_slice(&expert_fn(&res.recv_x[r * 8..(r + 1) * 8]));
        }
        let y = combine(&c, &res.handle, &expert_out)?;
        // Expected over rank-0-owned pairs only (single-sender recv per rank
        // keeps token indices disjoint in the mirror handle).
        let mut expect = vec![0.0f32; tokens * 8];
        for t in 0..tokens {
            for ki in 0..k {
                let i = t * k + ki;
                if c.owner_rank(idx[i]) != 0 {
                    continue;
                }
                let f = expert_fn(row(&x, tokens, t));
                for d in 0..8 {
                    expect[t * 8 + d] += w[i] * f[d];
                }
            }
        }
        let mut covered = vec![false; tokens];
        for &(t, _) in &res.handle.token_slots {
            covered[t as usize] = true;
        }
        for t in 0..tokens {
            if covered[t] {
                assert_close_slice(&y[t * 8..(t + 1) * 8], &expect[t * 8..(t + 1) * 8], 1e-5, "prop");
            }
        }
        Ok(())
    });
}
