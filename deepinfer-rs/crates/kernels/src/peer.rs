//! NVLink / Infinity Fabric **peer collectives for expert parallelism**.
//!
//! This is the single-process multi-GPU counterpart of the CPU
//! [`deepinfer-ep`](deepinfer_ep) crate's dispatch/combine, built directly
//! on the peer-access primitives of [`deepinfer_gpu`] — the layer the
//! NVL72/GB300 NVLink5 domain (130 TB/s rack bisection) exists for:
//!
//! 1. **Planning on the host** ([`build_send_plan`]): routing results are
//!    tiny (`tokens × K` i32), so each rank reads its `topk_idx` back and
//!    the host — seeing all ranks, being one process — computes the exact
//!    DeepEP-documented receive order `(expert, sender rank, token index,
//!    k)` with block-contiguous expert ownership
//!    (`owner(e) = e * world / num_experts`). Pure, deterministic, and
//!    unit-tested without any GPU.
//! 2. **Packing on the sender** ([`Kernels::gather_rows_indexed`]): each
//!    rank gathers its quantized rows into one packed send buffer
//!    (destination-major, then token, then k) with a device-resident index
//!    list — one local kernel per rank.
//! 3. **One peer copy per (destination, sender) pair**
//!    ([`PeerMesh::all_to_all_dispatch_fp8`]): contiguous
//!    `memcpy_peer_async` transfers on per-rank EP streams, ordered by an
//!    event mesh (`stream_d` waits for `event_s` after rank `s`'s gather),
//!    and closed by a full mesh barrier so send buffers can be reused.
//! 4. **Combine in reverse**
//!    ([`PeerMesh::all_to_all_combine_f32`]): the owner rank gathers its
//!    expert-output rows per home rank, one peer copy per pair brings them
//!    home, and the standard [`Kernels::moe_combine`] applies the routing
//!    weights (fixed k-ascending accumulation — deterministic).
//!
//! The v1 trade-offs (documented, matching the reference-first posture of
//! this workspace): the receive side keeps the raw
//! `(sender, token, k)` row order with per-row expert metadata for the
//! caller's aligned scatter, and the plans involve one small `topk_idx`
//! read-back per rank per step. Upgrading to NVSHMEM-style symmetric
//! buffers with sender-side RDMA directly into the aligned layout is the
//! follow-up the docs/GPU.md upgrade table tracks.

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::{Buffer, Device, Event, Stream};

use crate::Kernels;

/// One (token, k) routing pair destined for another rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendSlot {
    /// Home token index.
    pub token: u32,
    /// Routing slot within the token.
    pub k: u32,
    /// Routed expert id.
    pub expert: u32,
}

/// A rank's send plan: which (token, k) pairs go to which destination
/// rank, in the deterministic `(destination, token asc, k asc)` order the
/// packed send buffer is laid out in.
#[derive(Debug, Clone, Default)]
pub struct SendPlan {
    /// Per destination rank: the slots destined there (token-asc, k-asc).
    pub per_dst: Vec<Vec<SendSlot>>,
    /// Cumulative row offsets into the packed send buffer (`world + 1`).
    pub offsets: Vec<usize>,
    /// Total rows sent by this rank.
    pub total: usize,
}

/// Owner rank of `expert` under block-contiguous ownership (the
/// `deepinfer-ep` contract: `e * world / num_experts`).
pub fn expert_owner(expert: usize, num_experts: usize, world: usize) -> usize {
    if world <= 1 {
        return 0;
    }
    (expert * world) / num_experts
}

/// Builds the send plan of one rank from its routing result.
///
/// `topk_idx`: `[tokens, k]` (entries `< 0` are padding and are skipped,
/// matching the dispatch kernels). Rows of the packed send buffer are
/// ordered `(destination asc, token asc, k asc)` — a deterministic total
/// order, so identical inputs give bit-identical plans.
///
/// # Errors
/// [`Error::InvalidArgument`] on shape/ownership-domain violations.
pub fn build_send_plan(
    topk_idx: &[i32],
    tokens: usize,
    k: usize,
    world: usize,
    num_experts: usize,
) -> Result<SendPlan> {
    ensure!(world >= 1, InvalidArgument, "world must be >= 1");
    ensure!(num_experts >= 1, InvalidArgument, "num_experts must be >= 1");
    ensure!(k >= 1, InvalidArgument, "k must be >= 1");
    let n = tokens
        .checked_mul(k)
        .ok_or_else(|| Error::CapacityExceeded("tokens*k overflows".into()))?;
    ensure!(
        topk_idx.len() == n,
        DimensionMismatch,
        "topk_idx has {} entries, expected tokens*k = {n}",
        topk_idx.len()
    );
    let mut plan = SendPlan {
        per_dst: vec![Vec::new(); world],
        offsets: vec![0usize; world + 1],
        total: 0,
    };
    for t in 0..tokens {
        for kk in 0..k {
            let e = topk_idx[t * k + kk];
            if e < 0 {
                continue; // padding
            }
            let e = e as usize;
            ensure!(
                e < num_experts,
                OutOfRange,
                "expert id {e} outside [0, {num_experts})"
            );
            let dst = expert_owner(e, num_experts, world);
            plan.per_dst[dst].push(SendSlot {
                token: t as u32,
                k: kk as u32,
                expert: e as u32,
            });
        }
    }
    for d in 0..world {
        plan.offsets[d] = plan.total;
        plan.total += plan.per_dst[d].len();
    }
    plan.offsets[world] = plan.total;
    Ok(plan)
}

/// An all-pairs peer-access mesh of devices plus the per-rank EP streams
/// and barrier events the collectives schedule on.
///
/// All devices must come from the same backend (an NVL72 domain or an
/// Infinity Fabric island). Peer access is enabled for every ordered pair;
/// pairs that cannot address each other fail construction with a clear
/// error (the caller can then fall back to routing through the host).
pub struct PeerMesh {
    devices: Vec<Device>,
    streams: Vec<Stream>,
    events: Vec<Event>,
}

impl std::fmt::Debug for PeerMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerMesh")
            .field("world", &self.devices.len())
            .field(
                "devices",
                &self
                    .devices
                    .iter()
                    .map(|d| d.props().name.clone())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PeerMesh {
    /// Builds the mesh: enables all-pairs peer access, creates one EP
    /// stream and one barrier event per rank.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on an empty / mixed-backend device list;
    /// [`Error::Internal`] when a pair cannot access each other over the
    /// fabric or a stream/event cannot be created.
    pub fn new(devices: &[Device]) -> Result<PeerMesh> {
        ensure!(!devices.is_empty(), InvalidArgument, "PeerMesh needs at least one device");
        let backend = devices[0].backend();
        for (i, d) in devices.iter().enumerate() {
            ensure!(
                d.backend() == backend,
                InvalidArgument,
                "device {i} has backend {:?} but the mesh is {:?}",
                d.backend(),
                backend
            );
        }
        for (i, a) in devices.iter().enumerate() {
            for (j, b) in devices.iter().enumerate() {
                if i == j {
                    continue;
                }
                a.enable_peer_access(b).map_err(|e| {
                    Error::internal(format!(
                        "peer access {i} -> {j} unavailable ({}): {e}",
                        b.props().name
                    ))
                })?;
            }
        }
        let streams = devices
            .iter()
            .map(|d| d.stream())
            .collect::<Result<Vec<_>>>()?;
        let events = devices
            .iter()
            .map(|d| d.event())
            .collect::<Result<Vec<_>>>()?;
        Ok(PeerMesh {
            devices: devices.to_vec(),
            streams,
            events,
        })
    }

    /// Mesh size (ranks).
    pub fn world(&self) -> usize {
        self.devices.len()
    }

    /// The device of rank `r`.
    pub fn device(&self, r: usize) -> &Device {
        &self.devices[r]
    }

    /// The EP stream of rank `r`.
    pub fn stream(&self, r: usize) -> &Stream {
        &self.streams[r]
    }

    /// Gathers `rows` rows of `row_bytes` from `src` into `dst` by the
    /// device-resident index list `idx` (`[rows]` u32) on rank `r`'s EP
    /// stream (the packing step of dispatch and of combine).
    pub fn gather_indexed(
        &self,
        r: usize,
        k: &Kernels,
        src: &Buffer,
        dst: &Buffer,
        idx: &Buffer,
        rows: usize,
        row_bytes: usize,
    ) -> Result<()> {
        k.gather_indexed(src, dst, idx, rows, row_bytes, &self.streams[r])
    }

    /// **FP8 all-to-all dispatch.** Per rank `r`: `send_xq`
    /// (`[plan.total, hidden]` u8) and `send_xsf`
    /// (`[plan.total, hidden / 128]` u8) are the packed send buffers
    /// (built with [`PeerMesh::gather_indexed`] from the host plans), and
    /// `plans[r]` is rank r's [`SendPlan`].
    ///
    /// Returns per rank the receive buffers `(recv_xq, recv_xsf)` with
    /// rows in the deterministic `(sender, token, k)` order — the receive
    /// order the CPU `deepinfer-ep` crate documents — sized to the exact
    /// received row count.
    ///
    /// Synchronization: every destination stream first waits for all
    /// senders' packing work (via the event mesh), then the per-pair peer
    /// copies run, then a full mesh barrier retires the send buffers.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on plan/buffer inconsistencies;
    /// [`Error::Internal`] on driver rejections.
    pub fn all_to_all_dispatch_fp8(
        &self,
        send_xq: &[Buffer],
        send_xsf: &[Buffer],
        plans: &[SendPlan],
        hidden: usize,
    ) -> Result<Vec<(Buffer, Buffer)>> {
        let world = self.world();
        ensure!(hidden % 128 == 0, InvalidArgument, "fp8 dispatch needs hidden % 128 == 0");
        ensure!(
            send_xq.len() == world && send_xsf.len() == world && plans.len() == world,
            InvalidArgument,
            "dispatch inputs must cover all {world} ranks"
        );
        let sf_row = hidden / 128;
        // Receive sizes per rank (host-known: single process).
        let mut recv_rows = vec![0usize; world];
        for plan in plans {
            for (d, slots) in plan.per_dst.iter().enumerate() {
                recv_rows[d] += slots.len();
            }
        }
        // Phase 0: every rank's packing is complete (record events).
        for (r, ev) in self.events.iter().enumerate() {
            ev.record(&self.streams[r])?;
        }
        // Phase 1: per-pair contiguous peer copies on the destination's
        // stream, after it waits for the sender's packing event.
        let mut recv_owned: Vec<(Buffer, Buffer)> = Vec::with_capacity(world);
        let mut recv_views: Vec<(Buffer, Buffer)> = Vec::with_capacity(world);
        for d in 0..world {
            let dev = &self.devices[d];
            let xq = dev.alloc(recv_rows[d].max(1) * hidden)?;
            let xsf = dev.alloc(recv_rows[d].max(1) * sf_row)?;
            recv_views.push((
                xq.slice(0, recv_rows[d].max(1) * hidden)?,
                xsf.slice(0, recv_rows[d].max(1) * sf_row)?,
            ));
            recv_owned.push((xq, xsf));
        }
        for d in 0..world {
            for s in 0..world {
                if s == d || plans[s].per_dst[d].is_empty() {
                    continue;
                }
                // Destination waits for the sender's packing to be visible.
                self.streams[d].wait_event(&self.devices[d], &self.events[s])?;
                let rows = plans[s].per_dst[d].len();
                let off = plans[s].offsets[d];
                let dst_dev = &self.devices[d];
                let src_dev = &self.devices[s];
                // SAFETY-of-layout: offsets/rows come from the same plans
                // the packing used; buffers were allocated to match.
                let sxq = send_xq[s].slice(off * hidden, rows * hidden)?;
                let sxsf = send_xsf[s].slice(off * sf_row, rows * sf_row)?;
                dst_dev.memcpy_peer_async(&recv_views[d].0, src_dev, &sxq, &self.streams[d])?;
                dst_dev.memcpy_peer_async(&recv_views[d].1, src_dev, &sxsf, &self.streams[d])?;
            }
        }
        // Phase 2: mesh barrier — no send buffer is reused until every
        // copy that reads it has been enqueued (and ordered) on its
        // destination stream.
        for r in 0..world {
            self.events[r].record(&self.streams[r])?;
        }
        for d in 0..world {
            for s in 0..world {
                if d != s {
                    self.streams[d].wait_event(&self.devices[d], &self.events[s])?;
                }
            }
        }
        Ok(recv_owned)
    }

    /// **Combine in reverse**: per rank `r`, `send_out`
    /// (`[rows_r, hidden]` f32) holds the expert outputs gathered for the
    /// home ranks (built with [`PeerMesh::gather_indexed`]); `per_home`
    /// gives the row counts per destination (home) rank. Returns per home
    /// rank one receive buffer whose row layout the caller defines via the
    /// same counts (host-composable maps).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on count/buffer inconsistencies;
    /// [`Error::Internal`] on driver rejections.
    pub fn all_to_all_combine_f32(
        &self,
        send_out: &[Buffer],
        per_home: &[usize],
        hidden: usize,
    ) -> Result<Vec<Buffer>> {
        let world = self.world();
        ensure!(
            send_out.len() == world && per_home.len() == world,
            InvalidArgument,
            "combine inputs must cover all {world} ranks"
        );
        for (r, &n) in per_home.iter().enumerate() {
            let need = n * hidden;
            ensure!(
                send_out[r].len() >= need,
                InvalidArgument,
                "rank {r} send buffer holds {} bytes but {n} rows x {hidden} = {need}",
                send_out[r].len()
            );
        }
        for (r, ev) in self.events.iter().enumerate() {
            ev.record(&self.streams[r])?;
        }
        let mut recv_owned = Vec::with_capacity(world);
        let mut recv_views = Vec::with_capacity(world);
        for d in 0..world {
            let rows_d = per_home[d];
            let buf = self.devices[d].alloc(rows_d.max(1) * hidden * 4)?;
            recv_views.push(buf.slice(0, rows_d.max(1) * hidden * 4)?);
            recv_owned.push(buf);
        }
        // Offsets of each sender's block inside each destination buffer:
        // sender-major concatenation (deterministic).
        let mut send_base = vec![0usize; world];
        let mut acc = 0usize;
        for s in 0..world {
            send_base[s] = acc;
            acc += per_home[s];
        }
        for d in 0..world {
            for s in 0..world {
                if per_home[s] == 0 {
                    continue;
                }
                if d != s {
                    self.streams[d].wait_event(&self.devices[d], &self.events[s])?;
                }
                let rows = per_home[s];
                let dst = recv_views[d].slice(send_base[s] * hidden * 4, rows * hidden * 4)?;
                let src = send_out[s].slice(0, rows * hidden * 4)?;
                self.devices[d].memcpy_peer_async(&dst, &self.devices[s], &src, &self.streams[d])?;
            }
        }
        for r in 0..world {
            self.events[r].record(&self.streams[r])?;
        }
        for d in 0..world {
            for s in 0..world {
                if d != s {
                    self.streams[d].wait_event(&self.devices[d], &self.events[s])?;
                }
            }
        }
        Ok(recv_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_owner_matches_deepinfer_ep_block_ownership() {
        // owner(e) = e * world / E, block-contiguous.
        assert_eq!(expert_owner(0, 256, 8), 0);
        assert_eq!(expert_owner(31, 256, 8), 0);
        assert_eq!(expert_owner(32, 256, 8), 1);
        assert_eq!(expert_owner(255, 256, 8), 7);
        // world=1: everything local.
        assert_eq!(expert_owner(41, 256, 1), 0);
        // E not divisible by world: still monotone, in range.
        let mut prev = 0;
        for e in 0..10usize {
            let o = expert_owner(e, 10, 3);
            assert!(o < 3);
            assert!(o >= prev);
            prev = o;
        }
    }

    #[test]
    fn send_plan_orders_by_destination_then_token_then_k() {
        // tokens=2, k=2, world=2, E=4 => owners: e0,e1 -> 0; e2,e3 -> 1.
        let topk = [2, 0, 3, 1, -1, 2]; // t0: (2,0) t1: (3,1); t2? no, tokens=3? -> tokens=3,k=2
        let tokens = 3;
        let plan = build_send_plan(&topk, tokens, 2, 2, 4).unwrap();
        assert_eq!(plan.per_dst[0].len(), 2); // (t0,k1,e0) (t1,k1,e1)
        assert_eq!(plan.per_dst[1].len(), 3); // (t0,k0,e2) (t1,k0,e3) (t2,k1,e2)
        assert_eq!(plan.total, 5);
        assert_eq!(plan.offsets, [0, 2, 5]);
        // Slot order within each destination: token asc, then k asc.
        assert_eq!(
            plan.per_dst[1]
                .iter()
                .map(|s| (s.token, s.k))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 1)]
        );
        assert_eq!(
            plan.per_dst[0]
                .iter()
                .map(|s| (s.token, s.k))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)]
        );
        // Padded entries skipped.
        assert!(plan.per_dst.iter().all(|v| v.iter().all(|s| s.token != 2 || s.k != 0)));
    }

    #[test]
    fn send_plan_validates_and_edges() {
        assert!(build_send_plan(&[0], 1, 2, 1, 4).is_err()); // len mismatch
        assert!(build_send_plan(&[4], 1, 1, 1, 4).is_err()); // expert out of range
        assert!(build_send_plan(&[-1; 4], 2, 2, 2, 4).unwrap().total == 0); // all padding
        let empty = build_send_plan(&[], 0, 2, 2, 4).unwrap();
        assert_eq!(empty.total, 0);
        assert_eq!(empty.offsets, [0, 0, 0]);
        // world=1 keeps everything local.
        let one = build_send_plan(&[0, 3, 1, 2], 2, 2, 1, 4).unwrap();
        assert_eq!(one.per_dst[0].len(), 4);
        assert_eq!(one.offsets, [0, 4]);
    }

    #[test]
    fn send_plan_is_deterministic() {
        let topk: Vec<i32> = (0..64).map(|i| (i * 37 + 11) % 8).collect();
        let a = build_send_plan(&topk, 32, 2, 4, 8).unwrap();
        let b = build_send_plan(&topk, 32, 2, 4, 8).unwrap();
        assert_eq!(a.offsets, b.offsets);
        assert_eq!(a.per_dst, b.per_dst);
    }

    #[test]
    fn peer_mesh_rejects_empty() {
        // No GPU needed for the validation error path.
        assert!(PeerMesh::new(&[]).is_err());
    }
}
