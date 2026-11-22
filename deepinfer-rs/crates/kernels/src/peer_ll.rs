//! NVSHMEM-style **symmetric-buffer expert parallelism** (the low-latency
//! DeepEP communication pattern on raw driver primitives).
//!
//! Every rank owns one [`SymmetricHeap`] with an identical layout; kernels
//! write peers' heaps directly over NVLink / Infinity Fabric (in-process
//! peer pointers, or IPC-imported pointers for multi-process serving — see
//! [`Session`]), and progress lives in system-scope atomics inside the
//! heaps. One dispatch is: reserve a channel slot on the owner rank
//! (`atom.sys`), write the FP8 payload + metadata, publish with
//! `red.release.sys`; the receiver groups rows into the masked per-expert
//! layout after a device-side step barrier — no host readback, no
//! `topk_idx` download, no per-step allocation.
//!
//! The step protocol is documented in `kernels/peer_ll.cu`; this module is
//! the host-side orchestration (launch order, event-ordered resets, the
//! pointer tables the kernels consume).

use std::path::{Path, PathBuf};

use deepinfer_core::{ensure, Error, Result};
use deepinfer_gpu::{Buffer, Device};

use crate::peer::PeerMesh;
use crate::Kernels;

/// One rank's symmetric heap: the fixed-layout buffer every peer writes
/// into (see `kernels/peer_ll.cu` for the layout contract).
pub struct SymmetricHeap {
    buf: Buffer,
    k_channels: usize,
    cap: usize,
    tokens_max: usize,
    hidden: usize,
}

impl std::fmt::Debug for SymmetricHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymmetricHeap")
            .field("k_channels", &self.k_channels)
            .field("cap", &self.cap)
            .field("tokens_max", &self.tokens_max)
            .field("hidden", &self.hidden)
            .field(
                "bytes",
                &heap_bytes(self.k_channels, self.cap, self.tokens_max, self.hidden),
            )
            .finish()
    }
}

/// Aligned dispatch row bytes: `[hidden e4m3 | hidden/128 ue8m0 | meta 16]`.
fn row_bytes(hidden: usize) -> usize {
    (hidden + (hidden >> 7) + 16 + 15) & !15
}

/// Counter cells at the heap front: `counts[K] produced[K] expected[K]`
/// plus `step_done, combine_done, valid_pairs, pad` (3K + 6 i32 words).
fn cells(k: usize) -> usize {
    3 * k + 6
}

/// Channel-region offset (16-byte aligned counter block end).
fn chan_off(k: usize) -> usize {
    (cells(k) * 4 + 15) & !15
}

/// Combine-region offset.
fn combine_off(k: usize, cap: usize, hidden: usize) -> usize {
    chan_off(k) + k * cap * row_bytes(hidden)
}

/// Total heap bytes for the given geometry.
fn heap_bytes(k: usize, cap: usize, tokens_max: usize, hidden: usize) -> usize {
    combine_off(k, cap, hidden) + tokens_max * k * hidden * 4
}

impl SymmetricHeap {
    /// Allocate a heap on `dev` for the given routing geometry.
    ///
    /// `k_channels` is the MoE top-k, `cap` the per-channel capacity (the
    /// worst-case number of rows one rank may receive per channel in one
    /// step — size it to the batch bound), `tokens_max` this rank's maximum
    /// concurrent tokens, `hidden` the model width (must be a multiple of
    /// 128 for the FP8 scale row).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on degenerate geometry;
    /// [`Error::CapacityExceeded`] when the allocation fails.
    pub fn alloc(
        dev: &Device,
        k_channels: usize,
        cap: usize,
        tokens_max: usize,
        hidden: usize,
    ) -> Result<SymmetricHeap> {
        ensure!(k_channels >= 1, InvalidArgument, "k_channels must be >= 1");
        ensure!(cap >= 1, InvalidArgument, "cap must be >= 1");
        ensure!(tokens_max >= 1, InvalidArgument, "tokens_max must be >= 1");
        ensure!(
            hidden % 128 == 0,
            InvalidArgument,
            "hidden must be a multiple of 128"
        );
        let bytes = heap_bytes(k_channels, cap, tokens_max, hidden);
        let buf = dev
            .alloc(bytes)
            .map_err(|e| Error::CapacityExceeded(format!("symmetric heap ({bytes} B): {e}")))?;
        Ok(SymmetricHeap {
            buf,
            k_channels,
            cap,
            tokens_max,
            hidden,
        })
    }

    /// Raw heap address (the value peers write through).
    pub fn ptr(&self) -> u64 {
        self.buf.ptr()
    }

    /// Heap size in bytes.
    pub fn bytes(&self) -> usize {
        heap_bytes(self.k_channels, self.cap, self.tokens_max, self.hidden)
    }

    /// Number of i32 counter cells (`3K + 6`).
    pub fn cells(&self) -> usize {
        cells(self.k_channels)
    }
}

/// One peer's published record: `(ipc handle, k, cap, tokens, hidden)`.
pub type PeerRecord = ([u8; 64], usize, usize, usize, usize);

/// File-rendezvous for **multi-process** symmetric heaps: each process
/// publishes its IPC handle + geometry under one directory and imports the
/// peers' handles (the same role NVSHMEM's bootstrapping plays; here it is
/// a plain shared filesystem, so no extra library or job scheduler is
/// required).
pub struct Session {
    dir: PathBuf,
}

impl Session {
    /// Create (or attach to) the session directory.
    ///
    /// # Errors
    /// [`Error::Internal`] when the directory cannot be created.
    pub fn new(dir: impl AsRef<Path>) -> Result<Session> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("session dir {}: {e}", dir.display())))?;
        Ok(Session { dir })
    }

    fn rank_path(&self, rank: usize) -> PathBuf {
        self.dir.join(format!("rank{rank}.bin"))
    }

    /// Publish this rank's export handle and geometry.
    ///
    /// # Errors
    /// [`Error::Internal`] on I/O failure.
    pub fn publish(
        &self,
        rank: usize,
        handle: &[u8; 64],
        k: usize,
        cap: usize,
        tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        let mut blob = Vec::with_capacity(64 + 16);
        blob.extend_from_slice(handle);
        blob.extend_from_slice(&(k as u32).to_le_bytes());
        blob.extend_from_slice(&(cap as u32).to_le_bytes());
        blob.extend_from_slice(&(tokens as u32).to_le_bytes());
        blob.extend_from_slice(&(hidden as u32).to_le_bytes());
        let tmp = self.dir.join(format!("rank{rank}.tmp"));
        std::fs::write(&tmp, &blob)
            .map_err(|e| Error::internal(format!("publish rank{rank}: {e}")))?;
        std::fs::rename(&tmp, self.rank_path(rank))
            .map_err(|e| Error::internal(format!("publish rank{rank}: {e}")))?;
        Ok(())
    }

    /// Wait until every peer has published; returns their
    /// `(handle, k, cap, tokens, hidden)` records in rank order.
    ///
    /// # Errors
    /// [`Error::Internal`] on I/O failure or a geometry mismatch.
    pub fn wait_peers(&self, world: usize) -> Result<Vec<PeerRecord>> {
        let mut out = Vec::with_capacity(world);
        for r in 0..world {
            let path = self.rank_path(r);
            let blob = loop {
                match std::fs::read(&path) {
                    Ok(b) if b.len() == 80 => break b,
                    Ok(_) => {
                        return Err(Error::internal(format!(
                            "rank{r} published a malformed record"
                        )))
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        std::thread::sleep(std::time::Duration::from_millis(25));
                    }
                    Err(e) => return Err(Error::internal(format!("read rank{r}: {e}"))),
                }
            };
            let mut handle = [0u8; 64];
            handle.copy_from_slice(&blob[0..64]);
            let rd =
                |o: usize| u32::from_le_bytes(blob[o..o + 4].try_into().expect("4 bytes")) as usize;
            out.push((handle, rd(64), rd(68), rd(72), rd(76)));
        }
        Ok(out)
    }
}

/// The symmetric-heap expert-parallel engine over a [`PeerMesh`].
///
/// Owns one heap per rank plus the per-rank device pointer tables the
/// kernels consume. All launches are enqueued on the mesh's per-rank
/// streams; nothing synchronizes except the documented device-side
/// barriers.
pub struct SymmetricEp {
    mesh: PeerMesh,
    heaps: Vec<SymmetricHeap>,
    /// Device array `[world]` of heap addresses, one table per rank
    /// (shared in-process; per-rank imports for multi-process).
    ptrs: Vec<Buffer>,
    world: usize,
    k: usize,
    cap: usize,
    tokens_max: usize,
    hidden: usize,
}

impl std::fmt::Debug for SymmetricEp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymmetricEp")
            .field("world", &self.heaps.len())
            .field("k", &self.k)
            .field("cap", &self.cap)
            .field("tokens_max", &self.tokens_max)
            .field("hidden", &self.hidden)
            .finish()
    }
}

impl SymmetricEp {
    /// Allocate one symmetric heap per rank of the mesh (single process).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on geometry violations;
    /// [`Error::CapacityExceeded`] when a heap allocation fails.
    pub fn new(
        mesh: PeerMesh,
        k: usize,
        cap: usize,
        tokens_max: usize,
        hidden: usize,
    ) -> Result<SymmetricEp> {
        let world = mesh.world();
        let heaps: Vec<SymmetricHeap> = (0..world)
            .map(|r| SymmetricHeap::alloc(mesh.device(r), k, cap, tokens_max, hidden))
            .collect::<Result<_>>()?;
        let ptrs = Self::pointer_tables(&mesh, heaps.iter().map(|h| h.ptr()))?;
        Ok(SymmetricEp {
            mesh,
            heaps,
            ptrs,
            world,
            k,
            cap,
            tokens_max,
            hidden,
        })
    }

    /// Multi-process construction: allocate this rank's heap, publish its
    /// IPC handle through the [`Session`], import every peer's handle, and
    /// build the local pointer table.
    ///
    /// # Errors
    /// As [`SymmetricEp::new`], plus [`Error::Internal`] on IPC or session
    /// failures.
    pub fn new_multiprocess(
        mesh: PeerMesh,
        my_rank: usize,
        session: &Session,
        k: usize,
        cap: usize,
        tokens_max: usize,
        hidden: usize,
    ) -> Result<SymmetricEp> {
        ensure!(
            mesh.world() == 1,
            InvalidArgument,
            "a multi-process SymmetricEp uses a single-device mesh (one process per rank)"
        );
        let dev = mesh.device(0);
        let my_heap = SymmetricHeap::alloc(dev, k, cap, tokens_max, hidden)?;
        let handle = dev.ipc_export(&my_heap.buf)?;
        session.publish(my_rank, &handle, k, cap, tokens_max, hidden)?;
        let peers = session.wait_peers(my_rank + 1)?;
        // The pointer table: imported peer pointers plus our own buffer.
        let world = peers.len();
        let mut ptr_values: Vec<u64> = Vec::with_capacity(world);
        let mut imported: Vec<Buffer> = Vec::with_capacity(world);
        for (r, (h, pk, pcap, ptok, phid)) in peers.iter().enumerate() {
            ensure!(*pk == k && *pcap == cap && *ptok == tokens_max && *phid == hidden,
                InvalidArgument,
                "peer {r} geometry mismatch ({pk},{pcap},{ptok},{phid}) vs ({k},{cap},{tokens_max},{hidden})");
            if r == my_rank {
                ptr_values.push(my_heap.ptr());
            } else {
                let b = dev.ipc_open(h)?;
                ptr_values.push(b.ptr());
                imported.push(b);
            }
        }
        let table = dev.alloc_from(&ptr_values)?;
        // Keep the imported buffers alive for as long as the engine: fold
        // them into the heap list (they are only pointer owners; geometry
        // lives in the struct fields).
        let mut heaps = vec![my_heap];
        heaps.extend(imported.into_iter().map(|b| SymmetricHeap {
            buf: b,
            k_channels: k,
            cap,
            tokens_max,
            hidden,
        }));
        let ptrs = vec![table];
        let world = peers.len();
        Ok(SymmetricEp {
            mesh,
            heaps,
            ptrs,
            world,
            k,
            cap,
            tokens_max,
            hidden,
        })
    }

    fn pointer_tables(mesh: &PeerMesh, ptrs: impl Iterator<Item = u64>) -> Result<Vec<Buffer>> {
        let values: Vec<u64> = ptrs.collect();
        let tables = (0..mesh.world())
            .map(|r| mesh.device(r).alloc_from(&values))
            .collect::<Result<_>>()?;
        Ok(tables)
    }

    /// Mesh size (ranks).
    pub fn world(&self) -> usize {
        self.world
    }

    /// Routing-slot count (MoE top-k).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Per-channel capacity.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The underlying mesh (streams/events for caller orchestration).
    pub fn mesh(&self) -> &PeerMesh {
        &self.mesh
    }

    /// The heap of rank `r` (local allocations only).
    pub fn heap(&self, r: usize) -> Option<&SymmetricHeap> {
        self.heaps.get(r)
    }

    // -- granular step operations (all async; see kernels/peer_ll.cu) ------

    /// Reset every rank's symmetric counters, ordered before any dispatch
    /// via the mesh's event barrier (resets must complete process-wide
    /// before the first sender increments).
    ///
    /// # Errors
    /// [`Error::Internal`] on launch failures.
    pub fn reset_all(&self, k: &Kernels) -> Result<()> {
        let world = self.world();
        for r in 0..world {
            let cells = self.heaps[r].cells();
            k.launch1d(
                "di_ep_reset_ll_kernel",
                cells,
                128,
                0,
                deepinfer_gpu::KernelArgs::new()
                    .raw_ptr(self.heaps[r].ptr())
                    .i32(cells as i32),
                self.mesh.stream(r),
            )?;
        }
        // Event barrier: every dispatch stream waits for every reset.
        for r in 0..world {
            self.mesh.record(r)?;
        }
        for d in 0..world {
            for s in 0..world {
                self.mesh.wait(d, s)?;
            }
        }
        Ok(())
    }

    /// Dispatch rank `r`'s quantized activations to the owner ranks
    /// (`tokens <= tokens_max`). Launches the per-row dispatch kernel plus
    /// the step-barrier tail kernel on rank r's stream.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on capacity violations;
    /// [`Error::Internal`] on launch failures.
    pub fn dispatch(
        &self,
        k: &Kernels,
        r: usize,
        xq: &Buffer,
        xsf: &Buffer,
        topk_idx: &Buffer,
        tokens: usize,
        num_experts: usize,
    ) -> Result<()> {
        let world = self.world();
        ensure!(
            tokens <= self.tokens_max,
            CapacityExceeded,
            "dispatch of {tokens} tokens exceeds tokens_max = {}",
            self.tokens_max
        );
        let s = self.mesh.stream(r);
        let f = k.f("di_ep_dispatch_ll_kernel")?;
        f.launch(
            self.mesh.device(r),
            deepinfer_gpu::Launch {
                grid: (tokens as u32, self.k as u32, 1),
                block: (128, 1, 1),
                shared: 0,
            },
            &deepinfer_gpu::KernelArgs::new()
                .ptr(xq)
                .ptr(xsf)
                .ptr(topk_idx)
                .ptr(&self.ptrs[r])
                .i32(r as i32)
                .i32(world as i32)
                .i32(tokens as i32)
                .i32(self.hidden as i32)
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(num_experts as i32),
            s,
        )?;
        let d = k.f("di_ep_dispatch_done_ll_kernel")?;
        d.launch(
            self.mesh.device(r),
            deepinfer_gpu::Launch {
                grid: (1, 1, 1),
                block: (32, 1, 1),
                shared: 0,
            },
            &deepinfer_gpu::KernelArgs::new()
                .ptr(&self.ptrs[r])
                .i32(world as i32)
                .i32(self.k as i32),
            s,
        )
    }

    /// Group rank `r`'s received channel rows into the masked per-expert
    /// layout: `aq [E*cap, hidden]` u8, `asf [E*cap, hidden/128]` u8,
    /// `meta2 [E*cap, 4]` i32, `counters [E]` i32 (masked_m), `drops` i32.
    /// Spins on the device step barrier; enqueue after all ranks'
    /// [`SymmetricEp::dispatch`].
    ///
    /// # Errors
    /// [`Error::Internal`] on launch failures.
    pub fn group(
        &self,
        k: &Kernels,
        r: usize,
        counters: &Buffer,
        aq: &Buffer,
        asf: &Buffer,
        meta2: &Buffer,
        drops: &Buffer,
    ) -> Result<()> {
        let world = self.world();
        let tiles = self.cap.div_ceil(128);
        k.launch1d(
            "di_ep_group_ll_kernel",
            self.k * tiles,
            128,
            0,
            deepinfer_gpu::KernelArgs::new()
                .ptr(&self.ptrs[r])
                .i32(r as i32)
                .i32(world as i32)
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(self.hidden as i32)
                .ptr(counters)
                .ptr(aq)
                .ptr(asf)
                .ptr(meta2)
                .ptr(drops),
            self.mesh.stream(r),
        )
    }

    /// Write rank `r`'s valid expert outputs (`y [E*cap, hidden]` f32 in
    /// the masked layout) back to the home ranks' combine slots.
    ///
    /// # Errors
    /// [`Error::Internal`] on launch failures.
    pub fn combine(
        &self,
        k: &Kernels,
        r: usize,
        y: &Buffer,
        meta2: &Buffer,
        counters: &Buffer,
        num_experts: usize,
    ) -> Result<()> {
        let tiles = self.cap.div_ceil(128);
        let f = k.f("di_ep_combine_ll_kernel")?;
        f.launch(
            self.mesh.device(r),
            deepinfer_gpu::Launch {
                grid: (tiles as u32, num_experts as u32, 1),
                block: (128, 1, 1),
                shared: 0,
            },
            &deepinfer_gpu::KernelArgs::new()
                .ptr(y)
                .ptr(meta2)
                .ptr(counters)
                .ptr(&self.ptrs[r])
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(self.hidden as i32)
                .i32(num_experts as i32),
            self.mesh.stream(r),
        )
    }

    /// Reduce rank `r`'s combine buffer into `out [tokens, hidden]` f32
    /// with the routing weights (`topk_w [tokens, K]` f32,
    /// `topk_idx [tokens, K]` i32; `-1` slots contribute zero). Spins until
    /// every dispatched pair came home.
    ///
    /// # Errors
    /// [`Error::Internal`] on launch failures.
    pub fn reduce(
        &self,
        k: &Kernels,
        r: usize,
        topk_w: &Buffer,
        topk_idx: &Buffer,
        out: &Buffer,
        tokens: usize,
    ) -> Result<()> {
        k.launch1d(
            "di_ep_reduce_ll_kernel",
            (tokens * self.hidden).max(1),
            256,
            0,
            deepinfer_gpu::KernelArgs::new()
                .ptr(&self.ptrs[r])
                .i32(r as i32)
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(self.hidden as i32)
                .i32(tokens as i32)
                .ptr(topk_w)
                .ptr(topk_idx)
                .ptr(out),
            self.mesh.stream(r),
        )
    }
}

#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_ep_reset_ll_kernel",
    "di_ep_dispatch_ll_kernel",
    "di_ep_dispatch_done_ll_kernel",
    "di_ep_group_ll_kernel",
    "di_ep_combine_ll_kernel",
    "di_ep_reduce_ll_kernel",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_geometry_matches_kernel_formulas() {
        // The Rust mirror must agree with the .cu formulas exactly.
        for k in [1usize, 8, 16] {
            for hidden in [128usize, 7168] {
                let rb = row_bytes(hidden);
                assert_eq!(rb % 16, 0);
                assert_eq!(rb, (hidden + hidden / 128 + 16 + 15) & !15);
                let co = chan_off(k);
                assert_eq!(co, ((3 * k + 6) * 4 + 15) & !15);
                let cb = combine_off(k, 64, hidden);
                assert_eq!(cb, co + k * 64 * rb);
                let total = heap_bytes(k, 64, 32, hidden);
                assert_eq!(total, cb + 32 * k * hidden * 4);
                assert!(total > 0);
            }
        }
    }

    #[test]
    fn cells_cover_all_counters() {
        // counts[K] + produced[K] + expected[K] + 3 scalars + pad.
        assert_eq!(cells(8), 30);
        assert_eq!(cells(1), 9);
    }

    #[test]
    fn session_roundtrips_a_record() {
        let dir =
            std::env::temp_dir().join(format!("deepinfer-ep-session-test-{}", std::process::id()));
        let s = Session::new(&dir).expect("session dir");
        let mut h = [7u8; 64];
        h[0] = 42;
        s.publish(0, &h, 8, 64, 32, 7168).expect("publish");
        let peers = s.wait_peers(1).expect("wait");
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].0, h);
        assert_eq!(peers[0].1, 8);
        assert_eq!(peers[0].2, 64);
        assert_eq!(peers[0].3, 32);
        assert_eq!(peers[0].4, 7168);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
