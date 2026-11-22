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
use deepinfer_gpu::{Buffer, Device, Stream};

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
    /// Local expert slots (0 = legacy heap without the fused workspace).
    experts: usize,
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
                &heap_bytes_fused(
                    self.k_channels,
                    self.cap,
                    self.tokens_max,
                    self.hidden,
                    self.experts,
                ),
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

/// Total heap bytes for the given geometry (legacy, no fused regions).
#[allow(dead_code)] // the layout mirror is pinned by the unit tests
fn heap_bytes(k: usize, cap: usize, tokens_max: usize, hidden: usize) -> usize {
    combine_off(k, cap, hidden) + tokens_max * k * hidden * 4
}

/// Fused masked-workspace offset (mirror of the device `ep_masked_off`).
fn masked_off(k: usize, cap: usize, tokens_max: usize, hidden: usize) -> usize {
    (combine_off(k, cap, hidden) + tokens_max * k * hidden * 4 + 15) & !15
}

/// Experts-per-rank bound for a balanced default placement.
pub fn experts_per_rank(world: usize, num_experts: usize) -> usize {
    if world == 0 {
        0
    } else {
        num_experts.div_ceil(world)
    }
}

/// Default rank map: expert `e` lives on rank `e * world / num_experts`
/// (the DeepEP contiguous-range placement; device mirror of the NULL-map
/// branch of `di_ep_dispatch_ll_kernel`).
pub fn default_rank_map(world: usize, num_experts: usize) -> Vec<i32> {
    (0..num_experts)
        .map(|e| (e * world / num_experts.max(1)) as i32)
        .collect()
}

/// Default local slot map of `rank`: `slot = e - first_e(rank)` where
/// `first_e(rank) = ceil(rank * num_experts / world)` — the exact lower
/// boundary of the contiguous range the device's `dst = e*world/E` formula
/// assigns to `rank` (the boundaries are `ceil`-spaced, so they only agree
/// with `rank * ceil(E/world)` when `E % world == 0`). Every expert routed
/// to `rank` by [`default_rank_map`] maps into `[0, experts_per_rank)` and
/// every other expert maps to `-1`.
pub fn default_slot_map(world: usize, num_experts: usize, rank: usize) -> Vec<i32> {
    let lo = (rank * num_experts).div_ceil(world.max(1));
    let hi = ((rank + 1) * num_experts).div_ceil(world.max(1));
    (0..num_experts)
        .map(|e| if e >= lo && e < hi { (e - lo) as i32 } else { -1 })
        .collect()
}

/// Byte offset of `y` inside the fused heap (mirror of `ep_fused_ptrs`).
fn fused_y_off(k: usize, cap: usize, tokens_max: usize, hidden: usize, e: usize) -> usize {
    let mb = masked_off(k, cap, tokens_max, hidden);
    let asf = mb + e * cap * hidden;
    let meta = asf + e * cap * (hidden / 128);
    let cnt = meta + e * cap * 16;
    (cnt + e * 4 + 4 + 15) & !15 // drops then 16B align
}

/// Byte offset of the completion table inside the fused heap.
fn fused_ctab_off(
    k: usize,
    cap: usize,
    tokens_max: usize,
    hidden: usize,
    e: usize,
) -> usize {
    fused_y_off(k, cap, tokens_max, hidden, e) + e * cap * hidden * 4
}

/// Total heap bytes including the fused expert workspace.
fn heap_bytes_fused(
    k: usize,
    cap: usize,
    tokens_max: usize,
    hidden: usize,
    experts: usize,
) -> usize {
    fused_ctab_off(k, cap, tokens_max, hidden, experts) + tokens_max * k * 8
}

impl SymmetricHeap {
    /// Allocate a legacy heap on `dev` for the given routing geometry
    /// (no fused expert workspace; see [`SymmetricHeap::alloc_fused`]).
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
        SymmetricHeap::alloc_fused(dev, k_channels, cap, tokens_max, hidden, 0)
    }

    /// Allocate a heap that additionally carries the **fused expert
    /// workspace**: the masked per-expert activations/scales/meta/
    /// counters, the expert-output rows `y` (which peers read remotely)
    /// and the completion table. `experts` is the per-rank expert-slot
    /// bound ([`experts_per_rank`] under the default placement).
    ///
    /// # Errors
    /// As [`SymmetricHeap::alloc`].
    pub fn alloc_fused(
        dev: &Device,
        k_channels: usize,
        cap: usize,
        tokens_max: usize,
        hidden: usize,
        experts: usize,
    ) -> Result<SymmetricHeap> {
        ensure!(k_channels >= 1, InvalidArgument, "k_channels must be >= 1");
        ensure!(cap >= 1, InvalidArgument, "cap must be >= 1");
        ensure!(tokens_max >= 1, InvalidArgument, "tokens_max must be >= 1");
        ensure!(
            hidden % 128 == 0,
            InvalidArgument,
            "hidden must be a multiple of 128"
        );
        let bytes = heap_bytes_fused(k_channels, cap, tokens_max, hidden, experts);
        let buf = dev
            .alloc(bytes)
            .map_err(|e| Error::CapacityExceeded(format!("symmetric heap ({bytes} B): {e}")))?;
        Ok(SymmetricHeap {
            buf,
            k_channels,
            cap,
            tokens_max,
            hidden,
            experts,
        })
    }

    /// Raw heap address (the value peers write through).
    pub fn ptr(&self) -> u64 {
        self.buf.ptr()
    }

    /// Heap size in bytes.
    pub fn bytes(&self) -> usize {
        heap_bytes_fused(
            self.k_channels,
            self.cap,
            self.tokens_max,
            self.hidden,
            self.experts,
        )
    }

    /// Number of i32 counter cells (`3K + 6`).
    pub fn cells(&self) -> usize {
        cells(self.k_channels)
    }

    /// Local expert-slot bound (0 = legacy heap).
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// A sliced view of a heap region (for handing heap slices to the
    /// grouped masked GEMM).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when the range is outside the heap.
    fn region(&self, offset: usize, len: usize) -> Result<Buffer> {
        self.buf.slice(offset, len)
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
    /// Per-rank local expert slots (0 = legacy heaps).
    experts: usize,
}

impl std::fmt::Debug for SymmetricEp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymmetricEp")
            .field("world", &self.heaps.len())
            .field("k", &self.k)
            .field("cap", &self.cap)
            .field("tokens_max", &self.tokens_max)
            .field("hidden", &self.hidden)
            .field("experts", &self.experts)
            .finish()
    }
}

impl SymmetricEp {
    /// Allocate one symmetric heap per rank of the mesh (single process,
    /// legacy layout without the fused expert workspace).
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
        SymmetricEp::new_with_experts(mesh, k, cap, tokens_max, hidden, 0)
    }

    /// Allocate one fused-layout symmetric heap per rank: the heaps carry
    /// the masked expert workspace, the expert-output rows and the
    /// completion table (see [`SymmetricEp::group_fused`] and friends).
    ///
    /// # Errors
    /// As [`SymmetricEp::new`].
    pub fn new_with_experts(
        mesh: PeerMesh,
        k: usize,
        cap: usize,
        tokens_max: usize,
        hidden: usize,
        experts: usize,
    ) -> Result<SymmetricEp> {
        let world = mesh.world();
        let heaps: Vec<SymmetricHeap> = (0..world)
            .map(|r| {
                SymmetricHeap::alloc_fused(mesh.device(r), k, cap, tokens_max, hidden, experts)
            })
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
            experts,
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
            experts: 0,
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
            experts: 0,
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
    /// the step-barrier tail kernel on rank r's stream. `rank_map`
    /// (`[num_experts]` i32, EPLB placement) overrides the default
    /// contiguous-range map when present.
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
        rank_map: Option<&Buffer>,
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
                .u64(rank_map.map(|m| m.ptr()).unwrap_or(0))
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

    // -- fused expert path (P1-9; requires new_with_experts heaps) --------

    /// The fused masked-workspace buffer views of rank `r`'s heap — hand
    /// these straight to `gemm_grouped_masked_nt` (`aq`/`asf` as the A
    /// operands with `masked_m = counters`, `y` as D) so the grouped GEMM
    /// computes directly over the symmetric heap layout.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on legacy (non-fused) heaps;
    /// [`Error::Internal`] on slice failures.
    pub fn fused_buffers(&self, r: usize) -> Result<FusedBuffers> {
        ensure!(
            self.experts > 0,
            InvalidArgument,
            "fused_buffers needs new_with_experts heaps"
        );
        let e = self.experts;
        let mb = masked_off(self.k, self.cap, self.tokens_max, self.hidden);
        let asf = mb + e * self.cap * self.hidden;
        let meta = asf + e * self.cap * (self.hidden / 128);
        let cnt = meta + e * self.cap * 16;
        let y = fused_y_off(self.k, self.cap, self.tokens_max, self.hidden, e);
        let ctab = fused_ctab_off(self.k, self.cap, self.tokens_max, self.hidden, e);
        let heap = &self.heaps[r];
        Ok(FusedBuffers {
            aq: heap.region(mb, e * self.cap * self.hidden)?,
            asf: heap.region(asf, e * self.cap * (self.hidden / 128))?,
            counters: heap.region(cnt, e * 4)?,
            y: heap.region(y, e * self.cap * self.hidden * 4)?,
            ctab: heap.region(ctab, self.tokens_max * self.k * 8)?,
        })
    }

    /// Fused group: scatter rank `r`'s received channel rows into the
    /// heap's masked region using the local slot map (`slot_map[e]` = the
    /// local slot of logical expert `e` on rank `r`; see
    /// [`default_slot_map`]). Enqueue after every rank's dispatch.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on legacy heaps or a short slot map;
    /// [`Error::Internal`] on launch failures.
    pub fn group_fused(&self, k: &Kernels, r: usize, slot_map: &Buffer) -> Result<()> {
        ensure!(
            self.experts > 0,
            InvalidArgument,
            "group_fused needs new_with_experts heaps"
        );
        let world = self.world();
        let tiles = self.cap.div_ceil(128);
        k.launch1d(
            "di_ep_group_fused_ll_kernel",
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
                .i32(self.experts as i32)
                .i32(self.tokens_max as i32)
                .ptr(slot_map),
            self.mesh.stream(r),
        )
    }

    /// Fused publish: write one completion entry per valid expert row into
    /// the home ranks' completion tables (replaces the row-copying
    /// [`SymmetricEp::combine`]). Enqueue after this rank's grouped GEMM
    /// over [`SymmetricEp::fused_buffers`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on legacy heaps;
    /// [`Error::Internal`] on launch failures.
    pub fn publish(&self, k: &Kernels, r: usize) -> Result<()> {
        ensure!(
            self.experts > 0,
            InvalidArgument,
            "publish needs new_with_experts heaps"
        );
        let tiles = self.cap.div_ceil(128);
        let f = k.f("di_ep_publish_ll_kernel")?;
        f.launch(
            self.mesh.device(r),
            deepinfer_gpu::Launch {
                grid: (tiles as u32, self.experts as u32, 1),
                block: (128, 1, 1),
                shared: 0,
            },
            &deepinfer_gpu::KernelArgs::new()
                .ptr(&self.ptrs[r])
                .i32(r as i32)
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(self.hidden as i32)
                .i32(self.experts as i32)
                .i32(self.tokens_max as i32),
            self.mesh.stream(r),
        )
    }

    /// Fused reduce: read the expert outputs directly from the peers'
    /// heaps through the completion table (no combine staging copy).
    /// Deterministic: fixed k-ascending accumulation.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on legacy heaps;
    /// [`Error::Internal`] on launch failures.
    pub fn reduce_remote(
        &self,
        k: &Kernels,
        r: usize,
        topk_w: &Buffer,
        topk_idx: &Buffer,
        out: &Buffer,
        tokens: usize,
    ) -> Result<()> {
        ensure!(
            self.experts > 0,
            InvalidArgument,
            "reduce_remote needs new_with_experts heaps"
        );
        k.launch1d(
            "di_ep_reduce_remote_ll_kernel",
            (tokens * self.hidden).max(1),
            256,
            0,
            deepinfer_gpu::KernelArgs::new()
                .ptr(&self.ptrs[r])
                .i32(r as i32)
                .i32(self.k as i32)
                .i32(self.cap as i32)
                .i32(self.hidden as i32)
                .i32(self.experts as i32)
                .i32(self.tokens_max as i32)
                .i32(tokens as i32)
                .ptr(topk_w)
                .ptr(topk_idx)
                .ptr(out),
            self.mesh.stream(r),
        )
    }

    /// Zero the fused masked counters of every rank (the masked-layout
    /// `counters`/`drops` regions; the symmetric step counters are handled
    /// by [`SymmetricEp::reset_all`]). Call before dispatch of each step;
    /// the trailing event barrier orders the zeroing before any dispatch,
    /// exactly like `reset_all`.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] on legacy heaps;
    /// [`Error::Internal`] on upload failures.
    pub fn reset_fused(&self) -> Result<()> {
        ensure!(
            self.experts > 0,
            InvalidArgument,
            "reset_fused needs new_with_experts heaps"
        );
        for r in 0..self.world() {
            let b = self.fused_buffers(r)?;
            let zero = vec![0u8; (self.experts + 1) * 4];
            let dst = b.counters.slice(0, (self.experts + 1) * 4)?;
            self.mesh
                .stream(r)
                .write_async(self.mesh.device(r), &dst, &zero)?;
        }
        for r in 0..self.world() {
            self.mesh.record(r)?;
        }
        for d in 0..self.world() {
            for s in 0..self.world() {
                self.mesh.wait(d, s)?;
            }
        }
        Ok(())
    }
}

/// Sliced views of one rank's fused masked workspace (see
/// [`SymmetricEp::fused_buffers`]).
pub struct FusedBuffers {
    /// Masked e4m3 activations `[E*cap, hidden]` (A operand).
    pub aq: Buffer,
    /// Masked ue8m0 scales `[E*cap, hidden/128]`.
    pub asf: Buffer,
    /// Masked row counts `[E]` i32 (the `masked_m` GEMM operand).
    pub counters: Buffer,
    /// Expert outputs `[E*cap, hidden]` f32 (the D operand).
    pub y: Buffer,
    /// Completion table `[tokens_max, K]` u64.
    pub ctab: Buffer,
}

#[cfg(test)]
pub(crate) const KERNEL_NAMES: &[&str] = &[
    "di_ep_reset_ll_kernel",
    "di_ep_dispatch_ll_kernel",
    "di_ep_dispatch_done_ll_kernel",
    "di_ep_group_ll_kernel",
    "di_ep_combine_ll_kernel",
    "di_ep_reduce_ll_kernel",
    // P1-9 fused path.
    "di_ep_group_fused_ll_kernel",
    "di_ep_publish_ll_kernel",
    "di_ep_reduce_remote_ll_kernel",
    "di_ep_topk_histogram_ll_kernel",
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

impl Kernels {
    /// EPLB feedback: histogram one routing decision (`topk_idx [tokens, k]`
    /// i32; `-1` padding slots skipped) into `hist [num_experts]` i32 with
    /// integer atomics — order-independent, hence deterministic. The host
    /// drains `hist` periodically (e.g. every N serve steps) and feeds the
    /// counts to `deepinfer-eplb`'s `ExpertStats`, whose moving average
    /// drives `rebalance_experts`; the resulting placement becomes the
    /// dispatch `rank_map` + per-rank `slot_map` tables.
    ///
    /// # Errors
    /// [`Error::Internal`] on launch failures.
    pub fn topk_histogram(
        &self,
        topk_idx: &Buffer,
        hist: &Buffer,
        tokens: usize,
        k: usize,
        num_experts: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.launch1d(
            "di_ep_topk_histogram_ll_kernel",
            (tokens * k).max(1),
            256,
            0,
            deepinfer_gpu::KernelArgs::new()
                .ptr(topk_idx)
                .ptr(hist)
                .i32(tokens as i32)
                .i32(k as i32)
                .i32(num_experts as i32),
            stream,
        )
    }
}

#[cfg(test)]
mod fused_tests {
    use super::*;

    #[test]
    fn fused_offsets_match_device_formulas() {
        for k in [1usize, 8, 16] {
            for hidden in [128usize, 7168] {
                for experts in [1usize, 32, 256] {
                    let (cap, tokens) = (64usize, 32usize);
                    let mb = masked_off(k, cap, tokens, hidden);
                    // 16-byte aligned and past the legacy combine region.
                    assert_eq!(mb % 16, 0);
                    assert!(mb >= combine_off(k, cap, hidden) + tokens * k * hidden * 4);
                    let asf = mb + experts * cap * hidden;
                    let meta = asf + experts * cap * (hidden / 128);
                    let cnt = meta + experts * cap * 16;
                    let y = fused_y_off(k, cap, tokens, hidden, experts);
                    // y is 16B aligned, after counters+drops.
                    assert_eq!(y % 16, 0);
                    assert!(y >= cnt + experts * 4 + 4);
                    let ctab = fused_ctab_off(k, cap, tokens, hidden, experts);
                    assert_eq!(ctab, y + experts * cap * hidden * 4);
                    let total = heap_bytes_fused(k, cap, tokens, hidden, experts);
                    assert_eq!(total, ctab + tokens * k * 8);
                    // Regions are strictly ordered and non-overlapping.
                    assert!(mb < asf && asf < meta && meta < cnt && cnt <= y && y < ctab);
                }
            }
        }
    }

    #[test]
    fn default_maps_partition_the_experts() {
        for world in [1usize, 2, 4, 8] {
            for e in [1usize, 8, 256, 257] {
                let per = experts_per_rank(world, e);
                let rmap = default_rank_map(world, e);
                assert_eq!(rmap.len(), e);
                let mut slots: Vec<Vec<i32>> = (0..world)
                    .map(|r| default_slot_map(world, e, r))
                    .collect();
                for (ex, &dst) in rmap.iter().enumerate() {
                    let dst = dst as usize;
                    assert!(dst < world, "rank out of range");
                    let slot = slots[dst][ex];
                    assert!(
                        slot >= 0 && (slot as usize) < per,
                        "expert {ex} -> rank {dst} slot {slot} outside [0, {per})"
                    );
                    // The slot map of every OTHER rank rejects the expert.
                    for (r, sm) in slots.iter_mut().enumerate() {
                        if r != dst {
                            assert_eq!(sm[ex], -1, "expert {ex} also mapped on rank {r}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn default_maps_agree_with_the_device_formula() {
        // dst = e * world / num_experts (the kernel's NULL-map branch).
        let (world, e) = (8usize, 32usize);
        let rmap = default_rank_map(world, e);
        for ex in 0..e {
            assert_eq!(rmap[ex] as usize, ex * world / e);
        }
        // Contiguous ranges: rank r owns [r*per, (r+1)*per).
        let per = experts_per_rank(world, e);
        assert_eq!(per, 4);
        let sm = default_slot_map(world, e, 3);
        assert_eq!(sm[12], 0);
        assert_eq!(sm[15], 3);
        assert_eq!(sm[16], -1);
    }

    #[test]
    fn completion_entry_packing() {
        // (rank << 48) | offset — rank in the top 16 bits, offset below.
        let rank = 7usize;
        let off = 0x0000_1234_5678_9abcusize;
        let entry = ((rank as u64) << 48) | off as u64;
        assert_eq!((entry >> 48) as usize, rank);
        assert_eq!((entry & 0x0000_FFFF_FFFF_FFFF) as usize, off);
        // Heaps are far below 2^48 bytes, so the split is lossless.
        assert!(heap_bytes_fused(16, 512, 4096, 7168, 256) < (1usize << 48));
    }
}
