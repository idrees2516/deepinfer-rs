//! Events and stream-capture graphs — the synchronization and replay
//! substrate of the GPU stack.
//!
//! * [`Event`] — driver events for cross-stream and cross-device ordering.
//!   The peer-collective EP layer builds its all-to-all dispatch/combine
//!   barriers from these; they are also the portable way to overlap copies
//!   with compute on separate streams.
//! * [`CaptureGuard`] / [`GraphExec`] — capture a sequence of **kernel
//!   launches** on one stream into a driver graph (CUDA graphs / HIP
//!   graphs), instantiate it once, and replay the whole sequence with a
//!   single [`GraphExec::launch`]. This is the decode-latency weapon of
//!   every production inference runtime: a V4.1-Flash decode step is
//!   hundreds of small launches, and graph replay collapses their launch
//!   overhead to one call.
//!
//! # Capture contract
//!
//! While a [`CaptureGuard`] is alive, **only kernel launches** may be
//! enqueued on the captured stream. Host↔device copies are captured as
//! memcpy nodes whose legality depends on host-memory pinning across
//! driver versions, so the decode path is structured to do all uploads
//! *before* replay (outside the graph), through fixed device buffers whose
//! addresses are baked into the graph. Every scalar that changes between
//! replays — sequence length, cache write rows, attention window starts —
//! must live in device memory and be read by the kernels; the
//! `deepinfer-kernels` `_dev` kernel family exists exactly for this
//! (mirroring how production attention kernels take `seq_lens` as device
//! int arrays).
//!
//! Both vendors' capture modes are `RELAXED` (other threads may keep using
//! the driver while this thread captures).

use deepinfer_core::{ensure, Result};

use crate::device::{Device, Stream};
use crate::ffi::{cuda, hip};

/// An event on a device's context/streams.
///
/// Created via [`Device::event`]; recorded on a stream with
/// [`Event::record`]; other streams (possibly on other devices with peer
/// access) wait on it with [`Stream::wait_event`].
pub struct Event {
    dev: Device,
    handle: cuda::Handle,
}

// SAFETY: event handles are thread-safe in both vendors' APIs.
unsafe impl Send for Event {}
// SAFETY: see above; synchronization semantics are the caller's concern.
unsafe impl Sync for Event {}

impl Device {
    /// Creates an event on this device.
    ///
    /// # Errors
    /// [`Error::Internal`] when the driver rejects the creation or the
    /// (optional) event entry points are missing from an old driver.
    pub fn event(&self) -> Result<Event> {
        self.make_current()?;
        match (self.gpu().cuda(), self.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .event_create
                    .ok_or_else(crate::device::no_events_error)?;
                let mut h: cuda::Handle = std::ptr::null_mut();
                // SAFETY: out-pointer valid; flags = CU_EVENT_DEFAULT.
                let r = unsafe { (f)(&mut h, cuda::CU_EVENT_DEFAULT) };
                ensure!(r == 0, Internal, "cuEventCreate: {}", c.error_string(r));
                Ok(Event {
                    dev: self.clone(),
                    handle: h,
                })
            }
            (_, Some(h)) => {
                let f = h
                    .event_create
                    .ok_or_else(crate::device::no_events_error)?;
                let mut ev: hip::Handle = std::ptr::null_mut();
                // SAFETY: out-pointer valid.
                let r = unsafe { (f)(&mut ev) };
                ensure!(r == 0, Internal, "hipEventCreate: {}", h.error_string(r));
                Ok(Event {
                    dev: self.clone(),
                    handle: ev,
                })
            }
            _ => unreachable!("one backend is always loaded"),
        }
    }
}

impl Event {
    /// Records the event at the current position of `stream`.
    ///
    /// # Errors
    /// [`Error::Internal`] on a driver rejection (e.g. recording on a
    /// stream from a different context without peer access).
    pub fn record(&self, stream: &Stream) -> Result<()> {
        self.dev.make_current()?;
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .event_record
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: event and stream handles valid in the current context.
                let r = unsafe { (f)(self.handle, stream.raw()) };
                ensure!(r == 0, Internal, "cuEventRecord: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                let f = h
                    .event_record
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: see above.
                let r = unsafe { (f)(self.handle, stream.raw()) };
                ensure!(r == 0, Internal, "hipEventRecord: {}", h.error_string(r));
            }
            _ => unreachable!("one backend is always loaded"),
        }
        Ok(())
    }

    /// Blocks the host until the event completes.
    ///
    /// # Errors
    /// [`Error::Internal`] on a driver rejection.
    pub fn synchronize(&self) -> Result<()> {
        self.dev.make_current()?;
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .event_synchronize
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: event handle valid.
                let r = unsafe { (f)(self.handle) };
                ensure!(r == 0, Internal, "cuEventSynchronize: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                let f = h
                    .event_synchronize
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: see above.
                let r = unsafe { (f)(self.handle) };
                ensure!(r == 0, Internal, "hipEventSynchronize: {}", h.error_string(r));
            }
            _ => unreachable!("one backend is always loaded"),
        }
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // Best-effort destroy; failure cannot be reported from Drop.
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                if let Some(f) = c.event_destroy {
                    // SAFETY: handle valid until this drop.
                    unsafe { (f)(self.handle) };
                }
            }
            (_, Some(h)) => {
                if let Some(f) = h.event_destroy {
                    // SAFETY: see above.
                    unsafe { (f)(self.handle) };
                }
            }
            _ => unreachable!("one backend is always loaded"),
        }
    }
}

impl Stream {
    /// Makes all subsequent work on this stream wait for `event` (which may
    /// have been recorded on another stream or, with peer access enabled,
    /// another device).
    ///
    /// # Errors
    /// [`Error::Internal`] on a driver rejection.
    pub fn wait_event(&self, dev: &Device, event: &Event) -> Result<()> {
        dev.make_current()?;
        match (dev.gpu().cuda(), dev.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .stream_wait_event
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: stream and event handles valid in the current context.
                let r = unsafe { (f)(self.raw(), event.handle, 0) };
                ensure!(r == 0, Internal, "cuStreamWaitEvent: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                let f = h
                    .stream_wait_event
                    .ok_or_else(crate::device::no_events_error)?;
                // SAFETY: see above.
                let r = unsafe { (f)(self.raw(), event.handle, 0) };
                ensure!(r == 0, Internal, "hipStreamWaitEvent: {}", h.error_string(r));
            }
            _ => unreachable!("one backend is always loaded"),
        }
        Ok(())
    }
}

/// Whether the driver of `dev` exposes stream-capture graphs (CUDA >= 11.4
/// or ROCm >= 5.7). When false, [`Device::begin_capture`] returns
/// [`deepinfer_core::Error::Internal`] with a clear message and callers
/// should fall back to eager launches.
pub fn graph_supported(dev: &Device) -> bool {
    match (dev.gpu().cuda(), dev.gpu().hip()) {
        (Some(c), _) => c.stream_begin_capture.is_some() && c.stream_end_capture.is_some(),
        (_, Some(h)) => h.stream_begin_capture.is_some() && h.stream_end_capture.is_some(),
        _ => false,
    }
}

/// A live stream capture. Created by [`Device::begin_capture`]; finished by
/// [`CaptureGuard::end`], which instantiates the captured graph into a
/// replayable [`GraphExec`].
///
/// While the guard is alive, only kernel launches may be enqueued on the
/// captured stream (see the module-level capture contract). Dropping the
/// guard without calling `end` aborts the capture cleanly (the stream
/// returns to normal operation).
pub struct CaptureGuard<'a> {
    dev: Device,
    stream: &'a Stream,
    active: bool,
}

impl Device {
    /// Begins capturing kernel launches on `stream` (which must belong to
    /// this device). Capture mode is `RELAXED` on both vendors.
    ///
    /// # Errors
    /// [`Error::Internal`] when graphs are unsupported by the driver or the
    /// stream cannot enter capture (e.g. it has pending cross-stream
    /// dependencies the driver disallows).
    pub fn begin_capture<'a>(&self, stream: &'a Stream) -> Result<CaptureGuard<'a>> {
        self.make_current()?;
        match (self.gpu().cuda(), self.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .stream_begin_capture
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: stream handle valid in the current context.
                let r = unsafe { (f)(stream.raw(), cuda::CU_STREAM_CAPTURE_MODE_RELAXED) };
                ensure!(
                    r == 0,
                    Internal,
                    "cuStreamBeginCapture: {}",
                    c.error_string(r)
                );
            }
            (_, Some(h)) => {
                let f = h
                    .stream_begin_capture
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: see above.
                let r = unsafe { (f)(stream.raw(), hip::HIP_STREAM_CAPTURE_MODE_RELAXED) };
                ensure!(
                    r == 0,
                    Internal,
                    "hipStreamBeginCapture: {}",
                    h.error_string(r)
                );
            }
            _ => unreachable!("one backend is always loaded"),
        }
        Ok(CaptureGuard {
            dev: self.clone(),
            stream,
            active: true,
        })
    }
}

impl CaptureGuard<'_> {
    /// Ends the capture and instantiates the captured graph.
    ///
    /// The intermediate `CUgraph`/`hipGraph_t` is destroyed immediately
    /// after instantiation (the executable form is independent of it).
    ///
    /// # Errors
    /// [`Error::Internal`] when the capture was invalid (illegal ops, work
    /// on other unresolved streams) or instantiation fails.
    pub fn end(mut self) -> Result<GraphExec> {
        self.finish()
    }

    fn finish(&mut self) -> Result<GraphExec> {
        if !self.active {
            return Err(deepinfer_core::Error::internal(
                "capture already finished",
            ));
        }
        self.active = false;
        self.dev.make_current()?;
        let mut graph: cuda::Handle = std::ptr::null_mut();
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                let end = c
                    .stream_end_capture
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: stream handle valid; out-pointer valid.
                let r = unsafe { (end)(self.stream.raw(), &mut graph) };
                ensure!(r == 0, Internal, "cuStreamEndCapture: {}", c.error_string(r));
                let inst = c
                    .graph_instantiate
                    .ok_or_else(crate::device::no_graphs_error)?;
                let mut exec: cuda::Handle = std::ptr::null_mut();
                // SAFETY: graph handle from the driver; out-pointer valid; flags 0.
                let r = unsafe { (inst)(&mut exec, graph, 0) };
                ensure!(
                    r == 0,
                    Internal,
                    "cuGraphInstantiateWithFlags: {}",
                    c.error_string(r)
                );
                if let Some(destroy) = c.graph_destroy {
                    // SAFETY: graph handle valid; exec is independent after
                    // instantiation per driver contract.
                    unsafe { (destroy)(graph) };
                }
                Ok(GraphExec {
                    dev: self.dev.clone(),
                    handle: exec,
                })
            }
            (_, Some(h)) => {
                let end = h
                    .stream_end_capture
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: see above.
                let r = unsafe { (end)(self.stream.raw(), &mut graph) };
                ensure!(r == 0, Internal, "hipStreamEndCapture: {}", h.error_string(r));
                let inst = h
                    .graph_instantiate
                    .ok_or_else(crate::device::no_graphs_error)?;
                let mut exec: hip::Handle = std::ptr::null_mut();
                // SAFETY: see above.
                let r = unsafe { (inst)(&mut exec, graph, 0) };
                ensure!(
                    r == 0,
                    Internal,
                    "hipGraphInstantiateWithFlags: {}",
                    h.error_string(r)
                );
                if let Some(destroy) = h.graph_destroy {
                    // SAFETY: see above.
                    unsafe { (destroy)(graph) };
                }
                Ok(GraphExec {
                    dev: self.dev.clone(),
                    handle: exec,
                })
            }
            _ => unreachable!("one backend is always loaded"),
        }
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Abort path: end the capture (leaving it open would poison the
        // thread) and destroy the returned graph without instantiating it.
        // Errors are swallowed — Drop cannot report.
        self.active = false;
        let _ = self.dev.make_current();
        let mut graph: cuda::Handle = std::ptr::null_mut();
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                if let Some(end) = c.stream_end_capture {
                    // SAFETY: stream handle valid.
                    if unsafe { (end)(self.stream.raw(), &mut graph) } == 0 {
                        if let Some(destroy) = c.graph_destroy {
                            // SAFETY: graph handle from the driver.
                            unsafe { (destroy)(graph) };
                        }
                    }
                }
            }
            (_, Some(h)) => {
                if let Some(end) = h.stream_end_capture {
                    // SAFETY: see above.
                    if unsafe { (end)(self.stream.raw(), &mut graph) } == 0 {
                        if let Some(destroy) = h.graph_destroy {
                            // SAFETY: see above.
                            unsafe { (destroy)(graph) };
                        }
                    }
                }
            }
            _ => unreachable!("one backend is always loaded"),
        }
    }
}

/// An instantiated, replayable graph.
///
/// Produced by [`CaptureGuard::end`]. All buffer addresses are baked in at
/// capture time — every buffer used by the captured kernels must outlive
/// this handle. Replays are enqueued with [`GraphExec::launch`] on any
/// stream of the owning device and are bit-identical to re-running the
/// captured launch sequence by hand (the kernels are deterministic).
pub struct GraphExec {
    dev: Device,
    handle: cuda::Handle,
}

// SAFETY: graph exec handles are thread-safe in both vendors' APIs.
unsafe impl Send for GraphExec {}
// SAFETY: see above.
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Enqueues one replay of the graph on `stream`.
    ///
    /// # Errors
    /// [`Error::Internal`] on a driver rejection.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        self.dev.make_current()?;
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                let f = c
                    .graph_launch
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: exec and stream handles valid in the current context.
                let r = unsafe { (f)(self.handle, stream.raw()) };
                ensure!(r == 0, Internal, "cuGraphLaunch: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                let f = h
                    .graph_launch
                    .ok_or_else(crate::device::no_graphs_error)?;
                // SAFETY: see above.
                let r = unsafe { (f)(self.handle, stream.raw()) };
                ensure!(r == 0, Internal, "hipGraphLaunch: {}", h.error_string(r));
            }
            _ => unreachable!("one backend is always loaded"),
        }
        Ok(())
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // Best-effort destroy; failure cannot be reported from Drop.
        match (self.dev.gpu().cuda(), self.dev.gpu().hip()) {
            (Some(c), _) => {
                if let Some(f) = c.graph_exec_destroy {
                    // SAFETY: handle valid until this drop.
                    unsafe { (f)(self.handle) };
                }
            }
            (_, Some(h)) => {
                if let Some(f) = h.graph_exec_destroy {
                    // SAFETY: see above.
                    unsafe { (f)(self.handle) };
                }
            }
            _ => unreachable!("one backend is always loaded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_support_probe_is_callable_without_gpu() {
        // Must not panic on a GPU-less machine.
        for b in crate::Gpu::probe_backends() {
            if let Ok(gpu) = crate::Gpu::new(b) {
                if let Ok(dev) = gpu.device(0) {
                    let _ = graph_supported(&dev);
                }
            }
        }
    }
}
