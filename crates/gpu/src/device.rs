//! Unified GPU device abstraction over the CUDA driver API and HIP.
//!
//! One API, two backends, selected at runtime:
//! * **NVIDIA** — GB200 NVL72 / B200 (`sm_100`), GB300 / B300 (`sm_103`),
//!   plus any SM80+ GPU (Hopper `sm_90` etc.) for development machines.
//! * **AMD** — Instinct MI335X and the MI300/MI350 families, with the compile
//!   target taken from the device's own `gcnArchName` (e.g. `gfx942`,
//!   `gfx950`), so the same binary serves whatever ROCm GPU it meets.
//!
//! All raw handles are owned types with `Drop`; contexts are process-lifetime
//! (CUDA primary contexts are intentionally retained — see `Device` docs).

use std::ffi::c_void;
use std::sync::Arc;

use deepinfer_core::{ensure, Error, Result};

use crate::ffi::{cuda, hip, hiprtc, nvrtc};
use crate::dl;

/// Which vendor stack is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA driver + NVRTC.
    Cuda,
    /// AMD HIP runtime + hipRTC.
    Hip,
}

impl Backend {
    /// Short name used in cache keys and logs.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
        }
    }
}

/// Detected device architecture (drives JIT flags).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuArch {
    /// NVIDIA compute capability, e.g. (10, 0) for B200/GB200 NVL72 or
    /// (10, 3) for B300/GB300 Blackwell Ultra.
    Sm(u32, u32),
    /// AMD GCN arch name, e.g. `gfx942` (MI300X/MI325X class) or `gfx950`
    /// (MI350 class). Suffixes after `:` are stripped.
    Gfx(String),
}

impl GpuArch {
    /// Canonical string (e.g. `sm_100`, `gfx942`).
    pub fn as_str(&self) -> String {
        match self {
            GpuArch::Sm(maj, min) => format!("sm_{maj}{min}"),
            GpuArch::Gfx(name) => name.clone(),
        }
    }

    /// NVRTC `--gpu-architecture` flag. We emit plain `sm_XXX` CUBIN targets:
    /// our kernels use portable `mma.sync` (no arch-specific instructions),
    /// so no `a`/`f` suffix is required. Switch to `sm_100a`-style targets
    /// when the tcgen05 path lands (docs/GPU.md "Upgrade paths").
    pub fn nvrtc_flag(&self) -> String {
        match self {
            GpuArch::Sm(maj, min) => format!("--gpu-architecture=sm_{maj}{min}"),
            GpuArch::Gfx(name) => format!("--gpu-architecture={name}"),
        }
    }

    /// Fallback PTX flag when the local NVRTC is older than the device
    /// (e.g. CUDA 12.8 NVRTC meeting a B300): compile `compute_XXX` PTX and
    /// let the driver JIT it.
    pub fn nvrtc_ptx_flag(&self) -> Option<String> {
        match self {
            GpuArch::Sm(maj, min) => Some(format!("--gpu-architecture=compute_{maj}{min}")),
            GpuArch::Gfx(_) => None,
        }
    }

    /// hipRTC compile flag (target from `gcnArchName`).
    pub fn hiprtc_flag(&self) -> String {
        match self {
            GpuArch::Gfx(name) => format!("--gpu-architecture={name}"),
            GpuArch::Sm(..) => unreachable!("hiprtc on NVIDIA arch"),
        }
    }
}

/// Static device properties.
#[derive(Debug, Clone)]
pub struct DeviceProps {
    /// Vendor stack in use.
    pub backend: Backend,
    /// Device product name (e.g. "NVIDIA B200", "AMD Instinct MI335X").
    pub name: String,
    /// Architecture target for JIT.
    pub arch: GpuArch,
    /// SM count (NV) or CU count attribute (AMD); used for grid sizing.
    pub sm_count: u32,
    /// Total device memory in bytes.
    pub total_mem: u64,
    /// Max threads per block (NV; AMD defaults to 1024).
    pub max_threads_per_block: u32,
    /// Max dynamic shared memory per block in bytes (NV; conservative 32 KiB on AMD).
    pub max_shared: u32,
}

struct GpuInner {
    backend: Backend,
    cuda: Option<cuda::Cuda>,
    hip: Option<hip::Hip>,
    nvrtc: Option<nvrtc::Nvrtc>,
    hiprtc: Option<hiprtc::HipRtc>,
}

// SAFETY: all vendor entry points used here are thread-safe (CUDA driver API
// with per-thread current contexts; HIP runtime is thread-safe), and Lib
// handles are Send+Sync by construction (never unloaded).
unsafe impl Send for GpuInner {}
unsafe impl Sync for GpuInner {}

/// A loaded GPU stack: one driver + one RTC compiler for a backend.
#[derive(Clone)]
pub struct Gpu {
    inner: Arc<GpuInner>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("backend", &self.inner.backend)
            .finish()
    }
}

impl Gpu {
    /// Probe which backends have their libraries present (no device needed).
    pub fn probe_backends() -> Vec<Backend> {
        let mut out = Vec::new();
        if dl::probe(&["libcuda.so.1", "libcuda.so"]) {
            out.push(Backend::Cuda);
        }
        if dl::probe(&["libamdhip64.so.6", "libamdhip64.so"]) {
            out.push(Backend::Hip);
        }
        out
    }

    /// Load the driver and RTC compiler for a backend.
    pub fn new(backend: Backend) -> Result<Gpu> {
        match backend {
            Backend::Cuda => {
                let c = cuda::Cuda::load()?;
                // SAFETY: flags must be 0 per cuInit contract.
                let r = unsafe { (c.init)(cuda::CU_INIT_FLAGS) };
                ensure!(r == 0, Internal, "cuInit: {}", c.error_string(r));
                let rtc = nvrtc::Nvrtc::load().ok();
                Ok(Gpu {
                    inner: Arc::new(GpuInner {
                        backend,
                        cuda: Some(c),
                        hip: None,
                        nvrtc: rtc,
                        hiprtc: None,
                    }),
                })
            }
            Backend::Hip => {
                let h = hip::Hip::load()?;
                let rtc = hiprtc::HipRtc::load().ok();
                Ok(Gpu {
                    inner: Arc::new(GpuInner {
                        backend,
                        cuda: None,
                        hip: Some(h),
                        nvrtc: None,
                        hiprtc: rtc,
                    }),
                })
            }
        }
    }

    /// Backend served by this instance.
    pub fn backend(&self) -> Backend {
        self.inner.backend
    }

    /// NVRTC version when the CUDA RTC is available.
    pub fn nvrtc_version(&self) -> Option<(i32, i32)> {
        self.inner.nvrtc.as_ref().map(|n| n.version())
    }

    /// Number of visible devices.
    pub fn device_count(&self) -> Result<usize> {
        let mut n: i32 = 0;
        match (&self.inner.cuda, &self.inner.hip) {
            (Some(c), _) => {
                // SAFETY: valid out-pointer.
                let r = unsafe { (c.device_get_count)(&mut n) };
                ensure!(r == 0, Internal, "cuDeviceGetCount: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                // SAFETY: valid out-pointer.
                let r = unsafe { (h.get_device_count)(&mut n) };
                ensure!(r == 0, Internal, "hipGetDeviceCount: {}", h.error_string(r));
            }
            _ => unreachable!("one backend always loaded"),
        }
        Ok(n.max(0) as usize)
    }

    /// Open a device by ordinal (retains its primary context on CUDA — kept
    /// for the process lifetime, which is the right trade-off for an
    /// inference server and removes a class of use-after-release bugs).
    pub fn device(&self, ordinal: usize) -> Result<Device> {
        let count = self.device_count()?;
        ensure!(ordinal < count, InvalidArgument, "device ordinal out of range");
        match (&self.inner.cuda, &self.inner.hip) {
            (Some(c), _) => {
                let mut dev: i32 = 0;
                // SAFETY: valid out-pointer.
                let r = unsafe { (c.device_get)(&mut dev, ordinal as i32) };
                ensure!(r == 0, Internal, "cuDeviceGet: {}", c.error_string(r));
                let mut name = [0i8; 256];
                // SAFETY: name buffer valid; device handle from driver.
                let r = unsafe { (c.device_get_name)(name.as_mut_ptr(), name.len() as i32, dev) };
                ensure!(r == 0, Internal, "cuDeviceGetName: {}", c.error_string(r));
                let name = i8_to_string(&name);
                let attr = |a: i32| -> Result<i32> {
                    let mut v: i32 = 0;
                    // SAFETY: valid out-pointer; valid device.
                    let r = unsafe { (c.device_get_attribute)(&mut v, a, dev) };
                    ensure!(r == 0, Internal, "cuDeviceGetAttribute: {}", c.error_string(r));
                    Ok(v)
                };
                let maj = attr(cuda::ATTR_CC_MAJOR)? as u32;
                let min = attr(cuda::ATTR_CC_MINOR)? as u32;
                let mut total: usize = 0;
                // SAFETY: valid out-pointer.
                let r = unsafe { (c.device_total_mem)(&mut total, dev) };
                ensure!(r == 0, Internal, "cuDeviceTotalMem: {}", c.error_string(r));
                let mut ctx: cuda::Handle = std::ptr::null_mut();
                // SAFETY: valid out-pointer; retained for process lifetime.
                let r = unsafe { (c.primary_ctx_retain)(&mut ctx, dev) };
                ensure!(r == 0, Internal, "cuDevicePrimaryCtxRetain: {}", c.error_string(r));
                let props = DeviceProps {
                    backend: Backend::Cuda,
                    name,
                    arch: GpuArch::Sm(maj, min),
                    sm_count: attr(cuda::ATTR_MULTIPROCESSOR_COUNT)?.max(1) as u32,
                    total_mem: total as u64,
                    max_threads_per_block: attr(cuda::ATTR_MAX_THREADS_PER_BLOCK)?.max(128) as u32,
                    max_shared: attr(cuda::ATTR_SHARED_MEMORY_PER_BLOCK_OPTIN)?.max(16 * 1024) as u32,
                };
                Ok(Device {
                    gpu: self.clone(),
                    id: dev as i32,
                    ctx,
                    props,
                })
            }
            (_, Some(h)) => {
                // SAFETY: valid out-pointer.
                let r = unsafe { (h.set_device)(ordinal as i32) };
                ensure!(r == 0, Internal, "hipSetDevice: {}", h.error_string(r));
                let mut name = [0i8; 256];
                // SAFETY: name buffer valid.
                let r = unsafe { (h.device_get_name)(name.as_mut_ptr(), name.len() as i32, ordinal as i32) };
                ensure!(r == 0, Internal, "hipDeviceGetName: {}", h.error_string(r));
                let name = i8_to_string(&name);
                let mut free = 0usize;
                let mut total = 0usize;
                // SAFETY: valid out-pointers.
                let r = unsafe { (h.mem_get_info)(&mut free, &mut total) };
                ensure!(r == 0, Internal, "hipMemGetInfo: {}", h.error_string(r));
                // SM/CU count: attribute query mirrors CUDA numbering; fall
                // back to a conservative default if the runtime declines.
                let sm = match {
                    let mut v: i32 = 0;
                    // SAFETY: valid out-pointer.
                    let r = unsafe { (h.device_get_attribute)(&mut v, hip::ATTR_MULTIPROCESSOR_COUNT, ordinal as i32) };
                    if r == 0 { Some(v) } else { None }
                } {
                    Some(v) => v.max(1) as u32,
                    None => 120,
                };
                let arch = amd_arch_from_env_or_scan(h, ordinal as i32);
                let props = DeviceProps {
                    backend: Backend::Hip,
                    name,
                    arch,
                    sm_count: sm,
                    total_mem: total as u64,
                    max_threads_per_block: 1024,
                    max_shared: 32 * 1024,
                };
                Ok(Device {
                    gpu: self.clone(),
                    id: ordinal as i32,
                    ctx: std::ptr::null_mut(),
                    props,
                })
            }
            _ => unreachable!("one backend always loaded"),
        }
    }
}

/// Decode a C `char[N]` buffer into a Rust String.
fn i8_to_string(buf: &[i8]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let bytes: Vec<u8> = buf[..len].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Determine the AMD compile target.
///
/// `DEEPINFER_HIP_ARCH` (e.g. `gfx942`) wins if set; otherwise we call
/// `hipGetDeviceProperties` with a generous zeroed buffer and scan for the
/// `gcnArchName` string (`gfx<digits>`, stopping at `:`-suffixes or NUL).
/// Scanning avoids depending on the exact `hipDeviceProp_t` layout, which has
/// changed across ROCm releases — an ABI-resilience requirement for a
/// dlopen-based stack.
fn amd_arch_from_env_or_scan(h: &hip::Hip, device: i32) -> GpuArch {
    if let Ok(v) = std::env::var("DEEPINFER_HIP_ARCH") {
        let v = v.trim().to_string();
        if v.starts_with("gfx") && v[3..].chars().all(|c| c.is_ascii_digit()) {
            return GpuArch::Gfx(v);
        }
    }
    // Buffer must be at least as large as the largest hipDeviceProp_t we
    // might meet (name + uuid + gcnArchName + scalars; ~1.1 KiB in ROCm 6,
    // allocate 8 KiB for headroom across future versions).
    let mut buf = vec![0u8; 8192];
    // SAFETY: buf is 8 KiB of writable memory; the vendor call writes at most
    // sizeof(hipDeviceProp_t) < 8 KiB bytes into it.
    let r = unsafe { (h.get_device_properties)(buf.as_mut_ptr() as *mut c_void, device) };
    if r != 0 {
        return GpuArch::Gfx("gfx942".to_string());
    }
    // Scan for b"gfx" followed by 3 digits; take the first plausible hit.
    let s = &buf[..buf.len() - 8];
    for i in 0..s.len().saturating_sub(6) {
        if &s[i..i + 3] == b"gfx" {
            let mut j = i + 3;
            while j < s.len() && s[j].is_ascii_digit() && j - i < 12 {
                j += 1;
            }
            if j >= i + 6 {
                let name = String::from_utf8_lossy(&s[i..j]).into_owned();
                return GpuArch::Gfx(name);
            }
        }
    }
    GpuArch::Gfx("gfx942".to_string())
}

/// An open device: owns (retains) its compute context.
///
/// Cloning shares the same underlying (process-lifetime) context and is the
/// intended way to hand the device to long-lived structures.
#[derive(Clone)]
pub struct Device {
    gpu: Gpu,
    id: i32,
    ctx: cuda::Handle,
    props: DeviceProps,
}

// SAFETY: device handles are usable from any thread (CUDA driver API is
// thread-safe; we set the current context before every call).
unsafe impl Send for Device {}
// SAFETY: see above; all methods synchronize internally where required.
unsafe impl Sync for Device {}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("id", &self.id)
            .field("props", &self.props)
            .finish()
    }
}

impl Device {
    /// Static properties of this device.
    pub fn props(&self) -> &DeviceProps {
        &self.props
    }

    /// Backend of this device.
    pub fn backend(&self) -> Backend {
        self.props.backend
    }

    /// NVRTC compiler handle when available (CUDA backend).
    pub(crate) fn nvrtc(&self) -> Option<&nvrtc::Nvrtc> {
        self.gpu.inner.nvrtc.as_ref()
    }

    /// hipRTC compiler handle when available (HIP backend).
    pub(crate) fn hiprtc(&self) -> Option<&hiprtc::HipRtc> {
        self.gpu.inner.hiprtc.as_ref()
    }

    /// Make this device's context current on the calling thread (CUDA) or
    /// select the device (HIP). Called internally before driver calls.
    fn make_current(&self) -> Result<()> {
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: ctx was retained at Device creation and is never released.
                let r = unsafe { (c.ctx_set_current)(self.ctx) };
                ensure!(r == 0, Internal, "cuCtxSetCurrent: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                // SAFETY: ordinal is valid.
                let r = unsafe { (h.set_device)(self.id) };
                ensure!(r == 0, Internal, "hipSetDevice: {}", h.error_string(r));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Create an execution stream.
    pub fn stream(&self) -> Result<Stream> {
        self.make_current()?;
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                let mut s: cuda::Handle = std::ptr::null_mut();
                // SAFETY: out-pointer valid; flags 0 = default stream flags.
                let r = unsafe { (c.stream_create)(&mut s, 0) };
                ensure!(r == 0, Internal, "cuStreamCreate: {}", c.error_string(r));
                Ok(Stream {
                    gpu: self.gpu.clone(),
                    handle: s,
                })
            }
            (_, Some(h)) => {
                let mut s: hip::Handle = std::ptr::null_mut();
                // SAFETY: out-pointer valid.
                let r = unsafe { (h.stream_create)(&mut s) };
                ensure!(r == 0, Internal, "hipStreamCreate: {}", h.error_string(r));
                Ok(Stream {
                    gpu: self.gpu.clone(),
                    handle: s,
                })
            }
            _ => unreachable!(),
        }
    }

    /// Allocate device memory.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        ensure!(bytes > 0, InvalidArgument, "allocation size must be > 0");
        self.make_current()?;
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                let mut p: cuda::DevicePtr = 0;
                // SAFETY: out-pointer valid; size checked above.
                let r = unsafe { (c.mem_alloc)(&mut p, bytes) };
                ensure!(r == 0, Internal, "cuMemAlloc({bytes}): {}", c.error_string(r));
                Ok(Buffer {
                    gpu: self.gpu.clone(),
                    ptr: p,
                    len: bytes,
                })
            }
            (_, Some(h)) => {
                let mut p: hip::Handle = std::ptr::null_mut();
                // SAFETY: out-pointer valid; size checked above.
                let r = unsafe { (h.malloc)(&mut p, bytes) };
                ensure!(r == 0, Internal, "hipMalloc({bytes}): {}", h.error_string(r));
                Ok(Buffer {
                    gpu: self.gpu.clone(),
                    ptr: p as u64,
                    len: bytes,
                })
            }
            _ => unreachable!(),
        }
    }

    /// Allocate and fill device memory from a host slice.
    pub fn alloc_from<T: Copy>(&self, data: &[T]) -> Result<Buffer> {
        let bytes = data.len() * std::mem::size_of::<T>();
        let buf = self.alloc(bytes)?;
        self.write(&buf, data)?;
        Ok(buf)
    }

    /// Upload a host slice into an existing buffer (synchronous).
    pub fn write<T: Copy>(&self, buf: &Buffer, data: &[T]) -> Result<()> {
        let bytes = data.len() * std::mem::size_of::<T>();
        ensure!(bytes <= buf.len, InvalidArgument, "host slice larger than buffer");
        self.make_current()?;
        // SAFETY: data is a valid host buffer of `bytes` readable bytes; the
        // device pointer was allocated with `buf.len >= bytes`.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe {
                (c.memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes)
            },
            (_, Some(h)) => unsafe {
                (h.memcpy_htod)(buf.ptr as *mut c_void, data.as_ptr() as *const c_void, bytes)
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpy HtoD: {}", self.err_str(r));
        Ok(())
    }

    /// Download a buffer into a host slice (synchronous).
    pub fn read<T: Copy>(&self, buf: &Buffer, out: &mut [T]) -> Result<()> {
        let bytes = out.len() * std::mem::size_of::<T>();
        ensure!(bytes <= buf.len, InvalidArgument, "host slice larger than buffer");
        self.make_current()?;
        // SAFETY: out is a valid writable host buffer; device pointer valid.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe { (c.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, buf.ptr, bytes) },
            (_, Some(h)) => unsafe {
                (h.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, buf.ptr as *mut c_void, bytes)
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpy DtoH: {}", self.err_str(r));
        Ok(())
    }

    /// Device-to-device copy within the same device.
    pub fn copy_within(&self, dst: &Buffer, src: &Buffer) -> Result<()> {
        let bytes = dst.len.min(src.len);
        ensure!(bytes > 0, InvalidArgument, "empty copy");
        self.make_current()?;
        // SAFETY: both pointers valid for `bytes` on the current device.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe { (c.memcpy_dtod)(dst.ptr, src.ptr, bytes) },
            (_, Some(h)) => unsafe {
                (h.memcpy_dtod)(dst.ptr as *mut c_void, src.ptr as *mut c_void, bytes)
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpy DtoD: {}", self.err_str(r));
        Ok(())
    }

    /// Enable peer access from this device to `peer` (both must be from the
    /// same backend). Safe to call repeatedly; already-enabled is a no-op.
    pub fn enable_peer_access(&self, peer: &Device) -> Result<()> {
        ensure!(self.backend() == peer.backend(), InvalidArgument, "peer devices must share a backend");
        if self.id == peer.id {
            return Ok(());
        }
        self.make_current()?;
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                let mut ok: i32 = 0;
                // SAFETY: valid device handles.
                let r = unsafe { (c.device_can_access_peer)(&mut ok, self.id, peer.id) };
                ensure!(r == 0, Internal, "cuDeviceCanAccessPeer: {}", c.error_string(r));
                if ok == 0 {
                    return Err(Error::internal("peer access not supported between devices"));
                }
                // SAFETY: peer context is retained (Device creation).
                let r = unsafe { (c.ctx_enable_peer_access)(peer.ctx, 0) };
                // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 708
                ensure!(
                    r == 0 || r == 708,
                    Internal,
                    "cuCtxEnablePeerAccess: {}",
                    c.error_string(r)
                );
                Ok(())
            }
            (_, Some(h)) => {
                let mut ok: i32 = 0;
                // SAFETY: valid ordinals.
                let r = unsafe { (h.device_can_access_peer)(&mut ok, self.id, peer.id) };
                ensure!(r == 0, Internal, "hipDeviceCanAccessPeer: {}", h.error_string(r));
                if ok == 0 {
                    return Err(Error::internal("peer access not supported between devices"));
                }
                // SAFETY: valid ordinal; flags 0.
                let r = unsafe { (h.enable_peer_access)(peer.id, 0) };
                // hipErrorPeerAccessAlreadyEnabled = 14? treat 0 and "already" as ok
                ensure!(
                    r == 0 || r == 14,
                    Internal,
                    "hipEnablePeerAccess: {}",
                    h.error_string(r)
                );
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    /// Asynchronous peer copy `src` (on `src_dev`) -> `dst` (on this device).
    /// Both devices must have peer access enabled.
    pub fn memcpy_peer_async(
        &self,
        dst: &Buffer,
        src_dev: &Device,
        src: &Buffer,
        stream: &Stream,
    ) -> Result<()> {
        let bytes = dst.len.min(src.len);
        ensure!(bytes > 0, InvalidArgument, "empty peer copy");
        ensure!(self.backend() == src_dev.backend(), InvalidArgument, "peer copy across backends");
        self.make_current()?;
        // SAFETY: pointers valid on their devices for `bytes`; peer access
        // enabled by the caller via enable_peer_access.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe {
                (c.memcpy_peer_async)(dst.ptr, src_dev.ctx, src.ptr, self.ctx, bytes, stream.handle)
            },
            (_, Some(h)) => unsafe {
                (h.memcpy_peer_async)(
                    dst.ptr as *mut c_void,
                    self.id,
                    src.ptr as *const c_void,
                    src_dev.id,
                    bytes,
                    stream.handle,
                )
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpyPeerAsync: {}", self.err_str(r));
        Ok(())
    }

    /// Load a compiled code object (CUBIN/PTX on CUDA, HSACO on HIP) as a
    /// module of kernels.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        self.make_current()?;
        // Keep the image alive for the duration of the load call by handing
        // the driver a pointer into a stable copy.
        let image = image.to_vec();
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                let mut m: cuda::Handle = std::ptr::null_mut();
                // SAFETY: image is a valid CUBIN/PTX image of image.len bytes.
                let r = unsafe { (c.module_load_data)(&mut m, image.as_ptr() as *const c_void) };
                ensure!(r == 0, Internal, "cuModuleLoadData: {}", c.error_string(r));
                Ok(Module {
                    gpu: self.gpu.clone(),
                    handle: m,
                })
            }
            (_, Some(h)) => {
                let mut m: hip::Handle = std::ptr::null_mut();
                // SAFETY: image is a valid code object; zero JIT options.
                let r = unsafe {
                    (h.module_load_data_ex)(&mut m, image.as_ptr() as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut())
                };
                ensure!(r == 0, Internal, "hipModuleLoadDataEx: {}", h.error_string(r));
                Ok(Module {
                    gpu: self.gpu.clone(),
                    handle: m,
                })
            }
            _ => unreachable!(),
        }
    }

    fn err_str(&self, code: u32) -> String {
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => c.error_string(code),
            (_, Some(h)) => h.error_string(code),
            _ => unreachable!(),
        }
    }
}

/// An execution stream. Destroyed on drop.
pub struct Stream {
    gpu: Gpu,
    handle: cuda::Handle,
}

// SAFETY: streams are thread-safe execution queues in both vendors' APIs.
unsafe impl Send for Stream {}
// SAFETY: see above; ordering between streams is the caller's concern.
unsafe impl Sync for Stream {}

impl Stream {
    /// Block until all work queued on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: stream handle valid.
                let r = unsafe { (c.stream_synchronize)(self.handle) };
                ensure!(r == 0, Internal, "cuStreamSynchronize: {}", c.error_string(r));
            }
            (_, Some(h)) => {
                // SAFETY: stream handle valid.
                let r = unsafe { (h.stream_synchronize)(self.handle) };
                ensure!(r == 0, Internal, "hipStreamSynchronize: {}", h.error_string(r));
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Asynchronous H2D upload on this stream.
    pub fn write_async<T: Copy>(&self, dev: &Device, buf: &Buffer, data: &[T]) -> Result<()> {
        let bytes = data.len() * std::mem::size_of::<T>();
        ensure!(bytes <= buf.len, InvalidArgument, "host slice larger than buffer");
        dev.make_current()?;
        // SAFETY: host slice readable for `bytes`; device pointer valid.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe {
                (c.memcpy_htod_async)(buf.ptr, data.as_ptr() as *const c_void, bytes, self.handle)
            },
            (_, Some(h)) => unsafe {
                (h.memcpy_htod_async)(
                    buf.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    bytes,
                    self.handle,
                )
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpyHtoDAsync: {}", self.err_str(r));
        Ok(())
    }

    /// Asynchronous D2H download on this stream.
    pub fn read_async<T: Copy>(&self, dev: &Device, buf: &Buffer, out: &mut [T]) -> Result<()> {
        let bytes = out.len() * std::mem::size_of::<T>();
        ensure!(bytes <= buf.len, InvalidArgument, "host slice larger than buffer");
        dev.make_current()?;
        // SAFETY: host slice writable for `bytes`; device pointer valid.
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => unsafe {
                (c.memcpy_dtoh_async)(out.as_mut_ptr() as *mut c_void, buf.ptr, bytes, self.handle)
            },
            (_, Some(h)) => unsafe {
                (h.memcpy_dtoh_async)(
                    out.as_mut_ptr() as *mut c_void,
                    buf.ptr as *mut c_void,
                    bytes,
                    self.handle,
                )
            },
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "memcpyDtoHAsync: {}", self.err_str(r));
        Ok(())
    }

    fn err_str(&self, code: u32) -> String {
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => c.error_string(code),
            (_, Some(h)) => h.error_string(code),
            _ => unreachable!(),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Best-effort destroy; leaks only if the vendor call fails, which we
        // cannot report from Drop. Not unloading libs keeps fn pointers valid.
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: handle valid until this drop; no pending work may
                // exist — callers must synchronize before dropping (documented).
                unsafe { (c.stream_destroy)(self.handle) };
            }
            (_, Some(h)) => {
                // SAFETY: see above.
                unsafe { (h.stream_destroy)(self.handle) };
            }
            _ => unreachable!(),
        }
    }
}

/// Device memory allocation. Freed on drop.
pub struct Buffer {
    gpu: Gpu,
    ptr: u64,
    len: usize,
}

// SAFETY: device pointers are plain addresses, valid process-wide once
// allocated; the vendor allocator is thread-safe.
unsafe impl Send for Buffer {}
// SAFETY: see above.
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Device address (CUdeviceptr / hip device pointer).
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false (zero-length allocations are rejected).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Best-effort free (see Stream::drop note).
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: pointer was allocated via cuMemAlloc and not freed yet.
                unsafe { (c.mem_free)(self.ptr) };
            }
            (_, Some(h)) => {
                // SAFETY: pointer was allocated via hipMalloc and not freed yet.
                unsafe { (h.free)(self.ptr as *mut c_void) };
            }
            _ => unreachable!(),
        }
    }
}

/// A loaded code object.
pub struct Module {
    gpu: Gpu,
    handle: cuda::Handle,
}

// SAFETY: module handles are thread-safe in both APIs.
unsafe impl Send for Module {}
// SAFETY: see above.
unsafe impl Sync for Module {}

impl Module {
    /// Look up a `extern "C"` kernel by name.
    pub fn function(&self, name: &str) -> Result<Function> {
        let cname =
            std::ffi::CString::new(name).map_err(|_| Error::internal("NUL in kernel name"))?;
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                let mut f: cuda::Handle = std::ptr::null_mut();
                // SAFETY: module handle valid; name NUL-terminated.
                let r = unsafe { (c.module_get_function)(&mut f, self.handle, cname.as_ptr()) };
                ensure!(r == 0, Internal, "cuModuleGetFunction({name}): {}", c.error_string(r));
                Ok(Function {
                    gpu: self.gpu.clone(),
                    handle: f,
                })
            }
            (_, Some(h)) => {
                let mut f: hip::Handle = std::ptr::null_mut();
                // SAFETY: module handle valid; name NUL-terminated.
                let r = unsafe { (h.module_get_function)(&mut f, self.handle, cname.as_ptr()) };
                ensure!(r == 0, Internal, "hipModuleGetFunction({name}): {}", h.error_string(r));
                Ok(Function {
                    gpu: self.gpu.clone(),
                    handle: f,
                })
            }
            _ => unreachable!(),
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // Best-effort unload.
        match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: handle valid; all Functions taken from it are
                // reference-stable after unload per driver contract.
                unsafe { (c.module_unload)(self.handle) };
            }
            (_, Some(h)) => {
                // SAFETY: see above.
                unsafe { (h.module_unload)(self.handle) };
            }
            _ => unreachable!(),
        }
    }
}

/// A kernel entry point from a [`Module`].
#[derive(Clone)]
pub struct Function {
    gpu: Gpu,
    handle: cuda::Handle,
}

// SAFETY: function handles are thread-safe.
unsafe impl Send for Function {}
// SAFETY: see above.
unsafe impl Sync for Function {}

/// Launch geometry.
#[derive(Debug, Clone, Copy)]
pub struct Launch {
    /// Grid dimensions (blocks).
    pub grid: (u32, u32, u32),
    /// Block dimensions (threads).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory bytes.
    pub shared: u32,
}

impl Function {
    /// Launch the kernel with packed arguments on `stream`.
    pub fn launch(&self, dev: &Device, launch: Launch, args: &KernelArgs, stream: &Stream)
        -> Result<()> {
        dev.make_current()?;
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.offsets.len());
        for &off in &args.offsets {
            // SAFETY: offsets are within the args blob; the blob outlives the
            // launch call (the driver copies parameters synchronously).
            ptrs.push(unsafe { args.blob.as_ptr().add(off) as *mut u8 });
        }
        let r = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => {
                // SAFETY: function handle valid; grid/block dims validated by
                // the driver; kernelParams is an array of pointers to argument
                // values, each valid for the duration of the call.
                unsafe {
                    (c.launch_kernel)(
                        self.handle,
                        launch.grid.0,
                        launch.grid.1,
                        launch.grid.2,
                        launch.block.0,
                        launch.block.1,
                        launch.block.2,
                        launch.shared,
                        stream.handle,
                        ptrs.as_mut_ptr() as *mut *mut c_void,
                        std::ptr::null_mut(),
                    )
                }
            }
            (_, Some(h)) => {
                // SAFETY: see above (hipModuleLaunchKernel has the same shape).
                unsafe {
                    (h.module_launch_kernel)(
                        self.handle,
                        launch.grid.0,
                        launch.grid.1,
                        launch.grid.2,
                        launch.block.0,
                        launch.block.1,
                        launch.block.2,
                        launch.shared,
                        stream.handle,
                        ptrs.as_mut_ptr() as *mut *mut c_void,
                        std::ptr::null_mut(),
                    )
                }
            }
            _ => unreachable!(),
        };
        let err_text = match (&self.gpu.inner.cuda, &self.gpu.inner.hip) {
            (Some(c), _) => c.error_string(r),
            (_, Some(h)) => h.error_string(r),
            _ => unreachable!(),
        };
        ensure!(r == 0, Internal, "kernel launch: {err_text}");
        Ok(())
    }
}

/// Builder for kernel argument blobs (the `void** kernelParams` payload).
///
/// Push order must match the kernel's C parameter order exactly. Scalars are
/// naturally aligned inside the blob; device pointers are pushed as 8-byte
/// addresses.
#[derive(Default, Clone)]
pub struct KernelArgs {
    blob: Vec<u8>,
    offsets: Vec<usize>,
}

impl KernelArgs {
    /// Empty argument list.
    pub fn new() -> Self {
        KernelArgs::default()
    }

    fn push_aligned(&mut self, bytes: &[u8], align: usize) {
        while self.blob.len() % align != 0 {
            self.blob.push(0);
        }
        self.offsets.push(self.blob.len());
        self.blob.extend_from_slice(bytes);
    }

    /// `u32` parameter.
    pub fn u32(mut self, v: u32) -> Self {
        self.push_aligned(&v.to_ne_bytes(), 4);
        self
    }

    /// `int` parameter.
    pub fn i32(mut self, v: i32) -> Self {
        self.push_aligned(&v.to_ne_bytes(), 4);
        self
    }

    /// `size_t` / `unsigned long long` parameter.
    pub fn u64(mut self, v: u64) -> Self {
        self.push_aligned(&v.to_ne_bytes(), 8);
        self
    }

    /// `size_t` parameter from a Rust usize.
    pub fn usize(self, v: usize) -> Self {
        self.u64(v as u64)
    }

    /// `float` parameter.
    pub fn f32(mut self, v: f32) -> Self {
        self.push_aligned(&v.to_ne_bytes(), 4);
        self
    }

    /// `double` parameter.
    pub fn f64(mut self, v: f64) -> Self {
        self.push_aligned(&v.to_ne_bytes(), 8);
        self
    }

    /// Device pointer parameter from a [`Buffer`].
    pub fn ptr(self, b: &Buffer) -> Self {
        self.u64(b.ptr())
    }

    /// Device pointer parameter from a raw address.
    pub fn raw_ptr(self, p: u64) -> Self {
        self.u64(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_alignment_and_order() {
        let a = KernelArgs::new().u32(1).u64(2).f32(3.5).i32(-1);
        // u32 at 0, u64 aligned to 8, f32 at 16, i32 at 20.
        assert_eq!(a.offsets, vec![0, 8, 16, 20]);
        assert_eq!(a.blob.len(), 24);
    }

    #[test]
    fn arch_flags() {
        assert_eq!(
            GpuArch::Sm(10, 0).nvrtc_flag(),
            "--gpu-architecture=sm_100"
        );
        assert_eq!(
            GpuArch::Sm(10, 3).nvrtc_flag(),
            "--gpu-architecture=sm_103"
        );
        assert_eq!(
            GpuArch::Sm(10, 3).nvrtc_ptx_flag().unwrap(),
            "--gpu-architecture=compute_103"
        );
        assert_eq!(
            GpuArch::Gfx("gfx942".into()).hiprtc_flag(),
            "--gpu-architecture=gfx942"
        );
        assert_eq!(GpuArch::Sm(10, 0).as_str(), "sm_100");
        assert_eq!(GpuArch::Gfx("gfx950".into()).as_str(), "gfx950");
    }

    #[test]
    fn probe_backends_is_callable_without_gpu() {
        // Must not panic on a GPU-less machine; may return empty.
        let _ = Gpu::probe_backends();
    }

    #[test]
    fn cuda_init_without_driver_is_graceful() {
        if dl::probe(&["libcuda.so.1", "libcuda.so"]) {
            return; // machine has a driver; nothing to assert
        }
        assert!(Gpu::new(Backend::Cuda).is_err());
    }
}
