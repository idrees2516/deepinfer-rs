//! CUDA Driver API (12.x/13.x) subset, resolved at runtime via dlopen.
//!
//! Declarations transcribed from `cuda.h` (Driver API). We use the driver API,
//! not the runtime API, because that is what module loading + JIT requires and
//! it is the smaller, stabler surface. Only the entry points DeepInfer needs
//! are declared.

use std::ffi::c_void;

use crate::dl::Lib;
use deepinfer_core::Result;

pub(crate) type CuResult = u32;
/// Opaque CUcontext / CUstream / CUmodule / CUfunction handles.
pub(crate) type Handle = *mut c_void;
/// Device pointer (CUdeviceptr is an unsigned long long on all supported ABIs).
pub(crate) type DevicePtr = u64;

// cuDeviceGetAttribute selectors (cuda.h values).
pub(crate) const ATTR_MAX_THREADS_PER_BLOCK: i32 = 1;
pub(crate) const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
pub(crate) const ATTR_CC_MAJOR: i32 = 75;
pub(crate) const ATTR_CC_MINOR: i32 = 76;
pub(crate) const ATTR_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;

// cuInit flags must be 0.
pub(crate) const CU_INIT_FLAGS: u32 = 0;

pub(crate) type FnCuInit = unsafe extern "C" fn(u32) -> CuResult;
pub(crate) type FnCuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const std::ffi::c_char) -> CuResult;
pub(crate) type FnCuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
pub(crate) type FnCuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> CuResult;
pub(crate) type FnCuDeviceGetName = unsafe extern "C" fn(*mut std::ffi::c_char, i32, i32) -> CuResult;
pub(crate) type FnCuDeviceTotalMem = unsafe extern "C" fn(*mut usize, i32) -> CuResult;
pub(crate) type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> CuResult;
pub(crate) type FnCuDeviceCanAccessPeer = unsafe extern "C" fn(*mut i32, i32, i32) -> CuResult;
pub(crate) type FnCuPrimaryCtxRetain = unsafe extern "C" fn(*mut Handle, i32) -> CuResult;
pub(crate) type FnCuPrimaryCtxRelease = unsafe extern "C" fn(i32) -> CuResult;
pub(crate) type FnCuCtxSetCurrent = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuStreamCreate = unsafe extern "C" fn(*mut Handle, u32) -> CuResult;
pub(crate) type FnCuStreamDestroy = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuStreamSynchronize = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuMemAlloc = unsafe extern "C" fn(*mut DevicePtr, usize) -> CuResult;
pub(crate) type FnCuMemFree = unsafe extern "C" fn(DevicePtr) -> CuResult;
pub(crate) type FnCuMemcpyHtoD = unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> CuResult;
pub(crate) type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> CuResult;
pub(crate) type FnCuMemcpyDtoD = unsafe extern "C" fn(DevicePtr, DevicePtr, usize) -> CuResult;
pub(crate) type FnCuMemcpyHtoDAsync =
    unsafe extern "C" fn(DevicePtr, *const c_void, usize, Handle) -> CuResult;
pub(crate) type FnCuMemcpyDtoHAsync =
    unsafe extern "C" fn(*mut c_void, DevicePtr, usize, Handle) -> CuResult;
pub(crate) type FnCuMemcpyPeerAsync =
    unsafe extern "C" fn(DevicePtr, Handle, DevicePtr, Handle, usize, Handle) -> CuResult;
pub(crate) type FnCuCtxEnablePeerAccess = unsafe extern "C" fn(Handle, u32) -> CuResult;
pub(crate) type FnCuModuleLoadData = unsafe extern "C" fn(*mut Handle, *const c_void) -> CuResult;
pub(crate) type FnCuModuleUnload = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut Handle, Handle, *const std::ffi::c_char) -> CuResult;
pub(crate) type FnCuLaunchKernel = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut c_void,
) -> CuResult;

// ---- Events (cross-stream / cross-device ordering) -------------------------
pub(crate) type FnCuEventCreate = unsafe extern "C" fn(*mut Handle, u32) -> CuResult;
pub(crate) type FnCuEventRecord = unsafe extern "C" fn(Handle, Handle) -> CuResult;
pub(crate) type FnCuEventSynchronize = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuEventDestroy = unsafe extern "C" fn(Handle) -> CuResult;
pub(crate) type FnCuStreamWaitEvent = unsafe extern "C" fn(Handle, Handle, u32) -> CuResult;

// ---- Async device-to-device copy -------------------------------------------
pub(crate) type FnCuMemcpyDtoDAsync =
    unsafe extern "C" fn(DevicePtr, DevicePtr, usize, Handle) -> CuResult;

// ---- CUDA Graphs (stream capture) ------------------------------------------
pub(crate) type FnCuGraphDestroy = unsafe extern "C" fn(Handle) -> CuResult;
/// `CUstreamCaptureMode`: 0 = GLOBAL, 1 = THREAD_LOCAL, 2 = RELAXED.
pub(crate) type FnCuStreamBeginCapture = unsafe extern "C" fn(Handle, i32) -> CuResult;
pub(crate) type FnCuStreamEndCapture = unsafe extern "C" fn(Handle, *mut Handle) -> CuResult;
/// Present since CUDA 11.4; the arity-stable instantiation entry point
/// (the plain `cuGraphInstantiate` symbol changed arity across versions).
pub(crate) type FnCuGraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut Handle, Handle, u64) -> CuResult;
pub(crate) type FnCuGraphLaunch = unsafe extern "C" fn(Handle, Handle) -> CuResult;
pub(crate) type FnCuGraphExecDestroy = unsafe extern "C" fn(Handle) -> CuResult;

/// `CU_EVENT_DEFAULT`.
pub(crate) const CU_EVENT_DEFAULT: u32 = 0;
/// `CU_STREAM_CAPTURE_MODE_RELAXED` — other threads may use the driver
/// while this thread captures (the mode vLLM-class runtimes use).
pub(crate) const CU_STREAM_CAPTURE_MODE_RELAXED: i32 = 2;

/// All CUDA driver entry points we need, bound to one open `libcuda`.
pub struct Cuda {
    #[allow(dead_code)]
    pub(crate) lib: Lib,
    pub(crate) init: FnCuInit,
    pub(crate) get_error_string: FnCuGetErrorString,
    pub(crate) device_get_count: FnCuDeviceGetCount,
    pub(crate) device_get: FnCuDeviceGet,
    pub(crate) device_get_name: FnCuDeviceGetName,
    pub(crate) device_total_mem: FnCuDeviceTotalMem,
    pub(crate) device_get_attribute: FnCuDeviceGetAttribute,
    pub(crate) device_can_access_peer: FnCuDeviceCanAccessPeer,
    pub(crate) primary_ctx_retain: FnCuPrimaryCtxRetain,
    #[allow(dead_code)]
    pub(crate) primary_ctx_release: FnCuPrimaryCtxRelease,
    pub(crate) ctx_set_current: FnCuCtxSetCurrent,
    pub(crate) stream_create: FnCuStreamCreate,
    pub(crate) stream_destroy: FnCuStreamDestroy,
    pub(crate) stream_synchronize: FnCuStreamSynchronize,
    pub(crate) mem_alloc: FnCuMemAlloc,
    pub(crate) mem_free: FnCuMemFree,
    pub(crate) memcpy_htod: FnCuMemcpyHtoD,
    pub(crate) memcpy_dtoh: FnCuMemcpyDtoH,
    pub(crate) memcpy_dtod: FnCuMemcpyDtoD,
    pub(crate) memcpy_htod_async: FnCuMemcpyHtoDAsync,
    pub(crate) memcpy_dtoh_async: FnCuMemcpyDtoHAsync,
    pub(crate) memcpy_peer_async: FnCuMemcpyPeerAsync,
    pub(crate) ctx_enable_peer_access: FnCuCtxEnablePeerAccess,
    pub(crate) module_load_data: FnCuModuleLoadData,
    pub(crate) module_unload: FnCuModuleUnload,
    pub(crate) module_get_function: FnCuModuleGetFunction,
    pub(crate) launch_kernel: FnCuLaunchKernel,
    // Optional entries: events (driver >= 8-ish, always present in practice)
    // and graphs (CUDA >= 11.4). Resolved with sym_opt so older stacks still
    // load; the safe wrappers report a clean error when absent.
    pub(crate) event_create: Option<FnCuEventCreate>,
    pub(crate) event_record: Option<FnCuEventRecord>,
    pub(crate) event_synchronize: Option<FnCuEventSynchronize>,
    pub(crate) event_destroy: Option<FnCuEventDestroy>,
    pub(crate) stream_wait_event: Option<FnCuStreamWaitEvent>,
    pub(crate) memcpy_dtod_async: Option<FnCuMemcpyDtoDAsync>,
    pub(crate) graph_destroy: Option<FnCuGraphDestroy>,
    pub(crate) stream_begin_capture: Option<FnCuStreamBeginCapture>,
    pub(crate) stream_end_capture: Option<FnCuStreamEndCapture>,
    pub(crate) graph_instantiate: Option<FnCuGraphInstantiateWithFlags>,
    pub(crate) graph_launch: Option<FnCuGraphLaunch>,
    pub(crate) graph_exec_destroy: Option<FnCuGraphExecDestroy>,
}

impl Cuda {
    /// Load `libcuda` and resolve every symbol we depend on.
    pub fn load() -> Result<Cuda> {
        let lib = Lib::open(&["libcuda.so.1", "libcuda.so"])?;
        // SAFETY: all types below are transcribed from cuda.h; see module docs.
        unsafe {
            Ok(Cuda {
                init: lib.sym("cuInit")?,
                get_error_string: lib.sym("cuGetErrorString")?,
                device_get_count: lib.sym("cuDeviceGetCount")?,
                device_get: lib.sym("cuDeviceGet")?,
                device_get_name: lib.sym("cuDeviceGetName")?,
                device_total_mem: lib.sym("cuDeviceTotalMem_v2")?,
                device_get_attribute: lib.sym("cuDeviceGetAttribute")?,
                device_can_access_peer: lib.sym("cuDeviceCanAccessPeer")?,
                primary_ctx_retain: lib.sym("cuDevicePrimaryCtxRetain")?,
                primary_ctx_release: lib.sym("cuDevicePrimaryCtxRelease")?,
                ctx_set_current: lib.sym("cuCtxSetCurrent")?,
                stream_create: lib.sym("cuStreamCreate")?,
                stream_destroy: lib.sym("cuStreamDestroy")?,
                stream_synchronize: lib.sym("cuStreamSynchronize")?,
                mem_alloc: lib.sym("cuMemAlloc_v2")?,
                mem_free: lib.sym("cuMemFree_v2")?,
                memcpy_htod: lib.sym("cuMemcpyHtoD_v2")?,
                memcpy_dtoh: lib.sym("cuMemcpyDtoH_v2")?,
                memcpy_dtod: lib.sym("cuMemcpyDtoD_v2")?,
                memcpy_htod_async: lib.sym("cuMemcpyHtoDAsync_v2")?,
                memcpy_dtoh_async: lib.sym("cuMemcpyDtoHAsync_v2")?,
                memcpy_peer_async: lib.sym("cuMemcpyPeerAsync")?,
                ctx_enable_peer_access: lib.sym("cuCtxEnablePeerAccess")?,
                module_load_data: lib.sym("cuModuleLoadData")?,
                module_unload: lib.sym("cuModuleUnload")?,
                module_get_function: lib.sym("cuModuleGetFunction")?,
                launch_kernel: lib.sym("cuLaunchKernel")?,
                event_create: lib.sym_opt("cuEventCreate")?,
                event_record: lib.sym_opt("cuEventRecord")?,
                event_synchronize: lib.sym_opt("cuEventSynchronize")?,
                event_destroy: lib.sym_opt("cuEventDestroy")?,
                stream_wait_event: lib.sym_opt("cuStreamWaitEvent")?,
                memcpy_dtod_async: lib.sym_opt("cuMemcpyDtoDAsync_v2")?,
                graph_destroy: lib.sym_opt("cuGraphDestroy")?,
                stream_begin_capture: lib.sym_opt("cuStreamBeginCapture")?,
                stream_end_capture: lib.sym_opt("cuStreamEndCapture")?,
                graph_instantiate: lib.sym_opt("cuGraphInstantiateWithFlags")?,
                graph_launch: lib.sym_opt("cuGraphLaunch")?,
                graph_exec_destroy: lib.sym_opt("cuGraphExecDestroy")?,
                lib,
            })
        }
    }

    /// Human-readable text for a CUresult code.
    pub fn error_string(&self, code: CuResult) -> String {
        let mut p: *const std::ffi::c_char = std::ptr::null();
        // SAFETY: out-pointer is valid; result only read on success.
        let r = unsafe { (self.get_error_string)(code, &mut p) };
        if r == 0 && !p.is_null() {
            // SAFETY: vendor contract: points to a static NUL-terminated string.
            unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
        } else {
            format!("CUDA error {code}")
        }
    }
}
