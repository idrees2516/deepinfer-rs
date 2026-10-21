//! HIP runtime FFI, transcribed from `hip/hip_runtime_api.h` (ROCm 5.x–7.x).
//!
//! We use the HIP *module* API (hipModuleLoadDataEx / hipModuleLaunchKernel)
//! rather than hipLaunchKernel with C++ templates, because it accepts a
//! code object produced by hipRTC and takes plain function pointers — exactly
//! parallel to the CUDA driver path.
//!
//! One deliberate ABI-resilience choice: `hipGetDeviceProperties` writes a
//! `hipDeviceProp_t` whose layout has changed across ROCm versions. Instead of
//! transcribing the struct, callers pass a generously sized zeroed buffer and
//! scan it for the `gfx...` string of `gcnArchName` (see `device.rs`). The
//! numeric enum for `hipDeviceGetAttribute` mirrors CUDA's numbering for the
//! attributes we use.

use std::ffi::c_void;

use crate::dl::Lib;
use deepinfer_core::Result;

pub(crate) type HipResult = u32;
pub(crate) type Handle = *mut c_void;

// hipDeviceAttribute_t values (CUDA-compatible numbering for these entries).
pub(crate) const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;

// hipMemcpyKind — not needed; we use the explicit named entry points.

pub(crate) type FnGetErrorString = unsafe extern "C" fn(HipResult) -> *const std::ffi::c_char;
pub(crate) type FnGetDeviceCount = unsafe extern "C" fn(*mut i32) -> HipResult;
pub(crate) type FnSetDevice = unsafe extern "C" fn(i32) -> HipResult;
pub(crate) type FnDeviceGetName = unsafe extern "C" fn(*mut std::ffi::c_char, i32, i32) -> HipResult;
pub(crate) type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipResult;
pub(crate) type FnDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> HipResult;
/// Opaque-buffer form of hipGetDeviceProperties(hipDeviceProp_t*, int).
pub(crate) type FnGetDeviceProperties = unsafe extern "C" fn(*mut c_void, i32) -> HipResult;
pub(crate) type FnStreamCreate = unsafe extern "C" fn(*mut Handle) -> HipResult;
pub(crate) type FnStreamDestroy = unsafe extern "C" fn(Handle) -> HipResult;
pub(crate) type FnStreamSynchronize = unsafe extern "C" fn(Handle) -> HipResult;
pub(crate) type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipResult;
pub(crate) type FnFree = unsafe extern "C" fn(*mut c_void) -> HipResult;
pub(crate) type FnMemcpyHtoD = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> HipResult;
pub(crate) type FnMemcpyDtoH = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> HipResult;
pub(crate) type FnMemcpyDtoD = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> HipResult;
pub(crate) type FnMemcpyHtoDAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, Handle) -> HipResult;
pub(crate) type FnMemcpyDtoHAsync =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, Handle) -> HipResult;
pub(crate) type FnMemcpyPeerAsync =
    unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, usize, Handle) -> HipResult;
pub(crate) type FnDeviceCanAccessPeer = unsafe extern "C" fn(*mut i32, i32, i32) -> HipResult;
pub(crate) type FnEnablePeerAccess = unsafe extern "C" fn(i32, u32) -> HipResult;
pub(crate) type FnModuleLoadDataEx = unsafe extern "C" fn(
    *mut Handle,
    *const c_void,
    u32,
    *mut i32,       // hipJitOption*
    *mut *mut c_void, // option values
) -> HipResult;
pub(crate) type FnModuleUnload = unsafe extern "C" fn(Handle) -> HipResult;
pub(crate) type FnModuleGetFunction =
    unsafe extern "C" fn(*mut Handle, Handle, *const std::ffi::c_char) -> HipResult;
pub(crate) type FnModuleLaunchKernel = unsafe extern "C" fn(
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
    *mut *mut c_void,
) -> HipResult;

/// HIP runtime entry points bound to one open `libamdhip64`.
pub struct Hip {
    #[allow(dead_code)]
    lib: Lib,
    pub(crate) get_error_string: FnGetErrorString,
    pub(crate) get_device_count: FnGetDeviceCount,
    pub(crate) set_device: FnSetDevice,
    pub(crate) device_get_name: FnDeviceGetName,
    pub(crate) mem_get_info: FnMemGetInfo,
    pub(crate) device_get_attribute: FnDeviceGetAttribute,
    pub(crate) get_device_properties: FnGetDeviceProperties,
    pub(crate) stream_create: FnStreamCreate,
    pub(crate) stream_destroy: FnStreamDestroy,
    pub(crate) stream_synchronize: FnStreamSynchronize,
    pub(crate) malloc: FnMalloc,
    pub(crate) free: FnFree,
    pub(crate) memcpy_htod: FnMemcpyHtoD,
    pub(crate) memcpy_dtoh: FnMemcpyDtoH,
    pub(crate) memcpy_dtod: FnMemcpyDtoD,
    pub(crate) memcpy_htod_async: FnMemcpyHtoDAsync,
    pub(crate) memcpy_dtoh_async: FnMemcpyDtoHAsync,
    pub(crate) memcpy_peer_async: FnMemcpyPeerAsync,
    pub(crate) device_can_access_peer: FnDeviceCanAccessPeer,
    pub(crate) enable_peer_access: FnEnablePeerAccess,
    pub(crate) module_load_data_ex: FnModuleLoadDataEx,
    pub(crate) module_unload: FnModuleUnload,
    pub(crate) module_get_function: FnModuleGetFunction,
    pub(crate) module_launch_kernel: FnModuleLaunchKernel,
}

impl Hip {
    /// Load `libamdhip64`.
    pub fn load() -> Result<Hip> {
        let lib = Lib::open(&["libamdhip64.so.6", "libamdhip64.so"])?;
        // SAFETY: types transcribed from hip_runtime_api.h.
        unsafe {
            Ok(Hip {
                get_error_string: lib.sym("hipGetErrorString")?,
                get_device_count: lib.sym("hipGetDeviceCount")?,
                set_device: lib.sym("hipSetDevice")?,
                device_get_name: lib.sym("hipDeviceGetName")?,
                mem_get_info: lib.sym("hipMemGetInfo")?,
                device_get_attribute: lib.sym("hipDeviceGetAttribute")?,
                get_device_properties: lib.sym("hipGetDeviceProperties")?,
                stream_create: lib.sym("hipStreamCreate")?,
                stream_destroy: lib.sym("hipStreamDestroy")?,
                stream_synchronize: lib.sym("hipStreamSynchronize")?,
                malloc: lib.sym("hipMalloc")?,
                free: lib.sym("hipFree")?,
                memcpy_htod: lib.sym("hipMemcpyHtoD")?,
                memcpy_dtoh: lib.sym("hipMemcpyDtoH")?,
                memcpy_dtod: lib.sym("hipMemcpyDtoD")?,
                memcpy_htod_async: lib.sym("hipMemcpyHtoDAsync")?,
                memcpy_dtoh_async: lib.sym("hipMemcpyDtoHAsync")?,
                memcpy_peer_async: lib.sym("hipMemcpyPeerAsync")?,
                device_can_access_peer: lib.sym("hipDeviceCanAccessPeer")?,
                enable_peer_access: lib.sym("hipEnablePeerAccess")?,
                module_load_data_ex: lib.sym("hipModuleLoadDataEx")?,
                module_unload: lib.sym("hipModuleUnload")?,
                module_get_function: lib.sym("hipModuleGetFunction")?,
                module_launch_kernel: lib.sym("hipModuleLaunchKernel")?,
                lib,
            })
        }
    }

    /// Human-readable text for a hipError_t code.
    pub fn error_string(&self, code: HipResult) -> String {
        // SAFETY: returns a static string per HIP contract.
        let p = unsafe { (self.get_error_string)(code) };
        if p.is_null() {
            format!("HIP error {code}")
        } else {
            // SAFETY: static NUL-terminated.
            unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
}
