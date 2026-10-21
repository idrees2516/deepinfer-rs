//! NVRTC (runtime compilation of CUDA C++) FFI, transcribed from `nvrtc.h`.
//!
//! Design mirrors DeepGEMM's DeepJIT (`DG_JIT_USE_NVRTC` — reported ~10x faster
//! JIT than the nvcc-spawn path) and `deepgemm-rs`: kernels are embedded as
//! strings, compiled to a CUBIN for the exact device arch on first use, and
//! cached on disk keyed by a SHA-256 of (source, options, arch, backend).

use std::ffi::c_void;

use crate::dl::Lib;
use deepinfer_core::Result;

pub(crate) type NvrtcResult = u32;
pub(crate) type Program = *mut c_void;

pub(crate) type FnVersion = unsafe extern "C" fn(*mut i32, *mut i32) -> NvrtcResult;
pub(crate) type FnCreateProgram = unsafe extern "C" fn(
    *mut Program,
    *const std::ffi::c_char, // src
    *const std::ffi::c_char, // name
    i32,                     // numHeaders
    *const *const std::ffi::c_char,
    *const *const std::ffi::c_char,
) -> NvrtcResult;
pub(crate) type FnCompileProgram =
    unsafe extern "C" fn(Program, i32, *const *const std::ffi::c_char) -> NvrtcResult;
pub(crate) type FnGetLogSize = unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult;
pub(crate) type FnGetLog = unsafe extern "C" fn(Program, *mut std::ffi::c_char) -> NvrtcResult;
pub(crate) type FnGetCubinSize = unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult;
pub(crate) type FnGetCubin = unsafe extern "C" fn(Program, *mut std::ffi::c_char) -> NvrtcResult;
pub(crate) type FnGetPtxSize = unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult;
pub(crate) type FnGetPtx = unsafe extern "C" fn(Program, *mut std::ffi::c_char) -> NvrtcResult;
pub(crate) type FnDestroyProgram = unsafe extern "C" fn(*mut Program) -> NvrtcResult;
pub(crate) type FnGetErrorString =
    unsafe extern "C" fn(NvrtcResult) -> *const std::ffi::c_char;

/// NVRTC entry points bound to one open `libnvrtc`.
pub struct Nvrtc {
    #[allow(dead_code)]
    lib: Lib,
    pub(crate) version: FnVersion,
    pub(crate) create_program: FnCreateProgram,
    pub(crate) compile_program: FnCompileProgram,
    pub(crate) get_log_size: FnGetLogSize,
    pub(crate) get_log: FnGetLog,
    pub(crate) get_cubin_size: FnGetCubinSize,
    pub(crate) get_cubin: FnGetCubin,
    pub(crate) get_ptx_size: FnGetPtxSize,
    pub(crate) get_ptx: FnGetPtx,
    pub(crate) destroy_program: FnDestroyProgram,
    pub(crate) get_error_string: FnGetErrorString,
}

impl Nvrtc {
    /// Load `libnvrtc` (any major CUDA version).
    pub fn load() -> Result<Nvrtc> {
        let lib = Lib::open(&[
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so",
        ])?;
        // SAFETY: types transcribed from nvrtc.h.
        unsafe {
            Ok(Nvrtc {
                version: lib.sym("nvrtcVersion")?,
                create_program: lib.sym("nvrtcCreateProgram")?,
                compile_program: lib.sym("nvrtcCompileProgram")?,
                get_log_size: lib.sym("nvrtcGetProgramLogSize")?,
                get_log: lib.sym("nvrtcGetProgramLog")?,
                get_cubin_size: lib.sym("nvrtcGetCUBINSize")?,
                get_cubin: lib.sym("nvrtcGetCUBIN")?,
                get_ptx_size: lib.sym("nvrtcGetPTXSize")?,
                get_ptx: lib.sym("nvrtcGetPTX")?,
                destroy_program: lib.sym("nvrtcDestroyProgram")?,
                get_error_string: lib.sym("nvrtcGetErrorString")?,
                lib,
            })
        }
    }

    /// `(major, minor)` of the loaded NVRTC.
    pub fn version(&self) -> (i32, i32) {
        let (mut a, mut b) = (0i32, 0i32);
        // SAFETY: valid out-pointers.
        unsafe { (self.version)(&mut a, &mut b) };
        (a, b)
    }

    /// Error text for an NVRTC code.
    pub fn error_string(&self, code: NvrtcResult) -> String {
        // SAFETY: returns a static string per nvrtc.h contract.
        let p = unsafe { (self.get_error_string)(code) };
        if p.is_null() {
            format!("NVRTC error {code}")
        } else {
            // SAFETY: static NUL-terminated.
            unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
}
