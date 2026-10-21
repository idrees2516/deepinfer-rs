//! hipRTC FFI, transcribed from `hiprtc.h` (ROCm 5.x–7.x).
//!
//! Kernel sources are shared with the NVIDIA path and guarded with
//! `#if defined(__HIP__)` in the C++ layer; here we only compile and fetch
//! the native code object (`hiprtcGetCode`), which `hipModuleLoadDataEx`
//! accepts directly. This is the same flow as the public hipRTC examples:
//! options `["-D__HIP_PLATFORM_AMD__", "--gpu-architecture=<gfx target>"]`
//! with the target taken from the device's `gcnArchName` (see `ffi::hip`).

use std::ffi::c_void;

use crate::dl::Lib;
use deepinfer_core::Result;

pub(crate) type HiprtcResult = u32;
pub(crate) type Program = *mut c_void;

pub(crate) type FnCreateProgram = unsafe extern "C" fn(
    *mut Program,
    *const std::ffi::c_char, // src
    *const std::ffi::c_char, // name
    i32,                     // numHeaders
    *const *const std::ffi::c_char,
    *const *const std::ffi::c_char,
) -> HiprtcResult;
pub(crate) type FnCompileProgram =
    unsafe extern "C" fn(Program, i32, *const *const std::ffi::c_char) -> HiprtcResult;
pub(crate) type FnGetLogSize = unsafe extern "C" fn(Program, *mut usize) -> HiprtcResult;
pub(crate) type FnGetLog = unsafe extern "C" fn(Program, *mut std::ffi::c_char) -> HiprtcResult;
pub(crate) type FnGetCodeSize = unsafe extern "C" fn(Program, *mut usize) -> HiprtcResult;
pub(crate) type FnGetCode = unsafe extern "C" fn(Program, *mut std::ffi::c_char) -> HiprtcResult;
pub(crate) type FnDestroyProgram = unsafe extern "C" fn(*mut Program) -> HiprtcResult;

/// hipRTC entry points bound to one open `libhiprtc`.
pub struct HipRtc {
    #[allow(dead_code)]
    lib: Lib,
    pub(crate) create_program: FnCreateProgram,
    pub(crate) compile_program: FnCompileProgram,
    pub(crate) get_log_size: FnGetLogSize,
    pub(crate) get_log: FnGetLog,
    pub(crate) get_code_size: FnGetCodeSize,
    pub(crate) get_code: FnGetCode,
    pub(crate) destroy_program: FnDestroyProgram,
}

impl HipRtc {
    /// Load `libhiprtc`.
    pub fn load() -> Result<HipRtc> {
        let lib = Lib::open(&["libhiprtc.so.1", "libhiprtc.so"])?;
        // SAFETY: types transcribed from hiprtc.h.
        unsafe {
            Ok(HipRtc {
                create_program: lib.sym("hiprtcCreateProgram")?,
                compile_program: lib.sym("hiprtcCompileProgram")?,
                get_log_size: lib.sym("hiprtcGetProgramLogSize")?,
                get_log: lib.sym("hiprtcGetProgramLog")?,
                get_code_size: lib.sym("hiprtcGetCodeSize")?,
                get_code: lib.sym("hiprtcGetCode")?,
                destroy_program: lib.sym("hiprtcDestroyProgram")?,
                lib,
            })
        }
    }
}
