//! Audited `dlopen`/`dlsym` loader — the root of the workspace's only unsafe
//! boundary.
//!
//! SAFETY MODEL (see docs/GPU.md):
//! * We never link against CUDA/HIP at build time. Libraries are opened by
//!   name at runtime, so binaries build and run on machines with no GPU
//!   stack at all (matching DeepGEMM's DeepJIT and `deepgemm-rs` design).
//! * `Lib::sym` transmutes a `void*` to a typed function pointer. The type is
//!   chosen exclusively by our own typed FFI modules (`ffi::cuda`, `ffi::hip`,
//!   `ffi::nvrtc`, `ffi::hiprtc`), whose declarations are transcribed from the
//!   vendor headers (CUDA Driver API 12.x/13.x, HIP/hipRTC ROCm 5.x–7.x).
//! * Handles are `Send`/`Sync`: `dlopen`/`dlsym` are thread-safe (glibc holds
//!   an internal lock around handle bookkeeping, and resolved function
//!   pointers are immutable once loaded).
//! * No library is ever unloaded (`Lib` has no `Drop`), eliminating the
//!   classic use-after-`dlclose` hazard for function pointers held in structs.

use std::ffi::{c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;

use deepinfer_core::{ensure, Error, Result};

// RTLD flags (glibc): NOW = resolve all symbols eagerly (fail fast),
// LOCAL = do not pollute the global symbol namespace.
const RTLD_NOW: i32 = 0x2;
const RTLD_LOCAL: i32 = 0;

extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    fn dlerror() -> *const std::ffi::c_char;
}

/// An open shared-library handle that is never closed.
pub struct Lib {
    handle: *mut c_void,
    #[allow(dead_code)]
    path: std::path::PathBuf,
}

// SAFETY: dlopen handles are valid process-wide; dlsym results are immutable.
unsafe impl Send for Lib {}
// SAFETY: see above; all vendor calls we make through these pointers are
// themselves documented as thread-safe by NVIDIA/AMD.
unsafe impl Sync for Lib {}

impl std::fmt::Debug for Lib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lib").field("path", &self.path).finish()
    }
}

fn last_dl_error() -> String {
    // SAFETY: dlerror returns a thread-local NUL-terminated string or NULL;
    // we only read it immediately after a failed call on this thread.
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Lib {
    /// Try each candidate SONAME in order and open the first that loads.
    pub fn open(names: &[&str]) -> Result<Lib> {
        for name in names {
            let cname = CString::new(*name).map_err(|_| Error::internal("NUL in lib name"))?;
            // SAFETY: cname is a valid NUL-terminated string; RTLD_NOW|RTLD_LOCAL
            // is a valid flag combination. Handle checked below.
            let handle = unsafe { dlopen(cname.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(Lib {
                    handle,
                    path: std::path::PathBuf::from(std::ffi::OsStr::from_bytes(name.as_bytes())),
                });
            }
        }
        let tried = names.join(", ");
        let reason = last_dl_error();
        Err(Error::internal(format!(
            "GPU library not found (tried: {tried}): {reason}"
        )))
    }

    /// Resolve a symbol to a typed function pointer.
    ///
    /// # Safety
    /// The caller must pass a `T` whose ABI matches the actual symbol. In this
    /// workspace `T` is always one of the typed declarations in `ffi::*`,
    /// which are transcriptions of the vendor headers.
    pub unsafe fn sym<T: Copy>(&self, name: &str) -> Result<T> {
        let cname = CString::new(name).map_err(|_| Error::internal("NUL in symbol name"))?;
        // SAFETY: valid handle + NUL-terminated name; contract per dlsym(3).
        let _ = unsafe { dlerror() }; // clear stale error
        let ptr = unsafe { dlsym(self.handle, cname.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::internal(format!(
                "symbol `{name}` missing from {}",
                self.path.display()
            )));
        }
        // dlsym returns the symbol ADDRESS; for a function symbol that value
        // IS the function pointer. We write the pointer value (not the bytes
        // it points at) into a MaybeUninit<T> — the libloading technique,
        // restricted to pointer-sized Copy types (fn pointers).
        let addr = ptr as usize;
        ensure!(
            std::mem::size_of::<T>() == std::mem::size_of::<usize>(),
            Internal,
            "sym: T must be pointer-sized (an extern fn pointer type)"
        );
        let mut dst = std::mem::MaybeUninit::<T>::uninit();
        // SAFETY: writing size_of::<T>() (== pointer-sized) bytes into an
        // uninitialized local; T is Copy so no drop issues; the value read
        // back is the dlsym-returned address reinterpreted as T.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &addr as *const usize as *const u8,
                dst.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            );
            Ok(dst.assume_init())
        }
    }

    /// Resolve a symbol that may legitimately be absent across vendor
    /// versions; returns `None` in that case.
    ///
    /// # Safety
    /// Same contract as [`Lib::sym`].
    pub unsafe fn sym_opt<T: Copy>(&self, name: &str) -> Result<Option<T>> {
        let cname = CString::new(name).map_err(|_| Error::internal("NUL in symbol name"))?;
        // SAFETY: see sym().
        let _ = unsafe { dlerror() };
        let ptr = unsafe { dlsym(self.handle, cname.as_ptr()) };
        if ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: see sym() — writes the dlsym address value as T.
            let addr = ptr as usize;
            ensure!(
                std::mem::size_of::<T>() == std::mem::size_of::<usize>(),
                Internal,
                "sym_opt: T must be pointer-sized (an extern fn pointer type)"
            );
            let mut dst = std::mem::MaybeUninit::<T>::uninit();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &addr as *const usize as *const u8,
                    dst.as_mut_ptr() as *mut u8,
                    std::mem::size_of::<T>(),
                );
                Ok(Some(dst.assume_init()))
            }
        }
    }
}

/// Probe whether a library can be loaded at all (used for graceful
/// no-GPU environments: we report "not present" instead of an error).
pub fn probe(names: &[&str]) -> bool {
    Lib::open(names).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libc_is_loadable() {
        assert!(probe(&["libc.so.6", "libc.so"]));
    }

    #[test]
    fn missing_lib_reports_not_probeable() {
        assert!(!probe(&["lib_definitely_not_here_so_9x.so"]));
    }

    #[test]
    fn sym_resolves_a_known_function() {
        let lib = Lib::open(&["libc.so.6", "libc.so"]).expect("libc");
        // SAFETY: strlen has a stable C ABI: (const char*) -> size_t.
        let f: unsafe extern "C" fn(*const std::ffi::c_char) -> usize =
            unsafe { lib.sym("strlen") }.expect("strlen symbol");
        let c = CString::new("deepinfer").unwrap();
        // SAFETY: c is NUL-terminated.
        assert_eq!(unsafe { f(c.as_ptr()) }, 9);
    }

    #[test]
    fn sym_opt_missing_returns_none() {
        let lib = Lib::open(&["libc.so.6", "libc.so"]).expect("libc");
        // SAFETY: result discarded if present; symbol does not exist.
        let r: Option<unsafe extern "C" fn()> =
            unsafe { lib.sym_opt("definitely_not_a_libc_symbol_xyz") }.expect("no error");
        assert!(r.is_none());
    }
}
