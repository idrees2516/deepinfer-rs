//! Runtime kernel compilation (NVRTC / hipRTC) with a content-addressed
//! on-disk cache.
//!
//! Environment contract (mirroring DeepGEMM's `DG_JIT_*` variables, renamed
//! under the DeepInfer namespace):
//! * `DEEPINFER_JIT_CACHE`  — `0` disables the disk cache (default: on).
//! * `DEEPINFER_JIT_DEBUG`  — `1` prints compile decisions and cache hits.
//! * `DEEPINFER_CACHE_DIR`  — overrides the cache root
//!   (default: `$XDG_CACHE_HOME`/`$HOME/.cache` + `/deepinfer-rs/jit`).
//!
//! Cache key: SHA-256 over (backend tag, arch tag, kernel name, extra
//! options, source text). Cache hit => zero compile latency, which is what
//! makes JIT-based kernels viable for serving.

use std::ffi::{c_void, CString};
use std::io::Write;
use std::path::PathBuf;

use deepinfer_core::{ensure, Error, Result};

use crate::device::{Backend, Device, GpuArch, Module};
use crate::ffi::nvrtc::Nvrtc;
use crate::ffi::hiprtc::HipRtc;
use crate::hash::hex_digest;

/// A compiled, ready-to-load kernel image.
pub enum KernelImage {
    /// CUBIN (or PTX) for CUDA `cuModuleLoadData`.
    Cubin(Vec<u8>),
    /// HIP code object (HSACO) for `hipModuleLoadDataEx`.
    CodeObject(Vec<u8>),
}

impl KernelImage {
    /// Raw image bytes.
    pub fn bytes(&self) -> &[u8] {
        match self {
            KernelImage::Cubin(b) => b,
            KernelImage::CodeObject(b) => b,
        }
    }
}

fn cache_dir() -> Option<PathBuf> {
    if std::env::var("DEEPINFER_JIT_CACHE").ok().as_deref() == Some("0") {
        return None;
    }
    let base = match std::env::var("DEEPINFER_CACHE_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            let home = std::env::var("XDG_CACHE_HOME")
                .ok()
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .or_else(|| std::env::var("HOME").ok().map(PathBuf::from))?;
            home.join(".cache")
        }
    };
    Some(base.join("deepinfer-rs").join("jit"))
}

fn debug_log(msg: &str) {
    if std::env::var("DEEPINFER_JIT_DEBUG").ok().as_deref() == Some("1") {
        eprintln!("[deepinfer-jit] {msg}");
    }
}

/// Compile one translation unit for `arch` and load it on `device`.
///
/// `extra_options` are appended to the vendor defaults (e.g. `-DFOO=1`).
/// The same function is used for both backends; the source is expected to be
/// the shared dual-target CUDA/HIP C++ used by `deepinfer-kernels`.
pub fn compile_and_load(
    device: &Device,
    source: &str,
    kernel_name: &str,
    arch: &GpuArch,
    extra_options: &[String],
) -> Result<Module> {
    let image = compile(device, source, kernel_name, arch, extra_options)?;
    device.load_module(image.bytes())
}

/// Compile one translation unit, going through the disk cache when enabled.
pub fn compile(
    device: &Device,
    source: &str,
    kernel_name: &str,
    arch: &GpuArch,
    extra_options: &[String],
) -> Result<KernelImage> {
    let backend = device.backend();
    let key = hex_digest(&[
        backend.name().as_bytes(),
        arch.as_str().as_bytes(),
        kernel_name.as_bytes(),
        extra_options.join("\u{1}").as_bytes(),
        source.as_bytes(),
    ]);
    let dir = cache_dir();
    let cache_path = dir.as_ref().map(|d| d.join(format!("{key}.bin")));

    if let Some(path) = &cache_path {
        if let Ok(bytes) = std::fs::read(path) {
            debug_log(&format!("cache hit {kernel_name} ({key:.12})"));
            return Ok(match backend {
                Backend::Cuda => KernelImage::Cubin(bytes),
                Backend::Hip => KernelImage::CodeObject(bytes),
            });
        }
    }

    debug_log(&format!("compiling {kernel_name} for {}", arch.as_str()));
    let image = match backend {
        Backend::Cuda => {
            let rtc = device
                .nvrtc()
                .ok_or_else(|| Error::internal("libnvrtc not present; cannot JIT CUDA kernels"))?;
            compile_cuda(rtc, source, kernel_name, arch, extra_options)?
        }
        Backend::Hip => {
            let rtc = device
                .hiprtc()
                .ok_or_else(|| Error::internal("libhiprtc not present; cannot JIT HIP kernels"))?;
            compile_hip(rtc, source, kernel_name, arch, extra_options)?
        }
    };

    if let Some(path) = &cache_path {
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_ok() {
                if let Ok(mut f) = std::fs::File::create(path) {
                    let _ = f.write_all(image.bytes());
                }
            }
        }
    }
    Ok(image)
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::internal("NUL byte in source/option"))
}

fn compile_cuda(
    rtc: &Nvrtc,
    source: &str,
    kernel_name: &str,
    arch: &GpuArch,
    extra: &[String],
) -> Result<KernelImage> {
    let src = cstring(source)?;
    let name = cstring(&format!("{kernel_name}.cu"))?;
    let mut prog: *mut c_void = std::ptr::null_mut();
    // SAFETY: NVRTC program creation per nvrtc.h; out-pointer valid, source
    // NUL-terminated, zero headers.
    let r = unsafe {
        (rtc.create_program)(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    ensure!(r == 0, Internal, "nvrtcCreateProgram: {}", rtc.error_string(r));

    let run_compile = |flags: &[String]| -> Result<(u32, String)> {
        let cflags: Vec<CString> = flags.iter().map(|f| cstring(f)).collect::<Result<_>>()?;
        let ptrs: Vec<*const std::ffi::c_char> = cflags.iter().map(|f| f.as_ptr()).collect();
        // SAFETY: program handle valid; options NUL-terminated.
        let r = unsafe { (rtc.compile_program)(prog, ptrs.len() as i32, ptrs.as_ptr()) };
        let log = program_log_nvrtc(rtc, prog);
        Ok((r, log))
    };

    let mut flags: Vec<String> = vec![
        "--std=c++17".to_string(),
        arch.nvrtc_flag(),
        "-default-device".to_string(),
    ];
    flags.extend(extra.iter().cloned());

    let (mut r, mut log) = run_compile(&flags)?;
    if r != 0 {
        // Older NVRTC + newer GPU: retry with PTX for the driver JIT.
        if let Some(ptx_flag) = arch.nvrtc_ptx_flag() {
            debug_log("sm target rejected; retrying with PTX for driver JIT");
            flags[1] = ptx_flag;
            let (r2, log2) = run_compile(&flags)?;
            r = r2;
            log = log2;
        }
    }
    if r != 0 {
        return Err(Error::internal(format!(
            "NVRTC compile failed for {kernel_name} ({}):\n{log}",
            rtc.error_string(r)
        )));
    }
    if !log.trim().is_empty() {
        debug_log(&format!("{kernel_name} nvrtc log: {log}"));
    }

    let want_ptx = flags.iter().any(|f| f.starts_with("--gpu-architecture=compute_"));
    let bytes = if want_ptx {
        get_blob_nvrtc(rtc, prog, kernel_name, true)?
    } else {
        get_blob_nvrtc(rtc, prog, kernel_name, false)?
    };
    // SAFETY: destroys the program we created; handle becomes invalid, we
    // never touch it again.
    unsafe { (rtc.destroy_program)(&mut prog) };
    Ok(KernelImage::Cubin(bytes))
}

fn get_blob_nvrtc(rtc: &Nvrtc, prog: *mut c_void, kernel_name: &str, ptx: bool) -> Result<Vec<u8>> {
    let mut size: usize = 0;
    // SAFETY: program handle valid; out-pointer valid.
    let r = if ptx {
        unsafe { (rtc.get_ptx_size)(prog, &mut size) }
    } else {
        unsafe { (rtc.get_cubin_size)(prog, &mut size) }
    };
    ensure!(r == 0, Internal, "nvrtcGet*CUBINSize: {}", rtc.error_string(r));
    let mut buf = vec![0u8; size + 1];
    // SAFETY: buffer of `size+1` writable bytes; vendor writes at most `size`.
    let r = if ptx {
        unsafe { (rtc.get_ptx)(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) }
    } else {
        unsafe { (rtc.get_cubin)(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) }
    };
    ensure!(r == 0, Internal, "nvrtcGet*CUBIN: {}", rtc.error_string(r));
    buf.truncate(size);
    debug_assert!(!buf.is_empty(), "empty {kernel_name} image");
    Ok(buf)
}

fn program_log_nvrtc(rtc: &Nvrtc, prog: *mut c_void) -> String {
    let mut size: usize = 0;
    // SAFETY: handle valid; out-pointer valid.
    if unsafe { (rtc.get_log_size)(prog, &mut size) } != 0 || size == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; size + 1];
    // SAFETY: buffer of `size+1` writable bytes.
    if unsafe { (rtc.get_log)(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) } != 0 {
        return String::new();
    }
    buf.truncate(size);
    String::from_utf8_lossy(&buf).into_owned()
}

fn compile_hip(
    rtc: &HipRtc,
    source: &str,
    kernel_name: &str,
    arch: &GpuArch,
    extra: &[String],
) -> Result<KernelImage> {
    let src = cstring(source)?;
    let name = cstring(&format!("{kernel_name}.hip"))?;
    let mut prog: *mut c_void = std::ptr::null_mut();
    // SAFETY: per hiprtc.h; out-pointer valid; zero headers.
    let r = unsafe {
        (rtc.create_program)(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    ensure!(r == 0, Internal, "hiprtcCreateProgram failed");

    let mut flags: Vec<String> = vec![
        "-D__HIP_PLATFORM_AMD__".to_string(),
        arch.hiprtc_flag(),
    ];
    flags.extend(extra.iter().cloned());
    let cflags: Vec<CString> = flags.iter().map(|f| cstring(f)).collect::<Result<_>>()?;
    let ptrs: Vec<*const std::ffi::c_char> = cflags.iter().map(|f| f.as_ptr()).collect();
    // SAFETY: handle valid; options NUL-terminated.
    let r = unsafe { (rtc.compile_program)(prog, ptrs.len() as i32, ptrs.as_ptr()) };

    let mut log = String::new();
    {
        let mut size: usize = 0;
        // SAFETY: handle valid; out-pointer valid.
        if unsafe { (rtc.get_log_size)(prog, &mut size) } == 0 && size > 0 {
            let mut buf = vec![0u8; size + 1];
            // SAFETY: buffer of `size+1` writable bytes.
            if unsafe { (rtc.get_log)(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) } == 0 {
                buf.truncate(size);
                log = String::from_utf8_lossy(&buf).into_owned();
            }
        }
    }
    if r != 0 {
        // SAFETY: destroys the program we created.
        unsafe { (rtc.destroy_program)(&mut prog) };
        return Err(Error::internal(format!(
            "hipRTC compile failed for {kernel_name}:\n{log}"
        )));
    }
    if !log.trim().is_empty() {
        debug_log(&format!("{kernel_name} hiprtc log: {log}"));
    }

    let mut size: usize = 0;
    // SAFETY: handle valid; out-pointer valid.
    let r = unsafe { (rtc.get_code_size)(prog, &mut size) };
    ensure!(r == 0, Internal, "hiprtcGetCodeSize failed");
    let mut buf = vec![0u8; size];
    // SAFETY: buffer of exactly `size` writable bytes.
    let r = unsafe { (rtc.get_code)(prog, buf.as_mut_ptr() as *mut std::ffi::c_char) };
    ensure!(r == 0, Internal, "hiprtcGetCode failed");
    // SAFETY: destroys the program we created.
    unsafe { (rtc.destroy_program)(&mut prog) };
    Ok(KernelImage::CodeObject(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_respects_disable() {
        // Not a great unit test (mutates env), but scoped: run serially with
        // other env-touching tests by using a unique var state.
        // SAFETY of env mutation: tests are single-threaded per crate.
        let key = "DEEPINFER_JIT_CACHE_TEST_MARKER";
        std::env::set_var(key, "1");
        std::env::set_var("DEEPINFER_JIT_CACHE", "0");
        assert!(cache_dir().is_none());
        std::env::remove_var("DEEPINFER_JIT_CACHE");
        std::env::remove_var(key);
    }

    #[test]
    fn key_is_stable_and_sensitive() {
        let a = hex_digest(&[b"cuda", b"sm_100", b"k", b"", b"src"]);
        let b = hex_digest(&[b"cuda", b"sm_100", b"k", b"", b"src"]);
        let c = hex_digest(&[b"cuda", b"sm_103", b"k", b"", b"src"]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
