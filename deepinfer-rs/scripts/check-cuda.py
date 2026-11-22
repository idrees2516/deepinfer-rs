#!/usr/bin/env python3
"""Offline CUDA compile gate for deepinfer-rs kernels.

Compiles every embedded kernel translation unit exactly the way the
runtime JIT does (COMMON + module source, `--std=c++17 -default-device
--use_fast_math=off`) through NVRTC, then assembles the emitted PTX with
ptxas for each target architecture. This validates syntax, semantics AND
inline PTX (cp.async / mma / atom.sys strings) without a GPU present.

Usage:
    python3 scripts/check-cuda.py [--ptx-only] [module ...]

Requires the pip wheels `nvidia-cuda-nvrtc-cu12` and (unless --ptx-only)
`nvidia-cuda-nvcc-cu12`. Exit code 0 = every module compiled clean on
every arch.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KDIR = os.path.join(ROOT, "crates", "kernels", "kernels")


def find_lib(patterns):
    for p in patterns:
        if os.path.exists(p):
            return p
    import glob
    for pat in patterns:
        g = glob.glob(os.path.expanduser(mat) if (mat := pat) else "")
        if g:
            return sorted(g)[0]
    return None


NVRTC = find_lib([
    "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    "/usr/local/cuda/lib64/libnvrtc.so",
])
PTXAS = find_lib([
    "/home/z/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas",
    "/usr/local/cuda/bin/ptxas",
])

# (module name, file, extra files prepended after COMMON, arch list)
MODULES = [
    ("elementwise", ["common.cuh", "elementwise.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("gemm", ["common.cuh", "gemm.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("attention", ["common.cuh", "attention.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("dsa", ["common.cuh", "dsa.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("moe", ["common.cuh", "moe.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("decode", ["common.cuh", "decode_params.h", "decode.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("peer_ll", ["common.cuh", "peer_ll.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("serve", ["common.cuh", "serve.cu"], ["sm_80", "sm_90", "sm_100"]),
    ("sampler", ["common.cuh", "sampler.cu"], ["sm_80", "sm_90", "sm_100"]),
    # tcgen05 family targets (arch-specific `a` suffix, as sm100_module does).
    ("gemm_sm100", ["common.cuh", "gemm_sm100.cu"], ["sm_100a", "sm_103a"]),
]

nvrtc = None


def nvrtc_init():
    global nvrtc
    nvrtc = ctypes.CDLL(NVRTC)
    nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
    nvrtc.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
    nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_char_p)]
    nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
    nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
    nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_size_t)]
    nvrtc.nvrtcGetPTX.restype = ctypes.c_int
    nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    nvrtc.nvrtcVersion.restype = ctypes.c_int
    nvrtc.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int),
                                   ctypes.POINTER(ctypes.c_int)]


def compile_ptx(source, name, arch):
    prog = ctypes.c_void_p()
    src = source.encode()
    nm = f"{name}.cu".encode()
    r = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, nm, 0, None, None)
    if r != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {r}")
    opts = [b"--std=c++17", f"--gpu-architecture={arch}".encode(),
            b"-default-device"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    r = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value + 1)
    nvrtc.nvrtcGetProgramLog(prog, log)
    log_s = log.value.decode(errors="replace")
    if r != 0:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"NVRTC failed ({arch}):\n{log_s}")
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value + 1)
    nvrtc.nvrtcGetPTX(prog, ptx)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value[:ptx_size.value], log_s


def assemble(ptx, arch):
    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as f:
        f.write(ptx)
        path = f.name
    try:
        r = subprocess.run([PTXAS, f"-arch={arch}", path, "-o", os.devnull],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas failed ({arch}):\n{r.stdout}\n{r.stderr}")
        return r.stderr
    finally:
        os.unlink(path)


def main():
    global NVRTC, PTXAS
    args = sys.argv[1:]
    ptx_only = "--ptx-only" in args
    args = [a for a in args if not a.startswith("--")]
    if ptx_only:
        PTXAS = None
    if NVRTC is None:
        print("libnvrtc not found (pip install nvidia-cuda-nvrtc-cu12)", file=sys.stderr)
        return 2
    if PTXAS is None and not ptx_only:
        print("ptxas not found (pip install nvidia-cuda-nvcc-cu12); "
              "PTX-only mode", file=sys.stderr)
    nvrtc_init()
    major, minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    print(f"NVRTC {major.value}.{minor.value}; ptxas "
          f"{('on' if PTXAS else 'off')}")

    mods = MODULES
    if args:
        want = set(args)
        mods = [m for m in MODULES if m[0] in want]
        missing = want - {m[0] for m in mods}
        if missing:
            print(f"unknown module(s): {sorted(missing)}", file=sys.stderr)
            return 2

    failures = 0
    for name, files, archs in mods:
        parts = []
        for f in files:
            with open(os.path.join(KDIR, f)) as fh:
                parts.append(fh.read())
        source = "\n".join(parts)
        for arch in archs:
            try:
                ptx, log = compile_ptx(source, name, arch)
                tag = f"{name:12s} {arch:9s} PTX ok"
                if PTXAS:
                    err = assemble(ptx, arch)
                    n_regs = ""
                    for line in (err or "").splitlines():
                        if "registers" in line:
                            n_regs = " " + line.strip()
                            break
                    tag += f", SASS ok{n_regs}"
                print(tag)
            except RuntimeError as e:
                failures += 1
                print(f"FAIL {name} {arch}:\n{e}", file=sys.stderr)
    if failures:
        print(f"\n{failures} failure(s)", file=sys.stderr)
        return 1
    print("\nall modules compiled clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
