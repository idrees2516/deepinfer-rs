# GPU build environment for the cutile (CUDA Rust) backend.
# Source this before cargo build/test of the deepinfer-cutile crate.
# Toolchain pieces are extracted (no root needed) under /home/z/my-project/toolchain/prefix.
# Runtime (JIT + launch) still requires a real NVIDIA GPU with CUDA 13.3+ and driver.

TOOLCHAIN_PREFIX="${TOOLCHAIN_PREFIX:-/home/z/my-project/toolchain/prefix}"

export CUDA_HOME="$TOOLCHAIN_PREFIX/usr/local/cuda-13.4"
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_TARGET_DIR="x86_64-linux"
export LIBCLANG_PATH="$TOOLCHAIN_PREFIX/usr/lib/llvm-19/lib"
export LD_LIBRARY_PATH="$TOOLCHAIN_PREFIX/usr/lib/llvm-19/lib:$TOOLCHAIN_PREFIX/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# clang builtin headers (stddef.h etc.) live outside the default search path in the local prefix
export BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=$TOOLCHAIN_PREFIX/usr/lib/llvm-19/lib/clang/19"

export PATH="$HOME/.cargo/bin:$PATH"
