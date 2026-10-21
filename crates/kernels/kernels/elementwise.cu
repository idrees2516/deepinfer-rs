// elementwise.cu — norms, RoPE, activations, quantization, KV-cache encode.
// Numerics are ports of deepinfer-core / deepinfer-fp8 (bit-exact codecs;
// reductions use fixed tree orders within a block, validated by tolerance
// against the sequential CPU reference in the golden tests).

// RMSNorm over rows of `cols` with gain. out = x / sqrt(mean(x^2)+eps) * g.
extern "C" __global__ void __launch_bounds__(256)
di_rmsnorm_kernel(const float* __restrict__ x,
                  const float* __restrict__ gain,
                  float* __restrict__ out,
                  int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (u64)row * cols;
    float* orow = out + (u64)row * cols;
    __shared__ float red[256];
    float part = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        part += xr[i] * xr[i];
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        orow[i] = xr[i] * inv * gain[i];
    }
}

// Interleaved RoPE (DeepSeek-V4.1 convention) via a host-precomputed
// cos/sin table `cs` laid out [max_pos][half][2]. `conj` selects the
// attention-output conjugate rotation (O-RoPE): (a,b) -> (a*c + b*s,
// -a*s + b*c) instead of (a*c - b*s, a*s + b*c). Bit-exact vs the CPU
// (same per-element operations, no reassociation possible).
extern "C" __global__ void __launch_bounds__(256)
di_rope_kernel(const float* __restrict__ cs,
               const u32* __restrict__ pos,
               float* __restrict__ x,
               int rows, int row_len, int offset, int half, int conj) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * half;
         idx += gridDim.x * blockDim.x) {
        int r = idx / half;
        int j = idx - r * half;
        u32 p = pos[r];
        float c = cs[((u64)p * half + j) << 1];
        float s = cs[(((u64)p * half + j) << 1) + 1];
        int i0 = r * row_len + offset + 2 * j;
        float a = x[i0];
        float b = x[i0 + 1];
        if (conj) {
            x[i0] = a * c + b * s;
            x[i0 + 1] = -a * s + b * c;
        } else {
            x[i0] = a * c - b * s;
            x[i0 + 1] = a * s + b * c;
        }
    }
}

// out[i] = silu(g[i]) * u[i]; silu(z) = z / (1 + expf(-z)).
extern "C" __global__ void __launch_bounds__(256)
di_silu_mul_kernel(const float* __restrict__ g,
                   const float* __restrict__ u,
                   float* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float z = g[i];
        out[i] = (z / (1.0f + expf(-z))) * u[i];
    }
}

// y[i] += x[i].
extern "C" __global__ void __launch_bounds__(256)
di_add_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        y[i] += x[i];
    }
}

// Per-token-group E4M3 + UE8M0 quantization (DeepGEMM activation recipe,
// granularity group x cols along K). One warp handles one (row, group):
//   sf = 2^ceil(log2(amax/448)) (zero group -> 2^-127, byte 0x00)
//   q  = e4m3_rne(sanitize(v)/sf, saturate)
// sanitize: NaN -> +0 (never emit the E4M3 NaN payload); +-Inf saturates.
extern "C" __global__ void __launch_bounds__(256)
di_quant_per_token_group_kernel(const float* __restrict__ x,
                                u8* __restrict__ q,
                                u8* __restrict__ sf,
                                int rows, int cols, int group) {
    int warps_per_block = (int)(blockDim.x / DI_WARP);
    int gw = blockIdx.x * warps_per_block + di_warp_id();
    int groups_per_row = cols / group;
    int total = rows * groups_per_row;
    if (gw >= total) return;
    int r = gw / groups_per_row;
    int g = gw - r * groups_per_row;
    const float* xs = x + (u64)r * cols + (u64)g * group;
    float amax = 0.0f;
    for (int i = di_lane_id(); i < group; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float ratio = amax / 448.0f;
    u8 sfb = 0x00u;
    if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
    float scale = di_e8m0_to_f32(sfb);
    u8* qs = q + (u64)r * cols + (u64)g * group;
    for (int i = di_lane_id(); i < group; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        qs[i] = di_f32_to_e4m3(v / scale, true);
    }
    if (di_lane_id() == 0) sf[(u64)r * groups_per_row + g] = sfb;
}

// FlashMLA V4.1 FP8 KV-cache row encode: 512 f32 dims -> 528 bytes
// (512 e4m3 values then 16 ue8m0 scales, one per 32 consecutive dims).
extern "C" __global__ void __launch_bounds__(64)
di_kv_encode_fp8_kernel(const float* __restrict__ x,
                        u8* __restrict__ out,
                        int tokens, int dims /* 512 */) {
    // One warp per (token, group of 32).
    int gw = blockIdx.x * (int)(blockDim.x / DI_WARP) + di_warp_id();
    int groups = dims / 32;
    int total = tokens * groups;
    if (gw >= total) return;
    int t = gw / groups;
    int g = gw - t * groups;
    const float* xs = x + (u64)t * dims + (u64)g * 32;
    u8* row = out + (u64)t * 528;
    float amax = 0.0f;
    for (int i = di_lane_id(); i < 32; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float ratio = amax / 448.0f;
    u8 sfb = 0x00u;
    if (ratio > 0.0f) sfb = di_f32_to_e8m0_ceil(ratio);
    float scale = di_e8m0_to_f32(sfb);
    for (int i = di_lane_id(); i < 32; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        row[g * 32 + i] = di_f32_to_e4m3(v / scale, true);
    }
    if (di_lane_id() == 0) row[dims + g] = sfb;
}

// FlashMLA V4.1 FP4 KV-cache row encode: 512 f32 dims -> 288 bytes
// (256B packed e2m1, even dim LOW nibble, then 32 e4m3 scales per 16 dims;
// scale = e4m3_rne(amax/6) clamped to [2^-9, 448]).
extern "C" __global__ void __launch_bounds__(64)
di_kv_encode_fp4_kernel(const float* __restrict__ x,
                        u8* __restrict__ out,
                        int tokens, int dims /* 512 */) {
    int gw = blockIdx.x * (int)(blockDim.x / DI_WARP) + di_warp_id();
    int groups = dims / 16;
    int total = tokens * groups;
    if (gw >= total) return;
    int t = gw / groups;
    int g = gw - t * groups;
    const float* xs = x + (u64)t * dims + (u64)g * 16;
    u8* row = out + (u64)t * 288;
    float amax = 0.0f;
    for (int i = di_lane_id(); i < 16; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = di_warp_reduce_max(amax);
    float sv = amax / 6.0f;
    if (sv < 1.953125e-3f) sv = 1.953125e-3f; // 2^-9: smallest positive e4m3
    if (sv > 448.0f) sv = 448.0f;
    u8 sb = di_f32_to_e4m3(sv, true);
    float scale = di_e4m3_to_f32(sb);
    for (int i = di_lane_id(); i < 16; i += DI_WARP) {
        float v = xs[i];
        if (isnan(v)) v = 0.0f;
        u8 nib = di_f32_to_e2m1(v / scale);
        int idx = g * 16 + i;
        if ((idx & 1) == 0) {
            row[idx >> 1] = nib; // low nibble
        } else {
            row[idx >> 1] |= (u8)(nib << 4); // high nibble
        }
    }
    if (di_lane_id() == 0) row[256 + g] = sb;
}

// Cast f32 -> bf16 (RNE) elementwise.
extern "C" __global__ void __launch_bounds__(256)
di_f32_to_bf16_kernel(const float* __restrict__ x, u16* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = di_f32_to_bf16(x[i]);
    }
}

// Cast bf16 -> f32 elementwise (exact).
extern "C" __global__ void __launch_bounds__(256)
di_bf16_to_f32_kernel(const u16* __restrict__ x, float* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = di_bf16_to_f32(x[i]);
    }
}
