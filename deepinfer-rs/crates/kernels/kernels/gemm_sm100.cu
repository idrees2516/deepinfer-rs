// gemm_sm100.cu — tcgen05 (5th-gen tensor core) FP8 e4m3 GEMM with TMA,
// for the architecture-specific sm_100a / sm_103a targets (B200/B300, i.e.
// the GB200/GB300 NVL72 compute planes). NOT compiled for other targets —
// deepinfer-kernels only builds this module when the device reports
// compute capability >= 10.0.
//
// Semantics: exactly `di_gemm_fp8_nt_kernel` (DeepGEMM 1d1d recipe):
//   A [M, K] u8 e4m3, a_sf [M, K/128] ue8m0 (activations 1x128)
//   B [N, K] u8 e4m3, b_sf [ceil(N/128), K/128] ue8m0 (weights 128x128)
//   D [M, N] f32 = sum over 128-k blocks of (sa[m,b] * sb[n/128,b]) *
//                  dequant(A) @ dequant(B)^T
// Requires K % 128 == 0 (the CPU fp8_gemm_nt contract).
//
// Hardware mapping (PTX ISA 9.x "tcgen05" + CUTLASS SM100 UMMA + DeepGEMM
// sm100_fp8_fp4_gemm_1d1d, all cross-checked):
//   * cp.async.bulk.tensor.2d with SWIZZLE_128B tensor maps move A/B tiles
//     into shared memory (one TMA per tile, mbarrier complete_tx tracking).
//   * tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale consumes e4m3
//     operands straight from SMEM (K-major, 128B-swizzled descriptors) and
//     applies the UE8M0 scale factors from Tensor Memory, accumulating f32
//     in TMEM — the SM100 way of doing DeepGEMM's per-128-k rescale without
//     any intermediate register round-trip.
//   * Scale factors ride TMEM as packed UE8M0 x4 words: one word per matrix
//     line (A row / B row), all four bytes equal to the line's 128-k-block
//     scale, uploaded per k-block by a loader warp through SMEM and
//     tcgen05.cp.32x128b.warpx4 (the DeepGEMM "UTCCP transposer" pattern).
//     SFA_ID/SFB_ID in the instruction descriptor selects the byte (= the
//     32-element sub-block inside the 128-k block); with gran-128 scales all
//     four bytes are identical, so every sfid is exact.
//   * Warp specialization: w0 = TMA producer, w1 = MMA issue, w2 = scale
//     loader, w3..w6 = epilogue (Data-path Layout D: warp w covers TMEM
//     lanes w*32..w*32+31), w7 idle. 3-stage circular pipeline.
//
// Tile shape: BLOCK_M=128, BLOCK_N=256, BLOCK_K=128 (one 128-elem k-block
// per stage), STAGES=3. TMEM: 256 accumulator columns + 4*STAGES SFA + 
// 8*STAGES SFB = 292 -> allocate 320 columns. Shared memory ~149 KiB
// (opt-in via cuFuncSetAttribute, handled by the Rust launcher).
//
// Determinism: fixed grids, fixed issue order, fixed TMEM lane mapping —
// bit-identical results across runs on the same device/driver. Results are
// NOT bit-identical to the CPU reference (different accumulation order);
// the golden tests use tolerances like every other GEMM path.

#define SM100_BM 128
#define SM100_BN 256
#define SM100_BK 128          // one 128-element k block per stage
#define SM100_UMMA_K 32       // e4m3 dense K32 per tcgen05.mma
#define SM100_STAGES 3
#define SM100_THREADS 256     // 8 warps

// TMEM column layout.
#define SM100_TMEM_COLS 320   // rounded to a multiple of 32 (>= 292)
#define SM100_TMEM_ACC 0      // accumulator columns [0, 256)
#define SM100_TMEM_SFA 256    // 4 * STAGES columns
#define SM100_TMEM_SFB 268    // 8 * STAGES columns

// Shared-memory layout (bytes, from the dynamic smem base).
#define SM100_A_OFF 0
#define SM100_B_OFF (SM100_A_OFF + SM100_STAGES * SM100_BM * SM100_BK)
#define SM100_SFA_OFF (SM100_B_OFF + SM100_STAGES * SM100_BN * SM100_BK)
#define SM100_SFB_OFF (SM100_SFA_OFF + SM100_STAGES * 128 * 4)
#define SM100_BAR_OFF (SM100_SFB_OFF + SM100_STAGES * SM100_BN * 4)
#define SM100_TMEMCELL_OFF (SM100_BAR_OFF + 10 * 8)
#define SM100_SMEM_BYTES ((SM100_TMEMCELL_OFF + 15) & ~15)

// TMA bytes per stage (A tile + B tile).
#define SM100_TMA_BYTES ((SM100_BM + SM100_BN) * SM100_BK)

#if !defined(__HIP__)

// Opaque 128-byte CUtensorMap (cuda.h layout; NVRTC has no headers).
struct alignas(64) CUTensorMap { unsigned long long data[16]; };

// ---- PTX helpers (transcribed from the PTX ISA / CUTLASS arch layer) ----

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(bar), "r"(phase));
}

__device__ __forceinline__ void tma_2d_copy(unsigned dst, const void* tmap,
                                            int x, int y, unsigned bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(bar) : "memory");
}

__device__ __forceinline__ void tmem_alloc(unsigned dst_cell, unsigned ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst_cell), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(unsigned taddr, unsigned ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;");
}

// Shared-memory matrix descriptor for tcgen05 (K-major, SWIZZLE_128B).
// bits [0,14): start_address >> 4; bits [32,46): SBO >> 4; bits [46,48):
// version = 1 (SM100); bits [61,64): layout = 2 (SWIZZLE_128B). LBO is
// implicit (one swizzle atom spans the full BLOCK_K).
__device__ __forceinline__ unsigned long long make_smem_desc_128b(unsigned addr) {
    const unsigned sbo = 8u * SM100_BK; // 8-row atoms, 128 B each
    return ((unsigned long long)(addr >> 4))
         | ((unsigned long long)(sbo >> 4) << 32)
         | (1ull << 46)
         | (2ull << 61);
}

// SMEM descriptor for the scale-factor UTCCP source (SWIZZLE_NONE,
// atoms of 8 rows x 16 B, SBO = 128 B, LBO = 0 — DeepGEMM make_sf_desc).
__device__ __forceinline__ unsigned long long make_sf_desc(unsigned addr) {
    return ((unsigned long long)(addr >> 4))
         | ((unsigned long long)((8u * 16u) >> 4) << 32)
         | (1ull << 46);
}

// Instruction descriptor for kind::mxf8f6f4 (CUTLASS
// InstrDescriptorBlockScaled): e4m3 x e4m3 -> f32, K-major A and B, UE8M0
// scale format, M = 128, N = 256, runtime sfid selection.
__device__ __forceinline__ unsigned make_idesc(unsigned sfa_id, unsigned sfb_id) {
    return (0u << 2)                    // dense
         | ((sfb_id & 3u) << 4)         // Matrix B Scale Factor ID
         | (0u << 7)                    // a_format = E4M3
         | (0u << 10)                   // b_format = E4M3
         | (0u << 13) | (0u << 14)      // no negate
         | (0u << 15) | (0u << 16)      // K-major A and B
         | ((SM100_BN >> 3) << 17)      // n_dim
         | (1u << 23)                   // scale format = UE8M0
         | ((SM100_BM >> 4) << 24)      // m_dim
         | ((sfa_id & 3u) << 29)        // Matrix A Scale Factor ID
         | (0u << 31);                  // dense K32
}

// tcgen05.mma kind::mxf8f6f4 with block scales (single-thread issue).
__device__ __forceinline__ void mma_mxf8f6f4(unsigned d_tmem,
                                             unsigned long long a_desc,
                                             unsigned long long b_desc,
                                             unsigned idesc,
                                             unsigned tmem_sfa,
                                             unsigned tmem_sfb,
                                             unsigned accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3,"
        " [%5], [%6], p; \n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate),
           "r"(tmem_sfa), "r"(tmem_sfb));
}

// tcgen05.cp — SMEM -> TMEM scale-factor upload, 32 source rows of 128 b
// broadcast across the four 32-lane groups (CUTLASS
// SM100_UTCCP_4x32dp128bit_1cta).
__device__ __forceinline__ void utccp_sf(unsigned long long sf_desc, unsigned tmem_col) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(tmem_col), "l"(sf_desc));
}

__device__ __forceinline__ void umma_commit(unsigned bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(bar) : "memory");
}

__device__ __forceinline__ void tmem_ld_x32(unsigned taddr, float* v) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
          "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]),
          "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]),
          "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]),
          "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]),
          "=f"(v[20]), "=f"(v[21]), "=f"(v[22]), "=f"(v[23]),
          "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]),
          "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31])
        : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

// ---------------------------------------------------------------------------
// The kernel. Grid = (ceil(N / 256), ceil(M / 128)); block = 256 threads.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(SM100_THREADS)
di_gemm_fp8_nt_sm100a_kernel(const __grid_constant__ CUTensorMap tmap_a,
                              const __grid_constant__ CUTensorMap tmap_b,
                              const unsigned char* __restrict__ a_sf,
                              const unsigned char* __restrict__ b_sf,
                              float* __restrict__ d,
                              int M, int N, int K) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const unsigned smem = smem_u32(smem_raw);
    const unsigned bar_full0 = smem + SM100_BAR_OFF;          // [3] A/B TMA
    const unsigned bar_sffull0 = bar_full0 + 3 * 8;           // [3] SF smem
    const unsigned bar_empty0 = bar_sffull0 + 3 * 8;          // [3] stage retire
    const unsigned bar_tmem = bar_empty0 + 3 * 8;             // [1] accumulator
    const unsigned tmem_cell = smem + SM100_TMEMCELL_OFF;

    const int m0 = (int)blockIdx.y * SM100_BM;
    const int n0 = (int)blockIdx.x * SM100_BN;
    const int kb_blocks = K >> 7;
    const unsigned warp = (int)(threadIdx.x >> 5);
    const unsigned lane = (int)(threadIdx.x & 31);

    // ---- setup: barriers, tensor memory ----
    if (warp == 0 && lane == 0) {
        for (int s = 0; s < SM100_STAGES; ++s) {
            mbar_init(bar_full0 + s * 8, 1);       // TMA thread arrives+expect
            mbar_init(bar_sffull0 + s * 8, 32);    // SF loader warp arrives
            mbar_init(bar_empty0 + s * 8, 1);      // tcgen05.commit arrives
        }
        mbar_init(bar_tmem, 1);                     // final commit arrives
        fence_mbarrier_init();
    } else if (warp == 1) {
        // One full warp allocates the tensor memory block.
        tmem_alloc(tmem_cell, SM100_TMEM_COLS);
    }
    __syncthreads();
    const unsigned tmem_base = *(volatile unsigned*)(
        (const unsigned char*)smem_raw + SM100_TMEMCELL_OFF);

    const int num_kb = kb_blocks; // iterations along K

    // ---- warp 0: TMA producer ----
    if (warp == 0) {
        if (lane == 0) {
            for (int i = 0; i < num_kb; ++i) {
                const int s = i % SM100_STAGES;
                const unsigned round = (unsigned)(i / SM100_STAGES);
                // Stage s reusable? (round 0 passes immediately: parity 1
                // is the "previous" phase of a fresh barrier.)
                mbar_wait(bar_empty0 + s * 8, (round & 1u) ^ 1u);
                const unsigned a_dst = smem + SM100_A_OFF + s * (SM100_BM * SM100_BK);
                const unsigned b_dst = smem + SM100_B_OFF + s * (SM100_BN * SM100_BK);
                // Coordinates: inner (k elements) first, then rows.
                tma_2d_copy(a_dst, &tmap_a, i * SM100_BK, m0, bar_full0 + s * 8);
                tma_2d_copy(b_dst, &tmap_b, i * SM100_BK, n0, bar_full0 + s * 8);
                mbar_arrive_expect_tx(bar_full0 + s * 8, SM100_TMA_BYTES);
            }
        }
    }
    // ---- warp 1: MMA issue ----
    else if (warp == 1) {
        for (int i = 0; i < num_kb; ++i) {
            const int s = i % SM100_STAGES;
            const unsigned round = (unsigned)(i / SM100_STAGES);
            const unsigned parity = round & 1u;
            mbar_wait(bar_full0 + s * 8, parity);
            mbar_wait(bar_sffull0 + s * 8, parity);
            // The stage-s TMEM scale columns are read by round r-1's MMAs;
            // wait for them to retire before overwriting (round 0 passes).
            mbar_wait(bar_empty0 + s * 8, parity ^ 1u);
            tcgen05_after_thread_sync();
            if (lane == 0) {
                // Scale factors -> TMEM. SFA covers 128 lines (1 x 4 cols);
                // SFB covers 256 lines (2 x 4 cols).
                utccp_sf(make_sf_desc(smem + SM100_SFA_OFF + s * 128 * 4),
                         tmem_base + SM100_TMEM_SFA + 4 * s);
                utccp_sf(make_sf_desc(smem + SM100_SFB_OFF + s * SM100_BN * 4),
                         tmem_base + SM100_TMEM_SFB + 8 * s);
                utccp_sf(make_sf_desc(smem + SM100_SFB_OFF + s * SM100_BN * 4 + 512),
                         tmem_base + SM100_TMEM_SFB + 8 * s + 4);
                // 4 MMAs of K=32 over the 128-element k block.
                const unsigned a_base = smem + SM100_A_OFF + s * (SM100_BM * SM100_BK);
                const unsigned b_base = smem + SM100_B_OFF + s * (SM100_BN * SM100_BK);
                #pragma unroll
                for (int kk = 0; kk < SM100_BK / SM100_UMMA_K; ++kk) {
                    const unsigned off = kk * SM100_UMMA_K; // bytes (e4m3)
                    const unsigned sfid = off / 32u;
                    mma_mxf8f6f4(
                        tmem_base + SM100_TMEM_ACC,
                        make_smem_desc_128b(a_base + off),
                        make_smem_desc_128b(b_base + off),
                        make_idesc(sfid, sfid),
                        tmem_base + SM100_TMEM_SFA + 4 * s,
                        tmem_base + SM100_TMEM_SFB + 8 * s,
                        (i > 0) ? 1u : 0u);
                }
                umma_commit(bar_empty0 + s * 8);
                if (i == num_kb - 1) umma_commit(bar_tmem);
            }
            __syncwarp();
        }
    }
    // ---- warp 2: scale loader + transposer ----
    else if (warp == 2) {
        for (int i = 0; i < num_kb; ++i) {
            const int s = i % SM100_STAGES;
            const unsigned round = (unsigned)(i / SM100_STAGES);
            const unsigned parity = round & 1u;
            // SF smem of stage s reusable (round 0 passes).
            mbar_wait(bar_empty0 + s * 8, parity ^ 1u);
            const int kb = i;
            // SFA: [32][4] words; word (l, q) = replicate(a_sf[(q*32+l)][kb]).
            unsigned* sfa = (unsigned*)(smem_raw + SM100_SFA_OFF + s * 512);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int m = q * 32 + (int)lane;
                unsigned char b0 = 0;
                if (m < M) b0 = a_sf[(u64)m * kb_blocks + kb];
                sfa[lane * 4 + q] = (unsigned)b0 * 0x01010101u;
            }
            // SFB: 256 lines = 2 groups of [32][4];
            // word (g, l, q) = replicate(b_sf[((g*128 + q*32 + l) >> 7)][kb]).
            unsigned* sfb = (unsigned*)(smem_raw + SM100_SFB_OFF + s * SM100_BN * 4);
            #pragma unroll
            for (int g = 0; g < SM100_BN / 128; ++g) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int n = g * 128 + q * 32 + (int)lane;
                    unsigned char b0 = 0;
                    if (n < N) b0 = b_sf[(u64)(n >> 7) * kb_blocks + kb];
                    sfb[g * 128 + lane * 4 + q] = (unsigned)b0 * 0x01010101u;
                }
            }
            __syncwarp();
            fence_view_async_shared();
            mbar_arrive(bar_sffull0 + s * 8);
        }
    }
    // ---- warps 3..6: epilogue (Layout D; group-local warp = TMEM lanes) ----
    else if (warp >= 3 && warp <= 6) {
        mbar_wait(bar_tmem, 0);
        tcgen05_after_thread_sync();
        const int ew = (int)warp - 3; // 0..3
        const int row = m0 + ew * 32 + (int)lane;
        #pragma unroll 1
        for (int c = 0; c < SM100_BN; c += 32) {
            float v[32];
            const unsigned taddr = tmem_base + SM100_TMEM_ACC
                                 + ((unsigned)(ew * 32) << 16)
                                 + (unsigned)c;
            tmem_ld_x32(taddr, v);
            tmem_ld_wait();
            if (row < M) {
                float* drow = d + (u64)row * N + n0 + c;
                const int ncols = N - (n0 + c);
                if (ncols >= 32 && ((N & 3) == 0)) {
                    // 8 x 16 B vector stores (drow is 16 B aligned: the base
                    // is 256 B aligned, row*N*4 keeps it when N % 4 == 0,
                    // and n0 + c is a multiple of 32).
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        *(int4*)(drow + j * 4) =
                            *(const int4*)(const void*)(v + j * 4);
                    }
                } else {
                    for (int j = 0; j < 32 && j < ncols; ++j) drow[j] = v[j];
                }
            }
        }
        tcgen05_before_thread_sync();
    }

    __syncthreads();
    // One full warp returns the tensor memory (warp 1 allocated it).
    if (warp == 1) tmem_dealloc(tmem_base, SM100_TMEM_COLS);
}

#endif // !__HIP__
