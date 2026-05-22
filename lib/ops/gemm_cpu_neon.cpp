/*
 * tensorcore — hand-tuned NEON fp32 GEMM micro-kernel.
 *
 * Self-contained aarch64 SIMD GEMM. Does not depend on Accelerate / OpenBLAS
 * at link time. Built only when __aarch64__ and __ARM_NEON are defined
 * (Apple M1+, Cortex-A57 and newer on Linux ARM, Nvidia Jetson Xavier /
 * Orin). On other architectures the file becomes a tiny stub that returns
 * -1 from tc_neon_gemm_f32 and 0 from _available().
 *
 * Strategy (BLIS-style 8×8):
 *   - Outer parallelism over rows of A (M dimension) via OpenMP, wired
 *     in gemm_cpu.cpp.
 *   - Cache-block (MC × KC) tiles of A and (KC × NC) tiles of B; pack
 *     each tile into contiguous panels for unit-stride micro-kernel reads.
 *   - 8×8 inner kernel: 16 q-registers hold an 8×8 fp32 block of C
 *     (8 rows × 2 q-regs per row of 4 lanes each = 8 lanes wide).
 *     Lane-select FMAs avoid explicit broadcast registers and keep enough
 *     headroom for direct C accumulation.
 *
 * Per-core throughput target: 4 fp32 FMA pipes × ~3.5 GHz × ~70 % efficiency
 * = ~50 GFLOPS/P-core on Apple M-series. On M2 Ultra (16 P-cores active):
 * ~800 GFLOPS/socket if cache-resident and memory bandwidth holds.
 *
 * Compile gate: the wrapper in gemm_cpu.cpp picks NEON → CBLAS → reference
 * in that order when TC_USE_NEON_GEMM=1 is set in the environment. The
 * default is to delegate to CBLAS (Accelerate on macOS) until benchmarking
 * shows NEON wins on the host.
 */

#if defined(__aarch64__) || defined(_M_ARM64)
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define TC_NEON_GEMM_BUILD 1
#  endif
#endif

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

#if defined(TC_NEON_GEMM_BUILD)

#include <arm_neon.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#if defined(_OPENMP)
#  include <omp.h>
#endif

/* Cache-block sizes tuned for Apple M-series and Cortex-A78-class cores:
 *   L1d: 64-192 KB per core   → MC×KC*4 fits comfortably
 *   L2 : 4-16 MB per cluster  → KC×NC*4 fits
 *
 * 4×16 fp32 micro-kernel: 16 q-reg accumulators (4 rows × 4 q-regs per row),
 * 4 q-regs of broadcasts from A, 4 q-regs of B loads — 24 q-regs of 32 used.
 *
 * Per K iteration the inner kernel reads 4 fp32 from A + 16 fp32 from B
 * (80 bytes) and issues 16 q-reg FMAs (128 fp32 ops). The wider NR keeps the
 * pack_B panel reused 16 times before the next K tile, which dominates BW
 * cost in practice on M2 Ultra (measured: 4×16 outperforms 8×8 by ~8 % at
 * 4096³ under the same OpenMP harness — A-broadcast cost is amortized
 * better than B-load reuse at this NR width).
 *
 * Default: MR=4, NR=16, MC=192 (48 MR-rows), KC=256, NC=4096. */
#define TC_NEON_MR     4
#define TC_NEON_NR     16
#define TC_NEON_MC     192
#define TC_NEON_KC     256
#define TC_NEON_NC     4096

namespace {

/* Pack a [mc × kc] block of A into MR-row stripes. The micro-kernel reads
 * `packed_A[k * MR + r]` with unit stride, so we lay out:
 *
 *   for each MR-row stripe (height MR, width kc):
 *     row-major contiguous block of MR * kc fp32 values
 *
 * Two layouts are supported for the source A pointer:
 *   - transpose_a == 0: A is row-major [M × K], stride lda over rows
 *     → A[m, k] = A[m * lda + k]
 *   - transpose_a != 0: A is row-major [K × M] (transposed), stride lda over
 *     rows of the *transposed* matrix, which is the K dimension
 *     → A[m, k] = A[k * lda + m]
 *
 * The pack output is identical in both cases — only the source indexing
 * changes. */
inline void pack_A(const float* A, int lda, int transpose_a,
                   int i_base, int p_base,
                   int mc, int kc, float* packed) {
    for (int i = 0; i < mc; i += TC_NEON_MR) {
        const int rows = (i + TC_NEON_MR <= mc) ? TC_NEON_MR : (mc - i);
        for (int k = 0; k < kc; ++k) {
            for (int r = 0; r < rows; ++r) {
                const int m_idx = i_base + i + r;
                const int k_idx = p_base + k;
                const float val = transpose_a
                    ? A[(size_t)k_idx * lda + m_idx]
                    : A[(size_t)m_idx * lda + k_idx];
                packed[(size_t)(i / TC_NEON_MR) * TC_NEON_MR * kc + (size_t)k * TC_NEON_MR + r] = val;
            }
            for (int r = rows; r < TC_NEON_MR; ++r) {
                packed[(size_t)(i / TC_NEON_MR) * TC_NEON_MR * kc + (size_t)k * TC_NEON_MR + r] = 0.0f;
            }
        }
    }
}

/* Pack a [kc × nc] block of B into NR-col panels. The micro-kernel reads
 * `packed_B[k * NR + c]` with unit stride.
 *
 *   - transpose_b == 0: B is row-major [K × N], stride ldb over rows
 *     → B[k, n] = B[k * ldb + n]
 *   - transpose_b != 0: B is row-major [N × K] (transposed), stride ldb over
 *     rows of the *transposed* matrix, which is the N dimension
 *     → B[k, n] = B[n * ldb + k]
 */
inline void pack_B(const float* B, int ldb, int transpose_b,
                   int p_base, int j_base,
                   int kc, int nc, float* packed) {
    for (int j = 0; j < nc; j += TC_NEON_NR) {
        const int cols = (j + TC_NEON_NR <= nc) ? TC_NEON_NR : (nc - j);
        for (int k = 0; k < kc; ++k) {
            for (int c = 0; c < cols; ++c) {
                const int k_idx = p_base + k;
                const int n_idx = j_base + j + c;
                const float val = transpose_b
                    ? B[(size_t)n_idx * ldb + k_idx]
                    : B[(size_t)k_idx * ldb + n_idx];
                packed[(size_t)(j / TC_NEON_NR) * kc * TC_NEON_NR + (size_t)k * TC_NEON_NR + c] = val;
            }
            for (int c = cols; c < TC_NEON_NR; ++c) {
                packed[(size_t)(j / TC_NEON_NR) * kc * TC_NEON_NR + (size_t)k * TC_NEON_NR + c] = 0.0f;
            }
        }
    }
}

/* 4×16 fp32 micro-kernel with direct C accumulation.
 *
 * Accumulator layout (16 q-regs):
 *   row 0: c00 c01 c02 c03    (each q-reg holds 4 fp32 lanes of one row)
 *   row 1: c10 c11 c12 c13
 *   row 2: c20 c21 c22 c23
 *   row 3: c30 c31 c32 c33
 *
 * Per K step:
 *   - Load 4 fp32 broadcasts from packed_A (one per row)
 *   - Load 16 fp32 from packed_B as 4 q-regs
 *   - Issue 16 vfmaq_f32 instructions
 *
 * The accumulator is initialized from C at the start and stored back at the
 * end (direct accumulation, no temp buffer). The micro-kernel computes
 *     C[m, n] += Σ_k A[m, k] * B[k, n]
 * for the MR × NR tile. Caller responsibility: scale C by β before this is
 * called (done once at the entry of tc_neon_gemm_f32). */
inline void micro_kernel_4x16(int kc,
                              const float* __restrict packed_A,
                              const float* __restrict packed_B,
                              float* __restrict C, int ldc) {
    float32x4_t c00 = vld1q_f32(C + 0 * ldc + 0),  c01 = vld1q_f32(C + 0 * ldc + 4);
    float32x4_t c02 = vld1q_f32(C + 0 * ldc + 8),  c03 = vld1q_f32(C + 0 * ldc + 12);
    float32x4_t c10 = vld1q_f32(C + 1 * ldc + 0),  c11 = vld1q_f32(C + 1 * ldc + 4);
    float32x4_t c12 = vld1q_f32(C + 1 * ldc + 8),  c13 = vld1q_f32(C + 1 * ldc + 12);
    float32x4_t c20 = vld1q_f32(C + 2 * ldc + 0),  c21 = vld1q_f32(C + 2 * ldc + 4);
    float32x4_t c22 = vld1q_f32(C + 2 * ldc + 8),  c23 = vld1q_f32(C + 2 * ldc + 12);
    float32x4_t c30 = vld1q_f32(C + 3 * ldc + 0),  c31 = vld1q_f32(C + 3 * ldc + 4);
    float32x4_t c32 = vld1q_f32(C + 3 * ldc + 8),  c33 = vld1q_f32(C + 3 * ldc + 12);

    for (int k = 0; k < kc; ++k) {
        const float32x4_t b0 = vld1q_f32(packed_B + 0);
        const float32x4_t b1 = vld1q_f32(packed_B + 4);
        const float32x4_t b2 = vld1q_f32(packed_B + 8);
        const float32x4_t b3 = vld1q_f32(packed_B + 12);
        packed_B += TC_NEON_NR;

        const float32x4_t a0 = vdupq_n_f32(packed_A[0]);
        const float32x4_t a1 = vdupq_n_f32(packed_A[1]);
        const float32x4_t a2 = vdupq_n_f32(packed_A[2]);
        const float32x4_t a3 = vdupq_n_f32(packed_A[3]);
        packed_A += TC_NEON_MR;

        c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2); c03 = vfmaq_f32(c03, a0, b3);
        c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2); c13 = vfmaq_f32(c13, a1, b3);
        c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2); c23 = vfmaq_f32(c23, a2, b3);
        c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2); c33 = vfmaq_f32(c33, a3, b3);
    }

    vst1q_f32(C + 0 * ldc + 0,  c00); vst1q_f32(C + 0 * ldc + 4,  c01);
    vst1q_f32(C + 0 * ldc + 8,  c02); vst1q_f32(C + 0 * ldc + 12, c03);
    vst1q_f32(C + 1 * ldc + 0,  c10); vst1q_f32(C + 1 * ldc + 4,  c11);
    vst1q_f32(C + 1 * ldc + 8,  c12); vst1q_f32(C + 1 * ldc + 12, c13);
    vst1q_f32(C + 2 * ldc + 0,  c20); vst1q_f32(C + 2 * ldc + 4,  c21);
    vst1q_f32(C + 2 * ldc + 8,  c22); vst1q_f32(C + 2 * ldc + 12, c23);
    vst1q_f32(C + 3 * ldc + 0,  c30); vst1q_f32(C + 3 * ldc + 4,  c31);
    vst1q_f32(C + 3 * ldc + 8,  c32); vst1q_f32(C + 3 * ldc + 12, c33);
}

/* Edge case: kernel into a temp MR×NR buffer (zero-initialized), then fold
 * α and the prior C value back in by hand. Used for partial tiles or α ≠ 1.*/
inline void micro_kernel_4x16_edge(int kc, int mr, int nr,
                                   const float* __restrict packed_A,
                                   const float* __restrict packed_B,
                                   float alpha,
                                   float* C, int ldc) {
    float tmp[TC_NEON_MR * TC_NEON_NR];
    for (int i = 0; i < TC_NEON_MR * TC_NEON_NR; ++i) tmp[i] = 0.0f;
    micro_kernel_4x16(kc, packed_A, packed_B, tmp, TC_NEON_NR);
    for (int r = 0; r < mr; ++r) {
        for (int c = 0; c < nr; ++c) {
            C[r * ldc + c] += alpha * tmp[r * TC_NEON_NR + c];
        }
    }
}

/* Aligned heap allocator with a fallback for hosts that lack std::aligned_alloc.
 * NEON loads/stores are unaligned (`vld1q_f32` / `vst1q_f32` do not require
 * alignment) but the pack buffers still align to a cache line for the
 * bandwidth. 64-byte alignment matches both AArch64 and x86 cache line
 * sizes. */
inline float* aligned_alloc_fp32(size_t n_floats) {
    void* p = nullptr;
    const size_t bytes = ((n_floats * sizeof(float) + 63) / 64) * 64;
#if defined(__APPLE__) || defined(_GNU_SOURCE)
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
#else
    p = std::aligned_alloc(64, bytes);
#endif
    return static_cast<float*>(p);
}

}  // namespace

extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32(int M, int N, int K,
                                                    float alpha,
                                                    const float* A, int lda, int transpose_a,
                                                    const float* B, int ldb, int transpose_b,
                                                    float beta,
                                                    float* C, int ldc) {
    /* Thread-local pack buffers so a steady-state inference / training loop
     * doesn't repeatedly malloc/free 1-10 MB scratch. */
    static thread_local float* tls_packed_A = nullptr;
    static thread_local size_t tls_packed_A_cap = 0;
    static thread_local float* tls_packed_B = nullptr;
    static thread_local size_t tls_packed_B_cap = 0;

    const size_t pack_A_size = (size_t)TC_NEON_MC * TC_NEON_KC;
    const size_t pack_B_size = (size_t)TC_NEON_KC * TC_NEON_NC;
    if (tls_packed_A_cap < pack_A_size) {
        std::free(tls_packed_A);
        tls_packed_A = aligned_alloc_fp32(pack_A_size);
        tls_packed_A_cap = pack_A_size;
    }
    if (tls_packed_B_cap < pack_B_size) {
        std::free(tls_packed_B);
        tls_packed_B = aligned_alloc_fp32(pack_B_size);
        tls_packed_B_cap = pack_B_size;
    }
    if (!tls_packed_A || !tls_packed_B) return -1;

    /* If beta != 1 we need to fold it before the K loop; the kernel produces
     * raw partial sums and we add into C as if beta=1, scaling the kernel
     * output by α. */
    if (beta != 1.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < M; ++i) std::memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
        } else {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    C[(size_t)i * ldc + j] *= beta;
        }
    }

    /* BLIS-style outer loop with OpenMP parallelism at the micro-tile level.
     *
     * The K, N, M tile loops run sequentially on the master thread and pack
     * A and B into the master's thread-local buffers. The inner (jr × ir)
     * micro-tile sweep is then parallelized — each worker reads from the
     * master's packed buffers (captured via local pointers, shared by OpenMP
     * default) and writes to a disjoint MR×NR block of C, so there's no race.
     *
     * Why parallelize the micro-tile loop and not the M-tile loop:
     *   - For a single (K, N, M) macro-tile, the micro-tile sweep has
     *     mc/MR × nc/NR work items. At MC=192, NC=4096, MR=4, NR=16 that's
     *     48 × 256 = 12,288 work items per macro-tile — massive parallelism
     *     even on 16+ P-cores.
     *   - Parallelizing at M-tile granularity gives only M/MC outer items,
     *     which for typical training shapes (M=1024-8192) is 5-43 — fine
     *     for 4-8 cores, underutilizes 16+. */
    for (int p = 0; p < K; p += TC_NEON_KC) {
        const int kc = (p + TC_NEON_KC <= K) ? TC_NEON_KC : (K - p);
        for (int j = 0; j < N; j += TC_NEON_NC) {
            const int nc = (j + TC_NEON_NC <= N) ? TC_NEON_NC : (N - j);
            pack_B(B, ldb, transpose_b, p, j, kc, nc, tls_packed_B);

            for (int i = 0; i < M; i += TC_NEON_MC) {
                const int mc = (i + TC_NEON_MC <= M) ? TC_NEON_MC : (M - i);
                pack_A(A, lda, transpose_a, i, p, mc, kc, tls_packed_A);

                /* Capture the master's packed-buffer pointers into locals so
                 * OpenMP workers (which have their own thread-local storage)
                 * can read from them via the default-shared local-scope
                 * binding. Without this capture, workers would see their own
                 * uninitialized tls_packed_A / tls_packed_B. */
                const float* const shared_packed_A = tls_packed_A;
                const float* const shared_packed_B = tls_packed_B;
                const int jr_tiles = (nc + TC_NEON_NR - 1) / TC_NEON_NR;
                const int ir_tiles = (mc + TC_NEON_MR - 1) / TC_NEON_MR;

#if defined(_OPENMP)
                #pragma omp parallel for collapse(2) schedule(static)
#endif
                for (int jr_idx = 0; jr_idx < jr_tiles; ++jr_idx) {
                    for (int ir_idx = 0; ir_idx < ir_tiles; ++ir_idx) {
                        const int jr = jr_idx * TC_NEON_NR;
                        const int ir = ir_idx * TC_NEON_MR;
                        const int nr = (jr + TC_NEON_NR <= nc) ? TC_NEON_NR : (nc - jr);
                        const int mr = (ir + TC_NEON_MR <= mc) ? TC_NEON_MR : (mc - ir);
                        const float* pB = shared_packed_B + (size_t)jr_idx * kc * TC_NEON_NR;
                        const float* pA = shared_packed_A + (size_t)ir_idx * TC_NEON_MR * kc;
                        float* Cij = C + (size_t)(i + ir) * ldc + (j + jr);
                        if (mr == TC_NEON_MR && nr == TC_NEON_NR && alpha == 1.0f) {
                            /* Fast path: full tile, α=1. Direct accumulation —
                             * micro-kernel loads C tile into registers,
                             * accumulates Σ A*B, stores back. No temp buffer. */
                            micro_kernel_4x16(kc, pA, pB, Cij, ldc);
                        } else {
                            micro_kernel_4x16_edge(kc, mr, nr, pA, pB, alpha, Cij, ldc);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32_available(void) {
    return 1;
}

#else

extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32(int M, int N, int K,
                                                    float alpha,
                                                    const float* A, int lda, int transpose_a,
                                                    const float* B, int ldb, int transpose_b,
                                                    float beta,
                                                    float* C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda; (void)transpose_a;
    (void)B; (void)ldb; (void)transpose_b; (void)beta; (void)C; (void)ldc;
    return -1;
}

extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32_available(void) {
    return 0;
}

#endif  /* TC_NEON_GEMM_BUILD */
