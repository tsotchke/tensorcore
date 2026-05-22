/*
 * tensorcore — hand-tuned AVX2 + FMA fp32 GEMM micro-kernel.
 *
 * Self-contained x86_64 SIMD GEMM. Does not depend on OpenBLAS / MKL /
 * Accelerate at link time. Built only when __AVX2__ and __FMA__ are defined
 * (Haswell-EP and newer; Sandy Bridge / Ivy Bridge fall back to the reference
 * loop or to CBLAS if available).
 *
 * Strategy (BLIS-style 6×16):
 *   - Outer parallelism over rows of A (M dimension) via OpenMP.
 *   - Cache-block (MC×KC) tiles of A and (KC×NC) tiles of B; pack each tile
 *     into contiguous row-major / column-major buffers so the micro-kernel
 *     reads with unit stride.
 *   - 6×16 inner kernel: 12 ymm registers hold a 6×16 fp32 block of C;
 *     6 broadcast-loads from A and 2 loads from B per K step; one vfmadd231
 *     per accumulator.
 *
 * Per-core throughput target: 32 fp32 FMA/cycle × 2.2 GHz × ~70% efficiency
 * = ~50 GFLOPS/core. On a 22-core Xeon E5-2699 v4: ~1.1 TFLOPS/socket =
 * ~2.2 TFLOPS dual-socket if memory bandwidth holds.
 *
 * Compile gate: lives behind TC_HAS_AVX2_GEMM in CMakeLists. The wrapper in
 * gemm_cpu.cpp picks AVX2 → CBLAS → reference in that order.
 */

#if defined(__x86_64__) || defined(_M_X64)
#  if defined(__AVX2__) && defined(__FMA__)
#    define TC_AVX2_GEMM_BUILD 1
#  endif
#endif

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

#if defined(TC_AVX2_GEMM_BUILD)

#include "tensorcore/tensorcore.h"

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

/* Cache-block sizes tuned for Haswell-EP / Broadwell-EP / Skylake-SP:
 *   L1d: 32 KB per core   → MR×KC*4 fits comfortably
 *   L2 : 256 KB per core  → MC×KC*4 + NR*KC*4 should fit
 *   L3 : shared            → NC×KC*4 fits per-socket
 *
 * Default: MR=6, NR=16, MC=144 (24 MR-rows), KC=256, NC=4096. */
#define TC_AVX2_MR     6
#define TC_AVX2_NR     16
#define TC_AVX2_MC     144
#define TC_AVX2_KC     256
#define TC_AVX2_NC     4096

namespace {

/* Pack a [MC × KC] block of A (M-stride lda, K-stride 1) into a panel that
 * the micro-kernel reads in MR-row stripes. Layout:
 *
 *   for each MR-row stripe (height MR, width KC):
 *     row-major contiguous block of MR*KC fp32 values
 *
 * This lets the inner kernel broadcast-load A[mr_idx][k] with unit stride
 * across k. */
inline void pack_A(const float* A, int lda, int mc, int kc, float* packed) {
    for (int i = 0; i < mc; i += TC_AVX2_MR) {
        const int rows = (i + TC_AVX2_MR <= mc) ? TC_AVX2_MR : (mc - i);
        for (int k = 0; k < kc; ++k) {
            for (int r = 0; r < rows; ++r) {
                packed[(size_t)(i / TC_AVX2_MR) * TC_AVX2_MR * kc + (size_t)k * TC_AVX2_MR + r]
                    = A[(size_t)(i + r) * lda + k];
            }
            /* zero-fill the partial MR row when we're at the bottom edge */
            for (int r = rows; r < TC_AVX2_MR; ++r) {
                packed[(size_t)(i / TC_AVX2_MR) * TC_AVX2_MR * kc + (size_t)k * TC_AVX2_MR + r] = 0.0f;
            }
        }
    }
}

/* Pack a [KC × NC] block of B into NR-col panels:
 *
 *   for each NR-col panel (height KC, width NR):
 *     row-major contiguous block of KC*NR fp32 values
 *
 * Inner kernel does two ymm loads from B at each k step (NR=16 = 2 ymm). */
inline void pack_B(const float* B, int ldb, int kc, int nc, float* packed) {
    for (int j = 0; j < nc; j += TC_AVX2_NR) {
        const int cols = (j + TC_AVX2_NR <= nc) ? TC_AVX2_NR : (nc - j);
        for (int k = 0; k < kc; ++k) {
            for (int c = 0; c < cols; ++c) {
                packed[(size_t)(j / TC_AVX2_NR) * kc * TC_AVX2_NR + (size_t)k * TC_AVX2_NR + c]
                    = B[(size_t)k * ldb + (j + c)];
            }
            for (int c = cols; c < TC_AVX2_NR; ++c) {
                packed[(size_t)(j / TC_AVX2_NR) * kc * TC_AVX2_NR + (size_t)k * TC_AVX2_NR + c] = 0.0f;
            }
        }
    }
}

/* 6×16 fp32 micro-kernel. Computes C[0:6][0:16] += A[0:6][0:kc] * B[0:kc][0:16]
 * where A is packed in MR-row stripes (one stripe of MR*kc starting at packed_A)
 * and B is packed in NR-col panels (one panel of kc*NR starting at packed_B). */
inline void micro_kernel_6x16(int kc,
                              const float* __restrict packed_A,
                              const float* __restrict packed_B,
                              float* __restrict C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        /* Load B's 16-wide row at this k. */
        const __m256 b0 = _mm256_loadu_ps(packed_B + 0);
        const __m256 b1 = _mm256_loadu_ps(packed_B + 8);
        packed_B += TC_AVX2_NR;

        /* Broadcast A's 6 values at this k. */
        const __m256 a0 = _mm256_broadcast_ss(packed_A + 0);
        const __m256 a1 = _mm256_broadcast_ss(packed_A + 1);
        const __m256 a2 = _mm256_broadcast_ss(packed_A + 2);
        const __m256 a3 = _mm256_broadcast_ss(packed_A + 3);
        const __m256 a4 = _mm256_broadcast_ss(packed_A + 4);
        const __m256 a5 = _mm256_broadcast_ss(packed_A + 5);
        packed_A += TC_AVX2_MR;

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    /* Store. The kernel zero-initializes C accumulators; caller adds α and β. */
    _mm256_storeu_ps(C + 0 * ldc + 0, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc + 0, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc + 0, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

/* Edge case: kernel into a temp 6×16 buffer when the C tile is partial. */
inline void micro_kernel_6x16_edge(int kc, int mr, int nr,
                                   const float* __restrict packed_A,
                                   const float* __restrict packed_B,
                                   float alpha, float beta,
                                   float* C, int ldc) {
    float tmp[TC_AVX2_MR * TC_AVX2_NR];
    micro_kernel_6x16(kc, packed_A, packed_B, tmp, TC_AVX2_NR);
    /* Write back with α/β handling and edge clamping. */
    for (int r = 0; r < mr; ++r) {
        for (int c = 0; c < nr; ++c) {
            const float prev = beta != 0.0f ? C[r * ldc + c] : 0.0f;
            C[r * ldc + c] = alpha * tmp[r * TC_AVX2_NR + c] + beta * prev;
        }
    }
}

/* Aligned heap allocator with a fallback for hosts that lack std::aligned_alloc.
 * BLIS micro-kernels rely on 32-byte alignment for AVX2 ymm loads — we use
 * unaligned loads (`_mm256_loadu_ps`) above so misalignment is forgiven, but
 * the pack buffers still align to a cache line for the bandwidth. */
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

extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda,
                                                   const float* B, int ldb,
                                                   float beta,
                                                   float* C, int ldc) {
    /* Allocate pack buffers from thread-local pools so a steady-state inference
     * loop doesn't repeatedly malloc/free 1-50 MB scratch. */
    static thread_local float* tls_packed_A = nullptr;
    static thread_local size_t tls_packed_A_cap = 0;
    static thread_local float* tls_packed_B = nullptr;
    static thread_local size_t tls_packed_B_cap = 0;

    const size_t pack_A_size = (size_t)TC_AVX2_MC * TC_AVX2_KC;
    const size_t pack_B_size = (size_t)TC_AVX2_KC * TC_AVX2_NC;
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

    /* If beta != 0 we need to read C; for simplicity, scale C up-front then
     * accumulate from the kernel as if beta=1, scaling the kernel output by α. */
    if (beta != 1.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < M; ++i) std::memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
        } else {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    C[(size_t)i * ldc + j] *= beta;
        }
    }

    /* Outer M loop is parallelizable; the OpenMP wrapper in gemm_cpu.cpp does
     * that. Here we run sequentially per-thread on a sub-range; the caller
     * pre-partitions M before calling. */
    for (int p = 0; p < K; p += TC_AVX2_KC) {
        const int kc = (p + TC_AVX2_KC <= K) ? TC_AVX2_KC : (K - p);
        for (int j = 0; j < N; j += TC_AVX2_NC) {
            const int nc = (j + TC_AVX2_NC <= N) ? TC_AVX2_NC : (N - j);
            pack_B(B + (size_t)p * ldb + j, ldb, kc, nc, tls_packed_B);

            for (int i = 0; i < M; i += TC_AVX2_MC) {
                const int mc = (i + TC_AVX2_MC <= M) ? TC_AVX2_MC : (M - i);
                pack_A(A + (size_t)i * lda + p, lda, mc, kc, tls_packed_A);

                for (int jr = 0; jr < nc; jr += TC_AVX2_NR) {
                    const int nr = (jr + TC_AVX2_NR <= nc) ? TC_AVX2_NR : (nc - jr);
                    const float* pB = tls_packed_B + (size_t)(jr / TC_AVX2_NR) * kc * TC_AVX2_NR;
                    for (int ir = 0; ir < mc; ir += TC_AVX2_MR) {
                        const int mr = (ir + TC_AVX2_MR <= mc) ? TC_AVX2_MR : (mc - ir);
                        const float* pA = tls_packed_A + (size_t)(ir / TC_AVX2_MR) * TC_AVX2_MR * kc;
                        float* Cij = C + (size_t)(i + ir) * ldc + (j + jr);
                        if (mr == TC_AVX2_MR && nr == TC_AVX2_NR && alpha == 1.0f) {
                            /* Fast path: full tile, alpha=1.
                             * The kernel zero-inits and stores; here we need
                             * to accumulate into C, so we temp + add. */
                            float tmp[TC_AVX2_MR * TC_AVX2_NR];
                            micro_kernel_6x16(kc, pA, pB, tmp, TC_AVX2_NR);
                            for (int r = 0; r < TC_AVX2_MR; ++r) {
                                __m256 t0 = _mm256_loadu_ps(tmp + r * TC_AVX2_NR);
                                __m256 t1 = _mm256_loadu_ps(tmp + r * TC_AVX2_NR + 8);
                                __m256 c0 = _mm256_loadu_ps(Cij + r * ldc);
                                __m256 c1 = _mm256_loadu_ps(Cij + r * ldc + 8);
                                _mm256_storeu_ps(Cij + r * ldc,     _mm256_add_ps(t0, c0));
                                _mm256_storeu_ps(Cij + r * ldc + 8, _mm256_add_ps(t1, c1));
                            }
                        } else {
                            micro_kernel_6x16_edge(kc, mr, nr, pA, pB, alpha, 1.0f, Cij, ldc);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32_available(void) {
    return 1;
}

#else

extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda,
                                                   const float* B, int ldb,
                                                   float beta,
                                                   float* C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}

extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32_available(void) {
    return 0;
}

#endif  /* TC_AVX2_GEMM_BUILD */
