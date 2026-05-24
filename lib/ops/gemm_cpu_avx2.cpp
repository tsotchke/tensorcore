/*
 * tensorcore — hand-tuned AVX2 + FMA fp32 GEMM micro-kernel.
 *
 * Self-contained x86_64 SIMD GEMM. Does not depend on OpenBLAS / MKL /
 * Accelerate at link time. Built only when __AVX2__ and __FMA__ are defined
 * (Haswell-EP and newer; Sandy Bridge / Ivy Bridge fall back to the reference
 * loop or to CBLAS if available).
 *
 * Strategy (BLIS-style 6×16):
 *   - OpenMP fanout over the independent micro-kernel tile grid after A/B
 *     panels are packed once into shared read-only buffers.
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
#include <algorithm>
#include <mutex>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Cache-block sizes tuned for Haswell-EP / Broadwell-EP / Skylake-SP:
 *   L1d: 32 KB per core   → MR×KC*4 fits comfortably
 *   L2 : 256 KB per core  → MC×KC*4 + NR*KC*4 should fit
 *   L3 : shared            → NC×KC*4 fits per-socket
 *
 * With OpenMP outer parallelism over M, the macro-kernel packs B once for the
 * current K/N panel and gives each worker a private A panel. That avoids the
 * pathological "pack B per worker" variant while also avoiding an OpenMP
 * region per tiny micro-tile block.
 *
 * Default: MR=6, NR=16, MC=72 (12 MR-rows), KC=256, NC=512. */
#define TC_AVX2_MR     6
#define TC_AVX2_NR     16
#define TC_AVX2_MC     72
#define TC_AVX2_KC     256
#define TC_AVX2_NC     512

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

/* BLIS-style macro-kernel with shared pack buffers. The OpenMP parallel
 * region lives INSIDE this function around the inner micro-kernel tile
 * grid, not as an outer wrapper. The pack work happens once per
 * (K-tile, N-tile, M-tile) on the caller thread; workers only read packed
 * panels and write disjoint C tiles.
 *
 * Why not the simpler "split M across threads" approach: each thread
 * would independently pack B for the full N, multiplying the pack work
 * and total memory traffic linearly with thread count. On 44 threads at
 * 4096³ that's 44 × 64 MB ≈ 2.8 GB of redundant pack writes — drops
 * throughput from 81 GFLOPS (1 thread) to 31 GFLOPS (44 threads). */
static int tc_avx2_gemm_f32_slice(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta,
                                  float* C, int ldc) {
    /* Single shared pack buffer (no thread_local — those would be per-OMP-thread
     * and cause cache thrashing as we observed). Persist across calls and
     * guard against concurrent public tc_gemm callers using the AVX2 opt-in
     * path while still allowing OpenMP workers inside one call. */
    static std::mutex shared_pack_mutex;
    std::lock_guard<std::mutex> shared_pack_lock(shared_pack_mutex);
    static float* shared_packed_A = nullptr;
    static size_t shared_packed_A_cap = 0;
    static float* shared_packed_B = nullptr;
    static size_t shared_packed_B_cap = 0;
    /* For backward-compat with the symbol naming used earlier (keep diffs minimal). */
    auto& tls_packed_A = shared_packed_A;
    auto& tls_packed_A_cap = shared_packed_A_cap;
    auto& tls_packed_B = shared_packed_B;
    auto& tls_packed_B_cap = shared_packed_B_cap;

    const size_t pack_A_size = (size_t)TC_AVX2_MC * TC_AVX2_KC;
    const size_t pack_B_size = (size_t)TC_AVX2_KC * TC_AVX2_NC;

#if defined(_OPENMP)
    const char* threads_env = std::getenv("TC_AVX2_THREADS");
    const int requested_threads = (threads_env && threads_env[0]) ? std::atoi(threads_env) : 0;
    const int omp_threads = (requested_threads > 0) ? requested_threads : omp_get_max_threads();
    const bool allow_parallel = requested_threads != 1 && omp_threads > 1 && !omp_in_parallel();
    const int pack_A_slots = (allow_parallel && omp_threads > 1) ? omp_threads : 1;
#else
    const int pack_A_slots = 1;
#endif

    const size_t pack_A_pool_size = pack_A_size * (size_t)pack_A_slots;
    if (tls_packed_A_cap < pack_A_pool_size) {
        std::free(tls_packed_A);
        tls_packed_A = aligned_alloc_fp32(pack_A_pool_size);
        tls_packed_A_cap = pack_A_pool_size;
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

    for (int p = 0; p < K; p += TC_AVX2_KC) {
        const int kc = (p + TC_AVX2_KC <= K) ? TC_AVX2_KC : (K - p);
        for (int j = 0; j < N; j += TC_AVX2_NC) {
            const int nc = (j + TC_AVX2_NC <= N) ? TC_AVX2_NC : (N - j);
            pack_B(B + (size_t)p * ldb + j, ldb, kc, nc, tls_packed_B);

            const int i_blocks = (M + TC_AVX2_MC - 1) / TC_AVX2_MC;
#if defined(_OPENMP)
            const bool use_parallel = allow_parallel && i_blocks >= 2;
#pragma omp parallel for schedule(static) if(use_parallel) num_threads(omp_threads)
#endif
            for (int ib = 0; ib < i_blocks; ++ib) {
                const int i = ib * TC_AVX2_MC;
                const int mc = (i + TC_AVX2_MC <= M) ? TC_AVX2_MC : (M - i);
#if defined(_OPENMP)
                const int thread_slot = omp_in_parallel() ? omp_get_thread_num() : 0;
#else
                const int thread_slot = 0;
#endif
                float* packed_A_panel = tls_packed_A + (size_t)thread_slot * pack_A_size;
                pack_A(A + (size_t)i * lda + p, lda, mc, kc, packed_A_panel);

                const int jr_tiles = (nc + TC_AVX2_NR - 1) / TC_AVX2_NR;
                const int ir_tiles = (mc + TC_AVX2_MR - 1) / TC_AVX2_MR;
                for (int jt = 0; jt < jr_tiles; ++jt) {
                    for (int it = 0; it < ir_tiles; ++it) {
                        const int jr = jt * TC_AVX2_NR;
                        const int ir = it * TC_AVX2_MR;
                        const int nr = (jr + TC_AVX2_NR <= nc) ? TC_AVX2_NR : (nc - jr);
                        const float* pB = tls_packed_B + (size_t)(jr / TC_AVX2_NR) * kc * TC_AVX2_NR;
                        const int mr = (ir + TC_AVX2_MR <= mc) ? TC_AVX2_MR : (mc - ir);
                        const float* pA = packed_A_panel + (size_t)(ir / TC_AVX2_MR) * TC_AVX2_MR * kc;
                        float* Cij = C + (size_t)(i + ir) * ldc + (j + jr);
                        if (mr == TC_AVX2_MR && nr == TC_AVX2_NR && alpha == 1.0f) {
                            float tmp[TC_AVX2_MR * TC_AVX2_NR];
                            micro_kernel_6x16(kc, pA, pB, tmp, TC_AVX2_NR);
                            for (int r = 0; r < TC_AVX2_MR; ++r) {
                                __m256 t0v = _mm256_loadu_ps(tmp + r * TC_AVX2_NR);
                                __m256 t1v = _mm256_loadu_ps(tmp + r * TC_AVX2_NR + 8);
                                __m256 c0  = _mm256_loadu_ps(Cij + r * ldc);
                                __m256 c1  = _mm256_loadu_ps(Cij + r * ldc + 8);
                                _mm256_storeu_ps(Cij + r * ldc,     _mm256_add_ps(t0v, c0));
                                _mm256_storeu_ps(Cij + r * ldc + 8, _mm256_add_ps(t1v, c1));
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

/* Public entry point. The macro-kernel packs each A/B panel once, then fans
 * out independent 6×16 tile work to OpenMP threads when available. Set
 * TC_AVX2_THREADS=1 to force serial execution for A/B comparisons; set it
 * to N>1 to cap the internal worker count. */
extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda,
                                                   const float* B, int ldb,
                                                   float beta,
                                                   float* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    /* The slice function now contains its own OpenMP parallel region around
     * the inner kernel iterations, with shared pack buffers. This is the
     * BLIS-style approach: pack once, fan out the small-tile matmul work. */
    return tc_avx2_gemm_f32_slice(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
