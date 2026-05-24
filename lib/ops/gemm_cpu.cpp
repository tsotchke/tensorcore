/*
 * tensorcore - portable CPU GEMM backend.
 *
 * Correctness-first implementation for non-Apple mesh workers. This is not a
 * replacement for Accelerate/AMX/SME on Apple; it is the minimal non-CUDA path
 * that keeps tensorcore's ABI usable on Linux CPU nodes.
 */

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(TC_HAS_CBLAS)
#  if defined(__APPLE__)
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

#if defined(_WIN32)
#define TC_INTERNAL_SYMBOL
#else
#define TC_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#endif

/* Forward decl for the AVX2 micro-kernel in gemm_cpu_avx2.cpp. Returns 0 on
 * success, non-zero on internal failure (in which case the caller falls back
 * to CBLAS or the reference loop). The AVX2 path is preferred for fp32 GEMM
 * on x86_64 with AVX2+FMA - it's self-contained, no BLAS dependency. */
extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda,
                                                   const float* B, int ldb,
                                                   float beta,
                                                   float* C, int ldc);
extern "C" TC_INTERNAL_SYMBOL int tc_avx2_gemm_f32_available(void);

/* Forward decl for the NEON micro-kernel in gemm_cpu_neon.cpp. Same contract
 * as the AVX2 path: returns 0 on success, non-zero on internal failure. Built
 * on aarch64 with __ARM_NEON; everywhere else it stubs to -1 / 0.
 *
 * NEON's signature additionally takes transpose flags for A and B because the
 * pack functions handle transposed source layouts natively (no on-the-fly
 * matrix transposition needed). */
extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32(int M, int N, int K,
                                                   float alpha,
                                                   const float* A, int lda, int transpose_a,
                                                   const float* B, int ldb, int transpose_b,
                                                   float beta,
                                                   float* C, int ldc);
extern "C" TC_INTERNAL_SYMBOL int tc_neon_gemm_f32_available(void);

/* Forward decl for the Apple AMX matrix-coprocessor backend in
 * gemm_cpu_amx.cpp. Same contract as the NEON path - returns 0 on success,
 * non-zero on (unsupported configuration | runtime failure). Built only on
 * Apple Silicon (__APPLE__ && __aarch64__); stubbed to -1 / 0 elsewhere.
 *
 * Session-1 capabilities (see header in gemm_cpu_amx.cpp):
 *   fp32, !transpose, alpha=1, beta=0, M%16==0, N%16==0. All other cases return -1
 *   from the kernel so the dispatcher can fall through to NEON / CBLAS. */
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32(int M, int N, int K,
                                                  float alpha,
                                                  const float* A, int lda, int transpose_a,
                                                  const float* B, int ldb, int transpose_b,
                                                  float beta,
                                                  float* C, int ldc);
extern "C" TC_INTERNAL_SYMBOL int tc_amx_gemm_f32_available(void);

namespace {

bool validate_desc(const tc_gemm_desc* d) {
    /* K==0 is the BLAS degenerate sgemm (C := beta*C, no MAC accumulated).
     * Public ABI honors it; handled by a short-circuit in tc_gemm. */
    return d && d->M > 0 && d->N > 0 && d->K >= 0;
}

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool checked_add(size_t a, size_t b, size_t* out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int32_t effective_lda(const tc_gemm_desc* d) {
    return d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
}

int32_t effective_ldb(const tc_gemm_desc* d) {
    return d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
}

int32_t effective_ldc(const tc_gemm_desc* d) {
    return d->ldc ? d->ldc : d->N;
}

bool matrix_storage_bytes(int32_t rows, int32_t cols, int32_t ld,
                          tc_dtype_t dtype, size_t* out) {
    const size_t elem_size = tc_dtype_size(dtype);
    size_t row_offset = 0;
    size_t elems = 0;
    if (rows <= 0 || cols <= 0 || ld < cols || elem_size == 0) return false;
    if (!checked_mul((size_t)(rows - 1), (size_t)ld, &row_offset)) return false;
    if (!checked_add(row_offset, (size_t)cols, &elems)) return false;
    return checked_mul(elems, elem_size, out);
}

tc_status_t validate_gemm_buffers(tc_context* ctx,
                                  const tc_gemm_desc* d,
                                  const tc_buffer* A,
                                  const tc_buffer* B,
                                  tc_buffer* C) {
    const int32_t a_rows = d->transpose_a ? d->K : d->M;
    const int32_t a_cols = d->transpose_a ? d->M : d->K;
    const int32_t b_rows = d->transpose_b ? d->N : d->K;
    const int32_t b_cols = d->transpose_b ? d->K : d->N;
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    if (!matrix_storage_bytes(a_rows, a_cols, effective_lda(d), d->a_dtype, &a_bytes) ||
        !matrix_storage_bytes(b_rows, b_cols, effective_ldb(d), d->b_dtype, &b_bytes) ||
        !matrix_storage_bytes(d->M, d->N, effective_ldc(d), d->c_dtype, &c_bytes)) {
        return TC_ERR_INVALID_ARG;
    }
    tc_status_t s = tc_buffer_validate(ctx, A, a_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, B, b_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, C, c_bytes);
}

uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    const uint32_t f = v.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t round = (mant >> (shift - 1)) & 1u;
        return (uint16_t)(sign | ((mant >> shift) + round));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    const uint32_t round = (mant >> 12) & 1u;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant >> 13) + round));
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            ++exp;
            mant &= 0x3FFu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = {out};
    return v.f;
}

bool supports_cpu_gemm(const tc_gemm_desc* d) {
    const bool f32 =
        d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32 &&
        d->c_dtype == TC_DTYPE_F32 && d->accum_dtype == TC_DTYPE_F32;
    const bool f16 =
        d->a_dtype == TC_DTYPE_F16 && d->b_dtype == TC_DTYPE_F16 &&
        d->c_dtype == TC_DTYPE_F16 && d->accum_dtype == TC_DTYPE_F32;
    const bool bf16 =
        d->a_dtype == TC_DTYPE_BF16 && d->b_dtype == TC_DTYPE_BF16 &&
        d->c_dtype == TC_DTYPE_BF16 && d->accum_dtype == TC_DTYPE_F32;
    const bool i8 =
        d->a_dtype == TC_DTYPE_I8 && d->b_dtype == TC_DTYPE_I8 &&
        d->c_dtype == TC_DTYPE_I32 && d->accum_dtype == TC_DTYPE_I32;
    return f32 || f16 || bf16 || i8;
}

float load_a_f32(const float* A, const tc_gemm_desc* d, int m, int k) {
    const int32_t lda = effective_lda(d);
    return d->transpose_a ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
}

float load_b_f32(const float* B, const tc_gemm_desc* d, int k, int n) {
    const int32_t ldb = effective_ldb(d);
    return d->transpose_b ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
}

float load_a_f16(const uint16_t* A, const tc_gemm_desc* d, int m, int k) {
    const int32_t lda = effective_lda(d);
    const uint16_t v = d->transpose_a ? A[(size_t)k * lda + m] : A[(size_t)m * lda + k];
    return f16_to_f32(v);
}

float load_b_f16(const uint16_t* B, const tc_gemm_desc* d, int k, int n) {
    const int32_t ldb = effective_ldb(d);
    const uint16_t v = d->transpose_b ? B[(size_t)n * ldb + k] : B[(size_t)k * ldb + n];
    return f16_to_f32(v);
}

/* BF16 <-> FP32: trivially-aligned formats. BF16 is the high 16 bits of an
 * FP32 (sign + 8 exp + 7 mantissa). These helpers are used by both CBLAS
 * bf16 fallback and the always-compiled K==0 short-circuit. */
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v = { (uint32_t)b << 16 };
    return v.f;
}

static inline uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t lower = v.u & 0xFFFFu;
    const uint32_t upper = v.u >> 16;
    const uint32_t round =
        (lower > 0x8000u) || ((lower == 0x8000u) && (upper & 1u));
    return (uint16_t)(upper + round);
}

#if defined(TC_HAS_CBLAS)
void tc_cblas_sgemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
                    int32_t m, int32_t n, int32_t k,
                    float alpha,
                    const float* A, int32_t lda,
                    const float* B, int32_t ldb,
                    float beta,
                    float* C, int32_t ldc) {
#  if defined(__APPLE__) && defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#  endif
    cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#  if defined(__APPLE__) && defined(__clang__)
#    pragma clang diagnostic pop
#  endif
}

/* Dequantize an fp16 [rows x cols] matrix into an fp32 buffer.
 * Respects the leading dimension; the dst is packed (ld = cols).
 * OpenMP-parallelized; saturates all available cores on the dequant pass. */
void dequant_fp16_to_fp32(const uint16_t* src, int32_t rows, int32_t cols,
                          int32_t ld, float* dst) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t r = 0; r < rows; ++r) {
        const uint16_t* row = src + (size_t)r * ld;
        float* dst_row = dst + (size_t)r * cols;
        for (int32_t c = 0; c < cols; ++c) dst_row[c] = f16_to_f32(row[c]);
    }
}

/* Quantize an fp32 [rows x cols] packed buffer back into an fp16 matrix with
 * the given leading dimension. OpenMP-parallel. */
void quantize_fp32_to_fp16(const float* src, int32_t rows, int32_t cols,
                           int32_t ld, uint16_t* dst) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t r = 0; r < rows; ++r) {
        uint16_t* dst_row = dst + (size_t)r * ld;
        const float* src_row = src + (size_t)r * cols;
        for (int32_t c = 0; c < cols; ++c) dst_row[c] = f32_to_f16(src_row[c]);
    }
}

/* No #pragma omp here: this code path runs from inside Python via the
 * PyTorch bridge, where libomp is already loaded by PyTorch. A second
 * libomp init from tensorcore's link of OpenMP::OpenMP_CXX triggers OMP
 * Error #15 ("multiple OMP runtimes linked") and the KMP_DUPLICATE_LIB_OK
 * workaround segfaults. BF16 dequant/quant is a few hundred MB of
 * straight-line work — fast enough serial even at 4096x4096. The fp32
 * path doesn't hit this because it doesn't dequant. */
void dequant_bf16_to_fp32(const uint16_t* src, int32_t rows, int32_t cols,
                          int32_t ld, float* dst) {
    for (int32_t r = 0; r < rows; ++r) {
        const uint16_t* row = src + (size_t)r * ld;
        float* dst_row = dst + (size_t)r * cols;
        for (int32_t c = 0; c < cols; ++c) dst_row[c] = bf16_to_f32(row[c]);
    }
}

void quantize_fp32_to_bf16(const float* src, int32_t rows, int32_t cols,
                           int32_t ld, uint16_t* dst) {
    for (int32_t r = 0; r < rows; ++r) {
        uint16_t* dst_row = dst + (size_t)r * ld;
        const float* src_row = src + (size_t)r * cols;
        for (int32_t c = 0; c < cols; ++c) dst_row[c] = f32_to_bf16(src_row[c]);
    }
}

tc_status_t gemm_compute_cblas_f32(const tc_gemm_desc* d,
                                   const float* A, const float* B, float* C) {
    /* tensorcore's descriptor is row-major; cblas with CblasRowMajor honors
     * that directly. transpose_a/b map to CblasTrans / CblasNoTrans. */
    const CBLAS_TRANSPOSE ta = d->transpose_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = d->transpose_b ? CblasTrans : CblasNoTrans;
    tc_cblas_sgemm(ta, tb, d->M, d->N, d->K, d->alpha,
                   A, effective_lda(d), B, effective_ldb(d),
                   d->beta, C, effective_ldc(d));
    return TC_OK;
}

tc_status_t gemm_compute_cblas_bf16(const tc_gemm_desc* d,
                                    const uint16_t* A, const uint16_t* B,
                                    uint16_t* C) {
    /* BF16 path: same shape as the fp16 path but with bf16 dequant/quant.
     * BF16 dequant/quant is one shift each (vs the branchier fp16 path),
     * so this is slightly cheaper at the conversion step. The Accelerate
     * sgemm call in the middle is identical. */
    static thread_local std::vector<float> tls_Af, tls_Bf, tls_Cf;

    const int32_t a_rows = d->transpose_a ? d->K : d->M;
    const int32_t a_cols = d->transpose_a ? d->M : d->K;
    const int32_t b_rows = d->transpose_b ? d->N : d->K;
    const int32_t b_cols = d->transpose_b ? d->K : d->N;

    const size_t Af_n = (size_t)a_rows * a_cols;
    const size_t Bf_n = (size_t)b_rows * b_cols;
    const size_t Cf_n = (size_t)d->M * d->N;
    if (tls_Af.size() < Af_n) tls_Af.resize(Af_n);
    if (tls_Bf.size() < Bf_n) tls_Bf.resize(Bf_n);
    if (tls_Cf.size() < Cf_n) tls_Cf.resize(Cf_n);
    float* Af = tls_Af.data();
    float* Bf = tls_Bf.data();
    float* Cf = tls_Cf.data();

    dequant_bf16_to_fp32(A, a_rows, a_cols, effective_lda(d), Af);
    dequant_bf16_to_fp32(B, b_rows, b_cols, effective_ldb(d), Bf);
    if (d->beta != 0.0f) {
        dequant_bf16_to_fp32(C, d->M, d->N, effective_ldc(d), Cf);
    }

    const CBLAS_TRANSPOSE ta = d->transpose_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = d->transpose_b ? CblasTrans : CblasNoTrans;
    tc_cblas_sgemm(ta, tb, d->M, d->N, d->K, d->alpha,
                   Af, a_cols, Bf, b_cols,
                   d->beta, Cf, d->N);
    quantize_fp32_to_bf16(Cf, d->M, d->N, effective_ldc(d), C);
    return TC_OK;
}

tc_status_t gemm_compute_cblas_f16(const tc_gemm_desc* d,
                                   const uint16_t* A, const uint16_t* B,
                                   uint16_t* C) {
    /* fp16 inputs: dequant A and B to fp32 packed buffers, sgemm, requant.
     * The dequant + sgemm + requant is still much faster than the naive
     * triple-loop on any multithreaded CBLAS. */
    /* Thread-local scratch buffers grow monotonically across calls. A steady-
     * state inference loop pays the malloc cost once, then amortizes it to
     * zero. Without this, each fp16 GEMM call allocs and frees ~3*M*K*4 bytes
     * and the malloc dominates the actual GEMM cost at 4096^3. */
    static thread_local std::vector<float> tls_Af, tls_Bf, tls_Cf;

    const int32_t a_rows = d->transpose_a ? d->K : d->M;
    const int32_t a_cols = d->transpose_a ? d->M : d->K;
    const int32_t b_rows = d->transpose_b ? d->N : d->K;
    const int32_t b_cols = d->transpose_b ? d->K : d->N;

    const size_t Af_n = (size_t)a_rows * a_cols;
    const size_t Bf_n = (size_t)b_rows * b_cols;
    const size_t Cf_n = (size_t)d->M * d->N;
    if (tls_Af.size() < Af_n) tls_Af.resize(Af_n);
    if (tls_Bf.size() < Bf_n) tls_Bf.resize(Bf_n);
    if (tls_Cf.size() < Cf_n) tls_Cf.resize(Cf_n);
    float* Af = tls_Af.data();
    float* Bf = tls_Bf.data();
    float* Cf = tls_Cf.data();

    dequant_fp16_to_fp32(A, a_rows, a_cols, effective_lda(d), Af);
    dequant_fp16_to_fp32(B, b_rows, b_cols, effective_ldb(d), Bf);
    if (d->beta != 0.0f) {
        dequant_fp16_to_fp32(C, d->M, d->N, effective_ldc(d), Cf);
    }

    const CBLAS_TRANSPOSE ta = d->transpose_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tb = d->transpose_b ? CblasTrans : CblasNoTrans;
    tc_cblas_sgemm(ta, tb, d->M, d->N, d->K, d->alpha,
                   Af, a_cols, Bf, b_cols,
                   d->beta, Cf, d->N);
    quantize_fp32_to_fp16(Cf, d->M, d->N, effective_ldc(d), C);
    return TC_OK;
}
#endif  /* TC_HAS_CBLAS */

/* Cache the in-tree-kernel opt-in env vars once per thread. Env vars do not
 * change mid-process for these flags (they're configuration, not run-time
 * inputs). Reading getenv on every gemm_compute call added ~30 ns of
 * syscall-ish overhead and showed up as a +5-10 % penalty for the very
 * small matmuls coming through the PyTorch bridge. */
struct gemm_env_cache {
    bool prefer_avx2;
    bool prefer_amx;
    bool prefer_neon;
};
static const gemm_env_cache& load_gemm_env() {
    static thread_local gemm_env_cache cache = []() {
        gemm_env_cache c{};
        const char* e;
        e = std::getenv("TC_USE_AVX2_GEMM"); c.prefer_avx2 = (e && e[0] == '1');
        e = std::getenv("TC_USE_AMX_GEMM");  c.prefer_amx  = (e && e[0] == '1');
        e = std::getenv("TC_USE_NEON_GEMM"); c.prefer_neon = (e && e[0] == '1');
        return c;
    }();
    return cache;
}

tc_status_t gemm_compute(const tc_gemm_desc* d, const void* A, const void* B, void* C) {
    const int32_t ldc = effective_ldc(d);

    /* Path priority for fp32 GEMM on x86_64:
     *   1. AVX2 in-tree kernel    (self-contained, no BLAS dep, ~800 GFLOPS+)
     *   2. CBLAS (MKL > OpenBLAS) (~1.5-2 TFLOPS on Haswell-EP dual-socket)
     *   3. Reference triple-loop  (~1 GFLOPS - correctness only)
     *
     * Path priority for fp32 GEMM on aarch64:
     *   1. NEON in-tree kernel    (self-contained, no BLAS dep, ~50 GFLOPS/core)
     *   2. CBLAS (Accelerate on Apple, OpenBLAS on Linux ARM)
     *   3. Reference triple-loop
     *
     * For now both in-tree kernels are opt-in via TC_USE_AVX2_GEMM / TC_USE_NEON_GEMM
     * so we can A/B against the BLAS delegate. Once the OpenMP outer loops are
     * wired (Phase 1.x), the defaults flip. */
    /* CUDA GEMM dispatch happens in tc_gemm() proper (it has tc_buffer*
     * which we need; gemm_compute only has raw pointers). */

    const auto& env = load_gemm_env();

    if (env.prefer_avx2 && tc_avx2_gemm_f32_available() &&
        d->c_dtype == TC_DTYPE_F32 && d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32) {
        if (tc_avx2_gemm_f32(d->M, d->N, d->K,
                             d->alpha,
                             (const float*)A, effective_lda(d),
                             (const float*)B, effective_ldb(d),
                             d->beta,
                             (float*)C, ldc) == 0) {
            return TC_OK;
        }
        /* Fall through to CBLAS / reference on AVX2 internal failure. */
    }

    /* Apple AMX matrix coprocessor - preferred when available and the
     * caller has opted in via TC_USE_AMX_GEMM=1. AMX delivers ~10x NEON's
     * fp32 throughput on Apple Silicon. The kernel itself returns -1 for
     * invalid arguments or scratch-allocation failures, so we fall through
     * to the NEON / CBLAS path cleanly without a duplicate guard. */
    if (env.prefer_amx && tc_amx_gemm_f32_available() &&
        d->c_dtype == TC_DTYPE_F32 && d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32) {
        if (tc_amx_gemm_f32(d->M, d->N, d->K,
                            d->alpha,
                            (const float*)A, effective_lda(d), d->transpose_a ? 1 : 0,
                            (const float*)B, effective_ldb(d), d->transpose_b ? 1 : 0,
                            d->beta,
                            (float*)C, ldc) == 0) {
            return TC_OK;
        }
        /* Fall through to NEON / CBLAS / reference. */
    }

    if (env.prefer_neon && tc_neon_gemm_f32_available() &&
        d->c_dtype == TC_DTYPE_F32 && d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32) {
        /* The NEON pack functions handle transposed A and B natively, so no
         * dispatch guard is needed beyond the dtype check. */
        if (tc_neon_gemm_f32(d->M, d->N, d->K,
                             d->alpha,
                             (const float*)A, effective_lda(d), d->transpose_a ? 1 : 0,
                             (const float*)B, effective_ldb(d), d->transpose_b ? 1 : 0,
                             d->beta,
                             (float*)C, ldc) == 0) {
            return TC_OK;
        }
        /* Fall through to CBLAS / reference on NEON internal failure. */
    }

#if defined(TC_HAS_CBLAS)
    /* Fast path: delegate fp32 and fp16 GEMM to CBLAS (Accelerate / OpenBLAS /
     * MKL). Two orders of magnitude faster than the triple-loop reference. */
    if (d->c_dtype == TC_DTYPE_F32 &&
        d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32) {
        return gemm_compute_cblas_f32(d, (const float*)A, (const float*)B, (float*)C);
    }
    if (d->c_dtype == TC_DTYPE_F16 &&
        d->a_dtype == TC_DTYPE_F16 && d->b_dtype == TC_DTYPE_F16) {
        return gemm_compute_cblas_f16(d, (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)C);
    }
    if (d->c_dtype == TC_DTYPE_BF16 &&
        d->a_dtype == TC_DTYPE_BF16 && d->b_dtype == TC_DTYPE_BF16) {
        return gemm_compute_cblas_bf16(d, (const uint16_t*)A, (const uint16_t*)B, (uint16_t*)C);
    }
    /* I32 (int8 inputs) falls through to the integer reference below. */
#endif

    if (d->c_dtype == TC_DTYPE_I32) {
        const int32_t lda = effective_lda(d);
        const int32_t ldb = effective_ldb(d);
        const int8_t* Ai = (const int8_t*)A;
        const int8_t* Bi = (const int8_t*)B;
        int32_t* Ci = (int32_t*)C;
        for (int m = 0; m < d->M; ++m) {
            for (int n = 0; n < d->N; ++n) {
                int64_t sum = 0;
                for (int k = 0; k < d->K; ++k) {
                    const size_t ai = d->transpose_a ? (size_t)k * lda + m : (size_t)m * lda + k;
                    const size_t bi = d->transpose_b ? (size_t)n * ldb + k : (size_t)k * ldb + n;
                    sum += (int32_t)Ai[ai] * (int32_t)Bi[bi];
                }
                const size_t idx = (size_t)m * ldc + n;
                Ci[idx] = (int32_t)(d->alpha * (float)sum + d->beta * (float)Ci[idx]);
            }
        }
        return TC_OK;
    }

    if (d->c_dtype == TC_DTYPE_F32) {
        const float* Af = (const float*)A;
        const float* Bf = (const float*)B;
        float* Cf = (float*)C;
        for (int m = 0; m < d->M; ++m) {
            for (int n = 0; n < d->N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < d->K; ++k) {
                    sum += load_a_f32(Af, d, m, k) * load_b_f32(Bf, d, k, n);
                }
                const size_t idx = (size_t)m * ldc + n;
                Cf[idx] = d->alpha * sum + d->beta * Cf[idx];
            }
        }
        return TC_OK;
    }

    const uint16_t* Ah = (const uint16_t*)A;
    const uint16_t* Bh = (const uint16_t*)B;
    uint16_t* Ch = (uint16_t*)C;
    for (int m = 0; m < d->M; ++m) {
        for (int n = 0; n < d->N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < d->K; ++k) {
                sum += load_a_f16(Ah, d, m, k) * load_b_f16(Bh, d, k, n);
            }
            const size_t idx = (size_t)m * ldc + n;
            const float prev = f16_to_f32(Ch[idx]);
            Ch[idx] = f32_to_f16(d->alpha * sum + d->beta * prev);
        }
    }
    return TC_OK;
}

bool batched_matrix_bytes(int32_t rows,
                          int32_t cols,
                          int32_t ld,
                          tc_dtype_t dtype,
                          int32_t batch,
                          int64_t stride_elems,
                          size_t* out) {
    size_t single_elems = 0;
    size_t total_elems = 0;
    const size_t elem_size = tc_dtype_size(dtype);
    if (rows <= 0 || cols <= 0 || ld < cols || batch <= 0 || elem_size == 0) return false;
    size_t row_offset = 0;
    if (!checked_mul((size_t)(rows - 1), (size_t)ld, &row_offset)) return false;
    if (!checked_add(row_offset, (size_t)cols, &single_elems)) return false;
    if (batch == 1) {
        total_elems = single_elems;
    } else {
        if (stride_elems < 0 || (uint64_t)stride_elems < single_elems) return false;
        size_t batch_offset = 0;
        if (!checked_mul((size_t)(batch - 1), (size_t)stride_elems, &batch_offset)) {
            return false;
        }
        if (!checked_add(batch_offset, single_elems, &total_elems)) return false;
    }
    return checked_mul(total_elems, elem_size, out);
}

}  // namespace

#if defined(TC_ENABLE_CUDA)
extern "C" tc_status_t tc_cuda_gemm(tc_context* ctx,
                                     const tc_gemm_desc* desc,
                                     const tc_buffer* A,
                                     const tc_buffer* B,
                                     tc_buffer* C);
extern "C" int tc_cuda_is_active(void);
#endif

#if defined(TC_ENABLE_HIP)
extern "C" tc_status_t tc_hip_gemm(tc_context* ctx,
                                    const tc_gemm_desc* desc,
                                    const tc_buffer* A,
                                    const tc_buffer* B,
                                    tc_buffer* C);
extern "C" int tc_hip_is_active(void);
#endif

extern "C" tc_status_t tc_gemm(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer* C) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!validate_desc(desc) || !A || !B || !C) return TC_ERR_INVALID_ARG;

    /* K=0 short-circuit: BLAS degenerate GEMM = C := beta*C. Validate only
     * C (A,B can have any layout when K=0 — they're not touched), then
     * scale or zero. */
    if (desc->K == 0) {
        if (!supports_cpu_gemm(desc)) return TC_ERR_UNSUPPORTED_DTYPE;
        size_t c_bytes = 0;
        if (!matrix_storage_bytes(desc->M, desc->N, effective_ldc(desc),
                                  desc->c_dtype, &c_bytes)) {
            return TC_ERR_INVALID_ARG;
        }
        tc_status_t cs = tc_buffer_validate(ctx, C, c_bytes);
        if (cs != TC_OK) return cs;
        void* Cp = nullptr;
        cs = tc_buffer_map(C, &Cp);
        if (cs != TC_OK) return cs;
        const int32_t ldc = effective_ldc(desc);
        if (desc->c_dtype == TC_DTYPE_F32) {
            float* Cd = (float*)Cp;
            if (desc->beta == 0.0f) {
                for (int32_t i = 0; i < desc->M; ++i)
                    std::memset(Cd + (size_t)i * ldc, 0, (size_t)desc->N * sizeof(float));
            } else {
                for (int32_t i = 0; i < desc->M; ++i) {
                    float* row = Cd + (size_t)i * ldc;
                    for (int32_t j = 0; j < desc->N; ++j) row[j] *= desc->beta;
                }
            }
        } else if (desc->c_dtype == TC_DTYPE_F16) {
            uint16_t* Cd = (uint16_t*)Cp;
            if (desc->beta == 0.0f) {
                for (int32_t i = 0; i < desc->M; ++i)
                    std::memset(Cd + (size_t)i * ldc, 0, (size_t)desc->N * sizeof(uint16_t));
            } else {
                for (int32_t i = 0; i < desc->M; ++i) {
                    uint16_t* row = Cd + (size_t)i * ldc;
                    for (int32_t j = 0; j < desc->N; ++j) {
                        row[j] = f32_to_f16(f16_to_f32(row[j]) * desc->beta);
                    }
                }
            }
        } else if (desc->c_dtype == TC_DTYPE_BF16) {
            uint16_t* Cd = (uint16_t*)Cp;
            if (desc->beta == 0.0f) {
                for (int32_t i = 0; i < desc->M; ++i)
                    std::memset(Cd + (size_t)i * ldc, 0, (size_t)desc->N * sizeof(uint16_t));
            } else {
                for (int32_t i = 0; i < desc->M; ++i) {
                    uint16_t* row = Cd + (size_t)i * ldc;
                    for (int32_t j = 0; j < desc->N; ++j) {
                        row[j] = f32_to_bf16(bf16_to_f32(row[j]) * desc->beta);
                    }
                }
            }
        } else if (desc->c_dtype == TC_DTYPE_I32) {
            int32_t* Cd = (int32_t*)Cp;
            if (desc->beta == 0.0f) {
                for (int32_t i = 0; i < desc->M; ++i)
                    std::memset(Cd + (size_t)i * ldc, 0, (size_t)desc->N * sizeof(int32_t));
            } else {
                for (int32_t i = 0; i < desc->M; ++i) {
                    int32_t* row = Cd + (size_t)i * ldc;
                    for (int32_t j = 0; j < desc->N; ++j) {
                        row[j] = (int32_t)(desc->beta * (float)row[j]);
                    }
                }
            }
        }
        return tc_record_dispatch("tc_gemm", TC_BACKEND_PORTABLE_CPU, TC_OK);
    }

    /* Defer the CPU dtype gate until after CUDA dispatch: bf16/i8 may not
     * be supported by the portable CPU path but ARE supported by CUDA. */
    tc_status_t s = validate_gemm_buffers(ctx, desc, A, B, C);
    if (s != TC_OK) return s;

#if defined(TC_ENABLE_CUDA) || defined(TC_ENABLE_HIP)
    const bool same_float =
        (desc->c_dtype == TC_DTYPE_F32 || desc->c_dtype == TC_DTYPE_F16 ||
         desc->c_dtype == TC_DTYPE_BF16) &&
        desc->a_dtype == desc->c_dtype && desc->b_dtype == desc->c_dtype;
#endif
#if defined(TC_ENABLE_CUDA)
    const bool i8_to_i32 =
        desc->a_dtype == TC_DTYPE_I8 && desc->b_dtype == TC_DTYPE_I8 &&
        desc->c_dtype == TC_DTYPE_I32;
    /* CUDA dispatch: when a CUDA-enabled context is active, route supported
     * GEMM calls into cuBLAS. TC_DISABLE_CUDA_GEMM=1 / TC_CUDA_GEMM=0 /
     * TC_USE_CUDA_GEMM=0 force CPU fallback for debugging. Supported dtype combos:
     *   - fp32 in, fp32 out, fp32 accum
     *   - fp16 in, fp16 out, fp32 or fp16 accum (TC_CUDA_FP16_ACCUM=1)
     *   - bf16 in, bf16 out, fp32 accum
     *   - int8 in, int32 out, int32 accum (K must be multiple of 16)
     */
    if (tc_cuda_is_active() && (same_float || i8_to_i32)) {
        tc_status_t cs = tc_cuda_gemm(ctx, desc, A, B, C);
        if (cs == TC_OK) {
            return tc_record_dispatch("tc_gemm", TC_BACKEND_CUDA, TC_OK);
        }
        /* Fall through to CPU path on CUDA error (e.g. no device). */
    }
#endif

#if defined(TC_ENABLE_HIP)
    /* HIP/chipStar dispatch: vendor-neutral GPU path via hipBLAS. CUDA stays
     * preferred on NVIDIA when both runtimes are linked; HIP covers non-CUDA
     * SPIR-V devices and explicit chipStar builds. */
    if (tc_hip_is_active() && same_float) {
        tc_status_t hs = tc_hip_gemm(ctx, desc, A, B, C);
        if (hs == TC_OK) {
            return tc_record_dispatch("tc_gemm", TC_BACKEND_HIP, TC_OK);
        }
        /* Fall through to CPU path on HIP dtype/runtime failures. */
    }
#endif

    /* Now apply the CPU dtype gate. Only reached if no GPU backend served it. */
    if (!supports_cpu_gemm(desc)) return TC_ERR_UNSUPPORTED_DTYPE;

    void* Ap = nullptr;
    void* Bp = nullptr;
    void* Cp = nullptr;
    s = tc_buffer_map((tc_buffer*)A, &Ap);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)B, &Bp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(C, &Cp);
    if (s != TC_OK) return s;

    s = gemm_compute(desc, Ap, Bp, Cp);
    return tc_record_dispatch("tc_gemm", TC_BACKEND_PORTABLE_CPU, s);
}

extern "C" tc_status_t tc_gemm_async(tc_context* ctx,
                                     const tc_gemm_desc* desc,
                                     const tc_buffer* A,
                                     const tc_buffer* B,
                                     tc_buffer* C,
                                     tc_stream* stream) {
    (void)stream;
    return tc_gemm(ctx, desc, A, B, C);
}

extern "C" tc_status_t tc_gemm_batched(tc_context* ctx,
                                       const tc_gemm_batched_desc* bd,
                                       const tc_buffer* A,
                                       const tc_buffer* B,
                                       tc_buffer* C) {
    if (!ctx || !bd || !A || !B || !C) return TC_ERR_INVALID_ARG;
    const tc_gemm_desc& d = bd->base;
    if (!validate_desc(&d) || bd->batch <= 0) return TC_ERR_INVALID_ARG;
    if (!supports_cpu_gemm(&d)) return TC_ERR_UNSUPPORTED_DTYPE;
    if (bd->batch > 1 && (bd->stride_a <= 0 || bd->stride_b <= 0 || bd->stride_c <= 0)) {
        return TC_ERR_INVALID_SHAPE;
    }

    const int32_t a_rows = d.transpose_a ? d.K : d.M;
    const int32_t a_cols = d.transpose_a ? d.M : d.K;
    const int32_t b_rows = d.transpose_b ? d.N : d.K;
    const int32_t b_cols = d.transpose_b ? d.K : d.N;
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    if (!batched_matrix_bytes(a_rows, a_cols, effective_lda(&d),
                              d.a_dtype, bd->batch, bd->stride_a, &a_bytes) ||
        !batched_matrix_bytes(b_rows, b_cols, effective_ldb(&d),
                              d.b_dtype, bd->batch, bd->stride_b, &b_bytes) ||
        !batched_matrix_bytes(d.M, d.N, effective_ldc(&d),
                              d.c_dtype, bd->batch, bd->stride_c, &c_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    tc_status_t s = tc_buffer_validate(ctx, A, a_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, B, b_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, C, c_bytes);
    if (s != TC_OK) return s;

    uint8_t* Ap = nullptr;
    uint8_t* Bp = nullptr;
    uint8_t* Cp = nullptr;
    s = tc_buffer_map((tc_buffer*)A, (void**)&Ap);
    if (s != TC_OK) return s;
    s = tc_buffer_map((tc_buffer*)B, (void**)&Bp);
    if (s != TC_OK) return s;
    s = tc_buffer_map(C, (void**)&Cp);
    if (s != TC_OK) return s;

    const size_t a_elem = tc_dtype_size(d.a_dtype);
    const size_t b_elem = tc_dtype_size(d.b_dtype);
    const size_t c_elem = tc_dtype_size(d.c_dtype);
    const size_t stride_a = (size_t)((bd->batch == 1 || bd->stride_a == 0)
        ? (int64_t)a_rows * effective_lda(&d) : bd->stride_a);
    const size_t stride_b = (size_t)((bd->batch == 1 || bd->stride_b == 0)
        ? (int64_t)b_rows * effective_ldb(&d) : bd->stride_b);
    const size_t stride_c = (size_t)((bd->batch == 1 || bd->stride_c == 0)
        ? (int64_t)d.M * effective_ldc(&d) : bd->stride_c);

    for (int b = 0; b < bd->batch; ++b) {
        s = gemm_compute(&d,
                         Ap + (size_t)b * stride_a * a_elem,
                         Bp + (size_t)b * stride_b * b_elem,
                         Cp + (size_t)b * stride_c * c_elem);
        if (s != TC_OK) return s;
    }
    return tc_record_dispatch("tc_gemm_batched", TC_BACKEND_PORTABLE_CPU, TC_OK);
}
