/*
 * tensorcore — HIP GEMM dispatch.
 *
 * Routes tc_gemm calls into chipStar's hipBLAS port when the HIP backend
 * is selected. hipBLAS provides vendor-tuned GEMM across Intel/NVIDIA/AMD
 * via the chipStar SPIR-V dispatch layer (or native rocBLAS on AMD when
 * available).
 *
 * Compile gate: TC_ENABLE_HIP=ON in CMake. Without it, returns
 * TC_ERR_UNSUPPORTED_FAMILY so the public dispatch falls back to CBLAS
 * or the reference loop.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/hip.h"

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_HIP_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_HIP_INTERNAL
#endif

#if defined(TC_ENABLE_HIP)
#  include <hip/hip_runtime.h>
#  include <hipblas/hipblas.h>
#endif

extern "C" TC_HIP_INTERNAL void tc_hip_set_last_kernel(const char* name);

namespace {

#if defined(TC_ENABLE_HIP)

struct HipBlasHandle {
    hipblasHandle_t h = nullptr;
    bool            initialized = false;
};

HipBlasHandle& handle() {
    static HipBlasHandle hh;
    if (!hh.initialized) {
        if (hipblasCreate(&hh.h) == HIPBLAS_STATUS_SUCCESS) {
            hh.initialized = true;
        }
    }
    return hh;
}

/* Translate tc_gemm_desc transpose flags into hipBLAS HIPBLAS_OP_*. */
hipblasOperation_t trans_of(bool t) { return t ? HIPBLAS_OP_T : HIPBLAS_OP_N; }

tc_status_t hip_gemm_sgemm(const tc_gemm_desc* d,
                           const tc_buffer* A, const tc_buffer* B,
                           tc_buffer* C) {
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    /* tc_buffer's storage is host-mapped on integrated GPUs (Apple is Metal
     * here; HIP integrated is xavier-class), but discrete GPUs need explicit
     * device pointers. chipStar's hipMalloc returns device pointers; the
     * tc_buffer's map() returns a host pointer that's transparently mirrored.
     * For v0, expect tc_buffer to expose a device pointer via a new helper. */
    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    /* hipBLAS uses column-major. Our C ABI is row-major. The standard trick:
     *   C^T = B^T * A^T   (row-major → column-major).
     * So call hipBLAS with M↔N swapped and transposes flipped. */
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;
    const hipblasOperation_t op_a = trans_of(d->transpose_b);
    const hipblasOperation_t op_b = trans_of(d->transpose_a);

    hipblasStatus_t s = hipblasSgemm(
        hh.h, op_a, op_b,
        M, N, K,
        &d->alpha,
        (const float*)Bp, lda,
        (const float*)Ap, ldb,
        &d->beta,
        (float*)Cp, ldc);
    tc_hip_set_last_kernel("hipblas_sgemm");
    return (s == HIPBLAS_STATUS_SUCCESS) ? TC_OK : TC_ERR_INTERNAL;
}

tc_status_t hip_gemm_hgemm(const tc_gemm_desc* d,
                           const tc_buffer* A, const tc_buffer* B,
                           tc_buffer* C) {
    /* hipBLAS hgemm available on NVIDIA + AMD; Intel route through
     * sgemm with fp16 → fp32 dequant. v0 just routes everything through
     * hgemm and lets chipStar's hipBLAS port pick the right impl. */
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    const hipblasHalf alpha = hipblasHalf((__half)d->alpha);
    const hipblasHalf beta = hipblasHalf((__half)d->beta);
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;

    hipblasStatus_t s = hipblasHgemm(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        &alpha,
        (const hipblasHalf*)Bp, lda,
        (const hipblasHalf*)Ap, ldb,
        &beta,
        (hipblasHalf*)Cp, ldc);
    tc_hip_set_last_kernel("hipblas_hgemm");
    return (s == HIPBLAS_STATUS_SUCCESS) ? TC_OK : TC_ERR_INTERNAL;
}

#endif  /* TC_ENABLE_HIP */

}  // namespace

/* Public-ish entry point — called from the main tc_gemm dispatcher when
 * the active backend is HIP. Returns TC_ERR_UNSUPPORTED_FAMILY when
 * TC_ENABLE_HIP is OFF so the dispatcher falls through to the next
 * backend (CBLAS / reference). */
extern "C" TC_HIP_INTERNAL tc_status_t tc_hip_gemm(tc_context* ctx,
                                                    const tc_gemm_desc* desc,
                                                    const tc_buffer* A,
                                                    const tc_buffer* B,
                                                    tc_buffer* C) {
#if !defined(TC_ENABLE_HIP)
    (void)ctx; (void)desc; (void)A; (void)B; (void)C;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx || !desc) return TC_ERR_INVALID_ARG;
    if (!A || !B || !C) return TC_ERR_INVALID_ARG;
    if (desc->a_dtype == TC_DTYPE_F32 && desc->b_dtype == TC_DTYPE_F32 &&
        desc->c_dtype == TC_DTYPE_F32) {
        return hip_gemm_sgemm(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_F16 && desc->b_dtype == TC_DTYPE_F16 &&
        desc->c_dtype == TC_DTYPE_F16) {
        return hip_gemm_hgemm(desc, A, B, C);
    }
    return TC_ERR_UNSUPPORTED_DTYPE;
#endif
}
