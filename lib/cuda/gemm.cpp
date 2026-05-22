/*
 * tensorcore — CUDA GEMM via cuBLAS.
 *
 * Gated on TC_ENABLE_CUDA=ON. Wires tc_gemm into cuBLAS sgemm / hgemm /
 * gemmEx for tensor-core paths. Returns TC_ERR_UNSUPPORTED_FAMILY when
 * CUDA is not compiled in, so the public dispatch falls through.
 *
 * Row-major C ABI → column-major cuBLAS: the standard transpose-trick
 * C^T = B^T * A^T applies (swap M↔N, swap A↔B, flip transposes).
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/cuda.h"

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define TC_CUDA_INTERNAL __attribute__((visibility("hidden")))
#else
#  define TC_CUDA_INTERNAL
#endif

#if defined(TC_ENABLE_CUDA)
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cuda_fp16.h>
#endif

extern "C" TC_CUDA_INTERNAL void tc_cuda_set_last_kernel(const char* name);

namespace {

#if defined(TC_ENABLE_CUDA)

struct CublasHandle {
    cublasHandle_t h = nullptr;
    bool initialized = false;
};

CublasHandle& handle() {
    static CublasHandle hh;
    if (!hh.initialized) {
        if (cublasCreate(&hh.h) == CUBLAS_STATUS_SUCCESS) {
            hh.initialized = true;
            /* Enable tensor-core paths whenever shape allows. */
            cublasSetMathMode(hh.h, CUBLAS_TENSOR_OP_MATH);
        }
    }
    return hh;
}

cublasOperation_t trans_of(bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }

tc_status_t cuda_gemm_sgemm(const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C) {
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    /* Row-major → column-major: compute (B^T × A^T) instead. */
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;

    cublasStatus_t s = cublasSgemm(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        &d->alpha,
        (const float*)Bp, lda,
        (const float*)Ap, ldb,
        &d->beta,
        (float*)Cp, ldc);
    tc_cuda_set_last_kernel("cublas_sgemm");
    return (s == CUBLAS_STATUS_SUCCESS) ? TC_OK : TC_ERR_INTERNAL;
}

tc_status_t cuda_gemm_hgemm(const tc_gemm_desc* d,
                            const tc_buffer* A, const tc_buffer* B,
                            tc_buffer* C) {
    /* fp16 via cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP — picks
     * tensor-core path on sm_70+ (Volta/Turing/Ampere/Ada/Hopper). */
    auto& hh = handle();
    if (!hh.initialized) return TC_ERR_INTERNAL;

    void *Ap = nullptr, *Bp = nullptr, *Cp = nullptr;
    if (tc_buffer_map((tc_buffer*)A, &Ap) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map((tc_buffer*)B, &Bp) != TC_OK) return TC_ERR_INTERNAL;
    if (tc_buffer_map(C, &Cp) != TC_OK) return TC_ERR_INTERNAL;

    /* fp32 alpha/beta — cublasGemmEx uses fp32 scaling even with fp16
     * inputs/outputs when compute_type is CUDA_R_32F. */
    const float alpha_f = d->alpha;
    const float beta_f = d->beta;
    const int M = d->N, N = d->M, K = d->K;
    const int lda = d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
    const int ldb = d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
    const int ldc = d->ldc ? d->ldc : d->N;

    cublasStatus_t s = cublasGemmEx(
        hh.h, trans_of(d->transpose_b), trans_of(d->transpose_a),
        M, N, K,
        &alpha_f,
        Bp, CUDA_R_16F, lda,
        Ap, CUDA_R_16F, ldb,
        &beta_f,
        Cp, CUDA_R_16F, ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    tc_cuda_set_last_kernel("cublas_gemmex_fp16_tensorop");
    return (s == CUBLAS_STATUS_SUCCESS) ? TC_OK : TC_ERR_INTERNAL;
}

#endif  /* TC_ENABLE_CUDA */

}  // namespace

extern "C" TC_CUDA_INTERNAL tc_status_t tc_cuda_gemm(tc_context* ctx,
                                                     const tc_gemm_desc* desc,
                                                     const tc_buffer* A,
                                                     const tc_buffer* B,
                                                     tc_buffer* C) {
#if !defined(TC_ENABLE_CUDA)
    (void)ctx; (void)desc; (void)A; (void)B; (void)C;
    return TC_ERR_UNSUPPORTED_FAMILY;
#else
    if (!ctx || !desc || !A || !B || !C) return TC_ERR_INVALID_ARG;
    if (desc->a_dtype == TC_DTYPE_F32 && desc->b_dtype == TC_DTYPE_F32 &&
        desc->c_dtype == TC_DTYPE_F32) {
        return cuda_gemm_sgemm(desc, A, B, C);
    }
    if (desc->a_dtype == TC_DTYPE_F16 && desc->b_dtype == TC_DTYPE_F16 &&
        desc->c_dtype == TC_DTYPE_F16) {
        return cuda_gemm_hgemm(desc, A, B, C);
    }
    return TC_ERR_UNSUPPORTED_DTYPE;
#endif
}
