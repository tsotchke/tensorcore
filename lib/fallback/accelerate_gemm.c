/*
 * tensorcore — Accelerate CPU GEMM fallback.
 *
 * Used when:
 *   - GPU init failed
 *   - Sizes are too small to amortize GPU dispatch overhead
 *   - The user explicitly requested CPU path via TC_FORCE_CPU=1
 *
 * Accelerate on Apple Silicon auto-dispatches to AMX (M1-M3) or SME (M4+),
 * which is why we don't need a separate "AMX kernel".
 */

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include "tensorcore/tensorcore.h"

tc_status_t tc_accelerate_gemm_f32(const tc_gemm_desc* desc,
                                   const float* A,
                                   const float* B,
                                   float*       C) {
    if (!desc || !A || !B || !C) return TC_ERR_INVALID_ARG;

    const enum CBLAS_TRANSPOSE ta = desc->transpose_a ? CblasTrans : CblasNoTrans;
    const enum CBLAS_TRANSPOSE tb = desc->transpose_b ? CblasTrans : CblasNoTrans;
    const int M = desc->M, N = desc->N, K = desc->K;
    const int lda = desc->lda ? desc->lda : (desc->transpose_a ? M : K);
    const int ldb = desc->ldb ? desc->ldb : (desc->transpose_b ? K : N);
    const int ldc = desc->ldc ? desc->ldc : N;

    cblas_sgemm(CblasRowMajor, ta, tb,
                M, N, K,
                desc->alpha,
                A, lda,
                B, ldb,
                desc->beta,
                C, ldc);
    return TC_OK;
}
