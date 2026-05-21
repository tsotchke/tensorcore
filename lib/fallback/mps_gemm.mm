/*
 * tensorcore — MPS GEMM fallback.
 *
 * Used when the requested (dtype, family) combo isn't yet covered by our
 * simdgroup_matrix kernels. MPSMatrixMultiplication itself dispatches to
 * simdgroup_matrix kernels internally on Apple7+ when shape/dtype match, so
 * this is a safety net rather than a slow path.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdio>

static MPSDataType to_mps_dtype(tc_dtype_t d) {
    switch (d) {
        case TC_DTYPE_F16:  return MPSDataTypeFloat16;
        case TC_DTYPE_F32:  return MPSDataTypeFloat32;
        case TC_DTYPE_BF16: {
            /* MPSDataTypeBFloat16 added in macOS 14.0. */
            if (@available(macOS 14.0, iOS 16.0, *)) {
                return MPSDataTypeBFloat16;
            }
            return MPSDataTypeInvalid;
        }
        case TC_DTYPE_I8:   return MPSDataTypeInt8;
        case TC_DTYPE_I32:  return MPSDataTypeInt32;
        default:            return MPSDataTypeInvalid;
    }
}

/* bf16 ↔ fp32 lifts (CPU side). bf16 = high 16 bits of fp32. */
static inline float bf16_to_f32(uint16_t bits) {
    union { uint32_t u; float f; } v = { ((uint32_t)bits) << 16 };
    return v.f;
}
static inline uint16_t f32_to_bf16(float x) {
    union { float f; uint32_t u; } v = { x };
    /* Round-to-nearest-even of the high half. */
    uint32_t r = v.u + 0x7FFF + ((v.u >> 16) & 1);
    return (uint16_t)(r >> 16);
}

static inline int effective_lda(const tc_gemm_desc* desc) {
    return desc->lda ? desc->lda : (desc->transpose_a ? desc->M : desc->K);
}

static inline int effective_ldb(const tc_gemm_desc* desc) {
    return desc->ldb ? desc->ldb : (desc->transpose_b ? desc->K : desc->N);
}

static inline int effective_ldc(const tc_gemm_desc* desc) {
    return desc->ldc ? desc->ldc : desc->N;
}

/* Software bf16 GEMM: bf16 -> fp32 (host convert) -> tc_gemm(fp32) -> bf16.
 * Used as a fallback when the device lacks bf16 simdgroup_matrix AND MPS
 * doesn't support bf16 in matmul (Apple<9 path). */
static tc_status_t bf16_via_fp32(tc_context* ctx,
                                 const tc_gemm_desc* desc,
                                 const tc_buffer* A,
                                 const tc_buffer* B,
                                 tc_buffer*       C) {
    const int M = desc->M, N = desc->N, K = desc->K;
    const int lda = effective_lda(desc);
    const int ldb = effective_ldb(desc);
    const int ldc = effective_ldc(desc);
    uint16_t *Ap, *Bp, *Cp;
    tc_buffer_map((tc_buffer*)A, (void**)&Ap);
    tc_buffer_map((tc_buffer*)B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    tc_buffer *Af = NULL, *Bf = NULL, *Cf = NULL;
    float *Afp = NULL, *Bfp = NULL, *Cfp = NULL;
    tc_gemm_desc d;
    tc_status_t status = TC_ERR_ALLOC;
    if (tc_buffer_alloc(ctx, (size_t)M * K * sizeof(float), &Af) != TC_OK) goto cleanup;
    if (tc_buffer_alloc(ctx, (size_t)K * N * sizeof(float), &Bf) != TC_OK) goto cleanup;
    if (tc_buffer_alloc(ctx, (size_t)M * N * sizeof(float), &Cf) != TC_OK) goto cleanup;

    tc_buffer_map(Af, (void**)&Afp);
    tc_buffer_map(Bf, (void**)&Bfp);
    tc_buffer_map(Cf, (void**)&Cfp);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const int src = desc->transpose_a ? k * lda + m : m * lda + k;
            Afp[m * K + k] = bf16_to_f32(Ap[src]);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const int src = desc->transpose_b ? n * ldb + k : k * ldb + n;
            Bfp[k * N + n] = bf16_to_f32(Bp[src]);
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            Cfp[m * N + n] = bf16_to_f32(Cp[m * ldc + n]);
        }
    }

    d = *desc;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = false;
    d.transpose_b = false;
    d.lda = 0;
    d.ldb = 0;
    d.ldc = 0;
    status = tc_gemm(ctx, &d, Af, Bf, Cf);
    if (status != TC_OK) goto cleanup;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            Cp[m * ldc + n] = f32_to_bf16(Cfp[m * N + n]);
        }
    }
    status = TC_OK;

cleanup:
    if (Af) tc_buffer_free(ctx, Af);
    if (Bf) tc_buffer_free(ctx, Bf);
    if (Cf) tc_buffer_free(ctx, Cf);
    return status;
}

/* Software int8 GEMM: i8 -> fp32 (exact lift) -> tc_gemm(fp32) -> i32.
 * fp32 has 24-bit mantissa so exact for K ≤ 2^16 with int8 inputs in
 * [-128, 127]. Used when device lacks i8 simdgroup_matrix. */
static tc_status_t i8_via_fp32(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer*       C) {
    const int M = desc->M, N = desc->N, K = desc->K;
    const int lda = effective_lda(desc);
    const int ldb = effective_ldb(desc);
    const int ldc = effective_ldc(desc);
    int8_t  *Ap, *Bp; int32_t *Cp;
    tc_buffer_map((tc_buffer*)A, (void**)&Ap);
    tc_buffer_map((tc_buffer*)B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    tc_buffer *Af = NULL, *Bf = NULL, *Cf = NULL;
    float *Afp = NULL, *Bfp = NULL, *Cfp = NULL;
    tc_gemm_desc d;
    tc_status_t status = TC_ERR_ALLOC;
    if (tc_buffer_alloc(ctx, (size_t)M * K * sizeof(float), &Af) != TC_OK) goto cleanup;
    if (tc_buffer_alloc(ctx, (size_t)K * N * sizeof(float), &Bf) != TC_OK) goto cleanup;
    if (tc_buffer_alloc(ctx, (size_t)M * N * sizeof(float), &Cf) != TC_OK) goto cleanup;

    tc_buffer_map(Af, (void**)&Afp);
    tc_buffer_map(Bf, (void**)&Bfp);
    tc_buffer_map(Cf, (void**)&Cfp);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const int src = desc->transpose_a ? k * lda + m : m * lda + k;
            Afp[m * K + k] = (float)Ap[src];
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const int src = desc->transpose_b ? n * ldb + k : k * ldb + n;
            Bfp[k * N + n] = (float)Bp[src];
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            Cfp[m * N + n] = (float)Cp[m * ldc + n];
        }
    }

    d = *desc;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = false;
    d.transpose_b = false;
    d.lda = 0;
    d.ldb = 0;
    d.ldc = 0;
    status = tc_gemm(ctx, &d, Af, Bf, Cf);
    if (status != TC_OK) goto cleanup;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            Cp[m * ldc + n] = (int32_t)Cfp[m * N + n];
        }
    }
    status = TC_OK;

cleanup:
    if (Af) tc_buffer_free(ctx, Af);
    if (Bf) tc_buffer_free(ctx, Bf);
    if (Cf) tc_buffer_free(ctx, Cf);
    return status;
}

extern "C" tc_status_t tc_mps_gemm(tc_context* ctx,
                                   const tc_gemm_desc* desc,
                                   const tc_buffer* A,
                                   const tc_buffer* B,
                                   tc_buffer*       C) {
    if (!ctx || !desc || !A || !B || !C) return TC_ERR_INVALID_ARG;

    /* bf16: MPSMatrixMultiplication asserts on bf16, so route through the
     * fp32 software fallback when the device lacks bf16 simdgroup_matrix. */
    if (desc->a_dtype == TC_DTYPE_BF16 && desc->b_dtype == TC_DTYPE_BF16) {
        return bf16_via_fp32(ctx, desc, A, B, C);
    }
    /* i8 -> i32: same SW fallback through fp32. */
    if (desc->a_dtype == TC_DTYPE_I8 && desc->b_dtype == TC_DTYPE_I8 &&
        desc->c_dtype == TC_DTYPE_I32) {
        return i8_via_fp32(ctx, desc, A, B, C);
    }

    MPSDataType a_dt = to_mps_dtype(desc->a_dtype);
    MPSDataType b_dt = to_mps_dtype(desc->b_dtype);
    MPSDataType c_dt = to_mps_dtype(desc->c_dtype);
    if (a_dt == MPSDataTypeInvalid || b_dt == MPSDataTypeInvalid ||
        c_dt == MPSDataTypeInvalid) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    const NSUInteger M = desc->M, N = desc->N, K = desc->K;
    const NSUInteger lda = (NSUInteger)effective_lda(desc);
    const NSUInteger ldb = (NSUInteger)effective_ldb(desc);
    const NSUInteger ldc = (NSUInteger)effective_ldc(desc);
    const NSUInteger a_rows = desc->transpose_a ? K : M;
    const NSUInteger a_cols = desc->transpose_a ? M : K;
    const NSUInteger b_rows = desc->transpose_b ? N : K;
    const NSUInteger b_cols = desc->transpose_b ? K : N;

    @autoreleasepool {
        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:a_rows
                                                                       columns:a_cols
                                                                      rowBytes:lda * tc_dtype_size(desc->a_dtype)
                                                                      dataType:a_dt];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows
                                                                       columns:b_cols
                                                                      rowBytes:ldb * tc_dtype_size(desc->b_dtype)
                                                                      dataType:b_dt];
        MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:N
                                                                      rowBytes:ldc * tc_dtype_size(desc->c_dtype)
                                                                      dataType:c_dt];
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A->mtl descriptor:dA];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B->mtl descriptor:dB];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C->mtl descriptor:dC];

        MPSMatrixMultiplication* kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                                              transposeLeft:desc->transpose_a
                                             transposeRight:desc->transpose_b
                                                 resultRows:M
                                              resultColumns:N
                                            interiorColumns:K
                                                      alpha:(double)desc->alpha
                                                       beta:(double)desc->beta];

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        [kernel encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] mps gemm error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    return TC_OK;
}
