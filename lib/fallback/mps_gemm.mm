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
        case TC_DTYPE_I8:   return MPSDataTypeInt8;
        case TC_DTYPE_I32:  return MPSDataTypeInt32;
        default:            return MPSDataTypeInvalid;
    }
}

extern "C" tc_status_t tc_mps_gemm(tc_context* ctx,
                                   const tc_gemm_desc* desc,
                                   const tc_buffer* A,
                                   const tc_buffer* B,
                                   tc_buffer*       C) {
    if (!ctx || !desc || !A || !B || !C) return TC_ERR_INVALID_ARG;

    MPSDataType a_dt = to_mps_dtype(desc->a_dtype);
    MPSDataType b_dt = to_mps_dtype(desc->b_dtype);
    MPSDataType c_dt = to_mps_dtype(desc->c_dtype);
    if (a_dt == MPSDataTypeInvalid || b_dt == MPSDataTypeInvalid ||
        c_dt == MPSDataTypeInvalid) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    const NSUInteger M = desc->M, N = desc->N, K = desc->K;
    const NSUInteger lda = desc->lda ? desc->lda : (desc->transpose_a ? M : K);
    const NSUInteger ldb = desc->ldb ? desc->ldb : (desc->transpose_b ? K : N);
    const NSUInteger ldc = desc->ldc ? desc->ldc : N;

    @autoreleasepool {
        MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:K
                                                                      rowBytes:lda * tc_dtype_size(desc->a_dtype)
                                                                      dataType:a_dt];
        MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                       columns:N
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
