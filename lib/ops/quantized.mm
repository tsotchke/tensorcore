/*
 * tensorcore — quantized matmul host dispatch.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "tensorcore/quantized.h"
#include "../core/internal.h"

#include <cstdio>

extern "C" size_t tc_quantized_size(tc_quant_t fmt, int N, int K) {
    if (N <= 0 || K <= 0 || K % 32 != 0) return 0;
    const size_t nblocks = (size_t)(K / 32);
    const size_t bytes_per_block = (fmt == TC_QUANT_Q4_0) ? 18 : 34;
    return (size_t)N * nblocks * bytes_per_block;
}

extern "C" tc_status_t tc_quantize_weights(tc_context* ctx,
                                           const tc_buffer* W_fp16,
                                           tc_buffer*       W_quant,
                                           tc_quant_t       fmt,
                                           int N, int K) {
    if (!ctx || !W_fp16 || !W_quant || N <= 0 || K <= 0 || K % 32 != 0)
        return TC_ERR_INVALID_ARG;
    if (fmt != TC_QUANT_Q4_0) {
        /* Q8_0 quantization kernel deferred — caller can manually pack. */
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, @"tc_quantize_q4_0", &err);
    if (!pso) return err;

    const uint32_t N_u = (uint32_t)N, K_u = (uint32_t)K;
    const uint32_t nblocks = (uint32_t)(K / 32);
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:W_fp16->mtl   offset:0 atIndex:0];
        [enc setBuffer:W_quant->mtl  offset:0 atIndex:1];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:2];
        [enc setBytes:&K_u length:sizeof(K_u) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nblocks, N_u, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

/* Internal helper used by both sync and async variants. */
static tc_status_t gemv_quant_encode(tc_context* ctx,
                                     id<MTLCommandBuffer> cmd,
                                     const tc_buffer* X,
                                     const tc_buffer* W_quant,
                                     tc_buffer*       Y,
                                     tc_quant_t       fmt,
                                     int M, int N, int K) {
    NSString* kname = (fmt == TC_QUANT_Q4_0) ? @"tc_q4_0_gemv_f16"
                    : (fmt == TC_QUANT_Q8_0) ? @"tc_q8_0_gemv_f16"
                    : nil;
    if (!kname) return TC_ERR_UNSUPPORTED_DTYPE;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, kname, &err);
    if (!pso) return err;
    const uint32_t M_u = (uint32_t)M, N_u = (uint32_t)N, K_u = (uint32_t)K;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:X->mtl       offset:0 atIndex:0];
    [enc setBuffer:W_quant->mtl offset:0 atIndex:1];
    [enc setBuffer:Y->mtl       offset:0 atIndex:2];
    [enc setBytes:&M_u length:sizeof(M_u) atIndex:3];
    [enc setBytes:&N_u length:sizeof(N_u) atIndex:4];
    [enc setBytes:&K_u length:sizeof(K_u) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(N_u, M_u, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return TC_OK;
}

extern "C" tc_status_t tc_gemv_quantized_async(tc_context* ctx,
                                               const tc_buffer* X,
                                               const tc_buffer* W_quant,
                                               tc_buffer*       Y,
                                               tc_quant_t       fmt,
                                               int M, int N, int K,
                                               tc_stream*       stream) {
    if (!ctx || !X || !W_quant || !Y || !stream ||
        M <= 0 || N <= 0 || K <= 0 || K % 32 != 0) return TC_ERR_INVALID_ARG;
    @autoreleasepool {
        id<MTLCommandQueue> q = stream->queue;
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        tc_status_t s = gemv_quant_encode(ctx, cmd, X, W_quant, Y, fmt, M, N, K);
        if (s != TC_OK) return s;
        [cmd commit];
        /* No wait. */
    }
    return TC_OK;
}

extern "C" tc_status_t tc_gemv_quantized(tc_context* ctx,
                                         const tc_buffer* X,
                                         const tc_buffer* W_quant,
                                         tc_buffer*       Y,
                                         tc_quant_t       fmt,
                                         int M, int N, int K) {
    if (!ctx || !X || !W_quant || !Y || M <= 0 || N <= 0 || K <= 0 || K % 32 != 0)
        return TC_ERR_INVALID_ARG;
    NSString* kname = (fmt == TC_QUANT_Q4_0) ? @"tc_q4_0_gemv_f16"
                    : (fmt == TC_QUANT_Q8_0) ? @"tc_q8_0_gemv_f16"
                    : nil;
    if (!kname) return TC_ERR_UNSUPPORTED_DTYPE;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, kname, &err);
    if (!pso) return err;

    const uint32_t M_u = (uint32_t)M, N_u = (uint32_t)N, K_u = (uint32_t)K;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl       offset:0 atIndex:0];
        [enc setBuffer:W_quant->mtl offset:0 atIndex:1];
        [enc setBuffer:Y->mtl       offset:0 atIndex:2];
        [enc setBytes:&M_u length:sizeof(M_u) atIndex:3];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:4];
        [enc setBytes:&K_u length:sizeof(K_u) atIndex:5];
        /* One threadgroup (= one simdgroup of 32 threads) per output cell. */
        [enc dispatchThreadgroups:MTLSizeMake(N_u, M_u, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] gemv_quantized: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    return TC_OK;
}
