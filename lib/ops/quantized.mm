/*
 * tensorcore - quantized matmul host dispatch.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "tensorcore/quantized.h"
#include "../core/internal.h"

#include <cstdio>
#include <limits>

namespace {

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool fp16_matrix_bytes(int rows, int cols, size_t* out) {
    size_t elems = 0;
    if (rows <= 0 || cols <= 0) return false;
    if (!checked_mul((size_t)rows, (size_t)cols, &elems)) return false;
    return checked_mul(elems, sizeof(uint16_t), out);
}

tc_status_t validate_quantize_buffers(tc_context* ctx,
                                      const tc_buffer* W_fp16,
                                      tc_buffer* W_quant,
                                      tc_quant_t fmt,
                                      int N,
                                      int K) {
    size_t fp16_bytes = 0;
    if (!fp16_matrix_bytes(N, K, &fp16_bytes)) return TC_ERR_INVALID_ARG;
    const size_t quant_bytes = tc_quantized_size(fmt, N, K);
    if (quant_bytes == 0) return TC_ERR_INVALID_ARG;

    tc_status_t s = tc_buffer_validate(ctx, W_fp16, fp16_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, W_quant, quant_bytes);
}

tc_status_t validate_gemv_quantized_buffers(tc_context* ctx,
                                            const tc_buffer* X,
                                            const tc_buffer* W_quant,
                                            tc_buffer* Y,
                                            tc_quant_t fmt,
                                            int M,
                                            int N,
                                            int K) {
    size_t x_bytes = 0;
    size_t y_bytes = 0;
    if (!fp16_matrix_bytes(M, K, &x_bytes) ||
        !fp16_matrix_bytes(M, N, &y_bytes)) {
        return TC_ERR_INVALID_ARG;
    }
    const size_t w_bytes = tc_quantized_size(fmt, N, K);
    if (w_bytes == 0) return TC_ERR_INVALID_ARG;

    tc_status_t s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, W_quant, w_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, Y, y_bytes);
}

} /* namespace */

extern "C" size_t tc_quantized_size(tc_quant_t fmt, int N, int K) {
    if (N <= 0 || K <= 0 || K % 32 != 0) return 0;
    const size_t nblocks = (size_t)(K / 32);
    size_t bytes_per_block = 0;
    switch (fmt) {
        case TC_QUANT_Q4_0: bytes_per_block = 18; break;
        case TC_QUANT_Q8_0: bytes_per_block = 34; break;
        default: return 0;
    }
    return (size_t)N * nblocks * bytes_per_block;
}

extern "C" tc_status_t tc_quantize_weights(tc_context* ctx,
                                           const tc_buffer* W_fp16,
                                           tc_buffer*       W_quant,
                                           tc_quant_t       fmt,
                                           int N, int K) {
    if (!ctx || !W_fp16 || !W_quant || N <= 0 || K <= 0 || K % 32 != 0)
        return TC_ERR_INVALID_ARG;
    tc_status_t s = validate_quantize_buffers(ctx, W_fp16, W_quant, fmt, N, K);
    if (s != TC_OK) return s;
    NSString* kname = nil;
    if (fmt == TC_QUANT_Q4_0) {
        kname = @"tc_quantize_q4_0";
    } else if (fmt == TC_QUANT_Q8_0) {
        kname = @"tc_quantize_q8_0";
    } else {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, kname, &err);
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
    /* v2 default for Q4_0 (matches llama.cpp pattern). Set TC_Q4_USE_V1=1
     * to fall back to the slower 1-sg-per-cell kernel. */
    NSString* kname = nil;
    bool use_v2 = false;
    if (fmt == TC_QUANT_Q4_0) {
        const char* v1 = getenv("TC_Q4_USE_V1");
        if (v1 && v1[0] == '1') {
            kname = @"tc_q4_0_gemv_f16";
        } else {
            kname = @"tc_q4_0_gemv_v2_f16";
            use_v2 = true;
        }
    } else if (fmt == TC_QUANT_Q8_0) {
        kname = @"tc_q8_0_gemv_f16";
    }
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
    if (use_v2) {
        /* v2: 64 threads = 2 simdgroups x 32; each TG produces NR0*NSG = 8 output rows. */
        const uint32_t TG_X = (N_u + 7) / 8;   /* ceil(N / 8) */
        [enc dispatchThreadgroups:MTLSizeMake(TG_X, M_u, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
        /* v1: 1 simdgroup per output cell. */
        [enc dispatchThreadgroups:MTLSizeMake(N_u, M_u, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
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
    if (stream->owner != ctx) return TC_ERR_INVALID_ARG;
    tc_status_t s = validate_gemv_quantized_buffers(ctx, X, W_quant, Y, fmt, M, N, K);
    if (s != TC_OK) return s;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = tc_stream_command_buffer(stream);
        if (!cmd) return TC_ERR_INTERNAL;
        s = gemv_quant_encode(ctx, cmd, X, W_quant, Y, fmt, M, N, K);
        if (s != TC_OK) return s;
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
    tc_status_t s = validate_gemv_quantized_buffers(ctx, X, W_quant, Y, fmt, M, N, K);
    if (s != TC_OK) return s;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        s = gemv_quant_encode(ctx, cmd, X, W_quant, Y, fmt, M, N, K);
        if (s != TC_OK) return s;
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
