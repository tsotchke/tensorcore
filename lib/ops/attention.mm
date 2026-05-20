/*
 * tensorcore — fused FlashAttention dispatch.
 *
 * v0.1 supports:
 *   - io_dtype F16, accum_dtype F32
 *   - head_dim ∈ {64}   (D=128 path lands in v0.2)
 *
 * Larger or unsupported configurations fall back to a non-fused path that
 * issues GEMM + softmax + GEMM (TC_BACKEND_MPS). The fallback is implemented
 * in lib/fallback/mps_attention.mm (stub for now — returns
 * TC_ERR_UNSUPPORTED_DTYPE so callers see a clean failure instead of a wrong
 * answer).
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdio>

#ifdef TC_HAVE_METAL4_SDK
extern "C" tc_status_t tc_tensorops_attention_attempt(tc_context* ctx,
                                                      const tc_attention_desc* d,
                                                      const tc_buffer* Q,
                                                      const tc_buffer* K,
                                                      const tc_buffer* V,
                                                      tc_buffer*       O,
                                                      tc_buffer*       LSE);
#endif

namespace {

constexpr uint32_t kBR = 32;
constexpr uint32_t kThreadsPerTG = 128;

struct KernelChoice {
    NSString* name;
    uint32_t  BR;
    uint32_t  threads;
};

KernelChoice kernel_name_for(const tc_attention_desc* d, tc_status_t* err) {
    *err = TC_OK;
    if (d->io_dtype == TC_DTYPE_F16 && d->accum_dtype == TC_DTYPE_F32) {
        if (d->head_dim == 64)  return { @"tc_flash_attention_f16_d64",  32, 128 };
        if (d->head_dim == 128) return { @"tc_flash_attention_f16_d128", 16, 128 };
    }
    *err = TC_ERR_UNSUPPORTED_DTYPE;
    return { nil, 0, 0 };
}

id<MTLComputePipelineState> resolve_pipeline(tc_context* ctx,
                                             NSString* name,
                                             bool causal,
                                             bool return_lse,
                                             bool use_window,
                                             bool use_alibi,
                                             tc_status_t* err) {
    NSString* key = [NSString stringWithFormat:@"%@:c=%d:l=%d:w=%d:a=%d", name,
                                                causal ? 1 : 0, return_lse ? 1 : 0,
                                                use_window ? 1 : 0, use_alibi ? 1 : 0];
    {
        id<MTLComputePipelineState> cached = nil;
        @synchronized(ctx->pipelines) {
            cached = [(TCPipelineCache*)ctx->pipelines pipelines][key];
        }
        if (cached) { if (err) *err = TC_OK; return cached; }
    }

    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&causal     type:MTLDataTypeBool atIndex:0];
    [cv setConstantValue:&return_lse type:MTLDataTypeBool atIndex:1];
    [cv setConstantValue:&use_window type:MTLDataTypeBool atIndex:2];
    [cv setConstantValue:&use_alibi  type:MTLDataTypeBool atIndex:3];

    NSError* nserr = nil;
    id<MTLFunction> fn = [ctx->library newFunctionWithName:name
                                            constantValues:cv
                                                     error:&nserr];
    if (!fn) {
        if (err) *err = TC_ERR_KERNEL_NOT_FOUND;
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:fn error:&nserr];
    if (!pso) {
        if (err) *err = TC_ERR_PIPELINE;
        return nil;
    }
    @synchronized(ctx->pipelines) {
        [(TCPipelineCache*)ctx->pipelines pipelines][key] = pso;
    }
    if (err) *err = TC_OK;
    return pso;
}

}  /* namespace */

extern "C" tc_status_t tc_attention_forward(tc_context* ctx,
                                            const tc_attention_desc* desc,
                                            const tc_buffer* Q,
                                            const tc_buffer* K,
                                            const tc_buffer* V,
                                            tc_buffer*       O,
                                            tc_buffer*       LSE) {
    if (!ctx)                 return TC_ERR_NOT_INITIALIZED;
    if (!desc || !Q || !K || !V || !O) return TC_ERR_INVALID_ARG;
    if (desc->batch <= 0 || desc->heads <= 0 || desc->seq_q <= 0 || desc->seq_kv <= 0) {
        return TC_ERR_INVALID_SHAPE;
    }
    if (desc->return_lse && !LSE) return TC_ERR_INVALID_ARG;

#ifdef TC_HAVE_METAL4_SDK
    if (ctx->info.supports_tensorops_m5) {
        tc_status_t s = tc_tensorops_attention_attempt(ctx, desc, Q, K, V, O, LSE);
        if (s == TC_OK) return TC_OK;
    }
#endif

    tc_status_t err = TC_OK;
    KernelChoice kc = kernel_name_for(desc, &err);
    if (!kc.name) return err;

    const bool use_window = (desc->window_size > 0);
    const bool use_alibi  = (desc->alibi_slopes != NULL);
    id<MTLComputePipelineState> pso = resolve_pipeline(ctx, kc.name,
                                                        desc->causal,
                                                        desc->return_lse,
                                                        use_window, use_alibi, &err);
    if (!pso) return err;

    const uint32_t batch    = (uint32_t)desc->batch;
    const uint32_t heads    = (uint32_t)desc->heads;
    const uint32_t kv_heads = (desc->kv_heads > 0) ? (uint32_t)desc->kv_heads : heads;
    const uint32_t seq_q    = (uint32_t)desc->seq_q;
    const uint32_t seq_kv   = (uint32_t)desc->seq_kv;
    const float    sm_scale = desc->softmax_scale;

    const uint32_t q_blocks = (seq_q + kc.BR - 1) / kc.BR;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:Q->mtl offset:0 atIndex:0];
        [enc setBuffer:K->mtl offset:0 atIndex:1];
        [enc setBuffer:V->mtl offset:0 atIndex:2];
        [enc setBuffer:O->mtl offset:0 atIndex:3];
        if (desc->return_lse && LSE) {
            [enc setBuffer:LSE->mtl offset:0 atIndex:4];
        }
        [enc setBytes:&batch    length:sizeof(batch)    atIndex:5];
        [enc setBytes:&heads    length:sizeof(heads)    atIndex:6];
        [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:7];
        [enc setBytes:&seq_q    length:sizeof(seq_q)    atIndex:8];
        [enc setBytes:&seq_kv   length:sizeof(seq_kv)   atIndex:9];
        [enc setBytes:&sm_scale length:sizeof(sm_scale) atIndex:10];
        if (use_window) {
            const uint32_t w = (uint32_t)desc->window_size;
            [enc setBytes:&w length:sizeof(w) atIndex:11];
        }
        if (use_alibi) {
            /* For v0.1, use a single slope (the head-mean). Multi-head per-head
             * slopes will land in v0.2 via a buffer binding. */
            float slope = 0.0f;
            for (int h = 0; h < desc->heads; ++h) slope += desc->alibi_slopes[h];
            slope /= (float)desc->heads;
            [enc setBytes:&slope length:sizeof(slope) atIndex:12];
        }

        [enc dispatchThreadgroups:MTLSizeMake(q_blocks, heads, batch)
            threadsPerThreadgroup:MTLSizeMake(kc.threads, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            fprintf(stderr, "[tensorcore] attention dispatch error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
}

extern "C" tc_status_t tc_attention_backward(tc_context* ctx,
                                             const tc_attention_desc* desc,
                                             const tc_buffer* Q,
                                             const tc_buffer* K,
                                             const tc_buffer* V,
                                             const tc_buffer* O,
                                             const tc_buffer* dO,
                                             const tc_buffer* LSE,
                                             tc_buffer*       dQ,
                                             tc_buffer*       dK,
                                             tc_buffer*       dV) {
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!desc || !Q || !K || !V || !O || !dO || !LSE || !dQ || !dK || !dV)
        return TC_ERR_INVALID_ARG;
    if (desc->io_dtype != TC_DTYPE_F16 || desc->accum_dtype != TC_DTYPE_F32)
        return TC_ERR_UNSUPPORTED_DTYPE;
    if (desc->head_dim != 64 && desc->head_dim != 128)
        return TC_ERR_UNSUPPORTED_DTYPE;

    const uint32_t batch    = (uint32_t)desc->batch;
    const uint32_t heads    = (uint32_t)desc->heads;
    const uint32_t kv_heads = (desc->kv_heads > 0) ? (uint32_t)desc->kv_heads : heads;
    const uint32_t seq_q    = (uint32_t)desc->seq_q;
    const uint32_t seq_kv   = (uint32_t)desc->seq_kv;
    const float    sm_scale = desc->softmax_scale;

    const uint32_t BR  = (desc->head_dim == 64) ? 32 : 16;
    const uint32_t BC  = BR;
    const uint32_t TPG = 128;

    NSString* dq_name  = (desc->head_dim == 64)
        ? @"tc_flash_attention_backward_dq"
        : @"tc_flash_attention_backward_dq_d128";
    NSString* dkv_name = (desc->head_dim == 64)
        ? @"tc_flash_attention_backward_dk_dv"
        : @"tc_flash_attention_backward_dk_dv_d128";

    tc_status_t err = TC_OK;

    bool causal = desc->causal;
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&causal type:MTLDataTypeBool atIndex:0];

    NSError* nserr = nil;
    id<MTLFunction> fn_dq =
        [ctx->library newFunctionWithName:dq_name
                           constantValues:cv error:&nserr];
    if (!fn_dq) return TC_ERR_KERNEL_NOT_FOUND;
    id<MTLComputePipelineState> pso_dq =
        [ctx->device newComputePipelineStateWithFunction:fn_dq error:&nserr];
    if (!pso_dq) return TC_ERR_PIPELINE;

    id<MTLFunction> fn_dkv =
        [ctx->library newFunctionWithName:dkv_name
                           constantValues:cv error:&nserr];
    if (!fn_dkv) return TC_ERR_KERNEL_NOT_FOUND;
    id<MTLComputePipelineState> pso_dkv =
        [ctx->device newComputePipelineStateWithFunction:fn_dkv error:&nserr];
    if (!pso_dkv) return TC_ERR_PIPELINE;

    const uint32_t q_blocks  = (seq_q  + BR - 1) / BR;
    const uint32_t kv_blocks = (seq_kv + BC - 1) / BC;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

        /* dQ pass */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dq];
            [enc setBuffer:Q->mtl   offset:0 atIndex:0];
            [enc setBuffer:K->mtl   offset:0 atIndex:1];
            [enc setBuffer:V->mtl   offset:0 atIndex:2];
            [enc setBuffer:O->mtl   offset:0 atIndex:3];
            [enc setBuffer:dO->mtl  offset:0 atIndex:4];
            [enc setBuffer:LSE->mtl offset:0 atIndex:5];
            [enc setBuffer:dQ->mtl  offset:0 atIndex:6];
            [enc setBytes:&batch    length:sizeof(batch)    atIndex:7];
            [enc setBytes:&heads    length:sizeof(heads)    atIndex:8];
            [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:9];
            [enc setBytes:&seq_q    length:sizeof(seq_q)    atIndex:10];
            [enc setBytes:&seq_kv   length:sizeof(seq_kv)   atIndex:11];
            [enc setBytes:&sm_scale length:sizeof(sm_scale) atIndex:12];
            [enc dispatchThreadgroups:MTLSizeMake(q_blocks, heads, batch)
                threadsPerThreadgroup:MTLSizeMake(TPG, 1, 1)];
            [enc endEncoding];
        }
        /* dK, dV pass */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dkv];
            [enc setBuffer:Q->mtl   offset:0 atIndex:0];
            [enc setBuffer:K->mtl   offset:0 atIndex:1];
            [enc setBuffer:V->mtl   offset:0 atIndex:2];
            [enc setBuffer:O->mtl   offset:0 atIndex:3];
            [enc setBuffer:dO->mtl  offset:0 atIndex:4];
            [enc setBuffer:LSE->mtl offset:0 atIndex:5];
            [enc setBuffer:dK->mtl  offset:0 atIndex:6];
            [enc setBuffer:dV->mtl  offset:0 atIndex:7];
            [enc setBytes:&batch    length:sizeof(batch)    atIndex:8];
            [enc setBytes:&heads    length:sizeof(heads)    atIndex:9];
            [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:10];
            [enc setBytes:&seq_q    length:sizeof(seq_q)    atIndex:11];
            [enc setBytes:&seq_kv   length:sizeof(seq_kv)   atIndex:12];
            [enc setBytes:&sm_scale length:sizeof(sm_scale) atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake(kv_blocks, heads, batch)
                threadsPerThreadgroup:MTLSizeMake(TPG, 1, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] attention backward error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
    (void)err;
}

extern "C" tc_status_t tc_attention_forward_async(tc_context* ctx,
                                                  const tc_attention_desc* desc,
                                                  const tc_buffer* Q,
                                                  const tc_buffer* K,
                                                  const tc_buffer* V,
                                                  tc_buffer*       O,
                                                  tc_buffer*       LSE,
                                                  tc_stream*       stream) {
    /* Same as sync version, but uses the supplied stream's queue and does not
     * wait. */
    if (!ctx) return TC_ERR_NOT_INITIALIZED;
    if (!desc || !Q || !K || !V || !O) return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    KernelChoice kc = kernel_name_for(desc, &err);
    if (!kc.name) return err;
    const bool use_window = (desc->window_size > 0);
    const bool use_alibi  = (desc->alibi_slopes != NULL);
    id<MTLComputePipelineState> pso = resolve_pipeline(ctx, kc.name,
                                                        desc->causal,
                                                        desc->return_lse,
                                                        use_window, use_alibi, &err);
    if (!pso) return err;

    const uint32_t batch    = (uint32_t)desc->batch;
    const uint32_t heads    = (uint32_t)desc->heads;
    const uint32_t kv_heads = (desc->kv_heads > 0) ? (uint32_t)desc->kv_heads : heads;
    const uint32_t seq_q    = (uint32_t)desc->seq_q;
    const uint32_t seq_kv   = (uint32_t)desc->seq_kv;
    const float    sm_scale = desc->softmax_scale;
    const uint32_t q_blocks = (seq_q + kc.BR - 1) / kc.BR;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = stream ? tc_stream_command_buffer(stream)
                                          : [ctx->queue commandBuffer];
        if (!cmd) return TC_ERR_INTERNAL;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:Q->mtl offset:0 atIndex:0];
        [enc setBuffer:K->mtl offset:0 atIndex:1];
        [enc setBuffer:V->mtl offset:0 atIndex:2];
        [enc setBuffer:O->mtl offset:0 atIndex:3];
        if (desc->return_lse && LSE) [enc setBuffer:LSE->mtl offset:0 atIndex:4];
        [enc setBytes:&batch    length:sizeof(batch)    atIndex:5];
        [enc setBytes:&heads    length:sizeof(heads)    atIndex:6];
        [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:7];
        [enc setBytes:&seq_q    length:sizeof(seq_q)    atIndex:8];
        [enc setBytes:&seq_kv   length:sizeof(seq_kv)   atIndex:9];
        [enc setBytes:&sm_scale length:sizeof(sm_scale) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(q_blocks, heads, batch)
            threadsPerThreadgroup:MTLSizeMake(kc.threads, 1, 1)];
        [enc endEncoding];
        if (!stream) [cmd commit];
    }
    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
}
