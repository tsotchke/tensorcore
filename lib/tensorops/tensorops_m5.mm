/*
 * tensorcore — Metal 4 / TensorOps host dispatch (M5+ Neural Accelerator).
 *
 * Compiled only when CMake detects macOS 26.0+ SDK (TC_HAVE_METAL4_SDK).
 * Runtime gated on MTLGPUFamilyMetal4 + device-name contains "M5"/"M6"
 * (llama.cpp PR #16634 hardening — Metal 4 GPU family is reported on M3+ but
 * only M5+ has the dedicated tensor unit that makes this path a perf win).
 *
 * The kernels (tensorops_gemm.metal, tensorops_flash_attention.metal) use
 * mpp::tensor_ops directly. We dispatch them through ordinary MTLComputeCommandEncoder
 * + MTLBuffer — the Metal 4 *command encoder* (MTL4MachineLearningCommandEncoder)
 * is only needed for pre-compiled CoreML packages, which we are not using.
 *
 * Per Draw Things MFA v2.5 + Apple ML Research: this path delivers ~110 TFLOPS
 * fp16 on M5 Max (~5× M2 Ultra). On M3/M4 it's measurably slower than the
 * simdgroup_matrix path (no tensor unit silicon), hence the strict gating.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdio>
#include <cstring>

/* MTLGPUFamilyMetal4 may not be in older SDKs; provide the raw value for
 * forward-compat. Apple assigns it 5002 (per llama.cpp PR #16634 and the
 * MTLGPUFamily docs). */
#ifndef MTLGPUFamilyMetal4
#define MTLGPUFamilyMetal4 ((MTLGPUFamily)5002)
#endif

namespace {

/* Hardened detection: family + name. Per llama.cpp's empirical finding,
 * supportsFamily:MTLGPUFamilyMetal4 reports true on M3+ but the tensor unit
 * only delivers gains on M5+. */
bool runtime_supports_tensor_unit(id<MTLDevice> dev) {
    if (!dev) return false;
    BOOL fam = NO;
    @try {
        fam = [dev supportsFamily:MTLGPUFamilyMetal4];
    } @catch (...) {
        fam = NO;
    }
    if (!fam) return false;

    NSString* name = [dev name];
    if (!name) return false;
    /* Match "Apple M5", "Apple M6", future "Apple M7"... but exclude
     * incidental matches like "M5 Pro". The name format is e.g.
     * "Apple M5 Max" or "Apple M5". Substring search is sufficient. */
    if ([name containsString:@"M5"] || [name containsString:@"M6"]
        || [name containsString:@"M7"]) {
        return true;
    }
    return false;
}

NSString* tc4_kernel_name_for_gemm(const tc_gemm_desc* d, tc_status_t* err) {
    *err = TC_OK;
    if (d->a_dtype == TC_DTYPE_F16 && d->b_dtype == TC_DTYPE_F16 &&
        d->c_dtype == TC_DTYPE_F16 && d->accum_dtype == TC_DTYPE_F32) {
        return @"tc4_gemm_f16";
    }
    if (d->a_dtype == TC_DTYPE_BF16 && d->b_dtype == TC_DTYPE_BF16 &&
        d->c_dtype == TC_DTYPE_BF16 && d->accum_dtype == TC_DTYPE_F32) {
        return @"tc4_gemm_bf16";
    }
    if (d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32 &&
        d->c_dtype == TC_DTYPE_F32 && d->accum_dtype == TC_DTYPE_F32) {
        return @"tc4_gemm_f32";
    }
    *err = TC_ERR_UNSUPPORTED_DTYPE;
    return nil;
}

}  /* namespace */

extern "C" tc_status_t tc_tensorops_available(tc_context* ctx, bool* out) {
    if (!ctx || !out) return TC_ERR_INVALID_ARG;
    *out = runtime_supports_tensor_unit(ctx->device);
    return TC_OK;
}

/* GEMM via Metal 4 tensor_ops kernels.  Same dispatch shape as
 * lib/ops/gemm.mm but uses the tc4_gemm_* kernels in the metallib. */
extern "C" tc_status_t tc_tensorops_gemm_attempt(tc_context* ctx,
                                                 const tc_gemm_desc* desc,
                                                 const tc_buffer* A,
                                                 const tc_buffer* B,
                                                 tc_buffer*       C) {
    if (!ctx || !desc || !A || !B || !C) return TC_ERR_INVALID_ARG;
    if (!runtime_supports_tensor_unit(ctx->device)) {
        return TC_ERR_UNSUPPORTED_FAMILY;
    }

    tc_status_t err = TC_OK;
    NSString* kname = tc4_kernel_name_for_gemm(desc, &err);
    if (!kname) return err;

    /* v0.1 of this path: alpha=1, beta=0 only. */
    if (desc->alpha != 1.0f || desc->beta != 0.0f) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, kname, &err);
    if (!pso) return err;

    const uint32_t M = (uint32_t)desc->M;
    const uint32_t N = (uint32_t)desc->N;
    const uint32_t K = (uint32_t)desc->K;
    const float    alpha = desc->alpha;
    const float    beta  = desc->beta;

    constexpr uint32_t BM = 64;
    constexpr uint32_t BN = 64;
    constexpr uint32_t TPG = 128;   /* 4 simdgroups */

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:A->mtl offset:0 atIndex:0];
        [enc setBuffer:B->mtl offset:0 atIndex:1];
        [enc setBuffer:C->mtl offset:0 atIndex:2];
        [enc setBytes:&M     length:sizeof(M)     atIndex:3];
        [enc setBytes:&N     length:sizeof(N)     atIndex:4];
        [enc setBytes:&K     length:sizeof(K)     atIndex:5];
        [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];
        [enc setBytes:&beta  length:sizeof(beta)  atIndex:7];

        const uint32_t gx = (N + BN - 1) / BN;
        const uint32_t gy = (M + BM - 1) / BM;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(TPG, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            fprintf(stderr, "[tensorcore] tensorops gemm error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    tc_set_last_backend(TC_BACKEND_TENSOROPS_M5);
    return TC_OK;
}

/* Attention via the Metal 4 tensor_ops FlashAttention kernel. */
extern "C" tc_status_t tc_tensorops_attention_attempt(tc_context* ctx,
                                                      const tc_attention_desc* d,
                                                      const tc_buffer* Q,
                                                      const tc_buffer* K,
                                                      const tc_buffer* V,
                                                      tc_buffer*       O,
                                                      tc_buffer*       LSE) {
    if (!ctx || !d || !Q || !K || !V || !O) return TC_ERR_INVALID_ARG;
    if (!runtime_supports_tensor_unit(ctx->device)) {
        return TC_ERR_UNSUPPORTED_FAMILY;
    }

    if (d->io_dtype != TC_DTYPE_F16 || d->accum_dtype != TC_DTYPE_F32) {
        return TC_ERR_UNSUPPORTED_DTYPE;
    }

    NSString* kname = nil;
    if (d->head_dim == 64)       kname = @"tc4_flash_attention_f16_d64";
    else if (d->head_dim == 128) kname = @"tc4_flash_attention_f16_d128";
    else return TC_ERR_UNSUPPORTED_DTYPE;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = tc_pipeline_get(ctx, kname, &err);
    if (!pso) return err;

    const uint32_t batch    = (uint32_t)d->batch;
    const uint32_t heads    = (uint32_t)d->heads;
    const uint32_t kv_heads = (d->kv_heads > 0) ? (uint32_t)d->kv_heads : heads;
    const uint32_t seq_q    = (uint32_t)d->seq_q;
    const uint32_t seq_kv   = (uint32_t)d->seq_kv;
    const float    sm_scale = d->softmax_scale;
    const uint32_t BR       = 64;
    const uint32_t q_blocks = (seq_q + BR - 1) / BR;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:Q->mtl offset:0 atIndex:0];
        [enc setBuffer:K->mtl offset:0 atIndex:1];
        [enc setBuffer:V->mtl offset:0 atIndex:2];
        [enc setBuffer:O->mtl offset:0 atIndex:3];
        if (d->return_lse && LSE) [enc setBuffer:LSE->mtl offset:0 atIndex:4];
        [enc setBytes:&batch    length:sizeof(batch)    atIndex:5];
        [enc setBytes:&heads    length:sizeof(heads)    atIndex:6];
        [enc setBytes:&kv_heads length:sizeof(kv_heads) atIndex:7];
        [enc setBytes:&seq_q    length:sizeof(seq_q)    atIndex:8];
        [enc setBytes:&seq_kv   length:sizeof(seq_kv)   atIndex:9];
        [enc setBytes:&sm_scale length:sizeof(sm_scale) atIndex:10];

        [enc dispatchThreadgroups:MTLSizeMake(q_blocks, heads, batch)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    tc_set_last_backend(TC_BACKEND_TENSOROPS_M5);
    return TC_OK;
}
