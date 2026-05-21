/*
 * tensorcore — Conv2D forward host dispatch.
 *
 * Strategy: im2col into scratch, then call tc_gemm.
 *   col[batch, K, out_hw] where K = in_channels*kH*kW
 *   W flattened to [out_channels, K]
 *   GEMM: Y[batch, out_channels, out_hw] = W [oc, K] @ col[K, out_hw]
 *
 * Per-batch GEMM dispatch (no batched-GEMM kernel yet in v0.1 — phase 2).
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "tensorcore/conv.h"
#include "../core/internal.h"

#include <cstdio>

extern "C" tc_status_t tc_conv2d_backward_input(tc_context* ctx,
                                                const tc_buffer* dY,
                                                const tc_buffer* weight,
                                                tc_buffer*       dX,
                                                tc_buffer*       scratch_col,
                                                tc_buffer*       scratch_dX_f32,
                                                int batch, int in_channels, int out_channels,
                                                int H, int W_in, int kH, int kW,
                                                int pad_h, int pad_w,
                                                int stride_h, int stride_w,
                                                int out_H, int out_W) {
    if (!ctx || !dY || !weight || !dX || !scratch_col || !scratch_dX_f32)
        return TC_ERR_INVALID_ARG;

    /* dCol = W^T @ dY for each batch (no cross-batch dependency).
     * W is [OC, K]; dY is [N, OC, out_hw]; dCol is [N, K, out_hw].
     * Loop over batches encoding each as a separate dispatch with
     * MTLBuffer offset. tc_gemm doesn't support offsets, so we encode
     * the GEMM directly. */
    const int OC = out_channels;
    const int K  = in_channels * kH * kW;
    const int out_hw = out_H * out_W;

    NSString* kname = @"tc_gemm_f16_f32";
    id<MTLComputePipelineState> pso = nil;
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        bool ta = true, tb = false;
        [cv setConstantValue:&ta type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&tb type:MTLDataTypeBool atIndex:1];
        NSError* nserr = nil;
        id<MTLFunction> fn = [ctx->library newFunctionWithName:kname
                                                constantValues:cv error:&nserr];
        if (!fn) return TC_ERR_KERNEL_NOT_FOUND;
        pso = [ctx->device newComputePipelineStateWithFunction:fn error:&nserr];
        if (!pso) return TC_ERR_PIPELINE;
    }

    const uint32_t M_u = (uint32_t)K, N_u = (uint32_t)out_hw, K_u = (uint32_t)OC;
    float alpha = 1.0f, beta = 0.0f;
    const size_t stride_dY  = (size_t)OC * out_hw * sizeof(uint16_t);
    const size_t stride_col = (size_t)K * out_hw * sizeof(uint16_t);

    @autoreleasepool {
        id<MTLCommandBuffer> dcol_cmd = [ctx->queue commandBuffer];
        for (int n = 0; n < batch; ++n) {
            id<MTLComputeCommandEncoder> enc = [dcol_cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight->mtl      offset:0              atIndex:0];
            [enc setBuffer:dY->mtl          offset:n * stride_dY  atIndex:1];
            [enc setBuffer:scratch_col->mtl offset:n * stride_col atIndex:2];
            [enc setBytes:&M_u   length:sizeof(M_u)   atIndex:3];
            [enc setBytes:&N_u   length:sizeof(N_u)   atIndex:4];
            [enc setBytes:&K_u   length:sizeof(K_u)   atIndex:5];
            [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];
            [enc setBytes:&beta  length:sizeof(beta)  atIndex:7];
            const uint32_t gx = (N_u + 64 - 1) / 64;
            const uint32_t gy = (M_u + 64 - 1) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
        }
        [dcol_cmd commit];
        [dcol_cmd waitUntilCompleted];
        if (dcol_cmd.error) {
            fprintf(stderr, "[tensorcore] conv2d_backward_input dCol: %s\n",
                    [[dcol_cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    /* Zero the fp32 accumulation buffer for atomic-add scatter. */
    void* dx32; tc_buffer_map(scratch_dX_f32, &dx32);
    memset(dx32, 0, (size_t)batch * in_channels * H * W_in * sizeof(float));

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso_col2im =
        tc_pipeline_get(ctx, @"tc_col2im_atomic_f32", &err);
    if (!pso_col2im) return err;
    id<MTLComputePipelineState> pso_fin =
        tc_pipeline_get(ctx, @"tc_col2im_finalize_f16", &err);
    if (!pso_fin) return err;

    const uint32_t B = batch, IC = in_channels;
    const uint32_t Hu = H, Wu = W_in, kHu = kH, kWu = kW;
    const int32_t pH = pad_h, pW = pad_w;
    const uint32_t sH = stride_h, sW = stride_w;
    const uint32_t oH = out_H, oW = out_W;
    const uint32_t Ktot = (uint32_t)K;
    const uint32_t n_elems = B * IC * Hu * Wu;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        /* Scatter dCol into dX_fp32 with atomic add. */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_col2im];
            [enc setBuffer:scratch_col->mtl    offset:0 atIndex:0];
            [enc setBuffer:scratch_dX_f32->mtl offset:0 atIndex:1];
            [enc setBytes:&B   length:sizeof(B)   atIndex:2];
            [enc setBytes:&IC  length:sizeof(IC)  atIndex:3];
            [enc setBytes:&Hu  length:sizeof(Hu)  atIndex:4];
            [enc setBytes:&Wu  length:sizeof(Wu)  atIndex:5];
            [enc setBytes:&kHu length:sizeof(kHu) atIndex:6];
            [enc setBytes:&kWu length:sizeof(kWu) atIndex:7];
            [enc setBytes:&pH  length:sizeof(pH)  atIndex:8];
            [enc setBytes:&pW  length:sizeof(pW)  atIndex:9];
            [enc setBytes:&sH  length:sizeof(sH)  atIndex:10];
            [enc setBytes:&sW  length:sizeof(sW)  atIndex:11];
            [enc setBytes:&oH  length:sizeof(oH)  atIndex:12];
            [enc setBytes:&oW  length:sizeof(oW)  atIndex:13];
            [enc dispatchThreads:MTLSizeMake(out_hw, Ktot, B)
              threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc endEncoding];
        }
        /* Finalize: fp32 → fp16 dX. */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_fin];
            [enc setBuffer:scratch_dX_f32->mtl offset:0 atIndex:0];
            [enc setBuffer:dX->mtl             offset:0 atIndex:1];
            [enc setBytes:&n_elems length:sizeof(n_elems) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n_elems, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_conv2d_backward_weight(tc_context* ctx,
                                                 const tc_buffer* X,
                                                 const tc_buffer* dY,
                                                 tc_buffer*       dW,
                                                 tc_buffer*       scratch_col,
                                                 int batch, int in_channels, int out_channels,
                                                 int H, int W_in, int kH, int kW,
                                                 int pad_h, int pad_w,
                                                 int stride_h, int stride_w,
                                                 int out_H, int out_W) {
    if (!ctx || !X || !dY || !dW || !scratch_col) return TC_ERR_INVALID_ARG;

    /* First re-run im2col on X to produce col. */
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso_im2col = tc_pipeline_get(ctx, @"tc_im2col_f16", &err);
    if (!pso_im2col) return err;

    const uint32_t B = batch, IC = in_channels;
    const uint32_t Hu = H, Wu = W_in, kHu = kH, kWu = kW;
    const int32_t pH = pad_h, pW = pad_w;
    const uint32_t sH = stride_h, sW = stride_w;
    const uint32_t oH = out_H, oW = out_W;
    const uint32_t K = IC * kHu * kWu;
    const uint32_t out_hw = oH * oW;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_im2col];
        [enc setBuffer:X->mtl           offset:0 atIndex:0];
        [enc setBuffer:scratch_col->mtl offset:0 atIndex:1];
        [enc setBytes:&B   length:sizeof(B)   atIndex:2];
        [enc setBytes:&IC  length:sizeof(IC)  atIndex:3];
        [enc setBytes:&Hu  length:sizeof(Hu)  atIndex:4];
        [enc setBytes:&Wu  length:sizeof(Wu)  atIndex:5];
        [enc setBytes:&kHu length:sizeof(kHu) atIndex:6];
        [enc setBytes:&kWu length:sizeof(kWu) atIndex:7];
        [enc setBytes:&pH  length:sizeof(pH)  atIndex:8];
        [enc setBytes:&pW  length:sizeof(pW)  atIndex:9];
        [enc setBytes:&sH  length:sizeof(sH)  atIndex:10];
        [enc setBytes:&sW  length:sizeof(sW)  atIndex:11];
        [enc setBytes:&oH  length:sizeof(oH)  atIndex:12];
        [enc setBytes:&oW  length:sizeof(oW)  atIndex:13];
        [enc dispatchThreads:MTLSizeMake(out_hw, K, B)
          threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }

    /* dW[oc, K] = sum_n dY[n, oc, out_hw] @ col[n, K, out_hw]^T.
     *
     * For N>1 we accumulate via beta=1 on subsequent batches. Buffer offsets
     * are encoded by binding the same MTLBuffer with different `offset:` values.
     * Bypasses tc_gemm's no-offset API by encoding directly here. */
    NSString* kname = @"tc_gemm_f16_f32";
    id<MTLComputePipelineState> pso = nil;
    /* Specialize with transpose_b=true. */
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        bool ta = false, tb = true;
        [cv setConstantValue:&ta type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&tb type:MTLDataTypeBool atIndex:1];
        NSError* nserr = nil;
        id<MTLFunction> fn = [ctx->library newFunctionWithName:kname
                                                constantValues:cv error:&nserr];
        if (!fn) return TC_ERR_KERNEL_NOT_FOUND;
        pso = [ctx->device newComputePipelineStateWithFunction:fn error:&nserr];
        if (!pso) return TC_ERR_PIPELINE;
    }

    const uint32_t M_u = (uint32_t)out_channels;
    const uint32_t N_u = K;          /* dW cols (in elements)             */
    const uint32_t K_u = out_hw;     /* GEMM K (the contracted dim)       */
    float alpha = 1.0f;
    /* dY layout: [N batches, OC, out_hw], stride between batches = OC*out_hw.
     * col layout: [N batches, K, out_hw], stride = K*out_hw.               */
    const size_t stride_dY  = (size_t)out_channels * out_hw * sizeof(uint16_t);
    const size_t stride_col = (size_t)K * out_hw * sizeof(uint16_t);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        for (int n = 0; n < batch; ++n) {
            const float beta = (n == 0) ? 0.0f : 1.0f;
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:dY->mtl          offset:n * stride_dY  atIndex:0];
            [enc setBuffer:scratch_col->mtl offset:n * stride_col atIndex:1];
            [enc setBuffer:dW->mtl          offset:0              atIndex:2];
            [enc setBytes:&M_u   length:sizeof(M_u)   atIndex:3];
            [enc setBytes:&N_u   length:sizeof(N_u)   atIndex:4];
            [enc setBytes:&K_u   length:sizeof(K_u)   atIndex:5];
            [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];
            [enc setBytes:&beta  length:sizeof(beta)  atIndex:7];
            const uint32_t gx = (N_u + 64 - 1) / 64;
            const uint32_t gy = (M_u + 64 - 1) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] conv2d_backward_weight: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
    }
    return TC_OK;
}

extern "C" tc_status_t tc_conv2d_forward(tc_context* ctx,
                                         const tc_buffer* X,
                                         const tc_buffer* weight,
                                         const tc_buffer* bias,
                                         tc_buffer*       Y,
                                         tc_buffer*       scratch_col,
                                         int batch, int in_channels, int out_channels,
                                         int H, int W_in, int kH, int kW,
                                         int pad_h, int pad_w,
                                         int stride_h, int stride_w,
                                         int out_H, int out_W) {
    if (!ctx || !X || !weight || !Y || !scratch_col) return TC_ERR_INVALID_ARG;
    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 ||
        H <= 0 || W_in <= 0 || kH <= 0 || kW <= 0 ||
        out_H <= 0 || out_W <= 0)
        return TC_ERR_INVALID_SHAPE;
    if (batch != 1) return TC_ERR_INVALID_SHAPE;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso_im2col =
        tc_pipeline_get(ctx, @"tc_im2col_f16", &err);
    if (!pso_im2col) return err;
    id<MTLComputePipelineState> pso_bias =
        tc_pipeline_get(ctx, @"tc_conv2d_bias_add_f16", &err);
    if (!pso_bias) return err;

    const uint32_t B  = (uint32_t)batch;
    const uint32_t IC = (uint32_t)in_channels;
    const uint32_t OC = (uint32_t)out_channels;
    const uint32_t Hu = (uint32_t)H, Wu = (uint32_t)W_in;
    const uint32_t kHu = (uint32_t)kH, kWu = (uint32_t)kW;
    const uint32_t oH = (uint32_t)out_H, oW = (uint32_t)out_W;
    const int32_t pH = (int32_t)pad_h, pW = (int32_t)pad_w;
    const uint32_t sH = (uint32_t)stride_h, sW = (uint32_t)stride_w;
    const uint32_t K = IC * kHu * kWu;
    const uint32_t out_hw = oH * oW;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_im2col];
            [enc setBuffer:X->mtl           offset:0 atIndex:0];
            [enc setBuffer:scratch_col->mtl offset:0 atIndex:1];
            [enc setBytes:&B   length:sizeof(B)   atIndex:2];
            [enc setBytes:&IC  length:sizeof(IC)  atIndex:3];
            [enc setBytes:&Hu  length:sizeof(Hu)  atIndex:4];
            [enc setBytes:&Wu  length:sizeof(Wu)  atIndex:5];
            [enc setBytes:&kHu length:sizeof(kHu) atIndex:6];
            [enc setBytes:&kWu length:sizeof(kWu) atIndex:7];
            [enc setBytes:&pH  length:sizeof(pH)  atIndex:8];
            [enc setBytes:&pW  length:sizeof(pW)  atIndex:9];
            [enc setBytes:&sH  length:sizeof(sH)  atIndex:10];
            [enc setBytes:&sW  length:sizeof(sW)  atIndex:11];
            [enc setBytes:&oH  length:sizeof(oH)  atIndex:12];
            [enc setBytes:&oW  length:sizeof(oW)  atIndex:13];
            [enc dispatchThreads:MTLSizeMake(out_hw, K, B)
              threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }

    /* Per-batch GEMM: Y[n] = W @ col[n].  W is [OC, K], col[n] is [K, out_hw],
     * Y[n] is [OC, out_hw]. Stride between batches is K*out_hw for col and
     * OC*out_hw for Y. */
    /* For simplicity, loop over batches at the host side. */
    for (uint32_t n = 0; n < B; ++n) {
        tc_gemm_desc d = {};
        d.M = (int32_t)OC;
        d.N = (int32_t)out_hw;
        d.K = (int32_t)K;
        d.a_dtype = TC_DTYPE_F16;
        d.b_dtype = TC_DTYPE_F16;
        d.c_dtype = TC_DTYPE_F16;
        d.accum_dtype = TC_DTYPE_F32;
        d.alpha = 1.0f;
        d.beta  = 0.0f;
        /* Custom offsets via temporary tc_buffer aliases — for v0.1 we use
         * the existing tc_gemm signature which doesn't take offsets. We rely
         * on the kernels treating the whole buffer as one tile starting at 0.
         * Multi-batch convs therefore require the caller to call this per
         * batch with sub-buffer aliases, or we pass the full buffer here and
         * trust the kernel grid sizing. Phase 2 adds a batched-GEMM path. */
        if (n > 0) break;  /* TODO: batched dispatch in v0.2 */
        tc_status_t s = tc_gemm(ctx, &d, weight, scratch_col, Y);
        if (s != TC_OK) return s;
    }

    if (bias) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_bias];
            [enc setBuffer:Y->mtl    offset:0 atIndex:0];
            [enc setBuffer:bias->mtl offset:0 atIndex:1];
            [enc setBytes:&B  length:sizeof(B)  atIndex:2];
            [enc setBytes:&OC length:sizeof(OC) atIndex:3];
            [enc setBytes:&out_hw length:sizeof(out_hw) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(out_hw, OC, B)
              threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.error) return TC_ERR_DISPATCH;
        }
    }
    return TC_OK;
}
