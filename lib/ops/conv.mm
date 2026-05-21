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
#include <limits>

namespace {

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool conv_dims_valid(int batch, int in_channels, int out_channels,
                     int H, int W_in, int kH, int kW,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w,
                     int out_H, int out_W) {
    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 ||
        H <= 0 || W_in <= 0 || kH <= 0 || kW <= 0 ||
        pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0 ||
        out_H <= 0 || out_W <= 0) {
        return false;
    }
    const int expect_H = (H + 2 * pad_h - kH) / stride_h + 1;
    const int expect_W = (W_in + 2 * pad_w - kW) / stride_w + 1;
    return expect_H == out_H && expect_W == out_W;
}

bool conv_bytes(int batch, int in_channels, int out_channels,
                int H, int W_in, int kH, int kW,
                int out_H, int out_W,
                size_t* x_bytes, size_t* weight_bytes,
                size_t* y_bytes, size_t* scratch_col_bytes,
                size_t* dx_f32_bytes) {
    size_t x_elems = 0, w_elems = 0, y_elems = 0, k_elems = 0;
    size_t out_hw = 0, col_elems = 0, tmp = 0;
    if (!checked_mul((size_t)H, (size_t)W_in, &tmp) ||
        !checked_mul((size_t)batch, (size_t)in_channels, &x_elems) ||
        !checked_mul(x_elems, tmp, &x_elems) ||
        !checked_mul((size_t)out_channels, (size_t)in_channels, &w_elems) ||
        !checked_mul(w_elems, (size_t)kH, &w_elems) ||
        !checked_mul(w_elems, (size_t)kW, &w_elems) ||
        !checked_mul((size_t)out_H, (size_t)out_W, &out_hw) ||
        !checked_mul((size_t)batch, (size_t)out_channels, &y_elems) ||
        !checked_mul(y_elems, out_hw, &y_elems) ||
        !checked_mul((size_t)in_channels, (size_t)kH, &k_elems) ||
        !checked_mul(k_elems, (size_t)kW, &k_elems) ||
        !checked_mul((size_t)batch, k_elems, &col_elems) ||
        !checked_mul(col_elems, out_hw, &col_elems)) {
        return false;
    }
    return checked_mul(x_elems, sizeof(uint16_t), x_bytes) &&
           checked_mul(w_elems, sizeof(uint16_t), weight_bytes) &&
           checked_mul(y_elems, sizeof(uint16_t), y_bytes) &&
           checked_mul(col_elems, sizeof(uint16_t), scratch_col_bytes) &&
           checked_mul(x_elems, sizeof(float), dx_f32_bytes);
}

tc_status_t validate_conv_common(tc_context* ctx,
                                 int batch, int in_channels, int out_channels,
                                 int H, int W_in, int kH, int kW,
                                 int pad_h, int pad_w,
                                 int stride_h, int stride_w,
                                 int out_H, int out_W,
                                 size_t* x_bytes, size_t* weight_bytes,
                                 size_t* y_bytes, size_t* scratch_col_bytes,
                                 size_t* dx_f32_bytes) {
    if (!conv_dims_valid(batch, in_channels, out_channels, H, W_in, kH, kW,
                         pad_h, pad_w, stride_h, stride_w, out_H, out_W)) {
        return TC_ERR_INVALID_SHAPE;
    }
    if (!conv_bytes(batch, in_channels, out_channels, H, W_in, kH, kW,
                    out_H, out_W, x_bytes, weight_bytes, y_bytes,
                    scratch_col_bytes, dx_f32_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    (void)ctx;
    return TC_OK;
}

}  // namespace

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
    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0, scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(ctx, batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dY, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, weight, weight_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dX, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_dX_f32, dx_f32_bytes);
    if (s != TC_OK) return s;

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
    const uint32_t lda = (uint32_t)K;       /* weight is [OC, K], transposed */
    const uint32_t ldb = (uint32_t)out_hw;  /* dY is [OC, out_hw] */
    const uint32_t ldc = (uint32_t)out_hw;  /* dCol is [K, out_hw] */
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
            [enc setBytes:&lda   length:sizeof(lda)   atIndex:8];
            [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:9];
            [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:10];
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
    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0, scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(ctx, batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dY, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, dW, weight_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;

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
    const uint32_t lda = (uint32_t)out_hw;  /* dY is [OC, out_hw] */
    const uint32_t ldb = (uint32_t)out_hw;  /* col is [K, out_hw], transposed */
    const uint32_t ldc = (uint32_t)K;       /* dW is [OC, K] */
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
            [enc setBytes:&lda   length:sizeof(lda)   atIndex:8];
            [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:9];
            [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:10];
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
    size_t x_bytes = 0, weight_bytes = 0, y_bytes = 0, scratch_col_bytes = 0, dx_f32_bytes = 0;
    tc_status_t s = validate_conv_common(ctx, batch, in_channels, out_channels,
                                         H, W_in, kH, kW, pad_h, pad_w,
                                         stride_h, stride_w, out_H, out_W,
                                         &x_bytes, &weight_bytes, &y_bytes,
                                         &scratch_col_bytes, &dx_f32_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, weight, weight_bytes);
    if (s != TC_OK) return s;
    if (bias) {
        s = tc_buffer_validate(ctx, bias, (size_t)out_channels * sizeof(uint16_t));
        if (s != TC_OK) return s;
    }
    s = tc_buffer_validate(ctx, Y, y_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, scratch_col, scratch_col_bytes);
    if (s != TC_OK) return s;

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

    /* Per-batch GEMM: Y[n] = W @ col[n]. W is [OC, K], col[n] is
     * [K, out_hw], Y[n] is [OC, out_hw]. Bind per-batch MTLBuffer offsets
     * directly because the public GEMM API intentionally has no offset args. */
    NSString* kname = @"tc_gemm_f16_f32";
    id<MTLComputePipelineState> pso = nil;
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        bool ta = false, tb = false;
        [cv setConstantValue:&ta type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&tb type:MTLDataTypeBool atIndex:1];
        NSError* nserr = nil;
        id<MTLFunction> fn = [ctx->library newFunctionWithName:kname
                                                constantValues:cv error:&nserr];
        if (!fn) return TC_ERR_KERNEL_NOT_FOUND;
        pso = [ctx->device newComputePipelineStateWithFunction:fn error:&nserr];
        if (!pso) return TC_ERR_PIPELINE;
    }

    const uint32_t M_u = OC;
    const uint32_t N_u = out_hw;
    const uint32_t K_u = K;
    const uint32_t lda = K;       /* weight is [OC, K] */
    const uint32_t ldb = out_hw;  /* col is [K, out_hw] */
    const uint32_t ldc = out_hw;  /* Y is [OC, out_hw] */
    float alpha = 1.0f, beta = 0.0f;
    const size_t stride_col = (size_t)K * out_hw * sizeof(uint16_t);
    const size_t stride_Y = (size_t)OC * out_hw * sizeof(uint16_t);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        for (uint32_t n = 0; n < B; ++n) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:weight->mtl      offset:0              atIndex:0];
            [enc setBuffer:scratch_col->mtl offset:n * stride_col atIndex:1];
            [enc setBuffer:Y->mtl           offset:n * stride_Y   atIndex:2];
            [enc setBytes:&M_u   length:sizeof(M_u)   atIndex:3];
            [enc setBytes:&N_u   length:sizeof(N_u)   atIndex:4];
            [enc setBytes:&K_u   length:sizeof(K_u)   atIndex:5];
            [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];
            [enc setBytes:&beta  length:sizeof(beta)  atIndex:7];
            [enc setBytes:&lda   length:sizeof(lda)   atIndex:8];
            [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:9];
            [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:10];
            const uint32_t gx = (N_u + 64 - 1) / 64;
            const uint32_t gy = (M_u + 64 - 1) / 64;
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            fprintf(stderr, "[tensorcore] conv2d_forward GEMM: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return TC_ERR_DISPATCH;
        }
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
