/*
 * tensorcore — host dispatch for the fused training kernels.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdio>

namespace {

inline id<MTLComputePipelineState> pso_for(tc_context* ctx, NSString* name,
                                            tc_status_t* err) {
    return tc_pipeline_get(ctx, name, err);
}

/* Pick threadgroup size based on D — clamp to maxThreadsPerThreadgroup. */
inline uint32_t threads_for_d(uint32_t D) {
    uint32_t t = 256;
    if (D < 256) t = 128;
    if (D < 128) t = 64;
    if (D < 64)  t = 32;
    return t;
}

}  /* namespace */

extern "C" tc_status_t tc_rmsnorm_forward(tc_context* ctx,
                                          const tc_buffer* X, const tc_buffer* gamma,
                                          tc_buffer* Y, tc_buffer* rstd_out,
                                          int N, int D, float eps) {
    if (!ctx || !X || !gamma || !Y || !rstd_out || N <= 0 || D <= 0)
        return TC_ERR_INVALID_ARG;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_rmsnorm_forward", &err);
    if (!pso) return err;

    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl        offset:0 atIndex:0];
        [enc setBuffer:gamma->mtl    offset:0 atIndex:1];
        [enc setBuffer:Y->mtl        offset:0 atIndex:2];
        [enc setBuffer:rstd_out->mtl offset:0 atIndex:3];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:4];
        [enc setBytes:&D_u length:sizeof(D_u) atIndex:5];
        [enc setBytes:&eps length:sizeof(eps) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_rmsnorm_backward(tc_context* ctx,
                                           const tc_buffer* X, const tc_buffer* gamma,
                                           const tc_buffer* dY, const tc_buffer* rstd,
                                           tc_buffer* dX, tc_buffer* dgamma,
                                           int N, int D) {
    if (!ctx || !X || !gamma || !dY || !rstd || !dX || !dgamma || N <= 0 || D <= 0)
        return TC_ERR_INVALID_ARG;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso_bw = pso_for(ctx, @"tc_rmsnorm_backward", &err);
    if (!pso_bw) return err;
    id<MTLComputePipelineState> pso_dg = pso_for(ctx, @"tc_rmsnorm_reduce_dgamma", &err);
    if (!pso_dg) return err;

    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

        /* dX kernel */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_bw];
            [enc setBuffer:X->mtl      offset:0 atIndex:0];
            [enc setBuffer:gamma->mtl  offset:0 atIndex:1];
            [enc setBuffer:dY->mtl     offset:0 atIndex:2];
            [enc setBuffer:rstd->mtl   offset:0 atIndex:3];
            [enc setBuffer:dX->mtl     offset:0 atIndex:4];
            [enc setBuffer:dgamma->mtl offset:0 atIndex:5];
            [enc setBytes:&N_u length:sizeof(N_u) atIndex:6];
            [enc setBytes:&D_u length:sizeof(D_u) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
            [enc endEncoding];
        }
        /* dgamma reduction kernel */
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_dg];
            [enc setBuffer:X->mtl      offset:0 atIndex:0];
            [enc setBuffer:dY->mtl     offset:0 atIndex:1];
            [enc setBuffer:rstd->mtl   offset:0 atIndex:2];
            [enc setBuffer:dgamma->mtl offset:0 atIndex:3];
            [enc setBytes:&N_u length:sizeof(N_u) atIndex:4];
            [enc setBytes:&D_u length:sizeof(D_u) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(D_u, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_layernorm_forward(tc_context* ctx,
                                            const tc_buffer* X, const tc_buffer* gamma,
                                            const tc_buffer* beta,
                                            tc_buffer* Y, tc_buffer* mean_out,
                                            tc_buffer* rstd_out,
                                            int N, int D, float eps) {
    if (!ctx || !X || !gamma || !beta || !Y || !mean_out || !rstd_out)
        return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_layernorm_forward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl        offset:0 atIndex:0];
        [enc setBuffer:gamma->mtl    offset:0 atIndex:1];
        [enc setBuffer:beta->mtl     offset:0 atIndex:2];
        [enc setBuffer:Y->mtl        offset:0 atIndex:3];
        [enc setBuffer:mean_out->mtl offset:0 atIndex:4];
        [enc setBuffer:rstd_out->mtl offset:0 atIndex:5];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:6];
        [enc setBytes:&D_u length:sizeof(D_u) atIndex:7];
        [enc setBytes:&eps length:sizeof(eps) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_layernorm_backward(tc_context* ctx,
                                             const tc_buffer* X, const tc_buffer* gamma,
                                             const tc_buffer* dY,
                                             const tc_buffer* mean, const tc_buffer* rstd,
                                             tc_buffer* dX,
                                             int N, int D) {
    if (!ctx || !X || !gamma || !dY || !mean || !rstd || !dX)
        return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_layernorm_backward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl     offset:0 atIndex:0];
        [enc setBuffer:gamma->mtl offset:0 atIndex:1];
        [enc setBuffer:dY->mtl    offset:0 atIndex:2];
        [enc setBuffer:mean->mtl  offset:0 atIndex:3];
        [enc setBuffer:rstd->mtl  offset:0 atIndex:4];
        [enc setBuffer:dX->mtl    offset:0 atIndex:5];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:6];
        [enc setBytes:&D_u length:sizeof(D_u) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_rope_forward(tc_context* ctx,
                                       tc_buffer* X,
                                       const tc_buffer* cos_t, const tc_buffer* sin_t,
                                       int batch, int heads, int seq, int head_dim) {
    if (!ctx || !X || !cos_t || !sin_t || batch <= 0 || heads <= 0 || seq <= 0 ||
        head_dim <= 0 || head_dim % 2 != 0)
        return TC_ERR_INVALID_ARG;
    const size_t x_bytes = (size_t)batch * heads * seq * head_dim * sizeof(uint16_t);
    const size_t table_bytes = (size_t)seq * (head_dim / 2) * sizeof(float);
    tc_status_t s = tc_buffer_validate(ctx, X, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, cos_t, table_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, sin_t, table_bytes);
    if (s != TC_OK) return s;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_rope_forward", &err);
    if (!pso) return err;
    const uint32_t B = (uint32_t)batch, H = (uint32_t)heads,
                   S = (uint32_t)seq,   D = (uint32_t)head_dim;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl     offset:0 atIndex:0];
        [enc setBuffer:cos_t->mtl offset:0 atIndex:1];
        [enc setBuffer:sin_t->mtl offset:0 atIndex:2];
        [enc setBytes:&B length:sizeof(B) atIndex:3];
        [enc setBytes:&H length:sizeof(H) atIndex:4];
        [enc setBytes:&S length:sizeof(S) atIndex:5];
        [enc setBytes:&D length:sizeof(D) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(D / 2, S * H, B)
          threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_rope_backward(tc_context* ctx,
                                        tc_buffer* dX,
                                        const tc_buffer* cos_t, const tc_buffer* sin_t,
                                        int batch, int heads, int seq, int head_dim) {
    if (!ctx || !dX || !cos_t || !sin_t || batch <= 0 || heads <= 0 || seq <= 0 ||
        head_dim <= 0 || head_dim % 2 != 0)
        return TC_ERR_INVALID_ARG;
    const size_t x_bytes = (size_t)batch * heads * seq * head_dim * sizeof(uint16_t);
    const size_t table_bytes = (size_t)seq * (head_dim / 2) * sizeof(float);
    tc_status_t s = tc_buffer_validate(ctx, dX, x_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, cos_t, table_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, sin_t, table_bytes);
    if (s != TC_OK) return s;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_rope_backward", &err);
    if (!pso) return err;
    const uint32_t B = (uint32_t)batch, H = (uint32_t)heads,
                   S = (uint32_t)seq,   D = (uint32_t)head_dim;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:dX->mtl    offset:0 atIndex:0];
        [enc setBuffer:cos_t->mtl offset:0 atIndex:1];
        [enc setBuffer:sin_t->mtl offset:0 atIndex:2];
        [enc setBytes:&B length:sizeof(B) atIndex:3];
        [enc setBytes:&H length:sizeof(H) atIndex:4];
        [enc setBytes:&S length:sizeof(S) atIndex:5];
        [enc setBytes:&D length:sizeof(D) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(D / 2, S * H, B)
          threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_swiglu_forward(tc_context* ctx,
                                         const tc_buffer* gate, const tc_buffer* up,
                                         tc_buffer* out, int n) {
    if (!ctx || !gate || !up || !out || n <= 0) return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_swiglu_forward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)n;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:gate->mtl offset:0 atIndex:0];
        [enc setBuffer:up->mtl   offset:0 atIndex:1];
        [enc setBuffer:out->mtl  offset:0 atIndex:2];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N_u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_swiglu_backward(tc_context* ctx,
                                          const tc_buffer* gate, const tc_buffer* up,
                                          const tc_buffer* dout,
                                          tc_buffer* dgate, tc_buffer* dup, int n) {
    if (!ctx || !gate || !up || !dout || !dgate || !dup || n <= 0)
        return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_swiglu_backward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)n;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:gate->mtl  offset:0 atIndex:0];
        [enc setBuffer:up->mtl    offset:0 atIndex:1];
        [enc setBuffer:dout->mtl  offset:0 atIndex:2];
        [enc setBuffer:dgate->mtl offset:0 atIndex:3];
        [enc setBuffer:dup->mtl   offset:0 atIndex:4];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N_u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_softmax_forward(tc_context* ctx,
                                          const tc_buffer* X, tc_buffer* Y,
                                          int N, int D) {
    if (!ctx || !X || !Y || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_softmax_forward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl offset:0 atIndex:0];
        [enc setBuffer:Y->mtl offset:0 atIndex:1];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:2];
        [enc setBytes:&D_u length:sizeof(D_u) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_softmax_backward(tc_context* ctx,
                                           const tc_buffer* Y, const tc_buffer* dY,
                                           tc_buffer* dX, int N, int D) {
    if (!ctx || !Y || !dY || !dX || N <= 0 || D <= 0) return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_softmax_backward", &err);
    if (!pso) return err;
    const uint32_t N_u = (uint32_t)N, D_u = (uint32_t)D;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:Y->mtl  offset:0 atIndex:0];
        [enc setBuffer:dY->mtl offset:0 atIndex:1];
        [enc setBuffer:dX->mtl offset:0 atIndex:2];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:2 + 1];
        [enc setBytes:&D_u length:sizeof(D_u) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(N_u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_for_d(D_u), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_fused_rmsnorm_gemv(tc_context* ctx,
                                             const tc_buffer* X,
                                             const tc_buffer* gamma,
                                             const tc_buffer* W,
                                             tc_buffer*       Y,
                                             int M, int N, int K, float eps) {
    if (!ctx || !X || !gamma || !W || !Y || M <= 0 || N <= 0 || K <= 0)
        return TC_ERR_INVALID_ARG;
    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, @"tc_fused_rmsnorm_gemv_f16", &err);
    if (!pso) return err;

    const uint32_t M_u = (uint32_t)M, N_u = (uint32_t)N, K_u = (uint32_t)K;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X->mtl     offset:0 atIndex:0];
        [enc setBuffer:gamma->mtl offset:0 atIndex:1];
        [enc setBuffer:W->mtl     offset:0 atIndex:2];
        [enc setBuffer:Y->mtl     offset:0 atIndex:3];
        [enc setBytes:&M_u length:sizeof(M_u) atIndex:4];
        [enc setBytes:&N_u length:sizeof(N_u) atIndex:5];
        [enc setBytes:&K_u length:sizeof(K_u) atIndex:6];
        [enc setBytes:&eps length:sizeof(eps) atIndex:7];
        const uint32_t T = threads_for_d(K_u);
        [enc dispatchThreadgroups:MTLSizeMake(N_u, M_u, 1)
            threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}

extern "C" tc_status_t tc_adamw_step(tc_context* ctx,
                                     tc_buffer* params_fp32,
                                     tc_buffer* m_fp32, tc_buffer* v_fp32,
                                     const tc_buffer* grads,
                                     tc_dtype_t grad_dtype,
                                     int n,
                                     float lr, float beta1, float beta2, float eps,
                                     float wd, float bc1, float bc2) {
    if (!ctx || !params_fp32 || !m_fp32 || !v_fp32 || !grads || n <= 0)
        return TC_ERR_INVALID_ARG;

    NSString* kname = nil;
    if (grad_dtype == TC_DTYPE_F32)      kname = @"tc_adamw_step_f32";
    else if (grad_dtype == TC_DTYPE_F16) kname = @"tc_adamw_step_f16grad";
    else return TC_ERR_UNSUPPORTED_DTYPE;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = pso_for(ctx, kname, &err);
    if (!pso) return err;

    const uint32_t N_u = (uint32_t)n;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:params_fp32->mtl offset:0 atIndex:0];
        [enc setBuffer:m_fp32->mtl      offset:0 atIndex:1];
        [enc setBuffer:v_fp32->mtl      offset:0 atIndex:2];
        [enc setBuffer:grads->mtl       offset:0 atIndex:3];
        [enc setBytes:&N_u   length:sizeof(N_u)   atIndex:4];
        [enc setBytes:&lr    length:sizeof(lr)    atIndex:5];
        [enc setBytes:&beta1 length:sizeof(beta1) atIndex:6];
        [enc setBytes:&beta2 length:sizeof(beta2) atIndex:7];
        [enc setBytes:&eps   length:sizeof(eps)   atIndex:8];
        [enc setBytes:&wd    length:sizeof(wd)    atIndex:9];
        [enc setBytes:&bc1   length:sizeof(bc1)   atIndex:10];
        [enc setBytes:&bc2   length:sizeof(bc2)   atIndex:11];
        [enc dispatchThreads:MTLSizeMake(N_u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    return TC_OK;
}
