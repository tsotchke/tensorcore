/*
 * tensorcore — GEMM dispatch.
 *
 * Selects (kernel, backend) based on (dtype combo, GPU family, shape).
 * v0.1 dispatch table:
 *
 *   (F16,F16,F16,F32)              Apple7+   tc_gemm_f16_f32     simdgroup_matrix
 *   (F32,F32,F32,F32)              Apple7+   tc_gemm_f32_f32     simdgroup_matrix
 *   (BF16,BF16,BF16,F32)           Apple9+   tc_gemm_bf16_f32    simdgroup_matrix
 *   (I8, I8, I32, I32)             Apple10+  tc_gemm_i8_i32      simdgroup_matrix
 *   anything else / older family   any       MPSMatrixMultiplication fallback
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tensorcore/tensorcore.h"
#include "../core/internal.h"

#include <cstdio>
#include <limits>

extern "C" tc_status_t tc_mps_gemm(tc_context* ctx,
                                   const tc_gemm_desc* desc,
                                   const tc_buffer* A,
                                   const tc_buffer* B,
                                   tc_buffer*       C);

#ifdef TC_HAVE_METAL4_SDK
extern "C" tc_status_t tc_tensorops_gemm_attempt(tc_context* ctx,
                                                 const tc_gemm_desc* desc,
                                                 const tc_buffer* A,
                                                 const tc_buffer* B,
                                                 tc_buffer*       C);
#endif

namespace {

/* Tile-size heuristic: 128×128 wins for any dimension ≥ 256 on most shapes;
 * 64×64 is better for small shapes where 128×128's higher TG mem hurts
 * occupancy. */
struct TileChoice {
    NSString* kernel_name;
    uint32_t  BM, BN;
    uint32_t  threads_per_tg;
};

/* The 128×128 tile is built but not yet a perf win on Apple7/8 — register
 * pressure (16 acc fragments per simdgroup) causes spills. Phase-2 will refine
 * its layout. Default to 64×64 for v0.1. */
static bool use_128_tile(const tc_gemm_desc* /*d*/) {
    const char* opt = getenv("TC_USE_128_TILE");
    return opt && opt[0] == '1';
}

static bool use_async_kernel(const tc_gemm_desc* /*d*/) {
    const char* opt = getenv("TC_USE_ASYNC");
    return opt && opt[0] == '1';
}

TileChoice kernel_for(const tc_gemm_desc* d, tc_family_t fam, tc_context* ctx, tc_status_t* err) {
    *err = TC_OK;
    const bool big = use_128_tile(d);
    const bool async_path = use_async_kernel(d);
    const uint32_t BM = big ? 128 : 64;
    const uint32_t BN = big ? 128 : 64;
    const uint32_t T  = big ? 512 : 128;

    /* f16/f16/f16 + f32 accum */
    if (d->a_dtype == TC_DTYPE_F16 && d->b_dtype == TC_DTYPE_F16 &&
        d->c_dtype == TC_DTYPE_F16 && d->accum_dtype == TC_DTYPE_F32) {
        if (fam < TC_FAMILY_APPLE7) { *err = TC_ERR_UNSUPPORTED_FAMILY; return {nil,0,0,0}; }
        if (async_path && !d->transpose_a && !d->transpose_b) {
            /* If async kernels were stripped from the build (macOS 26+ SDK),
             * silently fall through to the sync path. */
            id<MTLFunction> probe = [ctx->library newFunctionWithName:@"tc_gemm_f16_f32_async"];
            if (probe) {
                const char* a128 = getenv("TC_USE_ASYNC_128");
                if (a128 && a128[0] == '1' &&
                    d->M >= 1024 && d->N >= 1024 && d->K >= 256) {
                    return { @"tc_gemm_f16_f32_async_128", 128, 128, 512 };
                }
                return { @"tc_gemm_f16_f32_async", 64, 64, 128 };
            }
        }
        return { big ? @"tc_gemm_f16_f32_128" : @"tc_gemm_f16_f32", BM, BN, T };
    }
    if (d->a_dtype == TC_DTYPE_F32 && d->b_dtype == TC_DTYPE_F32 &&
        d->c_dtype == TC_DTYPE_F32 && d->accum_dtype == TC_DTYPE_F32) {
        if (fam < TC_FAMILY_APPLE7) { *err = TC_ERR_UNSUPPORTED_FAMILY; return {nil,0,0,0}; }
        return { big ? @"tc_gemm_f32_f32_128" : @"tc_gemm_f32_f32", BM, BN, T };
    }
    if (d->a_dtype == TC_DTYPE_BF16 && d->b_dtype == TC_DTYPE_BF16 &&
        d->c_dtype == TC_DTYPE_BF16 && d->accum_dtype == TC_DTYPE_F32) {
        if (fam < TC_FAMILY_APPLE9) { *err = TC_ERR_UNSUPPORTED_FAMILY; return {nil,0,0,0}; }
        return { big ? @"tc_gemm_bf16_f32_128" : @"tc_gemm_bf16_f32", BM, BN, T };
    }
    if (d->a_dtype == TC_DTYPE_I8 && d->b_dtype == TC_DTYPE_I8 &&
        d->c_dtype == TC_DTYPE_I32 && d->accum_dtype == TC_DTYPE_I32) {
        if (fam < TC_FAMILY_APPLE10) { *err = TC_ERR_UNSUPPORTED_FAMILY; return {nil,0,0,0}; }
        /* 128-tile i8 variant lands in phase 2; for now use 64 tile. */
        return { @"tc_gemm_i8_i32", 64, 64, 128 };
    }
    *err = TC_ERR_UNSUPPORTED_DTYPE;
    return {nil,0,0,0};
}

bool validate(const tc_gemm_desc* d) {
    if (!d) return false;
    if (d->M <= 0 || d->N <= 0 || d->K <= 0) return false;
    if (d->transpose_a || d->transpose_b) return true;  /* allowed via function_constant */
    return true;
}

bool checked_mul(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

bool checked_add(size_t a, size_t b, size_t* out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool matrix_storage_bytes(int32_t rows, int32_t cols, int32_t ld,
                          tc_dtype_t dtype, size_t* out) {
    size_t row_offset = 0;
    size_t elems = 0;
    size_t bytes = 0;
    const size_t elem_size = tc_dtype_size(dtype);
    if (rows <= 0 || cols <= 0 || ld < cols || elem_size == 0) return false;
    if (!checked_mul((size_t)(rows - 1), (size_t)ld, &row_offset)) return false;
    if (!checked_add(row_offset, (size_t)cols, &elems)) return false;
    if (!checked_mul(elems, elem_size, &bytes)) return false;
    *out = bytes;
    return true;
}

int32_t effective_lda(const tc_gemm_desc* d) {
    return d->lda ? d->lda : (d->transpose_a ? d->M : d->K);
}

int32_t effective_ldb(const tc_gemm_desc* d) {
    return d->ldb ? d->ldb : (d->transpose_b ? d->K : d->N);
}

int32_t effective_ldc(const tc_gemm_desc* d) {
    return d->ldc ? d->ldc : d->N;
}

#ifdef TC_HAVE_METAL4_SDK
bool gemm_uses_default_layout(const tc_gemm_desc* d) {
    return effective_lda(d) == (d->transpose_a ? d->M : d->K) &&
           effective_ldb(d) == (d->transpose_b ? d->K : d->N) &&
           effective_ldc(d) == d->N;
}
#endif

tc_status_t validate_gemm_buffers(tc_context* ctx,
                                  const tc_gemm_desc* d,
                                  const tc_buffer* A,
                                  const tc_buffer* B,
                                  tc_buffer* C) {
    const int32_t a_rows = d->transpose_a ? d->K : d->M;
    const int32_t a_cols = d->transpose_a ? d->M : d->K;
    const int32_t b_rows = d->transpose_b ? d->N : d->K;
    const int32_t b_cols = d->transpose_b ? d->K : d->N;
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    if (!matrix_storage_bytes(a_rows, a_cols, effective_lda(d), d->a_dtype, &a_bytes) ||
        !matrix_storage_bytes(b_rows, b_cols, effective_ldb(d), d->b_dtype, &b_bytes) ||
        !matrix_storage_bytes(d->M, d->N, effective_ldc(d), d->c_dtype, &c_bytes)) {
        return TC_ERR_INVALID_ARG;
    }
    tc_status_t s = tc_buffer_validate(ctx, A, a_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, B, b_bytes);
    if (s != TC_OK) return s;
    return tc_buffer_validate(ctx, C, c_bytes);
}

bool batched_matrix_bytes(int32_t rows,
                          int32_t cols,
                          int32_t ld,
                          tc_dtype_t dtype,
                          int32_t batch,
                          int64_t stride_elems,
                          size_t* out) {
    size_t single_elems = 0;
    size_t total_elems = 0;
    const size_t elem_size = tc_dtype_size(dtype);
    if (rows <= 0 || cols <= 0 || ld < cols || batch <= 0 || elem_size == 0) return false;
    size_t row_offset = 0;
    if (!checked_mul((size_t)(rows - 1), (size_t)ld, &row_offset)) return false;
    if (!checked_add(row_offset, (size_t)cols, &single_elems)) return false;
    if (batch == 1) {
        total_elems = single_elems;
    } else {
        if (stride_elems < 0 || (uint64_t)stride_elems < single_elems) return false;
        size_t batch_offset = 0;
        if (!checked_mul((size_t)(batch - 1), (size_t)stride_elems, &batch_offset)) {
            return false;
        }
        if (!checked_add(batch_offset, single_elems, &total_elems)) return false;
    }
    return checked_mul(total_elems, elem_size, out);
}

id<MTLComputePipelineState> resolve_pipeline(tc_context* ctx,
                                             NSString* base_name,
                                             bool trans_a, bool trans_b,
                                             tc_status_t* err) {
    /* Specialize by function constants (transpose_a, transpose_b). Pipelines
     * are cached per (name, trans_a, trans_b); we encode the booleans in the
     * cache key string for now. */
    NSString* key = [NSString stringWithFormat:@"%@:ta=%d:tb=%d",
                                                base_name, trans_a ? 1 : 0, trans_b ? 1 : 0];
    /* Quick path: look up cached. */
    {
        NSError* nserr = nil;
        id<MTLComputePipelineState> cached = nil;
        @synchronized(ctx->pipelines) {
            cached = [(TCPipelineCache*)ctx->pipelines pipelines][key];
        }
        if (cached) { if (err) *err = TC_OK; return cached; }
        (void)nserr;
    }

    /* Specialized build. */
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&trans_a type:MTLDataTypeBool atIndex:0];
    [cv setConstantValue:&trans_b type:MTLDataTypeBool atIndex:1];

    NSError* nserr = nil;
    id<MTLFunction> fn = [ctx->library newFunctionWithName:base_name
                                            constantValues:cv
                                                     error:&nserr];
    if (!fn) {
        fprintf(stderr, "[tensorcore] specialize %s failed: %s\n",
                [base_name UTF8String],
                [[nserr localizedDescription] UTF8String]);
        if (err) *err = TC_ERR_KERNEL_NOT_FOUND;
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:fn error:&nserr];
    if (!pso) {
        fprintf(stderr, "[tensorcore] PSO %s failed: %s\n",
                [base_name UTF8String],
                [[nserr localizedDescription] UTF8String]);
        if (err) *err = TC_ERR_PIPELINE;
        return nil;
    }

    @synchronized(ctx->pipelines) {
        [(TCPipelineCache*)ctx->pipelines pipelines][key] = pso;
    }
    if (err) *err = TC_OK;
    return pso;
}

} /* namespace */

extern "C" tc_status_t tc_gemm(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer*       C) {
    if (!ctx)                  return TC_ERR_NOT_INITIALIZED;
    if (!validate(desc))       return TC_ERR_INVALID_ARG;
    if (!A || !B || !C)        return TC_ERR_INVALID_ARG;
    tc_status_t s = validate_gemm_buffers(ctx, desc, A, B, C);
    if (s != TC_OK) return s;

#ifdef TC_HAVE_METAL4_SDK
    /* Try the Metal 4 tensor_ops path first on M5+. It is gated on family
     * + device-name and silently returns TC_ERR_UNSUPPORTED_* otherwise. */
    if (ctx->info.supports_tensorops_m5 &&
        desc->alpha == 1.0f && desc->beta == 0.0f &&
        !desc->transpose_a && !desc->transpose_b &&
        gemm_uses_default_layout(desc)) {
        s = tc_tensorops_gemm_attempt(ctx, desc, A, B, C);
        if (s == TC_OK) return TC_OK;
        /* Anything else: fall through to the simdgroup_matrix path. */
    }
#endif

    tc_status_t err = TC_OK;
    TileChoice tile = kernel_for(desc, ctx->info.family, ctx, &err);
    if (!tile.kernel_name) {
        tc_set_last_backend(TC_BACKEND_MPS);
        return tc_mps_gemm(ctx, desc, A, B, C);
    }

    id<MTLComputePipelineState> pso = resolve_pipeline(ctx, tile.kernel_name,
                                                        desc->transpose_a,
                                                        desc->transpose_b,
                                                        &err);
    if (!pso) {
        tc_set_last_backend(TC_BACKEND_MPS);
        return tc_mps_gemm(ctx, desc, A, B, C);
    }

    const uint32_t M = (uint32_t)desc->M;
    const uint32_t N = (uint32_t)desc->N;
    const uint32_t K = (uint32_t)desc->K;
    const uint32_t lda = (uint32_t)effective_lda(desc);
    const uint32_t ldb = (uint32_t)effective_ldb(desc);
    const uint32_t ldc = (uint32_t)effective_ldc(desc);
    const float alpha = desc->alpha;
    const float beta  = desc->beta;

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
        [enc setBytes:&lda   length:sizeof(lda)   atIndex:8];
        [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:9];
        [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:10];

        const uint32_t groups_x = (N + tile.BN - 1) / tile.BN;
        const uint32_t groups_y = (M + tile.BM - 1) / tile.BM;
        [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(tile.threads_per_tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error) {
            fprintf(stderr, "[tensorcore] gemm dispatch error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            tc_set_last_backend(TC_BACKEND_NONE);
            return TC_ERR_DISPATCH;
        }
    }

    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
}

extern "C" tc_status_t tc_gemm_async(tc_context* ctx,
                                     const tc_gemm_desc* desc,
                                     const tc_buffer* A,
                                     const tc_buffer* B,
                                     tc_buffer*       C,
                                     tc_stream*       stream) {
    /* Same body as tc_gemm but does not wait. */
    if (!ctx)                  return TC_ERR_NOT_INITIALIZED;
    if (!validate(desc))       return TC_ERR_INVALID_ARG;
    if (!A || !B || !C)        return TC_ERR_INVALID_ARG;
    tc_status_t s = validate_gemm_buffers(ctx, desc, A, B, C);
    if (s != TC_OK) return s;

    tc_status_t err = TC_OK;
    auto fallback_gemm = [&]() -> tc_status_t {
        if (stream) {
            tc_status_t ss = tc_stream_sync(stream);
            if (ss != TC_OK) return ss;
        }
        return tc_mps_gemm(ctx, desc, A, B, C);
    };
    TileChoice tile = kernel_for(desc, ctx->info.family, ctx, &err);
    if (!tile.kernel_name) return fallback_gemm();

    id<MTLComputePipelineState> pso = resolve_pipeline(ctx, tile.kernel_name,
                                                        desc->transpose_a,
                                                        desc->transpose_b,
                                                        &err);
    if (!pso) return fallback_gemm();

    const uint32_t M = (uint32_t)desc->M;
    const uint32_t N = (uint32_t)desc->N;
    const uint32_t K = (uint32_t)desc->K;
    const uint32_t lda = (uint32_t)effective_lda(desc);
    const uint32_t ldb = (uint32_t)effective_ldb(desc);
    const uint32_t ldc = (uint32_t)effective_ldc(desc);
    const float alpha = desc->alpha;
    const float beta  = desc->beta;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = stream ? tc_stream_command_buffer(stream)
                                          : [ctx->queue commandBuffer];
        if (!cmd) return TC_ERR_INTERNAL;
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
        [enc setBytes:&lda   length:sizeof(lda)   atIndex:8];
        [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:9];
        [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:10];

        const uint32_t groups_x = (N + tile.BN - 1) / tile.BN;
        const uint32_t groups_y = (M + tile.BM - 1) / tile.BM;
        [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
            threadsPerThreadgroup:MTLSizeMake(tile.threads_per_tg, 1, 1)];
        [enc endEncoding];
        if (!stream) [cmd commit];
    }
    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
}

extern "C" tc_status_t tc_gemm_batched(tc_context* ctx,
                                       const tc_gemm_batched_desc* bd,
                                       const tc_buffer* A,
                                       const tc_buffer* B,
                                       tc_buffer*       C) {
    if (!ctx || !bd || !A || !B || !C) return TC_ERR_INVALID_ARG;
    const tc_gemm_desc& d = bd->base;
    if (bd->batch <= 0) return TC_ERR_INVALID_ARG;
    if (!validate(&d)) return TC_ERR_INVALID_ARG;

    /* Batched fast path: fp16 in/out + fp32 accum. Transposes are specialized
     * through the same function constants as single GEMM. */
    const bool fast_path =
        (d.a_dtype == TC_DTYPE_F16 && d.b_dtype == TC_DTYPE_F16 &&
         d.c_dtype == TC_DTYPE_F16 && d.accum_dtype == TC_DTYPE_F32);

    if (!fast_path) {
        if (bd->batch != 1) return TC_ERR_INVALID_SHAPE;
        return tc_gemm(ctx, &d, A, B, C);
    }

    if (bd->batch > 1 && (bd->stride_a <= 0 || bd->stride_b <= 0 || bd->stride_c <= 0)) {
        return TC_ERR_INVALID_SHAPE;
    }

    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    const int32_t a_rows = d.transpose_a ? d.K : d.M;
    const int32_t a_cols = d.transpose_a ? d.M : d.K;
    const int32_t b_rows = d.transpose_b ? d.N : d.K;
    const int32_t b_cols = d.transpose_b ? d.K : d.N;
    if (!batched_matrix_bytes(a_rows, a_cols, effective_lda(&d),
                              d.a_dtype, bd->batch, bd->stride_a, &a_bytes) ||
        !batched_matrix_bytes(b_rows, b_cols, effective_ldb(&d),
                              d.b_dtype, bd->batch, bd->stride_b, &b_bytes) ||
        !batched_matrix_bytes(d.M, d.N, effective_ldc(&d),
                              d.c_dtype, bd->batch, bd->stride_c, &c_bytes)) {
        return TC_ERR_INVALID_SHAPE;
    }
    tc_status_t s = tc_buffer_validate(ctx, A, a_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, B, b_bytes);
    if (s != TC_OK) return s;
    s = tc_buffer_validate(ctx, C, c_bytes);
    if (s != TC_OK) return s;

    tc_status_t err = TC_OK;
    id<MTLComputePipelineState> pso = resolve_pipeline(ctx, @"tc_gemm_f16_f32_batched",
                                                        d.transpose_a, d.transpose_b, &err);
    if (!pso) return err;

    const uint32_t M = (uint32_t)d.M;
    const uint32_t N = (uint32_t)d.N;
    const uint32_t K = (uint32_t)d.K;
    const float alpha = d.alpha;
    const float beta  = d.beta;
    const uint64_t sa = (uint64_t)bd->stride_a;
    const uint64_t sb = (uint64_t)bd->stride_b;
    const uint64_t sc = (uint64_t)bd->stride_c;
    const uint32_t lda = (uint32_t)effective_lda(&d);
    const uint32_t ldb = (uint32_t)effective_ldb(&d);
    const uint32_t ldc = (uint32_t)effective_ldc(&d);

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
        [enc setBytes:&sa    length:sizeof(sa)    atIndex:8];
        [enc setBytes:&sb    length:sizeof(sb)    atIndex:9];
        [enc setBytes:&sc    length:sizeof(sc)    atIndex:10];
        [enc setBytes:&lda   length:sizeof(lda)   atIndex:11];
        [enc setBytes:&ldb   length:sizeof(ldb)   atIndex:12];
        [enc setBytes:&ldc   length:sizeof(ldc)   atIndex:13];
        const uint32_t gx = (N + 64 - 1) / 64;
        const uint32_t gy = (M + 64 - 1) / 64;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, (NSUInteger)bd->batch)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) return TC_ERR_DISPATCH;
    }
    tc_set_last_backend(TC_BACKEND_SIMDGROUP_MATRIX);
    return TC_OK;
}
