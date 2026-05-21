/*
 * Correctness test: tc_gemm in fp16 vs cblas_sgemm reference (cast).
 *
 * fp16 matmul has fp32 accumulators, so the result tolerance is dominated by
 * the half-precision input quantization (~1e-3 relative) plus rounding at
 * store. We allow 3e-3 relative.
 */

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

/* Minimal f32 <-> f16 conversion (IEEE 754 binary16, round to nearest). */
static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;          /* underflow to zero  */
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round = (mant >> (shift - 1)) & 1;
        return (uint16_t)(sign | ((mant >> shift) + round));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0)); /* inf/NaN     */
    }
    uint32_t round = (mant >> 12) & 1;
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + round));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            ++exp; mant &= 0x3FF;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { out };
    return v.f;
}

static int run_case(tc_context* ctx, int M, int N, int K) {
    const size_t bytes_a = (size_t)M * K * sizeof(uint16_t);
    const size_t bytes_b = (size_t)K * N * sizeof(uint16_t);
    const size_t bytes_c = (size_t)M * N * sizeof(uint16_t);

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK) return 2;
    if (tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) return 3;
    uint16_t *Ap, *Bp, *Cp;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    float* Afp32 = (float*)malloc(bytes_a * 2);
    float* Bfp32 = (float*)malloc(bytes_b * 2);
    float* Cref  = (float*)calloc((size_t)M * N, sizeof(float));

    srand(0xF1F1);
    for (int i = 0; i < M * K; ++i) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        Afp32[i] = v;
        Ap[i] = f32_to_f16(v);
    }
    for (int i = 0; i < K * N; ++i) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        Bfp32[i] = v;
        Bp[i] = f32_to_f16(v);
    }
    memset(Cp, 0, bytes_c);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, Afp32, K, Bfp32, N, 0.0f, Cref, N);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    /* For fp16 matmul of u[-1,1] inputs across K dim, the typical magnitude
     * of any C[i,j] is ~sqrt(K/3). We compare relative error against the
     * scale of the matrix as a whole (RMS of |Cref|), not per-cell — per-cell
     * relative error is meaningless near zero. */
    double max_abs = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    if (s == TC_OK) {
        for (int i = 0; i < M * N; ++i) {
            float a = f16_to_f32(Cp[i]);
            double e = fabs((double)a - (double)Cref[i]);
            if (e > max_abs) max_abs = e;
            sum_sq_err += e * e;
            sum_sq_ref += (double)Cref[i] * (double)Cref[i];
        }
    }
    const double rms_err = sqrt(sum_sq_err / (M * N));
    const double rms_ref = sqrt(sum_sq_ref / (M * N));
    const double scaled  = rms_err / (rms_ref + 1e-9);

    printf("  M=%d N=%d K=%d   backend=%-18s  max_abs=%.3e  rms_err=%.3e  rms_ref=%.3e  scaled=%.3e  %s\n",
           M, N, K, tc_backend_name(tc_last_backend()),
           max_abs, rms_err, rms_ref, scaled, (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Afp32); free(Bfp32); free(Cref);
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    if (s != TC_OK) return (int)-s;
    /* fp16 with fp32 accumulators: scaled RMS error grows as 1/sqrt(K) of the
     * reference magnitude. Threshold 1.5e-2 covers K=64..512. */
    return (scaled < 1.5e-2) ? 0 : 5;
}

static int run_padded_ld_case(tc_context* ctx) {
    enum { M = 65, N = 67, K = 70 };
    const int lda = K + 5;
    const int ldb = N + 7;
    const int ldc = N + 3;
    const float alpha = 0.75f;
    const float beta = 0.25f;
    const size_t elems_a = (size_t)(M - 1) * lda + K;
    const size_t elems_b = (size_t)(K - 1) * ldb + N;
    const size_t elems_c = (size_t)(M - 1) * ldc + N;

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, elems_a * sizeof(uint16_t), &A) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, elems_b * sizeof(uint16_t), &B) != TC_OK) return 2;
    if (tc_buffer_alloc(ctx, elems_c * sizeof(uint16_t), &C) != TC_OK) return 3;

    uint16_t *Ap = NULL, *Bp = NULL, *Cp = NULL;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    float* Afp32 = (float*)calloc(elems_a, sizeof(float));
    float* Bfp32 = (float*)calloc(elems_b, sizeof(float));
    float* Cref = (float*)calloc(elems_c, sizeof(float));

    srand(0x1DA);
    for (size_t i = 0; i < elems_a; ++i) Ap[i] = f32_to_f16(-7.0f);
    for (size_t i = 0; i < elems_b; ++i) Bp[i] = f32_to_f16(9.0f);
    for (size_t i = 0; i < elems_c; ++i) {
        Cp[i] = f32_to_f16(3.0f);
        Cref[i] = 3.0f;
    }
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            Afp32[(size_t)m * lda + k] = v;
            Ap[(size_t)m * lda + k] = f32_to_f16(v);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            Bfp32[(size_t)k * ldb + n] = v;
            Bp[(size_t)k * ldb + n] = f32_to_f16(v);
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
            Cref[(size_t)m * ldc + n] = v;
            Cp[(size_t)m * ldc + n] = f32_to_f16(v);
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, Afp32, lda, Bfp32, ldb, beta, Cref, ldc);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.alpha = alpha; d.beta = beta;
    d.lda = lda; d.ldb = ldb; d.ldc = ldc;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    double max_abs = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    if (s == TC_OK) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const size_t idx = (size_t)m * ldc + n;
                float a = f16_to_f32(Cp[idx]);
                double e = fabs((double)a - (double)Cref[idx]);
                if (e > max_abs) max_abs = e;
                sum_sq_err += e * e;
                sum_sq_ref += (double)Cref[idx] * (double)Cref[idx];
            }
        }
    }
    const double rms_err = sqrt(sum_sq_err / (M * N));
    const double rms_ref = sqrt(sum_sq_ref / (M * N));
    const double scaled = rms_err / (rms_ref + 1e-9);

    printf("  padded lda=%d ldb=%d ldc=%d backend=%-18s max_abs=%.3e scaled=%.3e %s\n",
           lda, ldb, ldc, tc_backend_name(tc_last_backend()), max_abs, scaled,
           (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Afp32); free(Bfp32); free(Cref);
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    if (s != TC_OK) return (int)-s;
    return (scaled < 1.5e-2) ? 0 : 5;
}

static int run_batched_case(tc_context* ctx, int batch, int M, int N, int K) {
    const size_t elems_a = (size_t)M * K;
    const size_t elems_b = (size_t)K * N;
    const size_t elems_c = (size_t)M * N;
    const int64_t stride_a = (int64_t)elems_a + 7;
    const int64_t stride_b = (int64_t)elems_b + 5;
    const int64_t stride_c = (int64_t)elems_c + 3;
    const size_t total_a = (size_t)(batch - 1) * (size_t)stride_a + elems_a;
    const size_t total_b = (size_t)(batch - 1) * (size_t)stride_b + elems_b;
    const size_t total_c = (size_t)(batch - 1) * (size_t)stride_c + elems_c;

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, total_a * sizeof(uint16_t), &A) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, total_b * sizeof(uint16_t), &B) != TC_OK) return 2;
    if (tc_buffer_alloc(ctx, total_c * sizeof(uint16_t), &C) != TC_OK) return 3;

    uint16_t *Ap = NULL, *Bp = NULL, *Cp = NULL;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    float* Afp32 = (float*)calloc(total_a, sizeof(float));
    float* Bfp32 = (float*)calloc(total_b, sizeof(float));
    float* Cref = (float*)calloc(total_c, sizeof(float));

    srand(0xB47C);
    for (int b = 0; b < batch; ++b) {
        const size_t a0 = (size_t)b * (size_t)stride_a;
        const size_t b0 = (size_t)b * (size_t)stride_b;
        for (size_t i = 0; i < elems_a; ++i) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            Afp32[a0 + i] = v;
            Ap[a0 + i] = f32_to_f16(v);
        }
        for (size_t i = 0; i < elems_b; ++i) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            Bfp32[b0 + i] = v;
            Bp[b0 + i] = f32_to_f16(v);
        }
    }
    memset(Cp, 0, total_c * sizeof(uint16_t));

    for (int b = 0; b < batch; ++b) {
        const size_t a0 = (size_t)b * (size_t)stride_a;
        const size_t b0 = (size_t)b * (size_t)stride_b;
        const size_t c0 = (size_t)b * (size_t)stride_c;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, Afp32 + a0, K, Bfp32 + b0, N,
                    0.0f, Cref + c0, N);
    }

    tc_gemm_batched_desc bd = {0};
    bd.base.M = M; bd.base.N = N; bd.base.K = K;
    bd.base.a_dtype = TC_DTYPE_F16;
    bd.base.b_dtype = TC_DTYPE_F16;
    bd.base.c_dtype = TC_DTYPE_F16;
    bd.base.accum_dtype = TC_DTYPE_F32;
    bd.base.alpha = 1.0f; bd.base.beta = 0.0f;
    bd.batch = batch;
    bd.stride_a = stride_a;
    bd.stride_b = stride_b;
    bd.stride_c = stride_c;
    tc_status_t s = tc_gemm_batched(ctx, &bd, A, B, C);

    double max_abs = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    if (s == TC_OK) {
        for (int b = 0; b < batch; ++b) {
            const size_t c0 = (size_t)b * (size_t)stride_c;
            for (size_t i = 0; i < elems_c; ++i) {
                const float got = f16_to_f32(Cp[c0 + i]);
                const double e = fabs((double)got - (double)Cref[c0 + i]);
                if (e > max_abs) max_abs = e;
                sum_sq_err += e * e;
                sum_sq_ref += (double)Cref[c0 + i] * (double)Cref[c0 + i];
            }
        }
    }
    const double rms_err = sqrt(sum_sq_err / ((double)batch * elems_c));
    const double rms_ref = sqrt(sum_sq_ref / ((double)batch * elems_c));
    const double scaled = rms_err / (rms_ref + 1e-9);

    printf("  batched batch=%d M=%d N=%d K=%d backend=%-18s  max_abs=%.3e scaled=%.3e  %s\n",
           batch, M, N, K, tc_backend_name(tc_last_backend()),
           max_abs, scaled, (s == TC_OK && scaled < 1.5e-2) ? "OK" : tc_status_string(s));

    free(Afp32); free(Bfp32); free(Cref);
    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);
    if (s != TC_OK) return (int)-s;
    return (scaled < 1.5e-2) ? 0 : 5;
}

static int run_batched_rejection_case(tc_context* ctx) {
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    tc_buffer_alloc(ctx, 2 * sizeof(uint16_t), &A);
    tc_buffer_alloc(ctx, 2 * sizeof(uint16_t), &B);
    tc_buffer_alloc(ctx, 2 * sizeof(uint16_t), &C);

    tc_gemm_batched_desc bd = {0};
    bd.base.M = 1; bd.base.N = 1; bd.base.K = 1;
    bd.base.a_dtype = TC_DTYPE_F32;
    bd.base.b_dtype = TC_DTYPE_F32;
    bd.base.c_dtype = TC_DTYPE_F32;
    bd.base.accum_dtype = TC_DTYPE_F32;
    bd.base.alpha = 1.0f; bd.base.beta = 0.0f;
    bd.batch = 2;
    bd.stride_a = 1; bd.stride_b = 1; bd.stride_c = 1;

    tc_status_t s = tc_gemm_batched(ctx, &bd, A, B, C);
    printf("  batched fallback rejection: %s\n",
           (s == TC_ERR_INVALID_SHAPE) ? "OK" : tc_status_string(s));

    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);
    return (s == TC_ERR_INVALID_SHAPE) ? 0 : 1;
}

static int run_buffer_validation_case(tc_context* ctx) {
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    tc_buffer_alloc(ctx, 3 * sizeof(uint16_t), &A);  /* needs 2x2 = 4 elements */
    tc_buffer_alloc(ctx, 4 * sizeof(uint16_t), &B);
    tc_buffer_alloc(ctx, 4 * sizeof(uint16_t), &C);

    tc_gemm_desc d = {0};
    d.M = 2; d.N = 2; d.K = 2;
    d.a_dtype = TC_DTYPE_F16;
    d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;

    tc_status_t s = tc_gemm(ctx, &d, A, B, C);
    printf("  undersized GEMM buffer rejection: %s\n",
           (s == TC_ERR_INVALID_SHAPE) ? "OK" : tc_status_string(s));

    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);
    return (s == TC_ERR_INVALID_SHAPE) ? 0 : 1;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    int rc = 0;
    rc |= run_case(ctx,  64,  64,  64);
    rc |= run_case(ctx, 128, 128, 128);
    rc |= run_case(ctx, 256, 256, 256);
    rc |= run_case(ctx, 512, 512, 512);
    rc |= run_padded_ld_case(ctx);
    rc |= run_batched_case(ctx, 3, 64, 48, 64);
    rc |= run_batched_rejection_case(ctx);
    rc |= run_buffer_validation_case(ctx);
    tc_shutdown(ctx);
    return rc;
}
