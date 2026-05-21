/*
 * Correctness test: tc_gemm in fp32 vs cblas_sgemm reference.
 *
 * Build: handled by tests/CMakeLists.txt. Run: bin/test_gemm_f32.
 *
 * The kernel hits the simdgroup_matrix fp32 path on Apple7+ and the MPS
 * fallback elsewhere — both should match Accelerate to within ~1e-3 relative.
 */

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tensorcore/tensorcore.h"

extern tc_status_t tc_accelerate_gemm_f32(const tc_gemm_desc* desc,
                                          const float* A,
                                          const float* B,
                                          float* C);
extern tc_status_t tc_mps_gemm(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer* C);

static int check_f32_close(const char* label,
                           const float* got,
                           const float* want,
                           int count,
                           double tol) {
    double max_abs = 0.0;
    for (int i = 0; i < count; ++i) {
        const double e = fabs((double)got[i] - (double)want[i]);
        if (e > max_abs) max_abs = e;
    }
    printf("  %-28s max_abs=%.3e\n", label, max_abs);
    return (max_abs <= tol) ? 0 : 6;
}

static int run_accelerate_fallback_smoke(void) {
    enum { M = 2, N = 3, K = 4 };
    const float A[M * K] = {
         1.0f, -2.0f,  0.5f,  3.0f,
        -1.5f,  2.5f, -0.25f, 0.75f,
    };
    const float B[K * N] = {
         0.25f, -1.0f,  2.0f,
         1.5f,   0.5f, -0.5f,
        -2.0f,   1.0f,  0.75f,
         0.5f,  -1.5f,  1.25f,
    };
    const float C0[M * N] = {
         0.5f, -0.25f, 1.0f,
        -1.0f,  0.75f, 0.25f,
    };
    float C[M * N];
    float Cref[M * N];
    memcpy(C, C0, sizeof(C));
    memcpy(Cref, C0, sizeof(Cref));

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 0.75f;
    d.beta = -0.5f;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[m * K + k] * B[k * N + n];
            Cref[m * N + n] = d.alpha * sum + d.beta * Cref[m * N + n];
        }
    }

    tc_status_t s = tc_accelerate_gemm_f32(&d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  accelerate fallback smoke failed: %s\n", tc_status_string(s));
        return 6;
    }
    return check_f32_close("accelerate_f32_fallback", C, Cref, M * N, 1e-5);
}

static int run_mps_f32_fallback_smoke(tc_context* ctx) {
    enum { M = 3, N = 2, K = 4 };
    const size_t bytes_a = (size_t)M * K * sizeof(float);
    const size_t bytes_b = (size_t)K * N * sizeof(float);
    const size_t bytes_c = (size_t)M * N * sizeof(float);
    const float A_src[M * K] = {
         1.0f, -0.5f,  2.0f,  0.25f,
        -1.5f,  1.25f, 0.0f, -2.0f,
         0.75f, 2.5f, -1.0f,  1.5f,
    };
    const float B_src[K * N] = {
         0.5f,  -1.0f,
        -2.0f,   0.25f,
         1.5f,   2.0f,
        -0.75f,  1.25f,
    };
    const float C_src[M * N] = {
         1.0f, -1.0f,
         0.5f,  0.25f,
        -0.5f,  2.0f,
    };

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    float Cref[M * N];
    int rc = 6;

    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) {
        fprintf(stderr, "  mps f32 fallback smoke: allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  mps f32 fallback smoke: map failed\n");
        goto cleanup;
    }
    memcpy(Ap, A_src, bytes_a);
    memcpy(Bp, B_src, bytes_b);
    memcpy(Cp, C_src, bytes_c);
    memcpy(Cref, C_src, sizeof(Cref));

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.25f;
    d.beta = -0.25f;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A_src[m * K + k] * B_src[k * N + n];
            Cref[m * N + n] = d.alpha * sum + d.beta * Cref[m * N + n];
        }
    }

    tc_status_t s = tc_mps_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  mps f32 fallback smoke failed: %s\n", tc_status_string(s));
        goto cleanup;
    }
    rc = check_f32_close("mps_f32_fallback", Cp, Cref, M * N, 1e-4);

cleanup:
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
    return rc;
}

static int run_case(tc_context* ctx, int M, int N, int K, int trans_a, int trans_b) {
    const size_t bytes_a = (size_t)M * K * sizeof(float);
    const size_t bytes_b = (size_t)K * N * sizeof(float);
    const size_t bytes_c = (size_t)M * N * sizeof(float);

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK) return 2;
    if (tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) return 3;

    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    /* Deterministic random fill */
    srand(0xC0DE);
    for (int i = 0; i < M * K; ++i) Ap[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; ++i) Bp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    memset(Cp, 0, bytes_c);

    /* Reference: Accelerate sgemm */
    float* Cref = (float*)calloc((size_t)M * N, sizeof(float));
    cblas_sgemm(CblasRowMajor,
                trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans,
                M, N, K,
                1.0f, Ap, trans_a ? M : K,
                      Bp, trans_b ? K : N,
                0.0f, Cref, N);

    /* tensorcore */
    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = !!trans_a;
    d.transpose_b = !!trans_b;
    d.alpha = 1.0f; d.beta = 0.0f;

    tc_status_t s = tc_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  tc_gemm failed: %s\n", tc_status_string(s));
        free(Cref);
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
        return 4;
    }

    /* Compare. fp32 simdgroup_matrix with fp32 accum should be ~bit-exact for
     * small sizes; up to ~1e-4 rel for any reordering. */
    double max_abs = 0.0, max_rel = 0.0, sum_sq = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const double a = Cp[i], r = Cref[i];
        const double e = fabs(a - r);
        const double re = e / (fabs(r) + 1e-9);
        if (e > max_abs) max_abs = e;
        if (re > max_rel) max_rel = re;
        sum_sq += e * e;
    }
    const double rmse = sqrt(sum_sq / (M * N));

    const char* backend = tc_backend_name(tc_last_backend());
    printf("  M=%d N=%d K=%d ta=%d tb=%d   backend=%-18s  max_abs=%.3e  max_rel=%.3e  rmse=%.3e\n",
           M, N, K, trans_a, trans_b, backend, max_abs, max_rel, rmse);

    free(Cref);
    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);

    /* Tolerance: 1e-3 relative for fp32 matmul of u[-1,1] uniform inputs. */
    return (max_rel < 1e-3) ? 0 : 5;
}

static int run_padded_transpose_case(tc_context* ctx) {
    enum { M = 37, N = 41, K = 29, LDA = 40, LDB = 34, LDC = 48 };
    const size_t elems_a = (size_t)(K - 1) * LDA + M;
    const size_t elems_b = (size_t)(N - 1) * LDB + K;
    const size_t elems_c = (size_t)(M - 1) * LDC + N;
    const size_t bytes_a = elems_a * sizeof(float);
    const size_t bytes_b = elems_b * sizeof(float);
    const size_t bytes_c = elems_c * sizeof(float);
    const float alpha = 0.75f;
    const float beta = -0.25f;

    float *Ahost = (float*)malloc(bytes_a);
    float *Bhost = (float*)malloc(bytes_b);
    float *Chost = (float*)malloc(bytes_c);
    float *Cref = (float*)malloc(bytes_c);
    tc_buffer *Abuf = NULL, *Bbuf = NULL, *Cbuf = NULL;
    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    int rc = 7;

    if (!Ahost || !Bhost || !Chost || !Cref) {
        fprintf(stderr, "  padded transpose case: host allocation failed\n");
        goto cleanup;
    }

    for (size_t i = 0; i < elems_a; ++i) Ahost[i] = -37.0f;
    for (size_t i = 0; i < elems_b; ++i) Bhost[i] = 19.0f;
    for (size_t i = 0; i < elems_c; ++i) Chost[i] = -11.0f;

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            Ahost[k * LDA + m] = (float)(((k * 17 + m * 13) % 31) - 15) * 0.03125f;
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            Bhost[n * LDB + k] = (float)(((n * 11 + k * 7) % 29) - 14) * 0.02734375f;
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            Chost[m * LDC + n] = (float)(((m * 5 + n * 3) % 23) - 11) * 0.015625f;
        }
    }
    memcpy(Cref, Chost, bytes_c);

    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasTrans,
                M, N, K,
                alpha, Ahost, LDA,
                       Bhost, LDB,
                beta, Cref, LDC);

    if (tc_buffer_alloc(ctx, bytes_a, &Abuf) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_b, &Bbuf) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_c, &Cbuf) != TC_OK) {
        fprintf(stderr, "  padded transpose case: device allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(Abuf, (void**)&Ap) != TC_OK ||
        tc_buffer_map(Bbuf, (void**)&Bp) != TC_OK ||
        tc_buffer_map(Cbuf, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  padded transpose case: map failed\n");
        goto cleanup;
    }
    memcpy(Ap, Ahost, bytes_a);
    memcpy(Bp, Bhost, bytes_b);
    memcpy(Cp, Chost, bytes_c);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32;
    d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = 1;
    d.transpose_b = 1;
    d.alpha = alpha;
    d.beta = beta;
    d.lda = LDA;
    d.ldb = LDB;
    d.ldc = LDC;

    tc_status_t s = tc_gemm(ctx, &d, Abuf, Bbuf, Cbuf);
    if (s != TC_OK) {
        fprintf(stderr, "  padded transpose tc_gemm failed: %s\n", tc_status_string(s));
        goto cleanup;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq = 0.0;
    int padding_ok = 1;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const double got = Cp[m * LDC + n];
            const double want = Cref[m * LDC + n];
            const double e = fabs(got - want);
            const double re = e / (fabs(want) + 1e-9);
            if (e > max_abs) max_abs = e;
            if (re > max_rel) max_rel = re;
            sum_sq += e * e;
        }
        if (m < M - 1) {
            for (int n = N; n < LDC; ++n) {
                if (Cp[m * LDC + n] != -11.0f) padding_ok = 0;
            }
        }
    }
    const double rmse = sqrt(sum_sq / (M * N));
    const char* backend = tc_backend_name(tc_last_backend());
    printf("  padded transpose f32     backend=%-18s  max_abs=%.3e  max_rel=%.3e  rmse=%.3e  padding=%s\n",
           backend, max_abs, max_rel, rmse, padding_ok ? "OK" : "FAIL");
    rc = (max_rel < 1e-3 && max_abs < 1e-3 && padding_ok) ? 0 : 7;

cleanup:
    if (Abuf) tc_buffer_free(ctx, Abuf);
    if (Bbuf) tc_buffer_free(ctx, Bbuf);
    if (Cbuf) tc_buffer_free(ctx, Cbuf);
    free(Ahost);
    free(Bhost);
    free(Chost);
    free(Cref);
    return rc;
}

static int run_batched_padded_transpose_case(tc_context* ctx) {
    enum { batch = 2, M = 29, N = 31, K = 27, LDA = 34, LDB = 32, LDC = 38 };
    const size_t elems_a = (size_t)(K - 1) * LDA + M;
    const size_t elems_b = (size_t)(N - 1) * LDB + K;
    const size_t elems_c = (size_t)(M - 1) * LDC + N;
    const int64_t stride_a = (int64_t)elems_a + 7;
    const int64_t stride_b = (int64_t)elems_b + 11;
    const int64_t stride_c = (int64_t)elems_c + 13;
    const size_t total_a = (size_t)(batch - 1) * (size_t)stride_a + elems_a;
    const size_t total_b = (size_t)(batch - 1) * (size_t)stride_b + elems_b;
    const size_t total_c = (size_t)(batch - 1) * (size_t)stride_c + elems_c;
    const size_t bytes_a = total_a * sizeof(float);
    const size_t bytes_b = total_b * sizeof(float);
    const size_t bytes_c = total_c * sizeof(float);
    const float alpha = 1.25f;
    const float beta = -0.5f;

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    float *Ahost = (float*)malloc(bytes_a);
    float *Bhost = (float*)malloc(bytes_b);
    float *Chost = (float*)malloc(bytes_c);
    float *Cref = (float*)malloc(bytes_c);
    int rc = 8;

    if (!Ahost || !Bhost || !Chost || !Cref) {
        fprintf(stderr, "  batched padded transpose f32: host allocation failed\n");
        goto cleanup;
    }

    for (size_t i = 0; i < total_a; ++i) Ahost[i] = -37.0f;
    for (size_t i = 0; i < total_b; ++i) Bhost[i] = 19.0f;
    for (size_t i = 0; i < total_c; ++i) Chost[i] = -11.0f;

    for (int b = 0; b < batch; ++b) {
        const size_t a0 = (size_t)b * (size_t)stride_a;
        const size_t b0 = (size_t)b * (size_t)stride_b;
        const size_t c0 = (size_t)b * (size_t)stride_c;
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < M; ++m) {
                Ahost[a0 + (size_t)k * LDA + m] =
                    (float)(((b * 19 + k * 17 + m * 13) % 31) - 15) * 0.03125f;
            }
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                Bhost[b0 + (size_t)n * LDB + k] =
                    (float)(((b * 23 + n * 11 + k * 7) % 29) - 14) * 0.02734375f;
            }
        }
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                Chost[c0 + (size_t)m * LDC + n] =
                    (float)(((b * 3 + m * 5 + n * 3) % 23) - 11) * 0.015625f;
            }
        }
    }
    memcpy(Cref, Chost, bytes_c);

    for (int b = 0; b < batch; ++b) {
        const size_t a0 = (size_t)b * (size_t)stride_a;
        const size_t b0 = (size_t)b * (size_t)stride_b;
        const size_t c0 = (size_t)b * (size_t)stride_c;
        cblas_sgemm(CblasRowMajor,
                    CblasTrans,
                    CblasTrans,
                    M, N, K,
                    alpha, Ahost + a0, LDA,
                           Bhost + b0, LDB,
                    beta, Cref + c0, LDC);
    }

    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) {
        fprintf(stderr, "  batched padded transpose f32: device allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  batched padded transpose f32: map failed\n");
        goto cleanup;
    }
    memcpy(Ap, Ahost, bytes_a);
    memcpy(Bp, Bhost, bytes_b);
    memcpy(Cp, Chost, bytes_c);

    tc_gemm_batched_desc bd = {0};
    bd.base.M = M; bd.base.N = N; bd.base.K = K;
    bd.base.a_dtype = TC_DTYPE_F32;
    bd.base.b_dtype = TC_DTYPE_F32;
    bd.base.c_dtype = TC_DTYPE_F32;
    bd.base.accum_dtype = TC_DTYPE_F32;
    bd.base.transpose_a = 1;
    bd.base.transpose_b = 1;
    bd.base.alpha = alpha;
    bd.base.beta = beta;
    bd.base.lda = LDA;
    bd.base.ldb = LDB;
    bd.base.ldc = LDC;
    bd.batch = batch;
    bd.stride_a = stride_a;
    bd.stride_b = stride_b;
    bd.stride_c = stride_c;

    tc_status_t s = tc_gemm_batched(ctx, &bd, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  batched padded transpose f32 failed: %s\n", tc_status_string(s));
        goto cleanup;
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq = 0.0;
    int padding_ok = 1;
    for (int b = 0; b < batch; ++b) {
        const size_t c0 = (size_t)b * (size_t)stride_c;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const size_t idx = c0 + (size_t)m * LDC + n;
                const double got = Cp[idx];
                const double want = Cref[idx];
                const double e = fabs(got - want);
                const double re = e / (fabs(want) + 1e-9);
                if (e > max_abs) max_abs = e;
                if (re > max_rel) max_rel = re;
                sum_sq += e * e;
            }
            if (m < M - 1) {
                for (int n = N; n < LDC; ++n) {
                    if (Cp[c0 + (size_t)m * LDC + n] != -11.0f) padding_ok = 0;
                }
            }
        }
    }
    const double rmse = sqrt(sum_sq / ((double)batch * M * N));
    const char* backend = tc_backend_name(tc_last_backend());
    printf("  batched padded transpose f32 backend=%-18s  max_abs=%.3e  max_rel=%.3e  rmse=%.3e  padding=%s\n",
           backend, max_abs, max_rel, rmse, padding_ok ? "OK" : "FAIL");
    rc = (max_rel < 1e-3 && max_abs < 1e-3 && padding_ok) ? 0 : 8;

cleanup:
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
    free(Ahost);
    free(Bhost);
    free(Chost);
    free(Cref);
    return rc;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    tc_device_info info;
    tc_device_info_get(ctx, &info);
    printf("device=%s  family=Apple%d  unified=%s\n",
           info.name, (int)info.family, info.unified_memory ? "yes" : "no");

    int rc = 0;
    rc |= run_accelerate_fallback_smoke();
    rc |= run_mps_f32_fallback_smoke(ctx);
    /* Aligned shapes (no boundary path) */
    rc |= run_case(ctx,  64,  64,  64, 0, 0);
    rc |= run_case(ctx, 128, 128, 128, 0, 0);
    rc |= run_case(ctx, 256, 256, 256, 0, 0);
    rc |= run_case(ctx, 512, 512, 512, 0, 0);
    /* Non-aligned (exercises bounds-check slow path) */
    rc |= run_case(ctx,  65,  63,  47, 0, 0);
    rc |= run_case(ctx, 100, 200, 300, 0, 0);
    rc |= run_padded_transpose_case(ctx);
    rc |= run_batched_padded_transpose_case(ctx);

    tc_shutdown(ctx);
    return rc;
}
