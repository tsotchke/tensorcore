/*
 * Correctness test: tc_gemm in int8 vs int32 reference.
 *
 * Uses simdgroup_matrix<char,8,8> on Apple10+ (M4). Older silicon routes
 * through the MPS software fallback and should still match int32 reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

extern tc_status_t tc_mps_gemm(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer* C);

static int run_mps_i8_fallback_smoke(tc_context* ctx) {
    enum { M = 3, N = 4, K = 5 };
    const size_t ba = (size_t)M * K;
    const size_t bb = (size_t)K * N;
    const size_t bc = (size_t)M * N * sizeof(int32_t);
    const int8_t A_src[M * K] = {
         3, -2,  5,  1, -4,
        -7,  6,  0,  2,  1,
         4,  3, -5, -1,  2,
    };
    const int8_t B_src[K * N] = {
         2, -3,  1,  4,
        -1,  5, -2,  0,
         3,  1,  2, -4,
        -5,  2,  3,  1,
         4, -1,  0,  2,
    };

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    int8_t *Ap = NULL, *Bp = NULL;
    int32_t *Cp = NULL;
    int rc = 6;

    if (tc_buffer_alloc(ctx, ba, &A) != TC_OK ||
        tc_buffer_alloc(ctx, bb, &B) != TC_OK ||
        tc_buffer_alloc(ctx, bc, &C) != TC_OK) {
        fprintf(stderr, "  mps i8 fallback smoke: allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  mps i8 fallback smoke: map failed\n");
        goto cleanup;
    }
    memcpy(Ap, A_src, ba);
    memcpy(Bp, B_src, bb);
    memset(Cp, 0, bc);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_I8;
    d.b_dtype = TC_DTYPE_I8;
    d.c_dtype = TC_DTYPE_I32;
    d.accum_dtype = TC_DTYPE_I32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    tc_status_t s = tc_mps_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  mps i8 fallback smoke failed: %s\n", tc_status_string(s));
        goto cleanup;
    }

    int errors = 0;
    int64_t max_abs = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t want = 0;
            for (int k = 0; k < K; ++k) {
                want += (int32_t)Ap[m * K + k] * (int32_t)Bp[k * N + n];
            }
            int64_t e = (int64_t)Cp[m * N + n] - (int64_t)want;
            if (e < 0) e = -e;
            if (e > max_abs) max_abs = e;
            if (e != 0) ++errors;
        }
    }
    printf("  %-28s errors=%d/%d  max_abs=%lld\n",
           "mps_i8_sw_fallback", errors, M * N, (long long)max_abs);
    rc = (errors == 0) ? 0 : 6;

cleanup:
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
    return rc;
}

static int run_case(tc_context* ctx, int M, int N, int K) {
    const size_t ba = (size_t)M * K;
    const size_t bb = (size_t)K * N;
    const size_t bc = (size_t)M * N * sizeof(int32_t);

    tc_buffer *A, *B, *C;
    tc_buffer_alloc(ctx, ba, &A);
    tc_buffer_alloc(ctx, bb, &B);
    tc_buffer_alloc(ctx, bc, &C);
    int8_t  *Ap, *Bp;
    int32_t *Cp;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    srand(0x1788);
    for (size_t i = 0; i < ba; ++i) Ap[i] = (int8_t)((rand() & 0x7F) - 64);
    for (size_t i = 0; i < bb; ++i) Bp[i] = (int8_t)((rand() & 0x7F) - 64);
    memset(Cp, 0, bc);

    /* Reference: int32 matmul. */
    int32_t* Cr = calloc((size_t)M * N, sizeof(int32_t));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int k = 0; k < K; ++k) s += (int32_t)Ap[m*K+k] * (int32_t)Bp[k*N+n];
            Cr[m*N+n] = s;
        }

    tc_gemm_desc d = {0};
    d.M=M; d.N=N; d.K=K;
    d.a_dtype=TC_DTYPE_I8; d.b_dtype=TC_DTYPE_I8;
    d.c_dtype=TC_DTYPE_I32; d.accum_dtype=TC_DTYPE_I32;
    d.alpha=1.0f; d.beta=0.0f;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    if (s == TC_ERR_UNSUPPORTED_FAMILY) {
        printf("  M=%d N=%d K=%d   SKIPPED (i8 simdgroup_matrix requires Apple10+/M4+)\n",
               M, N, K);
        free(Cr);
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
        return 0;
    }

    int errors = 0;
    int64_t max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        int64_t e = (int64_t)Cp[i] - (int64_t)Cr[i];
        if (e < 0) e = -e;
        if (e > max_abs) max_abs = e;
        if (e != 0) ++errors;
    }
    printf("  M=%d N=%d K=%d   backend=%-18s  errors=%d/%d  max_abs=%lld  %s\n",
           M, N, K, tc_backend_name(tc_last_backend()),
           errors, M*N, (long long)max_abs,
           (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Cr);
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    /* int8 matmul should be bit-exact (integer arithmetic). */
    return (s == TC_OK && errors == 0) ? 0 : 5;
}

static int run_padded_transpose_beta_case(tc_context* ctx) {
    enum { M = 35, N = 33, K = 29, LDA = 39, LDB = 34, LDC = 40 };
    const size_t elems_a = (size_t)(K - 1) * LDA + M;
    const size_t elems_b = (size_t)(N - 1) * LDB + K;
    const size_t elems_c = (size_t)(M - 1) * LDC + N;
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    int8_t *Ap = NULL, *Bp = NULL;
    int32_t *Cp = NULL, *Cr = NULL;
    int rc = 7;

    if (tc_buffer_alloc(ctx, elems_a, &A) != TC_OK ||
        tc_buffer_alloc(ctx, elems_b, &B) != TC_OK ||
        tc_buffer_alloc(ctx, elems_c * sizeof(int32_t), &C) != TC_OK) {
        fprintf(stderr, "  padded transpose i8: allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  padded transpose i8: map failed\n");
        goto cleanup;
    }
    Cr = (int32_t*)calloc(elems_c, sizeof(int32_t));
    if (!Cr) {
        fprintf(stderr, "  padded transpose i8: reference allocation failed\n");
        goto cleanup;
    }

    memset(Ap, 0x3d, elems_a);
    memset(Bp, 0x27, elems_b);
    for (size_t i = 0; i < elems_c; ++i) {
        Cp[i] = -1234567;
        Cr[i] = -1234567;
    }

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            Ap[(size_t)k * LDA + m] = (int8_t)(((k * 17 + m * 13) % 31) - 15);
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            Bp[(size_t)n * LDB + k] = (int8_t)(((n * 11 + k * 7) % 29) - 14);
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const int32_t c0 = (int32_t)(((m * 5 + n * 3) % 23) - 11);
            Cp[(size_t)m * LDC + n] = c0;
            Cr[(size_t)m * LDC + n] = c0;
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)Ap[(size_t)k * LDA + m] *
                       (int32_t)Bp[(size_t)n * LDB + k];
            }
            Cr[(size_t)m * LDC + n] += sum;
        }
    }

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_I8;
    d.b_dtype = TC_DTYPE_I8;
    d.c_dtype = TC_DTYPE_I32;
    d.accum_dtype = TC_DTYPE_I32;
    d.transpose_a = 1;
    d.transpose_b = 1;
    d.alpha = 1.0f;
    d.beta = 1.0f;
    d.lda = LDA;
    d.ldb = LDB;
    d.ldc = LDC;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    int errors = 0;
    int64_t max_abs = 0;
    int padding_ok = 1;
    if (s == TC_OK) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const size_t idx = (size_t)m * LDC + n;
                int64_t e = (int64_t)Cp[idx] - (int64_t)Cr[idx];
                if (e < 0) e = -e;
                if (e > max_abs) max_abs = e;
                if (e != 0) ++errors;
            }
            if (m < M - 1) {
                for (int n = N; n < LDC; ++n) {
                    if (Cp[(size_t)m * LDC + n] != -1234567) padding_ok = 0;
                }
            }
        }
    }

    printf("  padded transpose beta i8 backend=%-18s  errors=%d/%d  max_abs=%lld padding=%s  %s\n",
           tc_backend_name(tc_last_backend()), errors, M * N, (long long)max_abs,
           padding_ok ? "OK" : "FAIL",
           (s == TC_OK && errors == 0 && padding_ok) ? "OK" : tc_status_string(s));
    rc = (s == TC_OK && errors == 0 && padding_ok) ? 0 : 7;

cleanup:
    free(Cr);
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
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
    if (!info.supports_i8_simdgroup) {
        printf("[note] device family=Apple%d lacks i8 simdgroup_matrix; "
               "testing SW fallback (i8 -> fp32 -> i32)\n", (int)info.family);
    } else {
        printf("[note] device family=Apple%d supports i8 simdgroup_matrix\n",
               (int)info.family);
    }
    int rc = 0;
    rc |= run_mps_i8_fallback_smoke(ctx);
    rc |= run_case(ctx, 64, 64, 64);
    rc |= run_case(ctx, 128, 128, 128);
    rc |= run_case(ctx, 256, 256, 256);
    rc |= run_padded_transpose_beta_case(ctx);
    tc_shutdown(ctx);
    return rc;
}
