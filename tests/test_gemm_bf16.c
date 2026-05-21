/*
 * Correctness test: tc_gemm in bf16 vs cblas_sgemm reference (cast).
 *
 * Uses simdgroup_matrix<bfloat,8,8> on Apple9+ (M3/A17 Pro). Older silicon
 * routes through the MPS software fallback and should still match reference.
 */

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

extern tc_status_t tc_mps_gemm(tc_context* ctx,
                               const tc_gemm_desc* desc,
                               const tc_buffer* A,
                               const tc_buffer* B,
                               tc_buffer* C);

static uint16_t f32_to_bf16(float x) {
    union { float f; uint32_t u; } v = {x};
    /* Round-to-nearest-even of the high half. */
    uint32_t r = v.u + 0x7FFF + ((v.u >> 16) & 1);
    return (uint16_t)(r >> 16);
}
static float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v = { ((uint32_t)b) << 16 };
    return v.f;
}

static int run_mps_bf16_fallback_smoke(tc_context* ctx) {
    enum { M = 3, N = 4, K = 5 };
    const size_t ba = (size_t)M * K * sizeof(uint16_t);
    const size_t bb = (size_t)K * N * sizeof(uint16_t);
    const size_t bc = (size_t)M * N * sizeof(uint16_t);
    const float A_src[M * K] = {
         1.0f, -0.5f,  0.25f,  2.0f, -1.5f,
         0.75f, 1.25f, -2.0f,  0.5f,  1.5f,
        -1.0f,  2.5f,  1.0f, -0.75f, 0.25f,
    };
    const float B_src[K * N] = {
         0.5f,  -1.0f,  1.5f,  0.25f,
        -0.75f,  2.0f, -0.5f,  1.0f,
         1.25f,  0.5f,  0.75f, -1.5f,
        -2.0f,   1.0f,  0.25f,  0.5f,
         0.75f, -0.25f, 2.0f,  -1.0f,
    };

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    uint16_t *Ap = NULL, *Bp = NULL, *Cp = NULL;
    int rc = 6;

    if (tc_buffer_alloc(ctx, ba, &A) != TC_OK ||
        tc_buffer_alloc(ctx, bb, &B) != TC_OK ||
        tc_buffer_alloc(ctx, bc, &C) != TC_OK) {
        fprintf(stderr, "  mps bf16 fallback smoke: allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "  mps bf16 fallback smoke: map failed\n");
        goto cleanup;
    }
    for (int i = 0; i < M * K; ++i) Ap[i] = f32_to_bf16(A_src[i]);
    for (int i = 0; i < K * N; ++i) Bp[i] = f32_to_bf16(B_src[i]);
    memset(Cp, 0, bc);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_BF16;
    d.b_dtype = TC_DTYPE_BF16;
    d.c_dtype = TC_DTYPE_BF16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    tc_status_t s = tc_mps_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "  mps bf16 fallback smoke failed: %s\n", tc_status_string(s));
        goto cleanup;
    }

    double max_abs = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_f32(Ap[m * K + k]) * bf16_to_f32(Bp[k * N + n]);
            }
            const float want = bf16_to_f32(f32_to_bf16(sum));
            const double e = fabs((double)bf16_to_f32(Cp[m * N + n]) - (double)want);
            if (e > max_abs) max_abs = e;
        }
    }
    printf("  %-28s max_abs=%.3e\n", "mps_bf16_sw_fallback", max_abs);
    rc = (max_abs <= 1e-2) ? 0 : 6;

cleanup:
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
    return rc;
}

static int run_case(tc_context* ctx, int M, int N, int K) {
    const size_t ba = (size_t)M * K * sizeof(uint16_t);
    const size_t bb = (size_t)K * N * sizeof(uint16_t);
    const size_t bc = (size_t)M * N * sizeof(uint16_t);
    tc_buffer *A, *B, *C;
    tc_buffer_alloc(ctx, ba, &A);
    tc_buffer_alloc(ctx, bb, &B);
    tc_buffer_alloc(ctx, bc, &C);
    uint16_t *Ap, *Bp, *Cp;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    float* Af = malloc(M * K * sizeof(float));
    float* Bf = malloc(K * N * sizeof(float));
    float* Cr = calloc((size_t)M * N, sizeof(float));

    srand(0xB10C);
    for (int i = 0; i < M*K; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*2.0f; Af[i]=v; Ap[i]=f32_to_bf16(v); }
    for (int i = 0; i < K*N; ++i) { float v = ((float)rand()/RAND_MAX-0.5f)*2.0f; Bf[i]=v; Bp[i]=f32_to_bf16(v); }
    memset(Cp, 0, bc);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, Af, K, Bf, N, 0.0f, Cr, N);

    tc_gemm_desc d = {0};
    d.M=M; d.N=N; d.K=K;
    d.a_dtype=TC_DTYPE_BF16; d.b_dtype=TC_DTYPE_BF16;
    d.c_dtype=TC_DTYPE_BF16; d.accum_dtype=TC_DTYPE_F32;
    d.alpha=1.0f; d.beta=0.0f;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);

    if (s == TC_ERR_UNSUPPORTED_FAMILY) {
        printf("  M=%d N=%d K=%d   SKIPPED (bf16 simdgroup_matrix requires Apple9+/M3+)\n",
               M, N, K);
        free(Af); free(Bf); free(Cr);
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
        return 0;
    }

    double rms_err = 0.0, rms_ref = 0.0, max_abs = 0.0;
    for (int i = 0; i < M*N; ++i) {
        const float a = bf16_to_f32(Cp[i]);
        const double e = fabs((double)a - (double)Cr[i]);
        rms_err += e*e;
        rms_ref += (double)Cr[i]*Cr[i];
        if (e > max_abs) max_abs = e;
    }
    rms_err = sqrt(rms_err / (M*N));
    rms_ref = sqrt(rms_ref / (M*N));
    const double scaled = rms_err / (rms_ref + 1e-9);
    printf("  M=%d N=%d K=%d   backend=%-18s  max_abs=%.3e  scaled=%.3e  %s\n",
           M, N, K, tc_backend_name(tc_last_backend()), max_abs, scaled,
           (s == TC_OK) ? "OK" : tc_status_string(s));

    free(Af); free(Bf); free(Cr);
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    /* bf16 has ~3x worse mantissa than fp16 — looser threshold. */
    /* When MPS bf16 fallback ran through the SW fp32 path, accuracy is
     * essentially fp32-quantized-to-bf16 ≈ 4e-3 RMS. */
    return (s == TC_OK && scaled < 5e-2) ? 0 : 5;
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
    if (!info.supports_bf16_simdgroup) {
        printf("[note] device family=Apple%d lacks bf16 simdgroup_matrix; "
               "testing MPS fallback path instead\n", (int)info.family);
    } else {
        printf("[note] device family=Apple%d supports bf16 simdgroup_matrix\n",
               (int)info.family);
    }
    int rc = 0;
    rc |= run_mps_bf16_fallback_smoke(ctx);
    rc |= run_case(ctx, 64, 64, 64);
    rc |= run_case(ctx, 128, 128, 128);
    rc |= run_case(ctx, 256, 256, 256);
    rc |= run_case(ctx, 512, 512, 512);
    tc_shutdown(ctx);
    return rc;
}
