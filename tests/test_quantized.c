/*
 * tensorcore - quantized GEMV correctness vs CPU dequant + fp16 GEMV.
 *
 * Generates random fp16 weights, quantizes to Q4_0/Q8_0 via GPU kernels, computes
 * GEMV via tc_gemv_quantized, compares against CPU references that dequantize
 * the same blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) { if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000; uint32_t sh = (uint32_t)(14 - exp);
        return (uint16_t)(sign | ((mant >> sh) + ((mant >> (sh-1)) & 1)));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | ((mant >> 13) + ((mant >> 12) & 1)));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0 && mant == 0) { union {uint32_t u; float f;} v = {sign}; return v.f; }
    if (exp == 31) { union {uint32_t u; float f;} v = {sign | 0x7F800000}; return v.f; }
    if (exp == 0) { while ((mant & 0x400) == 0) { mant <<= 1; --exp; } ++exp; mant &= 0x3FF; }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

static int backend_is_compute(const char* op) {
    const tc_backend_t b = tc_last_backend();
    if (b == TC_BACKEND_METAL_COMPUTE || b == TC_BACKEND_PORTABLE_CPU) return 1;
    fprintf(stderr, "%s backend was %s, expected metal_compute or portable_cpu\n",
            op, tc_backend_name(b));
    return 0;
}

/* CPU dequant + GEMV reference using the EXACT same Q4_0 blocks the kernel
 * produced (so we're testing kernel-to-kernel consistency, not the quantization
 * algorithm itself). */
static void ref_q4_0_gemv(int M, int N, int K, const uint16_t* X, const uint8_t* Wq,
                          float* Y) {
    const int nblocks = K / 32;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            const uint8_t* W_row = Wq + (size_t)n * nblocks * 18;
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t* block = W_row + (size_t)b * 18;
                const float scale = f16_to_f32(*(const uint16_t*)block);
                for (int i = 0; i < 32; ++i) {
                    const int k = b * 32 + i;
                    const uint8_t packed = block[2 + (i % 16)];
                    const int q = (i < 16) ? (packed & 0xF) : (packed >> 4);
                    const float w = scale * (float)(q - 8);
                    acc += (double)f16_to_f32(X[m * K + k]) * (double)w;
                }
            }
            Y[m * N + n] = (float)acc;
        }
    }
}

static void ref_q8_0_gemv(int M, int N, int K, const uint16_t* X, const uint8_t* Wq,
                          float* Y) {
    const int nblocks = K / 32;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            const uint8_t* W_row = Wq + (size_t)n * nblocks * 34;
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t* block = W_row + (size_t)b * 34;
                const float scale = f16_to_f32(*(const uint16_t*)block);
                const int8_t* qs = (const int8_t*)(block + 2);
                for (int i = 0; i < 32; ++i) {
                    const int k = b * 32 + i;
                    const float w = scale * (float)qs[i];
                    acc += (double)f16_to_f32(X[m * K + k]) * (double)w;
                }
            }
            Y[m * N + n] = (float)acc;
        }
    }
}

static int run_q4_case(tc_context* ctx, int M, int N, int K) {
    const size_t q_bytes = tc_quantized_size(TC_QUANT_Q4_0, N, K);

    tc_buffer *Xb = NULL, *Wfp16 = NULL, *Wq = NULL, *Yb = NULL;
    tc_stream* st = NULL;
    float* Yref = NULL;
    tc_status_t s = TC_OK;
    int rc = 5;

    if (tc_buffer_alloc(ctx, M*K*2, &Xb) != TC_OK ||
        tc_buffer_alloc(ctx, N*K*2, &Wfp16) != TC_OK ||
        tc_buffer_alloc(ctx, q_bytes, &Wq) != TC_OK ||
        tc_buffer_alloc(ctx, M*N*2, &Yb) != TC_OK) {
        fprintf(stderr, "buffer alloc failed\n");
        rc = 2;
        goto cleanup;
    }

    uint16_t *Xp, *Wp, *Yp; uint8_t *Wqp;
    if (tc_buffer_map(Xb,    (void**)&Xp)  != TC_OK ||
        tc_buffer_map(Wfp16, (void**)&Wp)  != TC_OK ||
        tc_buffer_map(Wq,    (void**)&Wqp) != TC_OK ||
        tc_buffer_map(Yb,    (void**)&Yp)  != TC_OK) {
        fprintf(stderr, "buffer map failed\n");
        rc = 2;
        goto cleanup;
    }

    srand(0xD4);
    for (int i = 0; i < M*K; ++i) Xp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f));
    for (int i = 0; i < N*K; ++i) Wp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    s = tc_quantize_weights(ctx, Wfp16, Wq, TC_QUANT_Q4_0, N, K);
    if (s != TC_OK) { fprintf(stderr, "quantize: %s\n", tc_status_string(s)); rc = 2; goto cleanup; }
    if (!backend_is_compute("quantize q4_0")) { rc = 2; goto cleanup; }

    s = tc_gemv_quantized(ctx, Xb, Wq, Yb, TC_QUANT_Q4_0, M, N, K);
    if (s != TC_OK) { fprintf(stderr, "gemv: %s\n", tc_status_string(s)); rc = 3; goto cleanup; }
    if (!backend_is_compute("gemv q4_0")) { rc = 3; goto cleanup; }

    Yref = malloc(M*N*sizeof(float));
    if (!Yref) { fprintf(stderr, "malloc failed\n"); rc = 2; goto cleanup; }
    ref_q4_0_gemv(M, N, K, Xp, Wqp, Yref);

    double se = 0, sr = 0, max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        float got = f16_to_f32(Yp[i]);
        double e = fabs((double)got - (double)Yref[i]);
        se += e * e; sr += (double)Yref[i] * Yref[i];
        if (e > max_abs) max_abs = e;
    }
    const double scaled = sqrt(se / (M*N)) / (sqrt(sr / (M*N)) + 1e-9);
    printf("  q4_0_gemv M=%d N=%d K=%d   max_abs=%.3e  rms_scaled=%.3e  %s\n",
           M, N, K, max_abs, scaled, (scaled < 5e-2) ? "OK" : "FAIL");

    /* Storage check: Q4_0 should be 18 bytes per 32-weight block. */
    const size_t expected = (size_t)N * (K / 32) * 18;
    printf("  q4_0 storage: %zu bytes (expected %zu) -> %.2f bits/weight  %s\n",
           q_bytes, expected,
           (double)q_bytes * 8.0 / (double)(N * K),
           (q_bytes == expected) ? "OK" : "FAIL");

    memset(Yp, 0, (size_t)M*N*sizeof(uint16_t));
    s = tc_stream_create(ctx, &st);
    if (s != TC_OK) { fprintf(stderr, "stream create: %s\n", tc_status_string(s)); rc = 4; goto cleanup; }
    s = tc_gemv_quantized_async(ctx, Xb, Wq, Yb, TC_QUANT_Q4_0, M, N, K, st);
    if (s == TC_OK) s = tc_stream_sync(st);
    if (s != TC_OK) { fprintf(stderr, "async gemv: %s\n", tc_status_string(s)); rc = 4; goto cleanup; }
    if (!backend_is_compute("async gemv q4_0")) { rc = 4; goto cleanup; }

    double ase = 0, asr = 0, amax_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        float got = f16_to_f32(Yp[i]);
        double e = fabs((double)got - (double)Yref[i]);
        ase += e * e; asr += (double)Yref[i] * Yref[i];
        if (e > amax_abs) amax_abs = e;
    }
    const double async_scaled = sqrt(ase / (M*N)) / (sqrt(asr / (M*N)) + 1e-9);
    printf("  q4_0_async M=%d N=%d K=%d max_abs=%.3e  rms_scaled=%.3e  %s\n",
           M, N, K, amax_abs, async_scaled, (async_scaled < 5e-2) ? "OK" : "FAIL");

    rc = (scaled < 5e-2 && async_scaled < 5e-2 && q_bytes == expected) ? 0 : 5;

cleanup:
    if (st) tc_stream_destroy(ctx, st);
    free(Yref);
    if (Xb) tc_buffer_free(ctx, Xb);
    if (Wfp16) tc_buffer_free(ctx, Wfp16);
    if (Wq) tc_buffer_free(ctx, Wq);
    if (Yb) tc_buffer_free(ctx, Yb);
    return rc;
}

static int run_q8_case(tc_context* ctx, int M, int N, int K) {
    const size_t q_bytes = tc_quantized_size(TC_QUANT_Q8_0, N, K);

    tc_buffer *Xb = NULL, *Wfp16 = NULL, *Wq = NULL, *Yb = NULL;
    float* Yref = NULL;
    tc_status_t s = TC_OK;
    int rc = 5;

    if (tc_buffer_alloc(ctx, M*K*2, &Xb) != TC_OK ||
        tc_buffer_alloc(ctx, N*K*2, &Wfp16) != TC_OK ||
        tc_buffer_alloc(ctx, q_bytes, &Wq) != TC_OK ||
        tc_buffer_alloc(ctx, M*N*2, &Yb) != TC_OK) {
        fprintf(stderr, "buffer alloc failed\n");
        rc = 2;
        goto cleanup;
    }

    uint16_t *Xp, *Wp, *Yp; uint8_t *Wqp;
    if (tc_buffer_map(Xb, (void**)&Xp) != TC_OK ||
        tc_buffer_map(Wfp16, (void**)&Wp) != TC_OK ||
        tc_buffer_map(Wq, (void**)&Wqp) != TC_OK ||
        tc_buffer_map(Yb, (void**)&Yp) != TC_OK) {
        fprintf(stderr, "buffer map failed\n");
        rc = 2;
        goto cleanup;
    }

    Yref = malloc((size_t)M*N*sizeof(float));
    if (!Yref) { fprintf(stderr, "malloc failed\n"); rc = 2; goto cleanup; }

    srand(0xE8);
    for (int i = 0; i < M*K; ++i) Xp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f));
    for (int i = 0; i < N*K; ++i) Wp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    s = tc_quantize_weights(ctx, Wfp16, Wq, TC_QUANT_Q8_0, N, K);
    if (s != TC_OK) { fprintf(stderr, "quantize q8_0: %s\n", tc_status_string(s)); rc = 2; goto cleanup; }
    if (!backend_is_compute("quantize q8_0")) { rc = 2; goto cleanup; }

    s = tc_gemv_quantized(ctx, Xb, Wq, Yb, TC_QUANT_Q8_0, M, N, K);
    if (s != TC_OK) { fprintf(stderr, "gemv q8_0: %s\n", tc_status_string(s)); rc = 3; goto cleanup; }
    if (!backend_is_compute("gemv q8_0")) { rc = 3; goto cleanup; }

    ref_q8_0_gemv(M, N, K, Xp, Wqp, Yref);

    double se = 0, sr = 0, max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        float got = f16_to_f32(Yp[i]);
        double e = fabs((double)got - (double)Yref[i]);
        se += e * e; sr += (double)Yref[i] * Yref[i];
        if (e > max_abs) max_abs = e;
    }
    const double scaled = sqrt(se / (M*N)) / (sqrt(sr / (M*N)) + 1e-9);
    printf("  q8_0_gemv M=%d N=%d K=%d   max_abs=%.3e  rms_scaled=%.3e  %s\n",
           M, N, K, max_abs, scaled, (scaled < 5e-2) ? "OK" : "FAIL");

    const size_t expected = (size_t)N * (K / 32) * 34;
    printf("  q8_0 storage: %zu bytes (expected %zu) -> %.2f bits/weight  %s\n",
           q_bytes, expected,
           (double)q_bytes * 8.0 / (double)(N * K),
           (q_bytes == expected) ? "OK" : "FAIL");

    rc = (scaled < 5e-2 && q_bytes == expected) ? 0 : 5;

cleanup:
    free(Yref);
    if (Xb) tc_buffer_free(ctx, Xb);
    if (Wfp16) tc_buffer_free(ctx, Wfp16);
    if (Wq) tc_buffer_free(ctx, Wq);
    if (Yb) tc_buffer_free(ctx, Yb);
    return rc;
}

static int run_fused_rmsnorm_quantized_case(tc_context* ctx, tc_quant_t fmt, int M, int N, int K) {
    const char* name = (fmt == TC_QUANT_Q4_0) ? "q4_0" : "q8_0";
    const size_t q_bytes = tc_quantized_size(fmt, N, K);
    const float eps = 1e-5f;

    tc_buffer *Xb = NULL, *gb = NULL, *Wfp16 = NULL, *Wq = NULL;
    tc_buffer *Yf = NULL, *Ys = NULL, *Xn = NULL, *rstd = NULL;
    tc_status_t s = TC_OK;
    int rc = 5;

    if (tc_buffer_alloc(ctx, M*K*2, &Xb) != TC_OK ||
        tc_buffer_alloc(ctx, K*2, &gb) != TC_OK ||
        tc_buffer_alloc(ctx, N*K*2, &Wfp16) != TC_OK ||
        tc_buffer_alloc(ctx, q_bytes, &Wq) != TC_OK ||
        tc_buffer_alloc(ctx, M*N*2, &Yf) != TC_OK ||
        tc_buffer_alloc(ctx, M*N*2, &Ys) != TC_OK ||
        tc_buffer_alloc(ctx, M*K*2, &Xn) != TC_OK ||
        tc_buffer_alloc(ctx, M*4, &rstd) != TC_OK) {
        fprintf(stderr, "buffer alloc failed\n");
        rc = 2;
        goto cleanup;
    }

    uint16_t *Xp, *gp, *Wp, *Yfp, *Ysp;
    if (tc_buffer_map(Xb, (void**)&Xp) != TC_OK ||
        tc_buffer_map(gb, (void**)&gp) != TC_OK ||
        tc_buffer_map(Wfp16, (void**)&Wp) != TC_OK ||
        tc_buffer_map(Yf, (void**)&Yfp) != TC_OK ||
        tc_buffer_map(Ys, (void**)&Ysp) != TC_OK) {
        fprintf(stderr, "buffer map failed\n");
        rc = 2;
        goto cleanup;
    }

    srand(fmt == TC_QUANT_Q4_0 ? 0x4F : 0x8F);
    for (int i = 0; i < M*K; ++i) Xp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f));
    for (int i = 0; i < K; ++i) gp[i] = f32_to_f16(0.5f + (float)rand()/RAND_MAX);
    for (int i = 0; i < N*K; ++i) Wp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    s = tc_quantize_weights(ctx, Wfp16, Wq, fmt, N, K);
    if (s != TC_OK) { fprintf(stderr, "quantize fused %s: %s\n", name, tc_status_string(s)); rc = 2; goto cleanup; }

    s = tc_fused_rmsnorm_gemv_quantized(ctx, Xb, gb, Wq, Yf, fmt, M, N, K, eps);
    if (s != TC_OK) { fprintf(stderr, "fused rmsnorm %s: %s\n", name, tc_status_string(s)); rc = 3; goto cleanup; }
    if (!backend_is_compute("fused rmsnorm quantized")) { rc = 3; goto cleanup; }

    s = tc_rmsnorm_forward(ctx, Xb, gb, Xn, rstd, M, K, eps);
    if (s != TC_OK) { fprintf(stderr, "rmsnorm separate %s: %s\n", name, tc_status_string(s)); rc = 4; goto cleanup; }
    s = tc_gemv_quantized(ctx, Xn, Wq, Ys, fmt, M, N, K);
    if (s != TC_OK) { fprintf(stderr, "gemv separate %s: %s\n", name, tc_status_string(s)); rc = 4; goto cleanup; }

    double se = 0, sr = 0, max_abs = 0;
    int nonfinite = 0;
    for (int i = 0; i < M*N; ++i) {
        float a = f16_to_f32(Yfp[i]);
        float b = f16_to_f32(Ysp[i]);
        if (!isfinite(a) || !isfinite(b)) {
            if (nonfinite < 4) {
                fprintf(stderr, "  nonfinite %s[%d]: fused=0x%04x separate=0x%04x\n",
                        name, i, Yfp[i], Ysp[i]);
            }
            nonfinite++;
        }
        double e = fabs((double)a - (double)b);
        se += e * e; sr += (double)b * b;
        if (e > max_abs) max_abs = e;
    }
    const double scaled = sqrt(se / (M*N)) / (sqrt(sr / (M*N)) + 1e-9);
    printf("  fused_rmsnorm_%s_gemv M=%d N=%d K=%d max_abs=%.3e  rms_scaled=%.3e  %s\n",
           name, M, N, K, max_abs, scaled, (scaled < 5e-3) ? "OK" : "FAIL");

    rc = (scaled < 5e-3) ? 0 : 5;

cleanup:
    if (Xb) tc_buffer_free(ctx, Xb);
    if (gb) tc_buffer_free(ctx, gb);
    if (Wfp16) tc_buffer_free(ctx, Wfp16);
    if (Wq) tc_buffer_free(ctx, Wq);
    if (Yf) tc_buffer_free(ctx, Yf);
    if (Ys) tc_buffer_free(ctx, Ys);
    if (Xn) tc_buffer_free(ctx, Xn);
    if (rstd) tc_buffer_free(ctx, rstd);
    return rc;
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    int rc = run_q4_case(ctx, 1, 128, 256);
    if (rc == 0) rc = run_q4_case(ctx, 1, 130, 256);
    if (rc == 0) rc = run_q8_case(ctx, 1, 129, 256);
    if (rc == 0) rc = run_fused_rmsnorm_quantized_case(ctx, TC_QUANT_Q4_0, 1, 97, 256);
    if (rc == 0) rc = run_fused_rmsnorm_quantized_case(ctx, TC_QUANT_Q8_0, 1, 96, 256);
    if (rc == 0) {
        const size_t invalid = tc_quantized_size((tc_quant_t)99, 128, 256);
        printf("  invalid quant size: %zu %s\n", invalid, (invalid == 0) ? "OK" : "FAIL");
        if (invalid != 0) rc = 5;
    }

    tc_shutdown(ctx);

    return rc;
}
