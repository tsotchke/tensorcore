/*
 * tensorcore — Q4_0 quantized GEMV correctness vs CPU dequant + fp16 GEMV.
 *
 * Generates random fp16 weights, quantizes to Q4_0 via GPU kernel, computes
 * GEMV via tc_gemv_quantized, compares against CPU reference that dequantizes
 * the same Q4_0 blocks. RMS-scaled error should be within Q4_0 quantization
 * noise (~3-5% on typical inputs — 4 bits is genuinely lossy).
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

/* CPU dequant + GEMV reference using the EXACT same Q4_0 blocks the kernel
 * produced (so we're testing kernel→kernel consistency, not the quantization
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
                    const uint8_t packed = block[2 + i/2];
                    const int q = (i & 1) ? (packed >> 4) : (packed & 0xF);
                    const float w = scale * (float)(q - 8);
                    acc += (double)f16_to_f32(X[m * K + k]) * (double)w;
                }
            }
            Y[m * N + n] = (float)acc;
        }
    }
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    const int M = 1, N = 128, K = 256;
    const size_t q_bytes = tc_quantized_size(TC_QUANT_Q4_0, N, K);

    tc_buffer *Xb, *Wfp16, *Wq, *Yb;
    tc_buffer_alloc(ctx, M*K*2, &Xb);
    tc_buffer_alloc(ctx, N*K*2, &Wfp16);
    tc_buffer_alloc(ctx, q_bytes, &Wq);
    tc_buffer_alloc(ctx, M*N*2, &Yb);

    uint16_t *Xp, *Wp, *Yp; uint8_t *Wqp;
    tc_buffer_map(Xb,    (void**)&Xp);
    tc_buffer_map(Wfp16, (void**)&Wp);
    tc_buffer_map(Wq,    (void**)&Wqp);
    tc_buffer_map(Yb,    (void**)&Yp);

    srand(0xD4);
    for (int i = 0; i < M*K; ++i) Xp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f));
    for (int i = 0; i < N*K; ++i) Wp[i] = f32_to_f16(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    s = tc_quantize_weights(ctx, Wfp16, Wq, TC_QUANT_Q4_0, N, K);
    if (s != TC_OK) { fprintf(stderr, "quantize: %s\n", tc_status_string(s)); return 2; }

    s = tc_gemv_quantized(ctx, Xb, Wq, Yb, TC_QUANT_Q4_0, M, N, K);
    if (s != TC_OK) { fprintf(stderr, "gemv: %s\n", tc_status_string(s)); return 3; }

    float* Yref = malloc(M*N*sizeof(float));
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

    free(Yref);
    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, Wfp16);
    tc_buffer_free(ctx, Wq); tc_buffer_free(ctx, Yb);
    tc_shutdown(ctx);

    return (scaled < 5e-2 && q_bytes == expected) ? 0 : 5;
}
