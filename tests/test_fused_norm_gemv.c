/*
 * Fused norm + GEMV correctness vs separate norm-forward + tc_gemm paths.
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

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    const int M = 1, K = 256, N = 128;
    const float eps = 1e-5f;

    tc_buffer *Xb, *gb, *bb, *Wb, *Yf, *Ys, *xn, *mean, *rstd;
    tc_buffer_alloc(ctx, M*K*2, &Xb);
    tc_buffer_alloc(ctx, K*2,   &gb);
    tc_buffer_alloc(ctx, K*2,   &bb);
    tc_buffer_alloc(ctx, K*N*2, &Wb);
    tc_buffer_alloc(ctx, M*N*2, &Yf);   /* fused output */
    tc_buffer_alloc(ctx, M*N*2, &Ys);   /* separate-path output */
    tc_buffer_alloc(ctx, M*K*2, &xn);
    tc_buffer_alloc(ctx, M*4,   &mean);
    tc_buffer_alloc(ctx, M*4,   &rstd);

    uint16_t *Xp, *gp, *bp, *Wp, *Yfp, *Ysp;
    tc_buffer_map(Xb, (void**)&Xp);
    tc_buffer_map(gb, (void**)&gp);
    tc_buffer_map(bb, (void**)&bp);
    tc_buffer_map(Wb, (void**)&Wp);
    tc_buffer_map(Yf, (void**)&Yfp);
    tc_buffer_map(Ys, (void**)&Ysp);

    srand(0x77);
    for (int i = 0; i < M*K; ++i) Xp[i] = f32_to_f16(((float)rand()/RAND_MAX-0.5f));
    for (int i = 0; i < K; ++i)   gp[i] = f32_to_f16(0.5f + (float)rand()/RAND_MAX);
    for (int i = 0; i < K; ++i)   bp[i] = f32_to_f16(((float)rand()/RAND_MAX-0.5f) * 0.2f);
    for (int i = 0; i < K*N; ++i) Wp[i] = f32_to_f16(((float)rand()/RAND_MAX-0.5f) * 0.1f);

    /* RMSNorm path 1: fused. */
    s = tc_fused_rmsnorm_gemv(ctx, Xb, gb, Wb, Yf, M, N, K, eps);
    if (s != TC_OK) { fprintf(stderr, "fused: %s\n", tc_status_string(s)); return 2; }

    /* RMSNorm path 2: separate. */
    s = tc_rmsnorm_forward(ctx, Xb, gb, xn, rstd, M, K, eps);
    if (s != TC_OK) { fprintf(stderr, "rmsnorm: %s\n", tc_status_string(s)); return 3; }
    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;
    s = tc_gemm(ctx, &d, xn, Wb, Ys);
    if (s != TC_OK) { fprintf(stderr, "gemm: %s\n", tc_status_string(s)); return 4; }

    double se = 0, sr = 0, max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        float a = f16_to_f32(Yfp[i]);
        float b = f16_to_f32(Ysp[i]);
        double e = fabs((double)a - (double)b);
        if (e > max_abs) max_abs = e;
        se += e * e; sr += (double)b * b;
    }
    const double scaled = sqrt(se / (M*N)) / (sqrt(sr / (M*N)) + 1e-9);
    printf("fused_rmsnorm_gemv M=%d K=%d N=%d   max_abs=%.3e rms_scaled=%.3e  %s\n",
           M, K, N, max_abs, scaled, (scaled < 5e-3) ? "OK" : "FAIL");

    memset(Yfp, 0, M*N*2);
    memset(Ysp, 0, M*N*2);

    /* LayerNorm path 1: fused. */
    s = tc_fused_layernorm_gemv(ctx, Xb, gb, bb, Wb, Yf, M, N, K, eps);
    if (s != TC_OK) { fprintf(stderr, "fused layernorm: %s\n", tc_status_string(s)); return 6; }

    /* LayerNorm path 2: separate. */
    s = tc_layernorm_forward(ctx, Xb, gb, bb, xn, mean, rstd, M, K, eps);
    if (s != TC_OK) { fprintf(stderr, "layernorm: %s\n", tc_status_string(s)); return 7; }
    s = tc_gemm(ctx, &d, xn, Wb, Ys);
    if (s != TC_OK) { fprintf(stderr, "layernorm gemm: %s\n", tc_status_string(s)); return 8; }

    double layer_se = 0, layer_sr = 0, layer_max_abs = 0;
    for (int i = 0; i < M*N; ++i) {
        float a = f16_to_f32(Yfp[i]);
        float b = f16_to_f32(Ysp[i]);
        double e = fabs((double)a - (double)b);
        if (e > layer_max_abs) layer_max_abs = e;
        layer_se += e * e; layer_sr += (double)b * b;
    }
    const double layer_scaled = sqrt(layer_se / (M*N)) / (sqrt(layer_sr / (M*N)) + 1e-9);
    printf("fused_layernorm_gemv M=%d K=%d N=%d max_abs=%.3e rms_scaled=%.3e  %s\n",
           M, K, N, layer_max_abs, layer_scaled, (layer_scaled < 5e-3) ? "OK" : "FAIL");

    tc_buffer_free(ctx, Xb); tc_buffer_free(ctx, gb); tc_buffer_free(ctx, bb); tc_buffer_free(ctx, Wb);
    tc_buffer_free(ctx, Yf); tc_buffer_free(ctx, Ys);
    tc_buffer_free(ctx, xn); tc_buffer_free(ctx, mean); tc_buffer_free(ctx, rstd);
    tc_shutdown(ctx);
    return (scaled < 5e-3 && layer_scaled < 5e-3) ? 0 : 5;
}
