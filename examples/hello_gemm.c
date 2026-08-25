/*
 * Minimal "hello tensorcore" example: one fp16 GEMM, print a few result cells.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0) { union {uint32_t u; float f;} v = {sign}; return v.f; }
    if (exp == 31){ union {uint32_t u; float f;} v = {sign | 0x7F800000}; return v.f; }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) { fprintf(stderr, "init failed\n"); return 1; }

    const int M = 256, N = 256, K = 256;
    tc_buffer *A, *B, *C;
    tc_buffer_alloc(ctx, M * K * 2, &A);
    tc_buffer_alloc(ctx, K * N * 2, &B);
    tc_buffer_alloc(ctx, M * N * 2, &C);

    uint16_t *Ap, *Bp, *Cp;
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);

    /* A: row-major identity-ish, B: row-major ramp */
    for (int i = 0; i < M*K; ++i) Ap[i] = f32_to_f16((i % K == i / K) ? 1.0f : 0.0f);
    for (int i = 0; i < K*N; ++i) Bp[i] = f32_to_f16((float)(i % 7) * 0.1f);
    memset(Cp, 0, M*N*2);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;

    tc_status_t s = tc_gemm(ctx, &d, A, B, C);
    printf("tc_gemm: %s  backend=%s\n",
           tc_status_string(s), tc_backend_name(tc_last_backend()));

    /* Print a small block of C */
    printf("C[0..3, 0..3]:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("  %.3f", (double)f16_to_f32(Cp[i*N + j]));
        }
        printf("\n");
    }

    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    tc_shutdown(ctx);
    return 0;
}
