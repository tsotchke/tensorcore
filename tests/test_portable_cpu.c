#include "tensorcore/tensorcore.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float f32_from_bits(uint32_t bits) {
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t f32_to_bits(float value) {
    uint32_t out;
    memcpy(&out, &value, sizeof(out));
    return out;
}

static float f16_to_f32(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t mant = bits & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) return f32_from_bits(sign);
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return f32_from_bits(sign | (uint32_t)(e + 127) << 23 | (mant << 13));
    }
    if (exp == 0x1fu) return f32_from_bits(sign | 0x7f800000u | (mant << 13));
    return f32_from_bits(sign | ((exp + (127u - 15u)) << 23) | (mant << 13));
}

static uint16_t f32_to_f16(float value) {
    const uint32_t bits = f32_to_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        ++half_exp;
        if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

static int expect_status(const char* name, tc_status_t got, tc_status_t want) {
    if (got == want) return 0;
    fprintf(stderr, "%s: got %s want %s\n", name, tc_status_string(got), tc_status_string(want));
    return 1;
}

static int run_padded_f32_gemm(tc_context* ctx) {
    const int M = 5, N = 4, K = 3;
    const int lda = M + 2, ldb = K + 3, ldc = N + 2;
    const size_t a_elems = (size_t)(K - 1) * lda + M;
    const size_t b_elems = (size_t)(N - 1) * ldb + K;
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    int rc = 0;

    if (tc_buffer_alloc(ctx, a_elems * sizeof(float), &A) != TC_OK ||
        tc_buffer_alloc(ctx, b_elems * sizeof(float), &B) != TC_OK ||
        tc_buffer_alloc(ctx, (size_t)M * ldc * sizeof(float), &C) != TC_OK) {
        rc = 1;
        goto cleanup;
    }
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);
    for (size_t i = 0; i < a_elems; ++i) Ap[i] = -37.0f;
    for (size_t i = 0; i < b_elems; ++i) Bp[i] = 19.0f;
    for (size_t i = 0; i < (size_t)M * ldc; ++i) Cp[i] = -11.0f;
    for (int k = 0; k < K; ++k)
        for (int m = 0; m < M; ++m)
            Ap[(size_t)k * lda + m] = 0.1f * (float)(1 + k * M + m);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            Bp[(size_t)n * ldb + k] = -0.05f * (float)(1 + n * K + k);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            Cp[(size_t)m * ldc + n] = 0.25f * (float)(m + n + 1);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F32; d.b_dtype = TC_DTYPE_F32;
    d.c_dtype = TC_DTYPE_F32; d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = true; d.transpose_b = true;
    d.alpha = 0.75f; d.beta = -0.25f;
    d.lda = lda; d.ldb = ldb; d.ldc = ldc;
    rc |= expect_status("padded f32 gemm", tc_gemm(ctx, &d, A, B, C), TC_OK);
    if (tc_last_backend() != TC_BACKEND_PORTABLE_CPU) rc = 1;

    float max_abs = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += Ap[(size_t)k * lda + m] * Bp[(size_t)n * ldb + k];
            }
            const float c0 = 0.25f * (float)(m + n + 1);
            const float want = 0.75f * acc - 0.25f * c0;
            const float err = fabsf(Cp[(size_t)m * ldc + n] - want);
            if (err > max_abs) max_abs = err;
        }
        for (int n = N; n < ldc; ++n) {
            if (Cp[(size_t)m * ldc + n] != -11.0f) rc = 1;
        }
    }
    if (max_abs > 1e-5f) rc = 1;

cleanup:
    if (C) tc_buffer_free(ctx, C);
    if (B) tc_buffer_free(ctx, B);
    if (A) tc_buffer_free(ctx, A);
    return rc;
}

static int run_padded_f16_gemm(tc_context* ctx) {
    const int M = 4, N = 3, K = 5;
    const int lda = M + 1, ldb = K + 2, ldc = N + 1;
    const size_t a_elems = (size_t)(K - 1) * lda + M;
    const size_t b_elems = (size_t)(N - 1) * ldb + K;
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    uint16_t *Ap = NULL, *Bp = NULL, *Cp = NULL;
    int rc = 0;

    if (tc_buffer_alloc(ctx, a_elems * sizeof(uint16_t), &A) != TC_OK ||
        tc_buffer_alloc(ctx, b_elems * sizeof(uint16_t), &B) != TC_OK ||
        tc_buffer_alloc(ctx, (size_t)M * ldc * sizeof(uint16_t), &C) != TC_OK) {
        rc = 1;
        goto cleanup;
    }
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);
    for (size_t i = 0; i < a_elems; ++i) Ap[i] = f32_to_f16(-37.0f);
    for (size_t i = 0; i < b_elems; ++i) Bp[i] = f32_to_f16(19.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < ldc; ++n)
            Cp[(size_t)m * ldc + n] = f32_to_f16(-11.0f);
    for (int k = 0; k < K; ++k)
        for (int m = 0; m < M; ++m)
            Ap[(size_t)k * lda + m] = f32_to_f16(0.025f * (float)(1 + k * M + m));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            Bp[(size_t)n * ldb + k] = f32_to_f16(-0.02f * (float)(1 + n * K + k));
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            Cp[(size_t)m * ldc + n] = f32_to_f16(0.05f * (float)(m + n + 1));

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_F16; d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.transpose_a = true; d.transpose_b = true;
    d.alpha = 0.75f; d.beta = -0.25f;
    d.lda = lda; d.ldb = ldb; d.ldc = ldc;
    rc |= expect_status("padded f16 gemm", tc_gemm(ctx, &d, A, B, C), TC_OK);
    if (tc_last_backend() != TC_BACKEND_PORTABLE_CPU) rc = 1;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += f16_to_f32(Ap[(size_t)k * lda + m]) *
                       f16_to_f32(Bp[(size_t)n * ldb + k]);
            }
            const float c0 = f16_to_f32(f32_to_f16(0.05f * (float)(m + n + 1)));
            const float want = f16_to_f32(f32_to_f16(0.75f * acc - 0.25f * c0));
            const float got = f16_to_f32(Cp[(size_t)m * ldc + n]);
            if (fabsf(got - want) > 3e-3f) rc = 1;
        }
        for (int n = N; n < ldc; ++n) {
            if (Cp[(size_t)m * ldc + n] != f32_to_f16(-11.0f)) rc = 1;
        }
    }

cleanup:
    if (C) tc_buffer_free(ctx, C);
    if (B) tc_buffer_free(ctx, B);
    if (A) tc_buffer_free(ctx, A);
    return rc;
}

static int run_batched_f32_gemm(tc_context* ctx) {
    const int batch = 2, M = 3, N = 3, K = 2;
    const int64_t sa = M * K + 1, sb = K * N + 2, sc = M * N + 1;
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    float *Ap = NULL, *Bp = NULL, *Cp = NULL;
    int rc = 0;
    tc_buffer_alloc(ctx, (size_t)((batch - 1) * sa + M * K) * sizeof(float), &A);
    tc_buffer_alloc(ctx, (size_t)((batch - 1) * sb + K * N) * sizeof(float), &B);
    tc_buffer_alloc(ctx, (size_t)((batch - 1) * sc + M * N) * sizeof(float), &C);
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M * K; ++i) Ap[(size_t)b * sa + i] = (float)(1 + b + i);
        for (int i = 0; i < K * N; ++i) Bp[(size_t)b * sb + i] = (float)(-2 + b + i);
        for (int i = 0; i < M * N; ++i) Cp[(size_t)b * sc + i] = 0.0f;
    }
    tc_gemm_batched_desc bd = {0};
    bd.base.M = M; bd.base.N = N; bd.base.K = K;
    bd.base.a_dtype = TC_DTYPE_F32; bd.base.b_dtype = TC_DTYPE_F32;
    bd.base.c_dtype = TC_DTYPE_F32; bd.base.accum_dtype = TC_DTYPE_F32;
    bd.base.alpha = 1.0f; bd.base.beta = 0.0f;
    bd.batch = batch; bd.stride_a = sa; bd.stride_b = sb; bd.stride_c = sc;
    rc |= expect_status("batched f32 gemm", tc_gemm_batched(ctx, &bd, A, B, C), TC_OK);
    for (int b = 0; b < batch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float want = 0.0f;
                for (int k = 0; k < K; ++k) {
                    want += Ap[(size_t)b * sa + m * K + k] * Bp[(size_t)b * sb + k * N + n];
                }
                if (fabsf(Cp[(size_t)b * sc + m * N + n] - want) > 1e-5f) rc = 1;
            }
        }
    }
    tc_buffer_free(ctx, C);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, A);
    return rc;
}

static int run_i8_gemm(tc_context* ctx) {
    const int M = 2, N = 2, K = 3;
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    int8_t *Ap = NULL, *Bp = NULL;
    int32_t *Cp = NULL;
    int rc = 0;
    tc_buffer_alloc(ctx, M * K, &A);
    tc_buffer_alloc(ctx, K * N, &B);
    tc_buffer_alloc(ctx, M * N * sizeof(int32_t), &C);
    tc_buffer_map(A, (void**)&Ap);
    tc_buffer_map(B, (void**)&Bp);
    tc_buffer_map(C, (void**)&Cp);
    int8_t av[] = {1, -2, 3, 4, 5, -6};
    int8_t bv[] = {-1, 2, 3, -4, 5, 6};
    memcpy(Ap, av, sizeof(av));
    memcpy(Bp, bv, sizeof(bv));
    memset(Cp, 0, M * N * sizeof(int32_t));
    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = TC_DTYPE_I8; d.b_dtype = TC_DTYPE_I8;
    d.c_dtype = TC_DTYPE_I32; d.accum_dtype = TC_DTYPE_I32;
    d.alpha = 1.0f; d.beta = 0.0f;
    rc |= expect_status("i8 gemm", tc_gemm(ctx, &d, A, B, C), TC_OK);
    int32_t want[] = {8, 28, -19, -48};
    for (int i = 0; i < M * N; ++i) {
        if (Cp[i] != want[i]) rc = 1;
    }
    tc_buffer_free(ctx, C);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, A);
    return rc;
}

static int run_quantized(tc_context* ctx) {
    const int M = 1, N = 3, K = 32;
    tc_buffer *X = NULL, *W = NULL, *Wq = NULL, *Y = NULL;
    uint16_t *Xp = NULL, *Wp = NULL, *Yp = NULL;
    uint8_t* Wqp = NULL;
    int rc = 0;
    const size_t q_bytes = tc_quantized_size(TC_QUANT_Q4_0, N, K);
    if (q_bytes != (size_t)N * 18u) rc = 1;
    tc_buffer_alloc(ctx, M * K * sizeof(uint16_t), &X);
    tc_buffer_alloc(ctx, N * K * sizeof(uint16_t), &W);
    tc_buffer_alloc(ctx, q_bytes, &Wq);
    tc_buffer_alloc(ctx, M * N * sizeof(uint16_t), &Y);
    tc_buffer_map(X, (void**)&Xp);
    tc_buffer_map(W, (void**)&Wp);
    tc_buffer_map(Wq, (void**)&Wqp);
    tc_buffer_map(Y, (void**)&Yp);
    for (int i = 0; i < M * K; ++i) Xp[i] = f32_to_f16(0.01f * (float)(i - 15));
    for (int i = 0; i < N * K; ++i) Wp[i] = f32_to_f16(0.02f * (float)((i % 17) - 8));
    rc |= expect_status("quantize q4", tc_quantize_weights(ctx, W, Wq, TC_QUANT_Q4_0, N, K), TC_OK);
    rc |= expect_status("gemv q4", tc_gemv_quantized(ctx, X, Wq, Y, TC_QUANT_Q4_0, M, N, K), TC_OK);
    for (int n = 0; n < N; ++n) {
        const uint8_t* block = Wqp + (size_t)n * 18u;
        const float scale = f16_to_f32(((const uint16_t*)block)[0]);
        float want = 0.0f;
        for (int i = 0; i < 16; ++i) {
            const uint8_t packed = block[2 + i];
            want += f16_to_f32(Xp[i]) * scale * (float)((packed & 0x0f) - 8);
            want += f16_to_f32(Xp[i + 16]) * scale * (float)((packed >> 4) - 8);
        }
        if (fabsf(f16_to_f32(Yp[n]) - f16_to_f32(f32_to_f16(want))) > 1e-3f) rc = 1;
    }
    tc_buffer_free(ctx, Y);
    tc_buffer_free(ctx, Wq);
    tc_buffer_free(ctx, W);
    tc_buffer_free(ctx, X);
    return rc;
}

static int run_distributed(tc_context* ctx) {
    tc_buffer *in = NULL, *out = NULL;
    tc_dist_ctx* d = NULL;
    float *inp = NULL, *outp = NULL;
    int rc = 0;
    tc_buffer_alloc(ctx, 4 * sizeof(float), &in);
    tc_buffer_alloc(ctx, 4 * sizeof(float), &out);
    tc_buffer_map(in, (void**)&inp);
    tc_buffer_map(out, (void**)&outp);
    for (int i = 0; i < 4; ++i) inp[i] = (float)i;
    rc |= expect_status("dist init", tc_dist_init(ctx, TC_DIST_SINGLE, 1, 0, "single://test", &d), TC_OK);
    rc |= expect_status("allreduce", tc_allreduce(d, in, 4, TC_DTYPE_F32, TC_REDUCE_SUM), TC_OK);
    rc |= expect_status("allgather", tc_allgather(d, in, out, 4, TC_DTYPE_F32), TC_OK);
    for (int i = 0; i < 4; ++i) {
        if (outp[i] != inp[i]) rc = 1;
    }
    if (tc_dist_world_size(d) != 1 || tc_dist_rank(d) != 0) rc = 1;
    rc |= expect_status("dist finalize", tc_dist_finalize(d), TC_OK);
    tc_buffer_free(ctx, out);
    tc_buffer_free(ctx, in);
    return rc;
}

int main(void) {
    tc_context* ctx = NULL;
    int rc = 0;
    rc |= expect_status("init", tc_init(&ctx), TC_OK);
    if (!ctx) return 1;

    tc_device_info info;
    rc |= expect_status("device info", tc_device_info_get(ctx, &info), TC_OK);
    if (strcmp(info.name, "portable-cpu") != 0) rc = 1;
    if (strcmp(tc_backend_name(TC_BACKEND_PORTABLE_CPU), "portable_cpu") != 0) rc = 1;

    rc |= run_padded_f32_gemm(ctx);
    rc |= run_padded_f16_gemm(ctx);
    rc |= run_batched_f32_gemm(ctx);
    rc |= run_i8_gemm(ctx);
    rc |= run_quantized(ctx);
    rc |= run_distributed(ctx);
    rc |= expect_status("attention unsupported",
                        tc_attention_forward(ctx, NULL, NULL, NULL, NULL, NULL, NULL),
                        TC_ERR_UNSUPPORTED_FAMILY);

    rc |= expect_status("shutdown", tc_shutdown(ctx), TC_OK);
    printf("portable CPU backend: %s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}
