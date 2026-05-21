#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorcore/tensorcore.h"

/* Minimal f32 <-> f16 conversion (IEEE 754 binary16, round to nearest). */
static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round = (mant >> (shift - 1)) & 1;
        return (uint16_t)(sign | ((mant >> shift) + round));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    }
    uint32_t round = (mant >> 12) & 1;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant >> 13) + round));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            ++exp;
            mant &= 0x3FF;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = {out};
    return v.f;
}

static int fill_inputs(uint16_t* A, uint16_t* B, uint16_t* C,
                       float* Af, float* Bf) {
    enum { M = 64, N = 64, K = 64 };
    for (int i = 0; i < M * K; ++i) {
        float v = (float)((i % 17) - 8) / 9.0f;
        Af[i] = f16_to_f32(f32_to_f16(v));
        A[i] = f32_to_f16(v);
    }
    for (int i = 0; i < K * N; ++i) {
        float v = (float)((i % 19) - 9) / 11.0f;
        Bf[i] = f16_to_f32(f32_to_f16(v));
        B[i] = f32_to_f16(v);
    }
    memset(C, 0, (size_t)M * N * sizeof(uint16_t));
    return 0;
}

static void reference_gemm(const float* A, const float* B, float* Cref) {
    enum { M = 64, N = 64, K = 64 };
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            Cref[m * N + n] = sum;
        }
    }
}

static int check_output(const uint16_t* C, const float* Cref,
                        double* out_scaled, double* out_max_abs) {
    enum { M = 64, N = 64 };
    double max_abs = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const double got = (double)f16_to_f32(C[i]);
        const double want = (double)Cref[i];
        const double err = fabs(got - want);
        if (err > max_abs) max_abs = err;
        sum_sq_err += err * err;
        sum_sq_ref += want * want;
    }
    const double rms_err = sqrt(sum_sq_err / (M * N));
    const double rms_ref = sqrt(sum_sq_ref / (M * N));
    const double scaled = rms_err / (rms_ref + 1e-9);
    *out_scaled = scaled;
    *out_max_abs = max_abs;
    return scaled <= 2.0e-2 ? 0 : 1;
}

int main(void) {
    enum { M = 64, N = 64, K = 64 };
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        if (s == TC_ERR_NO_DEVICE) {
            printf("tensorops_runtime_status=skipped_no_gpu reason=%s\n",
                   tc_status_string(s));
            return 0;
        }
        fprintf(stderr, "tensorops_runtime_status=failed_init reason=%s\n",
                tc_status_string(s));
        return 1;
    }

    tc_device_info info;
    s = tc_device_info_get(ctx, &info);
    if (s != TC_OK) {
        fprintf(stderr, "tc_device_info_get failed: %s\n", tc_status_string(s));
        tc_shutdown(ctx);
        return 1;
    }

    if (!info.supports_tensorops_m5) {
        printf("tensorops_runtime_status=skipped_no_m5 family=Apple%d device=\"%s\"\n",
               (int)info.family, info.name);
        tc_shutdown(ctx);
        return 0;
    }

    tc_buffer* A = NULL;
    tc_buffer* B = NULL;
    tc_buffer* C = NULL;
    uint16_t* Ap = NULL;
    uint16_t* Bp = NULL;
    uint16_t* Cp = NULL;
    float* Af = NULL;
    float* Bf = NULL;
    float* Cref = NULL;
    int rc = 1;

    if (tc_buffer_alloc(ctx, (size_t)M * K * sizeof(uint16_t), &A) != TC_OK ||
        tc_buffer_alloc(ctx, (size_t)K * N * sizeof(uint16_t), &B) != TC_OK ||
        tc_buffer_alloc(ctx, (size_t)M * N * sizeof(uint16_t), &C) != TC_OK) {
        fprintf(stderr, "tensorops runtime probe allocation failed\n");
        goto cleanup;
    }
    if (tc_buffer_map(A, (void**)&Ap) != TC_OK ||
        tc_buffer_map(B, (void**)&Bp) != TC_OK ||
        tc_buffer_map(C, (void**)&Cp) != TC_OK) {
        fprintf(stderr, "tensorops runtime probe map failed\n");
        goto cleanup;
    }

    Af = (float*)malloc((size_t)M * K * sizeof(float));
    Bf = (float*)malloc((size_t)K * N * sizeof(float));
    Cref = (float*)malloc((size_t)M * N * sizeof(float));
    if (!Af || !Bf || !Cref) {
        fprintf(stderr, "tensorops runtime probe host allocation failed\n");
        goto cleanup;
    }
    fill_inputs(Ap, Bp, Cp, Af, Bf);
    reference_gemm(Af, Bf, Cref);

    tc_gemm_desc d = {0};
    d.M = M;
    d.N = N;
    d.K = K;
    d.a_dtype = TC_DTYPE_F16;
    d.b_dtype = TC_DTYPE_F16;
    d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    d.alpha = 1.0f;
    d.beta = 0.0f;

    s = tc_gemm(ctx, &d, A, B, C);
    tc_backend_t backend = tc_last_backend();
    const char* backend_name = tc_backend_name(backend);

    if (s != TC_OK) {
        fprintf(stderr, "tensorops runtime GEMM failed: %s\n", tc_status_string(s));
        goto cleanup;
    }
    if (backend != TC_BACKEND_TENSOROPS_M5) {
        fprintf(stderr, "tensorops_runtime_status=failed backend=%s\n", backend_name);
        goto cleanup;
    }

    double scaled = 0.0;
    double max_abs = 0.0;
    if (check_output(Cp, Cref, &scaled, &max_abs) != 0) {
        fprintf(stderr,
                "tensorops_runtime_status=failed backend=%s scaled=%.6e max_abs=%.6e\n",
                backend_name, scaled, max_abs);
        goto cleanup;
    }

    printf("tensorops_runtime_status=passed backend=%s family=Apple%d device=\"%s\" dtype=f16 M=%d N=%d K=%d scaled=%.6e max_abs=%.6e\n",
           backend_name, (int)info.family, info.name, M, N, K, scaled, max_abs);
    rc = 0;

cleanup:
    free(Af);
    free(Bf);
    free(Cref);
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
    tc_shutdown(ctx);
    return rc;
}
