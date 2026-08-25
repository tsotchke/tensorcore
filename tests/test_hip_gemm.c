/*
 * tests/test_hip_gemm.c - optional HIP/chipStar backend smoke.
 *
 * Builds only when TC_ENABLE_HIP=ON. At runtime, exits 77 when no HIP
 * device is available so non-HIP CI hosts can still compile the tree.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/hip.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t mant = f & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - exp);
        return (uint16_t)(sign | ((mant >> sh) + ((mant >> (sh - 1)) & 1u)));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) |
                      ((mant >> 13) + ((mant >> 12) & 1u)));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    if (exp == 0 && mant == 0) {
        union { uint32_t u; float f; } v = {sign};
        return v.f;
    }
    if (exp == 31) {
        union { uint32_t u; float f; } v = {sign | 0x7f800000u};
        return v.f;
    }
    if (exp == 0) {
        while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
        ++exp;
        mant &= 0x3ffu;
    }
    union { uint32_t u; float f; } v = {
        sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13)
    };
    return v.f;
}

static int expect_backend(const char* label, const char* expected_kernel) {
    if (tc_last_backend() != TC_BACKEND_HIP) {
        fprintf(stderr, "%s used backend=%s, expected hip\n",
                label, tc_backend_name(tc_last_backend()));
        return 1;
    }
    if (strcmp(tc_hip_last_kernel_name(), expected_kernel) != 0) {
        fprintf(stderr, "%s used kernel=%s, expected %s\n",
                label, tc_hip_last_kernel_name(), expected_kernel);
        return 1;
    }
    return 0;
}

int main(void) {
    unsetenv("TC_USE_HIP_GEMM");
    unsetenv("TC_HIP_GEMM");
    unsetenv("TC_DISABLE_HIP_GEMM");

    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "tc_init failed\n");
        return 1;
    }

    tc_status_t init_s = tc_hip_init(ctx);
    if (init_s != TC_OK) {
        printf("[skip] no HIP/chipStar device available: %s\n", tc_status_string(init_s));
        tc_shutdown(ctx);
        return 77;
    }

    tc_hip_device_info info;
    if (tc_hip_device_count() <= 0 || tc_hip_device_at(0, &info) != TC_OK) {
        fprintf(stderr, "HIP initialized but no device was enumerable\n");
        tc_shutdown(ctx);
        return 1;
    }
    printf("HIP device: %s (vendor=%d, cu=%u, %.1fGB)\n",
           info.device_name, (int)info.vendor, info.compute_units,
           info.global_memory_bytes / 1e9);

    const float A_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float B_vals[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, sizeof(A_vals), &A) != TC_OK ||
        tc_buffer_alloc(ctx, sizeof(B_vals), &B) != TC_OK ||
        tc_buffer_alloc(ctx, sizeof(expected), &C) != TC_OK) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    void *Ap = NULL, *Bp = NULL, *Cp = NULL;
    if (tc_buffer_map(A, &Ap) != TC_OK ||
        tc_buffer_map(B, &Bp) != TC_OK ||
        tc_buffer_map(C, &Cp) != TC_OK) {
        fprintf(stderr, "map failed\n");
        return 1;
    }
    memcpy(Ap, A_vals, sizeof(A_vals));
    memcpy(Bp, B_vals, sizeof(B_vals));
    memset(Cp, 0, sizeof(expected));

    tc_gemm_desc d;
    memset(&d, 0, sizeof(d));
    d.M = 2; d.N = 2; d.K = 2;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F32;
    d.accum_dtype = TC_DTYPE_F32;
    tc_status_t s = tc_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "tc_gemm HIP fp32 failed: %s\n", tc_status_string(s));
        return 1;
    }
    if (expect_backend("fp32 identity", "hipblas_sgemm_staged")) return 1;

    float* out = (float*)Cp;
    for (int i = 0; i < 4; ++i) {
        if (fabsf(out[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "bad HIP GEMM output[%d]=%.6f expected %.6f\n",
                    i, out[i], expected[i]);
            return 1;
        }
    }
    printf("HIP f32 GEMM OK: kernel=hipblas_sgemm_staged\n");

    tc_buffer *Ah = NULL, *Bh = NULL, *Ch = NULL;
    const uint16_t Ah_vals[4] = {
        f32_to_f16(1.0f), f32_to_f16(2.0f),
        f32_to_f16(3.0f), f32_to_f16(4.0f),
    };
    const uint16_t Bh_vals[4] = {
        f32_to_f16(5.0f), f32_to_f16(6.0f),
        f32_to_f16(7.0f), f32_to_f16(8.0f),
    };
    if (tc_buffer_alloc(ctx, sizeof(Ah_vals), &Ah) != TC_OK ||
        tc_buffer_alloc(ctx, sizeof(Bh_vals), &Bh) != TC_OK ||
        tc_buffer_alloc(ctx, sizeof(Ah_vals), &Ch) != TC_OK) {
        fprintf(stderr, "fp16 alloc failed\n");
        return 1;
    }
    void *Ahp = NULL, *Bhp = NULL, *Chp = NULL;
    if (tc_buffer_map(Ah, &Ahp) != TC_OK ||
        tc_buffer_map(Bh, &Bhp) != TC_OK ||
        tc_buffer_map(Ch, &Chp) != TC_OK) {
        fprintf(stderr, "fp16 map failed\n");
        return 1;
    }
    memcpy(Ahp, Ah_vals, sizeof(Ah_vals));
    memcpy(Bhp, Bh_vals, sizeof(Bh_vals));
    memset(Chp, 0, sizeof(Ah_vals));

    memset(&d, 0, sizeof(d));
    d.M = 2; d.N = 2; d.K = 2;
    d.alpha = 1.0f; d.beta = 0.0f;
    d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F16;
    d.accum_dtype = TC_DTYPE_F32;
    s = tc_gemm(ctx, &d, Ah, Bh, Ch);
    if (s != TC_OK) {
        fprintf(stderr, "tc_gemm HIP fp16 failed: %s\n", tc_status_string(s));
        return 1;
    }
    if (expect_backend("fp16 identity", "hipblas_hgemm_staged")) return 1;
    uint16_t* hout = (uint16_t*)Chp;
    for (int i = 0; i < 4; ++i) {
        const float got = f16_to_f32(hout[i]);
        if (fabsf(got - expected[i]) > 1e-3f) {
            fprintf(stderr, "bad HIP hGEMM output[%d]=%.6f expected %.6f\n",
                    i, got, expected[i]);
            return 1;
        }
    }
    printf("HIP fp16 GEMM OK: kernel=hipblas_hgemm_staged\n");
    tc_buffer_free(ctx, Ah);
    tc_buffer_free(ctx, Bh);
    tc_buffer_free(ctx, Ch);

    setenv("TC_DISABLE_HIP_GEMM", "1", 1);
    memset(Cp, 0, sizeof(expected));
    s = tc_gemm(ctx, &d, A, B, C);
    if (s != TC_OK) {
        fprintf(stderr, "tc_gemm CPU fallback failed after disabling HIP: %s\n",
                tc_status_string(s));
        return 1;
    }
    if (tc_last_backend() == TC_BACKEND_HIP) {
        fprintf(stderr, "TC_DISABLE_HIP_GEMM did not force CPU fallback\n");
        return 1;
    }
    unsetenv("TC_DISABLE_HIP_GEMM");

    printf("HIP smoke OK: f32=hipblas_sgemm_staged fallback=%s\n",
           tc_backend_name(tc_last_backend()));
    tc_buffer_free(ctx, A);
    tc_buffer_free(ctx, B);
    tc_buffer_free(ctx, C);
    tc_shutdown(ctx);
    return 0;
}
