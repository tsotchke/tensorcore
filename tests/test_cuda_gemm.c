/*
 * tests/test_cuda_gemm.c - CUDA backend smoke + perf gate.
 *
 * Builds only when TC_ENABLE_CUDA=ON. At runtime, if no CUDA device is
 * present (e.g. nvidia-smi shows nothing or driver/library mismatch),
 * exits 77 (CMake's "skip" code).
 *
 * Validates:
 *   1. tc_cuda_init() succeeds and reports a device.
 *   2. tc_buffer_alloc returns CUDA-managed memory when
 *      TC_USE_CUDA_GEMM=1 (writes from host code remain coherent).
 *   3. tc_gemm() with TC_USE_CUDA_GEMM=1 routes to cuBLAS for fp32 + fp16.
 *   4. fp32 numerics: random-input identity (M=512, alpha=1, beta=0) error
 *      < 1e-3 vs CPU reference.
 *   5. Performance gate: fp32 4096^3 must exceed 15 TFLOPS on high-end
 *      Ampere+ devices (60+ SMs). Failures usually mean managed memory did
 *      not activate and the staged host/device path was used instead.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int is_high_end_ampere_or_newer(const tc_cuda_device_info* info) {
    return info && info->major >= 8 && info->multiprocessor_count >= 60;
}

static int expect_cuda_backend(const char* label, const char* expected_kernel) {
    if (tc_last_backend() != TC_BACKEND_CUDA) {
        fprintf(stderr, "%s used backend=%s, expected cuda\n",
                label, tc_backend_name(tc_last_backend()));
        return 1;
    }
    if (expected_kernel && strcmp(tc_cuda_last_kernel_name(), expected_kernel) != 0) {
        fprintf(stderr, "%s used kernel=%s, expected %s\n",
                label, tc_cuda_last_kernel_name(), expected_kernel);
        return 1;
    }
    return 0;
}

int main(void) {
    /* Opt into CUDA GEMM dispatch + managed-memory buffer allocations. */
    setenv("TC_USE_CUDA_GEMM", "1", 1);

    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "tc_init failed\n");
        return 1;
    }
    if (tc_cuda_init(ctx) != TC_OK) {
        fprintf(stderr, "[skip] no CUDA device available\n");
        tc_shutdown(ctx);
        return 77;
    }
    tc_cuda_device_info info;
    if (tc_cuda_device_at(0, &info) != TC_OK) {
        fprintf(stderr, "tc_cuda_device_at failed\n");
        tc_shutdown(ctx);
        return 1;
    }
    printf("CUDA device: %s (cc=%s, %.1fGB, %u SMs)\n",
           info.device_name, info.compute_capability,
           info.global_memory_bytes / 1e9, info.multiprocessor_count);

    /* --- Correctness: small fp32 GEMM vs known result. --- */
    {
        const int M = 8, N = 8, K = 8;
        tc_buffer *A, *B, *C;
        if (tc_buffer_alloc(ctx, M*K*sizeof(float), &A) != TC_OK ||
            tc_buffer_alloc(ctx, K*N*sizeof(float), &B) != TC_OK ||
            tc_buffer_alloc(ctx, M*N*sizeof(float), &C) != TC_OK) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        void *Ap, *Bp, *Cp;
        if (tc_buffer_map(A, &Ap) != TC_OK ||
            tc_buffer_map(B, &Bp) != TC_OK ||
            tc_buffer_map(C, &Cp) != TC_OK) {
            fprintf(stderr, "small fp32 map failed\n");
            return 1;
        }
        float *a = (float*)Ap, *b = (float*)Bp, *c = (float*)Cp;

        /* A = identity, B[i,j] = i*K+j. Expected C = B. */
        for (int i = 0; i < M*K; ++i) a[i] = 0;
        for (int i = 0; i < M; ++i) a[i*K + i] = 1.0f;
        for (int i = 0; i < K*N; ++i) b[i] = (float)i;
        memset(c, 0, M*N*sizeof(float));

        tc_gemm_desc d = {0};
        d.M = M; d.N = N; d.K = K;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F32;
        d.accum_dtype = TC_DTYPE_F32;
        if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
            fprintf(stderr, "tc_gemm fp32 small failed\n");
            return 1;
        }
        if (expect_cuda_backend("fp32 identity", "cublas_sgemm_managed")) return 1;
        double err = 0;
        for (int i = 0; i < M*N; ++i) {
            const float diff = c[i] - b[i];
            err += diff * diff;
        }
        err = sqrt(err / (M*N));
        printf("  identity GEMM 8^3 fp32 rms_err=%.2e %s\n", err,
               err < 1e-5 ? "OK" : "FAIL");
        if (err >= 1e-5) return 1;
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    }

    /* --- Perf: fp32 4096^3 must clear 15 TFLOPS. --- */
    {
        const int N = 4096;
        tc_buffer *A, *B, *C;
        if (tc_buffer_alloc(ctx, (size_t)N*N*sizeof(float), &A) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(float), &B) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(float), &C) != TC_OK) {
            fprintf(stderr, "fp32 perf alloc failed\n");
            return 1;
        }
        void *Ap, *Bp, *Cp;
        if (tc_buffer_map(A, &Ap) != TC_OK ||
            tc_buffer_map(B, &Bp) != TC_OK ||
            tc_buffer_map(C, &Cp) != TC_OK) {
            fprintf(stderr, "fp32 perf map failed\n");
            return 1;
        }
        float *a = (float*)Ap, *b = (float*)Bp;
        for (size_t i = 0; i < (size_t)N*N; ++i) {
            a[i] = 0.001f * (i % 512);
            b[i] = 0.001f * ((i * 17) % 512);
        }
        tc_gemm_desc d = {0};
        d.M = N; d.N = N; d.K = N;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F32;
        d.accum_dtype = TC_DTYPE_F32;
        if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
            fprintf(stderr, "tc_gemm fp32 warm failed\n");
            return 1;
        }
        const double t0 = now();
        for (int i = 0; i < 3; ++i) {
            if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
                fprintf(stderr, "tc_gemm fp32 perf failed\n");
                return 1;
            }
        }
        const double dt = (now() - t0) / 3.0;
        const double tflops = 2.0 * N * N * N / dt / 1e12;
        if (expect_cuda_backend("fp32 perf", "cublas_sgemm_managed")) return 1;
        printf("  fp32 GEMM 4096^3: %.3f ms, %.2f TFLOPS  last=%s\n",
               dt * 1000.0, tflops, tc_cuda_last_kernel_name());
        if (is_high_end_ampere_or_newer(&info) && tflops < 15.0) {
            fprintf(stderr, "FAIL: expected >15 TFLOPS, got %.2f (managed memory not active?)\n",
                    tflops);
            return 1;
        } else if (!is_high_end_ampere_or_newer(&info)) {
            printf("  fp32 perf gate skipped for this CUDA device class\n");
        }
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    }

    /* --- Perf: fp16 with default fp32-accum. --- */
    {
        const int N = 4096;
        tc_buffer *A, *B, *C;
        if (tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &A) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &B) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &C) != TC_OK) {
            fprintf(stderr, "fp16 perf alloc failed\n");
            return 1;
        }
        void *Ap, *Bp, *Cp;
        if (tc_buffer_map(A, &Ap) != TC_OK ||
            tc_buffer_map(B, &Bp) != TC_OK ||
            tc_buffer_map(C, &Cp) != TC_OK) {
            fprintf(stderr, "fp16 perf map failed\n");
            return 1;
        }
        uint16_t *a = (uint16_t*)Ap, *b = (uint16_t*)Bp;
        for (size_t i = 0; i < (size_t)N*N; ++i) { a[i] = 0x3C00; b[i] = 0x3C00; }

        tc_gemm_desc d = {0};
        d.M = N; d.N = N; d.K = N;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F16;
        d.accum_dtype = TC_DTYPE_F32;
        if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
            fprintf(stderr, "tc_gemm fp16 warm failed\n");
            return 1;
        }
        const double t0 = now();
        for (int i = 0; i < 5; ++i) {
            if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
                fprintf(stderr, "tc_gemm fp16 perf failed\n");
                return 1;
            }
        }
        const double dt = (now() - t0) / 5.0;
        const double tflops = 2.0 * N * N * N / dt / 1e12;
        if (expect_cuda_backend("fp16 fp32-accum perf",
                                "cublas_gemmex_fp16_tensorop_managed")) return 1;
        printf("  fp16 GEMM 4096^3 (fp32-accum): %.3f ms, %.2f TFLOPS  last=%s\n",
               dt * 1000.0, tflops, tc_cuda_last_kernel_name());
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    }

    /* --- Perf: fp16 with fp16-accum (high-rate tensor-core path). --- */
    setenv("TC_CUDA_FP16_ACCUM", "1", 1);
    {
        const int N = 4096;
        tc_buffer *A, *B, *C;
        if (tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &A) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &B) != TC_OK ||
            tc_buffer_alloc(ctx, (size_t)N*N*sizeof(uint16_t), &C) != TC_OK) {
            fprintf(stderr, "fp16-accum perf alloc failed\n");
            return 1;
        }
        void *Ap, *Bp, *Cp;
        if (tc_buffer_map(A, &Ap) != TC_OK ||
            tc_buffer_map(B, &Bp) != TC_OK ||
            tc_buffer_map(C, &Cp) != TC_OK) {
            fprintf(stderr, "fp16-accum perf map failed\n");
            return 1;
        }
        uint16_t *a = (uint16_t*)Ap, *b = (uint16_t*)Bp;
        for (size_t i = 0; i < (size_t)N*N; ++i) { a[i] = 0x3C00; b[i] = 0x3C00; }

        tc_gemm_desc d = {0};
        d.M = N; d.N = N; d.K = N;
        d.alpha = 1.0f; d.beta = 0.0f;
        d.a_dtype = d.b_dtype = d.c_dtype = TC_DTYPE_F16;
        d.accum_dtype = TC_DTYPE_F32;
        if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
            fprintf(stderr, "tc_gemm fp16-accum warm failed\n");
            return 1;
        }
        const double t0 = now();
        for (int i = 0; i < 5; ++i) {
            if (tc_gemm(ctx, &d, A, B, C) != TC_OK) {
                fprintf(stderr, "tc_gemm fp16-accum perf failed\n");
                return 1;
            }
        }
        const double dt = (now() - t0) / 5.0;
        const double tflops = 2.0 * N * N * N / dt / 1e12;
        if (expect_cuda_backend("fp16 fp16-accum perf",
                                "cublas_gemmex_fp16_tensorop_managed_fp16accum")) return 1;
        printf("  fp16 GEMM 4096^3 (fp16-accum): %.3f ms, %.2f TFLOPS  last=%s\n",
               dt * 1000.0, tflops, tc_cuda_last_kernel_name());
        tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
    }

    tc_shutdown(ctx);
    printf("OK\n");
    return 0;
}
