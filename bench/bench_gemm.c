/*
 * tensorcore — TFLOPS bench for tc_gemm.
 *
 * Reports median TFLOPS over N iterations after a warmup, plus the backend
 * that actually served the call. Sweeps square sizes 256..4096.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static void bench_one(tc_context* ctx, int M, int N, int K, tc_dtype_t dt) {
    const size_t elem = tc_dtype_size(dt);
    const size_t bytes_a = (size_t)M * K * elem;
    const size_t bytes_b = (size_t)K * N * elem;
    const size_t bytes_c = (size_t)M * N * elem;

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK) return;
    if (tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK) return;
    if (tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) return;

    /* Fill with something nonzero — values don't matter for timing. */
    void* p; tc_buffer_map(A, &p); memset(p, 0x3f, bytes_a);
    tc_buffer_map(B, &p); memset(p, 0x3f, bytes_b);
    tc_buffer_map(C, &p); memset(p, 0, bytes_c);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = dt; d.b_dtype = dt; d.c_dtype = dt;
    d.accum_dtype = (dt == TC_DTYPE_I8) ? TC_DTYPE_I32 : TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;

    /* warmup */
    for (int i = 0; i < 3; ++i) (void)tc_gemm(ctx, &d, A, B, C);

    const int ITERS = 10;
    double times[16] = {0};
    for (int i = 0; i < ITERS; ++i) {
        const double t0 = now_seconds();
        tc_status_t s = tc_gemm(ctx, &d, A, B, C);
        const double t1 = now_seconds();
        if (s != TC_OK) {
            printf("  %s  M=%d N=%d K=%d   FAIL %s\n",
                   tc_dtype_name(dt), M, N, K, tc_status_string(s));
            goto done;
        }
        times[i] = t1 - t0;
    }
    qsort(times, ITERS, sizeof(double), cmp_double);
    const double med = times[ITERS / 2];
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double tflops = flops / med / 1e12;
    printf("  %-5s  M=%5d N=%5d K=%5d   %-18s  median=%7.2f ms   %6.2f TFLOPS\n",
           tc_dtype_name(dt), M, N, K, tc_backend_name(tc_last_backend()),
           med * 1000.0, tflops);
done:
    tc_buffer_free(ctx, A); tc_buffer_free(ctx, B); tc_buffer_free(ctx, C);
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
    printf("=== tensorcore GEMM bench ===\n");
    printf("device=%s  family=Apple%d\n\n", info.name, (int)info.family);

    const int sizes[] = { 256, 512, 1024, 2048, 4096 };
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i) {
        int n = sizes[i];
        bench_one(ctx, n, n, n, TC_DTYPE_F16);
    }
    printf("\n");
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i) {
        int n = sizes[i];
        bench_one(ctx, n, n, n, TC_DTYPE_F32);
    }
    if (info.supports_bf16_simdgroup) {
        printf("\n");
        for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i) {
            int n = sizes[i];
            bench_one(ctx, n, n, n, TC_DTYPE_BF16);
        }
    }
    tc_shutdown(ctx);
    return 0;
}
