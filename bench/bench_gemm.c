/*
 * tensorcore — TFLOPS bench for tc_gemm.
 *
 * Reports median throughput over N iterations after a warmup, plus the backend
 * that actually served the call. Sweeps square sizes 256..4096 by default.
 *
 * Optional environment overrides:
 *   TC_BENCH_SIZES=256,512,1024
 *   TC_BENCH_DTYPES=f16,f32,bf16
 *   TC_BENCH_WARMUP=3
 *   TC_BENCH_ITERS=10
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

#define MAX_BENCH_SIZES 32
#define MAX_BENCH_DTYPES 8
#define MAX_BENCH_ITERS 64

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static const char* trim_token(char* s) {
    while (*s && isspace((unsigned char)*s)) ++s;
    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) --e;
    *e = '\0';
    return s;
}

static int only_spaces(const char* s) {
    while (*s) {
        if (!isspace((unsigned char)*s)) return 0;
        ++s;
    }
    return 1;
}

static int env_int(const char* name, int fallback, int min_value, int max_value) {
    const char* value = getenv(name);
    if (!value || !*value) return fallback;

    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || !only_spaces(end)) {
        fprintf(stderr, "warning: ignoring invalid %s=%s\n", name, value);
        return fallback;
    }
    if (parsed < min_value) parsed = min_value;
    if (parsed > max_value) parsed = max_value;
    return (int)parsed;
}

static size_t parse_sizes(int* sizes, size_t capacity) {
    static const int defaults[] = { 256, 512, 1024, 2048, 4096 };
    const char* env = getenv("TC_BENCH_SIZES");
    if (!env || !*env) {
        const size_t count = sizeof(defaults) / sizeof(defaults[0]);
        memcpy(sizes, defaults, count * sizeof(defaults[0]));
        return count;
    }

    char buf[512];
    snprintf(buf, sizeof(buf), "%s", env);
    size_t count = 0;
    for (char* tok = strtok(buf, ","); tok && count < capacity; tok = strtok(NULL, ",")) {
        const char* trimmed = trim_token(tok);
        char* end = NULL;
        long parsed = strtol(trimmed, &end, 10);
        if (end == trimmed || *trim_token(end) != '\0' || parsed <= 0 || parsed > 32768) {
            fprintf(stderr, "warning: ignoring invalid TC_BENCH_SIZES token '%s'\n", trimmed);
            continue;
        }
        sizes[count++] = (int)parsed;
    }
    if (count == 0) {
        fprintf(stderr, "warning: TC_BENCH_SIZES had no valid entries; using defaults\n");
        const size_t default_count = sizeof(defaults) / sizeof(defaults[0]);
        memcpy(sizes, defaults, default_count * sizeof(defaults[0]));
        return default_count;
    }
    return count;
}

static int parse_dtype_token(const char* token, tc_dtype_t* out) {
    if (strcmp(token, "f16") == 0 || strcmp(token, "fp16") == 0) {
        *out = TC_DTYPE_F16;
        return 1;
    }
    if (strcmp(token, "f32") == 0 || strcmp(token, "fp32") == 0) {
        *out = TC_DTYPE_F32;
        return 1;
    }
    if (strcmp(token, "bf16") == 0) {
        *out = TC_DTYPE_BF16;
        return 1;
    }
    return 0;
}

static size_t parse_dtypes(tc_dtype_t* dtypes, size_t capacity, int include_default_bf16) {
    const char* env = getenv("TC_BENCH_DTYPES");
    if (!env || !*env) {
        size_t count = 0;
        dtypes[count++] = TC_DTYPE_F16;
        dtypes[count++] = TC_DTYPE_F32;
        if (include_default_bf16 && count < capacity) dtypes[count++] = TC_DTYPE_BF16;
        return count;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", env);
    size_t count = 0;
    for (char* tok = strtok(buf, ","); tok && count < capacity; tok = strtok(NULL, ",")) {
        char* trimmed = (char*)trim_token(tok);
        for (char* p = trimmed; *p; ++p) *p = (char)tolower((unsigned char)*p);
        tc_dtype_t dt;
        if (!parse_dtype_token(trimmed, &dt)) {
            fprintf(stderr, "warning: ignoring invalid TC_BENCH_DTYPES token '%s'\n", trimmed);
            continue;
        }
        dtypes[count++] = dt;
    }
    if (count == 0) {
        fprintf(stderr, "warning: TC_BENCH_DTYPES had no valid entries; using f16,f32\n");
        dtypes[0] = TC_DTYPE_F16;
        dtypes[1] = TC_DTYPE_F32;
        return 2;
    }
    return count;
}

static void print_throughput(double tflops) {
    if (tflops >= 0.01) {
        printf("%6.2f TFLOPS\n", tflops);
    } else {
        printf("%6.2f GFLOPS\n", tflops * 1000.0);
    }
}

static void bench_one(tc_context* ctx, int M, int N, int K, tc_dtype_t dt, int warmup, int iters) {
    const size_t elem = tc_dtype_size(dt);
    const size_t bytes_a = (size_t)M * K * elem;
    const size_t bytes_b = (size_t)K * N * elem;
    const size_t bytes_c = (size_t)M * N * elem;

    tc_buffer *A = NULL, *B = NULL, *C = NULL;
    if (tc_buffer_alloc(ctx, bytes_a, &A) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_b, &B) != TC_OK ||
        tc_buffer_alloc(ctx, bytes_c, &C) != TC_OK) {
        printf("  %s  M=%d N=%d K=%d   FAIL allocation\n", tc_dtype_name(dt), M, N, K);
        goto done;
    }

    /* Fill with something nonzero — values don't matter for timing. */
    void* p; tc_buffer_map(A, &p); memset(p, 0x3f, bytes_a);
    tc_buffer_map(B, &p); memset(p, 0x3f, bytes_b);
    tc_buffer_map(C, &p); memset(p, 0, bytes_c);

    tc_gemm_desc d = {0};
    d.M = M; d.N = N; d.K = K;
    d.a_dtype = dt; d.b_dtype = dt; d.c_dtype = dt;
    d.accum_dtype = (dt == TC_DTYPE_I8) ? TC_DTYPE_I32 : TC_DTYPE_F32;
    d.alpha = 1.0f; d.beta = 0.0f;

    for (int i = 0; i < warmup; ++i) (void)tc_gemm(ctx, &d, A, B, C);

    double times[MAX_BENCH_ITERS] = {0};
    for (int i = 0; i < iters; ++i) {
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
    qsort(times, (size_t)iters, sizeof(double), cmp_double);
    const double med = times[iters / 2];
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double tflops = flops / med / 1e12;
    printf("  %-5s  M=%5d N=%5d K=%5d   %-18s  median=%7.2f ms   ",
           tc_dtype_name(dt), M, N, K, tc_backend_name(tc_last_backend()),
           med * 1000.0);
    print_throughput(tflops);
done:
    if (A) tc_buffer_free(ctx, A);
    if (B) tc_buffer_free(ctx, B);
    if (C) tc_buffer_free(ctx, C);
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
    printf("device=%s  family=Apple%d\n", info.name, (int)info.family);

    int sizes[MAX_BENCH_SIZES];
    tc_dtype_t dtypes[MAX_BENCH_DTYPES];
    const size_t size_count = parse_sizes(sizes, MAX_BENCH_SIZES);
    const size_t dtype_count = parse_dtypes(dtypes, MAX_BENCH_DTYPES,
                                            info.supports_bf16_simdgroup);
    const int warmup = env_int("TC_BENCH_WARMUP", 3, 0, 100);
    const int iters = env_int("TC_BENCH_ITERS", 10, 1, MAX_BENCH_ITERS);
    printf("warmup=%d  iters=%d\n\n", warmup, iters);

    for (size_t d = 0; d < dtype_count; ++d) {
        if (d > 0) printf("\n");
        for (size_t i = 0; i < size_count; ++i) {
            int n = sizes[i];
            bench_one(ctx, n, n, n, dtypes[d], warmup, iters);
        }
    }
    tc_shutdown(ctx);
    return 0;
}
