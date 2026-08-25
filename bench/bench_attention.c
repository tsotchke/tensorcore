/*
 * tensorcore — bench for tc_attention_forward (fp16, D=64).
 *
 * Reports tokens/sec equivalent at common transformer shapes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "tensorcore/tensorcore.h"

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

static int env_int(const char* name, int fallback, int min_value, int max_value) {
    const char* value = getenv(name);
    if (!value || !*value) return fallback;

    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value) return fallback;
    while (*end) {
        if (*end != ' ' && *end != '\t' && *end != '\n' && *end != '\r') return fallback;
        ++end;
    }
    if (parsed < min_value) parsed = min_value;
    if (parsed > max_value) parsed = max_value;
    return (int)parsed;
}

static void bench_one(tc_context* ctx, int B, int H, int S, int D, int causal,
                      int warmup, int iters) {
    const size_t qkv = (size_t)B * H * S * D;
    tc_buffer *Q = NULL, *K = NULL, *V = NULL, *O = NULL;
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &Q);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &K);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &V);
    tc_buffer_alloc(ctx, qkv * sizeof(uint16_t), &O);
    void* p;
    tc_buffer_map(Q, &p); memset(p, 0x3f, qkv * 2);
    tc_buffer_map(K, &p); memset(p, 0x3f, qkv * 2);
    tc_buffer_map(V, &p); memset(p, 0x3f, qkv * 2);
    tc_buffer_map(O, &p); memset(p, 0, qkv * 2);

    tc_attention_desc d = {0};
    d.batch = B; d.heads = H; d.seq_q = S; d.seq_kv = S; d.head_dim = D;
    d.io_dtype = TC_DTYPE_F16; d.accum_dtype = TC_DTYPE_F32;
    d.softmax_scale = 1.0f / sqrtf((float)D);
    d.causal = !!causal; d.return_lse = 0;

    /* warmup */
    for (int i = 0; i < warmup; ++i) (void)tc_attention_forward(ctx, &d, Q, K, V, O, NULL);

    double times[MAX_BENCH_ITERS] = {0};
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_seconds();
        tc_status_t s = tc_attention_forward(ctx, &d, Q, K, V, O, NULL);
        const double t1 = now_seconds();
        if (s != TC_OK) {
            printf("  B=%d H=%d S=%d D=%d   FAIL %s\n", B, H, S, D, tc_status_string(s));
            goto done;
        }
        times[i] = t1 - t0;
    }
    qsort(times, (size_t)iters, sizeof(double), cmp_double);
    const double med = times[iters / 2];
    /* FlashAttention work: 4 * B * H * S * S * D FLOPs (QK^T + softmax + PV ≈ 4SD per row) */
    const double flops = 4.0 * (double)B * H * (double)S * (double)S * D;
    const double tflops = flops / med / 1e12;
    printf("  B=%d H=%2d S=%5d D=%3d causal=%d   %-18s  median=%7.2f ms   %6.2f TFLOPS\n",
           B, H, S, D, causal, tc_backend_name(tc_last_backend()),
           med * 1000.0, tflops);
done:
    tc_buffer_free(ctx, Q); tc_buffer_free(ctx, K);
    tc_buffer_free(ctx, V); tc_buffer_free(ctx, O);
}

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }
    printf("=== tensorcore FlashAttention bench (fp16, D=64) ===\n\n");
    const int warmup = env_int("TC_ATTENTION_BENCH_WARMUP", 3, 0, 100);
    const int iters = env_int("TC_ATTENTION_BENCH_ITERS", 10, 1, MAX_BENCH_ITERS);
    if (env_int("TC_ATTENTION_BENCH_SINGLE", 0, 0, 1)) {
        bench_one(ctx,
                  env_int("TC_ATTENTION_BENCH_B", 1, 1, 64),
                  env_int("TC_ATTENTION_BENCH_H", 1, 1, 128),
                  env_int("TC_ATTENTION_BENCH_S", 16, 1, 32768),
                  env_int("TC_ATTENTION_BENCH_D", 64, 1, 256),
                  env_int("TC_ATTENTION_BENCH_CAUSAL", 1, 0, 1),
                  warmup,
                  iters);
        tc_shutdown(ctx);
        return 0;
    }
    /* Llama-style: H=32, D=128 — we'd use D=128 kernel; for v0.1 D=64 only. */
    bench_one(ctx, 1,  8,   512, 64, 1, warmup, iters);
    bench_one(ctx, 1,  8,  1024, 64, 1, warmup, iters);
    bench_one(ctx, 1,  8,  2048, 64, 1, warmup, iters);
    bench_one(ctx, 1, 16,  2048, 64, 1, warmup, iters);
    bench_one(ctx, 1, 16,  4096, 64, 1, warmup, iters);
    bench_one(ctx, 1, 32,  4096, 64, 1, warmup, iters);
    tc_shutdown(ctx);
    return 0;
}
