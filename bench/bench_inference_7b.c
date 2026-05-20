/*
 * tensorcore — synthetic Q4_0 7B-llama decode latency bench.
 *
 * Allocates random Q4_0 weights matching a 7B llama architecture and times
 * one decode step: the per-token GEMV work that dominates inference latency.
 *
 *   7B llama-1: layers=32, hidden=4096, heads=32, head_dim=128, mlp=11008
 *   Per token (M=1) per layer:
 *     - QKV projection : 3 × GEMV(hidden, hidden) = 3 × 4096×4096
 *     - O  projection  : 1 × GEMV(hidden, hidden)
 *     - MLP gate, up   : 2 × GEMV(hidden, mlp_dim)   = 2 × 4096×11008
 *     - MLP down       : 1 × GEMV(mlp_dim, hidden)   = 1 × 11008×4096
 *   Per layer total: 4 × 4096² + 3 × (4096 · 11008)
 *                  = 67M + 135M = 202M weights touched per token
 *   Whole forward (32 layers): 6.5B weights
 *
 * At Q4_0 = 4.5 bits = 0.5625 bytes per weight, that's 3.6 GB of weight
 * read per token. M2 Ultra has ~800 GB/s LPDDR5; theoretical decode rate
 * is ~220 tok/s. Measured llama.cpp on M2 Ultra: ~55-65 tok/s for 7B Q4_0.
 *
 * We're not loading a real model — just timing the GEMV throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    /* Llama-1 7B config. */
    const int hidden = 4096;
    const int heads  = 32;
    const int head_dim = 128;
    const int mlp_dim = 11008;
    const int n_layers = 32;
    (void)heads; (void)head_dim;

    printf("=== tensorcore 7B Q4_0 decode latency bench ===\n");
    printf("hidden=%d heads=%d head_dim=%d mlp_dim=%d layers=%d\n",
           hidden, heads, head_dim, mlp_dim, n_layers);

    /* One shared activation buffer reused across all GEMVs.
     * One shared output buffer.
     * Weight buffers per matmul (all Q4_0). */
    tc_buffer *x_h, *x_mlp, *y_h, *y_mlp;
    tc_buffer_alloc(ctx, hidden  * sizeof(uint16_t), &x_h);
    tc_buffer_alloc(ctx, mlp_dim * sizeof(uint16_t), &x_mlp);
    tc_buffer_alloc(ctx, hidden  * sizeof(uint16_t), &y_h);
    tc_buffer_alloc(ctx, mlp_dim * sizeof(uint16_t), &y_mlp);

    /* Allocate Q4_0 weight buffers for one layer's projections; we reuse them
     * across all "layers" — the loop over layers in this bench is just to
     * accumulate latency, not to do distinct work. */
    const size_t hh_q = tc_quantized_size(TC_QUANT_Q4_0, hidden, hidden);
    const size_t hm_q = tc_quantized_size(TC_QUANT_Q4_0, mlp_dim, hidden);
    const size_t mh_q = tc_quantized_size(TC_QUANT_Q4_0, hidden, mlp_dim);

    tc_buffer *Wq_hh, *Wq_hm, *Wq_mh;
    tc_buffer_alloc(ctx, hh_q, &Wq_hh);
    tc_buffer_alloc(ctx, hm_q, &Wq_hm);
    tc_buffer_alloc(ctx, mh_q, &Wq_mh);

    /* Fill with random Q4_0 bytes — we're benching latency, not correctness. */
    void* p;
    tc_buffer_map(Wq_hh, &p); memset(p, 0x37, hh_q);
    tc_buffer_map(Wq_hm, &p); memset(p, 0x37, hm_q);
    tc_buffer_map(Wq_mh, &p); memset(p, 0x37, mh_q);
    /* Fill activation with random bytes too. */
    tc_buffer_map(x_h, &p);   memset(p, 0x3f, hidden * 2);
    tc_buffer_map(x_mlp, &p); memset(p, 0x3f, mlp_dim * 2);

    printf("\nQ4_0 weight bytes: QKV+O (hidden²) %.1f MB each, "
           "MLP up/gate (hidden·mlp_dim) %.1f MB each, "
           "MLP down (mlp_dim·hidden) %.1f MB.\n",
           hh_q / (1024.0*1024.0), hm_q / (1024.0*1024.0), mh_q / (1024.0*1024.0));
    printf("Total weight footprint per 7B model: %.2f GB\n",
           (4 * hh_q + 2 * hm_q + mh_q) * n_layers / (1024.0*1024.0*1024.0));

    /* Warmup. */
    for (int i = 0; i < 5; ++i) {
        tc_gemv_quantized(ctx, x_h, Wq_hh, y_h, TC_QUANT_Q4_0, 1, hidden, hidden);
    }

    /* Async batched dispatch — fire all GEMVs into one stream, sync at the
     * end per token. Eliminates the per-call cmd-buffer round-trip. */
    tc_stream* st;
    tc_stream_create(ctx, &st);

    const int N_TOKENS = 20;
    const int N_REPEATS = 5;
    double times[N_REPEATS];
    for (int rep = 0; rep < N_REPEATS; ++rep) {
        double t0 = now_seconds();
        for (int tk = 0; tk < N_TOKENS; ++tk) {
            for (int layer = 0; layer < n_layers; ++layer) {
                tc_gemv_quantized_async(ctx, x_h, Wq_hh, y_h, TC_QUANT_Q4_0, 1, hidden, hidden, st);
                tc_gemv_quantized_async(ctx, x_h, Wq_hh, y_h, TC_QUANT_Q4_0, 1, hidden, hidden, st);
                tc_gemv_quantized_async(ctx, x_h, Wq_hh, y_h, TC_QUANT_Q4_0, 1, hidden, hidden, st);
                tc_gemv_quantized_async(ctx, x_h, Wq_hh, y_h, TC_QUANT_Q4_0, 1, hidden, hidden, st);
                tc_gemv_quantized_async(ctx, x_h, Wq_hm, y_mlp, TC_QUANT_Q4_0, 1, mlp_dim, hidden, st);
                tc_gemv_quantized_async(ctx, x_h, Wq_hm, y_mlp, TC_QUANT_Q4_0, 1, mlp_dim, hidden, st);
                tc_gemv_quantized_async(ctx, x_mlp, Wq_mh, y_h, TC_QUANT_Q4_0, 1, hidden, mlp_dim, st);
            }
            tc_stream_sync(st);   /* per-token sync */
        }
        times[rep] = now_seconds() - t0;
    }
    tc_stream_destroy(ctx, st);
    qsort(times, N_REPEATS, sizeof(double), cmp_double);
    const double best_dt = times[0];
    const double med_dt = times[N_REPEATS / 2];
    double per_token_ms = (med_dt / N_TOKENS) * 1000.0;
    double tps = N_TOKENS / med_dt;

    /* Weight read per token = 4 × hh + 2 × hm + 1 × mh, per layer × n_layers. */
    const double weight_bytes_per_token =
        (double)(4 * hh_q + 2 * hm_q + mh_q) * (double)n_layers;
    const double weight_gbps = weight_bytes_per_token * tps / (1024.0*1024.0*1024.0);

    printf("\nResults (%d tokens x %d repeats, %d layers, Q4_0 GEMVs only):\n",
           N_TOKENS, N_REPEATS, n_layers);
    printf("  median time    : %.3f s\n", med_dt);
    printf("  best time      : %.3f s\n", best_dt);
    printf("  median/token   : %.2f ms\n", per_token_ms);
    printf("  median tok/s   : %.1f\n", tps);
    printf("  median weight bw: %.1f GB/s\n", weight_gbps);
    printf("\nReference: llama.cpp on M2 Ultra Q4_0 7B reports ~55-65 tok/s.\n");
    printf("Note: this bench excludes attention + softmax/RoPE/RMSnorm — pure GEMV.\n");

    tc_buffer_free(ctx, x_h); tc_buffer_free(ctx, x_mlp);
    tc_buffer_free(ctx, y_h); tc_buffer_free(ctx, y_mlp);
    tc_buffer_free(ctx, Wq_hh); tc_buffer_free(ctx, Wq_hm); tc_buffer_free(ctx, Wq_mh);
    tc_shutdown(ctx);
    return 0;
}
