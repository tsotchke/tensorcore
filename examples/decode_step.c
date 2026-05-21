/*
 * decode_step.c - one synthetic Llama decode step end-to-end.
 *
 * Walks the same primitive sequence a real LLM decode loop would, against
 * randomly initialized Q4_0 weights at the 7B Llama shape. Demonstrates:
 *
 *   - tc_quantize_weights (Q4_0)
 *   - tc_fused_rmsnorm_gemv (the LLM inference workhorse)
 *   - tc_gemv_quantized / _async (KV projections + MLP)
 *   - tc_rope_forward (in-place positional encoding on Q/K)
 *   - tc_attention_forward (with GQA = no-op when kv_heads = heads)
 *   - tc_swiglu_forward
 *   - tc_stream_create / _sync (async batching pattern)
 *
 * This is *not* a working inference runtime - there's no real model, no
 * tokenizer, no sampling, no KV-cache management. It's the matrix-layer
 * skeleton that proves the assembly described in docs/inference.md.
 *
 * Build (automatically as part of the main project):
 *
 *     cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
 *     ./build/examples/decode_step
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "tensorcore/tensorcore.h"

/* --- Llama-7B-ish shape -------------------------------------------------- */

static const int HIDDEN     = 4096;
static const int MLP_DIM    = 11008;
static const int N_HEADS    = 32;
static const int N_KV_HEADS = 32;       /* set < N_HEADS to exercise GQA */
static const int HEAD_DIM   = 128;
static const int N_LAYERS   = 2;        /* keep small so the example is fast */
static const int SEQ_KV     = 32;       /* synthetic past-context length */
static const float RMS_EPS   = 1e-5f;
static const float ROPE_BASE = 10000.f;

/* --- Helpers ------------------------------------------------------------- */

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f & 0x7FFFFF);
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random_fp16(tc_buffer* buf, int n_elements, float scale, uint32_t seed) {
    uint16_t* p = NULL;
    tc_buffer_map(buf, (void**)&p);
    uint32_t state = seed;
    for (int i = 0; i < n_elements; ++i) {
        state = state * 1664525u + 1013904223u;
        float r = ((float)(state >> 8) / (float)(1u << 24)) * 2.f - 1.f;
        p[i] = f32_to_f16(r * scale);
    }
}

static tc_buffer* alloc_zero_fp16(tc_context* ctx, int n_elements) {
    tc_buffer* buf = NULL;
    tc_buffer_alloc(ctx, (size_t)n_elements * sizeof(uint16_t), &buf);
    void* p = NULL;
    tc_buffer_map(buf, &p);
    memset(p, 0, (size_t)n_elements * sizeof(uint16_t));
    return buf;
}

static tc_buffer* quantize_fp16_to_q4_0(tc_context* ctx,
                                        tc_buffer* W_fp16,
                                        int N, int K) {
    const size_t q_bytes = tc_quantized_size(TC_QUANT_Q4_0, N, K);
    tc_buffer* W_q = NULL;
    tc_buffer_alloc(ctx, q_bytes, &W_q);
    tc_status_t s = tc_quantize_weights(ctx, W_fp16, W_q, TC_QUANT_Q4_0, N, K);
    if (s != TC_OK) {
        fprintf(stderr, "tc_quantize_weights failed: %s\n", tc_status_string(s));
        exit(1);
    }
    return W_q;
}

/* --- Layer scratch (per-layer Q4_0 weights + norm gammas + KV-cache) ----- */

typedef struct {
    tc_buffer* W_q;       /* [hidden, hidden] Q4_0 */
    tc_buffer* W_k;       /* [n_kv_heads*head_dim, hidden] Q4_0 */
    tc_buffer* W_v;       /* [n_kv_heads*head_dim, hidden] Q4_0 */
    tc_buffer* W_o;       /* [hidden, hidden] Q4_0 */
    tc_buffer* W_gate;    /* [mlp_dim, hidden] Q4_0 */
    tc_buffer* W_up;      /* [mlp_dim, hidden] Q4_0 */
    tc_buffer* W_down;    /* [hidden, mlp_dim] Q4_0 */
    tc_buffer* attn_gamma;/* [hidden] fp16 */
    tc_buffer* mlp_gamma; /* [hidden] fp16 */
    tc_buffer* k_cache;   /* [seq_kv, n_kv_heads, head_dim] fp16 */
    tc_buffer* v_cache;   /* [seq_kv, n_kv_heads, head_dim] fp16 */
} layer_t;

static tc_buffer* quantize_random_weights(tc_context* ctx,
                                          int N, int K,
                                          uint32_t seed) {
    tc_buffer* W_fp16 = NULL;
    tc_buffer_alloc(ctx, (size_t)N * K * sizeof(uint16_t), &W_fp16);
    fill_random_fp16(W_fp16, N * K, 0.02f, seed);
    tc_buffer* W_q = quantize_fp16_to_q4_0(ctx, W_fp16, N, K);
    tc_buffer_free(ctx, W_fp16);
    return W_q;
}

static layer_t make_layer(tc_context* ctx, int layer_idx) {
    layer_t L;
    const uint32_t s = 1u + (uint32_t)layer_idx * 7919u;

    const int kv_dim = N_KV_HEADS * HEAD_DIM;

    L.W_q     = quantize_random_weights(ctx, HIDDEN,    HIDDEN,  s + 0);
    L.W_k     = quantize_random_weights(ctx, kv_dim,    HIDDEN,  s + 1);
    L.W_v     = quantize_random_weights(ctx, kv_dim,    HIDDEN,  s + 2);
    L.W_o     = quantize_random_weights(ctx, HIDDEN,    HIDDEN,  s + 3);
    L.W_gate  = quantize_random_weights(ctx, MLP_DIM,   HIDDEN,  s + 4);
    L.W_up    = quantize_random_weights(ctx, MLP_DIM,   HIDDEN,  s + 5);
    L.W_down  = quantize_random_weights(ctx, HIDDEN,    MLP_DIM, s + 6);

    tc_buffer_alloc(ctx, HIDDEN * sizeof(uint16_t), &L.attn_gamma);
    tc_buffer_alloc(ctx, HIDDEN * sizeof(uint16_t), &L.mlp_gamma);
    fill_random_fp16(L.attn_gamma, HIDDEN, 0.1f, s + 7);
    fill_random_fp16(L.mlp_gamma,  HIDDEN, 0.1f, s + 8);

    /* KV-cache prefilled with garbage to simulate past context */
    tc_buffer_alloc(ctx, (size_t)SEQ_KV * N_KV_HEADS * HEAD_DIM * sizeof(uint16_t), &L.k_cache);
    tc_buffer_alloc(ctx, (size_t)SEQ_KV * N_KV_HEADS * HEAD_DIM * sizeof(uint16_t), &L.v_cache);
    fill_random_fp16(L.k_cache, SEQ_KV * N_KV_HEADS * HEAD_DIM, 0.5f, s + 9);
    fill_random_fp16(L.v_cache, SEQ_KV * N_KV_HEADS * HEAD_DIM, 0.5f, s + 10);

    return L;
}

static void free_layer(tc_context* ctx, layer_t* L) {
    tc_buffer_free(ctx, L->W_q);
    tc_buffer_free(ctx, L->W_k);
    tc_buffer_free(ctx, L->W_v);
    tc_buffer_free(ctx, L->W_o);
    tc_buffer_free(ctx, L->W_gate);
    tc_buffer_free(ctx, L->W_up);
    tc_buffer_free(ctx, L->W_down);
    tc_buffer_free(ctx, L->attn_gamma);
    tc_buffer_free(ctx, L->mlp_gamma);
    tc_buffer_free(ctx, L->k_cache);
    tc_buffer_free(ctx, L->v_cache);
}

/* --- RoPE tables --------------------------------------------------------- */

static void make_rope_tables(tc_context* ctx,
                             tc_buffer** out_cos, tc_buffer** out_sin,
                             int seq, int head_dim) {
    tc_buffer_alloc(ctx, (size_t)seq * (head_dim / 2) * sizeof(float), out_cos);
    tc_buffer_alloc(ctx, (size_t)seq * (head_dim / 2) * sizeof(float), out_sin);
    float* cos_p = NULL;
    float* sin_p = NULL;
    tc_buffer_map(*out_cos, (void**)&cos_p);
    tc_buffer_map(*out_sin, (void**)&sin_p);
    for (int s = 0; s < seq; ++s) {
        for (int k = 0; k < head_dim / 2; ++k) {
            float freq = powf(ROPE_BASE, -2.f * k / (float)head_dim);
            float angle = (float)s * freq;
            cos_p[s * (head_dim / 2) + k] = cosf(angle);
            sin_p[s * (head_dim / 2) + k] = sinf(angle);
        }
    }
}

/* --- One decode step ----------------------------------------------------- */

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    printf("\n=== tensorcore synthetic Llama decode step ===\n");
    printf("hidden=%d heads=%d head_dim=%d kv_heads=%d mlp_dim=%d layers=%d\n",
           HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, MLP_DIM, N_LAYERS);
    printf("seq_kv=%d  (synthetic past context)\n\n", SEQ_KV);

    const int kv_dim = N_KV_HEADS * HEAD_DIM;

    /* RoPE tables - one entry per past+current position */
    tc_buffer* cos_t = NULL;
    tc_buffer* sin_t = NULL;
    make_rope_tables(ctx, &cos_t, &sin_t, SEQ_KV + 1, HEAD_DIM);

    /* Per-token activations */
    tc_buffer* x       = alloc_zero_fp16(ctx, HIDDEN);
    tc_buffer* x_norm  = alloc_zero_fp16(ctx, HIDDEN);
    tc_buffer* q       = alloc_zero_fp16(ctx, HIDDEN);            /* [1, N_HEADS * HEAD_DIM] */
    tc_buffer* k_step  = alloc_zero_fp16(ctx, kv_dim);
    tc_buffer* v_step  = alloc_zero_fp16(ctx, kv_dim);
    tc_buffer* o       = alloc_zero_fp16(ctx, HIDDEN);
    tc_buffer* gate    = alloc_zero_fp16(ctx, MLP_DIM);
    tc_buffer* up      = alloc_zero_fp16(ctx, MLP_DIM);
    tc_buffer* gu      = alloc_zero_fp16(ctx, MLP_DIM);

    /* Seed x with a synthetic "embedding lookup" output */
    fill_random_fp16(x, HIDDEN, 0.05f, 42u);

    /* Build the layers */
    printf("[setup] quantizing %d layers of Q4_0 weights...\n", N_LAYERS);
    const double t_setup_start = now_seconds();
    layer_t* layers = (layer_t*)calloc(N_LAYERS, sizeof(layer_t));
    for (int l = 0; l < N_LAYERS; ++l) {
        layers[l] = make_layer(ctx, l);
    }
    printf("[setup] done in %.2fs\n\n", now_seconds() - t_setup_start);

    /* Stream for async batching across all GEMVs */
    tc_stream* stream = NULL;
    tc_stream_create(ctx, &stream);

    /* Time the decode step */
    const double t0 = now_seconds();

    for (int l = 0; l < N_LAYERS; ++l) {
        layer_t* L = &layers[l];

        /* --- Attention block --- */

        /* Fused RMSnorm + Q projection */
        tc_fused_rmsnorm_gemv(ctx, x, L->attn_gamma, L->W_q, q,
                              1, HIDDEN, HIDDEN, RMS_EPS);

        /* For K, V we'd ideally re-norm but to keep this minimal we re-use
         * the same q-norm path; in a real runtime emit tc_rmsnorm_forward
         * once + 3 separate GEMVs or fold into a single W_qkv. */
        tc_gemv_quantized_async(ctx, x, L->W_k, k_step, TC_QUANT_Q4_0,
                                1, kv_dim, HIDDEN, stream);
        tc_gemv_quantized_async(ctx, x, L->W_v, v_step, TC_QUANT_Q4_0,
                                1, kv_dim, HIDDEN, stream);

        tc_stream_sync(stream);

        /* RoPE on Q and K at position SEQ_KV (the new token) */
        /* Note: tc_rope_forward expects [B, H, S, D]; we have S=1 per token. */
        tc_rope_forward(ctx, q, cos_t, sin_t, 1, N_HEADS, 1, HEAD_DIM);
        tc_rope_forward(ctx, k_step, cos_t, sin_t, 1, N_KV_HEADS, 1, HEAD_DIM);

        /* Append k_step / v_step into the KV-cache at position SEQ_KV.
         * (Your runtime owns this; here we leave the cache as-is and pretend
         * the new token's K/V are already part of the cache for the
         * attention call.) */

        /* Attention over the KV-cache */
        tc_attention_desc adesc = {0};
        adesc.batch       = 1;
        adesc.heads       = N_HEADS;
        adesc.kv_heads    = N_KV_HEADS;
        adesc.seq_q       = 1;
        adesc.seq_kv      = SEQ_KV;
        adesc.head_dim    = HEAD_DIM;
        adesc.io_dtype    = TC_DTYPE_F16;
        adesc.accum_dtype = TC_DTYPE_F32;
        adesc.softmax_scale = 1.f / sqrtf((float)HEAD_DIM);
        adesc.causal      = true;

        tc_attention_forward(ctx, &adesc, q, L->k_cache, L->v_cache, o, NULL);

        /* Output projection (residual fold not done; just project) */
        tc_gemv_quantized(ctx, o, L->W_o, x, TC_QUANT_Q4_0,
                          1, HIDDEN, HIDDEN);

        /* --- MLP block --- */

        /* Fused norm + gate; up via separate GEMV; SwiGLU; then down */
        tc_fused_rmsnorm_gemv(ctx, x, L->mlp_gamma, L->W_gate, gate,
                              1, MLP_DIM, HIDDEN, RMS_EPS);
        tc_gemv_quantized_async(ctx, x, L->W_up, up, TC_QUANT_Q4_0,
                                1, MLP_DIM, HIDDEN, stream);
        tc_stream_sync(stream);

        tc_swiglu_forward(ctx, gate, up, gu, MLP_DIM);

        tc_gemv_quantized(ctx, gu, L->W_down, x, TC_QUANT_Q4_0,
                          1, HIDDEN, MLP_DIM);

        printf("[layer %d] backend after MLP-down: %s\n",
               l, tc_backend_name(tc_last_backend()));
    }

    const double dt = now_seconds() - t0;
    printf("\n[decode] %d layers ran in %.1fms\n", N_LAYERS, dt * 1000.0);
    printf("[decode] backend (last GEMM-class call): %s\n",
           tc_backend_name(tc_last_backend()));

    /* Cleanup */
    tc_stream_destroy(ctx, stream);
    for (int l = 0; l < N_LAYERS; ++l) free_layer(ctx, &layers[l]);
    free(layers);

    tc_buffer_free(ctx, cos_t);
    tc_buffer_free(ctx, sin_t);
    tc_buffer_free(ctx, x);
    tc_buffer_free(ctx, x_norm);
    tc_buffer_free(ctx, q);
    tc_buffer_free(ctx, k_step);
    tc_buffer_free(ctx, v_step);
    tc_buffer_free(ctx, o);
    tc_buffer_free(ctx, gate);
    tc_buffer_free(ctx, up);
    tc_buffer_free(ctx, gu);

    tc_shutdown(ctx);
    printf("\n[done] tensorcore decode_step OK\n");
    return 0;
}
