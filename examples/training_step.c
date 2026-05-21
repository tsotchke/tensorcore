/*
 * training_step.c - one synthetic training iteration end-to-end.
 *
 * Demonstrates the forward + backward + optimizer step assembly for a small
 * RMSnorm -> Linear -> softmax block, the simplest non-trivial transformer
 * fragment. Runs a few SGD-style iterations to show the loss decreasing as
 * the parameters update.
 *
 * Exercises:
 *   - tc_rmsnorm_forward / tc_rmsnorm_backward
 *   - tc_gemm (forward) and tc_gemm with transpose_a / transpose_b (backward)
 *   - tc_softmax_forward / tc_softmax_backward
 *   - tc_adamw_step (fp32 master weights, fp16 grads)
 *
 * This is *not* a full transformer step - there's no attention here. It's
 * the minimal "shows every backward + optimizer call" example. See
 * tests/test_transformer_block.c for the full forward + backward + adamw of
 * a complete Llama-style block at small shapes.
 *
 * Build (automatic):
 *     cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
 *     ./build/examples/training_step
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "tensorcore/tensorcore.h"

/* --- Shapes -------------------------------------------------------------- */

static const int BATCH    = 4;
static const int IN_DIM   = 64;
static const int OUT_DIM  = 32;
static const int N_STEPS  = 15;
static const float LR     = 5e-3f;
static const float BETA1  = 0.9f;
static const float BETA2  = 0.999f;
static const float EPS    = 1e-8f;
static const float WD     = 0.0f;
static const float RMS_EPS = 1e-5f;

/* --- fp16 helpers -------------------------------------------------------- */

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
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x3FF);
    if (exp == 0)  { union {uint32_t u; float f;} v = {sign}; return v.f; }
    if (exp == 31) { union {uint32_t u; float f;} v = {sign | 0x7F800000}; return v.f; }
    union { uint32_t u; float f; } v = { sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13) };
    return v.f;
}

static uint32_t lcg(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
static float lcg_uniform(uint32_t* state) {
    return ((float)(lcg(state) >> 8) / (float)(1u << 24)) * 2.f - 1.f;
}

/* --- Buffer helpers ------------------------------------------------------ */

static tc_buffer* alloc_fp16(tc_context* ctx, int n) {
    tc_buffer* buf = NULL;
    tc_buffer_alloc(ctx, (size_t)n * sizeof(uint16_t), &buf);
    return buf;
}
static tc_buffer* alloc_fp32(tc_context* ctx, int n) {
    tc_buffer* buf = NULL;
    tc_buffer_alloc(ctx, (size_t)n * sizeof(float), &buf);
    return buf;
}

static void fill_random_fp16(tc_buffer* buf, int n, float scale, uint32_t seed) {
    uint16_t* p = NULL;
    tc_buffer_map(buf, (void**)&p);
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) p[i] = f32_to_f16(lcg_uniform(&state) * scale);
}
static void fill_constant_fp16(tc_buffer* buf, int n, float c) {
    uint16_t* p = NULL;
    tc_buffer_map(buf, (void**)&p);
    uint16_t h = f32_to_f16(c);
    for (int i = 0; i < n; ++i) p[i] = h;
}
static void fill_constant_fp32(tc_buffer* buf, int n, float c) {
    float* p = NULL;
    tc_buffer_map(buf, (void**)&p);
    for (int i = 0; i < n; ++i) p[i] = c;
}
static void copy_fp16_to_fp32(tc_buffer* src, tc_buffer* dst, int n) {
    uint16_t* s = NULL;
    float*    d = NULL;
    tc_buffer_map(src, (void**)&s);
    tc_buffer_map(dst, (void**)&d);
    for (int i = 0; i < n; ++i) d[i] = f16_to_f32(s[i]);
}
static void copy_fp32_to_fp16(tc_buffer* src, tc_buffer* dst, int n) {
    float*    s = NULL;
    uint16_t* d = NULL;
    tc_buffer_map(src, (void**)&s);
    tc_buffer_map(dst, (void**)&d);
    for (int i = 0; i < n; ++i) d[i] = f32_to_f16(s[i]);
}

/* --- Loss + gradient (computed on the host for clarity) ------------------ */

static float fused_softmax_ce_loss_and_dlogits(tc_buffer* probs_fp16,
                                              const int* targets,
                                              tc_buffer* dlogits_fp16,
                                              int B, int C) {
    /* For softmax + cross-entropy, the gradient w.r.t. logits collapses to
     * (probs - one_hot) / B. We write that directly into dlogits and skip
     * tc_softmax_backward - the Jacobian is already baked into this form.
     * Applying tc_softmax_backward to (probs - one_hot) would apply the
     * Jacobian a second time and give the wrong gradient. */
    uint16_t* p = NULL;
    tc_buffer_map(probs_fp16, (void**)&p);
    uint16_t* d = NULL;
    tc_buffer_map(dlogits_fp16, (void**)&d);

    float loss = 0.f;
    for (int b = 0; b < B; ++b) {
        const int t = targets[b];
        float pt = f16_to_f32(p[b * C + t]);
        if (pt < 1e-12f) pt = 1e-12f;
        loss -= logf(pt);
        for (int c = 0; c < C; ++c) {
            float v = f16_to_f32(p[b * C + c]);
            float g = (c == t) ? (v - 1.0f) : v;
            d[b * C + c] = f32_to_f16(g / (float)B);
        }
    }
    return loss / (float)B;
}

/* --- One iteration ------------------------------------------------------- */

int main(void) {
    tc_context* ctx = NULL;
    tc_status_t s = tc_init(&ctx);
    if (s != TC_OK && s != TC_ERR_ALREADY_INITIALIZED) {
        fprintf(stderr, "tc_init failed: %s\n", tc_status_string(s));
        return 1;
    }

    printf("\n=== tensorcore synthetic training step ===\n");
    printf("block: RMSnorm[%d] -> Linear[%d -> %d] -> softmax\n",
           IN_DIM, IN_DIM, OUT_DIM);
    printf("batch=%d  lr=%g  beta1=%g  beta2=%g  steps=%d\n\n",
           BATCH, LR, BETA1, BETA2, N_STEPS);

    /* --- Forward activations --- */
    tc_buffer* X       = alloc_fp16(ctx, BATCH * IN_DIM);
    tc_buffer* X_norm  = alloc_fp16(ctx, BATCH * IN_DIM);
    tc_buffer* rstd    = alloc_fp32(ctx, BATCH);
    tc_buffer* gamma   = alloc_fp16(ctx, IN_DIM);
    tc_buffer* W       = alloc_fp16(ctx, IN_DIM * OUT_DIM);
    tc_buffer* logits  = alloc_fp16(ctx, BATCH * OUT_DIM);
    tc_buffer* probs   = alloc_fp16(ctx, BATCH * OUT_DIM);

    /* --- Backward gradients --- */
    tc_buffer* dlogits = alloc_fp16(ctx, BATCH * OUT_DIM);
    tc_buffer* dX_norm = alloc_fp16(ctx, BATCH * IN_DIM);
    tc_buffer* dX      = alloc_fp16(ctx, BATCH * IN_DIM);
    tc_buffer* dW      = alloc_fp16(ctx, IN_DIM * OUT_DIM);
    tc_buffer* dgamma  = alloc_fp32(ctx, IN_DIM);   /* fp32 - matches tc_rmsnorm_backward */

    /* --- Optimizer state (fp32 master weights) --- */
    tc_buffer* W_fp32  = alloc_fp32(ctx, IN_DIM * OUT_DIM);
    tc_buffer* W_m     = alloc_fp32(ctx, IN_DIM * OUT_DIM);
    tc_buffer* W_v     = alloc_fp32(ctx, IN_DIM * OUT_DIM);
    tc_buffer* g_fp32  = alloc_fp32(ctx, IN_DIM);
    tc_buffer* g_m     = alloc_fp32(ctx, IN_DIM);
    tc_buffer* g_v     = alloc_fp32(ctx, IN_DIM);

    /* --- Initialize --- */
    fill_random_fp16(X,     BATCH * IN_DIM,  0.5f,  42u);
    fill_constant_fp16(gamma, IN_DIM, 1.0f);          /* gamma starts at 1 */
    fill_random_fp16(W,     IN_DIM * OUT_DIM, 0.1f, 7919u);
    copy_fp16_to_fp32(W,    W_fp32, IN_DIM * OUT_DIM);
    copy_fp16_to_fp32(gamma, g_fp32, IN_DIM);
    fill_constant_fp32(W_m, IN_DIM * OUT_DIM, 0.f);
    fill_constant_fp32(W_v, IN_DIM * OUT_DIM, 0.f);
    fill_constant_fp32(g_m, IN_DIM, 0.f);
    fill_constant_fp32(g_v, IN_DIM, 0.f);

    /* Synthetic targets: argmax of a fixed permutation of input dims */
    int targets[BATCH];
    {
        uint32_t state = 12345u;
        for (int b = 0; b < BATCH; ++b)
            targets[b] = (int)(lcg(&state) % (uint32_t)OUT_DIM);
    }

    /* --- Loop --- */
    for (int step = 1; step <= N_STEPS; ++step) {

        /* === Forward === */

        /* 1. RMSnorm */
        tc_rmsnorm_forward(ctx, X, gamma, X_norm, rstd, BATCH, IN_DIM, RMS_EPS);

        /* 2. Linear: logits = X_norm @ W */
        tc_gemm_desc d_fwd = {0};
        d_fwd.M = BATCH; d_fwd.N = OUT_DIM; d_fwd.K = IN_DIM;
        d_fwd.a_dtype = TC_DTYPE_F16;
        d_fwd.b_dtype = TC_DTYPE_F16;
        d_fwd.c_dtype = TC_DTYPE_F16;
        d_fwd.accum_dtype = TC_DTYPE_F32;
        d_fwd.alpha = 1.f; d_fwd.beta = 0.f;
        tc_gemm(ctx, &d_fwd, X_norm, W, logits);

        /* 3. Softmax */
        tc_softmax_forward(ctx, logits, probs, BATCH, OUT_DIM);

        /* === Loss === */
        const float loss = fused_softmax_ce_loss_and_dlogits(probs, targets,
                                                            dlogits, BATCH, OUT_DIM);

        /* === Backward ===
         *
         * The loss helper above wrote (probs - one_hot)/B directly into
         * `dlogits` - that *is* the gradient w.r.t. the logits for fused
         * softmax+CE. tc_softmax_backward is NOT called here because the
         * softmax Jacobian is already absorbed into this closed form;
         * calling it would apply the Jacobian a second time. */

        /* Linear backward:
         *   dW       = X_norm^T @ dlogits     (transpose_a=true)
         *   dX_norm  = dlogits @ W^T          (transpose_b=true) */
        tc_gemm_desc d_dW = {0};
        d_dW.M = IN_DIM; d_dW.N = OUT_DIM; d_dW.K = BATCH;
        d_dW.a_dtype = TC_DTYPE_F16;
        d_dW.b_dtype = TC_DTYPE_F16;
        d_dW.c_dtype = TC_DTYPE_F16;
        d_dW.accum_dtype = TC_DTYPE_F32;
        d_dW.alpha = 1.f; d_dW.beta = 0.f;
        d_dW.transpose_a = true;
        tc_gemm(ctx, &d_dW, X_norm, dlogits, dW);

        tc_gemm_desc d_dX = {0};
        d_dX.M = BATCH; d_dX.N = IN_DIM; d_dX.K = OUT_DIM;
        d_dX.a_dtype = TC_DTYPE_F16;
        d_dX.b_dtype = TC_DTYPE_F16;
        d_dX.c_dtype = TC_DTYPE_F16;
        d_dX.accum_dtype = TC_DTYPE_F32;
        d_dX.alpha = 1.f; d_dX.beta = 0.f;
        d_dX.transpose_b = true;
        tc_gemm(ctx, &d_dX, dlogits, W, dX_norm);

        /* RMSnorm backward: produces dX and dgamma. */
        tc_rmsnorm_backward(ctx, X, gamma, dX_norm, rstd,
                            dX, dgamma, BATCH, IN_DIM);

        /* === Optimizer === */

        const float bc1 = 1.f - powf(BETA1, (float)step);
        const float bc2 = 1.f - powf(BETA2, (float)step);

        /* W: fp32 master weights + fp16 grads. */
        tc_adamw_step(ctx, W_fp32, W_m, W_v, dW, TC_DTYPE_F16,
                      IN_DIM * OUT_DIM,
                      LR, BETA1, BETA2, EPS, WD, bc1, bc2);
        copy_fp32_to_fp16(W_fp32, W, IN_DIM * OUT_DIM);

        /* gamma: fp32 dgamma (tc_rmsnorm_backward writes fp32) feeds AdamW. */
        tc_adamw_step(ctx, g_fp32, g_m, g_v, dgamma, TC_DTYPE_F32,
                      IN_DIM,
                      LR, BETA1, BETA2, EPS, WD, bc1, bc2);
        copy_fp32_to_fp16(g_fp32, gamma, IN_DIM);

        printf("step %2d  loss=%.4f  backend(last GEMM)=%s\n",
               step, loss, tc_backend_name(tc_last_backend()));
    }

    /* --- Cleanup --- */
    tc_buffer_free(ctx, X); tc_buffer_free(ctx, X_norm); tc_buffer_free(ctx, rstd);
    tc_buffer_free(ctx, gamma); tc_buffer_free(ctx, W);
    tc_buffer_free(ctx, logits); tc_buffer_free(ctx, probs);
    tc_buffer_free(ctx, dlogits);
    tc_buffer_free(ctx, dX_norm); tc_buffer_free(ctx, dX);
    tc_buffer_free(ctx, dW); tc_buffer_free(ctx, dgamma);
    tc_buffer_free(ctx, W_fp32); tc_buffer_free(ctx, W_m); tc_buffer_free(ctx, W_v);
    tc_buffer_free(ctx, g_fp32); tc_buffer_free(ctx, g_m); tc_buffer_free(ctx, g_v);
    tc_shutdown(ctx);

    printf("\n[done] tensorcore training_step OK\n");
    return 0;
}
