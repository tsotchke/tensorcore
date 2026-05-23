/*
 * mesh_training_demo.c - split-rank Tensorcore + DiLoCo training demo.
 *
 * Single-rank mode:
 *     ./build/examples/mesh_training_demo
 *
 * Multi-rank mode, one process per host:
 *     ./mesh_training_demo --rank 0 --world 4 --url tcp://100.x.y.z:9100
 *     ./mesh_training_demo --rank 1 --world 4 --url tcp://100.x.y.z:9100
 *
 * Each rank runs local inner training steps for a tiny transformer-style
 * fragment:
 *     RMSNorm -> Linear -> softmax + cross-entropy -> backward -> AdamW
 *
 * After every K inner steps, DiLoCo all-reduces the fp32 master weight and
 * RMSNorm gamma anchors across ranks, then the fp16 forward weights are
 * refreshed from the synchronized masters.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/diloco.h"
#include "tensorcore/distributed.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    BATCH = 4,
    IN_DIM = 64,
    OUT_DIM = 32,
    DEFAULT_INNER_STEPS = 5,
    DEFAULT_OUTER_STEPS = 3,
};

static const float LR = 5e-3f;
static const float BETA1 = 0.9f;
static const float BETA2 = 0.999f;
static const float EPS = 1e-8f;
static const float WD = 0.0f;
static const float RMS_EPS = 1e-5f;

typedef struct DemoState {
    tc_buffer* X;
    tc_buffer* X_norm;
    tc_buffer* rstd;
    tc_buffer* gamma;
    tc_buffer* W;
    tc_buffer* logits;
    tc_buffer* probs;
    tc_buffer* dlogits;
    tc_buffer* dX_norm;
    tc_buffer* dX;
    tc_buffer* dW;
    tc_buffer* dgamma;
    tc_buffer* W_fp32;
    tc_buffer* W_m;
    tc_buffer* W_v;
    tc_buffer* g_fp32;
    tc_buffer* g_m;
    tc_buffer* g_v;
    int targets[BATCH];
} DemoState;

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint16_t f32_to_f16(float x) {
    union { float f; uint32_t u; } v = {x};
    uint32_t f = v.u;
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        union { uint32_t u; float f; } v = {sign};
        return v.f;
    }
    if (exp == 31) {
        union { uint32_t u; float f; } v = {sign | 0x7F800000u};
        return v.f;
    }
    union { uint32_t u; float f; } v = {
        sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13)
    };
    return v.f;
}

static uint32_t lcg(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float uniform_signed(uint32_t* state) {
    return ((float)(lcg(state) >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static int alloc_fp16(tc_context* ctx, int n, tc_buffer** out) {
    return tc_buffer_alloc(ctx, (size_t)n * sizeof(uint16_t), out) == TC_OK ? 0 : 1;
}

static int alloc_fp32(tc_context* ctx, int n, tc_buffer** out) {
    return tc_buffer_alloc(ctx, (size_t)n * sizeof(float), out) == TC_OK ? 0 : 1;
}

static void fill_random_fp16(tc_buffer* b, int n, float scale, uint32_t seed) {
    uint16_t* p = NULL;
    tc_buffer_map(b, (void**)&p);
    uint32_t st = seed;
    for (int i = 0; i < n; ++i) p[i] = f32_to_f16(uniform_signed(&st) * scale);
}

static void fill_constant_fp16(tc_buffer* b, int n, float value) {
    uint16_t* p = NULL;
    tc_buffer_map(b, (void**)&p);
    const uint16_t h = f32_to_f16(value);
    for (int i = 0; i < n; ++i) p[i] = h;
}

static void fill_constant_fp32(tc_buffer* b, int n, float value) {
    float* p = NULL;
    tc_buffer_map(b, (void**)&p);
    for (int i = 0; i < n; ++i) p[i] = value;
}

static void copy_fp16_to_fp32(tc_buffer* src, tc_buffer* dst, int n) {
    uint16_t* s = NULL;
    float* d = NULL;
    tc_buffer_map(src, (void**)&s);
    tc_buffer_map(dst, (void**)&d);
    for (int i = 0; i < n; ++i) d[i] = f16_to_f32(s[i]);
}

static void copy_fp32_to_fp16(tc_buffer* src, tc_buffer* dst, int n) {
    float* s = NULL;
    uint16_t* d = NULL;
    tc_buffer_map(src, (void**)&s);
    tc_buffer_map(dst, (void**)&d);
    for (int i = 0; i < n; ++i) d[i] = f32_to_f16(s[i]);
}

static float loss_and_dlogits(tc_buffer* probs, const int* targets,
                              tc_buffer* dlogits) {
    uint16_t* p = NULL;
    uint16_t* d = NULL;
    tc_buffer_map(probs, (void**)&p);
    tc_buffer_map(dlogits, (void**)&d);
    float loss = 0.0f;
    for (int b = 0; b < BATCH; ++b) {
        const int target = targets[b];
        float pt = f16_to_f32(p[b * OUT_DIM + target]);
        if (pt < 1e-12f) pt = 1e-12f;
        loss -= logf(pt);
        for (int c = 0; c < OUT_DIM; ++c) {
            const float prob = f16_to_f32(p[b * OUT_DIM + c]);
            const float grad = (c == target) ? (prob - 1.0f) : prob;
            d[b * OUT_DIM + c] = f32_to_f16(grad / (float)BATCH);
        }
    }
    return loss / (float)BATCH;
}

static int demo_alloc(tc_context* ctx, DemoState* st) {
    int rc = 0;
    rc |= alloc_fp16(ctx, BATCH * IN_DIM, &st->X);
    rc |= alloc_fp16(ctx, BATCH * IN_DIM, &st->X_norm);
    rc |= alloc_fp32(ctx, BATCH, &st->rstd);
    rc |= alloc_fp16(ctx, IN_DIM, &st->gamma);
    rc |= alloc_fp16(ctx, IN_DIM * OUT_DIM, &st->W);
    rc |= alloc_fp16(ctx, BATCH * OUT_DIM, &st->logits);
    rc |= alloc_fp16(ctx, BATCH * OUT_DIM, &st->probs);
    rc |= alloc_fp16(ctx, BATCH * OUT_DIM, &st->dlogits);
    rc |= alloc_fp16(ctx, BATCH * IN_DIM, &st->dX_norm);
    rc |= alloc_fp16(ctx, BATCH * IN_DIM, &st->dX);
    rc |= alloc_fp16(ctx, IN_DIM * OUT_DIM, &st->dW);
    rc |= alloc_fp32(ctx, IN_DIM, &st->dgamma);
    rc |= alloc_fp32(ctx, IN_DIM * OUT_DIM, &st->W_fp32);
    rc |= alloc_fp32(ctx, IN_DIM * OUT_DIM, &st->W_m);
    rc |= alloc_fp32(ctx, IN_DIM * OUT_DIM, &st->W_v);
    rc |= alloc_fp32(ctx, IN_DIM, &st->g_fp32);
    rc |= alloc_fp32(ctx, IN_DIM, &st->g_m);
    rc |= alloc_fp32(ctx, IN_DIM, &st->g_v);
    return rc;
}

static void demo_init(DemoState* st, int rank) {
    fill_random_fp16(st->X, BATCH * IN_DIM, 0.5f, 42u + (uint32_t)rank * 17u);
    fill_constant_fp16(st->gamma, IN_DIM, 1.0f);
    fill_random_fp16(st->W, IN_DIM * OUT_DIM, 0.1f, 7919u);
    copy_fp16_to_fp32(st->W, st->W_fp32, IN_DIM * OUT_DIM);
    copy_fp16_to_fp32(st->gamma, st->g_fp32, IN_DIM);
    fill_constant_fp32(st->W_m, IN_DIM * OUT_DIM, 0.0f);
    fill_constant_fp32(st->W_v, IN_DIM * OUT_DIM, 0.0f);
    fill_constant_fp32(st->g_m, IN_DIM, 0.0f);
    fill_constant_fp32(st->g_v, IN_DIM, 0.0f);
    uint32_t target_seed = 12345u + (uint32_t)rank * 97u;
    for (int b = 0; b < BATCH; ++b) {
        st->targets[b] = (int)(lcg(&target_seed) % (uint32_t)OUT_DIM);
    }
}

static void demo_free(tc_context* ctx, DemoState* st) {
    (void)ctx;
    tc_buffer_free(ctx, st->X); tc_buffer_free(ctx, st->X_norm);
    tc_buffer_free(ctx, st->rstd); tc_buffer_free(ctx, st->gamma);
    tc_buffer_free(ctx, st->W); tc_buffer_free(ctx, st->logits);
    tc_buffer_free(ctx, st->probs); tc_buffer_free(ctx, st->dlogits);
    tc_buffer_free(ctx, st->dX_norm); tc_buffer_free(ctx, st->dX);
    tc_buffer_free(ctx, st->dW); tc_buffer_free(ctx, st->dgamma);
    tc_buffer_free(ctx, st->W_fp32); tc_buffer_free(ctx, st->W_m);
    tc_buffer_free(ctx, st->W_v); tc_buffer_free(ctx, st->g_fp32);
    tc_buffer_free(ctx, st->g_m); tc_buffer_free(ctx, st->g_v);
}

static int run_inner_step(tc_context* ctx, DemoState* st, int step,
                          float* out_loss) {
    tc_status_t s = tc_rmsnorm_forward(ctx, st->X, st->gamma, st->X_norm,
                                       st->rstd, BATCH, IN_DIM, RMS_EPS);
    if (s != TC_OK) return 1;

    tc_gemm_desc fwd = {0};
    fwd.M = BATCH; fwd.N = OUT_DIM; fwd.K = IN_DIM;
    fwd.a_dtype = TC_DTYPE_F16; fwd.b_dtype = TC_DTYPE_F16;
    fwd.c_dtype = TC_DTYPE_F16; fwd.accum_dtype = TC_DTYPE_F32;
    fwd.alpha = 1.0f; fwd.beta = 0.0f;
    if (tc_gemm(ctx, &fwd, st->X_norm, st->W, st->logits) != TC_OK) return 2;
    if (tc_softmax_forward(ctx, st->logits, st->probs, BATCH, OUT_DIM) != TC_OK) return 3;
    *out_loss = loss_and_dlogits(st->probs, st->targets, st->dlogits);
    if (!isfinite(*out_loss)) return 4;

    tc_gemm_desc dW = {0};
    dW.M = IN_DIM; dW.N = OUT_DIM; dW.K = BATCH;
    dW.a_dtype = TC_DTYPE_F16; dW.b_dtype = TC_DTYPE_F16;
    dW.c_dtype = TC_DTYPE_F16; dW.accum_dtype = TC_DTYPE_F32;
    dW.alpha = 1.0f; dW.beta = 0.0f; dW.transpose_a = true;
    if (tc_gemm(ctx, &dW, st->X_norm, st->dlogits, st->dW) != TC_OK) return 5;

    tc_gemm_desc dX = {0};
    dX.M = BATCH; dX.N = IN_DIM; dX.K = OUT_DIM;
    dX.a_dtype = TC_DTYPE_F16; dX.b_dtype = TC_DTYPE_F16;
    dX.c_dtype = TC_DTYPE_F16; dX.accum_dtype = TC_DTYPE_F32;
    dX.alpha = 1.0f; dX.beta = 0.0f; dX.transpose_b = true;
    if (tc_gemm(ctx, &dX, st->dlogits, st->W, st->dX_norm) != TC_OK) return 6;

    s = tc_rmsnorm_backward(ctx, st->X, st->gamma, st->dX_norm, st->rstd,
                            st->dX, st->dgamma, BATCH, IN_DIM);
    if (s != TC_OK) return 7;

    const float bc1 = 1.0f - powf(BETA1, (float)step);
    const float bc2 = 1.0f - powf(BETA2, (float)step);
    s = tc_adamw_step(ctx, st->W_fp32, st->W_m, st->W_v, st->dW,
                      TC_DTYPE_F16, IN_DIM * OUT_DIM,
                      LR, BETA1, BETA2, EPS, WD, bc1, bc2);
    if (s != TC_OK) return 8;
    s = tc_adamw_step(ctx, st->g_fp32, st->g_m, st->g_v, st->dgamma,
                      TC_DTYPE_F32, IN_DIM,
                      LR, BETA1, BETA2, EPS, WD, bc1, bc2);
    if (s != TC_OK) return 9;
    copy_fp32_to_fp16(st->W_fp32, st->W, IN_DIM * OUT_DIM);
    copy_fp32_to_fp16(st->g_fp32, st->gamma, IN_DIM);
    return 0;
}

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage: %s [--rank R --world W --url tcp://host:port] "
            "[--inner K] [--outer O]\n",
            argv0);
}

int main(int argc, char** argv) {
    int rank = 0;
    int world = 1;
    int inner_steps = DEFAULT_INNER_STEPS;
    int outer_steps = DEFAULT_OUTER_STEPS;
    const char* url = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--world") && i + 1 < argc) world = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--url") && i + 1 < argc) url = argv[++i];
        else if (!strcmp(argv[i], "--inner") && i + 1 < argc) inner_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outer") && i + 1 < argc) outer_steps = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }
    if (rank < 0 || world <= 0 || rank >= world || inner_steps <= 0 || outer_steps <= 0) {
        usage(argv[0]);
        return 2;
    }
    if (world > 1 && !url) {
        fprintf(stderr, "--url is required when --world > 1\n");
        return 2;
    }

    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "[rank %d] tc_init failed\n", rank);
        return 1;
    }

    tc_dist_ctx* dist = NULL;
    const tc_dist_backend_t backend = (world == 1) ? TC_DIST_SINGLE : TC_DIST_GLOO;
    const char* rendezvous = url ? url : "single://mesh-training";
    const double t_init = now_seconds();
    if (tc_dist_init(ctx, backend, world, rank, rendezvous, &dist) != TC_OK) {
        fprintf(stderr, "[rank %d] tc_dist_init failed\n", rank);
        tc_shutdown(ctx);
        return 1;
    }
    printf("[rank %d/%d] rendezvous %.3fs via %s\n",
           rank, world, now_seconds() - t_init, rendezvous);

    DemoState st;
    memset(&st, 0, sizeof(st));
    if (demo_alloc(ctx, &st)) {
        fprintf(stderr, "[rank %d] allocation failed\n", rank);
        tc_dist_finalize(dist);
        tc_shutdown(ctx);
        return 1;
    }
    demo_init(&st, rank);

    tc_diloco_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.inner_steps = inner_steps;
    cfg.outer_lr = 1.0f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = (world > 1) ? TC_DILOCO_COMPRESS_FP16 : TC_DILOCO_COMPRESS_NONE;

    tc_diloco_ctx* dilo = NULL;
    if (tc_diloco_init(dist, &cfg, &dilo) != TC_OK ||
        tc_diloco_add_parameter(dilo, "linear.weight", st.W_fp32,
                                IN_DIM * OUT_DIM, TC_DTYPE_F32) != TC_OK ||
        tc_diloco_add_parameter(dilo, "rmsnorm.gamma", st.g_fp32,
                                IN_DIM, TC_DTYPE_F32) != TC_OK) {
        fprintf(stderr, "[rank %d] DiLoCo setup failed\n", rank);
        demo_free(ctx, &st);
        tc_dist_finalize(dist);
        tc_shutdown(ctx);
        return 1;
    }

    float first_loss = -1.0f;
    float last_loss = -1.0f;
    int rc = 0;
    const double t0 = now_seconds();
    int global_step = 0;
    for (int outer = 0; outer < outer_steps; ++outer) {
        for (int inner = 0; inner < inner_steps; ++inner) {
            ++global_step;
            float loss = 0.0f;
            const int step_rc = run_inner_step(ctx, &st, global_step, &loss);
            if (step_rc) {
                fprintf(stderr, "[rank %d] inner step failed at op %d\n", rank, step_rc);
                rc = 1;
                break;
            }
            if (first_loss < 0.0f) first_loss = loss;
            last_loss = loss;
            bool pending = false;
            if (tc_diloco_step(dilo, &pending) != TC_OK) {
                fprintf(stderr, "[rank %d] tc_diloco_step failed\n", rank);
                rc = 1;
                break;
            }
        }
        if (rc) break;
        if (tc_diloco_apply_outer(dilo) != TC_OK) {
            fprintf(stderr, "[rank %d] tc_diloco_apply_outer failed\n", rank);
            rc = 1;
            break;
        }
        copy_fp32_to_fp16(st.W_fp32, st.W, IN_DIM * OUT_DIM);
        copy_fp32_to_fp16(st.g_fp32, st.gamma, IN_DIM);
        printf("[rank %d] outer %d/%d loss=%.5f bytes=%.0f backend=%s\n",
               rank, outer + 1, outer_steps, last_loss,
               tc_diloco_last_outer_bytes_sent(dilo),
               tc_backend_name(tc_last_backend()));
        fflush(stdout);
    }

    if (tc_barrier(dist) != TC_OK) rc = 1;
    const double elapsed = now_seconds() - t0;
    printf("[rank %d] mesh_training_demo %s first_loss=%.5f last_loss=%.5f "
           "outer_steps=%llu elapsed=%.3fs\n",
           rank, rc ? "FAIL" : "OK", first_loss, last_loss,
           (unsigned long long)tc_diloco_outer_steps_completed(dilo), elapsed);

    tc_diloco_finalize(dilo);
    demo_free(ctx, &st);
    tc_dist_finalize(dist);
    tc_shutdown(ctx);
    return rc;
}
