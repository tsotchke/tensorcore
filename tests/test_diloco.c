/*
 * tests/test_diloco.c — single-rank DiLoCo runtime smoke test.
 *
 * Validates the outer-step machinery in lib/distributed/diloco.cpp at
 * world_size=1 (TC_DIST_SINGLE):
 *
 *   - init / finalize / add_parameter API surface
 *   - inner-step counter triggers outer-step boundary
 *   - apply_outer with no compression converges the model to itself
 *     (single rank, Δθ averaged with itself = identity transform)
 *   - apply_outer with no compression performs the local outer update
 *
 * Multi-rank dense behavior is covered by test_diloco_gloo_fork in the
 * portable CPU suite; this test is the single-process runtime smoke that
 * runs in the default tree.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/diloco.h"
#include "tensorcore/distributed.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint16_t f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | (mant ? 0x7e00u : 0x7c00u));
    int half_exp = (int)exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return sign;
        mant |= 0x800000u;
        const int shift = 14 - half_exp;
        const uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; ++half_exp; if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (rounded >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) {
            float r; uint32_t bits = sign; memcpy(&r, &bits, 4); return r;
        }
        int e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
        mant &= 0x03ffu;
        uint32_t bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        float r; memcpy(&r, &bits, 4); return r;
    }
    if (exp == 0x1fu) {
        uint32_t bits = sign | 0x7f800000u | (mant << 13);
        float r; memcpy(&r, &bits, 4); return r;
    }
    uint32_t bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    float r; memcpy(&r, &bits, 4); return r;
}

static int expect(const char* what, int ok) {
    if (!ok) fprintf(stderr, "DiLoCo: FAIL: %s\n", what);
    return ok ? 0 : 1;
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "DiLoCo: tc_init failed\n");
        return 1;
    }

    int rc = 0;

    /* Single-rank dist context — DiLoCo's all-reduce here is a no-op. */
    tc_dist_ctx* dist = NULL;
    rc |= expect("dist init", tc_dist_init(ctx, TC_DIST_SINGLE, 1, 0, "", &dist) == TC_OK);

    /* Configure DiLoCo: 5 inner steps per outer; SGD outer with lr=1.0
     * (no compression). With world_size=1 the all-reduce is identity, so
     * a single rank's Δθ becomes the global Δ̄θ, and lr=1.0 with SGD
     * outer applies Δ̄θ = Δθ directly: θ_anchor := θ_local; θ_local := θ_anchor.
     * Net effect after one outer step: θ_local unchanged from the inner
     * loop's end state. */
    tc_diloco_config cfg = {0};
    cfg.inner_steps = 5;
    cfg.outer_lr = 1.0f;
    cfg.outer_momentum = 0.0f;
    cfg.outer_optimizer = TC_DILOCO_OUTER_SGD;
    cfg.compress = TC_DILOCO_COMPRESS_NONE;
    cfg.async_overlap = false;
    cfg.tolerate_dropouts = false;

    tc_diloco_ctx* d = NULL;
    rc |= expect("diloco init", tc_diloco_init(dist, &cfg, &d) == TC_OK);

    /* Register a single fp16 parameter, [16 floats]. */
    const int N = 16;
    tc_buffer* theta = NULL;
    rc |= expect("buffer alloc", tc_buffer_alloc(ctx, (size_t)N * sizeof(uint16_t), &theta) == TC_OK);
    void* theta_p = NULL;
    tc_buffer_map(theta, &theta_p);
    uint16_t* t16 = (uint16_t*)theta_p;
    for (int i = 0; i < N; ++i) t16[i] = f32_to_f16(0.5f + 0.01f * i);

    rc |= expect("add param", tc_diloco_add_parameter(d, "p", theta, (size_t)N, TC_DTYPE_F16) == TC_OK);

    /* Simulate 5 inner steps where we increment θ_local by a constant.
     * After 5 increments of +0.1 per step, θ_local = θ_anchor + 0.5.
     * Outer-step Δθ = 0.5; SGD lr=1.0 → θ_anchor += 0.5; θ_local := θ_anchor.
     * So the final θ_local should be the initial θ + 0.5 (across all elems). */
    bool outer_pending = false;
    for (int step = 0; step < 5; ++step) {
        for (int i = 0; i < N; ++i) t16[i] = f32_to_f16(f16_to_f32(t16[i]) + 0.1f);
        rc |= expect("diloco step", tc_diloco_step(d, &outer_pending) == TC_OK);
    }
    rc |= expect("outer pending after 5 inner steps", outer_pending);
    rc |= expect("apply outer", tc_diloco_apply_outer(d) == TC_OK);

    /* Verify: after one outer step at world_size=1, θ_local should equal
     * the post-inner-loop state (within fp16 round-trip noise). */
    int converged = 1;
    for (int i = 0; i < N; ++i) {
        const float want = 0.5f + 0.01f * i + 0.5f;       /* initial + 5×0.1 */
        const float got = f16_to_f32(t16[i]);
        if (fabsf(got - want) > 0.05f) {
            fprintf(stderr, "DiLoCo: element %d: want %.3f got %.3f\n", i, want, got);
            converged = 0;
        }
    }
    rc |= expect("post-outer θ matches inner end-state", converged);

    /* Counter introspection. */
    rc |= expect("outer steps == 1", tc_diloco_outer_steps_completed(d) == 1);
    rc |= expect("inner steps == 5", tc_diloco_inner_steps_completed(d) == 5);

    /* Cleanup. */
    tc_diloco_finalize(d);
    tc_buffer_free(ctx, theta);
    tc_dist_finalize(dist);
    tc_shutdown(ctx);

    printf("DiLoCo single-rank: %s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}
