/*
 * tests/test_riemannian_offmanifold.c — the off-manifold clamp must REPORT.
 *
 * The Poincaré step clamps 1 − c‖x‖² up to kEpsNorm when a row is on or
 * outside the ball. That keeps the arithmetic finite while making the tangent
 * rescale 1/λ_x² ≈ 2.5e-31, so the row's gradient is annihilated rather than
 * projected. Silently. A loss curve is the only witness, and a bad one.
 *
 * This came out of the qLLM geometry A/B (2026-08-07), where the whole
 * question "is the hyperbolic factor actually learning?" was unanswerable
 * from the outside. It turned out to be fine — every row sat at √c‖x‖ ≈ 0.52
 * — but establishing that took a checkpoint dissection, because the engine
 * had no way to say so.
 *
 * Three properties, in the order they matter:
 *   1. in-ball rows report ZERO   — no false alarms, or the signal is noise
 *   2. off-ball rows report NON-ZERO and count exactly once per row
 *   3. reset() actually zeroes
 *
 * (2) is the one that must not be allowed to rot: a counter never observed
 * firing is indistinguishable from a counter wired to nothing.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/riemannian_adam.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fail_(const char* what) { fprintf(stderr, "FAIL: %s\n", what); return 1; }

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return fail_("tc_init");

    enum { N = 8, D = 16 };
    const size_t bytes = (size_t)N * D * sizeof(float);
    tc_buffer *bP, *bG, *bM, *bV;
    tc_buffer_alloc(ctx, bytes, &bP);
    tc_buffer_alloc(ctx, bytes, &bG);
    tc_buffer_alloc(ctx, bytes, &bM);
    tc_buffer_alloc(ctx, bytes, &bV);
    void *Pp = NULL, *Gp = NULL, *Mp = NULL, *Vp = NULL;
    tc_buffer_map(bP, &Pp); tc_buffer_map(bG, &Gp);
    tc_buffer_map(bM, &Mp); tc_buffer_map(bV, &Vp);
    float* P = (float*)Pp; float* G = (float*)Gp;
    float* M = (float*)Mp; float* V = (float*)Vp;

    const float c = 1.0f, lr = 1e-3f, b1 = 0.9f, b2 = 0.999f;
    const float eps = 1e-8f, wd = 0.0f;
    const float bc1 = 1.0f - b1, bc2 = 1.0f - b2;
    int rc = 0;

    /* --- 1. Well inside the ball: must be silent. ------------------------ *
     * √c‖x‖ ≈ 0.5, which is the operating point qLLM's embed_hyper actually
     * trains at (measured mean 0.5198). If this trips, the counter is
     * useless because every real run would drown in false positives. */
    for (int n = 0; n < N; ++n) {
        const float per = 0.5f / sqrtf((float)D);   /* ‖row‖ = 0.5 */
        for (int d = 0; d < D; ++d) {
            P[n * D + d] = per;
            G[n * D + d] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            M[n * D + d] = 0.0f;
            V[n * D + d] = 0.0f;
        }
    }
    tc_riemannian_offmanifold_reset();
    if (tc_riemannian_adam_step_poincare(ctx, bP, bG, bM, bV, N, D, c,
                                         lr, b1, b2, eps, wd, bc1, bc2) != TC_OK)
        rc |= fail_("poincare step (in-ball)");
    unsigned long long in_ball = tc_riemannian_offmanifold_count();
    if (in_ball != 0ULL) {
        fprintf(stderr, "FAIL: in-ball rows reported %llu off-manifold clamps, "
                        "expected 0\n", in_ball);
        rc = 1;
    } else {
        printf("in-ball  (sqrt(c)*||x||=0.50): clamps=0  OK\n");
    }

    /* --- 2. Deliberately outside the ball: must be loud. ----------------- *
     * √c‖x‖ = 4, so 1 − c‖x‖² = −15 and the clamp fires on entry for every
     * one of the N rows. Exactly N, not N*D: the condition is per-row. */
    for (int n = 0; n < N; ++n) {
        const float per = 4.0f / sqrtf((float)D);   /* ‖row‖ = 4 > 1/√c */
        for (int d = 0; d < D; ++d) {
            P[n * D + d] = per;
            G[n * D + d] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            M[n * D + d] = 0.0f;
            V[n * D + d] = 0.0f;
        }
    }
    tc_riemannian_offmanifold_reset();
    if (tc_riemannian_adam_step_poincare(ctx, bP, bG, bM, bV, N, D, c,
                                         lr, b1, b2, eps, wd, bc1, bc2) != TC_OK)
        rc |= fail_("poincare step (off-ball)");
    unsigned long long entry = 0ULL, retract = 0ULL;
    tc_riemannian_offmanifold_counts(&entry, &retract);
    if (entry != (unsigned long long)N) {
        fprintf(stderr, "FAIL: off-ball entry clamps = %llu, expected %d "
                        "(one per row)\n", entry, N);
        rc = 1;
    } else {
        printf("off-ball (sqrt(c)*||x||=4.00): entry=%llu retract=%llu  OK\n",
               entry, retract);
    }
    if (tc_riemannian_offmanifold_count() != entry + retract)
        rc |= fail_("count() != entry + retract");

    /* --- 3. reset() zeroes. ---------------------------------------------- */
    tc_riemannian_offmanifold_reset();
    if (tc_riemannian_offmanifold_count() != 0ULL)
        rc |= fail_("reset did not zero the counters");

    tc_buffer_free(ctx, bP); tc_buffer_free(ctx, bG);
    tc_buffer_free(ctx, bM); tc_buffer_free(ctx, bV);
    tc_shutdown(ctx);
    if (rc == 0) printf("test_riemannian_offmanifold: PASS\n");
    return rc;
}
