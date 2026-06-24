/*
 * tests/test_riemannian_adam.c — validates the three RiemannianAdam
 * variants:
 *
 *   1. Euclidean step matches plain AdamW (same formula path).
 *   2. After many Poincaré steps, the parameter stays inside the ball
 *      (‖x‖ < 1/√c − margin), no NaN, no boundary crossing.
 *   3. After many sphere steps, ‖x‖ stays 1.0 (numerical drift bounded
 *      by the per-step renormalization).
 *   4. Setting grad = 0 keeps params stationary (modulo weight decay).
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/riemannian_adam.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_(const char* what) { fprintf(stderr, "FAIL: %s\n", what); return 1; }

static float maxabs(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return fail_("tc_init");

    const int N = 8, D = 16;
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

    int rc = 0;
    srand(0x55);

    /* --- Euclidean: one step matches the standard AdamW formula. --- */
    for (int i = 0; i < N * D; ++i) {
        P[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
        G[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;
        M[i] = 0.0f; V[i] = 0.0f;
    }
    float P_ref[N * D];
    float M_ref[N * D] = {0};
    float V_ref[N * D] = {0};
    memcpy(P_ref, P, sizeof P_ref);
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.0f;
    const float bc1 = 1.0f - b1, bc2 = 1.0f - b2;
    /* tensorcore version */
    tc_riemannian_adam_step_euclidean(ctx, bP, bG, bM, bV, N, D,
                                       lr, b1, b2, eps, wd, bc1, bc2);
    /* hand-computed reference */
    for (int i = 0; i < N * D; ++i) {
        const float g = G[i];
        const float nm = b1 * M_ref[i] + (1.0f - b1) * g;
        const float nv = b2 * V_ref[i] + (1.0f - b2) * g * g;
        M_ref[i] = nm; V_ref[i] = nv;
        const float upd = (nm / bc1) / (sqrtf(nv / bc2) + eps);
        P_ref[i] = (P_ref[i] - lr * wd * P_ref[i]) - lr * upd;
    }
    {
        const float err = maxabs(P, P_ref, N * D);
        printf("  euclidean step ≡ AdamW       max_err=%.3e %s\n",
               err, err < 1e-6 ? "OK" : "FAIL");
        if (err >= 1e-6) rc = 1;
    }

    /* --- Poincaré: many steps stay inside the ball; no NaN. --- */
    for (int i = 0; i < N * D; ++i) {
        P[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.05f;   /* inside the ball */
        G[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;    /* nontrivial gradient */
        M[i] = 0.0f; V[i] = 0.0f;
    }
    const float c = 1.0f;
    const float sqrt_c = sqrtf(c);
    const float max_norm = (1.0f - 1e-5f) / sqrt_c;
    int finite_ok = 1, inside_ball = 1;
    for (int step = 0; step < 200; ++step) {
        const float bc1_t = 1.0f - powf(b1, (float)(step + 1));
        const float bc2_t = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_poincare(ctx, bP, bG, bM, bV, N, D, c,
                                          lr, b1, b2, eps, wd, bc1_t, bc2_t);
        for (int n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float x = P[n*D + d];
                if (!isfinite(x)) { finite_ok = 0; break; }
                s += x * x;
            }
            if (sqrtf(s) > max_norm + 1e-4f) inside_ball = 0;
        }
        if (!finite_ok) break;
    }
    printf("  poincare 200 steps finite     %s\n", finite_ok ? "OK" : "FAIL");
    printf("  poincare stays in ball        %s\n", inside_ball ? "OK" : "FAIL");
    if (!finite_ok || !inside_ball) rc = 1;

    /* --- Sphere: ‖x‖ stays ~1 after many steps. --- */
    for (int n = 0; n < N; ++n) {
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            P[n*D + d] = ((float)rand()/RAND_MAX - 0.5f);
            s += P[n*D + d] * P[n*D + d];
        }
        const float nx = sqrtf(s);
        for (int d = 0; d < D; ++d) P[n*D + d] /= nx;
    }
    for (int i = 0; i < N * D; ++i) {
        G[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        M[i] = 0.0f; V[i] = 0.0f;
    }
    float max_off_unit = 0.0f;
    for (int step = 0; step < 200; ++step) {
        const float bc1_t = 1.0f - powf(b1, (float)(step + 1));
        const float bc2_t = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_sphere(ctx, bP, bG, bM, bV, N, D,
                                        lr, b1, b2, eps, wd, bc1_t, bc2_t);
        for (int n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += P[n*D + d] * P[n*D + d];
            const float nrm = sqrtf(s);
            const float off = fabsf(nrm - 1.0f);
            if (off > max_off_unit) max_off_unit = off;
        }
    }
    printf("  sphere ‖x‖ stays 1 (off=%.2e) %s\n", max_off_unit,
           max_off_unit < 1e-4 ? "OK" : "FAIL");
    if (max_off_unit >= 1e-4f) rc = 1;

    /* --- Zero grad + zero wd: params should not move. --- */
    for (int i = 0; i < N * D; ++i) {
        P[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
        G[i] = 0.0f; M[i] = 0.0f; V[i] = 0.0f;
    }
    float P_initial[N * D];
    memcpy(P_initial, P, sizeof P_initial);
    tc_riemannian_adam_step_poincare(ctx, bP, bG, bM, bV, N, D, c,
                                      lr, b1, b2, eps, 0.0f, bc1, bc2);
    {
        const float err = maxabs(P, P_initial, N * D);
        printf("  poincare zero-grad no-op      max_err=%.3e %s\n",
               err, err < 1e-6 ? "OK" : "FAIL");
        if (err >= 1e-6) rc = 1;
    }

    tc_buffer_free(ctx, bP); tc_buffer_free(ctx, bG);
    tc_buffer_free(ctx, bM); tc_buffer_free(ctx, bV);
    tc_shutdown(ctx);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
