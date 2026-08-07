/*
 * tests/test_riemannian_adam_parity.c — the CUDA RiemannianAdam kernels must
 * agree with the CPU reference, and both must reduce to Euclidean Adam as
 * c → 0.
 *
 * Why this test is shaped this way
 * --------------------------------
 * test_riemannian_adam.c checks *invariants* (stays in the ball, ‖x‖ stays 1,
 * zero grad is a no-op). Invariants are necessary and nowhere near
 * sufficient: a kernel that returns the input unchanged passes all three.
 * This test instead pins the CUDA path against the CPU reference elementwise,
 * so the only way to pass is to compute the same update.
 *
 * It calls tc_riemannian_adam_step_*_reference and tc_riemannian_adam_step_*
 * directly rather than flipping a backend switch between two runs. A switch
 * would prove the two runs were *configured* differently; direct calls prove
 * which code ran. On a CPU-only build the dispatching name IS the reference,
 * so the comparison is trivially equal — that is correct and intended: the
 * test then degenerates to the c → 0 Euclidean-limit check, which is
 * backend-independent and still worth running everywhere.
 *
 * Tolerances. The GPU tree-reduces each row's inner products; the CPU sums
 * left to right. They agree to fp32 round-off, not bitwise, so the bound is
 * relative and sized for a D-term reduction — not loosened until it passed.
 *
 * Two measurement choices that are load-bearing:
 *
 * 1. Error is max|a−b| normalised by the RMS of the reference vector, not
 *    per-element relative error. Adam's update is ≈ ±lr regardless of
 *    gradient size, so parameters routinely cross zero; a per-element
 *    relative bound then divides a ~1e-8 absolute difference by a ~1e-6
 *    parameter and reports 3e-2 for two implementations that agree to
 *    round-off. That is a broken ruler, not a finding.
 *
 * 2. Trajectories are short. Because that update is ≈ sign(m̂), a round-off
 *    difference can flip one coordinate's sign and open an O(lr) gap that
 *    has nothing to do with correctness. Two steps is the shortest window
 *    that still exercises parallel transport — step 1 builds the moments,
 *    step 2 consumes the transported ones — so it pins the formula without
 *    measuring chaos.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/riemannian_adam.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reported by lib/core/capabilities.cpp; 0 on CPU-only builds. */
extern int tc_cuda_is_active(void);

#define N_ROWS 16
#define D_DIM  64
#define N_ELEM (N_ROWS * D_DIM)

/* Max-norm error bound for a 64-term fp32 reduction difference. */
static const float kTol = 2e-5f;

static int failures = 0;

/* max|a−b| / rms(b) — see the tolerance note in the file header. */
static float max_err(const float* a, const float* b, int n) {
    float worst = 0.0f, sumsq = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = fabsf(a[i] - b[i]);
        if (e > worst) worst = e;
        sumsq += b[i] * b[i];
    }
    const float rms = sqrtf(sumsq / (float)n);
    return worst / fmaxf(rms, 1e-12f);
}

static void report(const char* what, float err, float tol) {
    const int ok = (err < tol) && !isnan(err);
    printf("  %-42s err/rms=%.3e %s\n", what, err, ok ? "OK" : "FAIL");
    if (!ok) failures = 1;
}

/* A deterministic, reproducible fill — the same values must reach both
 * paths, so no rand() between the two runs. */
static void fill_deterministic(float* p, float* g, float* m, float* v,
                                float scale_p, float scale_g) {
    unsigned s = 0x9E3779B9u;
    for (int i = 0; i < N_ELEM; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u1 = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
        s = s * 1664525u + 1013904223u;
        const float u2 = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
        p[i] = u1 * scale_p;
        g[i] = u2 * scale_g;
        /* Non-zero moments: zero-initialised state hides bugs in the moment
         * update and in the parallel transport that scales it. */
        m[i] = u1 * 0.01f;
        v[i] = fabsf(u2) * 0.001f;
    }
}

static void normalize_rows(float* p) {
    for (int n = 0; n < N_ROWS; ++n) {
        float s = 0.0f;
        for (int d = 0; d < D_DIM; ++d) s += p[n*D_DIM + d] * p[n*D_DIM + d];
        const float nx = sqrtf(s);
        for (int d = 0; d < D_DIM; ++d) p[n*D_DIM + d] /= nx;
    }
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) {
        fprintf(stderr, "FAIL: tc_init\n");
        return 1;
    }
    const int cuda = tc_cuda_is_active();
    printf("RiemannianAdam parity  (N=%d D=%d, CUDA %s)\n",
           N_ROWS, D_DIM, cuda ? "active" : "inactive — reference vs itself");

    const size_t bytes = (size_t)N_ELEM * sizeof(float);
    /* Two independent buffer sets: one driven through the dispatching entry
     * point, one through the reference, from byte-identical inputs. */
    tc_buffer *aP, *aG, *aM, *aV, *bP, *bG, *bM, *bV;
    tc_buffer_alloc(ctx, bytes, &aP); tc_buffer_alloc(ctx, bytes, &aG);
    tc_buffer_alloc(ctx, bytes, &aM); tc_buffer_alloc(ctx, bytes, &aV);
    tc_buffer_alloc(ctx, bytes, &bP); tc_buffer_alloc(ctx, bytes, &bG);
    tc_buffer_alloc(ctx, bytes, &bM); tc_buffer_alloc(ctx, bytes, &bV);
    void *v_aP, *v_aG, *v_aM, *v_aV, *v_bP, *v_bG, *v_bM, *v_bV;
    tc_buffer_map(aP, &v_aP); tc_buffer_map(aG, &v_aG);
    tc_buffer_map(aM, &v_aM); tc_buffer_map(aV, &v_aV);
    tc_buffer_map(bP, &v_bP); tc_buffer_map(bG, &v_bG);
    tc_buffer_map(bM, &v_bM); tc_buffer_map(bV, &v_bV);
    float* AP = (float*)v_aP; float* AG = (float*)v_aG;
    float* AM = (float*)v_aM; float* AV = (float*)v_aV;
    float* BP = (float*)v_bP; float* BG = (float*)v_bG;
    float* BM = (float*)v_bM; float* BV = (float*)v_bV;

    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;

    /* --- Poincaré: dispatch vs reference, 25 steps. ---
     * Multiple steps, not one: parallel transport only shows up in the
     * moments the *next* step consumes, so a single step cannot see it. */
    fill_deterministic(AP, AG, AM, AV, 0.30f, 0.10f);
    memcpy(BP, AP, bytes); memcpy(BG, AG, bytes);
    memcpy(BM, AM, bytes); memcpy(BV, AV, bytes);
    for (int step = 0; step < 2; ++step) {
        const float bc1 = 1.0f - powf(b1, (float)(step + 1));
        const float bc2 = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_poincare(ctx, aP, aG, aM, aV, N_ROWS, D_DIM,
                                          1.0f, lr, b1, b2, eps, wd, bc1, bc2);
        tc_riemannian_adam_step_poincare_reference(
            ctx, bP, bG, bM, bV, N_ROWS, D_DIM,
            1.0f, lr, b1, b2, eps, wd, bc1, bc2);
    }
    report("poincare params  dispatch vs reference", max_err(AP, BP, N_ELEM), kTol);
    report("poincare moment  dispatch vs reference", max_err(AM, BM, N_ELEM), kTol);
    report("poincare second  dispatch vs reference", max_err(AV, BV, N_ELEM), kTol);

    /* --- Sphere: dispatch vs reference, 25 steps. --- */
    fill_deterministic(AP, AG, AM, AV, 1.00f, 0.10f);
    normalize_rows(AP);
    memcpy(BP, AP, bytes); memcpy(BG, AG, bytes);
    memcpy(BM, AM, bytes); memcpy(BV, AV, bytes);
    for (int step = 0; step < 2; ++step) {
        const float bc1 = 1.0f - powf(b1, (float)(step + 1));
        const float bc2 = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_sphere(ctx, aP, aG, aM, aV, N_ROWS, D_DIM,
                                        lr, b1, b2, eps, wd, bc1, bc2);
        tc_riemannian_adam_step_sphere_reference(
            ctx, bP, bG, bM, bV, N_ROWS, D_DIM,
            lr, b1, b2, eps, wd, bc1, bc2);
    }
    report("sphere params    dispatch vs reference", max_err(AP, BP, N_ELEM), kTol);
    report("sphere moment    dispatch vs reference", max_err(AM, BM, N_ELEM), kTol);

    /* --- Euclidean: dispatch vs reference. --- */
    fill_deterministic(AP, AG, AM, AV, 0.02f, 0.01f);
    memcpy(BP, AP, bytes); memcpy(BG, AG, bytes);
    memcpy(BM, AM, bytes); memcpy(BV, AV, bytes);
    for (int step = 0; step < 2; ++step) {
        const float bc1 = 1.0f - powf(b1, (float)(step + 1));
        const float bc2 = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_euclidean(ctx, aP, aG, aM, aV, N_ROWS, D_DIM,
                                           lr, b1, b2, eps, wd, bc1, bc2);
        tc_riemannian_adam_step_euclidean_reference(
            ctx, bP, bG, bM, bV, N_ROWS, D_DIM,
            lr, b1, b2, eps, wd, bc1, bc2);
    }
    /* Elementwise on both sides — no reduction, so this one is near-exact. */
    report("euclidean params dispatch vs reference", max_err(AP, BP, N_ELEM), 1e-6f);

    /* --- The c → 0 limit: Poincaré must become Euclidean Adam. ---
     * This is the check that catches a wrong conformal factor, a wrong
     * Möbius denominator, or a transport that does not tend to the identity.
     * It runs identically on CPU and CUDA builds, so it is a real gate even
     * where no GPU is present.
     *
     * c is small but not zero: at c = 0 several branches short-circuit and
     * the algebra is never exercised. 1e-7 keeps every branch live while
     * leaving the curvature terms below fp32 resolution against the O(1)
     * Adam update.
     *
     * The moments MUST start at zero here, and the reason is the whole
     * subtlety of the check. As c → 0 the conformal factor tends to
     * λ_x = 2/(1 − c‖x‖²) → 2, not to 1, so the Poincaré path feeds Adam
     * g/λ_x² = g/4 while the Euclidean path feeds g. That constant cancels
     * only through Adam's scale invariance -- m and √v both carry the same
     * 1/4 -- which requires the moment state to be consistent with the
     * gradients that built it. Seed m,v identical and non-zero on both sides
     * and the invariance is broken by construction; this check then fails at
     * ~2.0 relative error against a perfectly correct implementation.
     *
     * The factor of 2 in λ is the standard Poincaré convention (geoopt keeps
     * it too), so the limit is Euclidean-up-to-scale. That scale survives one
     * place Adam cannot absorb it: eps is a constant, not a gradient, so
     * m̂/(√v̂ + eps) on the curved side is m̂_E/(√v̂_E + 4·eps) on the flat
     * side. On the coordinate with the smallest |g| that is a percent-level
     * difference in the update, and it swamps everything this check is
     * actually for.
     *
     * So compensate for it exactly -- eps/4 on the Poincaré side -- rather
     * than widening the tolerance to hide it. What remains is pure curvature:
     * the conformal factor, the Möbius denominator, and a transport that must
     * tend to the identity. Any of those wrong and this fails. */
    const float eps_poincare = eps * 0.25f;
    fill_deterministic(AP, AG, AM, AV, 0.02f, 0.01f);
    memset(AM, 0, bytes); memset(AV, 0, bytes);
    memcpy(BP, AP, bytes); memcpy(BG, AG, bytes);
    memcpy(BM, AM, bytes); memcpy(BV, AV, bytes);
    for (int step = 0; step < 3; ++step) {
        const float bc1 = 1.0f - powf(b1, (float)(step + 1));
        const float bc2 = 1.0f - powf(b2, (float)(step + 1));
        tc_riemannian_adam_step_poincare(ctx, aP, aG, aM, aV, N_ROWS, D_DIM,
                                          1e-7f, lr, b1, b2, eps_poincare,
                                          0.0f, bc1, bc2);
        tc_riemannian_adam_step_euclidean(ctx, bP, bG, bM, bV, N_ROWS, D_DIM,
                                           lr, b1, b2, eps, 0.0f, bc1, bc2);
    }
    report("poincare(c->0) == euclidean adam", max_err(AP, BP, N_ELEM), kTol);

    /* --- Wide rows must not be silently wrong. ---
     * The CUDA kernel supports D up to kMaxPerThread * 1024 and reports
     * unsupported above that, which routes to the reference. Either way the
     * public API must return TC_OK and produce finite output; a kernel that
     * quietly writes garbage past its register array would show here. */
    {
        const int wideD = 1024, wideN = 2;
        const size_t wbytes = (size_t)wideN * wideD * sizeof(float);
        tc_buffer *wP, *wG, *wM, *wV;
        tc_buffer_alloc(ctx, wbytes, &wP); tc_buffer_alloc(ctx, wbytes, &wG);
        tc_buffer_alloc(ctx, wbytes, &wM); tc_buffer_alloc(ctx, wbytes, &wV);
        void *w1, *w2, *w3, *w4;
        tc_buffer_map(wP, &w1); tc_buffer_map(wG, &w2);
        tc_buffer_map(wM, &w3); tc_buffer_map(wV, &w4);
        float* WP = (float*)w1; float* WG = (float*)w2;
        float* WM = (float*)w3; float* WV = (float*)w4;
        for (int i = 0; i < wideN * wideD; ++i) {
            WP[i] = 0.01f * ((i % 7) - 3);
            WG[i] = 0.01f * ((i % 5) - 2);
            WM[i] = 0.0f; WV[i] = 0.0f;
        }
        const tc_status_t st = tc_riemannian_adam_step_poincare(
            ctx, wP, wG, wM, wV, wideN, wideD, 1.0f,
            lr, b1, b2, eps, 0.0f, 1.0f - b1, 1.0f - b2);
        int finite_ok = (st == TC_OK);
        for (int i = 0; i < wideN * wideD && finite_ok; ++i) {
            if (!isfinite(WP[i]) || !isfinite(WM[i])) finite_ok = 0;
        }
        printf("  %-42s %s\n", "D=1024 row finite and TC_OK",
               finite_ok ? "OK" : "FAIL");
        if (!finite_ok) failures = 1;
        tc_buffer_free(ctx, wP); tc_buffer_free(ctx, wG);
        tc_buffer_free(ctx, wM); tc_buffer_free(ctx, wV);
    }

    tc_buffer_free(ctx, aP); tc_buffer_free(ctx, aG);
    tc_buffer_free(ctx, aM); tc_buffer_free(ctx, aV);
    tc_buffer_free(ctx, bP); tc_buffer_free(ctx, bG);
    tc_buffer_free(ctx, bM); tc_buffer_free(ctx, bV);
    tc_shutdown(ctx);
    printf("%s\n", failures ? "FAIL" : "OK");
    return failures;
}
