/*
 * tests/test_poincare.c — validates the C-API Poincaré ops in
 * lib/ops/poincare_cpu.cpp against analytic identities:
 *
 *   1. Euclidean limit (c=0): mobius_add(x,y) = x+y, exp_map_zero(v)=v,
 *      log_map_zero(x)=x, distance(x,y) = 2||x-y|| (Ganea convention).
 *   2. Curved (c=1) identities: log_x(x) = 0, exp_x(0) = x, d(x,x)=0.
 *   3. Round-trip: exp_x(log_x(y)) ≈ y for small radii.
 *   4. Conformal factor at origin: λ_0 = 2.
 *
 * Diff-tests against torch reference are in the Python suite (poincare.py
 * autograd wrappers); this test exercises the raw C kernels.
 */

#include "tensorcore/tensorcore.h"
#include "tensorcore/poincare.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8
#define D 16

static int fail_(const char* what) {
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static void fill_small(float* p, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        /* Keep ‖row‖ small (< 0.5) so c=1 stays well inside the ball. */
        p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
}

int main(void) {
    tc_context* ctx = NULL;
    if (tc_init(&ctx) != TC_OK) return fail_("tc_init");

    /* Allocate buffers. */
    tc_buffer *bX, *bY, *bV, *bOut, *bDist, *bLam;
    if (tc_buffer_alloc(ctx, (size_t)N * D * sizeof(float), &bX) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, (size_t)N * D * sizeof(float), &bY) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, (size_t)N * D * sizeof(float), &bV) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, (size_t)N * D * sizeof(float), &bOut) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, (size_t)N * sizeof(float),     &bDist) != TC_OK) return 1;
    if (tc_buffer_alloc(ctx, (size_t)N * sizeof(float),     &bLam) != TC_OK) return 1;

    void *X = NULL, *Y = NULL, *V = NULL, *O = NULL, *Dout = NULL, *L = NULL;
    tc_buffer_map(bX, &X); tc_buffer_map(bY, &Y); tc_buffer_map(bV, &V);
    tc_buffer_map(bOut, &O); tc_buffer_map(bDist, &Dout); tc_buffer_map(bLam, &L);

    fill_small((float*)X, N * D, 1);
    fill_small((float*)Y, N * D, 2);
    fill_small((float*)V, N * D, 3);

    int rc = 0;

    /* ----- Euclidean limit (c=0) ----- */
    /* mobius_add(x, y, 0) should equal x + y. */
    tc_poincare_mobius_add(ctx, bX, bY, bOut, 0.0f, N, D);
    {
        float expected[N * D];
        for (int i = 0; i < N * D; ++i) expected[i] = ((float*)X)[i] + ((float*)Y)[i];
        const float e = max_abs_diff((float*)O, expected, N * D);
        printf("  mobius_add (c=0)         max_err=%.3e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        if (e >= 1e-6) rc = 1;
    }

    /* exp_map_zero(v, 0) should equal v. */
    tc_poincare_exp_map_zero(ctx, bV, bOut, 0.0f, N, D);
    {
        const float e = max_abs_diff((float*)O, (float*)V, N * D);
        printf("  exp_map_zero (c=0)       max_err=%.3e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        if (e >= 1e-6) rc = 1;
    }

    /* log_map_zero(x, 0) should equal x. */
    tc_poincare_log_map_zero(ctx, bX, bOut, 0.0f, N, D);
    {
        const float e = max_abs_diff((float*)O, (float*)X, N * D);
        printf("  log_map_zero (c=0)       max_err=%.3e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        if (e >= 1e-6) rc = 1;
    }

    /* distance(x, y, 0) should equal 2*||x-y|| (Ganea convention). */
    tc_poincare_distance(ctx, bX, bY, bDist, 0.0f, N, D);
    {
        float expected[N];
        for (int n = 0; n < N; ++n) {
            float s = 0.0f;
            const float* xr = (float*)X + n * D;
            const float* yr = (float*)Y + n * D;
            for (int d = 0; d < D; ++d) {
                const float dx = xr[d] - yr[d];
                s += dx * dx;
            }
            expected[n] = 2.0f * sqrtf(s);
        }
        const float e = max_abs_diff((float*)Dout, expected, N);
        printf("  distance (c=0) = 2|x-y|  max_err=%.3e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        if (e >= 1e-6) rc = 1;
    }

    /* conformal_factor(x=0, c) → 2. */
    memset(X, 0, (size_t)N * D * sizeof(float));
    tc_poincare_conformal_factor(ctx, bX, bLam, 1.0f, N, D);
    {
        float expected[N];
        for (int i = 0; i < N; ++i) expected[i] = 2.0f;
        const float e = max_abs_diff((float*)L, expected, N);
        printf("  conformal_factor(0,c)=2  max_err=%.3e %s\n", e, e < 1e-6 ? "OK" : "FAIL");
        if (e >= 1e-6) rc = 1;
    }
    /* Refill X for subsequent tests. */
    fill_small((float*)X, N * D, 1);

    /* ----- Curved (c=1) identities ----- */
    /* log_x(x) = 0. */
    tc_poincare_log_map(ctx, bX, bX, bOut, 1.0f, N, D);
    {
        float zero[N * D] = {0};
        const float e = max_abs_diff((float*)O, zero, N * D);
        printf("  log_x(x) = 0  (c=1)      max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5) rc = 1;
    }

    /* distance(x, x, c=1) = 0. */
    tc_poincare_distance(ctx, bX, bX, bDist, 1.0f, N, D);
    {
        float zero[N] = {0};
        const float e = max_abs_diff((float*)Dout, zero, N);
        printf("  distance(x,x,c=1)=0      max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5) rc = 1;
    }

    /* exp_x(0) = x. */
    memset(V, 0, (size_t)N * D * sizeof(float));
    tc_poincare_exp_map(ctx, bX, bV, bOut, 1.0f, N, D);
    {
        const float e = max_abs_diff((float*)O, (float*)X, N * D);
        printf("  exp_x(0) = x  (c=1)      max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5) rc = 1;
    }

    /* exp_x(log_x(y)) ≈ y (round-trip). Refill V with fresh randoms first. */
    fill_small((float*)V, N * D, 3);
    tc_poincare_log_map(ctx, bX, bY, bOut, 1.0f, N, D);
    /* Now bOut = log_x(y); plug into exp_x: */
    /* Use bOut as V, X as the base point, write into bV (reusing). */
    {
        /* Manual: copy bOut to bV, then exp_x(bV) into bOut. */
        tc_buffer_map(bOut, &O);
        memcpy(V, O, (size_t)N * D * sizeof(float));
        tc_poincare_exp_map(ctx, bX, bV, bOut, 1.0f, N, D);
        const float e = max_abs_diff((float*)O, (float*)Y, N * D);
        printf("  exp_x(log_x(y)) ≈ y      max_err=%.3e %s\n", e, e < 1e-3 ? "OK" : "FAIL");
        if (e >= 1e-3) rc = 1;
    }

    /* parallel_transport with x==y should be identity. */
    fill_small((float*)V, N * D, 5);
    tc_poincare_parallel_transport(ctx, bV, bX, bX, bOut, 1.0f, N, D);
    {
        const float e = max_abs_diff((float*)O, (float*)V, N * D);
        printf("  PT_{x→x}(v) = v          max_err=%.3e %s\n", e, e < 1e-5 ? "OK" : "FAIL");
        if (e >= 1e-5) rc = 1;
    }

    tc_buffer_free(ctx, bX); tc_buffer_free(ctx, bY); tc_buffer_free(ctx, bV);
    tc_buffer_free(ctx, bOut); tc_buffer_free(ctx, bDist); tc_buffer_free(ctx, bLam);
    tc_shutdown(ctx);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
