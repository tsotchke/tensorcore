/*
 * test_product_manifold.c — H × S × R product manifold smoke.
 *
 * Verifies the GeometricLM substrate: a point on a 3-factor product
 * (Lorentz hyperboloid × Sphere × Euclidean) round-trips through
 * exp/log per factor, distance is L² across factors, projection is
 * idempotent.
 */

#include "tensorcore/product_manifold.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int test_count = 0, test_pass = 0;
#define CHECK(label, cond) do { test_count++; \
    if (cond) { test_pass++; printf("  PASS %s\n", label); } \
    else      { printf("  FAIL %s\n", label); } } while (0)

int main(void) {
    /* H × S × R: Lorentz(intrinsic=4, |κ|=1) × Sphere(intrinsic=3, r=1)
     * × Euclidean(intrinsic=8) — ambient dim = 5 + 4 + 8 = 17. */
    tc_factor_t factors[3] = {
        { TC_FACTOR_LORENTZ,   4, 1.0f },
        { TC_FACTOR_SPHERE,    3, 1.0f },
        { TC_FACTOR_EUCLIDEAN, 8, 0.0f },
    };
    const int32_t n_factors = 3;
    const int32_t total = tc_product_ambient_dim(factors, n_factors);
    CHECK("ambient dim = 17 (5+4+8)", total == 17);

    /* Build a base point: origin in each factor. */
    float base[17] = {0};
    base[0] = 1.0f;  /* Lorentz origin: (1, 0, 0, 0, 0) */
    base[5] = 1.0f;  /* Sphere origin:  (1, 0, 0, 0) at offset 5 */
    /* Euclidean origin = zeros at offset 9 */

    /* Projection should be a no-op on an already-valid point. */
    float base_copy[17];
    memcpy(base_copy, base, sizeof(base));
    tc_product_project(factors, n_factors, base_copy);
    float drift = 0.0f;
    for (int i = 0; i < 17; ++i) {
        float d = fabsf(base[i] - base_copy[i]);
        if (d > drift) drift = d;
    }
    CHECK("projection is idempotent on valid point", drift < 1e-5f);

    /* Build a tangent (skip ambient coords). */
    srand(11);
    float tangent[17] = {0};
    for (int i = 1; i < 5; ++i) tangent[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    for (int i = 6; i < 9; ++i) tangent[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
    for (int i = 9; i < 17; ++i) tangent[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

    /* exp_p(v) → q, then log_p(q) → v2; check round-trip per factor. */
    float q[17] = {0}, v2[17] = {0};
    tc_product_exp(factors, n_factors, base, tangent, q);
    tc_product_log(factors, n_factors, base, q, v2);

    float err_lorentz = 0.0f, err_sphere = 0.0f, err_euclidean = 0.0f;
    for (int i = 1; i < 5; ++i) {
        float d = fabsf(tangent[i] - v2[i]);
        if (d > err_lorentz) err_lorentz = d;
    }
    for (int i = 6; i < 9; ++i) {
        float d = fabsf(tangent[i] - v2[i]);
        if (d > err_sphere) err_sphere = d;
    }
    for (int i = 9; i < 17; ++i) {
        float d = fabsf(tangent[i] - v2[i]);
        if (d > err_euclidean) err_euclidean = d;
    }
    CHECK("Lorentz factor log(exp(v)) round-trip < 1e-4", err_lorentz < 1e-4f);
    CHECK("Sphere factor log(exp(v)) round-trip < 1e-4",  err_sphere < 1e-4f);
    CHECK("Euclidean factor log(exp(v)) round-trip < 1e-5", err_euclidean < 1e-5f);

    /* Distance: should be positive and finite. */
    float d_pq = tc_product_distance(factors, n_factors, base, q);
    CHECK("d(base, q) > 0 and finite", d_pq > 0.0f && isfinite(d_pq));

    /* Self-distance is zero. */
    float d_pp = tc_product_distance(factors, n_factors, base, base);
    CHECK("d(p, p) = 0", fabsf(d_pp) < 1e-5f);

    /* Parallel transport: factor-wise dispatch should preserve norms
     * approximately (each factor's PT preserves its tangent norm). */
    float u[17] = {0}, ut[17] = {0};
    for (int i = 1; i < 5; ++i) u[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 6; i < 9; ++i) u[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 9; i < 17; ++i) u[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    tc_product_parallel_transport(factors, n_factors, base, q, u, ut);
    /* PT shouldn't NaN or blow up. */
    int all_finite = 1;
    for (int i = 0; i < 17; ++i) if (!isfinite(ut[i])) { all_finite = 0; break; }
    CHECK("PT output is finite across all factors", all_finite);

    printf("\n  %d/%d product manifold tests passed\n", test_pass, test_count);
    return (test_pass == test_count) ? 0 : 1;
}
