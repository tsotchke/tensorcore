/*
 * test_sphere.c — sphere manifold correctness smoke.
 *
 * Verifies:
 *   - project() lands on the sphere (||x|| = r)
 *   - exp(0) = base (identity)
 *   - log(exp(v)) round-trip within fp32 floor
 *   - distance(p, p) = 0
 *   - distance(p, q) matches ||log_p(q)||
 *   - parallel transport preserves tangent norm
 *   - slerp midpoint stays on sphere
 */

#include "tensorcore/sphere.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int test_count = 0, test_pass = 0;
#define CHECK(label, cond) do { test_count++; \
    if (cond) { test_pass++; printf("  PASS %s\n", label); } \
    else      { printf("  FAIL %s\n", label); } } while (0)

int main(void) {
    const size_t n = 16;
    const float r = 1.5f;

    /* Base = "north pole" (r, 0, ..., 0). */
    float base[16] = {0};
    base[0] = r;

    /* Random tangent at base (skip e_0). */
    srand(7);
    float v[16] = {0};
    for (size_t i = 1; i < n; ++i) v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;

    /* exp_p(v) → q */
    float q[16] = {0};
    tc_sphere_exp(base, v, q, n, r);
    float qq = tc_sphere_inner(q, q, n);
    CHECK("exp lands on sphere (||q||² = r²)", fabsf(qq - r*r) < 1e-4f);

    /* log_p(q) → v2; check round-trip */
    float v2[16] = {0};
    tc_sphere_log(base, q, v2, n, r);
    float err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(v[i] - v2[i]);
        if (d > err) err = d;
    }
    CHECK("log(exp(v)) round-trip < 1e-5", err < 1e-5f);

    /* distance(p, p) = 0 */
    float d_pp = tc_sphere_distance(base, base, n, r);
    CHECK("d(p, p) = 0", fabsf(d_pp) < 1e-5f);

    /* distance(p, q) matches ||log|| */
    float d_pq = tc_sphere_distance(base, q, n, r);
    float log_norm = sqrtf(tc_sphere_inner(v2, v2, n));
    CHECK("d(p, q) == ||log_p(q)||", fabsf(d_pq - log_norm) < 1e-3f);

    /* PT preserves norm */
    float u[16] = {0};
    for (size_t i = 1; i < n; ++i) u[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    tc_sphere_tangent_project(base, u, n);
    float norm_before = sqrtf(tc_sphere_inner(u, u, n));
    float ut[16] = {0};
    tc_sphere_parallel_transport(base, q, u, ut, n, r);
    float norm_after = sqrtf(tc_sphere_inner(ut, ut, n));
    CHECK("PT preserves tangent norm",
          fabsf(norm_before - norm_after) / (norm_before + 1e-10f) < 5e-3f);

    /* slerp midpoint on sphere */
    float m[16] = {0};
    tc_sphere_slerp(base, q, 0.5f, m, n, r);
    float mm = tc_sphere_inner(m, m, n);
    CHECK("slerp(0.5) on sphere", fabsf(mm - r*r) < 1e-4f);

    printf("\n  %d/%d sphere tests passed\n", test_pass, test_count);
    return (test_pass == test_count) ? 0 : 1;
}
