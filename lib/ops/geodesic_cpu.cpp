/*
 * tensorcore — Geodesic ODE solver (RK4).
 *
 * Math + ABI: include/tensorcore/geodesic.h.
 *
 * The geodesic equation γ̈ + Γ γ̇γ̇ = 0 splits into a first-order
 * system on phase space:
 *
 *   ṗ = v
 *   v̇^k = -Γ^k_{ij}(p) v^i v^j
 *
 * We integrate that system with classical RK4. Each RK4 substep
 * requires the full Christoffel tensor at the substep position, which
 * we obtain from tc_metric_christoffel (numerical central-difference).
 * That makes a single step cost (d * 2 + 1) metric evaluations: 2d
 * from each Christoffel call (one + / one - per coordinate) plus the
 * baseline g(p) for the inverse, repeated 4× for RK4 = 4 · (2d + 1).
 * For d ≤ 64 manifolds (the regime qLLM/Noesis use) that's well under
 * a millisecond per step, fine for offline geodesic tracing and
 * gradient-flow holonomy.
 */

#include "tensorcore/geodesic.h"
#include "tensorcore/metric.h"

#include <cstring>
#include <vector>

namespace {

/* Acceleration v̇^k = -Γ^k_{ij}(p) v^i v^j. Returns 0 on success. */
int christoffel_accel(tc_metric_fn fn, void* user,
                       int d, float h_christ,
                       const float* p, const float* v,
                       float* a_out) {
    std::vector<float> Gamma(static_cast<size_t>(d) * d * d);
    if (tc_metric_christoffel(fn, user, p, d, h_christ, Gamma.data()) != 0) {
        return 1;
    }
    for (int k = 0; k < d; ++k) {
        double sum = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                sum += static_cast<double>(Gamma[k * d * d + i * d + j]) *
                       static_cast<double>(v[i]) *
                       static_cast<double>(v[j]);
            }
        }
        a_out[k] = static_cast<float>(-sum);
    }
    return 0;
}

}  // namespace

extern "C" int tc_geodesic_step(tc_metric_fn fn, void* user,
                                 int dim, float dt, float h_christoffel,
                                 const float* pos_in, const float* vel_in,
                                 float* pos_out, float* vel_out) {
    const int d = dim;
    std::vector<float> p1(d), v1(d), a1(d);
    std::vector<float> p2(d), v2(d), a2(d);
    std::vector<float> p3(d), v3(d), a3(d);
    std::vector<float> p4(d), v4(d), a4(d);

    /* k1 */
    std::memcpy(p1.data(), pos_in, sizeof(float) * d);
    std::memcpy(v1.data(), vel_in, sizeof(float) * d);
    if (christoffel_accel(fn, user, d, h_christoffel,
                          p1.data(), v1.data(), a1.data()) != 0) {
        return 1;
    }

    /* k2 = f(state + dt/2 · k1) */
    for (int i = 0; i < d; ++i) {
        p2[i] = p1[i] + 0.5f * dt * v1[i];
        v2[i] = v1[i] + 0.5f * dt * a1[i];
    }
    if (christoffel_accel(fn, user, d, h_christoffel,
                          p2.data(), v2.data(), a2.data()) != 0) {
        return 2;
    }

    /* k3 = f(state + dt/2 · k2) */
    for (int i = 0; i < d; ++i) {
        p3[i] = p1[i] + 0.5f * dt * v2[i];
        v3[i] = v1[i] + 0.5f * dt * a2[i];
    }
    if (christoffel_accel(fn, user, d, h_christoffel,
                          p3.data(), v3.data(), a3.data()) != 0) {
        return 3;
    }

    /* k4 = f(state + dt · k3) */
    for (int i = 0; i < d; ++i) {
        p4[i] = p1[i] + dt * v3[i];
        v4[i] = v1[i] + dt * a3[i];
    }
    if (christoffel_accel(fn, user, d, h_christoffel,
                          p4.data(), v4.data(), a4.data()) != 0) {
        return 4;
    }

    /* Combine: x ← x + (dt/6) (k1 + 2 k2 + 2 k3 + k4). */
    const float c = dt / 6.0f;
    for (int i = 0; i < d; ++i) {
        pos_out[i] = p1[i] + c * (v1[i] + 2.0f * v2[i] + 2.0f * v3[i] + v4[i]);
        vel_out[i] = v1[i] + c * (a1[i] + 2.0f * a2[i] + 2.0f * a3[i] + a4[i]);
    }
    return 0;
}

extern "C" int tc_geodesic_integrate(tc_metric_fn fn, void* user,
                                      int dim, float dt, int n_steps,
                                      float h_christoffel,
                                      const float* pos0, const float* vel0,
                                      float* pos_out, float* vel_out) {
    const int d = dim;
    std::vector<float> buf_a(d), buf_b(d);
    std::vector<float> vel_a(d), vel_b(d);
    std::memcpy(buf_a.data(), pos0, sizeof(float) * d);
    std::memcpy(vel_a.data(), vel0, sizeof(float) * d);
    float* p_in  = buf_a.data();
    float* p_out = buf_b.data();
    float* v_in  = vel_a.data();
    float* v_out = vel_b.data();
    for (int s = 0; s < n_steps; ++s) {
        const int rc = tc_geodesic_step(fn, user, d, dt, h_christoffel,
                                        p_in, v_in, p_out, v_out);
        if (rc != 0) return rc;
        std::swap(p_in, p_out);
        std::swap(v_in, v_out);
    }
    /* After the swap, p_in / v_in hold the latest values. */
    std::memcpy(pos_out, p_in, sizeof(float) * d);
    std::memcpy(vel_out, v_in, sizeof(float) * d);
    return 0;
}
