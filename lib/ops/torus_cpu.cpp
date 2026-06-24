/*
 * tensorcore — Flat torus (CPU implementation).
 *
 * Math reference: see include/tensorcore/torus.h. All ops are per-
 * coordinate (the torus is a Cartesian product of circles). The only
 * subtlety is the shortest-displacement convention for log/distance:
 * pick the representative of (q - p) + 2πr Z in (-π r, π r].
 */

#include "tensorcore/torus.h"
#include <cmath>
#include <cstring>

namespace {

inline float wrap_to_period(float x, float period) {
    /* Maps x into [0, period). Uses fmod for stability with large inputs. */
    float w = std::fmod(x, period);
    if (w < 0.0f) w += period;
    return w;
}

inline float shortest_signed(float d, float period) {
    /* Map d into (-period/2, period/2]. */
    const float half = 0.5f * period;
    float w = std::fmod(d + half, period);
    if (w < 0.0f) w += period;
    return w - half;
}

}  // namespace

extern "C" void tc_torus_project(float* x, size_t n, float radius) {
    const float period = 6.28318530717958647692f * radius;  /* 2πr */
    for (size_t i = 0; i < n; ++i) {
        x[i] = wrap_to_period(x[i], period);
    }
}

extern "C" void tc_torus_exp(const float* base, const float* tangent,
                              float* point, size_t n, float radius) {
    const float period = 6.28318530717958647692f * radius;
    for (size_t i = 0; i < n; ++i) {
        point[i] = wrap_to_period(base[i] + tangent[i], period);
    }
}

extern "C" void tc_torus_log(const float* base, const float* point,
                              float* tangent, size_t n, float radius) {
    const float period = 6.28318530717958647692f * radius;
    for (size_t i = 0; i < n; ++i) {
        tangent[i] = shortest_signed(point[i] - base[i], period);
    }
}

extern "C" float tc_torus_distance(const float* p, const float* q, size_t n,
                                    float radius) {
    const float period = 6.28318530717958647692f * radius;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = shortest_signed(q[i] - p[i], period);
        s += (double)d * d;
    }
    return std::sqrt((float)s);
}

extern "C" void tc_torus_parallel_transport(const float* from, const float* to,
                                             const float* tangent, float* out,
                                             size_t n, float radius) {
    /* Flat connection: PT is the identity in coordinates. */
    (void)from; (void)to; (void)radius;
    if (out != tangent) std::memcpy(out, tangent, n * sizeof(float));
}
