#ifndef TENSORCORE_TORUS_H
#define TENSORCORE_TORUS_H

/*
 * tensorcore — Flat torus T^n = (R / 2πr Z)^n.
 *
 * Intrinsic coordinates: a point is an n-vector (θ_1, ..., θ_n) of
 * angles, each interpreted mod 2π · radius. The flat torus inherits
 * the Euclidean metric from R^n on its tangent space; geodesics are
 * straight lines in the universal cover (R^n) projected back to the
 * fundamental domain.
 *
 * Closed-form:
 *   exp_p(v) = (p + v) mod (2π r)              — tangent addition + wrap
 *   log_p(q) = shortest-displacement(q - p)    — accounts for winding
 *   d(p, q)² = Σ_i shortest(q_i - p_i)²        — Euclidean over wrapped coords
 *
 * `shortest-displacement(d)` picks the representative of d + 2πr Z
 * in (-π r, π r]. This is the right thing for periodic embeddings
 * (qLLM uses torus factors for circular features like wall-clock time
 * of day or phase angles).
 *
 * Parallel transport on the flat torus is identity in coordinates
 * (the connection is flat).
 *
 * Vectors are passed as flat float arrays of length `n` (intrinsic
 * dim — torus has no extra ambient coordinate, unlike sphere or
 * hyperboloid). Radius is a single float.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Project (wrap) angles into the fundamental domain [0, 2πr). */
void tc_torus_project(float* x, size_t n, float radius);

/* exp_p(v) = wrap(p + v). */
void tc_torus_exp(const float* base, const float* tangent,
                  float* point, size_t n, float radius);

/* log_p(q) = shortest displacement q - p (per coordinate, wrapped
 * into (-π r, π r]). */
void tc_torus_log(const float* base, const float* point,
                  float* tangent, size_t n, float radius);

/* Geodesic distance — L² of per-coordinate shortest displacements. */
float tc_torus_distance(const float* p, const float* q, size_t n,
                         float radius);

/* Parallel transport on the flat torus: identity in coordinates. */
void tc_torus_parallel_transport(const float* from, const float* to,
                                  const float* tangent, float* out,
                                  size_t n, float radius);

#ifdef __cplusplus
}
#endif
#endif
