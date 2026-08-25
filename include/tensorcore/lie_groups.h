#ifndef TENSORCORE_LIE_GROUPS_H
#define TENSORCORE_LIE_GROUPS_H

/*
 * tensorcore — Lie group exp/log for SU(2) and SO(3).
 *
 * Both groups admit closed-form matrix exponentials (no Padé approx
 * needed) because the algebra is 3-dimensional and the eigenvalues
 * of every algebra element are easy to compute.
 *
 * SU(2): 2×2 unitaries with det = 1. The algebra su(2) consists of
 * traceless anti-Hermitian 2×2 matrices, equivalent to a real
 * 3-vector v = (a, b, c) via H = i (a σ_x + b σ_y + c σ_z). Then
 *     exp(H) = cos(θ) I + i sin(θ)/θ · (a σ_x + b σ_y + c σ_z),
 *     θ = ||v||.
 *
 * SO(3): 3×3 rotation matrices with det = +1. The algebra so(3)
 * consists of skew-symmetric 3×3 matrices, equivalent to an axial
 * 3-vector ω = (ω_x, ω_y, ω_z). Rodrigues:
 *     R = I + sin(θ)/θ · K + (1 - cos(θ))/θ² · K²,
 *     K = [[0, -ω_z, ω_y], [ω_z, 0, -ω_x], [-ω_y, ω_x, 0]],
 *     θ = ||ω||.
 *
 * SU(2) is the double cover of SO(3); the bridge tc_su2_to_so3
 * converts a 2×2 unitary into the corresponding 3×3 rotation.
 *
 * Used by moonlab (holonomic gates via Berry phase parallel transport)
 * and qLLM (Lie group reductions on the H × S × R manifold).
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- SU(2) (2×2 complex unitaries with det=1) ---- *
 *
 * U is the standard 1-qubit gate layout: 8 floats, row-major
 * interleaved complex. The algebra vector v = (a, b, c) lives in R³. */

/* exp_{su(2)}((a, b, c)) → U (2×2 unitary). */
void tc_su2_exp(float a, float b, float c, float* U);

/* log_{SU(2)}(U) → (a, b, c) such that exp(i (a σ_x + b σ_y + c σ_z)) = U.
 * Branch: returns the principal log (θ ∈ [0, π]). Identity → (0,0,0). */
void tc_su2_log(const float* U, float* out_a, float* out_b, float* out_c);

/* Group multiplication: U · V. Both 2×2 unitaries in the standard layout. */
void tc_su2_mul(const float* U, const float* V, float* out_UV);

/* ---- SO(3) (3×3 real rotations) ---- *
 *
 * R is a 9-float array, row-major: R[3*i + j] = R_{ij}. */

/* Rodrigues exp: ω = (ω_x, ω_y, ω_z) → R (3×3 rotation). */
void tc_so3_exp(float wx, float wy, float wz, float* R);

/* Log of a rotation: R → ω, principal branch θ ∈ [0, π]. */
void tc_so3_log(const float* R, float* out_wx, float* out_wy, float* out_wz);

/* Rotation composition: R1 · R2. */
void tc_so3_mul(const float* R1, const float* R2, float* out_R12);

/* ---- Bridge: SU(2) → SO(3) (double cover) ---- *
 *
 * Maps a 2×2 SU(2) unitary to the 3×3 SO(3) rotation matrix it
 * implements on R³ via the adjoint representation. */
void tc_su2_to_so3(const float* U, float* R);

#ifdef __cplusplus
}
#endif
#endif
