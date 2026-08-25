/*
 * tensorcore — Holonomic gates (Berry phase via parallel transport).
 *
 * Math + ABI: include/tensorcore/holonomic.h.
 *
 * Composition uses tc_su2_exp + tc_su2_mul from lie_groups.h, so the
 * numerics inherit the Pauli-closed-form path (no Padé approximation).
 *
 * Berry phase extraction: the principal-branch phase of tr(U)/2.
 * For a z-aligned SU(2) generator exp(i (c σz)) the trace is
 * 2 cos(c), so the recovered phase equals c modulo 2π — exactly the
 * Berry phase. For non-aligned loops the trace still carries the
 * abelian holonomy (its argument), and the non-abelian SU(2) part is
 * recoverable by the caller from U directly.
 */

#include "tensorcore/holonomic.h"
#include "tensorcore/lie_groups.h"

#include <cmath>
#include <cstring>

extern "C" int tc_holonomic_compose_su2(const float* generators,
                                         int32_t n_segments,
                                         float* out_U) {
    /* Identity matrix initialiser, 2×2 row-major interleaved complex. */
    float acc[8] = {1, 0, 0, 0,
                    0, 0, 1, 0};
    if (n_segments < 0) return 1;
    for (int32_t s = 0; s < n_segments; ++s) {
        const float a = generators[3 * s + 0];
        const float b = generators[3 * s + 1];
        const float c = generators[3 * s + 2];
        float Useg[8];
        tc_su2_exp(a, b, c, Useg);
        float next[8];
        /* Left-multiply: acc ← Useg · acc, so that segment 0 is the
         * innermost (acts first on a state vector to the right). */
        tc_su2_mul(Useg, acc, next);
        std::memcpy(acc, next, sizeof(acc));
    }
    std::memcpy(out_U, acc, sizeof(acc));
    return 0;
}

extern "C" void tc_holonomic_berry_phase(const float* U, float* out_phase) {
    /* SU(2) holonomy angle.
     *
     * For U = exp(i v·σ) with ||v|| = θ, tr(U) = 2 cos(θ) is purely real
     * (the algebra is traceless), so the U(1) phase isn't atan2(im, re):
     * it's the rotation magnitude θ itself, which IS the geometric phase
     * picked up by a |+v̂⟩ eigenstate parallel-transported around the
     * closed loop.
     *
     * Layout: U[0..7] = row-major interleaved complex 2×2; U[0,0] = U[0]+iU[1],
     * U[1,1] = U[6]+iU[7]. tr(U)/2 = (U[0] + U[6]) / 2 + i (U[1] + U[7]) / 2.
     *
     * For SU(2), the trace's imag part is zero up to fp32 noise (det U = 1
     * forces tr U real). We extract θ as arccos(Re tr U / 2), with the sign
     * recovered from the imag part of U[0,0] (which carries the sign of v_z
     * for z-aligned generators — the only direction where the U(1) phase
     * unambiguously matches the algebra coefficient).
     */
    const float tr_re_half = 0.5f * (U[0] + U[6]);
    float cos_theta = tr_re_half;
    if (cos_theta >  1.0f) cos_theta =  1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;
    float theta = std::acos(cos_theta);
    /* Sign: U[0,0].imag = sin(θ) · v_z / θ; copy its sign onto θ so the
     * single-segment case exp(i c σ_z) → c (signed). */
    if (U[1] < 0.0f) theta = -theta;
    *out_phase = theta;
}
