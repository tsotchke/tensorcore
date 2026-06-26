#ifndef TENSORCORE_HOLONOMIC_H
#define TENSORCORE_HOLONOMIC_H

/*
 * tensorcore — Holonomic gates (Berry phase via parallel transport).
 *
 * A holonomic gate is the SU(2) unitary realised by adiabatically
 * driving a quantum system around a closed loop in some parameter
 * space. The loop's holonomy is purely geometric — it depends only
 * on the loop shape, not the parameterisation speed — which makes
 * holonomic gates intrinsically robust against timing noise.
 *
 * This API gives two equivalent surfaces:
 *
 *   tc_holonomic_compose_su2     : compose a discrete sequence of SU(2)
 *                                  generators (a, b, c)_k into a single
 *                                  loop unitary. Each generator is
 *                                  exp(i (a σx + b σy + c σz)), composed
 *                                  in order. Closing the loop means the
 *                                  vectors sum back to zero in algebra
 *                                  space.
 *
 *   tc_holonomic_berry_phase     : extract the abelian Berry phase
 *                                  (the U(1) part) of an SU(2) loop
 *                                  unitary U_loop. φ = arg(det U_loop)/2
 *                                  is the global phase picked up by
 *                                  parallel-transporting any state
 *                                  around the loop.
 *
 * The C surface is intentionally small — moonlab/QGTL hold the
 * physics-specific Hamiltonian-driving recipe; this gives them the
 * algebraic skeleton (Lie-group composition + abelian phase
 * extraction) without forcing them to re-derive SU(2) math.
 *
 * Uses tensorcore Lie-group primitives (include/tensorcore/lie_groups.h)
 * for SU(2) exp / mul and the standard 1-qubit gate layout (row-major
 * interleaved complex, 8 floats per unitary).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compose `n_segments` SU(2) generators into a loop unitary.
 * `generators` holds 3*n_segments floats: (a, b, c) per segment,
 * each producing exp(i (a σx + b σy + c σz)). The segment unitaries
 * are multiplied in order: U_loop = U_{n-1} · U_{n-2} · ... · U_0.
 * `out_U` is 8 floats (row-major interleaved complex).
 *
 * Closure check: caller is responsible for ensuring the algebra
 * vectors sum to zero (the loop closes). For abelian / commuting
 * generators that suffices; non-abelian closure is up to the caller's
 * physics — the holonomy is exactly the gate that comes out.
 *
 * Returns 0 on success. */
int tc_holonomic_compose_su2(const float* generators, int32_t n_segments,
                             float* out_U);

/* Abelian Berry phase of an SU(2) loop unitary.
 *
 * For U ∈ SU(2) of the form U = exp(iφ) · V with V ∈ SU(2) traceless-log,
 * φ is the global U(1) phase. From the 2×2 representation:
 *   U[0,0] = a + i b
 *   U[1,1] = a - i b  (when V is in the form exp(i v·σ))
 * we have det U = a² + b² · ... actually det U = 1 for SU(2) exactly,
 * so the abelian factor lives in U(2) ⊃ SU(2). For pure SU(2) loops
 * (det = 1), the Berry phase is encoded in the trace:
 *   tr(U_loop) / 2 = cos(θ),  arg(U[0,0]) = θ for the (z-axis-aligned) generator.
 * This function returns the principal-branch phase
 *   φ = atan2(Im tr(U)/2, Re tr(U)/2)
 * which equals the Berry phase modulo π for canonical SU(2) loops
 * driven by a z-aligned holonomy generator.
 *
 * out_phase ∈ (-π, π]. */
void tc_holonomic_berry_phase(const float* U, float* out_phase);

#ifdef __cplusplus
}
#endif
#endif
