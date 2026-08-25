#ifndef TENSORCORE_DENSITY_MATRIX_H
#define TENSORCORE_DENSITY_MATRIX_H

/*
 * tensorcore — Density matrix (open quantum systems).
 *
 * A density matrix ρ on n qubits is a 2^n × 2^n complex Hermitian
 * positive-semidefinite matrix with tr(ρ) = 1. Stored as a flat fp32
 * array of 2 · 4^n floats (interleaved complex, row-major over the
 * computational basis). The (i, j) element lives at
 *   rho[2 * (i * 2^n + j)]      (real part)
 *   rho[2 * (i * 2^n + j) + 1]  (imag part).
 *
 * Pure-state limit: ρ = |ψ⟩⟨ψ| has rank 1, purity tr(ρ²) = 1. Mixed
 * states have purity ∈ [1/2^n, 1).
 *
 * Replaces moonlab's per-project density-matrix kernels for the
 * common open-systems primitives. Channel application via Kraus
 * operators covers depolarising, dephasing, amplitude-damping and
 * any other CP-trace-preserving channel.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of fp32 elements in an n-qubit density matrix: 2 · 4^n. */
static inline size_t tc_dmstate_size(int n_qubits) {
    const size_t dim = (size_t)1 << n_qubits;
    return 2 * dim * dim;
}

/* Initialise ρ = |0…0⟩⟨0…0|: single 1 at (0, 0), rest zero. */
void tc_dmstate_zero(float* rho, int n_qubits);

/* Construct ρ = |ψ⟩⟨ψ| from a state-vector amplitude array (length
 * 2 · 2^n interleaved complex, same layout as tc_qstate_*). */
void tc_dmstate_from_pure(float* rho, const float* psi, int n_qubits);

/* Apply a 2×2 unitary U to the given qubit: ρ → U_q ρ U_q†.
 * U is laid out as the standard 1-qubit gate matrix (8 floats). */
void tc_dmstate_apply_1q_unitary(float* rho, int n_qubits, int qubit,
                                  const float* U);

/* Apply a 4×4 unitary U to (q_ctrl, q_targ): ρ → U ρ U†.
 * U layout matches tc_qstate_apply_2q_unitary (4×4 row-major
 * interleaved complex, 32 floats). Basis order |q_ctrl q_targ⟩
 * = |00⟩,|01⟩,|10⟩,|11⟩ — same as the pure-state 2q convention. */
void tc_dmstate_apply_2q_unitary(float* rho, int n_qubits,
                                  int q_ctrl, int q_targ, const float* U);

/* Partial trace: trace out one qubit, returning the reduced ρ on
 * the remaining n_qubits-1 qubits. `rho_out` must be allocated with
 * tc_dmstate_size(n_qubits - 1) floats.
 *
 * Formula: tr_q(ρ)_{i, j} = Σ_{k∈{0,1}} ρ_{i ⊕ kq, j ⊕ kq}
 * where ⊕ kq sets bit `q` to k in the index.
 *
 * Qubit-index convention: bit 0 is LSB. The qubit being traced is
 * removed from the index; remaining qubits compact downward
 * (so a 3-qubit ρ tracing out qubit 1 yields a 2-qubit ρ with
 *  what were qubits {0, 2} now as qubits {0, 1}). */
void tc_dmstate_partial_trace(const float* rho_in, int n_qubits,
                               int trace_qubit, float* rho_out);

/* Apply a Kraus channel on a single qubit: ρ → Σ_k K_k ρ K_k†.
 * `kraus_ops` is a flat array of `n_kraus` 2×2 complex matrices
 * (each 8 floats, same layout as the unitary path). The caller is
 * responsible for ensuring Σ_k K_k† K_k = I (trace preservation). */
void tc_dmstate_apply_kraus_1q(float* rho, int n_qubits, int qubit,
                                const float* kraus_ops, int n_kraus);

/* trace(ρ) = Σ_i ρ_{ii}. For valid density matrices this is 1+0i.
 * Returns the (real, imag) parts. */
void tc_dmstate_trace(const float* rho, int n_qubits,
                       float* out_re, float* out_im);

/* Purity = tr(ρ²) = Σ_{ij} |ρ_{ij}|². For pure states = 1; for the
 * maximally mixed state = 1/2^n. */
float tc_dmstate_purity(const float* rho, int n_qubits);

/* ---- Lindblad evolution (open quantum systems) ---- *
 *
 * Master equation:
 *   dρ/dt = -i [H, ρ]
 *          + Σ_k (L_k ρ L_k† - ½ {L_k† L_k, ρ})
 *
 * Hamiltonian H is described by Pauli terms (same struct as
 * tc_qstate_trotter_step). Each first-order substep:
 *   1. Hamiltonian evolution: ρ → U(dt) ρ U(dt)† where U(dt) is a
 *      first-order Trotter approximation of exp(-iH dt). Implemented
 *      by walking the same Pauli-string decomposition the state-vector
 *      Trotter uses (basis change + CNOT staircase + Rz + reverse) and
 *      applying every gate to ρ via the density-matrix conjugation
 *      kernels — preserves Hermiticity and trace.
 *   2. Dissipator: for each jump operator L_k (single-qubit, 2×2
 *      complex, 8 floats), compute
 *        ρ ← ρ + dt (L_k ρ L_k† - ½ (L_k† L_k ρ + ρ L_k† L_k)).
 *      Single-qubit jumps cover the standard noise models
 *      (depolarising, dephasing, amplitude damping, etc.) — multi-
 *      qubit jump operators can be added later.
 *
 * First-order error is O(dt²); composable across substeps. Trace
 * preservation holds to O(dt²) (small deviation accumulates over
 * many substeps — caller can rescale by 1/tr(ρ) if needed). */

/* tc_pauli_term_t is defined in tensorcore/quantum_gates.h; include
 * here so callers don't need a separate include. */
#include "tensorcore/quantum_gates.h"

void tc_dmstate_lindblad_step(float* rho, int n_qubits,
                               const tc_pauli_term_t* H_terms,
                               int n_H_terms,
                               const float* jump_ops,
                               const int*   jump_qubits,
                               int n_jumps,
                               float t, int n_substeps);

#ifdef __cplusplus
}
#endif
#endif
