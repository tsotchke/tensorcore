#ifndef TENSORCORE_QUANTUM_ATTENTION_H
#define TENSORCORE_QUANTUM_ATTENTION_H

/*
 * tensorcore — Quantum attention + entanglement measures.
 *
 * The Born-rule output projection in phase_attention.h handles the
 * GeometricLM scoring path. This header covers the genuinely-quantum
 * pieces that QGTL and moonlab use for attention over n-qubit state
 * vectors:
 *
 *   - tc_quantum_attention_score       : Born-rule overlap |⟨ψ_Q|ψ_K⟩|²
 *                                        between two n-qubit pure states.
 *   - tc_quantum_attention_softmax     : per-row softmax over [N_q, N_k]
 *                                        attention scores → attention
 *                                        weights summing to 1 per query.
 *   - tc_quantum_attention_apply       : weighted sum Σ_k attn[q,k] · V[k]
 *                                        with attention weights and
 *                                        per-key value vectors V.
 *   - tc_quantum_entanglement_entropy  : von Neumann entropy
 *                                        S(ρ_A) = -tr(ρ_A log ρ_A)
 *                                        of a single-qubit reduction of
 *                                        an n-qubit state. Reuses
 *                                        density_matrix.h's partial
 *                                        trace + an in-line eigensolve.
 *
 * State convention (matches quantum_gates.h): each n-qubit state is
 * a 2*2^n-float array, interleaved complex (state[2k]=re, state[2k+1]=im).
 *
 * Why these belong here, not under generic attention:
 *   - The overlap and entropy are quantum primitives — they depend on
 *     the amplitude + phase, not just the magnitude.
 *   - The downstream consumers (QGTL phase circuits, moonlab quantum
 *     attention layers) expect Born-rule semantics — every other
 *     "attention" head in those models routes through here.
 *
 * fp32 IO + accumulators. CPU-first; Metal/CUDA dispatch is a v0.2
 * follow-up.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Born-rule overlap between two n-qubit pure-state vectors:
 *
 *   score = |⟨ψ_Q | ψ_K⟩|² = |Σ_i ψ_Q[i]* · ψ_K[i]|²
 *
 * Output is a real number in [0, 1]. Both state vectors MUST be
 * normalized for the result to be a proper probability; this function
 * does NOT renormalize internally.
 *
 * state_q, state_k : 2 * 2^n_qubits floats, interleaved complex. */
float tc_quantum_attention_score(const float* state_q,
                                  const float* state_k,
                                  int n_qubits);

/* Per-row softmax over a [N_q, N_k] score matrix:
 *
 *   attn[q, k] = exp(scores[q, k] / T) / Σ_k' exp(scores[q, k'] / T)
 *
 * with optional temperature `T` (pass T = 1.0f for the standard
 * softmax). Numerically stabilised via per-row max-subtract. fp32.
 *
 * scores_in, attn_out: [N_q, N_k] fp32, row-major. May alias. */
void tc_quantum_attention_softmax(const float* scores_in,
                                   float* attn_out,
                                   int n_q, int n_k,
                                   float temperature);

/* Quantum-attention apply step. Given:
 *
 *   attn  : [N_q, N_k] attention weights (softmax-normalized rows)
 *   V     : [N_k, D_v] per-key value vectors (real fp32; no
 *           interleaved complex — typical attention output projection
 *           lives in real space even when the upstream scores are
 *           Born-rule overlaps of quantum states)
 *   out   : [N_q, D_v] output = attn @ V
 *
 * fp32 GEMM in disguise — implemented in lib/ops/quantum_attention_cpu.cpp
 * so callers don't have to round-trip through tc_gemm for the small
 * [N_q, N_k] shapes typical of moonlab circuits (N_q, N_k ≤ 64). */
void tc_quantum_attention_apply(const float* attn,
                                 const float* values,
                                 float* out,
                                 int n_q, int n_k, int d_v);

/* Single-qubit von Neumann entropy of an n-qubit pure state.
 *
 *   S(ρ_q) = -λ_+ log₂ λ_+ - λ_- log₂ λ_-
 *
 * where λ_± are the eigenvalues of the 2×2 reduced density matrix
 * obtained by partial-tracing every qubit EXCEPT `qubit_keep`. For a
 * pure state |ψ⟩ this equals the entanglement entropy between
 * `qubit_keep` and the rest.
 *
 * Returns 0 for a product state, log₂ 2 = 1 for a maximally-entangled
 * state. Output is in bits (log base 2).
 *
 * state : 2 * 2^n_qubits floats, interleaved complex. */
float tc_quantum_entanglement_entropy(const float* state,
                                       int n_qubits,
                                       int qubit_keep);

#ifdef __cplusplus
}
#endif
#endif
