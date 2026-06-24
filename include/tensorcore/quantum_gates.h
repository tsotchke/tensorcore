#ifndef TENSORCORE_QUANTUM_GATES_H
#define TENSORCORE_QUANTUM_GATES_H

/*
 * tensorcore — Quantum gates + state-vector apply.
 *
 * State vector: a complex fp32 array of length 2^n_qubits, stored as
 * interleaved (re, im) float pairs (= 2*2^n floats). Bit ordering:
 * qubit 0 is least significant, so basis state |q_{n-1}...q_1 q_0⟩
 * lives at amplitude index Σ q_i · 2^i.
 *
 * Single-qubit unitary U (2x2) is laid out row-major as 4 complex
 * numbers = 8 floats: [U00.re, U00.im, U01.re, U01.im,
 *                       U10.re, U10.im, U11.re, U11.im].
 * Two-qubit U (4x4) is 16 complex = 32 floats in row-major order.
 *
 * Gate enum values match QGTL's gate_type_t (quantum_base_types.h) so
 * callers can pass their `gate_type_t` straight into tc_qstate_apply_gate
 * without translation. Pauli/H/S/T are 0..6; RX/RY/RZ are 7..9;
 * CNOT/CY/CZ/SWAP are 10..13. Beyond 13 is QGTL-specific extensions.
 *
 * Replaces the per-project gate kernels in QGTL
 * (src/quantum_geometric/core/quantum_gate_operations.c) and moonlab
 * (src/algorithms/) at the math substrate level.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gate type enum — numeric values match QGTL's gate_type_t (0..13 here). */
typedef enum {
    TC_GATE_I    = 0,  /* identity */
    TC_GATE_X    = 1,  /* Pauli X (NOT) */
    TC_GATE_Y    = 2,  /* Pauli Y */
    TC_GATE_Z    = 3,  /* Pauli Z */
    TC_GATE_H    = 4,  /* Hadamard */
    TC_GATE_S    = 5,  /* S = sqrt(Z) */
    TC_GATE_T    = 6,  /* T = sqrt(S) = ^4√Z */
    TC_GATE_RX   = 7,  /* Rx(θ) = exp(-iθ/2 X) */
    TC_GATE_RY   = 8,  /* Ry(θ) = exp(-iθ/2 Y) */
    TC_GATE_RZ   = 9,  /* Rz(θ) = exp(-iθ/2 Z) */
    TC_GATE_CNOT = 10, /* controlled-X */
    TC_GATE_CY   = 11, /* controlled-Y */
    TC_GATE_CZ   = 12, /* controlled-Z */
    TC_GATE_SWAP = 13, /* SWAP */
    /* 3-qubit gates (numeric values match QGTL gate_type_t). */
    TC_GATE_CCX   = 18, /* Toffoli (controlled-controlled-X) */
    TC_GATE_CSWAP = 20, /* Fredkin (controlled SWAP) */
    /* Controlled rotations (numeric values match QGTL gate_type_t). */
    TC_GATE_CRX  = 22, /* controlled Rx(θ) */
    TC_GATE_CRY  = 23, /* controlled Ry(θ) */
    TC_GATE_CRZ  = 24, /* controlled Rz(θ) */
    TC_GATE_CH   = 25, /* controlled Hadamard */
    TC_GATE_SDG  = 26, /* S† */
    TC_GATE_TDG  = 27, /* T† */
} tc_gate_type_t;

/* ---- Gate matrices ---- *
 *
 * Each helper fills `out` with the gate's unitary in row-major
 * interleaved-complex layout. For 1-qubit gates: out[8]. For 2-qubit:
 * out[32]. Caller owns the storage. */

void tc_gate_matrix_1q(tc_gate_type_t type, const float* params, float* out);
void tc_gate_matrix_2q(tc_gate_type_t type, const float* params, float* out);
/* 3-qubit gate matrix: out[128] (8x8 complex, row-major interleaved). */
void tc_gate_matrix_3q(tc_gate_type_t type, const float* params, float* out);

/* ---- State vector ---- */

/* Number of fp32 elements for an n-qubit state: 2 * 2^n. */
static inline size_t tc_qstate_size(int n_qubits) {
    return (size_t)2 * ((size_t)1 << n_qubits);
}

/* Initialize |0...0⟩ — state[0] = 1+0i, rest = 0. */
void tc_qstate_zero(float* state, int n_qubits);

/* Apply a 2x2 unitary U to the named qubit (single-thread; in-place). */
void tc_qstate_apply_1q_unitary(float* state, int n_qubits, int qubit,
                                 const float* U);

/* Apply a 4x4 unitary U to the (ctrl, targ) qubit pair (in-place).
 * The 4 basis vectors of the 2-qubit subspace are ordered
 *   |q_ctrl q_targ⟩ = |00⟩, |01⟩, |10⟩, |11⟩
 * — i.e. q_targ is the LSB within the pair. */
void tc_qstate_apply_2q_unitary(float* state, int n_qubits,
                                 int q_ctrl, int q_targ,
                                 const float* U);

/* Apply an 8x8 unitary to a (q_a, q_b, q_c) qubit triple (in-place).
 * The 8 basis vectors of the 3-qubit subspace are ordered
 *   |q_a q_b q_c⟩ = |000⟩, |001⟩, ..., |111⟩
 * with q_c the LSB of the triple (analogous to the 2q convention). */
void tc_qstate_apply_3q_unitary(float* state, int n_qubits,
                                 int q_a, int q_b, int q_c,
                                 const float* U);

/* High-level dispatch: build the matrix from (type, params), then apply.
 * For 1q gates `qubits` holds one index; for 2q gates two indices. */
void tc_qstate_apply_gate(float* state, int n_qubits,
                           tc_gate_type_t type,
                           const int* qubits,
                           const float* params);

/* ---- Measurement helpers ---- */

/* P(qubit = 1) = Σ |state[i]|² over indices where bit `qubit` is set. */
float tc_qstate_prob_one(const float* state, int n_qubits, int qubit);

/* ||state||² = Σ |state[i]|² over all amplitudes. Unitary evolution
 * preserves this at 1.0 modulo FP32 round-off. */
float tc_qstate_norm_sq(const float* state, int n_qubits);

#ifdef __cplusplus
}
#endif
#endif
