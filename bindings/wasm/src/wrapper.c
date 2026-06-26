/*
 * wrapper.c — thin C re-export layer for the WASM bindings.
 *
 * We compile the portable-CPU subset of libtensorcore (the same
 * sources as the portable-cpu CMake build path) and use this file
 * to mark exactly which symbols are kept in the wasm module's
 * export table. Anything that doesn't appear here gets dead-code
 * eliminated by emcc.
 *
 * The js wrapper (wrapper.js) provides typed-array conversions on
 * top of these raw float-pointer entrypoints — JS callers don't see
 * the malloc/free dance.
 */

#include <emscripten/emscripten.h>
#include "tensorcore/lorentz.h"
#include "tensorcore/sphere.h"
#include "tensorcore/torus.h"
#include "tensorcore/lie_groups.h"
#include "tensorcore/quantum_gates.h"
#include "tensorcore/quantum_attention.h"
#include "tensorcore/holonomic.h"

/* ---- Lorentz ---- */
EMSCRIPTEN_KEEPALIVE
void tc_wasm_lorentz_exp(const float* base, const float* tangent, float* point,
                          int n, float curvature) {
    tc_lorentz_exp(base, tangent, point, (size_t)n, curvature);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_lorentz_log(const float* base, const float* point, float* tangent,
                          int n, float curvature) {
    tc_lorentz_log(base, point, tangent, (size_t)n, curvature);
}
EMSCRIPTEN_KEEPALIVE
float tc_wasm_lorentz_distance(const float* p, const float* q, int n, float curvature) {
    return tc_lorentz_distance(p, q, (size_t)n, curvature);
}

/* ---- Sphere ---- */
EMSCRIPTEN_KEEPALIVE
void tc_wasm_sphere_exp(const float* base, const float* tangent, float* point,
                         int n, float radius) {
    tc_sphere_exp(base, tangent, point, (size_t)n, radius);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_sphere_log(const float* base, const float* point, float* tangent,
                         int n, float radius) {
    tc_sphere_log(base, point, tangent, (size_t)n, radius);
}
EMSCRIPTEN_KEEPALIVE
float tc_wasm_sphere_distance(const float* p, const float* q, int n, float radius) {
    return tc_sphere_distance(p, q, (size_t)n, radius);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_sphere_slerp(const float* p, const float* q, float t, float* out,
                           int n, float radius) {
    tc_sphere_slerp(p, q, t, out, (size_t)n, radius);
}

/* ---- Torus ---- */
EMSCRIPTEN_KEEPALIVE
void tc_wasm_torus_exp(const float* base, const float* tangent, float* point,
                        int n, float radius) {
    tc_torus_exp(base, tangent, point, (size_t)n, radius);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_torus_log(const float* base, const float* point, float* tangent,
                        int n, float radius) {
    tc_torus_log(base, point, tangent, (size_t)n, radius);
}
EMSCRIPTEN_KEEPALIVE
float tc_wasm_torus_distance(const float* p, const float* q, int n, float radius) {
    return tc_torus_distance(p, q, (size_t)n, radius);
}

/* ---- Lie groups ---- */
EMSCRIPTEN_KEEPALIVE
void tc_wasm_su2_exp(float a, float b, float c, float* U) {
    tc_su2_exp(a, b, c, U);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_su2_log(const float* U, float* abc_out) {
    tc_su2_log(U, abc_out + 0, abc_out + 1, abc_out + 2);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_so3_exp(float wx, float wy, float wz, float* R) {
    tc_so3_exp(wx, wy, wz, R);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_su2_to_so3(const float* U, float* R) {
    tc_su2_to_so3(U, R);
}

/* ---- Quantum state-vector + gates ---- */
EMSCRIPTEN_KEEPALIVE
void tc_wasm_qstate_zero(float* state, int n_qubits) {
    tc_qstate_zero(state, n_qubits);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_qstate_apply_1q(float* state, int n_qubits, int target, const float* gate) {
    tc_qstate_apply_1q_unitary(state, n_qubits, target, gate);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_qstate_apply_2q(float* state, int n_qubits, int qa, int qb, const float* gate) {
    tc_qstate_apply_2q_unitary(state, n_qubits, qa, qb, gate);
}
EMSCRIPTEN_KEEPALIVE
float tc_wasm_qstate_prob_one(const float* state, int n_qubits, int qubit) {
    return tc_qstate_prob_one(state, n_qubits, qubit);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_gate_1q(int gate_type, float* out) {
    tc_gate_matrix_1q((tc_gate_type_t)gate_type, NULL, out);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_gate_2q(int gate_type, float* out) {
    tc_gate_matrix_2q((tc_gate_type_t)gate_type, NULL, out);
}

/* ---- Quantum attention + entanglement ---- */
EMSCRIPTEN_KEEPALIVE
float tc_wasm_quantum_attention_score(const float* state_q, const float* state_k,
                                       int n_qubits) {
    return tc_quantum_attention_score(state_q, state_k, n_qubits);
}
EMSCRIPTEN_KEEPALIVE
float tc_wasm_quantum_entanglement_entropy(const float* state, int n_qubits,
                                            int qubit_keep) {
    return tc_quantum_entanglement_entropy(state, n_qubits, qubit_keep);
}

/* ---- Holonomic ---- */
EMSCRIPTEN_KEEPALIVE
int tc_wasm_holonomic_compose_su2(const float* gens, int n_seg, float* out_U) {
    return tc_holonomic_compose_su2(gens, n_seg, out_U);
}
EMSCRIPTEN_KEEPALIVE
void tc_wasm_holonomic_berry_phase(const float* U, float* out_phase) {
    tc_holonomic_berry_phase(U, out_phase);
}
