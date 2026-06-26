/*
 * shim.h — single include surface for the TensorCore Swift system
 * library target. Pulls in only the headers Swift consumers actually
 * use (substrate ops; full tensorcore.h has too many internal types
 * for a tidy Swift import).
 */

#ifndef TC_SWIFT_SHIM_H
#define TC_SWIFT_SHIM_H

/* The TensorCore C ABI declares everything inside `extern "C"`. From
 * a Swift-system-library perspective we just need plain C
 * prototypes. We forward-declare the few opaque types Swift cares
 * about; the rest of the ABI is exposed through the included
 * headers. */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle. Declared as `void*` from Swift's perspective
 * (Swift's importer maps that to OpaquePointer). The real tc_context
 * is forward-declared inside libtensorcore; passing the void* through
 * the C ABI preserves the underlying pointer identity. */
typedef int tc_status_t;

int tc_init(void** out_ctx);
int tc_shutdown(void* ctx);
const char* tc_version(void);
const char* tc_status_string(int status);

/* ----- Geometric -----
 *
 * Same ABI as include/tensorcore/{lorentz,sphere,torus,lie_groups}.h */

void  tc_lorentz_exp(const float* base, const float* tangent,
                      float* point, unsigned long n, float curvature);
void  tc_lorentz_log(const float* base, const float* point,
                      float* tangent, unsigned long n, float curvature);
float tc_lorentz_distance(const float* p, const float* q,
                           unsigned long n, float curvature);
float tc_lorentz_minkowski(const float* a, const float* b, unsigned long n);

void  tc_sphere_exp(const float* base, const float* tangent,
                     float* point, unsigned long n, float radius);
void  tc_sphere_log(const float* base, const float* point,
                     float* tangent, unsigned long n, float radius);
float tc_sphere_distance(const float* p, const float* q,
                          unsigned long n, float radius);
void  tc_sphere_slerp(const float* p, const float* q, float t,
                       float* out, unsigned long n, float radius);

void  tc_torus_exp(const float* base, const float* tangent,
                    float* point, unsigned long n, float radius);
void  tc_torus_log(const float* base, const float* point,
                    float* tangent, unsigned long n, float radius);
float tc_torus_distance(const float* p, const float* q,
                         unsigned long n, float radius);

/* ----- Lie groups ----- */
void tc_su2_exp(float a, float b, float c, float* U);
void tc_su2_log(const float* U, float* out_a, float* out_b, float* out_c);
void tc_su2_mul(const float* U, const float* V, float* out);
void tc_so3_exp(float wx, float wy, float wz, float* R);
void tc_so3_log(const float* R, float* out_wx, float* out_wy, float* out_wz);
void tc_su2_to_so3(const float* U, float* R);

/* ----- Quantum state vector + gates ----- */
void  tc_qstate_zero(float* state, int n_qubits);
void  tc_qstate_apply_1q_unitary(float* state, int n_qubits, int target,
                                  const float* gate);
void  tc_qstate_apply_2q_unitary(float* state, int n_qubits, int qa, int qb,
                                  const float* gate);
float tc_qstate_prob_one(const float* state, int n_qubits, int qubit);
float tc_qstate_norm_sq(const float* state, int n_qubits);
void  tc_gate_matrix_1q(int gate_type, const float* params, float* out);
void  tc_gate_matrix_2q(int gate_type, const float* params, float* out);

/* ----- Quantum attention + entanglement ----- */
float tc_quantum_attention_score(const float* state_q, const float* state_k,
                                  int n_qubits);
void  tc_quantum_attention_softmax(const float* scores_in, float* attn_out,
                                    int n_q, int n_k, float temperature);
void  tc_quantum_attention_apply(const float* attn, const float* values,
                                  float* out, int n_q, int n_k, int d_v);
float tc_quantum_entanglement_entropy(const float* state, int n_qubits,
                                       int qubit_keep);

/* ----- Holonomic ----- */
int  tc_holonomic_compose_su2(const float* generators, int n_segments,
                               float* out_U);
void tc_holonomic_berry_phase(const float* U, float* out_phase);

#ifdef __cplusplus
}
#endif

#endif /* TC_SWIFT_SHIM_H */
