/*
 * tensorcore — Quantum gates + state-vector apply (CPU implementation).
 *
 * Math reference: see include/tensorcore/quantum_gates.h. Complex
 * arithmetic done on interleaved (re, im) float pairs. State vector
 * is 2*2^n floats (complex amplitudes); single-qubit application
 * sweeps 2^(n-1) bit-pair indices; two-qubit application sweeps
 * 2^(n-2) bit-quad indices.
 *
 * Conventions:
 *   - Qubit 0 is least significant.
 *   - Matrix layout: row-major interleaved complex.
 *   - Two-qubit basis order: |q_ctrl q_targ⟩ = |00⟩,|01⟩,|10⟩,|11⟩
 *     (q_targ is LSB of the 2-qubit pair).
 *   - sin/cos in fp32; gate matrices computed at fp32 throughout.
 *
 * Numerical safety: unitary evolution is exactly norm-preserving in
 * exact arithmetic. fp32 round-off drift is bounded by O(eps · depth).
 * The smoke test verifies ||state||² stays within 1e-5 of 1 after a
 * representative circuit (H × N + CNOT chain + measurement probe).
 */

#include "tensorcore/quantum_gates.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr float kInvSqrt2 = 0.70710678118654752440f;

/* Single complex multiply-add: out += a * b, on interleaved pairs. */
inline void cmadd(const float* a, const float* b, float* out) {
    /* (a.re + i a.im) * (b.re + i b.im)
     *   = (a.re b.re - a.im b.im) + i (a.re b.im + a.im b.re). */
    out[0] += a[0] * b[0] - a[1] * b[1];
    out[1] += a[0] * b[1] + a[1] * b[0];
}

/* Fill a 2x2 row-major interleaved-complex matrix. */
inline void set_1q(float* U,
                    float u00r, float u00i, float u01r, float u01i,
                    float u10r, float u10i, float u11r, float u11i) {
    U[0] = u00r; U[1] = u00i; U[2] = u01r; U[3] = u01i;
    U[4] = u10r; U[5] = u10i; U[6] = u11r; U[7] = u11i;
}

inline void set_4x4_zero(float* U) {
    std::memset(U, 0, 32 * sizeof(float));
}

}  // namespace

extern "C" void tc_gate_matrix_1q(tc_gate_type_t type, const float* params,
                                   float* U) {
    switch (type) {
        case TC_GATE_I:
            set_1q(U, 1,0, 0,0,  0,0, 1,0);
            break;
        case TC_GATE_X:
            set_1q(U, 0,0, 1,0,  1,0, 0,0);
            break;
        case TC_GATE_Y:
            /* Y = [[0, -i], [i, 0]] */
            set_1q(U, 0,0, 0,-1,  0,1, 0,0);
            break;
        case TC_GATE_Z:
            set_1q(U, 1,0, 0,0,  0,0, -1,0);
            break;
        case TC_GATE_H:
            set_1q(U,
                kInvSqrt2,0,  kInvSqrt2,0,
                kInvSqrt2,0, -kInvSqrt2,0);
            break;
        case TC_GATE_S:
            /* S = diag(1, i) */
            set_1q(U, 1,0, 0,0,  0,0, 0,1);
            break;
        case TC_GATE_T:
            /* T = diag(1, e^{iπ/4}) */
            set_1q(U, 1,0, 0,0,  0,0, kInvSqrt2,kInvSqrt2);
            break;
        case TC_GATE_SDG:
            set_1q(U, 1,0, 0,0,  0,0, 0,-1);
            break;
        case TC_GATE_TDG:
            set_1q(U, 1,0, 0,0,  0,0, kInvSqrt2,-kInvSqrt2);
            break;
        case TC_GATE_SX:
            /* √X = (1/2) [[1+i, 1-i], [1-i, 1+i]] */
            set_1q(U, 0.5f, 0.5f,  0.5f,-0.5f,
                       0.5f,-0.5f, 0.5f, 0.5f);
            break;
        case TC_GATE_RX: {
            /* Rx(θ) = [[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]] */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            set_1q(U, c,0,  0,-s,
                       0,-s, c,0);
            break;
        }
        case TC_GATE_RY: {
            /* Ry(θ) = [[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]] */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            set_1q(U, c,0, -s,0,
                       s,0,  c,0);
            break;
        }
        case TC_GATE_RZ: {
            /* Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]] */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            set_1q(U, c,-s, 0,0,
                       0,0,  c,s);
            break;
        }
        case TC_GATE_U1:
        case TC_GATE_PHASE: {
            /* U1(λ) = PHASE(λ) = diag(1, e^{iλ}) */
            const float lam = params[0];
            const float c = std::cos(lam), s = std::sin(lam);
            set_1q(U, 1,0, 0,0,
                       0,0, c,s);
            break;
        }
        case TC_GATE_U2: {
            /* U2(φ, λ) = (1/√2) [[1, -e^{iλ}],
             *                     [e^{iφ}, e^{i(φ+λ)}]] */
            const float phi = params[0], lam = params[1];
            const float cl = std::cos(lam), sl = std::sin(lam);
            const float cp = std::cos(phi), sp = std::sin(phi);
            const float cpl = std::cos(phi + lam), spl = std::sin(phi + lam);
            const float k = kInvSqrt2;
            set_1q(U,
                   k,0,             -k*cl, -k*sl,
                   k*cp,  k*sp,     k*cpl,  k*spl);
            break;
        }
        case TC_GATE_U3: {
            /* U3(θ, φ, λ) =
             *   [[cos(θ/2),               -e^{iλ} sin(θ/2)],
             *    [e^{iφ} sin(θ/2),  e^{i(φ+λ)} cos(θ/2)]] */
            const float t   = params[0] * 0.5f;
            const float phi = params[1];
            const float lam = params[2];
            const float ct = std::cos(t), st = std::sin(t);
            const float cl = std::cos(lam), sl = std::sin(lam);
            const float cp = std::cos(phi), sp = std::sin(phi);
            const float cpl = std::cos(phi + lam), spl = std::sin(phi + lam);
            set_1q(U,
                   ct, 0.0f,            -cl*st, -sl*st,
                   cp*st, sp*st,         cpl*ct,  spl*ct);
            break;
        }
        default:
            /* Unknown 1q gate: fill identity. */
            set_1q(U, 1,0, 0,0,  0,0, 1,0);
            break;
    }
}

extern "C" void tc_gate_matrix_2q(tc_gate_type_t type, const float* params,
                                   float* U) {
    set_4x4_zero(U);
    /* For row-major 4x4 interleaved-complex, element (i, j) starts at
     * U + 8*i + 2*j (i = row, j = col). */
    #define SET(i,j,re,im) do { U[8*(i) + 2*(j)] = (re); U[8*(i) + 2*(j) + 1] = (im); } while (0)
    switch (type) {
        case TC_GATE_CNOT:
            /* CNOT, basis order |00⟩|01⟩|10⟩|11⟩:
             *   |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩ */
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,3, 1,0);
            SET(3,2, 1,0);
            break;
        case TC_GATE_CY:
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,3, 0,-1);
            SET(3,2, 0, 1);
            break;
        case TC_GATE_CZ:
            SET(0,0,  1,0);
            SET(1,1,  1,0);
            SET(2,2,  1,0);
            SET(3,3, -1,0);
            break;
        case TC_GATE_SWAP:
            SET(0,0, 1,0);
            SET(1,2, 1,0);
            SET(2,1, 1,0);
            SET(3,3, 1,0);
            break;
        case TC_GATE_CRX: {
            /* Controlled Rx(θ): identity on |00⟩,|01⟩; Rx(θ) on the
             * (|10⟩, |11⟩) target-flip subspace. */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,2, c,0); SET(2,3, 0,-s);
            SET(3,2, 0,-s); SET(3,3, c,0);
            break;
        }
        case TC_GATE_CRY: {
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,2, c,0); SET(2,3, -s,0);
            SET(3,2, s,0); SET(3,3, c,0);
            break;
        }
        case TC_GATE_CRZ: {
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,2, c,-s);
            SET(3,3, c, s);
            break;
        }
        case TC_GATE_CH: {
            SET(0,0, 1,0);
            SET(1,1, 1,0);
            SET(2,2, kInvSqrt2,0); SET(2,3,  kInvSqrt2,0);
            SET(3,2, kInvSqrt2,0); SET(3,3, -kInvSqrt2,0);
            break;
        }
        case TC_GATE_ISWAP:
            /* iSWAP: swap |01⟩↔|10⟩ with phase i:
             *   [1, 0, 0, 0]
             *   [0, 0, i, 0]
             *   [0, i, 0, 0]
             *   [0, 0, 0, 1] */
            SET(0,0, 1,0);
            SET(1,2, 0,1);
            SET(2,1, 0,1);
            SET(3,3, 1,0);
            break;
        case TC_GATE_ZZ: {
            /* exp(-iθ/2 Z⊗Z) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2}). */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, c,-s);
            SET(1,1, c, s);
            SET(2,2, c, s);
            SET(3,3, c,-s);
            break;
        }
        case TC_GATE_XX: {
            /* exp(-iθ/2 X⊗X). c = cos(θ/2), s = sin(θ/2).
             *   [c, 0, 0, -is]
             *   [0, c, -is, 0]
             *   [0, -is, c, 0]
             *   [-is, 0, 0, c] */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, c,0);   SET(0,3, 0,-s);
            SET(1,1, c,0);   SET(1,2, 0,-s);
            SET(2,1, 0,-s);  SET(2,2, c,0);
            SET(3,0, 0,-s);  SET(3,3, c,0);
            break;
        }
        case TC_GATE_ECR: {
            /* Echoed Cross-Resonance (IBM native): up to phases equal to
             *   (1/√2) [[0, 1, 0, i],
             *           [1, 0, -i, 0],
             *           [0, i, 0, 1],
             *           [-i, 0, 1, 0]]
             * Built from RZX(π/2) + X⊗I + RZX(-π/2). */
            const float k = kInvSqrt2;
            SET(0,0, 0,0);   SET(0,1, k,0);   SET(0,2, 0,0);   SET(0,3, 0,k);
            SET(1,0, k,0);   SET(1,1, 0,0);   SET(1,2, 0,-k);  SET(1,3, 0,0);
            SET(2,0, 0,0);   SET(2,1, 0,k);   SET(2,2, 0,0);   SET(2,3, k,0);
            SET(3,0, 0,-k);  SET(3,1, 0,0);   SET(3,2, k,0);   SET(3,3, 0,0);
            break;
        }
        case TC_GATE_YY: {
            /* exp(-iθ/2 Y⊗Y).
             *   [c, 0, 0, +is]
             *   [0, c, -is, 0]
             *   [0, -is, c, 0]
             *   [+is, 0, 0, c] */
            const float t = params[0] * 0.5f;
            const float c = std::cos(t), s = std::sin(t);
            SET(0,0, c,0);   SET(0,3, 0, s);
            SET(1,1, c,0);   SET(1,2, 0,-s);
            SET(2,1, 0,-s);  SET(2,2, c,0);
            SET(3,0, 0, s);  SET(3,3, c,0);
            break;
        }
        default:
            /* Unknown 2q gate: fill identity. */
            SET(0,0, 1,0); SET(1,1, 1,0); SET(2,2, 1,0); SET(3,3, 1,0);
            break;
    }
    #undef SET
}

extern "C" void tc_qstate_zero(float* state, int n_qubits) {
    const size_t total = tc_qstate_size(n_qubits);
    std::memset(state, 0, total * sizeof(float));
    state[0] = 1.0f;  /* |0...0⟩ = 1 + 0i */
}

extern "C" void tc_qstate_apply_1q_unitary(float* state, int n_qubits,
                                            int qubit, const float* U) {
    const size_t N = (size_t)1 << n_qubits;       /* number of amplitudes */
    const size_t step = (size_t)1 << qubit;       /* stride to flip bit `qubit` */
    const size_t stride2 = step << 1;
    /* Sweep every 2-amp pair (idx0, idx1) that differ only in bit `qubit`. */
    for (size_t base = 0; base < N; base += stride2) {
        for (size_t off = 0; off < step; ++off) {
            const size_t i0 = base + off;
            const size_t i1 = i0 + step;
            float* a = state + 2 * i0;
            float* b = state + 2 * i1;
            const float ar = a[0], ai = a[1];
            const float br = b[0], bi = b[1];
            /* new_a = U[0,0]*a + U[0,1]*b */
            a[0] = U[0]*ar - U[1]*ai + U[2]*br - U[3]*bi;
            a[1] = U[0]*ai + U[1]*ar + U[2]*bi + U[3]*br;
            /* new_b = U[1,0]*a + U[1,1]*b */
            b[0] = U[4]*ar - U[5]*ai + U[6]*br - U[7]*bi;
            b[1] = U[4]*ai + U[5]*ar + U[6]*bi + U[7]*br;
        }
    }
}

extern "C" void tc_qstate_apply_2q_unitary(float* state, int n_qubits,
                                            int q_ctrl, int q_targ,
                                            const float* U) {
    /* For each amplitude index, the (ctrl, targ) bit pair takes one of
     * 4 values; index those by (b_ctrl, b_targ). The 4 amplitudes that
     * share all OTHER bits form one 2-qubit subsystem to which U applies.
     *
     * Iterate over the 2^(n-2) values of the "other bits" and apply U
     * to the 4 amplitudes (i_00, i_01, i_10, i_11) in each subsystem. */
    if (q_ctrl == q_targ) return;  /* invalid */
    const int q_lo = (q_ctrl < q_targ) ? q_ctrl : q_targ;
    const int q_hi = (q_ctrl < q_targ) ? q_targ : q_ctrl;
    const size_t step_lo = (size_t)1 << q_lo;
    const size_t step_hi = (size_t)1 << q_hi;
    const size_t N = (size_t)1 << n_qubits;
    /* Iterate over all amplitude indices with bits q_lo and q_hi cleared. */
    for (size_t i = 0; i < N; ++i) {
        if (i & step_lo) continue;
        if (i & step_hi) continue;
        /* Build the 4 indices in basis order |q_ctrl q_targ⟩. */
        const size_t i_ctrl0_targ0 = i;
        const size_t i_ctrl0_targ1 = i | ((size_t)1 << q_targ);
        const size_t i_ctrl1_targ0 = i | ((size_t)1 << q_ctrl);
        const size_t i_ctrl1_targ1 = i | ((size_t)1 << q_ctrl) | ((size_t)1 << q_targ);
        const size_t ids[4] = { i_ctrl0_targ0, i_ctrl0_targ1,
                                  i_ctrl1_targ0, i_ctrl1_targ1 };
        /* Pull amplitudes into a local 4-vector (complex). */
        float v[8];
        for (int k = 0; k < 4; ++k) {
            v[2*k]     = state[2 * ids[k]];
            v[2*k + 1] = state[2 * ids[k] + 1];
        }
        /* w = U @ v (4x4 complex matrix on 4-vec complex). */
        float w[8] = {0};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const float* u_rc = U + 8 * r + 2 * c;
                const float* v_c  = v + 2 * c;
                cmadd(u_rc, v_c, w + 2 * r);
            }
        }
        /* Write back. */
        for (int k = 0; k < 4; ++k) {
            state[2 * ids[k]]     = w[2*k];
            state[2 * ids[k] + 1] = w[2*k + 1];
        }
    }
}

/* Returns 1 if `type` is a 2-qubit gate, 0 otherwise. Updated whenever
 * a new 2-qubit gate enum value is added — this table is the dispatch
 * truth-source for tc_qstate_apply_gate. */
static int is_2q_gate(tc_gate_type_t t) {
    switch (t) {
        case TC_GATE_CNOT: case TC_GATE_CY:  case TC_GATE_CZ: case TC_GATE_SWAP:
        case TC_GATE_CRX:  case TC_GATE_CRY: case TC_GATE_CRZ: case TC_GATE_CH:
        case TC_GATE_ISWAP: case TC_GATE_ECR:
        case TC_GATE_XX:   case TC_GATE_YY:  case TC_GATE_ZZ:
            return 1;
        default:
            return 0;
    }
}

static int is_3q_gate(tc_gate_type_t t) {
    switch (t) {
        case TC_GATE_CCX: case TC_GATE_CSWAP:
            return 1;
        default:
            return 0;
    }
}

extern "C" void tc_qstate_apply_gate(float* state, int n_qubits,
                                      tc_gate_type_t type,
                                      const int* qubits,
                                      const float* params) {
    if (is_3q_gate(type)) {
        float U[128];
        tc_gate_matrix_3q(type, params, U);
        tc_qstate_apply_3q_unitary(state, n_qubits,
                                    qubits[0], qubits[1], qubits[2], U);
    } else if (is_2q_gate(type)) {
        float U[32];
        tc_gate_matrix_2q(type, params, U);
        tc_qstate_apply_2q_unitary(state, n_qubits, qubits[0], qubits[1], U);
    } else {
        float U[8];
        tc_gate_matrix_1q(type, params, U);
        tc_qstate_apply_1q_unitary(state, n_qubits, qubits[0], U);
    }
}

extern "C" float tc_qstate_prob_one(const float* state, int n_qubits, int qubit) {
    const size_t N = (size_t)1 << n_qubits;
    const size_t mask = (size_t)1 << qubit;
    float p = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        if (i & mask) {
            const float re = state[2 * i];
            const float im = state[2 * i + 1];
            p += re * re + im * im;
        }
    }
    return p;
}

extern "C" float tc_qstate_norm_sq(const float* state, int n_qubits) {
    const size_t N = (size_t)1 << n_qubits;
    float s = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        const float re = state[2 * i];
        const float im = state[2 * i + 1];
        s += re * re + im * im;
    }
    return s;
}

/* ---- 3-qubit gates ---- *
 *
 * For Toffoli/Fredkin (the only 3-qubit gates QGTL exposes), the unitary
 * is a permutation matrix — most entries are 0/1. The generic 8x8 mma
 * path handles arbitrary unitaries; we factor the matrix builders so
 * future parameterized 3-qubit gates (CCRZ, etc.) drop in cleanly. */

extern "C" void tc_gate_matrix_3q(tc_gate_type_t type, const float* params,
                                   float* U) {
    /* Zero the 8x8 matrix (128 floats). */
    std::memset(U, 0, 128 * sizeof(float));
    /* Element (i, j) starts at U + 16*i + 2*j (row-major, complex). */
    #define SET3(i, j, re, im) do { U[16*(i) + 2*(j)] = (re); \
                                      U[16*(i) + 2*(j) + 1] = (im); } while (0)
    switch (type) {
        case TC_GATE_CCX:
            /* Toffoli: identity except swap of |110⟩ ↔ |111⟩ (idx 6 ↔ 7).
             * Acts on the LSB qubit when both upper qubits are 1. */
            SET3(0,0, 1,0); SET3(1,1, 1,0); SET3(2,2, 1,0); SET3(3,3, 1,0);
            SET3(4,4, 1,0); SET3(5,5, 1,0);
            SET3(6,7, 1,0); SET3(7,6, 1,0);
            break;
        case TC_GATE_CSWAP:
            /* Fredkin: identity except swap of |101⟩ ↔ |110⟩ (idx 5 ↔ 6).
             * Swaps the two lower qubits when the top qubit (control) is 1. */
            SET3(0,0, 1,0); SET3(1,1, 1,0); SET3(2,2, 1,0); SET3(3,3, 1,0);
            SET3(4,4, 1,0); SET3(7,7, 1,0);
            SET3(5,6, 1,0); SET3(6,5, 1,0);
            break;
        default:
            /* Unknown 3q gate: identity. */
            for (int k = 0; k < 8; ++k) SET3(k, k, 1, 0);
            break;
    }
    (void)params;
    #undef SET3
}

extern "C" void tc_qstate_inner(const float* a, const float* b, int n_qubits,
                                 float* out_re, float* out_im) {
    /* <a|b> = Σ conj(a_i) · b_i */
    const size_t N = (size_t)1 << n_qubits;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const float ar = a[2 * i], ai = a[2 * i + 1];
        const float br = b[2 * i], bi = b[2 * i + 1];
        /* conj(a) · b = (ar - i ai)(br + i bi) = ar·br + ai·bi + i(ar·bi - ai·br) */
        re += (double)ar * br + (double)ai * bi;
        im += (double)ar * bi - (double)ai * br;
    }
    *out_re = (float)re;
    *out_im = (float)im;
}

/* Apply exp(-i α P) for a single Pauli string P given by `axes`/`qubits`.
 * Standard decomposition:
 *   1. Basis change: H on every X qubit, S† H on every Y qubit (mapping
 *      X→Z, Y→Z so the parity is collected on a Z basis).
 *   2. CNOT staircase from each active qubit into the "parity" qubit
 *      (chosen as the LAST entry in qubits[]).
 *   3. Rz(2α) on the parity qubit.
 *   4. Reverse CNOT staircase + reverse basis change.
 *
 * For identity-only terms (no X/Y/Z), evolution is a global phase
 * exp(-iα) — we just rescale every amplitude (real-rotate) and return. */
static void apply_pauli_evolution(float* state, int n_qubits,
                                   const int32_t* axes,
                                   const int32_t* qubits,
                                   int n_paulis, float alpha) {
    /* Filter out identity factors (TC_GATE_I = 0). */
    int active[32]; int n_active = 0;
    for (int i = 0; i < n_paulis; ++i) {
        if (axes[i] == TC_GATE_I) continue;
        active[n_active++] = i;
    }
    if (n_active == 0) {
        /* Pure global phase exp(-iα): multiply every amp by (cos α - i sin α). */
        const float c = std::cos(alpha), s = std::sin(alpha);
        const size_t N = (size_t)1 << n_qubits;
        for (size_t k = 0; k < N; ++k) {
            const float ar = state[2 * k], ai = state[2 * k + 1];
            state[2 * k]     = c * ar + s * ai;
            state[2 * k + 1] = c * ai - s * ar;
        }
        return;
    }

    /* Step 1: basis change so every active qubit is in Z basis. */
    float U_basis[8];
    for (int j = 0; j < n_active; ++j) {
        const int q = qubits[active[j]];
        if (axes[active[j]] == TC_GATE_X) {
            tc_gate_matrix_1q(TC_GATE_H, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
        } else if (axes[active[j]] == TC_GATE_Y) {
            /* Y = H · S → mapping Y → Z is achieved by S† then H.
             * Apply order is reverse for unitary on RIGHT (state acts on
             * |ψ⟩ left-to-right): we want the *new* basis to be Y's
             * eigenbasis, which is (S†)·H applied to the state. */
            tc_gate_matrix_1q(TC_GATE_SDG, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
            tc_gate_matrix_1q(TC_GATE_H, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
        }
        /* Z is already in Z basis — no change. */
    }

    /* Step 2: CNOT staircase from each active qubit into the parity qubit
     * (the last active qubit). */
    const int parity_qubit = qubits[active[n_active - 1]];
    float U_cnot[32];
    tc_gate_matrix_2q(TC_GATE_CNOT, nullptr, U_cnot);
    for (int j = 0; j < n_active - 1; ++j) {
        tc_qstate_apply_2q_unitary(state, n_qubits,
                                    qubits[active[j]], parity_qubit, U_cnot);
    }

    /* Step 3: Rz(2α) on the parity qubit. exp(-iα Z) = Rz(2α). */
    const float two_alpha = 2.0f * alpha;
    tc_gate_matrix_1q(TC_GATE_RZ, &two_alpha, U_basis);
    tc_qstate_apply_1q_unitary(state, n_qubits, parity_qubit, U_basis);

    /* Step 4: reverse CNOT staircase. */
    for (int j = n_active - 2; j >= 0; --j) {
        tc_qstate_apply_2q_unitary(state, n_qubits,
                                    qubits[active[j]], parity_qubit, U_cnot);
    }

    /* Step 5: reverse basis change. */
    for (int j = n_active - 1; j >= 0; --j) {
        const int q = qubits[active[j]];
        if (axes[active[j]] == TC_GATE_X) {
            tc_gate_matrix_1q(TC_GATE_H, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
        } else if (axes[active[j]] == TC_GATE_Y) {
            /* Reverse of (S† H) is (H† S) = (H · S). */
            tc_gate_matrix_1q(TC_GATE_H, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
            tc_gate_matrix_1q(TC_GATE_S, nullptr, U_basis);
            tc_qstate_apply_1q_unitary(state, n_qubits, q, U_basis);
        }
    }
}

extern "C" void tc_qstate_trotter_step(float* state, int n_qubits,
                                        const tc_pauli_term_t* terms,
                                        int n_terms,
                                        float t, int n_trotter_steps) {
    if (n_trotter_steps <= 0 || n_terms <= 0) return;
    const float dt = t / (float)n_trotter_steps;
    for (int step = 0; step < n_trotter_steps; ++step) {
        for (int k = 0; k < n_terms; ++k) {
            const tc_pauli_term_t* tk = &terms[k];
            const float alpha = tk->coef * dt;
            apply_pauli_evolution(state, n_qubits,
                                   tk->axes, tk->qubits,
                                   tk->n_paulis, alpha);
        }
    }
}

extern "C" void tc_quantum_geometric_tensor(const float* psi,
                                             const float* const* dpsi,
                                             int n_qubits, int n_params,
                                             float* out_G) {
    /* G_{ij} = <∂_i ψ | ∂_j ψ> - <∂_i ψ | ψ><ψ | ∂_j ψ> */
    /* Compute the n_params overlap scalars c_i = <ψ | ∂_i ψ> first;
     * <∂_i ψ | ψ> = conj(c_i). */
    std::vector<float> c_re(n_params), c_im(n_params);
    for (int i = 0; i < n_params; ++i) {
        tc_qstate_inner(psi, dpsi[i], n_qubits, &c_re[i], &c_im[i]);
    }
    /* Now fill G_{ij} = <∂_i ψ | ∂_j ψ> - conj(c_i) * c_j. */
    for (int i = 0; i < n_params; ++i) {
        for (int j = 0; j < n_params; ++j) {
            float inner_re, inner_im;
            tc_qstate_inner(dpsi[i], dpsi[j], n_qubits, &inner_re, &inner_im);
            /* conj(c_i) · c_j = (cr_i - i ci_i)(cr_j + i ci_j) */
            const float ci_cj_re = c_re[i] * c_re[j] + c_im[i] * c_im[j];
            const float ci_cj_im = c_re[i] * c_im[j] - c_im[i] * c_re[j];
            const size_t idx = (size_t)2 * ((size_t)i * n_params + j);
            out_G[idx]     = inner_re - ci_cj_re;
            out_G[idx + 1] = inner_im - ci_cj_im;
        }
    }
}

extern "C" void tc_qstate_apply_3q_unitary(float* state, int n_qubits,
                                            int q_a, int q_b, int q_c,
                                            const float* U) {
    if (q_a == q_b || q_a == q_c || q_b == q_c) return;  /* invalid */
    const size_t step_a = (size_t)1 << q_a;
    const size_t step_b = (size_t)1 << q_b;
    const size_t step_c = (size_t)1 << q_c;
    const size_t mask_abc = step_a | step_b | step_c;
    const size_t N = (size_t)1 << n_qubits;

    /* Iterate every amp index with bits q_a/q_b/q_c cleared.
     * For each, build the 8 sub-amp indices in unitary basis order
     *   |q_a q_b q_c⟩ = |000⟩, |001⟩, ..., |111⟩
     * and apply the 8x8 unitary. */
    for (size_t i = 0; i < N; ++i) {
        if (i & mask_abc) continue;
        size_t ids[8];
        for (int k = 0; k < 8; ++k) {
            const int qa_bit = (k >> 2) & 1;
            const int qb_bit = (k >> 1) & 1;
            const int qc_bit = (k     ) & 1;
            ids[k] = i
                   | ((size_t)qa_bit << q_a)
                   | ((size_t)qb_bit << q_b)
                   | ((size_t)qc_bit << q_c);
        }
        /* Pull amplitudes into a local 8-complex vector. */
        float v[16];
        for (int k = 0; k < 8; ++k) {
            v[2*k]     = state[2 * ids[k]];
            v[2*k + 1] = state[2 * ids[k] + 1];
        }
        /* w = U @ v (complex 8x8 on 8-complex vector). */
        float w[16] = {0};
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                const float ur_re = U[16*r + 2*c];
                const float ur_im = U[16*r + 2*c + 1];
                const float vc_re = v[2*c];
                const float vc_im = v[2*c + 1];
                w[2*r]     += ur_re * vc_re - ur_im * vc_im;
                w[2*r + 1] += ur_re * vc_im + ur_im * vc_re;
            }
        }
        /* Write back. */
        for (int k = 0; k < 8; ++k) {
            state[2 * ids[k]]     = w[2*k];
            state[2 * ids[k] + 1] = w[2*k + 1];
        }
    }
}
