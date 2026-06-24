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
        default:
            /* Unknown 2q gate: fill identity. */
            SET(0,0, 1,0); SET(1,1, 1,0); SET(2,2, 1,0); SET(3,3, 1,0);
            break;
    }
    (void)params;  /* No parameterized 2q gates yet. */
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

extern "C" void tc_qstate_apply_gate(float* state, int n_qubits,
                                      tc_gate_type_t type,
                                      const int* qubits,
                                      const float* params) {
    if (type >= TC_GATE_CNOT && type <= TC_GATE_SWAP) {
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
