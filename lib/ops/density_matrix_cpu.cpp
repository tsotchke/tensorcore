/*
 * tensorcore — Density matrix (CPU implementation).
 *
 * Math reference: see include/tensorcore/density_matrix.h. Complex
 * arithmetic on interleaved (re, im) float pairs. The 1-qubit unitary
 * application uses the bit-twiddling identity ρ → U ρ U† by sweeping
 * 2×2 sub-blocks of the 2^n × 2^n density matrix. Tile shape per
 * 1-qubit op: for every pair (i_other, j_other) over the n-1 remaining
 * qubits, the four matrix entries
 *   ρ_{(0,i_o), (0,j_o)}, ρ_{(0,i_o), (1,j_o)},
 *   ρ_{(1,i_o), (0,j_o)}, ρ_{(1,i_o), (1,j_o)}
 * form a 2×2 sub-block that maps as
 *   ρ_block → U · ρ_block · U†.
 */

#include "tensorcore/density_matrix.h"
#include <cstring>
#include <vector>

namespace {

/* Complex multiply-add: out += a * b on interleaved pairs. */
inline void cmadd(const float* a, const float* b, float* out) {
    out[0] += a[0] * b[0] - a[1] * b[1];
    out[1] += a[0] * b[1] + a[1] * b[0];
}

/* Compute U† (the conjugate transpose) from a 1q unitary U
 * stored as 8 interleaved floats. */
inline void unitary_conj_transpose(const float* U, float* Ud) {
    /* (Ud)[r][c] = conj(U[c][r]) */
    /* U[r][c] re/im at U[4r + 2c], U[4r + 2c + 1]. */
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            Ud[4*r + 2*c]     =  U[4*c + 2*r];
            Ud[4*r + 2*c + 1] = -U[4*c + 2*r + 1];
        }
    }
}

}  // namespace

extern "C" void tc_dmstate_zero(float* rho, int n_qubits) {
    const size_t N = tc_dmstate_size(n_qubits);
    std::memset(rho, 0, N * sizeof(float));
    rho[0] = 1.0f;
    /* rho[1] (imag) already zero. */
}

extern "C" void tc_dmstate_from_pure(float* rho, const float* psi, int n_qubits) {
    const size_t dim = (size_t)1 << n_qubits;
    /* ρ_{i,j} = ψ_i · conj(ψ_j) = (a_i + i b_i)(a_j - i b_j). */
    for (size_t i = 0; i < dim; ++i) {
        const float ai = psi[2 * i];
        const float bi = psi[2 * i + 1];
        for (size_t j = 0; j < dim; ++j) {
            const float aj = psi[2 * j];
            const float bj = psi[2 * j + 1];
            const float re = ai * aj + bi * bj;
            const float im = bi * aj - ai * bj;
            rho[2 * (i * dim + j)]     = re;
            rho[2 * (i * dim + j) + 1] = im;
        }
    }
}

extern "C" void tc_dmstate_apply_1q_unitary(float* rho, int n_qubits,
                                             int qubit, const float* U) {
    /* Apply ρ → U · ρ · U† by sweeping 2×2 sub-blocks. */
    const size_t dim = (size_t)1 << n_qubits;
    const size_t step = (size_t)1 << qubit;
    const size_t stride2 = step << 1;

    float Ud[8];
    unitary_conj_transpose(U, Ud);

    /* Working memory: full row of `dim` complex values for the
     * intermediate B = U · ρ, processed row-wise then column-wise. */
    std::vector<float> tmp(2 * dim * dim);
    std::memcpy(tmp.data(), rho, 2 * dim * dim * sizeof(float));

    /* Step 1: rho ← U · ρ. For each column j, sweep pairs of rows
     * (i0, i1) differing only in `qubit`, and update them via U. */
    for (size_t j = 0; j < dim; ++j) {
        for (size_t base = 0; base < dim; base += stride2) {
            for (size_t off = 0; off < step; ++off) {
                const size_t i0 = base + off;
                const size_t i1 = i0 + step;
                const float a_re = tmp[2 * (i0 * dim + j)];
                const float a_im = tmp[2 * (i0 * dim + j) + 1];
                const float b_re = tmp[2 * (i1 * dim + j)];
                const float b_im = tmp[2 * (i1 * dim + j) + 1];
                /* new_a = U[0,0]·a + U[0,1]·b */
                rho[2 * (i0 * dim + j)]     = U[0]*a_re - U[1]*a_im + U[2]*b_re - U[3]*b_im;
                rho[2 * (i0 * dim + j) + 1] = U[0]*a_im + U[1]*a_re + U[2]*b_im + U[3]*b_re;
                /* new_b = U[1,0]·a + U[1,1]·b */
                rho[2 * (i1 * dim + j)]     = U[4]*a_re - U[5]*a_im + U[6]*b_re - U[7]*b_im;
                rho[2 * (i1 * dim + j) + 1] = U[4]*a_im + U[5]*a_re + U[6]*b_im + U[7]*b_re;
            }
        }
    }

    /* Step 2: rho ← (U·ρ) · U† = rho · U†. For each row i, sweep pairs
     * of cols (j0, j1) differing only in `qubit`, and update via U†. */
    std::memcpy(tmp.data(), rho, 2 * dim * dim * sizeof(float));
    for (size_t i = 0; i < dim; ++i) {
        for (size_t base = 0; base < dim; base += stride2) {
            for (size_t off = 0; off < step; ++off) {
                const size_t j0 = base + off;
                const size_t j1 = j0 + step;
                const float a_re = tmp[2 * (i * dim + j0)];
                const float a_im = tmp[2 * (i * dim + j0) + 1];
                const float b_re = tmp[2 * (i * dim + j1)];
                const float b_im = tmp[2 * (i * dim + j1) + 1];
                /* For row vector v multiplied by U† on the right:
                 *   (v · U†)[j0] = v[j0] · Ud[0,0] + v[j1] · Ud[1,0]
                 *   (v · U†)[j1] = v[j0] · Ud[0,1] + v[j1] · Ud[1,1]
                 * where Ud = U† has Ud[r][c] = conj(U[c][r]). */
                const float u00r = Ud[0], u00i = Ud[1];
                const float u01r = Ud[2], u01i = Ud[3];
                const float u10r = Ud[4], u10i = Ud[5];
                const float u11r = Ud[6], u11i = Ud[7];
                rho[2 * (i * dim + j0)]     = a_re*u00r - a_im*u00i + b_re*u10r - b_im*u10i;
                rho[2 * (i * dim + j0) + 1] = a_re*u00i + a_im*u00r + b_re*u10i + b_im*u10r;
                rho[2 * (i * dim + j1)]     = a_re*u01r - a_im*u01i + b_re*u11r - b_im*u11i;
                rho[2 * (i * dim + j1) + 1] = a_re*u01i + a_im*u01r + b_re*u11i + b_im*u11r;
            }
        }
    }
}

extern "C" void tc_dmstate_apply_2q_unitary(float* rho, int n_qubits,
                                             int q_ctrl, int q_targ,
                                             const float* U) {
    /* ρ → U · ρ · U† where U is 4×4 acting on (q_ctrl, q_targ).
     * Sweep 4×4 sub-blocks of ρ: for each (i_other, j_other) pair
     * over the remaining n-2 qubits, gather the 16 sub-amps into a
     * 4×4 complex matrix B, compute U·B·U†, and write back. */
    if (q_ctrl == q_targ) return;
    const size_t dim = (size_t)1 << n_qubits;
    const size_t mask_ct = ((size_t)1 << q_ctrl) | ((size_t)1 << q_targ);

    /* Build U† once. */
    float Ud[32];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            Ud[8*r + 2*c]     =  U[8*c + 2*r];
            Ud[8*r + 2*c + 1] = -U[8*c + 2*r + 1];
        }
    }

    auto sub_idx = [q_ctrl, q_targ](size_t other, int basis_k) {
        const int b_ctrl = (basis_k >> 1) & 1;
        const int b_targ = basis_k & 1;
        return other | ((size_t)b_ctrl << q_ctrl) | ((size_t)b_targ << q_targ);
    };

    /* Iterate over all (i_other, j_other) pairs of indices with bits
     * q_ctrl/q_targ cleared. */
    std::vector<size_t> others;
    others.reserve(dim >> 2);
    for (size_t i = 0; i < dim; ++i) {
        if ((i & mask_ct) == 0) others.push_back(i);
    }

    for (size_t i_other : others) {
        for (size_t j_other : others) {
            /* Gather the 4×4 complex sub-block B[4][4] from ρ. */
            float B[32];
            for (int r = 0; r < 4; ++r) {
                const size_t ri = sub_idx(i_other, r);
                for (int c = 0; c < 4; ++c) {
                    const size_t cj = sub_idx(j_other, c);
                    B[8*r + 2*c]     = rho[2 * (ri * dim + cj)];
                    B[8*r + 2*c + 1] = rho[2 * (ri * dim + cj) + 1];
                }
            }
            /* T = U · B (4×4 complex matmul). */
            float T[32] = {0};
            for (int r = 0; r < 4; ++r) {
                for (int k = 0; k < 4; ++k) {
                    const float ur_re = U[8*r + 2*k];
                    const float ur_im = U[8*r + 2*k + 1];
                    for (int c = 0; c < 4; ++c) {
                        const float br_re = B[8*k + 2*c];
                        const float br_im = B[8*k + 2*c + 1];
                        T[8*r + 2*c]     += ur_re*br_re - ur_im*br_im;
                        T[8*r + 2*c + 1] += ur_re*br_im + ur_im*br_re;
                    }
                }
            }
            /* W = T · U† (4×4). */
            float W[32] = {0};
            for (int r = 0; r < 4; ++r) {
                for (int k = 0; k < 4; ++k) {
                    const float tr_re = T[8*r + 2*k];
                    const float tr_im = T[8*r + 2*k + 1];
                    for (int c = 0; c < 4; ++c) {
                        const float ud_re = Ud[8*k + 2*c];
                        const float ud_im = Ud[8*k + 2*c + 1];
                        W[8*r + 2*c]     += tr_re*ud_re - tr_im*ud_im;
                        W[8*r + 2*c + 1] += tr_re*ud_im + tr_im*ud_re;
                    }
                }
            }
            /* Scatter W back into ρ. */
            for (int r = 0; r < 4; ++r) {
                const size_t ri = sub_idx(i_other, r);
                for (int c = 0; c < 4; ++c) {
                    const size_t cj = sub_idx(j_other, c);
                    rho[2 * (ri * dim + cj)]     = W[8*r + 2*c];
                    rho[2 * (ri * dim + cj) + 1] = W[8*r + 2*c + 1];
                }
            }
        }
    }
}

extern "C" void tc_dmstate_partial_trace(const float* rho_in, int n_qubits,
                                          int trace_qubit, float* rho_out) {
    /* tr_q(ρ)_{i, j} = Σ_{k∈{0,1}} ρ_{i⊕kq, j⊕kq}.
     * Output ρ has n_qubits - 1 qubits. The remaining qubits compact:
     * bit `q` is removed from the index, higher bits shift down by 1. */
    const int n_out = n_qubits - 1;
    if (n_out < 0) return;
    const size_t dim_in  = (size_t)1 << n_qubits;
    const size_t dim_out = (size_t)1 << n_out;
    const size_t mask_q  = (size_t)1 << trace_qubit;
    const size_t low_mask  = mask_q - 1;        /* bits below trace_qubit */
    const size_t high_mask = ~(2 * mask_q - 1); /* bits above trace_qubit */

    auto expand = [low_mask, trace_qubit](size_t i_out, int k) {
        /* Insert bit `k` at position trace_qubit. */
        const size_t lo = i_out & low_mask;
        const size_t hi = (i_out & ~low_mask) << 1;
        return lo | hi | ((size_t)k << trace_qubit);
    };
    (void)high_mask; (void)mask_q;

    std::memset(rho_out, 0, 2 * dim_out * dim_out * sizeof(float));
    for (size_t i = 0; i < dim_out; ++i) {
        for (size_t j = 0; j < dim_out; ++j) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < 2; ++k) {
                const size_t i_in = expand(i, k);
                const size_t j_in = expand(j, k);
                re += rho_in[2 * (i_in * dim_in + j_in)];
                im += rho_in[2 * (i_in * dim_in + j_in) + 1];
            }
            rho_out[2 * (i * dim_out + j)]     = (float)re;
            rho_out[2 * (i * dim_out + j) + 1] = (float)im;
        }
    }
}

extern "C" void tc_dmstate_apply_kraus_1q(float* rho, int n_qubits, int qubit,
                                           const float* kraus_ops, int n_kraus) {
    /* ρ' = Σ_k K_k ρ K_k†. Allocate a zero accumulator, then for each
     * Kraus operator compute K · ρ_orig · K† and add to the accumulator. */
    const size_t N = tc_dmstate_size(n_qubits);
    std::vector<float> rho_orig(N);
    std::memcpy(rho_orig.data(), rho, N * sizeof(float));
    std::vector<float> rho_acc(N, 0.0f);
    std::vector<float> rho_tmp(N);

    for (int k = 0; k < n_kraus; ++k) {
        const float* Kk = kraus_ops + 8 * k;
        std::memcpy(rho_tmp.data(), rho_orig.data(), N * sizeof(float));
        tc_dmstate_apply_1q_unitary(rho_tmp.data(), n_qubits, qubit, Kk);
        for (size_t i = 0; i < N; ++i) rho_acc[i] += rho_tmp[i];
    }
    std::memcpy(rho, rho_acc.data(), N * sizeof(float));
    (void)cmadd;  /* declared inline above; not needed in this routine */
}

extern "C" void tc_dmstate_trace(const float* rho, int n_qubits,
                                  float* out_re, float* out_im) {
    const size_t dim = (size_t)1 << n_qubits;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        re += rho[2 * (i * dim + i)];
        im += rho[2 * (i * dim + i) + 1];
    }
    *out_re = (float)re;
    *out_im = (float)im;
}

/* ---- Lindblad evolution helpers ----
 *
 * Hamiltonian Trotter on ρ: walk the standard Pauli-string evolution
 * decomposition (basis change + CNOT staircase + Rz + reverse) but
 * applying every gate to ρ via tc_dmstate_apply_*_unitary so the
 * transformation is the conjugation ρ → U ρ U†. This preserves
 * Hermiticity and trace.
 *
 * Dissipator: for each single-qubit jump L, compute
 *   ρ ← ρ + dt · (L ρ L† - ½ (L†L ρ + ρ L†L)).
 * Single-qubit L acts on one qubit; the L ρ L† term is exactly what
 * tc_dmstate_apply_1q_unitary computes (the kernel doesn't require L
 * to be unitary). For the anticommutator we hand-roll the
 * row-only and column-only multiplications. */

namespace {

void unitary_2x2_from_gate(tc_gate_type_t type, const float* params, float* U) {
    tc_gate_matrix_1q(type, params, U);
}

void rho_apply_1q_left(float* rho, int n_qubits, int qubit, const float* M) {
    /* ρ ← M ρ.  Row sweep with M (no right-multiply by M†). */
    const size_t dim = (size_t)1 << n_qubits;
    const size_t step = (size_t)1 << qubit;
    const size_t stride2 = step << 1;
    std::vector<float> tmp(2 * dim * dim);
    std::memcpy(tmp.data(), rho, 2 * dim * dim * sizeof(float));
    for (size_t j = 0; j < dim; ++j) {
        for (size_t base = 0; base < dim; base += stride2) {
            for (size_t off = 0; off < step; ++off) {
                const size_t i0 = base + off;
                const size_t i1 = i0 + step;
                const float a_re = tmp[2 * (i0 * dim + j)];
                const float a_im = tmp[2 * (i0 * dim + j) + 1];
                const float b_re = tmp[2 * (i1 * dim + j)];
                const float b_im = tmp[2 * (i1 * dim + j) + 1];
                rho[2 * (i0 * dim + j)]     = M[0]*a_re - M[1]*a_im + M[2]*b_re - M[3]*b_im;
                rho[2 * (i0 * dim + j) + 1] = M[0]*a_im + M[1]*a_re + M[2]*b_im + M[3]*b_re;
                rho[2 * (i1 * dim + j)]     = M[4]*a_re - M[5]*a_im + M[6]*b_re - M[7]*b_im;
                rho[2 * (i1 * dim + j) + 1] = M[4]*a_im + M[5]*a_re + M[6]*b_im + M[7]*b_re;
            }
        }
    }
}

void rho_apply_1q_right(float* rho, int n_qubits, int qubit, const float* M) {
    /* ρ ← ρ M.  Column sweep with M. */
    const size_t dim = (size_t)1 << n_qubits;
    const size_t step = (size_t)1 << qubit;
    const size_t stride2 = step << 1;
    std::vector<float> tmp(2 * dim * dim);
    std::memcpy(tmp.data(), rho, 2 * dim * dim * sizeof(float));
    for (size_t i = 0; i < dim; ++i) {
        for (size_t base = 0; base < dim; base += stride2) {
            for (size_t off = 0; off < step; ++off) {
                const size_t j0 = base + off;
                const size_t j1 = j0 + step;
                const float a_re = tmp[2 * (i * dim + j0)];
                const float a_im = tmp[2 * (i * dim + j0) + 1];
                const float b_re = tmp[2 * (i * dim + j1)];
                const float b_im = tmp[2 * (i * dim + j1) + 1];
                /* (v · M)[j0] = v[j0]·M[0,0] + v[j1]·M[1,0]
                 * (v · M)[j1] = v[j0]·M[0,1] + v[j1]·M[1,1]
                 * Using M[r,c] at M[4r + 2c], M[4r + 2c + 1]. */
                const float m00r = M[0], m00i = M[1];
                const float m01r = M[2], m01i = M[3];
                const float m10r = M[4], m10i = M[5];
                const float m11r = M[6], m11i = M[7];
                rho[2 * (i * dim + j0)]     = a_re*m00r - a_im*m00i + b_re*m10r - b_im*m10i;
                rho[2 * (i * dim + j0) + 1] = a_re*m00i + a_im*m00r + b_re*m10i + b_im*m10r;
                rho[2 * (i * dim + j1)]     = a_re*m01r - a_im*m01i + b_re*m11r - b_im*m11i;
                rho[2 * (i * dim + j1) + 1] = a_re*m01i + a_im*m01r + b_re*m11i + b_im*m11r;
            }
        }
    }
}

/* Compute L† L for a 2x2 complex L. Result is 2x2 Hermitian. */
void mat2x2_conj_dot(const float* L, float* M) {
    /* M = L† L. */
    float Ld[8];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            Ld[4*r + 2*c]     =  L[4*c + 2*r];
            Ld[4*r + 2*c + 1] = -L[4*c + 2*r + 1];
        }
    }
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            float re = 0.0f, im = 0.0f;
            for (int k = 0; k < 2; ++k) {
                const float a_re = Ld[4*r + 2*k];
                const float a_im = Ld[4*r + 2*k + 1];
                const float b_re = L [4*k + 2*c];
                const float b_im = L [4*k + 2*c + 1];
                re += a_re*b_re - a_im*b_im;
                im += a_re*b_im + a_im*b_re;
            }
            M[4*r + 2*c]     = re;
            M[4*r + 2*c + 1] = im;
        }
    }
}

/* Apply a single Pauli-string evolution to ρ (via density-matrix
 * conjugation kernels). This is the open-system analog of the
 * apply_pauli_evolution helper in quantum_gates_cpu.cpp. */
void rho_apply_pauli_evolution(float* rho, int n_qubits,
                                const int32_t* axes,
                                const int32_t* qubits,
                                int n_paulis, float alpha) {
    int active[32]; int n_active = 0;
    for (int i = 0; i < n_paulis; ++i) {
        if (axes[i] == TC_GATE_I) continue;
        active[n_active++] = i;
    }
    if (n_active == 0) {
        /* Global phase: U = e^{-iα} I → U ρ U† = ρ. No-op on ρ. */
        return;
    }
    float U[8];
    /* Step 1: basis change to Z. */
    for (int j = 0; j < n_active; ++j) {
        const int q = qubits[active[j]];
        if (axes[active[j]] == TC_GATE_X) {
            unitary_2x2_from_gate(TC_GATE_H, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
        } else if (axes[active[j]] == TC_GATE_Y) {
            unitary_2x2_from_gate(TC_GATE_SDG, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
            unitary_2x2_from_gate(TC_GATE_H, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
        }
    }
    /* Step 2: CNOT staircase. */
    const int parity_qubit = qubits[active[n_active - 1]];
    float U_cnot[32];
    tc_gate_matrix_2q(TC_GATE_CNOT, nullptr, U_cnot);
    for (int j = 0; j < n_active - 1; ++j) {
        tc_dmstate_apply_2q_unitary(rho, n_qubits,
                                     qubits[active[j]], parity_qubit, U_cnot);
    }
    /* Step 3: Rz(2α) on parity qubit. */
    const float two_alpha = 2.0f * alpha;
    unitary_2x2_from_gate(TC_GATE_RZ, &two_alpha, U);
    tc_dmstate_apply_1q_unitary(rho, n_qubits, parity_qubit, U);
    /* Step 4: reverse staircase. */
    for (int j = n_active - 2; j >= 0; --j) {
        tc_dmstate_apply_2q_unitary(rho, n_qubits,
                                     qubits[active[j]], parity_qubit, U_cnot);
    }
    /* Step 5: reverse basis change. */
    for (int j = n_active - 1; j >= 0; --j) {
        const int q = qubits[active[j]];
        if (axes[active[j]] == TC_GATE_X) {
            unitary_2x2_from_gate(TC_GATE_H, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
        } else if (axes[active[j]] == TC_GATE_Y) {
            unitary_2x2_from_gate(TC_GATE_H, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
            unitary_2x2_from_gate(TC_GATE_S, nullptr, U);
            tc_dmstate_apply_1q_unitary(rho, n_qubits, q, U);
        }
    }
}

}  // namespace

extern "C" void tc_dmstate_lindblad_step(float* rho, int n_qubits,
                                          const tc_pauli_term_t* H_terms,
                                          int n_H_terms,
                                          const float* jump_ops,
                                          const int* jump_qubits,
                                          int n_jumps,
                                          float t, int n_substeps) {
    if (n_substeps <= 0) return;
    const float dt = t / (float)n_substeps;
    const size_t N = tc_dmstate_size(n_qubits);
    std::vector<float> tmp_rho(N), L_rho_Ld(N), Ldd_rho(N), rho_Ldd(N);

    for (int step = 0; step < n_substeps; ++step) {
        /* Hamiltonian Trotter (one round through each Pauli term). */
        for (int k = 0; k < n_H_terms; ++k) {
            const tc_pauli_term_t* tk = &H_terms[k];
            const float alpha = tk->coef * dt;
            rho_apply_pauli_evolution(rho, n_qubits,
                                       tk->axes, tk->qubits, tk->n_paulis,
                                       alpha);
        }
        /* Dissipator. */
        for (int k = 0; k < n_jumps; ++k) {
            const float* Lk = jump_ops + 8 * k;
            const int q = jump_qubits[k];

            /* L ρ L† via the existing 1q apply (works for non-unitary L). */
            std::memcpy(L_rho_Ld.data(), rho, N * sizeof(float));
            tc_dmstate_apply_1q_unitary(L_rho_Ld.data(), n_qubits, q, Lk);

            /* L†L as a 2x2 matrix M. */
            float M_LdL[8];
            mat2x2_conj_dot(Lk, M_LdL);

            /* L†L · ρ (left-apply). */
            std::memcpy(Ldd_rho.data(), rho, N * sizeof(float));
            rho_apply_1q_left(Ldd_rho.data(), n_qubits, q, M_LdL);

            /* ρ · L†L (right-apply). */
            std::memcpy(rho_Ldd.data(), rho, N * sizeof(float));
            rho_apply_1q_right(rho_Ldd.data(), n_qubits, q, M_LdL);

            /* ρ += dt · (L ρ L† - ½ (L†L ρ + ρ L†L)). */
            const float half_dt = 0.5f * dt;
            for (size_t i = 0; i < N; ++i) {
                rho[i] += dt * L_rho_Ld[i]
                        - half_dt * (Ldd_rho[i] + rho_Ldd[i]);
            }
        }
    }
    (void)tmp_rho;
}

extern "C" float tc_dmstate_purity(const float* rho, int n_qubits) {
    /* tr(ρ²) for Hermitian ρ equals Σ_{i,j} |ρ_{i,j}|². */
    const size_t dim = (size_t)1 << n_qubits;
    double s = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            const float re = rho[2 * (i * dim + j)];
            const float im = rho[2 * (i * dim + j) + 1];
            s += (double)re * re + (double)im * im;
        }
    }
    return (float)s;
}
