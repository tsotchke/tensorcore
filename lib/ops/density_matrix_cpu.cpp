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
