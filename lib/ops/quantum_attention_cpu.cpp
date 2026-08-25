/*
 * tensorcore — Quantum attention + entanglement-entropy CPU kernels.
 *
 * Math + ABI: include/tensorcore/quantum_attention.h.
 *
 * Born-rule overlap:
 *   <ψ_Q|ψ_K> = Σ_i conj(ψ_Q[i]) · ψ_K[i]
 *   = Σ_i (a_q - i b_q)(a_k + i b_k)
 *   = Σ_i (a_q a_k + b_q b_k) + i (a_q b_k - b_q a_k)
 *   score = |⟨…⟩|² = Re² + Im²
 *
 * Entanglement entropy: partial-trace down to a 2×2 reduced ρ,
 * eigendecompose, plug into S = -Σ λ log₂ λ. The 2×2 eigenproblem
 * has a closed-form solution (no LAPACK dependency).
 *
 * Softmax uses the standard max-subtract trick for fp32 stability.
 *
 * Apply step is a dense [N_q, N_k] × [N_k, D_v] GEMM done as a
 * triple loop; for the small attention shapes typical of moonlab
 * (N_q, N_k ≤ 64, D_v ≤ 256) this beats tc_gemm's setup overhead.
 */

#include "tensorcore/quantum_attention.h"
#include "tensorcore/density_matrix.h"

#include <cmath>
#include <cstring>
#include <vector>

extern "C" float tc_quantum_attention_score(const float* state_q,
                                             const float* state_k,
                                             int n_qubits) {
    const size_t dim = (size_t)1 << n_qubits;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const float aq = state_q[2 * i + 0];
        const float bq = state_q[2 * i + 1];
        const float ak = state_k[2 * i + 0];
        const float bk = state_k[2 * i + 1];
        re += static_cast<double>(aq) * ak + static_cast<double>(bq) * bk;
        im += static_cast<double>(aq) * bk - static_cast<double>(bq) * ak;
    }
    return static_cast<float>(re * re + im * im);
}

extern "C" void tc_quantum_attention_softmax(const float* scores_in,
                                              float* attn_out,
                                              int n_q, int n_k,
                                              float temperature) {
    const float inv_T = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    for (int q = 0; q < n_q; ++q) {
        const float* row_in  = scores_in + (size_t)q * n_k;
        float*       row_out = attn_out  + (size_t)q * n_k;
        /* max-subtract stabilisation */
        float row_max = row_in[0] * inv_T;
        for (int k = 1; k < n_k; ++k) {
            const float v = row_in[k] * inv_T;
            if (v > row_max) row_max = v;
        }
        double sum = 0.0;
        for (int k = 0; k < n_k; ++k) {
            const double e = std::exp(static_cast<double>(row_in[k] * inv_T - row_max));
            row_out[k] = static_cast<float>(e);
            sum += e;
        }
        const float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
        for (int k = 0; k < n_k; ++k) row_out[k] *= inv_sum;
    }
}

extern "C" void tc_quantum_attention_apply(const float* attn,
                                            const float* values,
                                            float* out,
                                            int n_q, int n_k, int d_v) {
    for (int q = 0; q < n_q; ++q) {
        float* out_row = out + (size_t)q * d_v;
        std::memset(out_row, 0, sizeof(float) * (size_t)d_v);
        for (int k = 0; k < n_k; ++k) {
            const float w = attn[(size_t)q * n_k + k];
            if (w == 0.0f) continue;
            const float* v_row = values + (size_t)k * d_v;
            for (int j = 0; j < d_v; ++j) out_row[j] += w * v_row[j];
        }
    }
}

namespace {

/* Closed-form eigenvalues of a 2×2 Hermitian matrix
 *   [[a, c], [conj(c), b]]   (a, b real; c complex)
 * λ_± = (a + b)/2 ± √((a-b)²/4 + |c|²).
 *
 * For a reduced density matrix from a pure state we know tr(ρ) = 1
 * and 0 ≤ λ_± ≤ 1, so the discriminant is non-negative.
 */
inline void eig_2x2_hermitian(double a, double b, double c_re, double c_im,
                               double* l_plus, double* l_minus) {
    const double half_sum = 0.5 * (a + b);
    const double half_diff = 0.5 * (a - b);
    const double off_sq = c_re * c_re + c_im * c_im;
    const double disc = std::sqrt(half_diff * half_diff + off_sq);
    *l_plus  = half_sum + disc;
    *l_minus = half_sum - disc;
}

inline double xlog2x_clamped(double x) {
    /* x log₂ x with 0 log 0 = 0 and a clamp on tiny negatives from fp32. */
    if (x <= 0.0) return 0.0;
    if (x > 1.0)  x = 1.0;
    return x * std::log2(x);
}

}  // namespace

extern "C" float tc_quantum_entanglement_entropy(const float* state,
                                                  int n_qubits,
                                                  int qubit_keep) {
    if (n_qubits <= 0 || qubit_keep < 0 || qubit_keep >= n_qubits) return 0.0f;

    /* Build the full density matrix ρ = |ψ⟩⟨ψ| (2^n × 2^n). */
    const size_t dim = (size_t)1 << n_qubits;
    const size_t rho_floats = 2 * dim * dim;
    std::vector<float> rho(rho_floats, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
        const float ai = state[2 * i + 0];
        const float bi = state[2 * i + 1];
        for (size_t j = 0; j < dim; ++j) {
            const float aj = state[2 * j + 0];
            const float bj = state[2 * j + 1];
            /* ρ_{ij} = ψ_i · conj(ψ_j) = (ai + i bi)(aj - i bj)
                     = (ai aj + bi bj) + i (bi aj - ai bj). */
            const size_t k = 2 * (i * dim + j);
            rho[k + 0] = ai * aj + bi * bj;
            rho[k + 1] = bi * aj - ai * bj;
        }
    }

    /* Successively partial-trace every qubit EXCEPT qubit_keep. After
     * each trace the qubit indices compact downward, so we have to
     * adjust qubit_keep accordingly. */
    int n = n_qubits;
    int keep = qubit_keep;
    while (n > 1) {
        /* Pick the lowest-index qubit that isn't `keep` to trace out. */
        int trace_q = (keep == 0) ? 1 : 0;
        std::vector<float> rho_out(2 * ((size_t)1 << (n - 1)) * ((size_t)1 << (n - 1)), 0.0f);
        tc_dmstate_partial_trace(rho.data(), n, trace_q, rho_out.data());
        rho = std::move(rho_out);
        if (trace_q < keep) keep -= 1;  /* keep's index drops by 1 */
        n -= 1;
    }

    /* rho is now a 2×2 reduced density matrix. ρ = [[a, c], [c*, b]]. */
    const double a   = rho[2 * (0 * 2 + 0) + 0];  /* ρ_{0,0}.re */
    const double b   = rho[2 * (1 * 2 + 1) + 0];  /* ρ_{1,1}.re */
    const double cre = rho[2 * (0 * 2 + 1) + 0];  /* ρ_{0,1}.re */
    const double cim = rho[2 * (0 * 2 + 1) + 1];  /* ρ_{0,1}.im */
    double l_plus, l_minus;
    eig_2x2_hermitian(a, b, cre, cim, &l_plus, &l_minus);

    const double S = -(xlog2x_clamped(l_plus) + xlog2x_clamped(l_minus));
    return static_cast<float>(S);
}
