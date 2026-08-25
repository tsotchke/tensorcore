/*
 * test_quantum_gates.c — quantum gate + state-vector apply smoke.
 *
 * Verifies:
 *   - Single-qubit gates (X, H, Z, RY) act correctly on |0⟩
 *   - Bell state preparation: H_0; CNOT_0,1 |00⟩ = (|00⟩ + |11⟩)/√2
 *   - Norm preservation across a representative 4-qubit circuit
 *   - Two-qubit SWAP: SWAP |10⟩ = |01⟩
 *   - RZ(π) on |+⟩ flips to -i|-⟩
 */

#include "tensorcore/quantum_gates.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int test_count = 0, test_pass = 0;
#define CHECK(label, cond) do { test_count++; \
    if (cond) { test_pass++; printf("  PASS %s\n", label); } \
    else      { printf("  FAIL %s\n", label); } } while (0)

#define APPROX(a, b, tol) (fabsf((a) - (b)) < (tol))

int main(void) {
    /* ===== Single qubit X on |0⟩ → |1⟩ ===== */
    {
        float state[4];  /* 1 qubit = 2 amps = 4 floats */
        tc_qstate_zero(state, 1);
        int qubits[1] = {0};
        tc_qstate_apply_gate(state, 1, TC_GATE_X, qubits, NULL);
        /* |1⟩ = (0, 1) = [(0,0), (1,0)] in re/im pairs */
        CHECK("X|0⟩ = |1⟩",
              APPROX(state[0], 0.0f, 1e-6f) &&
              APPROX(state[1], 0.0f, 1e-6f) &&
              APPROX(state[2], 1.0f, 1e-6f) &&
              APPROX(state[3], 0.0f, 1e-6f));
    }

    /* ===== Hadamard on |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2 ===== */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int qubits[1] = {0};
        tc_qstate_apply_gate(state, 1, TC_GATE_H, qubits, NULL);
        const float inv_sqrt2 = 0.70710678f;
        CHECK("H|0⟩ = |+⟩",
              APPROX(state[0], inv_sqrt2, 1e-5f) &&
              APPROX(state[2], inv_sqrt2, 1e-5f) &&
              APPROX(state[1], 0.0f, 1e-6f) &&
              APPROX(state[3], 0.0f, 1e-6f));
    }

    /* ===== RY(π) on |0⟩ → |1⟩ ===== */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int qubits[1] = {0};
        float params[1] = {(float)M_PI};
        tc_qstate_apply_gate(state, 1, TC_GATE_RY, qubits, params);
        CHECK("RY(π)|0⟩ ≈ |1⟩",
              APPROX(state[0], 0.0f, 1e-5f) &&
              APPROX(state[2], 1.0f, 1e-5f));
    }

    /* ===== Bell state: H_0 then CNOT_{0,1} on |00⟩ ===== */
    {
        float state[8];  /* 2 qubits = 4 amps = 8 floats */
        tc_qstate_zero(state, 2);
        int q0[1] = {0}, q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(state, 2, TC_GATE_CNOT, q01, NULL);
        const float inv_sqrt2 = 0.70710678f;
        /* Bell = (|00⟩ + |11⟩)/√2. Indices: |00⟩=0, |11⟩=3. */
        CHECK("Bell state: amp[|00⟩] = 1/√2",
              APPROX(state[0], inv_sqrt2, 1e-5f) &&
              APPROX(state[1], 0.0f, 1e-6f));
        CHECK("Bell state: amp[|01⟩] = 0",
              APPROX(state[2], 0.0f, 1e-6f) &&
              APPROX(state[3], 0.0f, 1e-6f));
        CHECK("Bell state: amp[|10⟩] = 0",
              APPROX(state[4], 0.0f, 1e-6f) &&
              APPROX(state[5], 0.0f, 1e-6f));
        CHECK("Bell state: amp[|11⟩] = 1/√2",
              APPROX(state[6], inv_sqrt2, 1e-5f) &&
              APPROX(state[7], 0.0f, 1e-6f));
        /* P(q0 = 1) should be 0.5 in the Bell state. */
        float p0 = tc_qstate_prob_one(state, 2, 0);
        float p1 = tc_qstate_prob_one(state, 2, 1);
        CHECK("Bell P(q0=1) = 0.5", APPROX(p0, 0.5f, 1e-5f));
        CHECK("Bell P(q1=1) = 0.5", APPROX(p1, 0.5f, 1e-5f));
    }

    /* ===== SWAP |10⟩ → |01⟩ ===== */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        /* Build |10⟩: amp index 2 (q1=1, q0=0). */
        state[0] = 0.0f;
        state[4] = 1.0f;  /* index 2 → offset 4 */
        int q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_SWAP, q01, NULL);
        /* SWAP gives |01⟩ (amp index 1 → offset 2). */
        CHECK("SWAP |10⟩ = |01⟩",
              APPROX(state[2], 1.0f, 1e-6f) &&
              APPROX(state[4], 0.0f, 1e-6f));
    }

    /* ===== CRZ(θ) on H ⊗ H |00⟩: controlled phase =====
     * After H_0 H_1 we have equal superposition (0.5, 0.5, 0.5, 0.5).
     * CRZ(π) with q_ctrl=0, q_targ=1 acts on amp indices where bit 0 = 1
     * (the control qubit). For those amps (idx 1 = q0=1,q1=0  and
     * idx 3 = q0=1,q1=1), the unitary's bottom-right 2x2 = diag(-i, +i)
     * applies: idx 1 (targ=0) → ×(-i), idx 3 (targ=1) → ×(+i).
     * Unchanged: idx 0 (q0=0,q1=0), idx 2 (q0=0,q1=1). */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        int q0[1] = {0}, q1[1] = {1}, q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q1, NULL);
        float params[1] = {(float)M_PI};
        tc_qstate_apply_gate(state, 2, TC_GATE_CRZ, q01, params);
        CHECK("CRZ(π): amp idx 0 = 0.5+0i",
              APPROX(state[0], 0.5f, 1e-5f) && APPROX(state[1], 0.0f, 1e-5f));
        CHECK("CRZ(π): amp idx 1 (ctrl=1,targ=0) = -0.5i",
              APPROX(state[2], 0.0f, 1e-5f) && APPROX(state[3], -0.5f, 1e-5f));
        CHECK("CRZ(π): amp idx 2 = 0.5+0i (unchanged)",
              APPROX(state[4], 0.5f, 1e-5f) && APPROX(state[5], 0.0f, 1e-5f));
        CHECK("CRZ(π): amp idx 3 (ctrl=1,targ=1) = +0.5i",
              APPROX(state[6], 0.0f, 1e-5f) && APPROX(state[7], 0.5f, 1e-5f));
    }

    /* ===== CRX(π) on amp index 1 (ctrl=1, targ=0):
     * The control bit (q_ctrl=0 = LSB) is set, so CRX rotates the
     * (|ctrl=1,targ=0⟩, |ctrl=1,targ=1⟩) = (amp idx 1, amp idx 3)
     * subspace as Rx(π) = -i X:
     *   new amp[1] = c·amp[1] - i s·amp[3] = 0·1 - i·1·0 = 0
     *   new amp[3] = -i s·amp[1] + c·amp[3] = -i·1·1 + 0 = -i  */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        /* Prepare amp index 1: state[2]=1, all else 0. */
        state[0] = 0.0f;
        state[2] = 1.0f;
        int q01[2] = {0, 1};
        float params[1] = {(float)M_PI};
        tc_qstate_apply_gate(state, 2, TC_GATE_CRX, q01, params);
        CHECK("CRX(π) ctrl=1: amp idx 1 ≈ 0",
              APPROX(state[2], 0.0f, 1e-5f) && APPROX(state[3], 0.0f, 1e-5f));
        CHECK("CRX(π) ctrl=1: amp idx 3 ≈ -i",
              APPROX(state[6], 0.0f, 1e-5f) && APPROX(state[7], -1.0f, 1e-5f));
    }

    /* ===== CH on H_0|00⟩ = (1/√2)(|q0=0⟩ + |q0=1⟩) ⊗ |q1=0⟩:
     * CH with q_ctrl=0, q_targ=1 applies H to q1 only when q0=1.
     *   |q0=0,q1=0⟩ (amp idx 0): ctrl=0, unchanged → 1/√2
     *   |q0=1,q1=0⟩ (amp idx 1): ctrl=1, H on q1 splits into
     *                              (|q0=1,q1=0⟩ + |q0=1,q1=1⟩)/√2
     *     so (1/√2)·(1/√2) = 0.5 goes to amp idx 1 AND amp idx 3.
     * Final: amp idx 0 = 1/√2, amp idx 1 = 0.5, amp idx 3 = 0.5. */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        int q0[1] = {0}, q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(state, 2, TC_GATE_CH, q01, NULL);
        const float inv_sqrt2 = 0.70710678f;
        CHECK("CH: amp idx 0 = 1/√2",
              APPROX(state[0], inv_sqrt2, 1e-5f));
        CHECK("CH: amp idx 1 = 0.5",
              APPROX(state[2], 0.5f, 1e-5f));
        CHECK("CH: amp idx 3 = 0.5",
              APPROX(state[6], 0.5f, 1e-5f));
    }

    /* ===== Toffoli (CCX) on |110⟩ (amp idx 3 with q0=1,q1=1,q2=0) =====
     * qubits = (q0, q1, q2). Unitary basis order |q_a q_b q_c⟩ where
     * q_c is LSB of the triple → we pass {q2, q1, q0} so that the
     * "target" qubit (LSB of unitary basis) is q0.
     * For |q0=1,q1=1,q2=0⟩ (state amp idx 3, offset 6), Toffoli flips
     * q0 (since q1=1 and q2=1... wait q2 is the upper control here.
     * Concretely: passing qubits=[q2,q1,q0] makes q_a=q2, q_b=q1, q_c=q0.
     * Unitary checks q_a=1 AND q_b=1, flips q_c.
     * For amp idx 3 (q2=0, q1=1, q0=1): q_a=0 → no flip → idx stays 3.
     * For amp idx 7 (q2=1, q1=1, q0=1): q_a=1 AND q_b=1 → flip q_c → amp idx 6.
     * So we prepare |111⟩ (amp idx 7, offset 14) and expect it to land
     * in amp idx 6 (offset 12). */
    {
        float state[16];   /* 3 qubits = 8 amps = 16 floats */
        tc_qstate_zero(state, 3);
        state[0] = 0.0f;
        state[14] = 1.0f;   /* prepare amp idx 7 = |q2=1,q1=1,q0=1⟩ */
        /* Pass qubits in unitary-basis order: q_a, q_b, q_c. q_c is the
         * target (gets flipped if both controls are 1). Use q_c = q0 (LSB),
         * q_b = q1, q_a = q2. */
        int qs[3] = {2, 1, 0};
        tc_qstate_apply_gate(state, 3, TC_GATE_CCX, qs, NULL);
        /* CCX flips q_c iff q_a AND q_b → state idx 7 → idx 6 (offset 12). */
        CHECK("Toffoli: amp idx 7 → 0", APPROX(state[14], 0.0f, 1e-6f));
        CHECK("Toffoli: amp idx 6 = 1",
              APPROX(state[12], 1.0f, 1e-6f) && APPROX(state[13], 0.0f, 1e-6f));
    }

    /* ===== Fredkin (CSWAP) on |101⟩ (q_a=1 → swap q_b ↔ q_c) =====
     * qubits passed as (q_a, q_b, q_c). For amp idx 5 (q_a=1, q_b=0, q_c=1):
     * q_a=1 triggers swap of q_b and q_c → result is |1,1,0⟩ = amp idx 6.
     * For amp idx 3 (q_a=0, q_b=1, q_c=1): q_a=0 → no swap → stays idx 3. */
    {
        float state[16];
        tc_qstate_zero(state, 3);
        state[0] = 0.0f;
        state[10] = 1.0f;  /* prepare amp idx 5 (q_a=1, q_b=0, q_c=1) */
        int qs[3] = {2, 1, 0};
        tc_qstate_apply_gate(state, 3, TC_GATE_CSWAP, qs, NULL);
        /* Fredkin swaps amp idx 5 ↔ idx 6 (offset 12). */
        CHECK("Fredkin: amp idx 5 → 0", APPROX(state[10], 0.0f, 1e-6f));
        CHECK("Fredkin: amp idx 6 = 1",
              APPROX(state[12], 1.0f, 1e-6f) && APPROX(state[13], 0.0f, 1e-6f));
        /* Norm preserved (unitary). */
        float n2 = tc_qstate_norm_sq(state, 3);
        CHECK("Fredkin preserves ||state||²", APPROX(n2, 1.0f, 1e-6f));
    }

    /* ===== ISWAP |10⟩ → i|01⟩ (and |01⟩ → i|10⟩) =====
     * |10⟩ has amp idx 2 (offset 4). ISWAP gives i|01⟩ → amp idx 1
     * (offset 2) with value (0, 1). */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        state[0] = 0.0f; state[4] = 1.0f;  /* |10⟩ */
        int q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_ISWAP, q01, NULL);
        /* ISWAP swaps the (|01⟩, |10⟩) subspace with phase i.
         * |10⟩ → i|01⟩: amp idx 1 = (re=0, im=1). */
        CHECK("ISWAP |10⟩ → i|01⟩: amp idx 2 = 0",
              APPROX(state[4], 0.0f, 1e-6f) && APPROX(state[5], 0.0f, 1e-6f));
        CHECK("ISWAP |10⟩ → i|01⟩: amp idx 1 = i",
              APPROX(state[2], 0.0f, 1e-6f) && APPROX(state[3], 1.0f, 1e-6f));
    }

    /* ===== U1(π) on |+⟩ = (|0⟩+|1⟩)/√2: phase on |1⟩ only =====
     * U1(π) = diag(1, e^{iπ}) = diag(1, -1). So |+⟩ → (|0⟩-|1⟩)/√2 = |-⟩. */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int q0[1] = {0};
        tc_qstate_apply_gate(state, 1, TC_GATE_H, q0, NULL);
        float params[1] = {(float)M_PI};
        tc_qstate_apply_gate(state, 1, TC_GATE_U1, q0, params);
        const float inv_sqrt2 = 0.70710678f;
        /* |-⟩ = (1/√2)|0⟩ - (1/√2)|1⟩ — amp[0] = +1/√2, amp[1] = -1/√2. */
        CHECK("U1(π) on |+⟩: amp[0] = 1/√2",
              APPROX(state[0], inv_sqrt2, 1e-5f));
        CHECK("U1(π) on |+⟩: amp[1] = -1/√2",
              APPROX(state[2], -inv_sqrt2, 1e-5f));
    }

    /* ===== U3(π, 0, 0) on |0⟩ ≈ |1⟩ (up to global phase) =====
     * U3(θ=π, φ=0, λ=0) = [[cos(π/2), -sin(π/2)], [sin(π/2), cos(π/2)]]
     * = [[0, -1], [1, 0]] — applied to |0⟩=(1,0) gives (0, 1) = |1⟩. */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int q0[1] = {0};
        float params[3] = {(float)M_PI, 0.0f, 0.0f};
        tc_qstate_apply_gate(state, 1, TC_GATE_U3, q0, params);
        CHECK("U3(π,0,0) on |0⟩ ≈ |1⟩",
              APPROX(state[0], 0.0f, 1e-5f) && APPROX(state[2], 1.0f, 1e-5f));
    }

    /* ===== ZZ(π) on H_0 H_1 |00⟩ — Ising rotation =====
     * Equal superposition (0.5, 0.5, 0.5, 0.5). ZZ(π) applies the diagonal
     * diag(e^{-iπ/2}, e^{iπ/2}, e^{iπ/2}, e^{-iπ/2}) = diag(-i, +i, +i, -i).
     * So amp[0] = 0.5·(-i), amp[1] = 0.5·(+i), amp[2] = 0.5·(+i), amp[3] = 0.5·(-i). */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        int q0[1] = {0}, q1[1] = {1}, q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q1, NULL);
        float params[1] = {(float)M_PI};
        tc_qstate_apply_gate(state, 2, TC_GATE_ZZ, q01, params);
        CHECK("ZZ(π) on |++⟩: amp[0] = -0.5i",
              APPROX(state[0], 0.0f, 1e-5f) && APPROX(state[1], -0.5f, 1e-5f));
        CHECK("ZZ(π) on |++⟩: amp[1] = +0.5i",
              APPROX(state[2], 0.0f, 1e-5f) && APPROX(state[3], 0.5f, 1e-5f));
        CHECK("ZZ(π) on |++⟩: amp[2] = +0.5i",
              APPROX(state[4], 0.0f, 1e-5f) && APPROX(state[5], 0.5f, 1e-5f));
        CHECK("ZZ(π) on |++⟩: amp[3] = -0.5i",
              APPROX(state[6], 0.0f, 1e-5f) && APPROX(state[7], -0.5f, 1e-5f));
    }

    /* ===== SX (√X) on |0⟩: applying SX twice should give X|0⟩ = |1⟩ =====
     * SX² = X. Apply twice and verify we land at |1⟩ up to global phase. */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int q0[1] = {0};
        tc_qstate_apply_gate(state, 1, TC_GATE_SX, q0, NULL);
        tc_qstate_apply_gate(state, 1, TC_GATE_SX, q0, NULL);
        /* SX² = X up to global phase i. So result is i|1⟩, which means
         * state[0..1] ≈ 0 and state[2..3] ≈ (0, 1). */
        float prob_one = state[2]*state[2] + state[3]*state[3];
        CHECK("SX²|0⟩ probability on |1⟩ ≈ 1.0", APPROX(prob_one, 1.0f, 1e-5f));
        float prob_zero = state[0]*state[0] + state[1]*state[1];
        CHECK("SX²|0⟩ probability on |0⟩ ≈ 0", APPROX(prob_zero, 0.0f, 1e-5f));
    }

    /* ===== ECR: unitary check via ECR · ECR† ≈ I =====
     * ECR is its own inverse up to a global phase (ECR² is a phase
     * times identity), so just verify norm preservation on a random
     * 2-qubit input. */
    {
        float state[8];
        tc_qstate_zero(state, 2);
        /* Mix: H on q0 then ECR. */
        int q0[1] = {0}, q01[2] = {0, 1};
        tc_qstate_apply_gate(state, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(state, 2, TC_GATE_ECR, q01, NULL);
        float n2 = tc_qstate_norm_sq(state, 2);
        CHECK("ECR preserves ||state||² = 1", APPROX(n2, 1.0f, 1e-5f));
    }

    /* ===== Trotter step: exp(-i (π/2) Z) on |+⟩ → -i|-⟩ =====
     * H_0 |0⟩ = |+⟩. Then apply Trotter evolution of H = Z with t = π/2,
     * 1 Trotter step. exp(-i (π/2) Z) = Rz(π) = diag(e^{-iπ/2}, e^{iπ/2})
     *                                         = diag(-i, +i).
     * On |+⟩ = (|0⟩+|1⟩)/√2 → (-i|0⟩ + i|1⟩)/√2 = -i · (|0⟩ - |1⟩)/√2
     *   = -i|-⟩. So state[0]=0, state[1]=-1/√2, state[2]=0, state[3]=+1/√2. */
    {
        float state[4];
        tc_qstate_zero(state, 1);
        int q0[1] = {0};
        tc_qstate_apply_gate(state, 1, TC_GATE_H, q0, NULL);
        /* Single Pauli term: Z on qubit 0, coefficient 1. */
        int32_t axes[1] = { TC_GATE_Z };
        int32_t qubits[1] = { 0 };
        tc_pauli_term_t term = {1, axes, qubits, 1.0f};
        tc_qstate_trotter_step(state, 1, &term, 1, (float)M_PI/2.0f, 1);
        const float inv_sqrt2 = 0.70710678f;
        CHECK("Trotter Z on |+⟩: amp[0] = 0-i/√2",
              APPROX(state[0], 0.0f, 1e-5f) && APPROX(state[1], -inv_sqrt2, 1e-5f));
        CHECK("Trotter Z on |+⟩: amp[1] = 0+i/√2",
              APPROX(state[2], 0.0f, 1e-5f) && APPROX(state[3], inv_sqrt2, 1e-5f));
        float n2 = tc_qstate_norm_sq(state, 1);
        CHECK("Trotter preserves ||state||²", APPROX(n2, 1.0f, 1e-5f));
    }

    /* ===== Trotter on a 2-qubit ZZ term: should match the closed-form ZZ gate =====
     * exp(-i α Z⊗Z) for α = π/2: Trotter step of one ZZ term should give
     * the same diag(-i, +i, +i, -i) phases on |++⟩ that the ZZ gate does. */
    {
        float ts[8], gs[8];
        /* Trotter path. */
        tc_qstate_zero(ts, 2);
        int q0[1] = {0}, q1[1] = {1};
        tc_qstate_apply_gate(ts, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(ts, 2, TC_GATE_H, q1, NULL);
        int32_t axes[2] = { TC_GATE_Z, TC_GATE_Z };
        int32_t qubits[2] = { 0, 1 };
        tc_pauli_term_t zzterm = {2, axes, qubits, 1.0f};
        tc_qstate_trotter_step(ts, 2, &zzterm, 1, (float)M_PI/2.0f, 1);
        /* Gate path. */
        tc_qstate_zero(gs, 2);
        tc_qstate_apply_gate(gs, 2, TC_GATE_H, q0, NULL);
        tc_qstate_apply_gate(gs, 2, TC_GATE_H, q1, NULL);
        float zzp[1] = {(float)M_PI};
        int q01[2] = {0, 1};
        tc_qstate_apply_gate(gs, 2, TC_GATE_ZZ, q01, zzp);
        float max_err = 0.0f;
        for (int i = 0; i < 8; ++i) {
            float d = fabsf(ts[i] - gs[i]);
            if (d > max_err) max_err = d;
        }
        CHECK("Trotter Z⊗Z matches closed-form ZZ gate", max_err < 1e-4f);
    }

    /* ===== Quantum Geometric Tensor: parameter-shift on Rz(θ)|+⟩ =====
     * |ψ(θ)⟩ = Rz(θ)|+⟩. ∂_θ |ψ⟩ = -i(Z/2)|ψ⟩ ⇒ <∂ψ|∂ψ> = 1/4, <ψ|∂ψ> = 0
     * (since <+|Z|+⟩ = 0). So QGT_{00} = 1/4 - 0 = 0.25. */
    {
        float psi[4], dpsi[4];
        tc_qstate_zero(psi, 1);
        int q0[1] = {0};
        tc_qstate_apply_gate(psi, 1, TC_GATE_H, q0, NULL);
        /* dpsi = -i/2 · Z · psi. psi = (1/√2, 0, 1/√2, 0).
         * Z · psi = (1/√2, 0, -1/√2, 0).
         * -i/2 · Z · psi = (0, -1/(2√2), 0,  1/(2√2)). */
        const float inv_sqrt2 = 0.70710678f;
        const float half_inv_sqrt2 = inv_sqrt2 * 0.5f;
        dpsi[0] = 0.0f;          dpsi[1] = -half_inv_sqrt2;
        dpsi[2] = 0.0f;          dpsi[3] =  half_inv_sqrt2;
        const float* d_arr[1] = { dpsi };
        float G[2];  /* 1×1 complex */
        tc_quantum_geometric_tensor(psi, d_arr, 1, 1, G);
        CHECK("QGT_{00} = 0.25 (Fubini-Study metric on Rz parameter)",
              APPROX(G[0], 0.25f, 1e-5f));
        CHECK("QGT_{00} imag = 0", APPROX(G[1], 0.0f, 1e-5f));
    }

    /* ===== 4-qubit circuit norm-preservation ===== */
    {
        float state[32];  /* 4 qubits = 16 amps = 32 floats */
        tc_qstate_zero(state, 4);
        /* H on every qubit, then a CNOT chain. */
        for (int q = 0; q < 4; ++q) {
            int qs[1] = {q};
            tc_qstate_apply_gate(state, 4, TC_GATE_H, qs, NULL);
        }
        for (int q = 0; q < 3; ++q) {
            int qs[2] = {q, q + 1};
            tc_qstate_apply_gate(state, 4, TC_GATE_CNOT, qs, NULL);
        }
        /* RZ rotations with a sweep of angles. */
        for (int q = 0; q < 4; ++q) {
            int qs[1] = {q};
            float params[1] = {0.5f * (q + 1)};
            tc_qstate_apply_gate(state, 4, TC_GATE_RZ, qs, params);
        }
        float n2 = tc_qstate_norm_sq(state, 4);
        CHECK("4-qubit circuit preserves ||state||² = 1",
              APPROX(n2, 1.0f, 1e-5f));
    }

    printf("\n  %d/%d quantum gate tests passed\n", test_pass, test_count);
    return (test_pass == test_count) ? 0 : 1;
}
