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
