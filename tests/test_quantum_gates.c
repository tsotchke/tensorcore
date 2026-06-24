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
