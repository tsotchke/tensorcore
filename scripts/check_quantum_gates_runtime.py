#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for tensorcore's quantum gates.

Exercises the state-vector apply path: single-qubit Pauli/H/RY,
two-qubit CNOT (Bell state preparation), SWAP, and unitarity over a
4-qubit H+CNOT+RZ circuit. Writes JSON keyed under
`checks.quantum_gates.*` consumed by the ICC oracle.

Usage:
    python3 scripts/check_quantum_gates_runtime.py \\
        --out build/quantum_gates_runtime_evidence.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "quantum_gates_runtime_evidence.json"))
    p.add_argument("--lib", default=os.environ.get("TENSORCORE_LIB",
                                                     str(Path(__file__).resolve().parent.parent
                                                         / "build" / "libtensorcore.dylib")))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"quantum_gates": {"runtime_status": "skipped_no_lib",
                                              "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped_no_lib reason={reason}")
    return 0


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()

    try:
        lib = ctypes.CDLL(args.lib)
    except OSError as exc:
        return _skip(out_path, f"libtensorcore not loadable: {exc}")
    try:
        import numpy as np
    except ImportError:
        return _skip(out_path, "numpy not installed")

    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int)

    def bind(name, ret, args_):
        try:
            f = getattr(lib, name)
        except AttributeError:
            return None
        f.restype = ret; f.argtypes = args_
        return f

    q_zero    = bind("tc_qstate_zero", None, [f32p, ctypes.c_int])
    q_apply_1 = bind("tc_qstate_apply_1q_unitary", None, [f32p, ctypes.c_int, ctypes.c_int, f32p])
    q_apply_2 = bind("tc_qstate_apply_2q_unitary", None, [f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p])
    q_apply_g = bind("tc_qstate_apply_gate", None, [f32p, ctypes.c_int, ctypes.c_int, i32p, f32p])
    q_prob1   = bind("tc_qstate_prob_one", ctypes.c_float, [f32p, ctypes.c_int, ctypes.c_int])
    q_norm    = bind("tc_qstate_norm_sq",  ctypes.c_float, [f32p, ctypes.c_int])

    if not all((q_zero, q_apply_g, q_prob1, q_norm)):
        payload = {"checks": {"quantum_gates": {"runtime_status": "failed",
                                                  "failure_reason": "missing public quantum symbols"}}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("runtime_status=failed reason=missing public quantum symbols", file=sys.stderr)
        return 1

    INV_SQRT2 = 0.70710678
    probes: dict[str, dict] = {}

    # === X|0⟩ = |1⟩ ===
    try:
        state = (ctypes.c_float * 4)(); q_zero(state, 1)
        qubits = (ctypes.c_int * 1)(0)
        q_apply_g(state, 1, 1, qubits, None)  # GATE_X = 1
        ok = (abs(state[0]) < 1e-6 and abs(state[1]) < 1e-6
              and abs(state[2] - 1.0) < 1e-6 and abs(state[3]) < 1e-6)
        probes["pauli_X_on_zero"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["pauli_X_on_zero"] = {"error": str(exc), "passed": False,
                                       "traceback": traceback.format_exc()}

    # === H|0⟩ = |+⟩ ===
    try:
        state = (ctypes.c_float * 4)(); q_zero(state, 1)
        qubits = (ctypes.c_int * 1)(0)
        q_apply_g(state, 1, 4, qubits, None)  # GATE_H = 4
        ok = (abs(state[0] - INV_SQRT2) < 1e-5 and abs(state[2] - INV_SQRT2) < 1e-5
              and abs(state[1]) < 1e-6 and abs(state[3]) < 1e-6)
        probes["hadamard_on_zero"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["hadamard_on_zero"] = {"error": str(exc), "passed": False,
                                        "traceback": traceback.format_exc()}

    # === RY(π)|0⟩ ≈ |1⟩ ===
    try:
        state = (ctypes.c_float * 4)(); q_zero(state, 1)
        qubits = (ctypes.c_int * 1)(0)
        params = (ctypes.c_float * 1)(math.pi)
        q_apply_g(state, 1, 8, qubits, params)  # GATE_RY = 8
        ok = (abs(state[0]) < 1e-5 and abs(state[2] - 1.0) < 1e-5)
        probes["ry_pi_on_zero"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["ry_pi_on_zero"] = {"error": str(exc), "passed": False,
                                     "traceback": traceback.format_exc()}

    # === Bell: H_0 then CNOT_{0,1} → (|00⟩+|11⟩)/√2 ===
    try:
        state = (ctypes.c_float * 8)(); q_zero(state, 2)
        q0 = (ctypes.c_int * 1)(0); q01 = (ctypes.c_int * 2)(0, 1)
        q_apply_g(state, 2, 4, q0, None)        # H on q0
        q_apply_g(state, 2, 10, q01, None)      # CNOT_{0,1} (GATE_CNOT = 10)
        # Expected: amp(|00⟩) = amp(|11⟩) = 1/√2; rest 0
        ok = (abs(state[0] - INV_SQRT2) < 1e-5 and abs(state[6] - INV_SQRT2) < 1e-5
              and abs(state[2]) < 1e-6 and abs(state[4]) < 1e-6)
        p_q0 = float(q_prob1(state, 2, 0))
        p_q1 = float(q_prob1(state, 2, 1))
        ok = ok and abs(p_q0 - 0.5) < 1e-5 and abs(p_q1 - 0.5) < 1e-5
        probes["bell_state_prep"] = {
            "state_re_im": list(state),
            "prob_q0_one": p_q0, "prob_q1_one": p_q1,
            "passed": ok,
        }
    except Exception as exc:
        probes["bell_state_prep"] = {"error": str(exc), "passed": False,
                                       "traceback": traceback.format_exc()}

    # === SWAP |10⟩ → |01⟩ ===
    try:
        state = (ctypes.c_float * 8)(); q_zero(state, 2)
        state[0] = 0.0; state[4] = 1.0  # |10⟩ at amp index 2 = float offset 4
        q01 = (ctypes.c_int * 2)(0, 1)
        q_apply_g(state, 2, 13, q01, None)      # SWAP = 13
        ok = abs(state[2] - 1.0) < 1e-6 and abs(state[4]) < 1e-6
        probes["swap_10_to_01"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["swap_10_to_01"] = {"error": str(exc), "passed": False,
                                     "traceback": traceback.format_exc()}

    # === CRZ(π) on H_0 H_1 |00⟩ — controlled phase rotation ===
    # Specifically exercises the new controlled-rotation dispatch path
    # (QGTL uses CRX/CRY/CRZ/CH 80+ times across its src/ tree).
    try:
        state = (ctypes.c_float * 8)(); q_zero(state, 2)
        q0 = (ctypes.c_int * 1)(0); q1 = (ctypes.c_int * 1)(1); q01 = (ctypes.c_int * 2)(0, 1)
        q_apply_g(state, 2, 4, q0, None)         # H_0
        q_apply_g(state, 2, 4, q1, None)         # H_1
        params = (ctypes.c_float * 1)(math.pi)
        q_apply_g(state, 2, 24, q01, params)     # CRZ(π) — TC_GATE_CRZ = 24
        # Expected: idx 0 = 0.5+0i, idx 1 (ctrl=1,targ=0) = -0.5i,
        #           idx 2 = 0.5+0i (unchanged), idx 3 (ctrl=1,targ=1) = +0.5i.
        ok = (abs(state[0] - 0.5) < 1e-5 and abs(state[1]) < 1e-5
              and abs(state[2]) < 1e-5 and abs(state[3] + 0.5) < 1e-5
              and abs(state[4] - 0.5) < 1e-5 and abs(state[5]) < 1e-5
              and abs(state[6]) < 1e-5 and abs(state[7] - 0.5) < 1e-5)
        n2 = float(q_norm(state, 2))
        ok = ok and abs(n2 - 1.0) < 1e-5
        probes["crz_pi_on_plusplus"] = {"state_re_im": list(state), "norm_sq": n2, "passed": ok}
    except Exception as exc:
        probes["crz_pi_on_plusplus"] = {"error": str(exc), "passed": False,
                                          "traceback": traceback.format_exc()}

    # === Toffoli (CCX) on |111⟩ → |110⟩ — exercises the 3-qubit dispatch ===
    try:
        state = (ctypes.c_float * 16)(); q_zero(state, 3)
        state[0] = 0.0
        state[14] = 1.0  # amp idx 7 (q2=1,q1=1,q0=1)
        qs = (ctypes.c_int * 3)(2, 1, 0)
        q_apply_g(state, 3, 18, qs, None)  # TC_GATE_CCX = 18
        ok = (abs(state[14]) < 1e-6 and abs(state[12] - 1.0) < 1e-6
              and abs(state[13]) < 1e-6)
        n2 = float(q_norm(state, 3))
        ok = ok and abs(n2 - 1.0) < 1e-5
        probes["toffoli_ccx"] = {"state_re_im": list(state), "norm_sq": n2, "passed": ok}
    except Exception as exc:
        probes["toffoli_ccx"] = {"error": str(exc), "passed": False,
                                   "traceback": traceback.format_exc()}

    # === ISWAP |10⟩ → i|01⟩ — exercises the entangling 2-qubit dispatch ===
    try:
        state = (ctypes.c_float * 8)(); q_zero(state, 2)
        state[0] = 0.0; state[4] = 1.0  # |10⟩ at amp idx 2 = offset 4
        q01 = (ctypes.c_int * 2)(0, 1)
        q_apply_g(state, 2, 21, q01, None)   # TC_GATE_ISWAP = 21
        ok = (abs(state[4]) < 1e-6 and abs(state[5]) < 1e-6
              and abs(state[2]) < 1e-6 and abs(state[3] - 1.0) < 1e-6)
        probes["iswap_10_to_i01"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["iswap_10_to_i01"] = {"error": str(exc), "passed": False,
                                        "traceback": traceback.format_exc()}

    # === U3(π, 0, 0) on |0⟩ ≈ |1⟩ — exercises 3-parameter dispatch ===
    try:
        state = (ctypes.c_float * 4)(); q_zero(state, 1)
        q0 = (ctypes.c_int * 1)(0)
        params = (ctypes.c_float * 3)(math.pi, 0.0, 0.0)
        q_apply_g(state, 1, 17, q0, params)  # TC_GATE_U3 = 17
        ok = abs(state[0]) < 1e-5 and abs(state[2] - 1.0) < 1e-5
        probes["u3_pi_0_0_on_zero"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["u3_pi_0_0_on_zero"] = {"error": str(exc), "passed": False,
                                          "traceback": traceback.format_exc()}

    # === ZZ(π) on |++⟩ — exercises Ising rotation ===
    try:
        state = (ctypes.c_float * 8)(); q_zero(state, 2)
        q0 = (ctypes.c_int * 1)(0); q1 = (ctypes.c_int * 1)(1); q01 = (ctypes.c_int * 2)(0, 1)
        q_apply_g(state, 2, 4, q0, None)         # H_0
        q_apply_g(state, 2, 4, q1, None)         # H_1
        params = (ctypes.c_float * 1)(math.pi)
        q_apply_g(state, 2, 35, q01, params)     # TC_GATE_ZZ = 35
        # amp[0]=-0.5i, amp[1]=+0.5i, amp[2]=+0.5i, amp[3]=-0.5i
        ok = (abs(state[0]) < 1e-5 and abs(state[1] + 0.5) < 1e-5
              and abs(state[2]) < 1e-5 and abs(state[3] - 0.5) < 1e-5
              and abs(state[4]) < 1e-5 and abs(state[5] - 0.5) < 1e-5
              and abs(state[6]) < 1e-5 and abs(state[7] + 0.5) < 1e-5)
        probes["zz_pi_on_plusplus"] = {"state_re_im": list(state), "passed": ok}
    except Exception as exc:
        probes["zz_pi_on_plusplus"] = {"error": str(exc), "passed": False,
                                          "traceback": traceback.format_exc()}

    # === Trotter step: Z evolution matches Rz closed-form ===
    try:
        # Bind trotter symbols
        class _PauliTerm(ctypes.Structure):
            _fields_ = [("n_paulis", ctypes.c_int32),
                        ("axes",     ctypes.POINTER(ctypes.c_int32)),
                        ("qubits",   ctypes.POINTER(ctypes.c_int32)),
                        ("coef",     ctypes.c_float)]
        q_trotter = bind("tc_qstate_trotter_step", None,
                          [f32p, ctypes.c_int, ctypes.POINTER(_PauliTerm),
                           ctypes.c_int, ctypes.c_float, ctypes.c_int])
        if q_trotter is None:
            raise RuntimeError("tc_qstate_trotter_step symbol missing")
        state = (ctypes.c_float * 4)(); q_zero(state, 1)
        q0 = (ctypes.c_int * 1)(0)
        q_apply_g(state, 1, 4, q0, None)   # H_0 → |+⟩
        axes = (ctypes.c_int32 * 1)(3)     # TC_GATE_Z = 3
        qubits = (ctypes.c_int32 * 1)(0)
        term = _PauliTerm(1, axes, qubits, 1.0)
        terms = (_PauliTerm * 1)(term)
        q_trotter(state, 1, terms, 1, math.pi / 2.0, 1)
        # Expected: -i|-⟩ = (0, -1/√2, 0, +1/√2).
        inv_sqrt2 = 0.70710678
        ok = (abs(state[0]) < 1e-5 and abs(state[1] + inv_sqrt2) < 1e-5
              and abs(state[2]) < 1e-5 and abs(state[3] - inv_sqrt2) < 1e-5)
        n2 = float(q_norm(state, 1))
        ok = ok and abs(n2 - 1.0) < 1e-5
        probes["trotter_z_on_plus"] = {"state_re_im": list(state), "norm_sq": n2, "passed": ok}
    except Exception as exc:
        probes["trotter_z_on_plus"] = {"error": str(exc), "passed": False,
                                          "traceback": traceback.format_exc()}

    # === Quantum Geometric Tensor: Rz parameter on |+⟩ should give G_{00} = 0.25 ===
    try:
        q_qgt = bind("tc_quantum_geometric_tensor", None,
                      [f32p, ctypes.POINTER(f32p), ctypes.c_int, ctypes.c_int, f32p])
        if q_qgt is None:
            raise RuntimeError("tc_quantum_geometric_tensor symbol missing")
        psi = (ctypes.c_float * 4)(); q_zero(psi, 1)
        q0 = (ctypes.c_int * 1)(0)
        q_apply_g(psi, 1, 4, q0, None)     # H_0 → |+⟩
        # dpsi = -i/2 · Z · psi = (0, -1/(2√2), 0, +1/(2√2))
        half_inv_sqrt2 = 0.70710678 * 0.5
        dpsi = (ctypes.c_float * 4)(0.0, -half_inv_sqrt2, 0.0, half_inv_sqrt2)
        dpsi_arr = (f32p * 1)(ctypes.cast(dpsi, f32p))
        G = (ctypes.c_float * 2)()  # 1×1 complex
        q_qgt(psi, dpsi_arr, 1, 1, G)
        ok = abs(G[0] - 0.25) < 1e-5 and abs(G[1]) < 1e-5
        probes["qgt_rz_on_plus"] = {"G_re_im": [G[0], G[1]],
                                       "expected_re": 0.25, "passed": ok}
    except Exception as exc:
        probes["qgt_rz_on_plus"] = {"error": str(exc), "passed": False,
                                        "traceback": traceback.format_exc()}

    # === Density matrix: |0⟩⟨0|, H applied, depolarising channel ===
    try:
        dm_zero    = bind("tc_dmstate_zero", None, [f32p, ctypes.c_int])
        dm_apply_u = bind("tc_dmstate_apply_1q_unitary", None,
                            [f32p, ctypes.c_int, ctypes.c_int, f32p])
        dm_apply_k = bind("tc_dmstate_apply_kraus_1q", None,
                            [f32p, ctypes.c_int, ctypes.c_int, f32p, ctypes.c_int])
        dm_trace   = bind("tc_dmstate_trace", None,
                            [f32p, ctypes.c_int, f32p, f32p])
        dm_purity  = bind("tc_dmstate_purity", ctypes.c_float, [f32p, ctypes.c_int])
        if any(fn is None for fn in (dm_zero, dm_apply_u, dm_apply_k, dm_trace, dm_purity)):
            raise RuntimeError("density-matrix symbols missing")

        rho = (ctypes.c_float * 8)()  # 1-qubit dm = 2x2 complex = 8 floats
        dm_zero(rho, 1)
        # |0⟩⟨0|: purity=1, trace=1
        pu0 = float(dm_purity(rho, 1))
        tr_re = ctypes.c_float(); tr_im = ctypes.c_float()
        dm_trace(rho, 1, ctypes.byref(tr_re), ctypes.byref(tr_im))
        zero_ok = abs(pu0 - 1.0) < 1e-5 and abs(tr_re.value - 1.0) < 1e-5

        # H |0⟩⟨0| H = |+⟩⟨+| — purity stays 1, ρ becomes [[0.5, 0.5], [0.5, 0.5]]
        inv_sqrt2 = 0.70710678
        H = (ctypes.c_float * 8)(inv_sqrt2, 0, inv_sqrt2, 0,
                                  inv_sqrt2, 0, -inv_sqrt2, 0)
        dm_apply_u(rho, 1, 0, H)
        pu_plus = float(dm_purity(rho, 1))
        plus_ok = (abs(pu_plus - 1.0) < 1e-4
                   and abs(rho[0] - 0.5) < 1e-5 and abs(rho[6] - 0.5) < 1e-5)

        # Depolarising channel: K_k = (1/2) {I, X, Y, Z} → maximally mixed.
        dm_zero(rho, 1)  # back to |0⟩⟨0|
        k = 0.5
        kraus = (ctypes.c_float * 32)(
            k,0, 0,0,  0,0, k,0,
            0,0, k,0,  k,0, 0,0,
            0,0, 0,-k, 0,k, 0,0,
            k,0, 0,0,  0,0, -k,0,
        )
        dm_apply_k(rho, 1, 0, kraus, 4)
        pu_mix = float(dm_purity(rho, 1))
        dm_trace(rho, 1, ctypes.byref(tr_re), ctypes.byref(tr_im))
        mix_ok = (abs(pu_mix - 0.5) < 1e-4 and abs(tr_re.value - 1.0) < 1e-5)

        probes["dmstate_pure_to_mixed"] = {
            "zero_purity": pu0, "zero_ok": zero_ok,
            "plus_purity": pu_plus, "plus_ok": plus_ok,
            "mixed_purity": pu_mix, "mixed_trace_re": tr_re.value, "mix_ok": mix_ok,
            "passed": (zero_ok and plus_ok and mix_ok),
        }
    except Exception as exc:
        probes["dmstate_pure_to_mixed"] = {"error": str(exc), "passed": False,
                                              "traceback": traceback.format_exc()}

    # === 4-qubit circuit norm preservation ===
    try:
        state = (ctypes.c_float * 32)(); q_zero(state, 4)
        for q in range(4):
            qubits = (ctypes.c_int * 1)(q)
            q_apply_g(state, 4, 4, qubits, None)            # H on every qubit
        for q in range(3):
            qubits = (ctypes.c_int * 2)(q, q + 1)
            q_apply_g(state, 4, 10, qubits, None)           # CNOT chain
        for q in range(4):
            qubits = (ctypes.c_int * 1)(q)
            params = (ctypes.c_float * 1)(0.5 * (q + 1))
            q_apply_g(state, 4, 9, qubits, params)          # RZ rotations
        n2 = float(q_norm(state, 4))
        ok = abs(n2 - 1.0) < 1e-5
        probes["four_qubit_unitarity"] = {"norm_sq": n2, "passed": ok}
    except Exception as exc:
        probes["four_qubit_unitarity"] = {"error": str(exc), "passed": False,
                                            "traceback": traceback.format_exc()}

    for _probe in probes.values():
        _probe["runtime_status"] = "passed" if _probe.get("passed", False) else "failed"
    all_passed = all(p.get("passed", False) for p in probes.values())
    payload = {
        "checks": {
            "quantum_gates": {
                "runtime_status": "passed" if all_passed else "failed",
                "probes": probes,
                "lib_path": args.lib,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "passed" if all_passed else "failed"
    print(f"runtime_status={status} probes={list(probes.keys())}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
