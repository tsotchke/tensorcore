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
