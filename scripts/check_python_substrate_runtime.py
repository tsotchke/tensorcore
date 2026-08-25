#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for the Python substrate wrappers.

The C kernels in lib/ops/*.cpp + lib/distributed/remote_shard.cpp are
the source of truth, but the Python wrappers in python/tensorcore/__init__.py
are how sibling repos (qLLM, Noesis, moonlab, QGTL) actually consume the
substrate. This smoke proves the wrappers correctly marshal NumPy
arrays into the C ABI for every shipped op family: Lorentz, Sphere,
Torus, SU(2) / SO(3) Lie groups, quantum state-vector + gates, metric
tensor, geodesic ODE, holonomic gate composition, shard plan.

Writes JSON keyed under `checks.python_substrate.*` for the ICC
`python-substrate-runtime-evidence` oracle.

Usage:
    python3 scripts/check_python_substrate_runtime.py \\
        --out build/python_substrate_runtime_evidence.json
"""

from __future__ import annotations

import argparse
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
                                         / "build" / "python_substrate_runtime_evidence.json"))
    p.add_argument("--lib", default=os.environ.get("TENSORCORE_LIB",
                                                     str(Path(__file__).resolve().parent.parent
                                                         / "build" / "libtensorcore.dylib")))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"python_substrate": {"runtime_status": "skipped",
                                                  "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped reason={reason}")
    return 0


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()

    os.environ.setdefault("TENSORCORE_LIB", args.lib)
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "python"))

    try:
        import numpy as np
    except ImportError:
        return _skip(out_path, "numpy not installed")
    try:
        import tensorcore as tc
    except OSError as exc:
        return _skip(out_path, f"libtensorcore not loadable: {exc}")
    except Exception as exc:
        return _skip(out_path, f"tensorcore import failed: {exc}")

    probes: dict[str, dict] = {}

    def probe(name, fn):
        try:
            result = fn()
            probes[name] = {**result,
                             "runtime_status": "passed" if result.get("passed", False) else "failed"}
        except Exception as exc:
            probes[name] = {
                "passed": False,
                "runtime_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    def lorentz_probe():
        rng = np.random.default_rng(7)
        base = np.zeros(8, dtype=np.float32); base[0] = 1.0
        v = np.concatenate([[0], rng.standard_normal(7) * 0.3]).astype(np.float32)
        q = tc.lorentz_exp(base, v)
        v2 = tc.lorentz_log(base, q)
        err = float(np.abs(v - v2).max())
        d_self = tc.lorentz_distance(base, base)
        d_pq = tc.lorentz_distance(base, q)
        return {"roundtrip_err": err, "d_self": d_self, "d_pq": d_pq,
                "passed": err < 1e-4 and abs(d_self) < 1e-5 and d_pq > 0}

    def sphere_probe():
        rng = np.random.default_rng(11)
        base = np.zeros(16, dtype=np.float32); base[0] = 1.0
        v = np.concatenate([[0], rng.standard_normal(15) * 0.2]).astype(np.float32)
        q = tc.sphere_exp(base, v)
        v2 = tc.sphere_log(base, q)
        mid = tc.sphere_slerp(base, q, 0.5)
        err = float(np.abs(v - v2).max())
        d_mid = tc.sphere_distance(base, mid)
        d_full = tc.sphere_distance(base, q)
        return {"roundtrip_err": err, "d_slerp_mid": d_mid, "d_full": d_full,
                "passed": err < 1e-4 and abs(2 * d_mid - d_full) < 1e-3}

    def torus_probe():
        base = np.array([0, 0, 0], dtype=np.float32)
        v = np.array([0.4, -0.3, 0.2], dtype=np.float32)
        q = tc.torus_exp(base, v); v2 = tc.torus_log(base, q)
        err = float(np.abs(v - v2).max())
        return {"roundtrip_err": err, "passed": err < 1e-4}

    def lie_su2_probe():
        a_in, b_in, c_in = 0.1, -0.2, math.pi / 4
        U = tc.su2_exp(a_in, b_in, c_in)
        a, b, c = tc.su2_log(U)
        err = max(abs(a - a_in), abs(b - b_in), abs(c - c_in))
        # SU(2) → SO(3) double cover bridge
        R = tc.su2_to_so3(U)
        det = float(np.linalg.det(R))
        orth = float(np.abs(R @ R.T - np.eye(3)).max())
        return {"su2_log_err": err, "so3_det": det, "so3_orthogonality_err": orth,
                "passed": err < 1e-4 and abs(det - 1.0) < 1e-3 and orth < 1e-3}

    def lie_so3_probe():
        R = tc.so3_exp(0.0, 0.0, math.pi / 2)
        wx, wy, wz = tc.so3_log(R)
        return {"recovered_wz": wz, "expected_wz": math.pi / 2,
                "passed": abs(wz - math.pi / 2) < 1e-4 and abs(wx) < 1e-4 and abs(wy) < 1e-4}

    def quantum_state_probe():
        state = tc.qstate_zero(2)
        H = tc.gate_matrix_1q(4)  # H = 4
        state = tc.qstate_apply_1q(state, 2, 0, H)
        p_one_q0 = tc.qstate_prob_one(state, 2, 0)
        norm = tc.qstate_norm_sq(state, 2)
        return {"prob_one_q0_after_H": p_one_q0, "norm_sq": norm,
                "passed": abs(p_one_q0 - 0.5) < 1e-5 and abs(norm - 1.0) < 1e-5}

    def metric_probe():
        m_euc = tc.metric_euclidean()
        p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        v = np.array([1, 2, 3], dtype=np.float32)
        w = np.array([4, 5, 6], dtype=np.float32)
        g_vw = tc.metric_apply(m_euc, p, v, w)
        # Euclidean: g_vw = v·w
        expect = float(v @ w)
        # Inverse of identity
        I = np.eye(3, dtype=np.float32)
        I_inv, rc = tc.metric_inverse(I)
        inv_err = float(np.abs(I_inv - I).max())
        # Sphere stereographic metric at origin gives g_xx = 4
        m_sph, holder = tc.metric_sphere_stereographic(1.0)
        v_x = np.array([1, 0, 0], dtype=np.float32)
        g_sphere = tc.metric_apply(m_sph, p * 0,  # origin
                                    v_x, v_x,
                                    user_ptr=ctypes_addr(holder))
        return {"g_vw_euc": g_vw, "expected_vw": expect,
                "inverse_id_err": inv_err,
                "g_xx_sphere_origin": g_sphere,
                "passed": abs(g_vw - expect) < 1e-4 and rc == 0
                          and inv_err < 1e-5 and abs(g_sphere - 4.0) < 1e-3}

    def geodesic_probe():
        m_euc = tc.metric_euclidean()
        pos0 = np.zeros(3, dtype=np.float32)
        vel0 = np.array([1, 0, 0], dtype=np.float32)
        p, v = tc.geodesic_integrate(m_euc, pos0, vel0, 0.01, 100)
        pos_err = float(np.abs(p - np.array([1, 0, 0])).max())
        vel_err = float(np.abs(v - vel0).max())
        return {"final_pos": p.tolist(), "final_vel": v.tolist(),
                "pos_err": pos_err, "vel_err": vel_err,
                "passed": pos_err < 1e-3 and vel_err < 1e-3}

    def holonomic_probe():
        U = tc.holonomic_compose_su2([[0, 0, math.pi / 4]])
        ph = tc.holonomic_berry_phase(U)
        U2 = tc.holonomic_compose_su2([[0, 0, math.pi / 4],
                                         [0, 0, -math.pi / 4]])
        ph2 = tc.holonomic_berry_phase(U2)
        return {"phase_pi_over_4": ph, "expected": math.pi / 4,
                "phase_closed_loop": ph2,
                "passed": abs(ph - math.pi / 4) < 1e-4 and abs(ph2) < 1e-3}

    def shard_plan_probe():
        plan = tc.shard_plan(n_peers=4, rows=10, cols=8)
        owners = [tc.shard_owner(plan, r) for r in range(10)]
        ranges = [tc.shard_local_range(plan, r) for r in range(4)]
        # Verify partition is a clean cover with no gaps
        covered = sum(hi - lo for lo, hi in ranges)
        return {"owners": owners, "ranges": ranges, "covered_rows": covered,
                "passed": covered == 10
                          and all(0 <= o < 4 for o in owners)}

    def quantum_attention_probe():
        # Born-rule overlap: same state = 1, orthogonal = 0.
        state0 = tc.qstate_zero(2)
        state1 = state0.copy(); state1[0] = 0.0; state1[2] = 1.0  # |10⟩
        s_self = tc.quantum_attention_score(state0, state0, 2)
        s_orth = tc.quantum_attention_score(state0, state1, 2)
        # Softmax sums to 1 per row.
        scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        attn = tc.quantum_attention_softmax(scores, 1, 3, temperature=1.0)
        row_sum = float(attn.sum())
        # Apply with identity → values unchanged.
        I = np.eye(2, dtype=np.float32)
        V = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out = tc.quantum_attention_apply(I, V, 2, 2, 2)
        out_err = float(np.abs(out - V).max())
        return {"score_self": s_self, "score_orth": s_orth,
                "softmax_row_sum": row_sum, "apply_id_err": out_err,
                "passed": (abs(s_self - 1.0) < 1e-5 and abs(s_orth) < 1e-5
                           and abs(row_sum - 1.0) < 1e-5 and out_err < 1e-6)}

    def quantum_entanglement_probe():
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 → S = 1 bit on either qubit.
        bell = np.zeros(8, dtype=np.float32)
        bell[0] = 1 / math.sqrt(2)  # |00⟩.re
        bell[6] = 1 / math.sqrt(2)  # |11⟩.re
        s_q0 = tc.quantum_entanglement_entropy(bell, 2, 0)
        s_q1 = tc.quantum_entanglement_entropy(bell, 2, 1)
        # Product state → S = 0 on every qubit.
        prod = tc.qstate_zero(2)
        s_prod = tc.quantum_entanglement_entropy(prod, 2, 0)
        return {"bell_entropy_q0_bits": s_q0,
                "bell_entropy_q1_bits": s_q1,
                "product_entropy_bits": s_prod,
                "passed": (abs(s_q0 - 1.0) < 1e-4 and abs(s_q1 - 1.0) < 1e-4
                           and abs(s_prod) < 1e-4)}

    # Helper for the metric probe — ctypes.byref address as an int.
    import ctypes
    def ctypes_addr(c_obj):
        return ctypes.addressof(c_obj)

    probe("lorentz_wrapper",      lorentz_probe)
    probe("sphere_wrapper",       sphere_probe)
    probe("torus_wrapper",        torus_probe)
    probe("lie_su2_wrapper",      lie_su2_probe)
    probe("lie_so3_wrapper",      lie_so3_probe)
    probe("quantum_state_wrapper", quantum_state_probe)
    probe("metric_wrapper",       metric_probe)
    probe("geodesic_wrapper",     geodesic_probe)
    probe("holonomic_wrapper",    holonomic_probe)
    probe("shard_plan_wrapper",   shard_plan_probe)
    probe("quantum_attention_wrapper", quantum_attention_probe)
    probe("quantum_entanglement_wrapper", quantum_entanglement_probe)

    all_passed = all(p.get("passed", False) for p in probes.values())
    payload = {
        "checks": {
            "python_substrate": {
                "runtime_status": "passed" if all_passed else "failed",
                "lib_path": args.lib,
                "probes": probes,
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
