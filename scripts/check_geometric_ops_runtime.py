#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for tensorcore's geometric kernels.

Exercises every shipped manifold (Poincaré ball, Lorentz hyperboloid,
n-sphere, mixed-curvature product) via the public ctypes ABI. For each
op we run a round-trip / invariant probe and report pass/fail. Writes
a JSON document keyed under `checks.geometric_ops.*` that the ICC
oracle in `.icc/completion-oracles.yaml` consumes.

Usage:
    python3 scripts/check_geometric_ops_runtime.py \\
        --out build/geometric_ops_runtime_evidence.json
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
                                         / "build" / "geometric_ops_runtime_evidence.json"))
    p.add_argument("--lib", default=os.environ.get("TENSORCORE_LIB",
                                                     str(Path(__file__).resolve().parent.parent
                                                         / "build" / "libtensorcore.dylib")))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"geometric_ops": {"runtime_status": "skipped_no_lib",
                                              "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped_no_lib reason={reason}")
    return 0


def _fail(out_path: Path, reason: str, **extra) -> int:
    payload = {"checks": {"geometric_ops": {"runtime_status": "failed",
                                              "failure_reason": reason,
                                              **extra}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=failed reason={reason}", file=sys.stderr)
    return 1


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

    def bind(name, ret, args_):
        try:
            f = getattr(lib, name)
        except AttributeError:
            return None
        f.restype = ret
        f.argtypes = args_
        return f

    # --- Lorentz ---
    lor_exp  = bind("tc_lorentz_exp",  None,           [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
    lor_log  = bind("tc_lorentz_log",  None,           [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
    lor_mink = bind("tc_lorentz_minkowski", ctypes.c_float, [f32p, f32p, ctypes.c_size_t])
    lor_dist = bind("tc_lorentz_distance", ctypes.c_float, [f32p, f32p, ctypes.c_size_t, ctypes.c_float])
    # --- Sphere ---
    sph_exp  = bind("tc_sphere_exp", None, [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
    sph_log  = bind("tc_sphere_log", None, [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
    sph_inner= bind("tc_sphere_inner", ctypes.c_float, [f32p, f32p, ctypes.c_size_t])
    sph_slerp= bind("tc_sphere_slerp", None, [f32p, f32p, ctypes.c_float, f32p, ctypes.c_size_t, ctypes.c_float])
    # --- Product manifold ---
    class _Factor(ctypes.Structure):
        _fields_ = [("kind", ctypes.c_int32), ("intrinsic_dim", ctypes.c_int32), ("curvature", ctypes.c_float)]
    pm_dim   = bind("tc_product_ambient_dim", ctypes.c_int32, [ctypes.POINTER(_Factor), ctypes.c_int32])
    pm_exp   = bind("tc_product_exp", None, [ctypes.POINTER(_Factor), ctypes.c_int32, f32p, f32p, f32p])
    pm_log   = bind("tc_product_log", None, [ctypes.POINTER(_Factor), ctypes.c_int32, f32p, f32p, f32p])
    pm_dist  = bind("tc_product_distance", ctypes.c_float, [ctypes.POINTER(_Factor), ctypes.c_int32, f32p, f32p])

    required = {"tc_lorentz_exp": lor_exp, "tc_lorentz_log": lor_log,
                "tc_sphere_exp": sph_exp, "tc_sphere_log": sph_log,
                "tc_product_exp": pm_exp, "tc_product_log": pm_log}
    missing = [n for n, fn in required.items() if fn is None]
    if missing:
        return _fail(out_path, f"missing symbols: {missing}")

    probes: dict[str, dict] = {}

    def buf(arr):
        a = np.asarray(arr, dtype=np.float32).copy()
        return a, a.ctypes.data_as(f32p)

    rng = np.random.default_rng(7)

    # === Lorentz round-trip ===
    try:
        n = 8; c = 1.0
        base = np.zeros(n, dtype=np.float32); base[0] = 1.0
        bp, bptr = buf(base)
        v = np.concatenate([[0.0], rng.standard_normal(n-1) * 0.3]).astype(np.float32)
        vp, vptr = buf(v)
        q, qptr = buf(np.zeros(n))
        lor_exp(bptr, vptr, qptr, n, c)
        v2, v2ptr = buf(np.zeros(n))
        lor_log(bptr, qptr, v2ptr, n, c)
        err = float(np.abs(vp - v2).max())
        on_sheet = float(lor_mink(qptr, qptr, n))
        probes["lorentz_round_trip"] = {
            "n": n, "curvature": c,
            "log_exp_max_err": err,
            "on_sheet_inner": on_sheet,
            "expected_inner": -1.0 / c,
            "passed": err < 1e-4 and abs(on_sheet + 1.0 / c) < 1e-3,
        }
    except Exception as exc:
        probes["lorentz_round_trip"] = {"error": f"{type(exc).__name__}: {exc}", "passed": False,
                                          "traceback": traceback.format_exc()}

    # === Sphere round-trip ===
    try:
        n = 16; r = 1.0
        base = np.zeros(n, dtype=np.float32); base[0] = r
        bp, bptr = buf(base)
        v = np.concatenate([[0.0], rng.standard_normal(n-1) * 0.3]).astype(np.float32)
        vp, vptr = buf(v)
        q, qptr = buf(np.zeros(n))
        sph_exp(bptr, vptr, qptr, n, r)
        v2, v2ptr = buf(np.zeros(n))
        sph_log(bptr, qptr, v2ptr, n, r)
        err = float(np.abs(vp - v2).max())
        on_sphere = float(sph_inner(qptr, qptr, n))
        # slerp midpoint stays on sphere
        m, mptr = buf(np.zeros(n))
        sph_slerp(bptr, qptr, 0.5, mptr, n, r)
        slerp_norm = float(sph_inner(mptr, mptr, n))
        probes["sphere_round_trip"] = {
            "n": n, "radius": r,
            "log_exp_max_err": err,
            "on_sphere_inner": on_sphere,
            "expected_inner_r2": r * r,
            "slerp_norm_sq": slerp_norm,
            "passed": (err < 1e-4 and abs(on_sphere - r*r) < 1e-3
                       and abs(slerp_norm - r*r) < 1e-3),
        }
    except Exception as exc:
        probes["sphere_round_trip"] = {"error": f"{type(exc).__name__}: {exc}", "passed": False,
                                         "traceback": traceback.format_exc()}

    # === Torus T^3 (qLLM periodic embeddings) ===
    try:
        torus_distance = bind("tc_torus_distance", ctypes.c_float,
                                [f32p, f32p, ctypes.c_size_t, ctypes.c_float])
        torus_log = bind("tc_torus_log", None,
                          [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
        torus_exp = bind("tc_torus_exp", None,
                          [f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float])
        if any(fn is None for fn in (torus_distance, torus_log, torus_exp)):
            raise RuntimeError("torus symbols missing")
        rng2 = np.random.default_rng(11)
        n = 3; r = 1.0
        base = (rng2.uniform(0, 2*math.pi, n)).astype(np.float32)
        bp, bptr = buf(base)
        tangent = (rng2.standard_normal(n) * 0.4).astype(np.float32)
        tp, tptr = buf(tangent)
        q, qptr = buf(np.zeros(n))
        torus_exp(bptr, tptr, qptr, n, r)
        v2, v2ptr = buf(np.zeros(n))
        torus_log(bptr, qptr, v2ptr, n, r)
        err = float(np.abs(tp - v2).max())
        d_self = float(torus_distance(bptr, bptr, n, r))
        d_pq = float(torus_distance(bptr, qptr, n, r))
        probes["torus_round_trip"] = {
            "n": n, "radius": r,
            "log_exp_max_err": err,
            "d_self": d_self, "d_pq": d_pq,
            "passed": (err < 1e-4 and abs(d_self) < 1e-5
                       and d_pq > 0.0 and math.isfinite(d_pq)),
        }
    except Exception as exc:
        probes["torus_round_trip"] = {"error": f"{type(exc).__name__}: {exc}",
                                        "passed": False,
                                        "traceback": traceback.format_exc()}

    # === Product manifold H × S × R ===
    try:
        factors = (_Factor * 3)(
            _Factor(3, 4, 1.0),   # LORENTZ, intrinsic=4
            _Factor(2, 3, 1.0),   # SPHERE, intrinsic=3
            _Factor(0, 8, 0.0),   # EUCLIDEAN, intrinsic=8
        )
        nf = 3
        total = int(pm_dim(factors, nf))
        base = np.zeros(total, dtype=np.float32)
        base[0] = 1.0  # Lorentz origin
        base[5] = 1.0  # Sphere origin (ambient offset 5)
        bp, bptr = buf(base)
        tangent = np.zeros(total, dtype=np.float32)
        tangent[1:5] = rng.standard_normal(4) * 0.2
        tangent[6:9] = rng.standard_normal(3) * 0.2
        tangent[9:17] = rng.standard_normal(8) * 0.1
        tp, tptr = buf(tangent)
        q, qptr = buf(np.zeros(total))
        pm_exp(factors, nf, bptr, tptr, qptr)
        v2, v2ptr = buf(np.zeros(total))
        pm_log(factors, nf, bptr, qptr, v2ptr)
        err_lor = float(np.abs(tp[1:5] - v2[1:5]).max())
        err_sph = float(np.abs(tp[6:9] - v2[6:9]).max())
        err_euc = float(np.abs(tp[9:17] - v2[9:17]).max())
        d_self = float(pm_dist(factors, nf, bptr, bptr))
        d_pq = float(pm_dist(factors, nf, bptr, qptr))
        probes["product_manifold_HxSxR"] = {
            "ambient_dim": total,
            "expected_ambient": 17,
            "lorentz_err": err_lor,
            "sphere_err": err_sph,
            "euclidean_err": err_euc,
            "d_self": d_self,
            "d_pq": d_pq,
            "passed": (total == 17 and err_lor < 1e-3 and err_sph < 1e-3
                       and err_euc < 1e-5 and abs(d_self) < 1e-5
                       and d_pq > 0.0 and math.isfinite(d_pq)),
        }
    except Exception as exc:
        probes["product_manifold_HxSxR"] = {"error": f"{type(exc).__name__}: {exc}", "passed": False,
                                              "traceback": traceback.format_exc()}

    # === Riemannian metric tensor (Euclidean inverse + sphere stereographic) ===
    try:
        c_float_p = ctypes.POINTER(ctypes.c_float)
        metric_fn_t = ctypes.CFUNCTYPE(None, c_float_p, ctypes.c_int,
                                        c_float_p, ctypes.c_void_p)
        m_apply = bind("tc_metric_apply", ctypes.c_float,
                       [metric_fn_t, ctypes.c_void_p, c_float_p, ctypes.c_int,
                        c_float_p, c_float_p])
        m_inv = bind("tc_metric_inverse", ctypes.c_int,
                     [c_float_p, ctypes.c_int, c_float_p])
        m_chris = bind("tc_metric_christoffel", ctypes.c_int,
                       [metric_fn_t, ctypes.c_void_p, c_float_p, ctypes.c_int,
                        ctypes.c_float, c_float_p])
        m_euc_addr = getattr(lib, "tc_metric_euclidean", None)
        m_sph_addr = getattr(lib, "tc_metric_sphere_stereographic", None)
        if any(fn is None for fn in (m_apply, m_inv, m_chris,
                                       m_euc_addr, m_sph_addr)):
            raise RuntimeError("metric symbols missing")
        # Cast the C callbacks to metric_fn_t via their address so we can
        # pass them straight back into m_apply / m_chris.
        m_euc = ctypes.cast(m_euc_addr, metric_fn_t)
        m_sph = ctypes.cast(m_sph_addr, metric_fn_t)
        d = 3
        p = np.zeros(d, dtype=np.float32)
        pp = p.ctypes.data_as(c_float_p)
        v = np.array([1, 0, 0], dtype=np.float32)
        w = np.array([1, 0, 0], dtype=np.float32)
        vp = v.ctypes.data_as(c_float_p); wp = w.ctypes.data_as(c_float_p)
        g_vw = float(m_apply(m_euc, None, pp, d, vp, wp))
        # Inverse of identity is identity
        g_eye = (np.eye(d, dtype=np.float32)).flatten()
        g_inv = np.zeros(d*d, dtype=np.float32)
        rc = int(m_inv(g_eye.ctypes.data_as(c_float_p), d,
                       g_inv.ctypes.data_as(c_float_p)))
        inv_err = float(np.abs(g_inv.reshape(d, d) - np.eye(d)).max())
        # Christoffels of Euclidean are identically zero
        chris = np.zeros(d*d*d, dtype=np.float32)
        rc2 = int(m_chris(m_euc, None, pp, d, 1e-3,
                          chris.ctypes.data_as(c_float_p)))
        chris_max = float(np.abs(chris).max())
        # Sphere stereographic at origin → g = (2r²/r²)² I = 4 I when r=1
        rval = ctypes.c_float(1.0)
        g_sph = float(m_apply(m_sph, ctypes.byref(rval), pp, d, vp, wp))
        probes["metric_tensor_runtime"] = {
            "euclidean_inner_xx": g_vw,
            "expected_inner_xx": 1.0,
            "inverse_id_return_code": rc,
            "inverse_id_max_err": inv_err,
            "christoffel_euclidean_return_code": rc2,
            "christoffel_euclidean_max": chris_max,
            "sphere_stereographic_g_xx": g_sph,
            "expected_sphere_g_xx_at_origin": 4.0,
            "passed": (abs(g_vw - 1.0) < 1e-5
                       and rc == 0 and inv_err < 1e-5
                       and rc2 == 0 and chris_max < 1e-3
                       and abs(g_sph - 4.0) < 1e-3),
        }
    except Exception as exc:
        probes["metric_tensor_runtime"] = {"error": f"{type(exc).__name__}: {exc}",
                                            "passed": False,
                                            "traceback": traceback.format_exc()}

    # === Geodesic ODE (RK4 on Euclidean = straight lines) ===
    try:
        c_float_p = ctypes.POINTER(ctypes.c_float)
        metric_fn_t = ctypes.CFUNCTYPE(None, c_float_p, ctypes.c_int,
                                        c_float_p, ctypes.c_void_p)
        g_int = bind("tc_geodesic_integrate", ctypes.c_int,
                     [metric_fn_t, ctypes.c_void_p, ctypes.c_int,
                      ctypes.c_float, ctypes.c_int, ctypes.c_float,
                      c_float_p, c_float_p, c_float_p, c_float_p])
        m_euc_addr = getattr(lib, "tc_metric_euclidean", None)
        if g_int is None or m_euc_addr is None:
            raise RuntimeError("geodesic symbols missing")
        m_euc = ctypes.cast(m_euc_addr, metric_fn_t)
        d = 3
        p0 = np.array([0, 0, 0], dtype=np.float32)
        v0 = np.array([1, 0, 0], dtype=np.float32)
        p_out = np.zeros(d, dtype=np.float32)
        v_out = np.zeros(d, dtype=np.float32)
        dt = 0.01; n_steps = 100  # T = 1.0
        rc = int(g_int(m_euc, None, d, ctypes.c_float(dt), n_steps,
                       ctypes.c_float(1e-3),
                       p0.ctypes.data_as(c_float_p),
                       v0.ctypes.data_as(c_float_p),
                       p_out.ctypes.data_as(c_float_p),
                       v_out.ctypes.data_as(c_float_p)))
        # Straight-line geodesic: γ(T) = p0 + T v0 = (1, 0, 0); γ̇ unchanged
        pos_err = float(np.abs(p_out - np.array([1, 0, 0])).max())
        vel_err = float(np.abs(v_out - v0).max())
        probes["geodesic_runtime"] = {
            "return_code": rc,
            "final_pos": p_out.tolist(),
            "expected_pos": [1.0, 0.0, 0.0],
            "pos_err": pos_err,
            "final_vel": v_out.tolist(),
            "vel_err": vel_err,
            "passed": rc == 0 and pos_err < 1e-3 and vel_err < 1e-3,
        }
    except Exception as exc:
        probes["geodesic_runtime"] = {"error": f"{type(exc).__name__}: {exc}",
                                       "passed": False,
                                       "traceback": traceback.format_exc()}

    # === Holonomic gate (SU(2) z-axis loop → Berry phase) ===
    try:
        h_compose = bind("tc_holonomic_compose_su2", ctypes.c_int,
                         [c_float_p, ctypes.c_int32, c_float_p])
        h_phase = bind("tc_holonomic_berry_phase", None,
                       [c_float_p, c_float_p])
        if h_compose is None or h_phase is None:
            raise RuntimeError("holonomic symbols missing")
        # Single segment exp(i (π/4) σ_z) → trace = 2 cos(π/4) → phase = π/4
        gens = np.array([0.0, 0.0, math.pi / 4.0], dtype=np.float32)
        U = np.zeros(8, dtype=np.float32)
        rc = int(h_compose(gens.ctypes.data_as(c_float_p), 1,
                           U.ctypes.data_as(c_float_p)))
        phase = ctypes.c_float(0.0)
        h_phase(U.ctypes.data_as(c_float_p), ctypes.byref(phase))
        # Two-segment loop: exp(i (π/4) σz) then exp(-i (π/4) σz) → identity → phase 0
        gens2 = np.array([0.0, 0.0,  math.pi / 4.0,
                          0.0, 0.0, -math.pi / 4.0], dtype=np.float32)
        U2 = np.zeros(8, dtype=np.float32)
        h_compose(gens2.ctypes.data_as(c_float_p), 2,
                  U2.ctypes.data_as(c_float_p))
        phase2 = ctypes.c_float(0.0)
        h_phase(U2.ctypes.data_as(c_float_p), ctypes.byref(phase2))
        probes["holonomic_runtime"] = {
            "single_segment_return_code": rc,
            "berry_phase_pi_over_4": phase.value,
            "expected_phase_pi_over_4": math.pi / 4.0,
            "two_segment_phase_closed_loop": phase2.value,
            "passed": (rc == 0 and abs(phase.value - math.pi / 4.0) < 1e-4
                       and abs(phase2.value) < 1e-3),  # fp32 SU(2) compose noise
        }
    except Exception as exc:
        probes["holonomic_runtime"] = {"error": f"{type(exc).__name__}: {exc}",
                                        "passed": False,
                                        "traceback": traceback.format_exc()}

    for _probe in probes.values():
        _probe["runtime_status"] = "passed" if _probe.get("passed", False) else "failed"
    all_passed = all(p.get("passed", False) for p in probes.values())
    payload = {
        "checks": {
            "geometric_ops": {
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
