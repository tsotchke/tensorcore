#!/usr/bin/env python3
"""bench_phase4.py — wall-time benchmarks for the Phase 4 substrate ops.

Phase 4 added Riemannian metric tensor + numerical Christoffels, the RK4
geodesic integrator, SU(2) holonomic gate composition + Berry-phase
extraction. This bench measures the per-call wall time of each new op
across realistic problem sizes so consumers know what they're paying.

Targets (informed by sibling-repo workloads on Apple M2 Ultra):
  - metric_christoffel  : ≤ 0.5 ms at d = 8 (Noesis embedding manifold)
  - geodesic_integrate  : ≤ 10 ms for 100-step path at d = 8
  - holonomic_compose   : ≤ 50 µs per 32-segment SU(2) loop
  - holonomic_phase     : ≤ 5 µs per extraction

Usage:
    TENSORCORE_LIB=$PWD/build/libtensorcore.dylib python3 bench/bench_phase4.py

Outputs a markdown table to stdout + optionally a JSON dump.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", default=os.environ.get("TENSORCORE_LIB",
                                                     str(Path(__file__).resolve().parent.parent
                                                         / "build" / "libtensorcore.dylib")))
    p.add_argument("--repeats", type=int, default=50,
                   help="number of iterations per measurement (default 50)")
    p.add_argument("--out-json", default=None,
                   help="optional path to dump full results as JSON")
    return p.parse_args()


def median_us(fn, repeats):
    """Run `fn` `repeats` times; return (median_us, p95_us)."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[max(0, int(len(samples) * 0.95) - 1)]
    return p50, p95


def main():
    args = parse_args()
    os.environ.setdefault("TENSORCORE_LIB", args.lib)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    import numpy as np
    import tensorcore as tc

    results = []

    def record(op, params, fn):
        # Warmup so allocations and any JIT effects settle before timing.
        for _ in range(5):
            fn()
        p50, p95 = median_us(fn, args.repeats)
        results.append({"op": op, "params": params,
                         "p50_us": round(p50, 2), "p95_us": round(p95, 2)})

    # --- metric_christoffel across d = {2, 4, 8, 16} on a Poincaré ball ---
    rng = np.random.default_rng(0)
    for d in (2, 4, 8, 16):
        m_p, c_holder = tc.metric_poincare(0.5)
        import ctypes
        user = ctypes.addressof(c_holder)
        point = (rng.uniform(-0.3, 0.3, d)).astype(np.float32)
        def go(m_p=m_p, point=point, user=user, d=d):
            # Allocate the output buffer inside the timed region too — that
            # matches how consumers actually call it.
            chris = np.zeros(d * d * d, dtype=np.float32)
            from ctypes import c_int, c_float, c_void_p, POINTER
            tc._lib.tc_metric_christoffel(m_p, ctypes.c_void_p(user),
                                          point.ctypes.data_as(POINTER(c_float)),
                                          c_int(d), c_float(1e-3),
                                          chris.ctypes.data_as(POINTER(c_float)))
        record("metric_christoffel", f"d={d} (Poincaré, c=0.5)", go)

    # --- geodesic_integrate: 100 steps on Poincaré ball for d = {2, 4, 8} ---
    for d in (2, 4, 8):
        m_p, c_holder = tc.metric_poincare(0.5)
        user = ctypes.addressof(c_holder)
        pos0 = (rng.uniform(-0.2, 0.2, d)).astype(np.float32)
        vel0 = (rng.standard_normal(d) * 0.1).astype(np.float32)
        def go(m_p=m_p, pos0=pos0, vel0=vel0, user=user):
            from ctypes import c_void_p
            tc.geodesic_integrate(m_p, pos0, vel0, 0.01, 100,
                                    h_christoffel=1e-3, user_ptr=user)
        record("geodesic_integrate", f"d={d} steps=100 (Poincaré)", go)

    # Euclidean geodesic (Christoffels identically zero — lighter inner work).
    for d in (2, 4, 8):
        m_e = tc.metric_euclidean()
        pos0 = np.zeros(d, dtype=np.float32)
        vel0 = np.ones(d, dtype=np.float32) * 0.1
        def go(m_e=m_e, pos0=pos0, vel0=vel0):
            tc.geodesic_integrate(m_e, pos0, vel0, 0.01, 100,
                                    h_christoffel=1e-3)
        record("geodesic_integrate", f"d={d} steps=100 (Euclidean)", go)

    # --- holonomic_compose at varying segment counts ---
    for n in (4, 16, 64):
        gens = rng.standard_normal((n, 3)).astype(np.float32) * 0.1
        def go(gens=gens):
            tc.holonomic_compose_su2(gens)
        record("holonomic_compose_su2", f"n_segments={n}", go)

    # --- holonomic_berry_phase (pure extraction; tiny per call) ---
    U = tc.holonomic_compose_su2([[0.1, -0.2, 0.3]])
    def go():
        tc.holonomic_berry_phase(U)
    record("holonomic_berry_phase", "1 call", go)

    # --- metric_inverse at d = {2, 4, 8, 16} on a random spd matrix ---
    for d in (2, 4, 8, 16):
        A = rng.standard_normal((d, d)).astype(np.float32)
        spd = (A @ A.T + np.eye(d, dtype=np.float32)).astype(np.float32)
        def go(spd=spd, d=d):
            tc.metric_inverse(spd)
        record("metric_inverse", f"d={d} (SPD)", go)

    # ---- Print markdown table ----
    print("\n## Phase 4 substrate benchmarks")
    print(f"libtensorcore: {args.lib}")
    print(f"version: {tc.version()}")
    print(f"repeats per measurement: {args.repeats}\n")
    print("| op | params | p50 (µs) | p95 (µs) |")
    print("|---|---|---:|---:|")
    for r in results:
        print(f"| `{r['op']}` | {r['params']} | {r['p50_us']:.2f} | {r['p95_us']:.2f} |")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "lib": args.lib,
            "version": tc.version(),
            "repeats": args.repeats,
            "results": results,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2) + "\n")
        print(f"\nJSON: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
