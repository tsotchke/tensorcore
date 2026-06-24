#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for tensorcore's mesh collectives.

Runs the test_mesh_collective binary (fork-based 2-peer smoke over
tc_remote_tensor_fetch on localhost) and translates its exit code +
output into structured JSON keyed under `checks.mesh_collective.*`.

The actual op probes live in the C test (tests/test_mesh_collective.c)
because fork-based testing through ctypes-driven Python is messier than
direct C. ICC just needs to see PASS/FAIL plus per-op coverage.

Usage:
    python3 scripts/check_mesh_collective_runtime.py \\
        --out build/mesh_collective_runtime_evidence.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "mesh_collective_runtime_evidence.json"))
    p.add_argument("--bin", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "tests" / "test_mesh_collective"))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"mesh_collective": {"runtime_status": "skipped_no_binary",
                                                "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped_no_binary reason={reason}")
    return 0


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    binary = Path(args.bin).expanduser()

    if not binary.exists() or not os.access(binary, os.X_OK):
        return _skip(out_path, f"test_mesh_collective not built: {binary}")

    try:
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        payload = {"checks": {"mesh_collective": {"runtime_status": "failed",
                                                    "failure_reason": "timeout"}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    stdout = result.stdout
    stderr = result.stderr
    rc = result.returncode

    # Parse per-op PASS lines from stdout.
    expected_ops = ["AllReduce SUM", "AllReduce AVG", "Broadcast", "AllGather"]
    probes = {}
    for op in expected_ops:
        key = op.lower().replace(" ", "_")
        ok = f"PASS rank 0 {op}" in stdout
        probes[key] = {
            "op": op,
            "passed": ok,
            "runtime_status": "passed" if ok else "failed",
        }

    all_passed = (rc == 0) and all(p["passed"] for p in probes.values())

    payload = {
        "checks": {
            "mesh_collective": {
                "runtime_status": "passed" if all_passed else "failed",
                "test_rc": rc,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
                "probes": probes,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "passed" if all_passed else "failed"
    print(f"runtime_status={status} ops={list(probes.keys())}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
