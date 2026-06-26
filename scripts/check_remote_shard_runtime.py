#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for tensorcore's remote_shard.

Runs the test_remote_shard binary (fork-based 2-peer smoke that exercises
tc_remote_shard_register + tc_remote_shard_owner +
tc_remote_shard_local_range + tc_remote_shard_get over the tc_remote
transport on localhost) and translates its PASS lines into structured JSON
keyed under `checks.remote_shard.*`.

Usage:
    python3 scripts/check_remote_shard_runtime.py \\
        --out build/remote_shard_runtime_evidence.json
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
                                         / "build" / "remote_shard_runtime_evidence.json"))
    p.add_argument("--bin", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "tests" / "test_remote_shard"))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"remote_shard": {"runtime_status": "skipped_no_binary",
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
        return _skip(out_path, f"test_remote_shard not built: {binary}")

    try:
        result = subprocess.run([str(binary)], capture_output=True, text=True,
                                  timeout=60)
    except subprocess.TimeoutExpired:
        payload = {"checks": {"remote_shard": {"runtime_status": "failed",
                                                 "failure_reason": "timeout"}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    stdout = result.stdout
    stderr = result.stderr
    rc = result.returncode

    # Parse per-op PASS lines from stdout. Rank 0 prints all PASS markers;
    # rank 1's success is implied by the final "ALL PASS" line + rc==0.
    expected_ops = ["shard_owner balanced",
                     "shard_get full_range",
                     "shard_get cross_owner_range",
                     "shard_drain_puts applied=1"]
    probes = {}
    for op in expected_ops:
        key = op.lower().replace(" ", "_")
        ok = f"PASS rank 0 {op}" in stdout
        probes[key] = {
            "op": op,
            "passed": ok,
            "runtime_status": "passed" if ok else "failed",
        }

    both_ranks_ok = ("rank0 fails=0" in stdout
                     and "rank1 fails=0" in stdout
                     and "ALL PASS" in stdout)
    probes["fork_2peer_both_ranks_ok"] = {
        "op": "fork-2peer rank0+rank1 zero fails",
        "passed": both_ranks_ok,
        "runtime_status": "passed" if both_ranks_ok else "failed",
    }

    all_passed = (rc == 0) and all(p["passed"] for p in probes.values())

    payload = {
        "checks": {
            "remote_shard": {
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
