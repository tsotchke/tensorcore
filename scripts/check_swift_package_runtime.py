#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for the TensorCore Swift package.

Runs `swift test` against bindings/swift/TensorCore/, parses per-test
PASS/FAIL into structured JSON keyed under `checks.swift_package.*`.

Skips cleanly when swift isn't on the host (Linux / Windows CI).

Usage:
    python3 scripts/check_swift_package_runtime.py \\
        --out build/swift_package_runtime_evidence.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "swift_package_runtime_evidence.json"))
    p.add_argument("--pkg", default=str(Path(__file__).resolve().parent.parent
                                          / "bindings" / "swift" / "TensorCore"))
    p.add_argument("--lib-dir", default=str(Path(__file__).resolve().parent.parent / "build"))
    p.add_argument("--include-dir", default=str(Path(__file__).resolve().parent.parent / "include"))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"swift_package": {"runtime_status": "skipped",
                                              "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped reason={reason}")
    return 0


def main():
    args = parse_args()
    out_path = Path(args.out).expanduser()
    pkg = Path(args.pkg).expanduser()

    if shutil.which("swift") is None:
        return _skip(out_path, "swift not installed (Apple toolchain required)")
    if not pkg.exists():
        return _skip(out_path, f"swift package dir missing: {pkg}")
    if not Path(args.lib_dir, "libtensorcore.dylib").exists():
        return _skip(out_path, f"libtensorcore.dylib missing in {args.lib_dir}")

    cmd = ["swift", "test",
            "-Xcc", f"-I{args.include_dir}",
            "-Xlinker", f"-L{args.lib_dir}",
            "-Xlinker", "-rpath", "-Xlinker", args.lib_dir]
    try:
        proc = subprocess.run(cmd, cwd=str(pkg), capture_output=True,
                                text=True, timeout=180)
    except subprocess.TimeoutExpired:
        payload = {"checks": {"swift_package": {"runtime_status": "failed",
                                                  "failure_reason": "swift test timeout"}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    # XCTest output looks like:
    #   Test Case '-[TensorCoreTests.TensorCoreTests testLorentzRoundtrip]' passed (0.001 seconds).
    # Pull (TestSuite, TestName, status) triples.
    per_test = re.findall(
        r"Test Case '-\[(\w+)\.(\w+)\s+(\w+)\]' (passed|failed)",
        proc.stdout + proc.stderr)
    probes = {}
    n_pass = 0
    n_fail = 0
    for _suite_module, _suite_class, name, status in per_test:
        passed = status == "passed"
        probes[name] = {"test": name, "passed": passed,
                          "runtime_status": "passed" if passed else "failed"}
        if passed: n_pass += 1
        else:      n_fail += 1

    overall_ok = (proc.returncode == 0 and n_fail == 0 and n_pass > 0)
    payload = {
        "checks": {
            "swift_package": {
                "runtime_status": "passed" if overall_ok else "failed",
                "test_rc": proc.returncode,
                "total_passed": n_pass,
                "total_failed": n_fail,
                "probes": probes,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    # Embed stderr/stdout only on failure (consistent with the
    # rust-crate smoke pattern — avoids ICC's free-text extractor
    # misreading success output).
    if not overall_ok:
        payload["checks"]["swift_package"]["stdout_tail"] = proc.stdout[-2000:]
        payload["checks"]["swift_package"]["stderr_tail"] = proc.stderr[-2000:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status={'passed' if overall_ok else 'failed'} passed={n_pass}/{n_pass + n_fail}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
