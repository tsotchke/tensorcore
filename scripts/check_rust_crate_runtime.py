#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for the tensorcore-rs crate.

Runs `cargo test` against bindings/rust/tensorcore-rs/ (the Rust
wrapper around libtensorcore.{dylib,so}) and translates per-test
pass/fail into structured JSON keyed under `checks.rust_crate.*`.

The Rust crate is one of the consumers of the tensorcore substrate;
this smoke proves the FFI surface is wired correctly + Rust callers
get the same numerics as Python / Eshkol.

Usage:
    python3 scripts/check_rust_crate_runtime.py \\
        --out build/rust_crate_runtime_evidence.json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "rust_crate_runtime_evidence.json"))
    p.add_argument("--crate", default=str(Path(__file__).resolve().parent.parent
                                            / "bindings" / "rust" / "tensorcore-rs"))
    p.add_argument("--lib", default=os.environ.get("TENSORCORE_LIB",
                                                     str(Path(__file__).resolve().parent.parent
                                                         / "build" / "libtensorcore.dylib")))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"rust_crate": {"runtime_status": "skipped",
                                            "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped reason={reason}")
    return 0


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    crate = Path(args.crate).expanduser()
    lib_path = Path(args.lib).expanduser()

    if shutil.which("cargo") is None:
        return _skip(out_path, "cargo not installed")
    if not crate.exists():
        return _skip(out_path, f"crate directory missing: {crate}")
    if not lib_path.exists():
        return _skip(out_path, f"libtensorcore missing: {lib_path}; run cmake build first")

    env = {**os.environ, "TENSORCORE_LIB_DIR": str(lib_path.parent),
            "CARGO_TERM_COLOR": "never"}
    try:
        # No --quiet: we need the per-test `test name ... ok` lines so we
        # can emit per-test probe entries in the ICC JSON.
        result = subprocess.run(["cargo", "test"],
                                  cwd=str(crate), capture_output=True, text=True,
                                  timeout=180, env=env)
    except subprocess.TimeoutExpired:
        payload = {"checks": {"rust_crate": {"runtime_status": "failed",
                                                "failure_reason": "cargo test timeout"}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    stdout = result.stdout
    stderr = result.stderr
    rc = result.returncode

    # Parse `test result: ok. N passed; M failed; K ignored; ...` lines.
    summary = re.findall(r"test result:\s*(\w+)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed",
                            stdout)
    total_passed = sum(int(p) for _, p, _ in summary)
    total_failed = sum(int(f) for _, _, f in summary)
    overall_ok = rc == 0 and total_failed == 0 and total_passed > 0

    # Per-test PASS lines look like: `test foo ... ok` / `test foo ... FAILED`
    per_test = re.findall(r"^test\s+([\w:]+)\s+\.\.\.\s+(ok|FAILED|ignored)\s*$",
                            stdout, flags=re.MULTILINE)
    probes = {}
    for name, status in per_test:
        key = name.replace("::", "_")
        passed = status == "ok"
        probes[key] = {
            "test": name,
            "passed": passed,
            "runtime_status": "passed" if passed else "failed",
        }

    payload = {
        "checks": {
            "rust_crate": {
                "runtime_status": "passed" if overall_ok else "failed",
                "test_rc": rc,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "probes": probes,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    # Embed stdout/stderr only on failure — cargo's normal-success output
    # contains substrings like "1 test compiled" that ICC's free-text
    # event extractor sometimes mistakes for failure tracebacks, so keep
    # the trace clean on the happy path.
    if not overall_ok:
        payload["checks"]["rust_crate"]["stdout_tail"] = stdout[-2000:]
        payload["checks"]["rust_crate"]["stderr_tail"] = stderr[-2000:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "passed" if overall_ok else "failed"
    print(f"runtime_status={status} passed={total_passed} failed={total_failed}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
