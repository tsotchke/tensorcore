#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for the tensorcore WASM bindings.

Runs `node bindings/wasm/test/smoke.mjs` against the pre-built
bindings/wasm/dist/ artifact, parses its stdout JSON tail into the
canonical checks.wasm_bindings.* shape.

Skips cleanly if either node or the built artifact is missing.

Usage:
    python3 scripts/check_wasm_bindings_runtime.py \\
        --out build/wasm_bindings_runtime_evidence.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "wasm_bindings_runtime_evidence.json"))
    p.add_argument("--smoke", default=str(Path(__file__).resolve().parent.parent
                                            / "bindings" / "wasm" / "test" / "smoke.mjs"))
    p.add_argument("--dist", default=str(Path(__file__).resolve().parent.parent
                                           / "bindings" / "wasm" / "dist"))
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"wasm_bindings": {"runtime_status": "skipped",
                                              "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped reason={reason}")
    return 0


def find_node():
    # Prefer system node, then emsdk-bundled node (which we use to
    # build the wasm).
    for p in ("node",):
        if shutil.which(p): return p
    emsdk_node = Path.home() / "emsdk" / "node"
    if emsdk_node.exists():
        for sub in sorted(emsdk_node.iterdir()):
            candidate = sub / "bin" / "node"
            if candidate.exists(): return str(candidate)
    return None


def main():
    args = parse_args()
    out_path = Path(args.out).expanduser()
    smoke = Path(args.smoke).expanduser()
    dist = Path(args.dist).expanduser()

    node = find_node()
    if not node:
        return _skip(out_path, "node not installed")
    if not smoke.exists():
        return _skip(out_path, f"smoke missing: {smoke}")
    if not (dist / "tensorcore.wasm").exists():
        return _skip(out_path, f"tensorcore.wasm missing: run bindings/wasm/build.sh first")

    proc = subprocess.run([node, str(smoke)], capture_output=True,
                            text=True, timeout=60)
    # The smoke prints a final JSON line; pull the last { ... } block.
    last_json = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "wasm_bindings" in line:
            try:
                last_json = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not last_json:
        payload = {"checks": {"wasm_bindings": {"runtime_status": "failed",
                                                  "failure_reason": "no JSON payload from smoke",
                                                  "stdout_tail": proc.stdout[-2000:]}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(last_json, indent=2, sort_keys=True) + "\n")
    top = last_json["checks"]["wasm_bindings"]["runtime_status"]
    print(f"runtime_status={top} passed={last_json['checks']['wasm_bindings'].get('total_passed')}")
    return 0 if top == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
