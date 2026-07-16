#!/usr/bin/env python3
"""Run Apple-family sprint gates and emit ICC-readable JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.apple_family_runtime_sprint.v1"
EVENT_KIND = "apple_family_gate"
GATES = (
    "official_family_contract",
    "selector_selftest",
    "m2_fallback_runtime",
    "m4_runtime",
    "m5_tensorops_runtime",
    "cross_platform_selftest",
    "documentation_truth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build-apple-family")
    parser.add_argument(
        "--m2-evidence",
        type=pathlib.Path,
        default=ROOT / "build-apple-family" / "apple_family_runtime_evidence.json",
    )
    parser.add_argument("--m4-evidence", type=pathlib.Path)
    parser.add_argument("--m5-evidence", type=pathlib.Path)
    parser.add_argument(
        "--trace-output",
        type=pathlib.Path,
        default=ROOT / "build-apple-family" / "apple-family-runtime-gates.jsonl",
    )
    parser.add_argument("--require-tracked-clean", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = "\n".join(value.strip() for value in (proc.stdout, proc.stderr) if value.strip())
    return {
        "argv": argv,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": combined[-1600:],
    }


def git_value(*args: str) -> str:
    result = run(["git", *args])
    return result["output_tail"].strip() if result["ok"] else ""


def physical_check(path: pathlib.Path | None, chip: str, head: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "argv": [],
            "ok": False,
            "returncode": None,
            "output_sha256": hashlib.sha256(b"").hexdigest(),
            "output_tail": f"physical {chip} evidence missing",
        }
    return run(
        [
            sys.executable,
            "scripts/check_apple_family_runtime_evidence.py",
            str(path),
            "--git-head",
            head,
            "--require-chip",
            chip,
            "--require-clean-head",
            "--require-pass",
        ]
    )


def cross_platform_check() -> dict[str, Any]:
    fixture = run([sys.executable, "scripts/check_apple_family_runtime_evidence_selftest.py"])
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cmake = (ROOT / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")
    required = ("ubuntu-latest", "macos-14", "windows-latest", "portable-cpu", "portable-windows")
    missing = [token for token in required if token not in workflow]
    if "test_apple_family_policy" not in cmake:
        missing.append("tests/CMakeLists.txt:test_apple_family_policy")
    if missing:
        fixture["ok"] = False
        fixture["returncode"] = 1
        fixture["output_tail"] += "\nmissing cross-platform contracts: " + ", ".join(missing)
    return fixture


def write_trace(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    head = git_value("rev-parse", "HEAD")
    tracked_dirty = bool(git_value("status", "--porcelain", "--untracked-files=no"))
    selector = args.build_dir.resolve() / "tests" / "test_apple_family_policy"
    checks = {
        "official_family_contract": run([sys.executable, "scripts/check_apple_family_docs.py"]),
        "selector_selftest": run([str(selector)]),
        "m2_fallback_runtime": physical_check(args.m2_evidence.resolve(), "M2", head),
        "m4_runtime": physical_check(args.m4_evidence.resolve() if args.m4_evidence else None, "M4", head),
        "m5_tensorops_runtime": physical_check(args.m5_evidence.resolve() if args.m5_evidence else None, "M5", head),
        "cross_platform_selftest": cross_platform_check(),
        "documentation_truth": run([sys.executable, "scripts/check_docs_links.py"]),
    }
    snippets = {
        "official_family_contract": "Apple tables, public MSL types, reserved ABI, source selector, and M4 authority policy agree",
        "selector_selftest": "portable selector maps M4=Apple9, M5=Apple10, Apple11=reserved, and integer matrix=false",
        "m2_fallback_runtime": "physical clean-head M2 proves Apple8, BF16 fallback, integer MPS, and TensorOps-negative selection",
        "m4_runtime": "authorized physical M4 proves Apple9, BF16 simdgroup, integer MPS, and TensorOps-negative selection",
        "m5_tensorops_runtime": "physical SDK26 M5 proves Apple10, floating matrix paths, integer MPS, and TensorOps execution",
        "cross_platform_selftest": "portable fixtures and selector CTest are wired across Ubuntu, macOS, and Windows CI",
        "documentation_truth": "all public local documentation links resolve",
    }
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events: list[dict[str, Any]] = []
    for gate in GATES:
        check = checks[gate]
        passed = bool(check["ok"])
        if args.require_tracked_clean and tracked_dirty:
            passed = False
        events.append(
            {
                "schema": SCHEMA,
                "kind": EVENT_KIND,
                "name": gate,
                "value": "PASS" if passed else "FAIL",
                "status": "PASS" if passed else "FAIL",
                "timestamp": timestamp,
                "git_head": head,
                "git_tracked_dirty": tracked_dirty,
                "checks": [gate],
                "snippet": snippets[gate] if passed else check["output_tail"],
            }
        )
    write_trace(args.trace_output, events)
    passed_count = sum(event["value"] == "PASS" for event in events)
    ok = passed_count == len(events)
    report = {
        "schema": SCHEMA,
        "ok": ok,
        "git_head": head,
        "git_tracked_dirty": tracked_dirty,
        "passed_gates": passed_count,
        "gate_count": len(events),
        "trace_output": str(args.trace_output),
        "checks": checks,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for event in events:
            print(f"{event['value']} {event['name']}: {event['snippet']}")
        print(f"Apple family runtime sprint {'OK' if ok else 'BLOCKED'}: {passed_count}/{len(events)} gates")
        print(f"ICC trace: {args.trace_output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
