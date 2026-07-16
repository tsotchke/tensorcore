#!/usr/bin/env python3
"""Run the complete topology-authority gate set and emit ICC JSONL evidence."""

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
SCHEMA = "tensorcore.topology_authority_sprint.v1"
EVENT_KIND = "topology_authority_gate"
GATE_NAMES = (
    "canonical_schema",
    "source_coverage",
    "identity_reconciliation",
    "drift_fail_closed",
    "scheduler_binding",
    "signed_snapshot",
    "public_redaction",
    "cross_platform_selftest",
    "documentation_truth",
)


def run_command(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "output_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "output_tail": combined[-1200:],
    }


def git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def write_trace(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-output",
        type=pathlib.Path,
        default=ROOT / "build" / "topology-authority-gates.jsonl",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-tracked-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable
    checks = {
        "authority": run_command([python, "scripts/topology_authority_selftest.py"]),
        "observation_wrapper": run_command([python, "scripts/wrap_topology_observation_selftest.py"]),
        "scheduler_binding": run_command([python, "scripts/topology_scheduler_binding_selftest.py"]),
        "documentation": run_command([python, "scripts/check_topology_authority_docs.py"]),
        "documentation_links": run_command([python, "scripts/check_docs_links.py"]),
        "inventory": run_command([python, "scripts/check_mesh_resource_inventory.py"]),
        "jobs": run_command([python, "scripts/check_mesh_resource_jobs.py"]),
    }
    tracked_dirty = bool(git_value("status", "--porcelain", "--untracked-files=no"))
    head = git_value("rev-parse", "HEAD")
    gate_checks = {
        "canonical_schema": ("authority", "inventory"),
        "source_coverage": ("authority", "observation_wrapper"),
        "identity_reconciliation": ("authority",),
        "drift_fail_closed": ("authority",),
        "scheduler_binding": ("scheduler_binding", "inventory", "jobs"),
        "signed_snapshot": ("authority", "scheduler_binding"),
        "public_redaction": ("authority", "documentation"),
        "cross_platform_selftest": ("authority", "observation_wrapper", "documentation"),
        "documentation_truth": ("documentation", "documentation_links"),
    }
    snippets = {
        "canonical_schema": "canonical policy/snapshot schemas and checked scheduler inventory passed",
        "source_coverage": "complete manifested source coverage and timestamped observation envelopes passed",
        "identity_reconciliation": "stable aliases, unmanaged identities, and alias-conflict cases passed",
        "drift_fail_closed": "unknown, stale, future, conflicting, incomplete, and mismatched inputs failed closed",
        "scheduler_binding": "scheduler rejected missing, stale, future, tampered, drifted, and unadmitted snapshots",
        "signed_snapshot": "HMAC-SHA256 round trip, scheduler verification, and tamper rejection passed",
        "public_redaction": "public projection omitted private aliases/source hashes and denied sensitive key classes",
        "cross_platform_selftest": "portable stdlib suites passed locally and are required on Ubuntu/macOS/Windows CI",
        "documentation_truth": "operator contract and all local Markdown links passed machine checks",
    }
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events = []
    for gate in GATE_NAMES:
        required_checks = gate_checks[gate]
        passed = all(checks[name]["ok"] for name in required_checks)
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
                "checks": list(required_checks),
                "snippet": snippets[gate]
                if passed
                else f"{gate} failed: "
                + ", ".join(name for name in required_checks if not checks[name]["ok"]),
            }
        )
    write_trace(args.trace_output, events)
    ok = all(event["value"] == "PASS" for event in events)
    report = {
        "schema": SCHEMA,
        "ok": ok,
        "git_head": head,
        "git_tracked_dirty": tracked_dirty,
        "trace_output": str(args.trace_output),
        "passed_gates": sum(event["value"] == "PASS" for event in events),
        "gate_count": len(events),
        "checks": checks,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for event in events:
            print(f"{event['value']} {event['name']}: {event['snippet']}")
        print(f"topology authority sprint {'OK' if ok else 'FAIL'}: {report['passed_gates']}/{report['gate_count']} gates")
        print(f"ICC trace: {args.trace_output}")
        if not ok:
            for name, check in checks.items():
                if not check["ok"]:
                    print(f"FAILED {name}: {check['output_tail']}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
