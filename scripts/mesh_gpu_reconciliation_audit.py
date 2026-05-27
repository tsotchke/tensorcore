#!/usr/bin/env python3
"""Run one scheduler-VM GPU reconciliation sweep plus scheduler audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mesh_resource_scheduler
import mesh_worker_gpu_reconcile_sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
DEFAULT_JOBS = ROOT / "configs" / "mesh_resource_jobs.json"
SCHEMA = "tensorcore.gpu_reconciliation_audit.v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def clean_reports_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.reconciliation.json"):
        if child.is_file():
            child.unlink()


def sweep_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        inventory_json=args.inventory_json,
        arbiter_status_json=args.arbiter_status_json,
        arbiter_cmd=args.arbiter_cmd,
        arbiter_timeout_sec=args.arbiter_timeout_sec,
        snapshot_json_dir=args.snapshot_json_dir,
        snapshot_timeout_sec=args.snapshot_timeout_sec,
        reports_dir=args.reports_dir,
        resource=args.resource,
        include_blocked=args.include_blocked,
        offline=args.offline,
    )


def audit_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        jobs_json=str(args.jobs_json),
        inventory_json=str(args.inventory_json),
        worker_reconciliation_json=[],
        worker_reconciliation_dir=[str(args.reports_dir)],
    )


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_reports_dir:
        clean_reports_dir(args.reports_dir)
    sweep = mesh_worker_gpu_reconcile_sweep.build_payload(sweep_args(args))
    if args.sweep_json:
        write_json(args.sweep_json, sweep)
    try:
        audit = mesh_resource_scheduler.cmd_audit(audit_args(args))
    except Exception as exc:
        audit = {
            "schema": "tensorcore.cluster_audit.result.v1",
            "ok": False,
            "checked_at_unix": time.time(),
            "errors": [f"scheduler audit failed: {exc}"],
            "job_count": 0,
            "worker_reconciliation_reports": [],
        }
    payload = {
        "schema": SCHEMA,
        "ok": bool(sweep.get("ok")) and bool(audit.get("ok")),
        "checked_at_unix": time.time(),
        "sweep": sweep,
        "audit": audit,
    }
    if args.audit_json:
        write_json(args.audit_json, payload)
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--jobs-json", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--arbiter-status-json", type=Path)
    parser.add_argument("--arbiter-cmd", default=mesh_worker_gpu_reconcile_sweep.DEFAULT_ARBITER_CMD)
    parser.add_argument("--arbiter-timeout-sec", type=float, default=10.0)
    parser.add_argument("--snapshot-json-dir", type=Path)
    parser.add_argument("--snapshot-timeout-sec", type=float, default=10.0)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--clean-reports-dir", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.arbiter_timeout_sec <= 0:
        parser.error("--arbiter-timeout-sec must be > 0")
    if args.snapshot_timeout_sec <= 0:
        parser.error("--snapshot-timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = run_once(args)
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(
            "GPU reconciliation audit ok: "
            f"{payload['sweep'].get('resource_count', 0)} resource(s)"
        )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
