#!/usr/bin/env python3
"""Reconcile worker GPU snapshots with central scheduler lease state."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "tensorcore.mesh_worker_gpu_reconciliation.v1"
SNAPSHOT_SCHEMA = "tensorcore.mesh_worker_gpu_snapshot.v1"


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def active_leases_for_resource(status: dict[str, Any], resource: str) -> list[dict[str, Any]]:
    leases = status.get("leases")
    if not isinstance(leases, list):
        return []
    out = []
    for lease in leases:
        if not isinstance(lease, dict):
            continue
        if lease.get("resource") != resource:
            continue
        if str(lease.get("status") or "active") in {"released", "expired", "cancelled"}:
            continue
        out.append(lease)
    return out


def cuda_apps(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    apps = snapshot.get("cuda_apps")
    if not isinstance(apps, list):
        return []
    return [app for app in apps if isinstance(app, dict) and isinstance(app.get("pid"), int)]


def classify_apps(
    apps: list[dict[str, Any]],
    *,
    allow_patterns: list[re.Pattern[str]],
    allowed_process_max_memory_mib: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = []
    unmanaged = []
    for app in apps:
        process_name = str(app.get("process_name") or app.get("raw") or "")
        used_memory = app.get("used_memory_mib")
        parse_ok = app.get("pid") is not None and isinstance(used_memory, int)
        pattern_ok = any(pattern.search(process_name) for pattern in allow_patterns)
        memory_ok = isinstance(used_memory, int) and used_memory <= allowed_process_max_memory_mib
        if parse_ok and pattern_ok and memory_ok:
            allowed.append(app)
        else:
            unmanaged.append(app)
    return allowed, unmanaged


def reconcile(
    snapshot: dict[str, Any],
    status: dict[str, Any],
    *,
    resource: str,
    allow_process_regex: list[str] | None = None,
    allowed_process_max_memory_mib: int = 64,
) -> dict[str, Any]:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    apps = cuda_apps(snapshot)
    patterns = [re.compile(pattern) for pattern in allow_process_regex or []]
    allowed_apps, unmanaged_apps = classify_apps(
        apps,
        allow_patterns=patterns,
        allowed_process_max_memory_mib=allowed_process_max_memory_mib,
    )
    leases = active_leases_for_resource(status, resource)
    errors: list[dict[str, Any]] = []
    action = "none"
    reason = "ok"
    if not snapshot.get("ok"):
        reason = "worker_snapshot_unhealthy"
        action = "drain"
        errors.append({"code": reason, "resource": resource, "snapshot_reason": snapshot.get("reason")})
    elif unmanaged_apps and not leases:
        reason = "stale_unknown_unleased_cuda"
        action = "drain"
        errors.append(
            {
                "code": reason,
                "resource": resource,
                "cuda_pids": [app["pid"] for app in unmanaged_apps],
            }
        )
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "reason": reason,
        "action": action,
        "checked_at_unix": time.time(),
        "resource": resource,
        "worker_host": snapshot.get("worker_host"),
        "cuda_app_count": len(apps),
        "cuda_pids": [app["pid"] for app in apps],
        "allowed_cuda_apps": allowed_apps,
        "unmanaged_cuda_apps": unmanaged_apps,
        "allowed_process_max_memory_mib": allowed_process_max_memory_mib,
        "allow_process_regex": allow_process_regex or [],
        "active_lease_ids": [lease.get("id") or lease.get("lease_id") for lease in leases],
        "active_lease_count": len(leases),
        "errors": errors,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--arbiter-status-json", type=Path, required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument(
        "--allow-process-regex",
        action="append",
        default=[],
        help="Allow matching CUDA process names below the memory cap.",
    )
    parser.add_argument("--allowed-process-max-memory-mib", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.allowed_process_max_memory_mib < 0:
        parser.error("--allowed-process-max-memory-mib must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = reconcile(
        read_json_object(args.snapshot_json),
        read_json_object(args.arbiter_status_json),
        resource=args.resource,
        allow_process_regex=args.allow_process_regex,
        allowed_process_max_memory_mib=args.allowed_process_max_memory_mib,
    )
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{payload['resource']}: GPU reconciliation ok")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
