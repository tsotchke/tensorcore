#!/usr/bin/env python3
"""Reconcile worker GPU snapshots with central scheduler lease state."""

from __future__ import annotations

import argparse
import json
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


def reconcile(snapshot: dict[str, Any], status: dict[str, Any], *, resource: str) -> dict[str, Any]:
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    apps = cuda_apps(snapshot)
    leases = active_leases_for_resource(status, resource)
    errors: list[dict[str, Any]] = []
    action = "none"
    reason = "ok"
    if apps and not leases:
        reason = "stale_unknown_unleased_cuda"
        action = "drain"
        errors.append(
            {
                "code": reason,
                "resource": resource,
                "cuda_pids": [app["pid"] for app in apps],
            }
        )
    elif not snapshot.get("ok"):
        reason = "worker_snapshot_unhealthy"
        action = "drain"
        errors.append({"code": reason, "resource": resource, "snapshot_reason": snapshot.get("reason")})
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
        "active_lease_ids": [lease.get("id") or lease.get("lease_id") for lease in leases],
        "active_lease_count": len(leases),
        "errors": errors,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--arbiter-status-json", type=Path, required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = reconcile(
        read_json_object(args.snapshot_json),
        read_json_object(args.arbiter_status_json),
        resource=args.resource,
    )
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{payload['resource']}: GPU reconciliation ok")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
