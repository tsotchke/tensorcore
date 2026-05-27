#!/usr/bin/env python3
"""Run GPU snapshot/reconciliation across Tensorcore CUDA inventory resources."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mesh_worker_gpu_reconcile
import mesh_worker_gpu_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
DEFAULT_ARBITER_CMD = os.environ.get("TC_MESH_ARBITER_CMD", "")
SCHEMA = "tensorcore.mesh_worker_gpu_reconciliation_sweep.v1"


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_resource_name(resource: str) -> str:
    return resource.replace(":", "_").replace("/", "_")


def load_inventory(path: Path) -> list[dict[str, Any]]:
    payload = read_json_object(path)
    if payload.get("schema") != "tensorcore.mesh_resources.v1":
        raise ValueError("--inventory-json schema must be tensorcore.mesh_resources.v1")
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ValueError("--inventory-json resources must be a list")
    out = []
    for row in resources:
        if not isinstance(row, dict):
            raise ValueError("--inventory-json resources contains a non-object row")
        out.append(row)
    return out


def active_scheduler_cuda_resources(
    rows: list[dict[str, Any]],
    *,
    include_blocked: bool,
    resource_filter: set[str],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        resource = str(row.get("id") or "")
        if resource_filter and resource not in resource_filter:
            continue
        if str(row.get("backend") or "").lower() != "cuda":
            continue
        if row.get("control_plane") != "tensorcore_scheduler":
            continue
        if not include_blocked and row.get("status", "active") == "blocked":
            continue
        out.append(row)
    return out


def reconciliation_config(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("gpu_reconciliation")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"resource {row.get('id')!r} gpu_reconciliation must be an object")
    cfg = dict(raw)
    cfg.setdefault("enabled", True)
    if not isinstance(cfg["enabled"], bool):
        raise ValueError(f"resource {row.get('id')!r} gpu_reconciliation.enabled must be a JSON boolean")
    if not cfg["enabled"]:
        return cfg
    cfg.setdefault("poll_host", row.get("node") or str(row.get("id") or "").split(":", 1)[0])
    cfg.setdefault("nvidia_smi", "nvidia-smi")
    cfg.setdefault("allow_process_regex", [])
    cfg.setdefault("allowed_process_max_memory_mib", 64)
    if not isinstance(cfg["poll_host"], str) or not cfg["poll_host"].strip():
        raise ValueError(f"resource {row.get('id')!r} gpu_reconciliation.poll_host must be non-empty")
    if not isinstance(cfg["nvidia_smi"], str) or not cfg["nvidia_smi"].strip():
        raise ValueError(f"resource {row.get('id')!r} gpu_reconciliation.nvidia_smi must be non-empty")
    if not isinstance(cfg["allow_process_regex"], list) or not all(
        isinstance(item, str) for item in cfg["allow_process_regex"]
    ):
        raise ValueError(f"resource {row.get('id')!r} gpu_reconciliation.allow_process_regex must be a string list")
    if not isinstance(cfg["allowed_process_max_memory_mib"], int) or cfg["allowed_process_max_memory_mib"] < 0:
        raise ValueError(
            f"resource {row.get('id')!r} gpu_reconciliation.allowed_process_max_memory_mib must be >= 0"
        )
    return cfg


def arbiter_status(args: argparse.Namespace) -> dict[str, Any]:
    if args.arbiter_status_json:
        return read_json_object(args.arbiter_status_json)
    argv = shlex.split(args.arbiter_cmd)
    if not argv:
        raise ValueError("--arbiter-cmd must not be empty")
    proc = subprocess.run(
        [*argv, "--json", "status"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.arbiter_timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"arbiter status failed rc={proc.returncode}: {proc.stderr.strip()[-500:]}")
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("arbiter status returned non-object JSON")
    return payload


def snapshot_path(snapshot_dir: Path, resource: str) -> Path:
    return snapshot_dir / f"{safe_resource_name(resource)}.snapshot.json"


def load_or_collect_snapshot(
    row: dict[str, Any],
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    resource = str(row["id"])
    if args.snapshot_json_dir:
        path = snapshot_path(args.snapshot_json_dir, resource)
        if path.exists():
            return read_json_object(path)
        if args.offline:
            raise FileNotFoundError(path)
    snap_args = argparse.Namespace(
        resource=resource,
        nvidia_smi=cfg["nvidia_smi"],
        ssh_host=cfg["poll_host"] if cfg.get("poll_host") else "",
        timeout_sec=args.snapshot_timeout_sec,
    )
    return mesh_worker_gpu_snapshot.build_payload(snap_args)


def reconcile_resource(
    row: dict[str, Any],
    status: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    resource = str(row["id"])
    cfg = reconciliation_config(row)
    if not cfg.get("enabled", True):
        return {
            "schema": "tensorcore.mesh_worker_gpu_reconciliation.v1",
            "ok": True,
            "reason": "disabled",
            "action": "skip",
            "checked_at_unix": time.time(),
            "resource": resource,
            "worker_host": row.get("node"),
            "cuda_app_count": 0,
            "cuda_pids": [],
            "active_lease_ids": [],
            "active_lease_count": 0,
            "errors": [],
            "disabled_reason": cfg.get("reason") or "",
        }
    snapshot = load_or_collect_snapshot(row, cfg, args)
    return mesh_worker_gpu_reconcile.reconcile(
        snapshot,
        status,
        resource=resource,
        allow_process_regex=list(cfg.get("allow_process_regex") or []),
        allowed_process_max_memory_mib=int(cfg.get("allowed_process_max_memory_mib", 64)),
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    rows = active_scheduler_cuda_resources(
        load_inventory(args.inventory_json),
        include_blocked=args.include_blocked,
        resource_filter=set(args.resource or []),
    )
    try:
        status = arbiter_status(args)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "ok": False,
            "checked_at_unix": time.time(),
            "resource_count": len(rows),
            "failed_count": 1,
            "errors": [
                {
                    "resource": "*",
                    "reason": "arbiter_status_unavailable",
                    "action": "block_sweep",
                    "error": str(exc),
                }
            ],
            "reports": [],
        }
    reports = []
    errors = []
    for row in rows:
        resource = str(row["id"])
        try:
            report = reconcile_resource(row, status, args)
        except Exception as exc:
            report = {
                "schema": "tensorcore.mesh_worker_gpu_reconciliation.v1",
                "ok": False,
                "reason": "reconciliation_exception",
                "action": "drain",
                "checked_at_unix": time.time(),
                "resource": resource,
                "worker_host": row.get("node"),
                "errors": [{"code": "reconciliation_exception", "error": str(exc)}],
            }
        reports.append(report)
        if args.reports_dir:
            write_json(
                args.reports_dir / f"{safe_resource_name(resource)}.reconciliation.json",
                report,
            )
        if report.get("ok") is not True:
            errors.append(
                {
                    "resource": resource,
                    "reason": report.get("reason"),
                    "action": report.get("action"),
                }
            )
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "checked_at_unix": time.time(),
        "resource_count": len(rows),
        "failed_count": len(errors),
        "errors": errors,
        "reports": reports,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--arbiter-status-json", type=Path)
    parser.add_argument("--arbiter-cmd", default=DEFAULT_ARBITER_CMD)
    parser.add_argument("--arbiter-timeout-sec", type=float, default=10.0)
    parser.add_argument("--snapshot-json-dir", type=Path)
    parser.add_argument("--snapshot-timeout-sec", type=float, default=10.0)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--include-blocked", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.arbiter_timeout_sec <= 0:
        parser.error("--arbiter-timeout-sec must be > 0")
    if args.snapshot_timeout_sec <= 0:
        parser.error("--snapshot-timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"GPU reconciliation sweep ok: {payload['resource_count']} resource(s)")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
