#!/usr/bin/env python3
"""Audit the Tensorcore mesh scheduler as a whole-system control plane."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.mesh_system_audit.v1"
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
SCHEDULER_CANDIDATES = [
    ROOT / "scripts" / "mesh_resource_scheduler.py",
    pathlib.Path(__file__).with_name("mesh-resource-scheduler"),
    pathlib.Path(__file__).with_name("mesh_resource_scheduler.py"),
]
DEFAULT_ALLOWED_CUDA_HELPERS = [
    r"steamwebhelper$",
    r"/opt/google/chrome/chrome",
]


def load_module(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scheduler_module(name: str) -> ModuleType:
    path = next((item for item in SCHEDULER_CANDIDATES if item.exists()), SCHEDULER_CANDIDATES[0])
    return load_module(name, path)


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def command(value: Any) -> list[str]:
    if isinstance(value, list):
        argv = [str(part) for part in value]
    elif isinstance(value, str):
        argv = shlex.split(value)
    else:
        return []
    if argv and argv[0].startswith("~"):
        argv[0] = str(pathlib.Path(argv[0]).expanduser())
    return argv


def load_json(path: pathlib.Path) -> dict:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def run_json_command(command: str, *, timeout: float) -> dict:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("empty command")
    proc = run_capture(argv, timeout=timeout)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"{argv!r} failed rc={proc.returncode}: {detail}")
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise RuntimeError(f"{argv!r} returned non-object JSON")
    return data


def resource_rows_by_id(status: dict) -> dict[str, dict]:
    rows = {}
    for row in status.get("resources") or []:
        if isinstance(row, dict) and row.get("id"):
            rows[str(row["id"])] = row
    return rows


def leases_by_resource(status: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for lease in status.get("leases") or []:
        if isinstance(lease, dict):
            out.setdefault(str(lease.get("resource") or ""), []).append(lease)
    return out


def lease_metadata(lease: dict) -> dict:
    metadata = lease.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def lease_tenant(lease: dict) -> str:
    metadata = lease_metadata(lease)
    owner = str(lease.get("owner") or "")
    return str(metadata.get("tenant") or owner.split(":", 1)[0] or owner)


def covered_cuda_pids(leases: list[dict]) -> set[int]:
    pids: set[int] = set()
    for lease in leases:
        identity = lease_metadata(lease).get("worker_identity")
        if not isinstance(identity, dict):
            continue
        for key in ("matched_cuda_pids", "cuda_pids"):
            values = identity.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    pids.add(int(value))
                except (TypeError, ValueError):
                    continue
    return pids


def parse_cuda_apps(stdout: str) -> list[dict[str, Any]]:
    apps = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 3:
            if line.strip():
                apps.append({"raw": line.strip(), "parse_error": "expected pid, process_name, memory"})
            continue
        pid_s, process_name, memory_s = parts
        try:
            pid: int | None = int(pid_s)
        except ValueError:
            pid = None
        try:
            memory_mib: int | None = int(memory_s)
        except ValueError:
            memory_mib = None
        apps.append({
            "pid": pid,
            "process_name": process_name,
            "used_memory_mib": memory_mib,
            "raw": line.strip(),
        })
    return apps


def normalize_cuda_probe_payload(payload: dict) -> dict:
    schema = payload.get("schema")
    if schema == "tensorcore.cuda_resource_admission.v1":
        ok = payload.get("ok") is True
        return {
            "ok": ok,
            "rc": 0,
            "apps": [] if ok else list(payload.get("blocked") or []),
            "payload": payload,
        }
    if schema == "tensorcore.windows_cuda_probe.evidence.v1":
        admission = payload.get("admission") if isinstance(payload.get("admission"), dict) else {}
        ok = (
            payload.get("runtime_status") == "ready"
            and admission.get("ok") is True
            and bool((payload.get("cuda_toolkit") or {}).get("nvcc_found"))
            and int(payload.get("device_count") or 0) > 0
        )
        return {
            "ok": ok,
            "rc": 0,
            "apps": [] if ok else list(admission.get("blocked") or []),
            "payload": payload,
        }
    raise ValueError(f"unsupported CUDA probe schema {schema!r}")


def probe_cuda_resource(row: dict, *, timeout: float) -> dict:
    custom_cmd = command(row.get("cuda_probe_cmd"))
    if custom_cmd:
        proc = run_capture(custom_cmd, timeout=timeout)
        if proc.returncode != 0:
            return {
                "ok": False,
                "rc": proc.returncode,
                "apps": [],
                "stderr_tail": proc.stderr.strip()[-1000:],
                "stdout_tail": proc.stdout.strip()[-1000:],
            }
        try:
            payload = json.loads(proc.stdout)
            if not isinstance(payload, dict):
                raise ValueError("custom CUDA probe returned non-object JSON")
            return normalize_cuda_probe_payload(payload)
        except Exception as exc:
            return {
                "ok": False,
                "rc": proc.returncode,
                "apps": [],
                "stderr_tail": str(exc)[-1000:],
                "stdout_tail": proc.stdout.strip()[-1000:],
            }

    node = str(row.get("node") or row["id"].split(":", 1)[0])
    proc = run_capture(
        [
            "ssh",
            node,
            "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits",
        ],
        timeout=timeout,
    )
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "apps": parse_cuda_apps(proc.stdout) if proc.returncode == 0 else [],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def helper_allowed(app: dict, patterns: list[re.Pattern[str]]) -> bool:
    name = str(app.get("process_name") or "")
    return any(pattern.search(name) for pattern in patterns)


def audit(
    *,
    inventory: dict[str, dict],
    jobs: list[dict],
    scheduler_state: dict | None,
    arbiter_status: dict,
    max_scheduler_age_sec: float,
    cuda_apps_by_resource: dict[str, dict] | None = None,
    allowed_cuda_helper_regexes: list[str] | None = None,
) -> dict:
    scheduler = load_scheduler_module("mesh_resource_scheduler_for_system_audit")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    resources_status = resource_rows_by_id(arbiter_status)
    leases = leases_by_resource(arbiter_status)

    missing = sorted(set(inventory) - set(resources_status))
    for resource in missing:
        errors.append({"resource": resource, "error": "inventory_resource_missing_from_arbiter_status"})

    unknown = sorted(set(resources_status) - set(inventory))
    for resource in unknown:
        warnings.append({"resource": resource, "warning": "arbiter_resource_not_in_inventory"})

    paused_jobs = [job for job in jobs if job.get("desired_state") == "paused"]
    paused_start_jobs = [job for job in paused_jobs if job.get("start_cmd")]
    for job in paused_start_jobs:
        if not job.get("preflight_cmd"):
            warnings.append({
                "job": job.get("id"),
                "resource": job.get("resource"),
                "warning": "paused_launchable_job_missing_preflight_cmd",
            })

    for resource_id, row in inventory.items():
        active = leases.get(resource_id, [])
        if row.get("status", "active") == "blocked" and active:
            errors.append({
                "resource": resource_id,
                "error": "blocked_resource_has_active_leases",
                "lease_ids": [lease.get("id") for lease in active],
            })
        if row.get("general_queue_eligible", True) is False and row.get("reserved_for"):
            rules = row.get("reserved_for") or []
            for lease in active:
                owner = str(lease.get("owner") or "")
                tenant = lease_tenant(lease)
                if not scheduler.principal_allowed_for_reserved_resource(owner, tenant, rules):
                    errors.append({
                        "resource": resource_id,
                        "lease": lease.get("id"),
                        "error": "reserved_resource_lease_owner_not_allowed",
                    })
        if (
            row.get("status", "active") == "active"
            and row.get("general_queue_eligible", True)
            and row.get("control_plane") == "tensorcore_scheduler"
        ):
            if not any(job["resource"] == resource_id for job in jobs):
                warnings.append({"resource": resource_id, "warning": "active_resource_has_no_scheduler_job"})

    if scheduler_state is not None:
        age = time.time() - float(scheduler_state.get("checked_at_unix") or 0.0)
        if scheduler_state.get("ok") is not True:
            errors.append({"error": "scheduler_state_not_ok", "state_errors": scheduler_state.get("errors")})
        if age > max_scheduler_age_sec:
            errors.append({
                "error": "scheduler_state_stale",
                "age_sec": round(age, 3),
                "max_age_sec": max_scheduler_age_sec,
            })

    helper_patterns = [
        re.compile(pattern)
        for pattern in (allowed_cuda_helper_regexes or DEFAULT_ALLOWED_CUDA_HELPERS)
    ]
    cuda_audits: dict[str, dict] = {}
    for resource_id, row in inventory.items():
        if str(row.get("backend") or "").lower() != "cuda":
            continue
        active = leases.get(resource_id, [])
        for lease in active:
            metadata = lease_metadata(lease)
            if metadata.get("surface") == "tensorcore_mesh_scheduler":
                if metadata.get("worker_identity_pending") is True or "worker_identity" not in metadata:
                    errors.append({
                        "resource": resource_id,
                        "lease": lease.get("id"),
                        "error": "scheduler_lease_missing_worker_identity",
                    })
        if not cuda_apps_by_resource or resource_id not in cuda_apps_by_resource:
            continue
        probe = cuda_apps_by_resource[resource_id]
        cuda_audits[resource_id] = probe
        if not probe.get("ok"):
            if row.get("status") == "blocked":
                warnings.append({"resource": resource_id, "warning": "blocked_cuda_probe_failed", "probe": probe})
            else:
                errors.append({"resource": resource_id, "error": "cuda_probe_failed", "probe": probe})
            continue
        covered = covered_cuda_pids(active)
        unmanaged = []
        for app in probe.get("apps") or []:
            pid = app.get("pid")
            if pid in covered:
                continue
            if helper_allowed(app, helper_patterns):
                continue
            unmanaged.append(app)
        if unmanaged:
            row_out = {
                "resource": resource_id,
                "apps": unmanaged,
            }
            if row.get("status") == "blocked":
                row_out["warning"] = "blocked_cuda_has_unmanaged_processes"
                warnings.append(row_out)
            else:
                row_out["error"] = "unmanaged_cuda_processes"
                errors.append(row_out)

    return {
        "schema": SCHEMA,
        "ok": not errors,
        "checked_at_unix": time.time(),
        "summary": {
            "inventory_resources": len(inventory),
            "expanded_jobs": len(jobs),
            "paused_jobs": len(paused_jobs),
            "paused_start_jobs": len(paused_start_jobs),
            "paused_start_jobs_with_preflight": sum(1 for job in paused_start_jobs if job.get("preflight_cmd")),
            "active_leases": len(arbiter_status.get("leases") or []),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
        "cuda_audits": cuda_audits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--jobs-json", type=pathlib.Path, required=True)
    parser.add_argument("--scheduler-state-json", type=pathlib.Path)
    parser.add_argument("--arbiter-status-json", type=pathlib.Path)
    parser.add_argument("--arbiter-cmd")
    parser.add_argument("--max-scheduler-age-sec", type=float, default=120.0)
    parser.add_argument("--probe-cuda", action="store_true")
    parser.add_argument("--probe-timeout-sec", type=float, default=10.0)
    parser.add_argument("--allow-cuda-helper-regex", action="append", default=[])
    parser.add_argument("--pretty-json", action="store_true")
    return parser.parse_args(argv)


def emit_text(payload: dict) -> None:
    status = "OK" if payload.get("ok") else "FAILED"
    summary = payload.get("summary") or {}
    print(
        f"mesh system audit {status}: "
        f"resources={summary.get('inventory_resources')} "
        f"jobs={summary.get('expanded_jobs')} "
        f"leases={summary.get('active_leases')} "
        f"errors={summary.get('errors')} warnings={summary.get('warnings')}"
    )
    for row in payload.get("errors") or []:
        print(f"ERROR {row}", file=sys.stderr)
    for row in payload.get("warnings") or []:
        print(f"WARN {row}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    scheduler = load_scheduler_module("mesh_resource_scheduler_for_system_audit_main")
    inventory = scheduler.load_inventory(str(args.inventory_json))
    jobs = scheduler.load_jobs(str(args.jobs_json), inventory=inventory)
    scheduler.validate_jobs_against_inventory(jobs, inventory)
    scheduler_state = load_json(args.scheduler_state_json) if args.scheduler_state_json else None
    if args.arbiter_status_json:
        arbiter_status = load_json(args.arbiter_status_json)
    elif args.arbiter_cmd:
        arbiter_status = run_json_command(args.arbiter_cmd, timeout=args.probe_timeout_sec)
    else:
        raise SystemExit("--arbiter-status-json or --arbiter-cmd is required")

    cuda_apps_by_resource = None
    if args.probe_cuda:
        cuda_apps_by_resource = {
            resource_id: probe_cuda_resource(row, timeout=args.probe_timeout_sec)
            for resource_id, row in inventory.items()
            if str(row.get("backend") or "").lower() == "cuda"
        }
    payload = audit(
        inventory=inventory,
        jobs=jobs,
        scheduler_state=scheduler_state,
        arbiter_status=arbiter_status,
        max_scheduler_age_sec=args.max_scheduler_age_sec,
        cuda_apps_by_resource=cuda_apps_by_resource,
        allowed_cuda_helper_regexes=args.allow_cuda_helper_regex or None,
    )
    if args.pretty_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        emit_text(payload)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
