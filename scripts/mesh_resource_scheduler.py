#!/usr/bin/env python3
"""Coordinate shared mesh resources through the Tsotchke arbiter.

This is a control-plane scheduler, not a workload-specific launcher. It reads
a small jobs file, probes known work, reconciles stale leases, and only starts a
job after it has claimed the requested resource. The policy is intentionally
non-destructive: live work is never killed for priority alone, and unknown
leases are treated as busy resources.
"""

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


DEFAULT_ARBITER_CMD = "~/.tsotchke/bin/tsotchke-arbiter"
SCHEMA = "tensorcore.mesh_resource_jobs.v1"
INVENTORY_SCHEMA = "tensorcore.mesh_resources.v1"
INVENTORY_STATUSES = {"active", "reserved", "blocked"}
RESOURCE_CLASSES = {"generic", "cuda_exclusive"}
DESIRED_STATES = {"running", "paused"}


def command(value: Any) -> list[str]:
    if isinstance(value, list):
        argv = [str(part) for part in value]
    elif isinstance(value, str):
        argv = shlex.split(value)
    else:
        return []
    if argv and argv[0].startswith("~"):
        argv[0] = str(Path(argv[0]).expanduser())
    return argv


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_json(argv: list[str], *, timeout: float) -> dict:
    proc = run_capture(argv, timeout=timeout)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"{argv!r} failed rc={proc.returncode}: {detail}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{argv!r} did not return JSON: {proc.stdout[:240]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{argv!r} returned non-object JSON")
    return data


def write_json(path: str, payload: dict) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


def load_jobs(path: str, inventory: dict[str, dict] | None = None) -> list[dict]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        jobs = raw.get("jobs")
    else:
        jobs = raw
    if not isinstance(jobs, list):
        raise SystemExit("--jobs-json must contain a list or an object with jobs")
    return [normalize_job(job, inventory=inventory) for job in jobs]


def load_inventory(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--inventory-json must contain a JSON object")
    if raw.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"--inventory-json schema must be {INVENTORY_SCHEMA}")
    resources = raw.get("resources")
    if not isinstance(resources, list):
        raise ValueError("--inventory-json resources must be a list")
    out: dict[str, dict] = {}
    for row in resources:
        if not isinstance(row, dict):
            raise ValueError("--inventory-json resources contains a non-object row")
        resource_id = row.get("id")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError("--inventory-json resource id must be a non-empty string")
        if resource_id in out:
            raise ValueError(f"--inventory-json has duplicate resource {resource_id!r}")
        capacity = row.get("capacity", 1)
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"resource {resource_id!r} capacity must be a positive integer")
        status = row.get("status", "active")
        if status not in INVENTORY_STATUSES:
            raise ValueError(
                f"resource {resource_id!r} status must be one of {sorted(INVENTORY_STATUSES)!r}"
            )
        backend = row.get("backend")
        if backend is not None and (not isinstance(backend, str) or not backend.strip()):
            raise ValueError(f"resource {resource_id!r} backend must be a non-empty string when set")
        if status == "blocked" and not str(row.get("blocked_reason") or "").strip():
            raise ValueError(f"resource {resource_id!r} status=blocked requires blocked_reason")
        general = row.get("general_queue_eligible", True)
        if not isinstance(general, bool):
            raise ValueError(f"resource {resource_id!r} general_queue_eligible must be a JSON boolean")
        reserved_for = row.get("reserved_for", [])
        if reserved_for is None:
            reserved_for = []
        if not isinstance(reserved_for, list) or not all(isinstance(item, str) for item in reserved_for):
            raise ValueError(f"resource {resource_id!r} reserved_for must be a string list")
        out[resource_id] = dict(row)
    return out


def owner_matches_rule(owner: str, rule: str) -> bool:
    if rule.endswith("*"):
        return owner.startswith(rule[:-1])
    return owner == rule


def owner_allowed_for_reserved_resource(owner: str, rules: list[str]) -> bool:
    return any(owner_matches_rule(owner, rule) for rule in rules)


def validate_jobs_against_inventory(jobs: list[dict], inventory: dict[str, dict]) -> None:
    if not inventory:
        return
    errors = []
    for job in jobs:
        resource = inventory.get(job["resource"])
        if resource is None:
            errors.append(f"job {job['id']!r} targets unknown resource {job['resource']!r}")
            continue
        status = str(resource.get("status", "active"))
        if status == "blocked" and job["desired_state"] == "running":
            reason = resource.get("blocked_reason") or "no blocked_reason provided"
            errors.append(f"job {job['id']!r} targets blocked resource {job['resource']!r}: {reason}")
        general = resource.get("general_queue_eligible", True)
        reserved_for = resource.get("reserved_for") or []
        if general is False and reserved_for:
            if not owner_allowed_for_reserved_resource(job["owner"], reserved_for):
                errors.append(
                    f"job {job['id']!r} owner {job['owner']!r} is not allowed "
                    f"to use reserved resource {job['resource']!r}"
                )
        elif general is False and status != "blocked" and job["desired_state"] == "running":
            errors.append(
                f"job {job['id']!r} targets non-general resource {job['resource']!r} "
                "with no reserved_for allow-list"
            )
        backend = str(resource.get("backend") or "").lower()
        if (
            backend == "cuda"
            and job["desired_state"] == "running"
            and job["resource_class"] != "cuda_exclusive"
        ):
            errors.append(
                f"job {job['id']!r} targets CUDA inventory resource {job['resource']!r} "
                "but resource_class is not cuda_exclusive"
            )
    if errors:
        raise ValueError("; ".join(errors))


def infer_resource_class(resource: str, inventory: dict[str, dict] | None = None) -> str:
    row = inventory.get(resource) if inventory else None
    if row and str(row.get("backend") or "").lower() == "cuda":
        return "cuda_exclusive"
    if ":cuda" in resource or resource.startswith("cuda"):
        return "cuda_exclusive"
    return "generic"


def require_str(job: dict, key: str) -> str:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"job {job.get('id', '<unknown>')!r} field {key!r} must be a non-empty string")
    return value


def require_bool(job: dict, key: str, default: bool) -> bool:
    value = job.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"job {job.get('id', '<unknown>')!r} field {key!r} must be a JSON boolean")
    return value


def validate_command_field(job: dict, key: str, *, required: bool) -> None:
    value = job.get(key)
    if value is None:
        if required:
            raise ValueError(f"job {job['id']!r} requires {key} for resource_class={job['resource_class']}")
        return
    if isinstance(value, str):
        if not shlex.split(value):
            raise ValueError(f"job {job['id']!r} field {key!r} must not be empty")
        return
    if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
        return
    raise ValueError(f"job {job['id']!r} field {key!r} must be a non-empty string or string list")


def normalize_job(job: Any, inventory: dict[str, dict] | None = None) -> dict:
    if not isinstance(job, dict):
        raise SystemExit("--jobs-json contains a non-object job")
    out = dict(job)
    out["id"] = require_str(out, "id")
    out["sync_id"] = str(out.get("sync_id") or out["id"])
    out["resource"] = require_str(out, "resource")
    out["owner"] = require_str(out, "owner")
    out["priority"] = int(out.get("priority", 0))
    out["enabled"] = require_bool(out, "enabled", True)
    out["ttl_sec"] = float(out.get("ttl_sec", 900.0))
    if out["ttl_sec"] <= 0:
        raise ValueError(f"job {out['id']!r} field 'ttl_sec' must be positive")
    out["desired_state"] = require_str(out, "desired_state") if "desired_state" in out else "running"
    if out["desired_state"] not in DESIRED_STATES:
        raise ValueError(
            f"job {out['id']!r} desired_state must be one of {sorted(DESIRED_STATES)!r}"
        )
    out["resource_class"] = str(
        out.get("resource_class") or infer_resource_class(out["resource"], inventory)
    )
    if out["resource_class"] not in RESOURCE_CLASSES:
        raise ValueError(
            f"job {out['id']!r} resource_class must be one of {sorted(RESOURCE_CLASSES)!r}"
        )
    requires_host_gate = out["resource_class"] == "cuda_exclusive"
    validate_command_field(out, "probe_cmd", required=False)
    validate_command_field(out, "start_cmd", required=out["desired_state"] == "running")
    validate_command_field(out, "completion_cmd", required=False)
    validate_command_field(out, "admission_cmd", required=requires_host_gate)
    validate_command_field(out, "post_start_probe_cmd", required=requires_host_gate)
    validate_command_field(out, "worker_identity_cmd", required=requires_host_gate)

    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.setdefault("surface", "tensorcore_mesh_scheduler")
    metadata["sync_job_id"] = out["sync_id"]
    metadata["job_id"] = out["id"]
    metadata["resource_class"] = out["resource_class"]
    metadata["scheduler_host"] = os.uname().nodename
    metadata["worker_identity_pending"] = requires_host_gate
    out["metadata"] = metadata
    return out


def probe_job(job: dict, *, timeout: float) -> dict:
    probe_cmd = command(job.get("probe_cmd"))
    if not probe_cmd:
        return {"live": None, "reason": "missing_probe_cmd", "rc": None}
    try:
        proc = run_capture(probe_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"live": None, "reason": "probe_timeout", "rc": None}
    return {
        "live": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "probe_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-240:],
        "stderr_tail": proc.stderr.strip()[-240:],
    }


def complete_job(job: dict, *, timeout: float) -> dict:
    completion_cmd = command(job.get("completion_cmd"))
    if not completion_cmd:
        return {"complete": False, "reason": "missing_completion_cmd", "rc": None}
    try:
        proc = run_capture(completion_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"complete": None, "reason": "completion_timeout", "rc": None}
    return {
        "complete": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "completion_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def admit_job(job: dict, *, timeout: float) -> dict:
    admission_cmd = command(job.get("admission_cmd"))
    if not admission_cmd:
        return {"admitted": True, "reason": "missing_admission_cmd", "rc": None}
    try:
        proc = run_capture(admission_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"admitted": None, "reason": "admission_timeout", "rc": None}
    return {
        "admitted": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "admission_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def lease_metadata(row: dict) -> dict:
    metadata = row.get("metadata")
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


def lease_sync_id(row: dict) -> str | None:
    metadata = lease_metadata(row)
    value = (
        row.get("sync_job_id")
        or row.get("sync_id")
        or metadata.get("sync_job_id")
        or metadata.get("sync_id")
    )
    return str(value) if value else None


def dedupe_leases(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str | int]] = set()
    for row in rows:
        lease_id = row.get("id")
        key = ("id", str(lease_id)) if lease_id else ("object", id(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def leases_for_resource(status: dict, resource: str) -> list[dict]:
    return [
        row
        for row in status.get("leases") or []
        if isinstance(row, dict) and row.get("resource") == resource
    ]


def job_for_lease(lease: dict, jobs: list[dict]) -> dict | None:
    sync_id = lease_sync_id(lease)
    for job in jobs:
        if lease.get("resource") != job["resource"]:
            continue
        if sync_id and sync_id == job["sync_id"]:
            return job
        if lease.get("owner") == job["owner"]:
            return job
    return None


def matching_leases(status: dict, job: dict) -> list[dict]:
    logical = []
    owner = []
    for row in leases_for_resource(status, job["resource"]):
        if lease_sync_id(row) == job["sync_id"]:
            logical.append(row)
        if row.get("owner") == job["owner"]:
            owner.append(row)
    if logical:
        return dedupe_leases(logical + owner)
    return dedupe_leases(owner)


def claim_metadata(job: dict, worker_identity: dict | None = None) -> dict:
    metadata = dict(job["metadata"])
    if worker_identity and worker_identity.get("ok"):
        metadata["worker_identity"] = worker_identity.get("identity")
        metadata["worker_identity_pending"] = False
    return metadata


def claim_job(
    job: dict,
    *,
    arbiter_cmd: list[str],
    timeout: float,
    worker_identity: dict | None = None,
) -> dict:
    metadata = claim_metadata(job, worker_identity=worker_identity)
    return run_json(
        arbiter_cmd
        + [
            "claim",
            job["resource"],
            "--owner",
            job["owner"],
            "--ttl-sec",
            str(job["ttl_sec"]),
            "--metadata-json",
            json.dumps(metadata, sort_keys=True),
            "--json",
        ],
        timeout=timeout,
    )


def heartbeat_lease(lease: dict, job: dict, *, arbiter_cmd: list[str], timeout: float) -> dict:
    return run_json(
        arbiter_cmd
        + ["heartbeat", str(lease["id"]), "--ttl-sec", str(job["ttl_sec"]), "--json"],
        timeout=timeout,
    )


def release_lease(lease: dict, *, arbiter_cmd: list[str], timeout: float) -> dict:
    return run_json(
        arbiter_cmd + ["release", str(lease["id"]), "--json"],
        timeout=timeout,
    )


def release_many(
    leases: list[dict],
    *,
    arbiter_cmd: list[str],
    timeout: float,
) -> tuple[list[dict], bool]:
    payloads = []
    ok = True
    for lease in leases:
        payload = release_lease(lease, arbiter_cmd=arbiter_cmd, timeout=timeout)
        payloads.append(payload)
        ok = ok and bool(payload.get("ok"))
    return payloads, ok


def start_job(job: dict, *, timeout: float) -> dict:
    start_cmd = command(job.get("start_cmd"))
    if not start_cmd:
        raise RuntimeError(f"job {job['id']} has no start_cmd")
    proc = run_capture(start_cmd, timeout=timeout)
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def post_start_probe_job(
    job: dict,
    *,
    timeout: float,
    interval: float,
) -> dict:
    probe_cmd = command(job.get("post_start_probe_cmd"))
    if not probe_cmd:
        if job.get("resource_class") != "cuda_exclusive":
            return {"verified": True, "reason": "not_required", "rc": None}
        return {"verified": False, "reason": "missing_post_start_probe_cmd", "rc": None}
    deadline = time.monotonic() + max(0.0, timeout)
    attempts = 0
    last: dict[str, Any] = {"verified": False, "reason": "not_run", "rc": None}
    while True:
        attempts += 1
        remaining = deadline - time.monotonic()
        if attempts > 1 and remaining <= 0:
            last["reason"] = "post_start_timeout"
            last["attempts"] = attempts - 1
            return last
        try:
            proc = run_capture(probe_cmd, timeout=max(0.1, min(remaining, 10.0)))
        except subprocess.TimeoutExpired:
            last = {
                "verified": None,
                "reason": "post_start_probe_timeout",
                "rc": None,
                "attempts": attempts,
            }
        else:
            last = {
                "verified": proc.returncode == 0,
                "reason": "ok" if proc.returncode == 0 else "post_start_probe_failed",
                "rc": proc.returncode,
                "stdout_tail": proc.stdout.strip()[-1000:],
                "stderr_tail": proc.stderr.strip()[-1000:],
                "attempts": attempts,
            }
            if proc.returncode == 0:
                return last
        if interval <= 0:
            return last
        if time.monotonic() >= deadline:
            last["reason"] = "post_start_timeout"
            return last
        time.sleep(max(0.0, min(interval, deadline - time.monotonic())))


def collect_worker_identity(job: dict, *, timeout: float) -> dict:
    identity_cmd = command(job.get("worker_identity_cmd"))
    if not identity_cmd:
        if job.get("resource_class") != "cuda_exclusive":
            return {"ok": True, "reason": "not_required", "rc": None, "identity": {}}
        return {"ok": False, "reason": "missing_worker_identity_cmd", "rc": None}
    try:
        proc = run_capture(identity_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "worker_identity_timeout", "rc": None}
    result: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "worker_identity_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = {"text": proc.stdout.strip()}
        if isinstance(parsed, dict):
            result["identity"] = parsed
        else:
            result["identity"] = {"value": parsed}
    return result


def enabled_running_jobs(jobs: list[dict], resource: str) -> list[dict]:
    return [
        job
        for job in jobs
        if job["resource"] == resource
        and job["enabled"]
        and job["desired_state"] == "running"
    ]


def enabled_jobs(jobs: list[dict], resource: str) -> list[dict]:
    return [
        job
        for job in jobs
        if job["resource"] == resource and job["enabled"]
    ]


def choose_candidate(
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    admissions: dict[str, dict],
    resource: str,
) -> dict | None:
    candidates = [
        job
        for job in enabled_running_jobs(jobs, resource)
        if probes[job["id"]]["live"] is False
        and completions[job["id"]]["complete"] is False
        and admissions[job["id"]]["admitted"] is True
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item["priority"], item["id"]))[0]


def reconcile_stale_leases(
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    resource: str,
    status: dict,
    *,
    arbiter_cmd: list[str],
    timeout: float,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    results = []
    errors = []
    for job in jobs:
        if job["resource"] != resource:
            continue
        job_probe = probes[job["id"]]
        if job_probe["live"] is not False:
            continue
        completion = completions[job["id"]]
        leases = matching_leases(status, job)
        if not leases:
            continue
        if completion["complete"] is True:
            action = (
                "would_release_completed_lease"
                if dry_run
                else "released_completed_lease"
            )
        else:
            action = "would_release_stale_lease" if dry_run else "released_stale_lease"
        result = {
            "resource": resource,
            "job": job["id"],
            "action": action,
            "lease_ids": [lease.get("id") for lease in leases],
            "probe": job_probe,
            "completion": completion,
            "ok": True,
        }
        if not dry_run:
            try:
                payloads, ok = release_many(
                    leases,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                result["arbiter"] = payloads
                result["ok"] = ok
            except Exception as exc:
                result["ok"] = False
                errors.append({"resource": resource, "job": job["id"], "error": str(exc)})
        results.append(result)
    return results, errors


def resource_leases_after_reconcile(
    leases: list[dict],
    stale_results: list[dict],
    dry_run: bool,
) -> list[dict]:
    if dry_run:
        return leases
    released = {
        str(lease_id)
        for row in stale_results
        for lease_id in row.get("lease_ids", [])
        if lease_id is not None and row.get("ok")
    }
    return [lease for lease in leases if str(lease.get("id")) not in released]


def handle_live_holders(
    live_jobs: list[dict],
    known_leases: list[dict],
    unknown_leases: list[dict],
    probes: dict[str, dict],
    *,
    arbiter_cmd: list[str],
    timeout: float,
    worker_identity_timeout: float,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    results = []
    errors = []
    if len(live_jobs) > 1:
        return [], [{
            "resource": live_jobs[0]["resource"] if live_jobs else "-",
            "error": "multiple live holders detected; refusing destructive arbitration",
            "jobs": [job["id"] for job in live_jobs],
        }]

    job = live_jobs[0]
    matches = [
        lease
        for lease in known_leases
        if job_for_lease(lease, [job]) is not None
    ]
    current = matches[0] if matches else None
    stale = matches[1:] if matches else []
    other_known = [
        lease
        for lease in known_leases
        if all(lease.get("id") != match.get("id") for match in matches)
    ]

    if current:
        action = "would_heartbeat_live_holder" if dry_run else "heartbeated_live_holder"
        identity = collect_worker_identity(job, timeout=worker_identity_timeout)
        result = {
            "resource": job["resource"],
            "job": job["id"],
            "action": action,
            "lease_id": current.get("id"),
            "stale_lease_ids": [lease.get("id") for lease in stale],
            "probe": probes[job["id"]],
            "worker_identity": identity,
            "ok": not other_known and bool(identity.get("ok")),
        }
        if other_known:
            result["action"] = "live_holder_conflicting_known_lease"
            result["other_lease_ids"] = [lease.get("id") for lease in other_known]
        if not dry_run:
            try:
                releases, releases_ok = release_many(
                    stale,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                payload = heartbeat_lease(
                    current,
                    job,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                result["arbiter"] = {"release": releases, "heartbeat": payload}
                result["ok"] = (
                    not other_known
                    and bool(identity.get("ok"))
                    and releases_ok
                    and bool(payload.get("ok"))
                )
            except Exception as exc:
                result["ok"] = False
                errors.append({"resource": job["resource"], "job": job["id"], "error": str(exc)})
        results.append(result)
        return results, errors

    if other_known:
        results.append({
            "resource": job["resource"],
            "job": job["id"],
            "action": "live_holder_blocked_by_known_lease_unknown_liveness",
            "lease_ids": [lease.get("id") for lease in other_known],
            "ok": True,
        })
        return results, errors

    if unknown_leases:
        results.append({
            "resource": job["resource"],
            "job": job["id"],
            "action": "live_holder_blocked_by_unknown_lease",
            "lease_ids": [lease.get("id") for lease in unknown_leases],
            "ok": True,
        })
        return results, errors

    action = "would_adopt_live_holder" if dry_run else "adopted_live_holder"
    identity = collect_worker_identity(job, timeout=worker_identity_timeout)
    result = {
        "resource": job["resource"],
        "job": job["id"],
        "action": action,
        "probe": probes[job["id"]],
        "worker_identity": identity,
        "ok": bool(identity.get("ok")),
    }
    if not identity.get("ok"):
        result["action"] = "live_holder_identity_failed"
        results.append(result)
        return results, errors
    if not dry_run:
        try:
            payload = claim_job(
                job,
                arbiter_cmd=arbiter_cmd,
                timeout=timeout,
                worker_identity=identity,
            )
            result["arbiter"] = payload
            result["ok"] = bool(payload.get("ok"))
            result["lease_id"] = payload.get("lease_id") or payload.get("id")
        except Exception as exc:
            result["ok"] = False
            errors.append({"resource": job["resource"], "job": job["id"], "error": str(exc)})
    results.append(result)
    return results, errors


def schedule_resource(
    resource: str,
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    admissions: dict[str, dict],
    status: dict,
    args: argparse.Namespace,
    *,
    arbiter_cmd: list[str],
) -> tuple[list[dict], list[dict]]:
    results, errors = reconcile_stale_leases(
        jobs,
        probes,
        completions,
        resource,
        status,
        arbiter_cmd=arbiter_cmd,
        timeout=args.timeout_sec,
        dry_run=args.dry_run,
    )
    leases = resource_leases_after_reconcile(
        leases_for_resource(status, resource),
        results,
        args.dry_run,
    )
    known_leases = [lease for lease in leases if job_for_lease(lease, jobs) is not None]
    unknown_leases = [lease for lease in leases if job_for_lease(lease, jobs) is None]
    live_jobs = [
        job
        for job in enabled_jobs(jobs, resource)
        if probes[job["id"]]["live"] is True
    ]

    if live_jobs:
        live_results, live_errors = handle_live_holders(
            live_jobs,
            known_leases,
            unknown_leases,
            probes,
            arbiter_cmd=arbiter_cmd,
            timeout=args.timeout_sec,
            worker_identity_timeout=args.worker_identity_timeout_sec,
            dry_run=args.dry_run,
        )
        results.extend(live_results)
        errors.extend(live_errors)
        return results, errors

    if unknown_leases:
        results.append({
            "resource": resource,
            "action": "resource_busy_unknown_lease",
            "lease_ids": [lease.get("id") for lease in unknown_leases],
            "ok": True,
        })
        return results, errors

    if known_leases:
        results.append({
            "resource": resource,
            "action": "resource_busy_known_lease_unknown_liveness",
            "lease_ids": [lease.get("id") for lease in known_leases],
            "ok": True,
        })
        return results, errors

    candidate = choose_candidate(jobs, probes, completions, admissions, resource)
    if candidate is None:
        unknown_completion_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if probes[job["id"]]["live"] is False
            and completions[job["id"]]["complete"] is None
        ]
        if unknown_completion_jobs:
            results.append({
                "resource": resource,
                "action": "idle_completion_unknown",
                "jobs": unknown_completion_jobs,
                "ok": True,
            })
            return results, errors
        admission_blocked_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if probes[job["id"]]["live"] is False
            and completions[job["id"]]["complete"] is False
            and admissions[job["id"]]["admitted"] is not True
        ]
        if admission_blocked_jobs:
            results.append({
                "resource": resource,
                "action": "idle_admission_blocked",
                "jobs": admission_blocked_jobs,
                "admissions": {
                    job_id: admissions[job_id]
                    for job_id in admission_blocked_jobs
                },
                "ok": True,
            })
            return results, errors
        completed_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if completions[job["id"]]["complete"] is True
        ]
        if completed_jobs:
            results.append({
                "resource": resource,
                "action": "idle_completed_jobs",
                "jobs": completed_jobs,
                "ok": True,
            })
            return results, errors
        results.append({"resource": resource, "action": "idle_no_candidate", "ok": True})
        return results, errors

    if args.dry_run:
        results.append({
            "resource": resource,
            "job": candidate["id"],
            "action": "would_claim_and_launch",
            "probe": probes[candidate["id"]],
            "completion": completions[candidate["id"]],
            "admission": admissions[candidate["id"]],
            "ok": True,
        })
        return results, errors

    claim_payload = claim_job(candidate, arbiter_cmd=arbiter_cmd, timeout=args.timeout_sec)
    result = {
        "resource": resource,
        "job": candidate["id"],
        "action": "claimed_and_launched",
        "arbiter": claim_payload,
        "completion": completions[candidate["id"]],
        "admission": admissions[candidate["id"]],
        "ok": bool(claim_payload.get("ok")),
    }
    if not claim_payload.get("ok"):
        results.append(result)
        return results, errors

    try:
        start_payload = start_job(candidate, timeout=args.start_timeout_sec)
        result["start"] = start_payload
        result["ok"] = bool(start_payload.get("ok"))
    except Exception as exc:
        result["ok"] = False
        result["start_error"] = str(exc)

    if result["ok"]:
        post_start = post_start_probe_job(
            candidate,
            timeout=args.post_start_timeout_sec,
            interval=args.post_start_interval_sec,
        )
        result["post_start"] = post_start
        result["ok"] = post_start.get("verified") is True
        if result["ok"]:
            identity = collect_worker_identity(
                candidate,
                timeout=args.worker_identity_timeout_sec,
            )
            result["worker_identity"] = identity
            result["ok"] = bool(identity.get("ok"))

    if not result["ok"]:
        lease_id = claim_payload.get("lease_id") or claim_payload.get("id")
        if lease_id and result.get("post_start", {}).get("verified") is not True:
            try:
                result["release_after_failed_start"] = release_lease(
                    {"id": lease_id},
                    arbiter_cmd=arbiter_cmd,
                    timeout=args.timeout_sec,
                )
            except Exception as exc:
                errors.append({
                    "resource": resource,
                    "job": candidate["id"],
                    "error": f"failed to release lease after failed start: {exc}",
                })
    results.append(result)
    return results, errors


def schedule_once(args: argparse.Namespace) -> dict:
    arbiter_cmd = command(args.arbiter_cmd)
    if not arbiter_cmd:
        raise SystemExit("--arbiter-cmd resolved to an empty command")
    inventory = load_inventory(args.inventory_json)
    jobs = load_jobs(args.jobs_json, inventory=inventory)
    validate_jobs_against_inventory(jobs, inventory)
    probes = {job["id"]: probe_job(job, timeout=args.probe_timeout_sec) for job in jobs}
    completions = {
        job["id"]: complete_job(job, timeout=args.probe_timeout_sec)
        for job in jobs
    }
    admissions = {
        job["id"]: admit_job(job, timeout=args.admission_timeout_sec)
        for job in jobs
    }
    status = run_json(arbiter_cmd + ["status", "--json"], timeout=args.timeout_sec)
    resources = sorted({job["resource"] for job in jobs})
    results = []
    errors = []
    for resource in resources:
        try:
            resource_results, resource_errors = schedule_resource(
                resource,
                jobs,
                probes,
                completions,
                admissions,
                status,
                args,
                arbiter_cmd=arbiter_cmd,
            )
            results.extend(resource_results)
            errors.extend(resource_errors)
        except Exception as exc:
            errors.append({"resource": resource, "error": str(exc)})
    return {
        "schema": "tensorcore.mesh_resource_scheduler.result.v1",
        "ok": not errors and all(row.get("ok", True) for row in results),
        "checked_at_unix": time.time(),
        "dry_run": args.dry_run,
        "results": results,
        "errors": errors,
    }


def emit_text(payload: dict) -> None:
    if "iteration" in payload:
        print(f"== iteration {payload['iteration']} ==")
    for row in payload["results"]:
        print(
            f"{row.get('resource', '-')}: {row.get('action')} "
            f"job={row.get('job', '-')} ok={row.get('ok', True)}"
        )
    for row in payload["errors"]:
        print(f"ERROR {row}", file=sys.stderr)


def emit_json(payload: dict, *, pretty: bool = False) -> None:
    if pretty:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    else:
        json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_loop(args: argparse.Namespace) -> int:
    iteration = 0
    all_ok = True
    try:
        while True:
            iteration += 1
            try:
                payload = schedule_once(args)
            except Exception as exc:
                payload = {
                    "schema": "tensorcore.mesh_resource_scheduler.result.v1",
                    "ok": False,
                    "checked_at_unix": time.time(),
                    "dry_run": args.dry_run,
                    "results": [],
                    "errors": [{"error": str(exc)}],
                }
            payload["iteration"] = iteration
            all_ok = all_ok and bool(payload.get("ok"))
            if args.state_json:
                write_json(args.state_json, payload)
            if args.json or args.pretty_json:
                emit_json(payload, pretty=args.pretty_json)
            else:
                emit_text(payload)
            if args.max_iterations > 0 and iteration >= args.max_iterations:
                break
            time.sleep(max(0.0, args.interval_sec))
    except KeyboardInterrupt:
        return 130
    return 0 if all_ok else 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arbiter-cmd", default=DEFAULT_ARBITER_CMD)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--inventory-json")
    parser.add_argument("--state-json")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--probe-timeout-sec", type=float, default=10.0)
    parser.add_argument("--admission-timeout-sec", type=float, default=10.0)
    parser.add_argument("--start-timeout-sec", type=float, default=60.0)
    parser.add_argument("--post-start-timeout-sec", type=float, default=30.0)
    parser.add_argument("--post-start-interval-sec", type=float, default=2.0)
    parser.add_argument("--worker-identity-timeout-sec", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_iterations < 0:
        parser.error("--max-iterations must be >= 0")
    if args.post_start_timeout_sec < 0:
        parser.error("--post-start-timeout-sec must be >= 0")
    if args.post_start_interval_sec < 0:
        parser.error("--post-start-interval-sec must be >= 0")
    if args.worker_identity_timeout_sec <= 0:
        parser.error("--worker-identity-timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.loop:
        return run_loop(args)
    try:
        payload = schedule_once(args)
    except Exception as exc:
        payload = {
            "schema": "tensorcore.mesh_resource_scheduler.result.v1",
            "ok": False,
            "checked_at_unix": time.time(),
            "dry_run": args.dry_run,
            "results": [],
            "errors": [{"error": str(exc)}],
        }
    if args.state_json:
        write_json(args.state_json, payload)
    if args.json or args.pretty_json:
        emit_json(payload, pretty=args.pretty_json)
    else:
        emit_text(payload)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
