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


DEFAULT_ARBITER_CMD = os.environ.get("TC_MESH_ARBITER_CMD", "tsotchke-arbiter")
SCHEMA = "tensorcore.mesh_resource_jobs.v1"
SUBMIT_SCHEMA = "tensorcore.job.v1"
INVENTORY_SCHEMA = "tensorcore.mesh_resources.v1"
INVENTORY_STATUSES = {"active", "reserved", "blocked"}
RESOURCE_CLASSES = {"generic", "cuda_exclusive"}
DESIRED_STATES = {"running", "paused"}
ROOT = Path(__file__).resolve().parents[1]
CONTROL_COMMANDS = {"submit", "status", "cancel", "drain", "undrain", "audit"}


def render_command_part(value: str, job: dict | None = None) -> str:
    if job is None:
        return value
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    replacements = {
        "authority_owner": job.get("authority_owner") or job.get("owner"),
        "authority_resource": job.get("authority_resource") or job.get("resource"),
        "backend": job.get("resource_backend"),
        "id": job.get("id"),
        "job_id": job.get("id"),
        "lease_id": job.get("lease_id"),
        "logical_id": job.get("logical_id"),
        "node": job.get("resource_node"),
        "owner": job.get("owner"),
        "repo_root": ROOT,
        "resource": job.get("resource"),
        "resource_class": job.get("resource_class"),
        "sync_id": job.get("sync_id"),
        "tenant": job.get("tenant"),
        "worker_alias": job.get("worker_alias") or metadata.get("worker_alias"),
        "worker_gpu_alias": job.get("worker_alias") or metadata.get("worker_alias"),
    }
    for key, replacement in metadata.items():
        if key not in replacements and isinstance(replacement, (str, int, float)):
            replacements[key] = replacement
    out = value
    for key, replacement in replacements.items():
        if replacement is not None:
            out = out.replace("{" + key + "}", str(replacement))
    return out


def resolve_repo_relative_part(value: str, *, executable: bool = False) -> str:
    if executable and value.startswith("~"):
        return str(Path(value).expanduser())
    path = Path(value)
    if path.is_absolute():
        return value
    if value.startswith(("scripts/", "configs/")):
        candidate = ROOT / path
        if candidate.exists():
            return str(candidate)
    return value


def command(value: Any, job: dict | None = None) -> list[str]:
    if isinstance(value, list):
        argv = [render_command_part(str(part), job) for part in value]
    elif isinstance(value, str):
        argv = shlex.split(render_command_part(value, job))
    else:
        return []
    return [resolve_repo_relative_part(part, executable=index == 0) for index, part in enumerate(argv)]


def job_cwd(job: dict) -> str | None:
    value = job.get("cwd")
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def job_env(job: dict) -> dict[str, str] | None:
    raw = job.get("env")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"job {job.get('id', '<unknown>')!r} env must be an object")
    env = dict(os.environ)
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"job {job.get('id', '<unknown>')!r} env keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"job {job.get('id', '<unknown>')!r} env[{key!r}] must be scalar")
        env[key] = str(value)
    return env


def run_capture_for_job(
    argv: list[str],
    job: dict,
    *,
    timeout: float,
    use_job_context: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not use_job_context:
        return run_capture(argv, timeout=timeout)
    return run_capture(argv, timeout=timeout, cwd=job_cwd(job), env=job_env(job))


def run_capture(
    argv: list[str],
    *,
    timeout: float,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        cwd=cwd,
        env=env,
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


def append_jsonl(path: str | Path | None, payload: dict) -> None:
    if not path:
        return
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def require_queue_event_log(args: argparse.Namespace, *, command: str) -> str:
    path = getattr(args, "event_log_jsonl", None)
    if not path:
        raise ValueError(
            f"{command} requires --event-log-jsonl unless --dry-run is set"
        )
    return str(path)


def parse_stdout_json(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def load_jobs(path: str, inventory: dict[str, dict] | None = None) -> list[dict]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        jobs = raw.get("jobs")
    else:
        jobs = raw
    if not isinstance(jobs, list):
        raise SystemExit("--jobs-json must contain a list or an object with jobs")
    expanded = []
    for job in jobs:
        expanded.extend(expand_job_resources(job, inventory or {}))
    return [normalize_job(job, inventory=inventory) for job in expanded]


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


def ensure_string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"{field} must be a string or string list")


def optional_string_list(value: Any, *, field: str) -> list[str] | None:
    if value is None:
        return None
    return ensure_string_list(value, field=field)


def resource_tags(row: dict) -> set[str]:
    tags = row.get("tags", [])
    if tags is None:
        return set()
    if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
        raise ValueError(f"resource {row.get('id', '<unknown>')!r} tags must be a string list")
    return set(tags)


def job_owner_and_tenant(job: dict) -> tuple[str, str]:
    owner = require_str(job, "owner")
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    tenant = str(job.get("tenant") or metadata.get("tenant") or owner.split(":", 1)[0]).strip()
    if not tenant:
        raise ValueError(f"job {job.get('id', '<unknown>')!r} field 'tenant' must be a non-empty string")
    return owner, tenant


def principal_allowed_for_reserved_resource(owner: str, tenant: str, rules: list[str]) -> bool:
    return any(
        owner_matches_rule(owner, rule) or owner_matches_rule(tenant, rule)
        for rule in rules
    )


def resource_allowed_for_job(row: dict, job: dict, *, include_reserved: bool) -> bool:
    status = str(row.get("status", "active"))
    if status == "blocked":
        return False
    owner, tenant = job_owner_and_tenant(job)
    general = row.get("general_queue_eligible", True)
    reserved_for = row.get("reserved_for") or []
    if general is False:
        if not reserved_for:
            return False
        if not principal_allowed_for_reserved_resource(owner, tenant, reserved_for):
            return False
    if status == "reserved" and not include_reserved:
        return bool(reserved_for and principal_allowed_for_reserved_resource(owner, tenant, reserved_for))
    return True


def resource_matches_pool(resource_id: str, row: dict, pool: Any) -> bool:
    if isinstance(pool, str):
        return (
            resource_id == pool
            or row.get("backend") == pool
            or row.get("class") == pool
            or pool in resource_tags(row)
        )
    if isinstance(pool, list):
        if not all(isinstance(item, str) for item in pool):
            raise ValueError("resource_pool list must contain only strings")
        return resource_id in pool
    if not isinstance(pool, dict):
        raise ValueError("resource_pool must be a string, string list, or object")

    exact = optional_string_list(
        pool.get("resource") if "resource" in pool else pool.get("resources"),
        field="resource_pool.resources",
    )
    if exact is not None and resource_id not in exact:
        return False

    backends = optional_string_list(pool.get("backend") or pool.get("backends"), field="resource_pool.backend")
    if backends is not None and str(row.get("backend") or "") not in backends:
        return False

    classes = optional_string_list(pool.get("class") or pool.get("classes"), field="resource_pool.class")
    if classes is not None and str(row.get("class") or "") not in classes:
        return False

    nodes = optional_string_list(pool.get("node") or pool.get("nodes"), field="resource_pool.node")
    if nodes is not None and str(row.get("node") or "") not in nodes:
        return False

    tags = optional_string_list(pool.get("tag") or pool.get("tags"), field="resource_pool.tags")
    if tags is not None and not set(tags).issubset(resource_tags(row)):
        return False

    min_memory = pool.get("min_memory_gib")
    if min_memory is not None:
        try:
            required = float(min_memory)
            available = float(row.get("memory_gib", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("resource_pool.min_memory_gib must be numeric") from exc
        if available < required:
            return False

    return True


def resource_candidates_for_job(job: Any, inventory: dict[str, dict]) -> tuple[list[str], bool]:
    if not isinstance(job, dict):
        raise SystemExit("--jobs-json contains a non-object job")
    selectors = [
        key
        for key in ("resource", "resources", "resource_pool", "resource_selector")
        if key in job and job.get(key) not in (None, "")
    ]
    if not selectors:
        raise ValueError(f"job {job.get('id', '<unknown>')!r} requires resource, resources, or resource_pool")
    if len(selectors) > 1:
        raise ValueError(
            f"job {job.get('id', '<unknown>')!r} must use only one of resource, resources, resource_pool"
        )
    selector = selectors[0]
    if selector == "resource":
        return [require_str(job, "resource")], False
    if selector == "resources":
        resources = ensure_string_list(job["resources"], field="resources")
        if not resources:
            raise ValueError(f"job {job.get('id', '<unknown>')!r} resources must not be empty")
        return resources, True
    if not inventory:
        raise ValueError(f"job {job.get('id', '<unknown>')!r} uses resource_pool but no inventory is loaded")

    pool = job.get(selector)
    include_reserved = bool(pool.get("include_reserved", False)) if isinstance(pool, dict) else False
    resources = [
        resource_id
        for resource_id, row in inventory.items()
        if resource_matches_pool(resource_id, row, pool)
        and resource_allowed_for_job(row, job, include_reserved=include_reserved)
    ]
    if not resources:
        raise ValueError(f"job {job.get('id', '<unknown>')!r} resource_pool matched no eligible resources")
    return resources, True


def expand_job_resources(job: Any, inventory: dict[str, dict]) -> list[dict]:
    resources, pooled = resource_candidates_for_job(job, inventory)
    if len(set(resources)) != len(resources):
        raise ValueError(f"job {job.get('id', '<unknown>')!r} has duplicate resource candidates")
    logical_id = require_str(job, "id")
    out = []
    for resource in resources:
        clone = dict(job)
        clone["logical_id"] = str(clone.get("logical_id") or logical_id)
        clone["resource"] = resource
        clone.pop("resources", None)
        clone.pop("resource_selector", None)
        if pooled:
            clone["id"] = f"{logical_id}@{resource}"
            clone.setdefault("sync_id", logical_id)
        out.append(clone)
    return out


def owner_matches_rule(owner: str, rule: str) -> bool:
    if rule.endswith("*"):
        return owner.startswith(rule[:-1])
    return owner == rule


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
            if not principal_allowed_for_reserved_resource(job["owner"], job["tenant"], reserved_for):
                errors.append(
                    f"job {job['id']!r} tenant {job['tenant']!r} owner {job['owner']!r} is not allowed "
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


def require_submit_bool(spec: dict, key: str, default: bool) -> bool:
    value = spec.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"tensorcore.job.v1 field {key!r} must be a JSON boolean")
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
    out["logical_id"] = str(out.get("logical_id") or out["id"])
    out["sync_id"] = str(out.get("sync_id") or out["id"])
    out["resource"] = require_str(out, "resource")
    out["owner"] = require_str(out, "owner")
    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    out["tenant"] = str(out.get("tenant") or metadata.get("tenant") or out["owner"].split(":", 1)[0]).strip()
    if not out["tenant"]:
        raise ValueError(f"job {out['id']!r} field 'tenant' must be a non-empty string")
    out["priority"] = int(out.get("priority", 0))
    out["max_parallel"] = int(out.get("max_parallel", 1))
    if out["max_parallel"] < 1:
        raise ValueError(f"job {out['id']!r} field 'max_parallel' must be >= 1")
    tenant_max_parallel = out.get("tenant_max_parallel")
    if tenant_max_parallel is not None:
        out["tenant_max_parallel"] = int(tenant_max_parallel)
        if out["tenant_max_parallel"] < 1:
            raise ValueError(f"job {out['id']!r} field 'tenant_max_parallel' must be >= 1")
    adopt_keys = out.get("adopt_unknown_lease_metadata_keys", [])
    if adopt_keys is None:
        adopt_keys = []
    if not isinstance(adopt_keys, list) or not all(isinstance(item, str) for item in adopt_keys):
        raise ValueError(
            f"job {out['id']!r} field 'adopt_unknown_lease_metadata_keys' must be a string list"
        )
    out["adopt_unknown_lease_metadata_keys"] = adopt_keys
    out["enabled"] = require_bool(out, "enabled", True)
    out["ttl_sec"] = float(out.get("ttl_sec", 900.0))
    if out["ttl_sec"] <= 0:
        raise ValueError(f"job {out['id']!r} field 'ttl_sec' must be positive")
    if out.get("cwd") is not None and not isinstance(out.get("cwd"), str):
        raise ValueError(f"job {out['id']!r} field 'cwd' must be a string when set")
    if out.get("env") is not None:
        job_env(out)
    if out.get("preemption_policy") is not None and not isinstance(out.get("preemption_policy"), dict):
        raise ValueError(f"job {out['id']!r} field 'preemption_policy' must be an object when set")
    if out.get("quality_gates") is not None and not isinstance(out.get("quality_gates"), list):
        raise ValueError(f"job {out['id']!r} field 'quality_gates' must be a list when set")
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
    inventory_row = inventory.get(out["resource"]) if inventory else None
    if inventory_row:
        out["resource_node"] = str(inventory_row.get("node") or out["resource"].split(":", 1)[0])
        out["resource_backend"] = str(inventory_row.get("backend") or "")
        out["inventory_class"] = str(inventory_row.get("class") or "")
        if not out.get("worker_alias") and inventory_row.get("worker_alias"):
            out["worker_alias"] = str(inventory_row["worker_alias"])
    else:
        out["resource_node"] = str(out.get("resource_node") or out["resource"].split(":", 1)[0])
        out["resource_backend"] = str(out.get("resource_backend") or "")
        out["inventory_class"] = str(out.get("inventory_class") or "")
    requires_host_gate = out["resource_class"] == "cuda_exclusive"
    validate_command_field(out, "probe_cmd", required=False)
    validate_command_field(out, "start_cmd", required=out["desired_state"] == "running")
    validate_command_field(out, "completion_cmd", required=False)
    validate_command_field(out, "admission_cmd", required=requires_host_gate)
    validate_command_field(out, "post_start_probe_cmd", required=requires_host_gate)
    validate_command_field(out, "worker_identity_cmd", required=requires_host_gate)

    metadata = dict(metadata)
    metadata.setdefault("surface", "tensorcore_mesh_scheduler")
    metadata["sync_job_id"] = out["sync_id"]
    metadata["job_id"] = out["id"]
    metadata["logical_job_id"] = out["logical_id"]
    metadata["resource_class"] = out["resource_class"]
    metadata["tenant"] = out["tenant"]
    metadata["resource"] = out["resource"]
    metadata["resource_backend"] = out["resource_backend"]
    metadata["resource_node"] = out["resource_node"]
    if out.get("worker_alias"):
        metadata["worker_alias"] = out["worker_alias"]
    metadata["scheduler_host"] = os.uname().nodename
    metadata["worker_identity_pending"] = requires_host_gate
    out["metadata"] = metadata
    return out


def probe_job(job: dict, *, timeout: float) -> dict:
    probe_cmd = command(job.get("probe_cmd"), job)
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
    completion_cmd = command(job.get("completion_cmd"), job)
    if not completion_cmd:
        return {"complete": False, "reason": "missing_completion_cmd", "rc": None}
    try:
        proc = run_capture(completion_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"complete": None, "reason": "completion_timeout", "rc": None}
    result = {
        "complete": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "completion_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    parsed = parse_stdout_json(proc.stdout)
    if parsed is not None:
        result["json"] = parsed
    return result


def admit_job(job: dict, *, timeout: float) -> dict:
    admission_cmd = command(job.get("admission_cmd"), job)
    if not admission_cmd:
        return {"admitted": True, "reason": "missing_admission_cmd", "rc": None}
    try:
        proc = run_capture(admission_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"admitted": None, "reason": "admission_timeout", "rc": None}
    result = {
        "admitted": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "admission_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    parsed = parse_stdout_json(proc.stdout)
    if parsed is not None:
        result["json"] = parsed
    return result


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


def lease_tenant(row: dict) -> str:
    metadata = lease_metadata(row)
    owner = str(row.get("owner") or "")
    return str(metadata.get("tenant") or owner.split(":", 1)[0] or owner)


def adoptable_unknown_leases(job: dict, leases: list[dict]) -> tuple[list[dict], list[dict]]:
    keys = job.get("adopt_unknown_lease_metadata_keys") or []
    if not keys:
        return [], list(leases)
    job_metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    adopted = []
    remaining = []
    for lease in leases:
        metadata = lease_metadata(lease)
        if lease_tenant(lease) != job["tenant"]:
            remaining.append(lease)
            continue
        if all(
            key in job_metadata
            and key in metadata
            and str(job_metadata[key]) == str(metadata[key])
            for key in keys
        ):
            adopted.append(lease)
        else:
            remaining.append(lease)
    return adopted, remaining


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


def heartbeat_lease(
    lease: dict,
    job: dict,
    *,
    arbiter_cmd: list[str],
    timeout: float,
    worker_identity: dict | None = None,
) -> dict:
    base = ["heartbeat", str(lease["id"]), "--ttl-sec", str(job["ttl_sec"])]
    if worker_identity is not None:
        metadata = claim_metadata(job, worker_identity=worker_identity)
        try:
            payload = run_json(
                arbiter_cmd
                + base
                + ["--metadata-json", json.dumps(metadata, sort_keys=True), "--json"],
                timeout=timeout,
            )
            payload["metadata_refreshed"] = True
            return payload
        except RuntimeError as exc:
            fallback = run_json(arbiter_cmd + base + ["--json"], timeout=timeout)
            fallback["metadata_refreshed"] = False
            fallback["metadata_refresh_error"] = str(exc)[-1000:]
            return fallback
    return run_json(arbiter_cmd + base + ["--json"], timeout=timeout)


def nested_get(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def job_evidence_path(job: dict) -> Path | None:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    raw = metadata.get("evidence_path")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def first_dict(*values: Any) -> dict:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def evidence_worker_identity(result: dict, leases: list[dict]) -> dict:
    identity = nested_get(result, ("worker_identity", "identity"))
    if isinstance(identity, dict):
        return identity
    for lease in leases:
        metadata = lease_metadata(lease)
        worker_identity = metadata.get("worker_identity")
        if isinstance(worker_identity, dict):
            return worker_identity
    return {}


def evidence_lease_metadata(leases: list[dict]) -> list[dict]:
    return [lease_metadata(lease) for lease in leases if lease_metadata(lease)]


def positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def write_scheduler_evidence(
    job: dict,
    result: dict,
    *,
    phase: str,
    leases: list[dict] | None = None,
) -> dict | None:
    path = job_evidence_path(job)
    if path is None:
        return None
    lease_rows = list(leases or [])
    admission = first_dict(nested_get(result, ("admission", "json")))
    start = first_dict(nested_get(result, ("start", "json")))
    post_start = first_dict(nested_get(result, ("post_start", "json")))
    completion = first_dict(nested_get(result, ("completion", "json")))
    artifact = first_dict(
        nested_get(completion, ("artifact",)),
        nested_get(post_start, ("artifact",)),
        nested_get(start, ("payload",)),
    )
    worker_identity = evidence_worker_identity(result, lease_rows)
    lease_metadata_rows = evidence_lease_metadata(lease_rows)
    claim = result.get("arbiter") if isinstance(result.get("arbiter"), dict) else {}
    lease_id = (
        result.get("lease_id")
        or claim.get("lease_id")
        or claim.get("id")
        or next((lease.get("id") for lease in lease_rows if lease.get("id")), None)
    )
    scheduler_lease_held = bool(lease_id) and (
        bool(claim.get("ok"))
        or any(
            metadata.get("surface") == "tensorcore_mesh_scheduler"
            for metadata in lease_metadata_rows
        )
    )
    worker_identity_recorded = bool(worker_identity) and (
        nested_get(result, ("worker_identity", "ok")) is True
        or any(metadata.get("worker_identity_pending") is False for metadata in lease_metadata_rows)
    )
    payload = {
        "schema": "tensorcore.windows_cuda_scheduled_smoke.evidence.v1",
        "schema_version": 1,
        "checked_at_unix": time.time(),
        "phase": phase,
        "resource": job["resource"],
        "job": job["id"],
        "driver_visible": admission.get("driver_ok") is True or positive_int(admission.get("device_count")),
        "toolchain_found": admission.get("toolchain_ok") is True or bool(artifact.get("nvcc_path")),
        "wddm_admission_ok": admission.get("admission_ok") is True and admission.get("ok") is True,
        "build_smoke_passed": artifact.get("build_ok") is True,
        "runtime_smoke_passed": artifact.get("runtime_ok") is True,
        "scheduler_lease_held": scheduler_lease_held,
        "worker_identity_recorded": worker_identity_recorded,
        "lease_id": lease_id,
        "admission": admission,
        "start": start,
        "post_start": post_start,
        "completion": completion,
        "smoke_artifact": artifact,
        "worker_identity": worker_identity,
        "worker_identity_heartbeat": result.get("worker_identity_heartbeat"),
        "lease_metadata": lease_metadata_rows,
    }
    write_json(str(path), payload)
    return {"ok": True, "path": str(path), "schema": payload["schema"]}


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
    start_cmd = command(job.get("start_cmd"), job)
    if not start_cmd:
        raise RuntimeError(f"job {job['id']} has no start_cmd")
    proc = run_capture_for_job(start_cmd, job, timeout=timeout)
    result = {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    parsed = parse_stdout_json(proc.stdout)
    if parsed is not None:
        result["json"] = parsed
    return result


def post_start_probe_job(
    job: dict,
    *,
    timeout: float,
    interval: float,
) -> dict:
    probe_cmd = command(job.get("post_start_probe_cmd"), job)
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
            parsed = parse_stdout_json(proc.stdout)
            if parsed is not None:
                last["json"] = parsed
            if proc.returncode == 0:
                return last
        if interval <= 0:
            return last
        if time.monotonic() >= deadline:
            last["reason"] = "post_start_timeout"
            return last
        time.sleep(max(0.0, min(interval, deadline - time.monotonic())))


def collect_worker_identity(job: dict, *, timeout: float) -> dict:
    identity_cmd = command(job.get("worker_identity_cmd"), job)
    if not identity_cmd:
        if job.get("resource_class") != "cuda_exclusive":
            return {"ok": True, "reason": "not_required", "rc": None, "identity": {}}
        return {"ok": False, "reason": "missing_worker_identity_cmd", "rc": None}
    try:
        proc = run_capture(identity_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "worker_identity_timeout", "rc": None}
    result: dict[str, Any] = {
        "ok": False,
        "reason": "worker_identity_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }
    if proc.returncode != 0:
        return result
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result["reason"] = "invalid_worker_identity_json"
        return result
    if not isinstance(parsed, dict):
        result["reason"] = "invalid_worker_identity_payload"
        return result
    result["identity"] = parsed
    if parsed.get("schema") != "tensorcore.mesh_worker_identity.v1":
        result["reason"] = "invalid_worker_identity_schema"
        return result
    if parsed.get("resource") != job.get("resource"):
        result["reason"] = "worker_identity_resource_mismatch"
        return result
    if parsed.get("ok") is not True:
        result["reason"] = str(parsed.get("reason") or "worker_identity_payload_not_ok")
        return result
    result["ok"] = True
    result["reason"] = "ok"
    return result


def launch_plan(job: dict) -> dict:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    plan = {
        "schema": "tensorcore.cluster_launch_plan.v1",
        "job": job.get("id"),
        "logical_job": job.get("logical_id"),
        "resource": job.get("resource"),
        "resource_class": job.get("resource_class"),
        "owner": job.get("owner"),
        "tenant": job.get("tenant"),
        "start_cmd": command(job.get("start_cmd"), job),
        "probe_cmd": command(job.get("probe_cmd"), job),
        "completion_cmd": command(job.get("completion_cmd"), job),
        "admission_cmd": command(job.get("admission_cmd"), job),
        "preflight_cmd": command(job.get("preflight_cmd"), job),
        "post_start_probe_cmd": command(job.get("post_start_probe_cmd"), job),
        "worker_identity_cmd": command(job.get("worker_identity_cmd"), job),
        "cwd": job_cwd(job),
        "env_keys": sorted((job.get("env") or {}).keys()) if isinstance(job.get("env"), dict) else [],
        "artifact_root": job.get("artifact_root") or metadata.get("artifact_root"),
        "evidence_path": str(job_evidence_path(job)) if job_evidence_path(job) else None,
        "finalizer_cmd": command(job.get("finalizer_cmd"), job),
        "preemption_policy": job.get("preemption_policy") or metadata.get("preemption_policy"),
        "quality_gates": job.get("quality_gates") or metadata.get("quality_gates") or [],
        "run_intent_required": metadata.get("require_run_intent") is not False,
    }
    return plan


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


def initial_scheduler_counts(
    jobs: list[dict],
    probes: dict[str, dict],
    status: dict,
) -> dict[str, dict[str, int]]:
    tenant_counts: dict[str, int] = {}
    logical_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    def note(job: dict) -> None:
        key = (job["resource"], job["logical_id"])
        if key in seen:
            return
        seen.add(key)
        tenant_counts[job["tenant"]] = tenant_counts.get(job["tenant"], 0) + 1
        logical_counts[job["logical_id"]] = logical_counts.get(job["logical_id"], 0) + 1

    for job in jobs:
        if probes[job["id"]]["live"] is True:
            note(job)
    for lease in status.get("leases") or []:
        if not isinstance(lease, dict):
            continue
        job = job_for_lease(lease, jobs)
        if job is not None and probes[job["id"]]["live"] is not False:
            note(job)
    return {"tenant": tenant_counts, "logical": logical_counts}


def logical_parallel_available(job: dict, counts: dict[str, dict[str, int]]) -> bool:
    return counts["logical"].get(job["logical_id"], 0) < job["max_parallel"]


def tenant_parallel_limit(job: dict, max_running_per_tenant: int) -> int:
    return int(job.get("tenant_max_parallel") or max_running_per_tenant or 0)


def tenant_parallel_available(
    job: dict,
    counts: dict[str, dict[str, int]],
    *,
    max_running_per_tenant: int,
) -> bool:
    limit = tenant_parallel_limit(job, max_running_per_tenant)
    if limit <= 0:
        return True
    return counts["tenant"].get(job["tenant"], 0) < limit


def mark_planned(job: dict, counts: dict[str, dict[str, int]]) -> None:
    counts["tenant"][job["tenant"]] = counts["tenant"].get(job["tenant"], 0) + 1
    counts["logical"][job["logical_id"]] = counts["logical"].get(job["logical_id"], 0) + 1


def choose_candidate(
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    admissions: dict[str, dict],
    resource: str,
    counts: dict[str, dict[str, int]],
    *,
    max_running_per_tenant: int,
) -> dict | None:
    candidates = [
        job
        for job in enabled_running_jobs(jobs, resource)
        if probes[job["id"]]["live"] is False
        and completions[job["id"]]["complete"] is False
        and admissions[job["id"]]["admitted"] is True
        and logical_parallel_available(job, counts)
        and tenant_parallel_available(
            job,
            counts,
            max_running_per_tenant=max_running_per_tenant,
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            counts["tenant"].get(item["tenant"], 0),
            -item["priority"],
            item["id"],
        ),
    )[0]


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
        if not dry_run and completion["complete"] is True and result.get("ok"):
            evidence = write_scheduler_evidence(
                job,
                result,
                phase="completed",
                leases=leases,
            )
            if evidence is not None:
                result["scheduler_evidence"] = evidence
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


def unknown_lease_quarantine_candidate(
    resource: str,
    unknown_leases: list[dict],
    *,
    age_threshold_sec: float,
) -> dict | None:
    if age_threshold_sec <= 0:
        return None
    now = time.time()
    candidates = []
    for lease in unknown_leases:
        metadata = lease.get("metadata") if isinstance(lease.get("metadata"), dict) else {}
        if isinstance(metadata.get("worker_identity"), dict):
            continue
        if metadata.get("worker_identity_pending") is False:
            continue
        timestamp = None
        for key in ("acquired_at", "created_at", "updated_at"):
            raw = lease.get(key)
            if isinstance(raw, (int, float)):
                timestamp = float(raw)
                break
        if timestamp is None:
            continue
        age_sec = max(0.0, now - timestamp)
        if age_sec < age_threshold_sec:
            continue
        candidates.append(
            {
                "id": lease.get("id"),
                "owner": lease.get("owner"),
                "age_sec": age_sec,
                "metadata_keys": sorted(str(key) for key in metadata.keys()),
            }
        )
    if not candidates:
        return None
    return {
        "resource": resource,
        "action": "stale_unknown_quarantine_candidate",
        "lease_ids": [row.get("id") for row in candidates],
        "ok": True,
        "quarantine_recommended": True,
        "evidence": {
            "age_threshold_sec": age_threshold_sec,
            "reason": "unknown lease has no scheduler job and no worker identity metadata",
            "unknown_leases": candidates,
        },
    }


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
                heartbeat_identity = (
                    identity if job.get("resource_class") == "cuda_exclusive" else None
                )
                payload = heartbeat_lease(
                    current,
                    job,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                    worker_identity=heartbeat_identity,
                )
                result["arbiter"] = {"release": releases, "heartbeat": payload}
                result["ok"] = (
                    not other_known
                    and bool(identity.get("ok"))
                    and releases_ok
                    and bool(payload.get("ok"))
                    and payload.get("metadata_refreshed", True) is not False
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

    adopted_unknown, remaining_unknown = adoptable_unknown_leases(job, unknown_leases)
    if adopted_unknown and not remaining_unknown:
        current = adopted_unknown[0]
        stale = adopted_unknown[1:]
        action = (
            "would_adopt_unknown_lease_live_holder"
            if dry_run
            else "adopted_unknown_lease_live_holder"
        )
        identity = collect_worker_identity(job, timeout=worker_identity_timeout)
        result = {
            "resource": job["resource"],
            "job": job["id"],
            "action": action,
            "lease_id": current.get("id"),
            "stale_lease_ids": [lease.get("id") for lease in stale],
            "probe": probes[job["id"]],
            "worker_identity": identity,
            "ok": bool(identity.get("ok")),
        }
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
                    worker_identity=identity,
                )
                result["arbiter"] = {"release": releases, "heartbeat": payload}
                result["ok"] = (
                    bool(identity.get("ok"))
                    and releases_ok
                    and bool(payload.get("ok"))
                    and payload.get("metadata_refreshed", True) is not False
                )
            except Exception as exc:
                result["ok"] = False
                errors.append({"resource": job["resource"], "job": job["id"], "error": str(exc)})
        results.append(result)
        return results, errors

    if unknown_leases:
        results.append({
            "resource": job["resource"],
            "job": job["id"],
            "action": "live_holder_blocked_by_unknown_lease",
            "lease_ids": [lease.get("id") for lease in remaining_unknown or unknown_leases],
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
    counts: dict[str, dict[str, int]],
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
        quarantine = unknown_lease_quarantine_candidate(
            resource,
            unknown_leases,
            age_threshold_sec=args.unknown_lease_quarantine_age_sec,
        )
        if quarantine is not None:
            results.append(quarantine)
        else:
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

    candidate = choose_candidate(
        jobs,
        probes,
        completions,
        admissions,
        resource,
        counts,
        max_running_per_tenant=args.max_running_per_tenant,
    )
    if candidate is None:
        parallel_blocked_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if probes[job["id"]]["live"] is False
            and completions[job["id"]]["complete"] is False
            and admissions[job["id"]]["admitted"] is True
            and not logical_parallel_available(job, counts)
        ]
        if parallel_blocked_jobs:
            results.append({
                "resource": resource,
                "action": "idle_logical_parallel_limit",
                "jobs": parallel_blocked_jobs,
                "ok": True,
            })
            return results, errors
        tenant_blocked_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if probes[job["id"]]["live"] is False
            and completions[job["id"]]["complete"] is False
            and admissions[job["id"]]["admitted"] is True
            and logical_parallel_available(job, counts)
            and not tenant_parallel_available(
                job,
                counts,
                max_running_per_tenant=args.max_running_per_tenant,
            )
        ]
        if tenant_blocked_jobs:
            results.append({
                "resource": resource,
                "action": "idle_tenant_parallel_limit",
                "jobs": tenant_blocked_jobs,
                "ok": True,
            })
            return results, errors
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
            "launch_plan": launch_plan(candidate),
            "ok": True,
        })
        mark_planned(candidate, counts)
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
    lease_id = claim_payload.get("lease_id") or claim_payload.get("id")
    if lease_id:
        candidate = dict(candidate)
        candidate["lease_id"] = str(lease_id)

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
        if result["ok"] and candidate.get("resource_class") == "cuda_exclusive":
            identity = collect_worker_identity(
                candidate,
                timeout=args.worker_identity_timeout_sec,
            )
            result["worker_identity"] = identity
            result["ok"] = bool(identity.get("ok"))
            if result["ok"]:
                lease_id = claim_payload.get("lease_id") or claim_payload.get("id")
                if lease_id:
                    try:
                        identity_heartbeat = heartbeat_lease(
                            {"id": lease_id},
                            candidate,
                            arbiter_cmd=arbiter_cmd,
                            timeout=args.timeout_sec,
                            worker_identity=identity,
                        )
                        result["worker_identity_heartbeat"] = identity_heartbeat
                        result["ok"] = (
                            bool(identity_heartbeat.get("ok"))
                            and identity_heartbeat.get("metadata_refreshed", True) is not False
                        )
                    except Exception as exc:
                        result["ok"] = False
                        errors.append({
                            "resource": resource,
                            "job": candidate["id"],
                            "error": f"failed to refresh worker identity metadata: {exc}",
                        })
    if result.get("ok"):
        evidence = write_scheduler_evidence(
            candidate,
            result,
            phase="launched",
            leases=[],
        )
        if evidence is not None:
            result["scheduler_evidence"] = evidence

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
    if result["ok"]:
        mark_planned(candidate, counts)
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
    counts = initial_scheduler_counts(jobs, probes, status)
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
                counts=counts,
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


def read_json_object(path: str | Path) -> dict:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_jobs_doc(path: str | Path, *, missing_ok: bool = False) -> dict:
    queue_path = Path(path).expanduser()
    if missing_ok and not queue_path.exists():
        return {"schema": SCHEMA, "jobs": []}
    payload = read_json_object(queue_path)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{queue_path} schema must be {SCHEMA}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"{queue_path} jobs must be a list")
    return {"schema": SCHEMA, "jobs": list(jobs)}


def write_jobs_doc(path: str | Path, doc: dict) -> None:
    write_json(str(Path(path).expanduser()), {"schema": SCHEMA, "jobs": doc.get("jobs", [])})


def submit_command_argv(value: Any, *, field: str) -> list[str]:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, str) and shlex.split(value):
        return shlex.split(value)
    if isinstance(value, dict):
        argv = value.get("argv")
        if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
            return list(argv)
        shell = value.get("shell")
        if isinstance(shell, str) and shell.strip():
            return ["sh", "-lc", shell]
    raise ValueError(f"{field} must be a command string, argv list, or object with argv/shell")


def optional_submit_command(spec: dict, *keys: str) -> list[str] | None:
    for key in keys:
        if key in spec and spec.get(key) not in (None, ""):
            return submit_command_argv(spec[key], field=key)
    return None


def submit_resources(spec: dict) -> tuple[str, Any]:
    resources = spec.get("resources")
    if isinstance(resources, dict):
        if resources.get("resource"):
            return "resource", resources["resource"]
        if resources.get("resources"):
            return "resources", resources["resources"]
        if resources.get("resource_pool"):
            return "resource_pool", resources["resource_pool"]
        if resources.get("selector"):
            return "resource_pool", resources["selector"]
    for key in ("resource", "resources", "resource_pool", "resource_selector"):
        if key in spec and spec.get(key) not in (None, ""):
            return ("resource_pool" if key == "resource_selector" else key), spec[key]
    raise ValueError("tensorcore.job.v1 requires resources.resource, resources.resources, or resources.selector")


def submit_spec_to_mesh_job(spec: dict) -> dict:
    if spec.get("schema") != SUBMIT_SCHEMA:
        raise ValueError(f"submit spec schema must be {SUBMIT_SCHEMA}")
    job_id = require_str(spec, "id")
    if "artifact" in spec and spec.get("artifact") is not None and not isinstance(spec.get("artifact"), dict):
        raise ValueError("tensorcore.job.v1 field 'artifact' must be an object when set")
    if "resources" in spec and spec.get("resources") is not None and not isinstance(spec.get("resources"), dict):
        raise ValueError("tensorcore.job.v1 field 'resources' must be an object when set")
    if "preemption_policy" in spec and spec.get("preemption_policy") is not None and not isinstance(spec.get("preemption_policy"), dict):
        raise ValueError("tensorcore.job.v1 field 'preemption_policy' must be an object when set")
    if "quality_gates" in spec and spec.get("quality_gates") is not None and not isinstance(spec.get("quality_gates"), list):
        raise ValueError("tensorcore.job.v1 field 'quality_gates' must be a list when set")
    command_spec = spec.get("command")
    start_cmd = optional_submit_command(spec, "start_cmd")
    if start_cmd is None:
        if command_spec is None:
            raise ValueError("tensorcore.job.v1 requires command or start_cmd")
        start_cmd = submit_command_argv(command_spec, field="command")
    selector_key, selector_value = submit_resources(spec)
    resources = spec.get("resources") if isinstance(spec.get("resources"), dict) else {}
    selector_backend = ""
    if isinstance(selector_value, dict):
        selector_backend = str(selector_value.get("backend") or "")
    metadata = dict(spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {})
    artifact = spec.get("artifact") if isinstance(spec.get("artifact"), dict) else {}
    preemption_policy = spec.get("preemption_policy") if isinstance(spec.get("preemption_policy"), dict) else {}
    quality_gates = spec.get("quality_gates") if isinstance(spec.get("quality_gates"), list) else []
    metadata.setdefault("surface", "tensorcore_cluster_scheduler")
    metadata.setdefault("submit_schema", SUBMIT_SCHEMA)
    metadata.setdefault("scheduler_contract", "tensorcore_job_v1")
    metadata.setdefault("preemption_policy", preemption_policy)
    metadata.setdefault("quality_gates", quality_gates)
    metadata.setdefault("artifact_root", artifact.get("root") or spec.get("artifact_root"))
    metadata.setdefault("evidence_path", artifact.get("evidence_path") or spec.get("evidence_path"))
    metadata.setdefault("require_run_intent", True)
    out: dict[str, Any] = {
        "id": job_id,
        "sync_id": str(spec.get("sync_id") or job_id),
        selector_key: selector_value,
        "resource_class": str(
            spec.get("resource_class")
            or resources.get("resource_class")
            or ("cuda_exclusive" if resources.get("exclusive") is True or selector_backend.lower() == "cuda" else "")
            or "generic"
        ),
        "owner": require_str(spec, "owner"),
        "tenant": str(spec.get("tenant") or require_str(spec, "owner").split(":", 1)[0]),
        "priority": int(spec.get("priority", 0)),
        "desired_state": str(spec.get("desired_state") or "running"),
        "ttl_sec": float(spec.get("ttl_sec", 900.0)),
        "enabled": require_submit_bool(spec, "enabled", True),
        "start_cmd": start_cmd,
        "metadata": metadata,
    }
    for source_key, target_key in (
        ("probe", "probe_cmd"),
        ("completion", "completion_cmd"),
        ("admission", "admission_cmd"),
        ("preflight", "preflight_cmd"),
        ("post_start_probe", "post_start_probe_cmd"),
        ("worker_identity", "worker_identity_cmd"),
        ("finalizer", "finalizer_cmd"),
    ):
        value = optional_submit_command(spec, target_key, source_key)
        if value is not None:
            out[target_key] = value
    if isinstance(command_spec, dict):
        if isinstance(command_spec.get("cwd"), str):
            out["cwd"] = command_spec["cwd"]
        if isinstance(command_spec.get("env"), dict):
            out["env"] = dict(command_spec["env"])
    if isinstance(spec.get("cwd"), str):
        out["cwd"] = spec["cwd"]
    if isinstance(spec.get("env"), dict):
        out["env"] = dict(spec["env"])
    if artifact.get("root") or spec.get("artifact_root"):
        out["artifact_root"] = artifact.get("root") or spec.get("artifact_root")
    if preemption_policy:
        out["preemption_policy"] = preemption_policy
    if quality_gates:
        out["quality_gates"] = quality_gates
    if "max_parallel" in spec:
        out["max_parallel"] = int(spec["max_parallel"])
    if "tenant_max_parallel" in spec:
        out["tenant_max_parallel"] = int(spec["tenant_max_parallel"])
    return out


def upsert_job(jobs: list[dict], job: dict, *, replace: bool) -> list[dict]:
    out = []
    found = False
    for row in jobs:
        if isinstance(row, dict) and row.get("id") == job["id"]:
            if not replace:
                raise ValueError(f"job {job['id']!r} already exists; pass --replace to update it")
            out.append(job)
            found = True
        else:
            out.append(row)
    if not found:
        out.append(job)
    return out


def queued_job_id_matches(row: dict, requested_id: str) -> bool:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        return False
    if requested_id == row_id:
        return True
    if "resource_pool" not in row and "resources" not in row and "resource_selector" not in row:
        return False
    return requested_id.startswith(f"{row_id}@") and len(requested_id) > len(row_id) + 1


def cmd_submit(args: argparse.Namespace) -> dict:
    inventory = load_inventory(args.inventory_json)
    spec = read_json_object(args.job_json)
    mesh_job = submit_spec_to_mesh_job(spec)
    expanded = [normalize_job(job, inventory=inventory) for job in expand_job_resources(mesh_job, inventory)]
    validate_jobs_against_inventory(expanded, inventory)
    payload = {
        "schema": "tensorcore.cluster_submit.result.v1",
        "ok": True,
        "checked_at_unix": time.time(),
        "dry_run": args.dry_run,
        "job": mesh_job,
        "expanded_jobs": [job["id"] for job in expanded],
        "launch_plans": [launch_plan(job) for job in expanded],
    }
    if args.dry_run:
        return payload
    event_log_jsonl = require_queue_event_log(args, command="submit")
    doc = load_jobs_doc(args.jobs_json, missing_ok=True)
    doc["jobs"] = upsert_job(doc["jobs"], mesh_job, replace=args.replace)
    write_jobs_doc(args.jobs_json, doc)
    append_jsonl(
        event_log_jsonl,
        {
            "schema": "tensorcore.scheduler_queue_event.v1",
            "event": "submit",
            "created_at_unix": payload["checked_at_unix"],
            "jobs_json": str(Path(args.jobs_json).expanduser()),
            "job_id": mesh_job["id"],
            "replace": bool(args.replace),
            "expanded_jobs": payload["expanded_jobs"],
        },
    )
    payload["jobs_json"] = str(Path(args.jobs_json).expanduser())
    payload["queued"] = True
    return payload


def cmd_status(args: argparse.Namespace) -> dict:
    inventory = load_inventory(args.inventory_json)
    jobs = load_jobs(args.jobs_json, inventory=inventory) if args.jobs_json else []
    status = {"leases": []}
    if not args.offline:
        status = run_json(command(args.arbiter_cmd) + ["status", "--json"], timeout=args.timeout_sec)
    resources = []
    for resource_id, row in sorted(inventory.items()):
        resource_jobs = [job for job in jobs if job["resource"] == resource_id]
        resource_leases = leases_for_resource(status, resource_id)
        resources.append({
            "resource": resource_id,
            "backend": row.get("backend"),
            "class": row.get("class"),
            "status": row.get("status", "active"),
            "control_plane": row.get("control_plane"),
            "jobs": [job["id"] for job in resource_jobs],
            "leases": [lease.get("id") for lease in resource_leases],
            "busy": bool(resource_leases),
        })
    return {
        "schema": "tensorcore.cluster_status.v1",
        "ok": True,
        "checked_at_unix": time.time(),
        "resources": resources,
        "jobs": [{"id": job["id"], "resource": job["resource"], "desired_state": job["desired_state"]} for job in jobs],
        "leases": status.get("leases") or [],
    }


def cmd_cancel(args: argparse.Namespace) -> dict:
    doc = load_jobs_doc(args.jobs_json)
    matched = []
    for job in doc["jobs"]:
        if not isinstance(job, dict) or not queued_job_id_matches(job, args.job_id):
            continue
        metadata = dict(job.get("metadata") if isinstance(job.get("metadata"), dict) else {})
        metadata["cancelled_at_unix"] = time.time()
        metadata["cancel_reason"] = args.reason
        job["metadata"] = metadata
        job["enabled"] = False
        job["desired_state"] = "paused"
        matched.append(job.get("id"))
    if not matched:
        raise ValueError(f"job {args.job_id!r} not found")
    checked_at = time.time()
    if not args.dry_run:
        event_log_jsonl = require_queue_event_log(args, command="cancel")
        write_jobs_doc(args.jobs_json, doc)
        append_jsonl(
            event_log_jsonl,
            {
                "schema": "tensorcore.scheduler_queue_event.v1",
                "event": "cancel",
                "created_at_unix": checked_at,
                "jobs_json": str(Path(args.jobs_json).expanduser()),
                "requested_job_id": args.job_id,
                "cancelled_jobs": matched,
                "reason": args.reason,
            },
        )
    return {
        "schema": "tensorcore.cluster_cancel.result.v1",
        "ok": True,
        "checked_at_unix": checked_at,
        "dry_run": args.dry_run,
        "cancelled_jobs": matched,
    }


def update_inventory_status(args: argparse.Namespace, *, drained: bool) -> dict:
    payload = read_json_object(args.inventory_json)
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ValueError("--inventory-json resources must be a list")
    found = False
    for row in resources:
        if not isinstance(row, dict) or row.get("id") != args.resource:
            continue
        found = True
        if drained:
            row["status"] = "blocked"
            row["blocked_reason"] = args.reason
        else:
            row["status"] = "active"
            row.pop("blocked_reason", None)
    if not found:
        raise ValueError(f"resource {args.resource!r} not found")
    if not args.dry_run:
        out_path = args.out_inventory_json or args.inventory_json
        write_json(str(Path(out_path).expanduser()), payload)
    return {
        "schema": "tensorcore.cluster_drain.result.v1",
        "ok": True,
        "checked_at_unix": time.time(),
        "dry_run": args.dry_run,
        "resource": args.resource,
        "status": "blocked" if drained else "active",
    }


def cmd_audit(args: argparse.Namespace) -> dict:
    inventory = load_inventory(args.inventory_json)
    jobs = load_jobs(args.jobs_json, inventory=inventory)
    errors: list[str] = []
    try:
        validate_jobs_against_inventory(jobs, inventory)
    except ValueError as exc:
        errors.append(str(exc))
    for job in jobs:
        if job["resource_class"] == "cuda_exclusive":
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            if metadata.get("require_run_intent") is False:
                errors.append(f"CUDA job {job['id']!r} disables run_intent requirement")
            if not job.get("admission_cmd") or not job.get("post_start_probe_cmd") or not job.get("worker_identity_cmd"):
                errors.append(f"CUDA job {job['id']!r} lacks admission/post-start/identity contract")
    reconciliation_paths = [Path(raw_path).expanduser() for raw_path in getattr(args, "worker_reconciliation_json", []) or []]
    for raw_dir in getattr(args, "worker_reconciliation_dir", []) or []:
        directory = Path(raw_dir).expanduser()
        if not directory.is_dir():
            errors.append(f"worker reconciliation dir {directory} is not a directory")
            continue
        reconciliation_paths.extend(sorted(directory.glob("*.reconciliation.json")))
    reconciliation_reports = []
    for path in reconciliation_paths:
        try:
            report = read_json_object(path)
        except Exception as exc:
            errors.append(f"worker reconciliation report {path} could not be read: {exc}")
            continue
        reconciliation_reports.append(report)
        if report.get("schema") != "tensorcore.mesh_worker_gpu_reconciliation.v1":
            errors.append(f"worker reconciliation report {path} has invalid schema")
            continue
        if report.get("ok") is not True:
            errors.append(
                "worker reconciliation failed for "
                f"{report.get('resource')}: {report.get('reason')}"
            )
    return {
        "schema": "tensorcore.cluster_audit.result.v1",
        "ok": not errors,
        "checked_at_unix": time.time(),
        "errors": errors,
        "job_count": len(jobs),
        "worker_reconciliation_reports": reconciliation_reports,
    }


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
    if argv and argv[0] in CONTROL_COMMANDS:
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="control_command", required=True)

        def add_output_flags(p: argparse.ArgumentParser) -> None:
            p.add_argument("--json", action="store_true")
            p.add_argument("--pretty-json", action="store_true")

        submit = sub.add_parser("submit", help="Validate and enqueue a tensorcore.job.v1 spec")
        submit.add_argument("--job-json", required=True)
        submit.add_argument("--jobs-json", required=True)
        submit.add_argument("--inventory-json", required=True)
        submit.add_argument("--event-log-jsonl")
        submit.add_argument("--replace", action="store_true")
        submit.add_argument("--dry-run", action="store_true")
        add_output_flags(submit)

        status = sub.add_parser("status", help="Report inventory, jobs, and arbiter leases")
        status.add_argument("--arbiter-cmd", default=DEFAULT_ARBITER_CMD)
        status.add_argument("--jobs-json")
        status.add_argument("--inventory-json", required=True)
        status.add_argument("--timeout-sec", type=float, default=10.0)
        status.add_argument("--offline", action="store_true")
        add_output_flags(status)

        cancel = sub.add_parser("cancel", help="Disable a queued job without releasing leases")
        cancel.add_argument("--jobs-json", required=True)
        cancel.add_argument("--event-log-jsonl")
        cancel.add_argument("--job-id", required=True)
        cancel.add_argument("--reason", default="operator_cancelled")
        cancel.add_argument("--dry-run", action="store_true")
        add_output_flags(cancel)

        drain = sub.add_parser("drain", help="Mark a resource blocked in an inventory file")
        drain.add_argument("--inventory-json", required=True)
        drain.add_argument("--out-inventory-json")
        drain.add_argument("--resource", required=True)
        drain.add_argument("--reason", required=True)
        drain.add_argument("--dry-run", action="store_true")
        add_output_flags(drain)

        undrain = sub.add_parser("undrain", help="Mark a resource active in an inventory file")
        undrain.add_argument("--inventory-json", required=True)
        undrain.add_argument("--out-inventory-json")
        undrain.add_argument("--resource", required=True)
        undrain.add_argument("--reason", default="operator_undrain")
        undrain.add_argument("--dry-run", action="store_true")
        add_output_flags(undrain)

        audit = sub.add_parser("audit", help="Validate queue and inventory scheduler contracts")
        audit.add_argument("--jobs-json", required=True)
        audit.add_argument("--inventory-json", required=True)
        audit.add_argument("--worker-reconciliation-json", action="append", default=[])
        audit.add_argument("--worker-reconciliation-dir", action="append", default=[])
        add_output_flags(audit)

        return parser.parse_args(argv)

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
    parser.add_argument("--unknown-lease-quarantine-age-sec", type=float, default=900.0)
    parser.add_argument("--max-running-per-tenant", type=int, default=0)
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
    if args.unknown_lease_quarantine_age_sec < 0:
        parser.error("--unknown-lease-quarantine-age-sec must be >= 0")
    if args.max_running_per_tenant < 0:
        parser.error("--max-running-per-tenant must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if getattr(args, "control_command", None):
        try:
            if args.control_command == "submit":
                payload = cmd_submit(args)
            elif args.control_command == "status":
                payload = cmd_status(args)
            elif args.control_command == "cancel":
                payload = cmd_cancel(args)
            elif args.control_command == "drain":
                payload = update_inventory_status(args, drained=True)
            elif args.control_command == "undrain":
                payload = update_inventory_status(args, drained=False)
            elif args.control_command == "audit":
                payload = cmd_audit(args)
            else:
                raise RuntimeError(f"unhandled control command {args.control_command!r}")
        except Exception as exc:
            payload = {
                "schema": "tensorcore.cluster_command.result.v1",
                "ok": False,
                "checked_at_unix": time.time(),
                "errors": [{"error": str(exc)}],
            }
        if args.json or args.pretty_json:
            emit_json(payload, pretty=args.pretty_json)
        else:
            if "results" in payload:
                emit_text(payload)
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("ok") else 2
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
