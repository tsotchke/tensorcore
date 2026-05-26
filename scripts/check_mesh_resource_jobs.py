#!/usr/bin/env python3
"""Validate checked-in Tensorcore mesh scheduler jobs against inventory."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
import shlex
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
DEFAULT_JOBS = ROOT / "configs" / "mesh_resource_jobs.json"
COMMAND_FIELDS = (
    "probe_cmd",
    "completion_cmd",
    "admission_cmd",
    "post_start_probe_cmd",
    "preflight_cmd",
    "worker_identity_cmd",
    "start_cmd",
)
PRIVATE_COMMAND_PATTERNS = (
    "~/.tsotchke/bin",
    "/.tsotchke/bin",
    "/Users/",
    "/home/tyr/.local/bin/",
    "/.local/bin/",
)
HOST_LOCAL_SCRIPT_PREFIXES = (
    "/data/qllm/runs/",
    "/home/tyr/bytehole/qllm/runs/",
)
REPO_RELATIVE_PREFIXES = ("scripts/", "configs/")
WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT = "windows_cuda_scheduled_smoke"
WINDOWS_CUDA_SMOKE_SCRIPT = "scripts/start_windows_cuda_smoke.py"
WINDOWS_PERSISTENT_LAUNCH_SCRIPT = "scripts/check_windows_persistent_launch.py"
WINDOWS_CUDA_SMOKE_MAX_DURATION_SEC = 5
HOST_LOCAL_SERVICE_STARTS = (
    "systemctl --user start",
    "systemctl start",
)
SCHEDULER_CANDIDATES = [
    ROOT / "scripts" / "mesh_resource_scheduler.py",
    pathlib.Path(__file__).with_name("mesh-resource-scheduler"),
    pathlib.Path(__file__).with_name("mesh_resource_scheduler.py"),
]


def load_module(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--jobs-json", type=pathlib.Path, default=DEFAULT_JOBS)
    return parser.parse_args()


def raw_command(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value]
    if isinstance(value, str):
        return shlex.split(value)
    return []


def command_parts(value: Any) -> list[str]:
    parts = [str(value)]
    argv = raw_command(value)
    parts.extend(argv)
    for item in argv:
        if " " not in item and "&&" not in item and ";" not in item:
            continue
        try:
            parts.extend(shlex.split(item))
        except ValueError:
            pass
    return parts


def command_has_part(value: Any, expected: str) -> bool:
    return expected in command_parts(value)


def command_flag_value(value: Any, flag: str) -> str | None:
    parts = command_parts(value)
    for index, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[index + 1]
    return None


def command_uses_host_local_service_start(value: Any) -> bool:
    text = " ".join(command_parts(value))
    return any(pattern in text for pattern in HOST_LOCAL_SERVICE_STARTS)


def validate_checked_in_command(
    errors: list[str],
    owner: str,
    field: str,
    value: Any,
) -> None:
    parts = command_parts(value)
    if len(parts) <= 1:
        return
    for part in parts:
        if any(pattern in part for pattern in PRIVATE_COMMAND_PATTERNS):
            errors.append(
                f"{owner} {field} must use git-checkout commands, not private wrapper path {part!r}"
            )
        if part.endswith(".sh") and part.startswith(HOST_LOCAL_SCRIPT_PREFIXES):
            errors.append(
                f"{owner} {field} must launch repo-owned scripts, not host-local script {part!r}"
            )
        if part.startswith(REPO_RELATIVE_PREFIXES) and not (ROOT / part).exists():
            errors.append(f"{owner} {field} references missing repo path {part!r}")


def validate_job_policy(errors: list[str], job: dict[str, Any]) -> None:
    job_id = job["id"]
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    if job.get("preflight_cmd") and "--json" not in command_parts(job.get("preflight_cmd")):
        errors.append(f"job {job_id!r} preflight_cmd must emit JSON with --json")
    if command_has_part(job.get("preflight_cmd"), "scripts/check_mesh_git_access.py"):
        if not command_has_part(job.get("preflight_cmd"), "--resource"):
            errors.append(
                f"job {job_id!r} check_mesh_git_access.py preflight_cmd must pass --resource"
            )
    if job.get("desired_state") == "paused":
        if not str(metadata.get("scheduler_pause_reason") or "").strip():
            errors.append(f"paused job {job_id!r} requires metadata.scheduler_pause_reason")
        if job.get("start_cmd") and not job.get("preflight_cmd"):
            errors.append(f"paused launchable job {job_id!r} requires preflight_cmd")
    if job.get("desired_state") == "running" and command_uses_host_local_service_start(job.get("start_cmd")):
        errors.append(
            f"running job {job_id!r} must use a repo-owned starter, not a host-local systemd unit"
        )
    if metadata.get("scheduler_contract") == WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT:
        if not command_has_part(job.get("start_cmd"), WINDOWS_CUDA_SMOKE_SCRIPT):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                f"requires {WINDOWS_CUDA_SMOKE_SCRIPT} start_cmd"
            )
        if not command_has_part(job.get("preflight_cmd"), WINDOWS_PERSISTENT_LAUNCH_SCRIPT):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                f"requires {WINDOWS_PERSISTENT_LAUNCH_SCRIPT} preflight_cmd"
            )
        if not command_has_part(job.get("post_start_probe_cmd"), "scripts/check_windows_cuda_smoke_artifact.py"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "requires Windows CUDA smoke artifact post_start_probe_cmd"
            )
        if not command_has_part(job.get("probe_cmd"), "scripts/check_windows_cuda_smoke_artifact.py"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "requires Windows CUDA smoke artifact probe_cmd"
            )
        if not command_has_part(job.get("probe_cmd"), "--require-live"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "probe_cmd must require a live smoke artifact"
            )
        if command_has_part(job.get("probe_cmd"), "--require-live-or-complete"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "probe_cmd must not treat completed artifacts as live"
            )
        if not command_has_part(job.get("post_start_probe_cmd"), "--require-live-or-complete"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "post_start_probe_cmd must accept live or completed smoke artifacts"
            )
        if not command_has_part(job.get("completion_cmd"), "scripts/check_windows_cuda_smoke_artifact.py"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "requires Windows CUDA smoke artifact completion_cmd"
            )
        if not command_has_part(job.get("completion_cmd"), "--require-complete"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "completion_cmd must require completed smoke artifacts"
            )
        if command_has_part(job.get("start_cmd"), "--foreground"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "must not use --foreground in scheduler start_cmd"
            )
        if command_has_part(job.get("start_cmd"), "--recover-foreground-timeout"):
            errors.append(
                f"job {job_id!r} {WINDOWS_CUDA_SCHEDULED_SMOKE_CONTRACT} "
                "must not use --recover-foreground-timeout in scheduler start_cmd"
            )
        raw_duration = command_flag_value(job.get("start_cmd"), "--duration-sec")
        try:
            duration = int(raw_duration or "")
        except ValueError:
            duration = None
        if duration is None:
            errors.append(f"job {job_id!r} Windows CUDA smoke start_cmd requires --duration-sec")
        elif duration > WINDOWS_CUDA_SMOKE_MAX_DURATION_SEC:
            errors.append(
                f"job {job_id!r} Windows CUDA smoke duration must be <= "
                f"{WINDOWS_CUDA_SMOKE_MAX_DURATION_SEC}s"
            )


def validate_jobs(inventory_path: pathlib.Path, jobs_path: pathlib.Path) -> list[str]:
    scheduler_path = next((item for item in SCHEDULER_CANDIDATES if item.exists()), SCHEDULER_CANDIDATES[0])
    scheduler = load_module("mesh_resource_scheduler_for_jobs_check", scheduler_path)
    inventory = scheduler.load_inventory(str(inventory_path))
    jobs = scheduler.load_jobs(str(jobs_path), inventory=inventory)
    errors: list[str] = []
    try:
        scheduler.validate_jobs_against_inventory(jobs, inventory)
    except ValueError as exc:
        errors.append(str(exc))

    job_ids = [job["id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        errors.append("expanded jobs must have unique ids")

    scheduler_resources = {
        resource_id
        for resource_id, row in inventory.items()
        if row.get("control_plane") == "tensorcore_scheduler"
        and row.get("status", "active") == "active"
    }
    job_resources = {job["resource"] for job in jobs}
    missing = sorted(scheduler_resources - job_resources)
    if missing:
        errors.append(f"tensorcore_scheduler resources without jobs: {', '.join(missing)}")

    for job in jobs:
        for field in COMMAND_FIELDS:
            if field in job:
                validate_checked_in_command(errors, f"job {job['id']!r}", field, job[field])
        validate_job_policy(errors, job)
        if job["desired_state"] == "running" and job["resource_class"] == "cuda_exclusive":
            for field in ("admission_cmd", "post_start_probe_cmd", "worker_identity_cmd"):
                if not job.get(field):
                    errors.append(f"running CUDA job {job['id']!r} requires {field}")
        if not str(job.get("tenant") or "").strip():
            errors.append(f"job {job['id']!r} must declare tenant")

    return errors


def main() -> int:
    args = parse_args()
    errors = validate_jobs(args.inventory_json, args.jobs_json)
    if errors:
        for error in errors:
            print(f"mesh resource jobs error: {error}")
        return 1
    print(f"mesh resource jobs OK: {args.jobs_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
