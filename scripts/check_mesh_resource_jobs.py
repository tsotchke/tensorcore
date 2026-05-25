#!/usr/bin/env python3
"""Validate checked-in Tensorcore mesh scheduler jobs against inventory."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
DEFAULT_JOBS = ROOT / "configs" / "mesh_resource_jobs.json"
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
