#!/usr/bin/env python3
"""Validate scheduler-owned Windows CUDA smoke evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any


SCHEMA = "tensorcore.windows_cuda_scheduled_smoke.evidence.v1"
SMOKE_SCHEMA = "tensorcore.windows_cuda_smoke.v1"
WORKER_IDENTITY_SCHEMA = "tensorcore.mesh_worker_identity.v1"
BOOLEAN_FIELDS = (
    "driver_visible",
    "toolchain_found",
    "wddm_admission_ok",
    "build_smoke_passed",
    "runtime_smoke_passed",
    "scheduler_lease_held",
    "worker_identity_recorded",
)
OBJECT_FIELDS = (
    "admission",
    "start",
    "post_start",
    "completion",
    "smoke_artifact",
    "worker_identity",
    "worker_identity_heartbeat",
)


def fail(message: str) -> int:
    print(f"Windows scheduled CUDA smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Windows scheduled CUDA smoke evidence invalid: could not read JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print("Windows scheduled CUDA smoke evidence invalid: evidence must be a JSON object", file=sys.stderr)
        return None
    return payload


def require_bool(payload: dict[str, Any], key: str, enabled: bool, message: str) -> str | None:
    if enabled and payload.get(key) is not True:
        return message
    return None


def validate_structure(payload: dict[str, Any]) -> str | None:
    if payload.get("phase") not in {"launched", "completed"}:
        return "phase must be launched or completed"
    for key in BOOLEAN_FIELDS:
        if not isinstance(payload.get(key), bool):
            return f"{key} must be boolean"
    for key in OBJECT_FIELDS:
        if key in payload and payload[key] is not None and not isinstance(payload[key], dict):
            return f"{key} must be an object"
    artifact = payload.get("smoke_artifact") if isinstance(payload.get("smoke_artifact"), dict) else {}
    if payload.get("build_smoke_passed") is True or payload.get("runtime_smoke_passed") is True:
        if not artifact:
            return "smoke_artifact is required when smoke flags are true"
        if artifact.get("schema") != SMOKE_SCHEMA:
            return f"smoke_artifact schema must be {SMOKE_SCHEMA!r}"
        if artifact.get("resource") != payload.get("resource"):
            return "smoke_artifact resource must match evidence resource"
    if payload.get("runtime_smoke_passed") is True and artifact.get("runtime_ok") is not True:
        return "smoke_artifact runtime_ok must be true when runtime_smoke_passed=true"
    if payload.get("build_smoke_passed") is True and artifact.get("build_ok") is not True:
        return "smoke_artifact build_ok must be true when build_smoke_passed=true"
    identity = payload.get("worker_identity") if isinstance(payload.get("worker_identity"), dict) else {}
    if payload.get("worker_identity_recorded") is True:
        if not identity:
            return "worker_identity must be an object when recorded"
        if identity.get("schema") != WORKER_IDENTITY_SCHEMA:
            return f"worker_identity schema must be {WORKER_IDENTITY_SCHEMA!r}"
        if identity.get("resource") != payload.get("resource"):
            return "worker_identity resource must match evidence resource"
        if identity.get("ok") is not True:
            return "worker_identity ok must be true when recorded"
    if payload.get("scheduler_lease_held") is True and not payload.get("lease_id"):
        return "lease_id is required when scheduler_lease_held=true"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--max-age-sec", type=float, default=0.0)
    parser.add_argument("--require-driver-visible", action="store_true")
    parser.add_argument("--require-toolchain", action="store_true")
    parser.add_argument("--require-wddm-admission", action="store_true")
    parser.add_argument("--require-build-smoke", action="store_true")
    parser.add_argument("--require-runtime-smoke", action="store_true")
    parser.add_argument("--require-scheduler-lease", action="store_true")
    parser.add_argument("--require-worker-identity", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    payload = load_json(args.path)
    if payload is None:
        return 1
    if payload.get("schema") != SCHEMA:
        return fail(f"schema must be {SCHEMA!r}")
    if payload.get("schema_version") != 1:
        return fail("schema_version must be 1")
    if not payload.get("resource"):
        return fail("resource is required")
    if not payload.get("job"):
        return fail("job is required")
    structure_error = validate_structure(payload)
    if structure_error:
        return fail(structure_error)

    if args.max_age_sec > 0:
        try:
            age = time.time() - float(payload.get("checked_at_unix") or 0.0)
        except (TypeError, ValueError):
            return fail("checked_at_unix must be numeric when --max-age-sec is used")
        if age > args.max_age_sec:
            return fail(f"evidence is stale: age_sec={age:.3f} max_age_sec={args.max_age_sec}")

    require_all = args.require_complete
    checks = [
        ("driver_visible", args.require_driver_visible or require_all, "driver must be visible"),
        ("toolchain_found", args.require_toolchain or require_all, "nvcc/toolchain must be found"),
        ("wddm_admission_ok", args.require_wddm_admission or require_all, "WDDM admission must be OK"),
        ("build_smoke_passed", args.require_build_smoke or require_all, "CUDA build smoke must pass"),
        ("runtime_smoke_passed", args.require_runtime_smoke or require_all, "CUDA runtime smoke must pass"),
        ("scheduler_lease_held", args.require_scheduler_lease or require_all, "scheduler lease must be held"),
        ("worker_identity_recorded", args.require_worker_identity or require_all, "worker identity must be recorded"),
    ]
    for key, enabled, message in checks:
        error = require_bool(payload, key, enabled, message)
        if error:
            return fail(error)

    if args.require_complete and payload.get("phase") != "completed":
        return fail("phase must be completed when --require-complete is used")
    print(
        "Windows scheduled CUDA smoke evidence OK: "
        f"resource={payload.get('resource')} job={payload.get('job')} phase={payload.get('phase')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
