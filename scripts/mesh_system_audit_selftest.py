#!/usr/bin/env python3
"""Selftests for scripts/mesh_system_audit.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import time
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_CANDIDATES = [
    ROOT / "scripts" / "mesh_system_audit.py",
    pathlib.Path(__file__).with_name("mesh-system-audit"),
    pathlib.Path(__file__).with_name("mesh_system_audit.py"),
]


def load_module() -> ModuleType:
    path = next((item for item in SCRIPT_CANDIDATES if item.exists()), SCRIPT_CANDIDATES[0])
    loader = importlib.machinery.SourceFileLoader("mesh_system_audit_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory() -> dict[str, dict[str, Any]]:
    return {
        "cosbox:cuda3090": {
            "id": "cosbox:cuda3090",
            "node": "cosbox",
            "backend": "cuda",
            "class": "cuda-training",
            "status": "active",
            "general_queue_eligible": True,
        },
        "jack-blupc:cuda3060": {
            "id": "jack-blupc:cuda3060",
            "node": "jack-blupc",
            "backend": "cuda",
            "class": "cuda-small-auxiliary",
            "status": "blocked",
            "blocked_reason": "not ready",
            "general_queue_eligible": False,
        },
    }


def jobs() -> list[dict[str, Any]]:
    return [
        {
            "id": "georefine",
            "resource": "cosbox:cuda3090",
            "tenant": "georefine",
        }
    ]


def lease(resource: str = "cosbox:cuda3090", *, pid: int = 1234) -> dict[str, Any]:
    return {
        "id": f"lease-{resource}",
        "resource": resource,
        "owner": "georefine:m2",
        "metadata": {
            "surface": "tensorcore_mesh_scheduler",
            "tenant": "georefine",
            "worker_identity_pending": False,
            "worker_identity": {
                "worker_host": "cosbox",
                "cuda_pids": [pid],
                "matched_cuda_pids": [pid],
            },
        },
    }


def status(rows: dict[str, dict[str, Any]], leases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "tsotchke.resource_leases.v1",
        "ok": True,
        "resources": [
            {
                "id": resource_id,
                "capacity": 1,
                "active": sum(1 for item in leases if item.get("resource") == resource_id),
                "leases": [item for item in leases if item.get("resource") == resource_id],
            }
            for resource_id in rows
        ],
        "leases": leases,
    }


def scheduler_state() -> dict[str, Any]:
    return {
        "schema": "tensorcore.mesh_resource_scheduler.result.v1",
        "ok": True,
        "checked_at_unix": time.time(),
        "errors": [],
    }


def test_ok_with_scheduled_cuda_pid() -> None:
    mod = load_module()
    payload = mod.audit(
        inventory=inventory(),
        jobs=jobs(),
        scheduler_state=scheduler_state(),
        arbiter_status=status(inventory(), [lease(pid=1234)]),
        max_scheduler_age_sec=120,
        cuda_apps_by_resource={
            "cosbox:cuda3090": {
                "ok": True,
                "apps": [{"pid": 1234, "process_name": "python", "used_memory_mib": 1000}],
            }
        },
    )
    assert payload["ok"] is True
    assert payload["summary"]["errors"] == 0


def test_blocked_resource_with_lease_fails() -> None:
    mod = load_module()
    payload = mod.audit(
        inventory=inventory(),
        jobs=jobs(),
        scheduler_state=scheduler_state(),
        arbiter_status=status(inventory(), [lease("jack-blupc:cuda3060", pid=44)]),
        max_scheduler_age_sec=120,
    )
    assert payload["ok"] is False
    assert payload["errors"][0]["error"] == "blocked_resource_has_active_leases"


def test_unmanaged_cuda_pid_fails() -> None:
    mod = load_module()
    payload = mod.audit(
        inventory=inventory(),
        jobs=jobs(),
        scheduler_state=scheduler_state(),
        arbiter_status=status(inventory(), [lease(pid=1234)]),
        max_scheduler_age_sec=120,
        cuda_apps_by_resource={
            "cosbox:cuda3090": {
                "ok": True,
                "apps": [{"pid": 9999, "process_name": "python", "used_memory_mib": 1000}],
            }
        },
    )
    assert payload["ok"] is False
    assert payload["errors"][0]["error"] == "unmanaged_cuda_processes"


def main() -> int:
    test_ok_with_scheduled_cuda_pid()
    test_blocked_resource_with_lease_fails()
    test_unmanaged_cuda_pid_fails()
    print("mesh system audit selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
