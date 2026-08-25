#!/usr/bin/env python3
"""Focused tests for scheduler admission through a signed topology snapshot."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import tempfile
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "mesh_resource_scheduler.py"
AUTHORITY = ROOT / "scripts" / "topology_authority.py"
SIGNING_KEY = "scheduler-binding-selftest"


def load_module(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "alpha:cuda0",
            "node": "alpha",
            "backend": "cuda",
            "class": "cuda-test",
            "capacity": 1,
            "status": "active",
            "control_plane": "tensorcore_scheduler",
            "general_queue_eligible": True,
        },
        {
            "id": "beta:retired",
            "node": "beta",
            "backend": "service",
            "class": "retired",
            "capacity": 1,
            "status": "blocked",
            "control_plane": "blocked",
            "general_queue_eligible": False,
            "blocked_reason": "retired selftest resource",
        },
    ]


def authority_resources(*, alpha_admission: str = "eligible") -> list[dict[str, Any]]:
    return [
        {
            "id": "alpha:cuda0",
            "authority_node_id": "alpha",
            "backend": "cuda",
            "class": "cuda-test",
            "capacity": 1,
            "scheduler_status": "active",
            "authority_admission": alpha_admission,
            "authority_reason": "scheduler_status",
        },
        {
            "id": "beta:retired",
            "authority_node_id": "beta",
            "backend": "service",
            "class": "retired",
            "capacity": 1,
            "scheduler_status": "blocked",
            "authority_admission": "blocked",
            "authority_reason": "scheduler_status",
        },
    ]


def write_inventory(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "inventory.json"
    path.write_text(
        json.dumps({"schema": "tensorcore.mesh_resources.v1", "resources": inventory_rows()}),
        encoding="utf-8",
    )
    return path


def write_snapshot(
    authority: ModuleType,
    directory: pathlib.Path,
    *,
    generated_at: datetime | None = None,
    alpha_admission: str = "eligible",
) -> pathlib.Path:
    snapshot = {
        "schema": "tensorcore.topology_snapshot.v1",
        "generated_at": authority.format_timestamp(generated_at or datetime.now(timezone.utc)),
        "policy_sha256": "selftest-policy",
        "sources": [],
        "nodes": [],
        "resources": authority_resources(alpha_admission=alpha_admission),
        "drift": {"clean": True, "error_count": 0, "warning_count": 0, "issues": []},
    }
    authority.attach_integrity(snapshot, SIGNING_KEY)
    path = directory / "topology.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def gate_args(snapshot: pathlib.Path | None, **overrides: Any) -> argparse.Namespace:
    values = {
        "topology_snapshot": str(snapshot) if snapshot else None,
        "require_topology_authority": True,
        "require_topology_signature": True,
        "topology_max_age_sec": 300.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def with_signing_key(function: Any) -> None:
    previous = os.environ.get("TC_TOPOLOGY_SIGNING_KEY")
    os.environ["TC_TOPOLOGY_SIGNING_KEY"] = SIGNING_KEY
    try:
        function()
    finally:
        if previous is None:
            os.environ.pop("TC_TOPOLOGY_SIGNING_KEY", None)
        else:
            os.environ["TC_TOPOLOGY_SIGNING_KEY"] = previous


def test_signed_snapshot_binds_inventory_and_job_metadata(
    scheduler: ModuleType,
    authority: ModuleType,
) -> None:
    with tempfile.TemporaryDirectory(prefix="tensorcore-scheduler-topology-") as raw:
        directory = pathlib.Path(raw)
        inventory = scheduler.load_inventory(str(write_inventory(directory)))
        snapshot = write_snapshot(authority, directory)
        gate = scheduler.topology_authority_gate(gate_args(snapshot), inventory)
        assert gate["ok"] is True, gate
        assert gate["integrity"] == "signature_valid"
        assert inventory["alpha:cuda0"]["topology_authority_admission"] == "eligible"
        normalized = scheduler.normalize_job(
            {
                "id": "alpha-paused",
                "resource": "alpha:cuda0",
                "owner": "tensorcore:selftest",
                "desired_state": "paused",
                "resource_class": "cuda_exclusive",
                "admission_cmd": ["admit"],
                "post_start_probe_cmd": ["post"],
                "worker_identity_cmd": ["identity"],
            },
            inventory=inventory,
        )
        assert normalized["metadata"]["topology_snapshot_sha256"] == gate["snapshot_sha256"]
        assert normalized["metadata"]["topology_authority_node_id"] == "alpha"
        assert normalized["metadata"]["topology_authority_admission"] == "eligible"


def test_missing_snapshot_is_rejected(scheduler: ModuleType, authority: ModuleType) -> None:
    del authority
    gate = scheduler.topology_authority_gate(gate_args(None), {})
    assert gate == {
        "ok": False,
        "required": True,
        "configured": False,
        "reason": "snapshot_required",
    }


def test_dry_run_may_omit_snapshot(scheduler: ModuleType, authority: ModuleType) -> None:
    del authority
    gate = scheduler.topology_authority_gate(gate_args(None, dry_run=True), {})
    assert gate == {
        "ok": True,
        "required": False,
        "configured": False,
        "reason": "not_configured",
    }


def test_quarantined_active_resource_is_rejected(
    scheduler: ModuleType,
    authority: ModuleType,
) -> None:
    with tempfile.TemporaryDirectory(prefix="tensorcore-scheduler-topology-") as raw:
        directory = pathlib.Path(raw)
        inventory = scheduler.load_inventory(str(write_inventory(directory)))
        snapshot = write_snapshot(authority, directory, alpha_admission="quarantined")
        gate = scheduler.topology_authority_gate(gate_args(snapshot), inventory)
        assert gate["ok"] is False
        assert gate["reason"] == "resource_binding_failed"
        assert any("not admitted" in error for error in gate["errors"])


def test_stale_snapshot_is_rejected(scheduler: ModuleType, authority: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="tensorcore-scheduler-topology-") as raw:
        directory = pathlib.Path(raw)
        inventory = scheduler.load_inventory(str(write_inventory(directory)))
        snapshot = write_snapshot(
            authority,
            directory,
            generated_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        gate = scheduler.topology_authority_gate(gate_args(snapshot, topology_max_age_sec=60.0), inventory)
        assert gate["ok"] is False
        assert gate["reason"] == "snapshot_stale"


def test_future_snapshot_is_rejected(scheduler: ModuleType, authority: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="tensorcore-scheduler-topology-") as raw:
        directory = pathlib.Path(raw)
        inventory = scheduler.load_inventory(str(write_inventory(directory)))
        snapshot = write_snapshot(
            authority,
            directory,
            generated_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        gate = scheduler.topology_authority_gate(gate_args(snapshot), inventory)
        assert gate["ok"] is False
        assert gate["reason"] == "snapshot_from_future"


def test_tampered_snapshot_is_rejected(scheduler: ModuleType, authority: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="tensorcore-scheduler-topology-") as raw:
        directory = pathlib.Path(raw)
        inventory = scheduler.load_inventory(str(write_inventory(directory)))
        snapshot = write_snapshot(authority, directory)
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload["resources"][0]["capacity"] = 999
        snapshot.write_text(json.dumps(payload), encoding="utf-8")
        gate = scheduler.topology_authority_gate(gate_args(snapshot), inventory)
        assert gate["ok"] is False
        assert gate["reason"] == "digest_mismatch"


def test_cli_defaults_to_required_signed_authority(
    scheduler: ModuleType,
    authority: ModuleType,
) -> None:
    del authority
    args = scheduler.parse_args(["--jobs-json", "jobs.json", "--inventory-json", "inventory.json"])
    assert args.require_topology_authority is True
    assert args.require_topology_signature is True
    submit = scheduler.parse_args(
        [
            "submit",
            "--job-json",
            "job.json",
            "--jobs-json",
            "jobs.json",
            "--inventory-json",
            "inventory.json",
        ]
    )
    assert submit.require_topology_authority is True
    assert submit.require_topology_signature is True
    bypass = scheduler.parse_args(
        [
            "--jobs-json",
            "jobs.json",
            "--inventory-json",
            "inventory.json",
            "--allow-unreconciled-topology",
            "--allow-unsigned-topology",
        ]
    )
    assert bypass.require_topology_authority is False
    assert bypass.require_topology_signature is False


def main() -> int:
    scheduler = load_module("mesh_resource_scheduler_topology_selftest", SCHEDULER)
    authority = load_module("topology_authority_scheduler_selftest", AUTHORITY)
    tests = [
        test_signed_snapshot_binds_inventory_and_job_metadata,
        test_missing_snapshot_is_rejected,
        test_dry_run_may_omit_snapshot,
        test_quarantined_active_resource_is_rejected,
        test_stale_snapshot_is_rejected,
        test_future_snapshot_is_rejected,
        test_tampered_snapshot_is_rejected,
        test_cli_defaults_to_required_signed_authority,
    ]

    def run() -> None:
        for test in tests:
            test(scheduler, authority)
            print(f"PASS {test.__name__}")

    with_signing_key(run)
    print(f"topology scheduler binding selftest OK: {len(tests)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
