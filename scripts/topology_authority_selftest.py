#!/usr/bin/env python3
"""Deterministic positive and fail-closed tests for topology_authority.py."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "scripts" / "topology_authority.py"
POLICY = ROOT / "configs" / "topology_authority.json"
AS_OF = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def load_authority() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("topology_authority_under_test", str(AUTHORITY))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declaration_payload() -> dict[str, Any]:
    return {
        "mesh": {"name": "synthetic"},
        "nodes": [
            {
                "name": "alpha",
                "ssh_alias": "alpha-ssh",
                "tailscale_name": "alpha",
                "os": "linux",
                "arch": "x86_64",
                "roles": ["external-compute"],
                "cloud": {
                    "provider": "gcp",
                    "project": "private-project",
                    "zone": "test-zone-1",
                    "instance": "alpha-cloud",
                },
                "capabilities": {
                    "compute": {
                        "class": "synthetic-cuda",
                        "accelerators": ["cpu", "cuda"],
                        "gpu_count": 1,
                        "gpu_memory_mb": 24576,
                    }
                },
            },
            {
                "name": "beta",
                "tailscale_name": "beta",
                "lifecycle": "retired",
                "os": "macos",
                "arch": "arm64",
                "roles": ["retired"],
                "capabilities": {"compute": {"accelerators": ["cpu", "metal"]}},
            },
        ],
    }


def tailscale_payload(profile_id: str, *, observed_at: str = "2026-07-16T11:59:00Z") -> dict[str, Any]:
    if profile_id == "mesh":
        status = {
            "BackendState": "Running",
            "Self": {"HostName": "alpha", "DNSName": "alpha.mesh.invalid.", "Online": True},
            "Peer": {
                "peer-beta": {"HostName": "beta", "DNSName": "beta.mesh.invalid.", "Online": False}
            },
        }
    else:
        status = {
            "BackendState": "Running",
            "Self": {"HostName": "personal-device", "Online": True},
            "Peer": {},
        }
    return {
        "schema": "tensorcore.tailscale_profile_observation.v1",
        "profile_id": profile_id,
        "observed_at": observed_at,
        "status": status,
    }


def cloud_payload(*, observed_at: str = "2026-07-16T11:58:00Z") -> dict[str, Any]:
    return {
        "schema": "tensorcore.cloud_inventory_observation.v1",
        "inventory_id": "gcp",
        "provider": "gcp",
        "observed_at": observed_at,
        "instances": [
            {
                "name": "alpha-cloud",
                "status": "RUNNING",
                "guestAccelerators": [
                    {"acceleratorType": "zones/test-zone-1/acceleratorTypes/nvidia-test-gpu"}
                ],
            }
        ],
    }


def scheduler_payload(*, node: str = "alpha", backend: str = "cuda") -> dict[str, Any]:
    return {
        "schema": "tensorcore.mesh_resources.v1",
        "resources": [
            {
                "id": "alpha:cuda0",
                "node": node,
                "backend": backend,
                "class": "cuda-test",
                "capacity": 1,
                "status": "active",
                "control_plane": "tensorcore_scheduler",
                "general_queue_eligible": True,
            },
            {
                "id": "beta:retired-service",
                "node": "beta",
                "backend": "service",
                "class": "retired",
                "capacity": 1,
                "status": "blocked",
                "control_plane": "blocked",
                "general_queue_eligible": False,
            },
        ],
    }


def normalize_inputs(
    mod: ModuleType,
    policy: dict[str, Any],
    *,
    declarations_payload: dict[str, Any] | None = None,
    mesh_payload: dict[str, Any] | None = None,
    personal_payload: dict[str, Any] | None = None,
    cloud_inventory_payload: dict[str, Any] | None = None,
    scheduler_inventory_payload: dict[str, Any] | None = None,
    include_personal: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    declarations, declaration_source = mod.normalize_declarations(
        declarations_payload or declaration_payload(), "computer_mesh", policy
    )
    observations: list[dict[str, Any]] = []
    sources = [declaration_source]
    rows, source = mod.normalize_tailscale_observation(mesh_payload or tailscale_payload("mesh"), "mesh")
    observations.extend(rows)
    sources.append(source)
    if include_personal:
        rows, source = mod.normalize_tailscale_observation(
            personal_payload or tailscale_payload("personal"), "personal"
        )
        observations.extend(rows)
        sources.append(source)
    rows, source = mod.normalize_cloud_observation(cloud_inventory_payload or cloud_payload(), "gcp")
    observations.extend(rows)
    sources.append(source)
    resources, scheduler_source = mod.normalize_scheduler_inventory(
        scheduler_inventory_payload or scheduler_payload(), "tensorcore", policy
    )
    sources.append(scheduler_source)
    return declarations, observations, resources, sources


def reconcile_inputs(
    mod: ModuleType,
    policy: dict[str, Any],
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    declarations, observations, resources, sources = normalize_inputs(mod, policy, **kwargs)
    return mod.reconcile(
        policy,
        declarations,
        observations,
        resources,
        sources,
        as_of=AS_OF,
        signing_key="selftest-signing-key",
    )


def test_clean_reconciliation(mod: ModuleType, policy: dict[str, Any]) -> None:
    private, public = reconcile_inputs(mod, policy)
    assert private["drift"]["clean"] is True, private["drift"]
    assert private["drift"]["warning_count"] == 1
    assert any(item["code"] == "unmanaged_observation" for item in private["drift"]["issues"])
    resources = {row["id"]: row for row in private["resources"]}
    assert resources["alpha:cuda0"]["authority_admission"] == "eligible"
    assert resources["beta:retired-service"]["authority_admission"] == "blocked"
    nodes = {row["id"]: row for row in private["nodes"]}
    assert nodes["alpha"]["state"] == {
        "observation": "fresh",
        "connectivity": "online",
        "admission": "eligible",
    }
    assert nodes["beta"]["state"]["admission"] == "blocked"
    assert mod.verify_snapshot(private, "selftest-signing-key", require_signature=True) == (
        True,
        "signature_valid",
    )
    assert mod.verify_snapshot(public, "selftest-signing-key", require_signature=True) == (
        True,
        "signature_valid",
    )
    assert mod.forbidden_public_paths(public, policy) == []
    assert all("aliases" not in row for row in public["nodes"])
    assert "input_sha256" not in public["source_coverage"][0]


def test_unknown_scheduler_node_fails_closed(mod: ModuleType, policy: dict[str, Any]) -> None:
    private, _ = reconcile_inputs(mod, policy, scheduler_inventory_payload=scheduler_payload(node="ghost"))
    assert private["drift"]["clean"] is False
    resource = next(row for row in private["resources"] if row["id"] == "alpha:cuda0")
    assert resource["authority_admission"] == "quarantined"
    assert any(item["code"] == "scheduler_resource_unknown_node" for item in private["drift"]["issues"])


def test_stale_observations_fail_closed(mod: ModuleType, policy: dict[str, Any]) -> None:
    stale = "2026-07-14T00:00:00Z"
    private, _ = reconcile_inputs(
        mod,
        policy,
        mesh_payload=tailscale_payload("mesh", observed_at=stale),
        personal_payload=tailscale_payload("personal", observed_at=stale),
        cloud_inventory_payload=cloud_payload(observed_at=stale),
    )
    resource = next(row for row in private["resources"] if row["id"] == "alpha:cuda0")
    assert private["drift"]["clean"] is False
    assert resource["authority_admission"] == "quarantined"
    assert resource["authority_reason"] == "node_stale_unknown"


def test_future_observation_is_blocking(mod: ModuleType, policy: dict[str, Any]) -> None:
    future = "2026-07-16T13:00:00Z"
    private, _ = reconcile_inputs(
        mod,
        policy,
        mesh_payload=tailscale_payload("mesh", observed_at=future),
    )
    assert private["drift"]["clean"] is False
    assert any(item["code"] == "observation_from_future" for item in private["drift"]["issues"])


def test_alias_conflict_fails_closed(mod: ModuleType, policy: dict[str, Any]) -> None:
    payload = declaration_payload()
    payload["nodes"].append(
        {
            "name": "gamma",
            "ssh_alias": "alpha",
            "tailscale_name": "gamma",
            "capabilities": {"compute": {"accelerators": ["cpu", "cuda"]}},
        }
    )
    private, _ = reconcile_inputs(mod, policy, declarations_payload=payload)
    assert private["drift"]["clean"] is False
    assert any(item["code"] == "declared_alias_conflict" for item in private["drift"]["issues"])
    resource = next(row for row in private["resources"] if row["id"] == "alpha:cuda0")
    assert resource["authority_admission"] == "quarantined"


def test_backend_conflict_fails_closed(mod: ModuleType, policy: dict[str, Any]) -> None:
    private, _ = reconcile_inputs(
        mod,
        policy,
        scheduler_inventory_payload=scheduler_payload(backend="metal"),
    )
    resource = next(row for row in private["resources"] if row["id"] == "alpha:cuda0")
    assert resource["authority_admission"] == "quarantined"
    assert any(
        item["code"] == "scheduler_backend_capability_mismatch" for item in private["drift"]["issues"]
    )


def test_missing_profile_is_blocking(mod: ModuleType, policy: dict[str, Any]) -> None:
    private, _ = reconcile_inputs(mod, policy, include_personal=False)
    assert private["drift"]["clean"] is False
    assert any(
        item["code"] == "missing_required_source" and item["source_id"] == "personal"
        for item in private["drift"]["issues"]
    )


def test_integrity_detects_tampering(mod: ModuleType, policy: dict[str, Any]) -> None:
    private, _ = reconcile_inputs(mod, policy)
    tampered = copy.deepcopy(private)
    tampered["resources"][0]["capacity"] = 999
    assert mod.verify_snapshot(tampered, "selftest-signing-key", require_signature=True) == (
        False,
        "digest_mismatch",
    )


def test_cli_round_trip(mod: ModuleType, policy: dict[str, Any]) -> None:
    del mod, policy
    with tempfile.TemporaryDirectory(prefix="tensorcore-topology-") as raw_directory:
        directory = pathlib.Path(raw_directory)
        inputs = {
            "nodes.json": declaration_payload(),
            "mesh.json": tailscale_payload("mesh"),
            "personal.json": tailscale_payload("personal"),
            "cloud.json": cloud_payload(),
            "scheduler.json": scheduler_payload(),
        }
        for name, payload in inputs.items():
            (directory / name).write_text(json.dumps(payload), encoding="utf-8")
        private_path = directory / "private.json"
        public_path = directory / "public.json"
        trace_path = directory / "trace.jsonl"
        key_path = directory / "topology-signing.key"
        key_path.write_text("cli-selftest-key\n", encoding="utf-8")
        key_path.chmod(0o600)
        env = dict(os.environ)
        env.pop("TC_TOPOLOGY_SIGNING_KEY", None)
        env["TC_TOPOLOGY_SIGNING_KEY_FILE"] = str(key_path)
        proc = subprocess.run(
            [
                sys.executable,
                str(AUTHORITY),
                "reconcile",
                "--policy",
                str(POLICY),
                "--declarations",
                f"computer_mesh={directory / 'nodes.json'}",
                "--tailscale-profile",
                f"mesh={directory / 'mesh.json'}",
                "--tailscale-profile",
                f"personal={directory / 'personal.json'}",
                "--cloud-inventory",
                f"gcp={directory / 'cloud.json'}",
                "--scheduler-inventory",
                f"tensorcore={directory / 'scheduler.json'}",
                "--as-of",
                "2026-07-16T12:00:00Z",
                "--private-output",
                str(private_path),
                "--public-output",
                str(public_path),
                "--icc-trace",
                str(trace_path),
                "--require-clean",
                "--require-signature",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        summary = json.loads(proc.stdout)
        assert summary["ok"] is True and summary["signed"] is True
        assert private_path.exists() and public_path.exists() and trace_path.exists()
        verify = subprocess.run(
            [
                sys.executable,
                str(AUTHORITY),
                "verify",
                str(private_path),
                "--policy",
                str(POLICY),
                "--require-signature",
                "--json",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        assert verify.returncode == 0, (verify.stdout, verify.stderr)
        assert json.loads(verify.stdout)["reason"] == "signature_valid"
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert {event["name"] for event in events} == {
            "canonical_schema",
            "source_coverage",
            "identity_reconciliation",
            "drift_fail_closed",
            "scheduler_binding",
            "signed_snapshot",
            "public_redaction",
        }
        assert all(event["value"] == "PASS" for event in events)


def main() -> int:
    mod = load_authority()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    tests = [
        test_clean_reconciliation,
        test_unknown_scheduler_node_fails_closed,
        test_stale_observations_fail_closed,
        test_future_observation_is_blocking,
        test_alias_conflict_fails_closed,
        test_backend_conflict_fails_closed,
        test_missing_profile_is_blocking,
        test_integrity_detects_tampering,
        test_cli_round_trip,
    ]
    for test in tests:
        test(mod, policy)
        print(f"PASS {test.__name__}")
    print(f"topology authority selftest OK: {len(tests)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
