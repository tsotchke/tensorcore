#!/usr/bin/env python3
"""Selftests for mesh_gpu_reconciliation_audit.py."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import pathlib
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    path = ROOT / "scripts" / "mesh_gpu_reconciliation_audit.py"
    loader = importlib.machinery.SourceFileLoader("mesh_gpu_reconciliation_audit_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def inventory() -> dict:
    return {
        "schema": "tensorcore.mesh_resources.v1",
        "resources": [
            {
                "id": "cosbox:cuda3090",
                "node": "cosbox",
                "backend": "cuda",
                "class": "cuda-training",
                "capacity": 1,
                "status": "active",
                "control_plane": "tensorcore_scheduler",
                "gpu_reconciliation": {
                    "enabled": True,
                    "poll_host": "cosbox",
                    "allow_process_regex": ["steamwebhelper$"],
                    "allowed_process_max_memory_mib": 64,
                },
            }
        ],
    }


def jobs() -> dict:
    return {
        "schema": "tensorcore.mesh_resource_jobs.v1",
        "jobs": [
            {
                "id": "qllm-phase1",
                "owner": "qllm:phase1",
                "tenant": "qllm",
                "resource": "cosbox:cuda3090",
                "resource_class": "cuda_exclusive",
                "priority": 10,
                "desired_state": "paused",
                "ttl_sec": 60,
                "probe_cmd": ["probe", "qllm-phase1"],
                "admission_cmd": ["admit", "qllm-phase1"],
                "post_start_probe_cmd": ["post", "qllm-phase1"],
                "worker_identity_cmd": ["identity", "qllm-phase1"],
                "metadata": {"scheduler_pause_reason": "selftest"},
            }
        ],
    }


def snapshot(process_name: str, used_memory_mib: int) -> dict:
    return {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "reason": "ok",
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "gpus": [],
        "cuda_apps": [
            {
                "pid": 1234,
                "process_name": process_name,
                "used_memory_mib": used_memory_mib,
            }
        ],
        "cuda_pids": [1234],
        "compute_app_count": 1,
    }


def args_for(root: pathlib.Path) -> argparse.Namespace:
    return argparse.Namespace(
        inventory_json=root / "inventory.json",
        jobs_json=root / "jobs.json",
        reports_dir=root / "reports",
        sweep_json=root / "sweep.json",
        audit_json=root / "audit.json",
        arbiter_status_json=root / "status.json",
        arbiter_cmd="arbiter",
        arbiter_timeout_sec=1.0,
        snapshot_json_dir=root / "snapshots",
        snapshot_timeout_sec=1.0,
        resource=[],
        include_blocked=False,
        offline=True,
        clean_reports_dir=True,
    )


def prepare_root(root: pathlib.Path, snap: dict) -> None:
    write_json(root / "inventory.json", inventory())
    write_json(root / "jobs.json", jobs())
    write_json(root / "status.json", {"leases": []})
    write_json(root / "snapshots" / "cosbox_cuda3090.snapshot.json", snap)
    stale = root / "reports" / "stale.reconciliation.json"
    write_json(stale, {"schema": "tensorcore.mesh_worker_gpu_reconciliation.v1", "ok": False})


def test_audit_passes_allowed_desktop_cuda_and_cleans_reports() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        prepare_root(root, snapshot("/home/cos/snap/steam/common/.local/share/Steam/steamwebhelper", 9))
        payload = mod.run_once(args_for(root))
        reports = sorted((root / "reports").glob("*.reconciliation.json"))
        audit_payload = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert audit_payload["ok"] is True
    assert [path.name for path in reports] == ["cosbox_cuda3090.reconciliation.json"]


def test_audit_fails_unmanaged_unleased_cuda() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        prepare_root(root, snapshot("python train.py", 8192))
        payload = mod.run_once(args_for(root))
    assert payload["ok"] is False
    assert payload["sweep"]["reports"][0]["reason"] == "stale_unknown_unleased_cuda"
    assert payload["audit"]["ok"] is False


def main() -> int:
    test_audit_passes_allowed_desktop_cuda_and_cleans_reports()
    test_audit_fails_unmanaged_unleased_cuda()
    print("mesh GPU reconciliation audit selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
