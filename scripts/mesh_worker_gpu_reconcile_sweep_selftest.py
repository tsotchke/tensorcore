#!/usr/bin/env python3
"""Selftests for mesh_worker_gpu_reconcile_sweep.py."""

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
    path = ROOT / "scripts" / "mesh_worker_gpu_reconcile_sweep.py"
    loader = importlib.machinery.SourceFileLoader("mesh_worker_gpu_reconcile_sweep_under_test", str(path))
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


def inventory(resources: list[dict]) -> dict:
    return {"schema": "tensorcore.mesh_resources.v1", "resources": resources}


def cuda_resource(resource: str = "cosbox:cuda3090", **overrides: object) -> dict:
    row = {
        "id": resource,
        "node": resource.split(":", 1)[0],
        "backend": "cuda",
        "class": "cuda-training",
        "capacity": 1,
        "status": "active",
        "control_plane": "tensorcore_scheduler",
        "gpu_reconciliation": {
            "enabled": True,
            "poll_host": resource.split(":", 1)[0],
            "allow_process_regex": ["steamwebhelper$", "/opt/google/chrome/chrome"],
            "allowed_process_max_memory_mib": 256,
        },
    }
    row.update(overrides)
    return row


def snapshot(apps: list[dict]) -> dict:
    return {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "reason": "ok",
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "gpus": [],
        "cuda_apps": apps,
        "cuda_pids": [app["pid"] for app in apps],
        "compute_app_count": len(apps),
    }


def args_for(
    root: pathlib.Path,
    *,
    inventory_path: pathlib.Path,
    status_path: pathlib.Path,
    snapshot_dir: pathlib.Path,
    reports_dir: pathlib.Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        inventory_json=inventory_path,
        arbiter_status_json=status_path,
        arbiter_cmd="arbiter",
        arbiter_timeout_sec=1.0,
        snapshot_json_dir=snapshot_dir,
        snapshot_timeout_sec=1.0,
        reports_dir=reports_dir,
        resource=[],
        include_blocked=False,
        offline=True,
    )


def test_sweep_allows_inventory_desktop_allowlist_and_writes_reports() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        inventory_path = write_json(root / "inventory.json", inventory([cuda_resource()]))
        status_path = write_json(root / "status.json", {"leases": []})
        snapshot_dir = root / "snapshots"
        reports_dir = root / "reports"
        write_json(
            snapshot_dir / "cosbox_cuda3090.snapshot.json",
            snapshot(
                [
                    {
                        "pid": 1,
                        "process_name": "/home/cos/snap/steam/common/.local/share/Steam/ubuntu12_64/steamwebhelper",
                        "used_memory_mib": 9,
                    },
                    {
                        "pid": 2,
                        "process_name": "/opt/google/chrome/chrome --type=gpu-process",
                        "used_memory_mib": 248,
                    },
                ]
            ),
        )
        payload = mod.build_payload(
            args_for(
                root,
                inventory_path=inventory_path,
                status_path=status_path,
                snapshot_dir=snapshot_dir,
                reports_dir=reports_dir,
            )
        )
        report = json.loads((reports_dir / "cosbox_cuda3090.reconciliation.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["resource_count"] == 1
    assert report["ok"] is True
    assert report["unmanaged_cuda_apps"] == []
    assert len(report["allowed_cuda_apps"]) == 2


def test_sweep_fails_unmanaged_unleased_cuda() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        inventory_path = write_json(root / "inventory.json", inventory([cuda_resource()]))
        status_path = write_json(root / "status.json", {"leases": []})
        snapshot_dir = root / "snapshots"
        write_json(
            snapshot_dir / "cosbox_cuda3090.snapshot.json",
            snapshot([{"pid": 1234, "process_name": "python train.py", "used_memory_mib": 8192}]),
        )
        payload = mod.build_payload(
            args_for(
                root,
                inventory_path=inventory_path,
                status_path=status_path,
                snapshot_dir=snapshot_dir,
            )
        )
    assert payload["ok"] is False
    assert payload["errors"][0]["resource"] == "cosbox:cuda3090"
    assert payload["reports"][0]["reason"] == "stale_unknown_unleased_cuda"


def test_sweep_skips_disabled_reconciliation_resource() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        row = cuda_resource(
            "jack-blupc:cuda3060",
            gpu_reconciliation={
                "enabled": False,
                "reason": "windows worker snapshot agent pending",
            },
        )
        inventory_path = write_json(root / "inventory.json", inventory([row]))
        status_path = write_json(root / "status.json", {"leases": []})
        payload = mod.build_payload(
            args_for(
                root,
                inventory_path=inventory_path,
                status_path=status_path,
                snapshot_dir=root / "snapshots",
            )
        )
    assert payload["ok"] is True
    assert payload["reports"][0]["action"] == "skip"
    assert payload["reports"][0]["disabled_reason"] == "windows worker snapshot agent pending"


def test_sweep_reports_missing_arbiter_as_structured_failure() -> None:
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        inventory_path = write_json(root / "inventory.json", inventory([cuda_resource()]))
        args = args_for(
            root,
            inventory_path=inventory_path,
            status_path=root / "missing-status.json",
            snapshot_dir=root / "snapshots",
        )
        args.arbiter_status_json = None
        args.arbiter_cmd = "__missing_tensorcore_arbiter__"
        payload = mod.build_payload(args)
    assert payload["ok"] is False
    assert payload["errors"][0]["reason"] == "arbiter_status_unavailable"
    assert payload["errors"][0]["action"] == "block_sweep"


def main() -> int:
    test_sweep_allows_inventory_desktop_allowlist_and_writes_reports()
    test_sweep_fails_unmanaged_unleased_cuda()
    test_sweep_skips_disabled_reconciliation_resource()
    test_sweep_reports_missing_arbiter_as_structured_failure()
    print("mesh worker GPU reconciliation sweep selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
