#!/usr/bin/env python3
"""Selftests for worker GPU snapshot and reconciliation scripts."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_parses_nvidia_smi_rows() -> None:
    mod = load_script("mesh_worker_gpu_snapshot_under_test", ROOT / "scripts" / "mesh_worker_gpu_snapshot.py")
    apps = mod.parse_compute_apps("1234, /usr/bin/python, 8192\n")
    gpus = mod.parse_gpu_rows("0, GPU-test, 00000000:01:00.0, RTX 3090, 24576, 1024, 23552, 7\n")
    assert apps[0]["pid"] == 1234
    assert apps[0]["used_memory_mib"] == 8192
    assert gpus[0]["uuid"] == "GPU-test"
    assert gpus[0]["memory_total_mib"] == 24576


def test_reconcile_drains_unleased_cuda() -> None:
    mod = load_script("mesh_worker_gpu_reconcile_under_test", ROOT / "scripts" / "mesh_worker_gpu_reconcile.py")
    snapshot = {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "cuda_apps": [{"pid": 1234, "process_name": "python", "used_memory_mib": 8192}],
    }
    status = {"leases": []}
    payload = mod.reconcile(snapshot, status, resource="cosbox:cuda3090")
    assert payload["ok"] is False
    assert payload["reason"] == "stale_unknown_unleased_cuda"
    assert payload["action"] == "drain"


def test_reconcile_allows_leased_cuda() -> None:
    mod = load_script("mesh_worker_gpu_reconcile_under_test", ROOT / "scripts" / "mesh_worker_gpu_reconcile.py")
    snapshot = {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "cuda_apps": [{"pid": 1234, "process_name": "python", "used_memory_mib": 8192}],
    }
    status = {"leases": [{"id": "lease-1", "resource": "cosbox:cuda3090", "status": "active"}]}
    payload = mod.reconcile(snapshot, status, resource="cosbox:cuda3090")
    assert payload["ok"] is True
    assert payload["active_lease_ids"] == ["lease-1"]


def main() -> int:
    test_snapshot_parses_nvidia_smi_rows()
    test_reconcile_drains_unleased_cuda()
    test_reconcile_allows_leased_cuda()
    print("mesh worker GPU snapshot selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
