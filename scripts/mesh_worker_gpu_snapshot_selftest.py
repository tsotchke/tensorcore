#!/usr/bin/env python3
"""Selftests for worker GPU snapshot and reconciliation scripts."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import argparse
import pathlib
import subprocess
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


def test_snapshot_supports_ssh_polling_command_prefix() -> None:
    mod = load_script("mesh_worker_gpu_snapshot_under_test", ROOT / "scripts" / "mesh_worker_gpu_snapshot.py")
    args = argparse.Namespace(ssh_host="cosbox", nvidia_smi="nvidia-smi")
    argv = mod.nvidia_smi_command(args, ["--query-gpu=index"])
    assert argv == ["ssh", "cosbox", "nvidia-smi", "--query-gpu=index"]
    args.ssh_host = ""
    assert mod.nvidia_smi_command(args, ["--query-gpu=index"]) == [
        "nvidia-smi",
        "--query-gpu=index",
    ]


def test_snapshot_records_worker_and_poller_hosts_for_ssh() -> None:
    mod = load_script("mesh_worker_gpu_snapshot_under_test", ROOT / "scripts" / "mesh_worker_gpu_snapshot.py")

    def fake_run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        if "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu" in argv:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="0, GPU-test, 00000000:01:00.0, RTX 3090, 24576, 1024, 23552, 7\n",
                stderr="",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="1234, python, 8192\n", stderr="")

    original_run_capture = mod.run_capture
    original_hostname = mod.socket.gethostname
    mod.run_capture = fake_run_capture
    mod.socket.gethostname = lambda: "poller"
    try:
        payload = mod.build_payload(
            argparse.Namespace(
                resource="cosbox:cuda3090",
                nvidia_smi="nvidia-smi",
                ssh_host="cosbox",
                timeout_sec=1.0,
            )
        )
    finally:
        mod.run_capture = original_run_capture
        mod.socket.gethostname = original_hostname

    assert payload["worker_host"] == "cosbox"
    assert payload["poller_host"] == "poller"
    assert payload["cuda_pids"] == [1234]
    assert "cmdline" not in payload["cuda_apps"][0]


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


def test_reconcile_allows_small_desktop_cuda_clients_without_lease() -> None:
    mod = load_script("mesh_worker_gpu_reconcile_under_test", ROOT / "scripts" / "mesh_worker_gpu_reconcile.py")
    snapshot = {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "cuda_apps": [
            {
                "pid": 1476880,
                "process_name": "/home/cos/snap/steam/common/.local/share/Steam/ubuntu12_64/steamwebhelper",
                "used_memory_mib": 9,
            },
            {
                "pid": 142744,
                "process_name": "/opt/google/chrome/chrome --type=gpu-process",
                "used_memory_mib": 248,
            },
        ],
    }
    payload = mod.reconcile(
        snapshot,
        {"leases": []},
        resource="cosbox:cuda3090",
        allow_process_regex=["steamwebhelper$", "/opt/google/chrome/chrome"],
        allowed_process_max_memory_mib=256,
    )
    assert payload["ok"] is True
    assert payload["reason"] == "ok"
    assert payload["action"] == "none"
    assert len(payload["allowed_cuda_apps"]) == 2
    assert payload["unmanaged_cuda_apps"] == []


def test_reconcile_blocks_allowed_process_over_memory_cap_without_lease() -> None:
    mod = load_script("mesh_worker_gpu_reconcile_under_test", ROOT / "scripts" / "mesh_worker_gpu_reconcile.py")
    snapshot = {
        "schema": "tensorcore.mesh_worker_gpu_snapshot.v1",
        "ok": True,
        "worker_host": "cosbox",
        "resource": "cosbox:cuda3090",
        "cuda_apps": [
            {
                "pid": 142744,
                "process_name": "/opt/google/chrome/chrome --type=gpu-process",
                "used_memory_mib": 1024,
            }
        ],
    }
    payload = mod.reconcile(
        snapshot,
        {"leases": []},
        resource="cosbox:cuda3090",
        allow_process_regex=["/opt/google/chrome/chrome"],
        allowed_process_max_memory_mib=256,
    )
    assert payload["ok"] is False
    assert payload["reason"] == "stale_unknown_unleased_cuda"
    assert payload["unmanaged_cuda_apps"][0]["pid"] == 142744


def main() -> int:
    test_snapshot_parses_nvidia_smi_rows()
    test_snapshot_supports_ssh_polling_command_prefix()
    test_snapshot_records_worker_and_poller_hosts_for_ssh()
    test_reconcile_drains_unleased_cuda()
    test_reconcile_allows_leased_cuda()
    test_reconcile_allows_small_desktop_cuda_clients_without_lease()
    test_reconcile_blocks_allowed_process_over_memory_cap_without_lease()
    print("mesh worker GPU snapshot selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
