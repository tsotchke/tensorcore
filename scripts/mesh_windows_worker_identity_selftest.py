#!/usr/bin/env python3
"""Selftests for scripts/mesh_windows_worker_identity.py."""

from __future__ import annotations

import argparse
import inspect
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mesh_windows_worker_identity.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("mesh_windows_worker_identity_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**kwargs: object) -> argparse.Namespace:
    defaults = {
        "target": "jack-blupc",
        "resource": "jack-blupc:cuda3060",
        "match_regex": "tensorcore_cuda_worker",
        "artifact_path": "",
        "allow_completed_artifact": False,
        "require_matching_process": True,
        "require_matched_cuda": True,
        "timeout_sec": 5.0,
        "json": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def raw_identity(*, cuda_pid: int | None = 1234) -> dict:
    apps = []
    if cuda_pid is not None:
        apps.append({
            "pid": cuda_pid,
            "process_name": "test_cuda_gemm.exe",
            "used_memory_mib": 512,
        })
    return {
        "computer_name": "DESKTOP-JACK-BL",
        "user": "tsotchke",
        "matched_processes": [{
            "pid": 1234,
            "ppid": 100,
            "name": "test_cuda_gemm.exe",
            "executable": "C:/tmp/test_cuda_gemm.exe",
            "args": "test_cuda_gemm.exe --tensorcore_cuda_worker",
        }],
        "cuda": {
            "ok": True,
            "apps": apps,
        },
    }


def test_matching_cuda_process_passes() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        assert target == "jack-blupc"
        assert "tensorcore_cuda_worker" in script
        return subprocess.CompletedProcess([], 0, json.dumps(raw_identity()), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is True
    assert payload["worker_host"] == "DESKTOP-JACK-BL"
    assert payload["matched_cuda_pids"] == [1234]


def test_remote_upload_uses_scp_not_ssh_stdin() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_powershell)
    assert '"scp"' in source
    assert "In.ReadToEnd" not in source
    assert "input=script" not in source


def test_matching_process_without_cuda_fails_when_required() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps(raw_identity(cuda_pid=None)), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is False
    assert payload["reason"] == "no_matched_cuda_process"


def test_no_matching_process_fails_when_required() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raw = raw_identity(cuda_pid=None)
        raw["matched_processes"] = []
        return subprocess.CompletedProcess([], 0, json.dumps(raw), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(require_matched_cuda=False))
    assert payload["ok"] is False
    assert payload["reason"] == "no_matching_process"


def test_opaque_wddm_rows_are_not_cuda_identity() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raw = raw_identity(cuda_pid=None)
        raw["cuda"]["apps"] = [{
            "pid": None,
            "process_name": "[Insufficient Permissions]",
            "used_memory_mib": None,
        }]
        return subprocess.CompletedProcess([], 0, json.dumps(raw), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is False
    assert payload["reason"] == "no_matched_cuda_process"
    assert payload["cuda_pids"] == []
    assert len(payload["ignored_opaque_wddm"]) == 1


def test_invalid_json_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "not-json", "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(require_matched_cuda=False))
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_probe_json"


def test_timeout_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["scp"], timeout)

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(require_matched_cuda=False))
    assert payload["ok"] is False
    assert payload["reason"] == "probe_timeout"


def test_artifact_timeout_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["scp"], timeout)

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(allow_completed_artifact=True, require_matching_process=False))
    assert payload["ok"] is False
    assert payload["reason"] == "probe_timeout"


def test_artifact_probe_non_object_json_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "[]", "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(allow_completed_artifact=True, require_matching_process=False))
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_probe_payload"


def test_completed_artifact_identity_passes_when_allowed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        assert "jack-blupc-cuda3060-smoke.json" in script
        raw = {
            "computer_name": "DESKTOP-JACK-BL",
            "user": "tsotchke",
            "reason": "ok",
            "artifact": {
                "schema": "tensorcore.windows_cuda_smoke.v1",
                "resource": "jack-blupc:cuda3060",
                "state": "completed",
                "ok": True,
                "runtime_ok": True,
                "smoke_pid": 100,
                "cuda_pid": 200,
                "executable": "C:/tmp/smoke.exe",
                "token": "unit",
            },
        }
        return subprocess.CompletedProcess([], 0, json.dumps(raw), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args(allow_completed_artifact=True, require_matching_process=False))
    assert payload["ok"] is True
    assert payload["worker_pid"] == 100
    assert payload["matched_cuda_pids"] == [200]


def main() -> int:
    test_matching_cuda_process_passes()
    test_remote_upload_uses_scp_not_ssh_stdin()
    test_matching_process_without_cuda_fails_when_required()
    test_no_matching_process_fails_when_required()
    test_opaque_wddm_rows_are_not_cuda_identity()
    test_invalid_json_fails_closed()
    test_timeout_fails_closed()
    test_artifact_timeout_fails_closed()
    test_artifact_probe_non_object_json_fails_closed()
    test_completed_artifact_identity_passes_when_allowed()
    print("Windows worker identity selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
