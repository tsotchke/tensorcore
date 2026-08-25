#!/usr/bin/env python3
"""Selftests for scripts/check_windows_cuda_smoke_artifact.py."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_windows_cuda_smoke_artifact.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("windows_cuda_smoke_check_under_test", str(SCRIPT))
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
        "artifact_path": "",
        "require_live": False,
        "require_live_or_complete": False,
        "require_complete": False,
        "live_max_age_sec": 15.0,
        "max_age_sec": 0.0,
        "timeout_sec": 20.0,
        "json": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def payload(state: str, *, process_alive: bool = False, runtime_ok: bool = True) -> dict:
    return {
        "schema": "tensorcore.windows_cuda_smoke.check.v1",
        "ok": True,
        "reason": "ok",
        "resource": "jack-blupc:cuda3060",
        "artifact_path": "C:/Users/test/AppData/Local/tensorcore/smoke.json",
        "process_alive": process_alive,
        "checked_at_unix": 1000,
        "artifact": {
            "schema": "tensorcore.windows_cuda_smoke.v1",
            "ok": runtime_ok,
            "state": state,
            "resource": "jack-blupc:cuda3060",
            "build_ok": True,
            "runtime_ok": runtime_ok,
            "checked_at_unix": 1000,
            "heartbeat_unix": 1000,
            "completed_at_unix": 1000,
        },
    }


def test_live_running_process_passes() -> None:
    mod = load_module()
    result = mod.evaluate(payload("running", process_alive=True), args(require_live=True))
    assert result["ok"] is True
    assert result["reason"] == "ok"


def test_live_running_fresh_heartbeat_passes_without_process_probe() -> None:
    mod = load_module()
    result = mod.evaluate(payload("running", process_alive=False), args(require_live=True))
    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["live_artifact_fresh"] is True


def test_live_running_stale_heartbeat_fails_without_process_probe() -> None:
    mod = load_module()
    stale = payload("running", process_alive=False)
    stale["artifact"]["heartbeat_unix"] = 900
    result = mod.evaluate(stale, args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "live_artifact_stale"


def test_default_artifact_path_expands_localappdata() -> None:
    mod = load_module()
    script = mod.render_probe("jack-blupc:cuda3060", "")
    assert "$ArtifactDir = Join-Path $env:LOCALAPPDATA 'tensorcore'" in script
    assert "$ArtifactPath = '$env:LOCALAPPDATA" not in script


def test_live_completed_artifact_fails() -> None:
    mod = load_module()
    result = mod.evaluate(payload("completed", process_alive=False), args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "not_live:completed"


def test_live_or_complete_running_fresh_heartbeat_passes() -> None:
    mod = load_module()
    result = mod.evaluate(payload("running", process_alive=False), args(require_live_or_complete=True))
    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["live_artifact_fresh"] is True


def test_live_or_complete_completed_artifact_passes() -> None:
    mod = load_module()
    result = mod.evaluate(payload("completed"), args(require_live_or_complete=True))
    assert result["ok"] is True
    assert result["reason"] == "ok"


def test_live_or_complete_completed_runtime_failure_fails() -> None:
    mod = load_module()
    result = mod.evaluate(payload("completed", runtime_ok=False), args(require_live_or_complete=True))
    assert result["ok"] is False
    assert result["reason"] == "complete_artifact_not_ok"


def test_live_or_complete_building_artifact_fails() -> None:
    mod = load_module()
    result = mod.evaluate(payload("building", process_alive=True), args(require_live_or_complete=True))
    assert result["ok"] is False
    assert result["reason"] == "not_live_or_complete:building"


def test_probe_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    bad = payload("running", process_alive=True)
    bad["schema"] = "wrong"
    result = mod.evaluate(bad, args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "invalid_probe_schema"


def test_artifact_resource_mismatch_fails_closed() -> None:
    mod = load_module()
    bad = payload("running", process_alive=True)
    bad["artifact"]["resource"] = "other:cuda"
    result = mod.evaluate(bad, args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "artifact_resource_mismatch"


def test_artifact_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    bad = payload("running", process_alive=True)
    bad["artifact"]["schema"] = "wrong"
    result = mod.evaluate(bad, args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "invalid_artifact_schema"


def test_live_building_artifact_fails() -> None:
    mod = load_module()
    result = mod.evaluate(payload("building", process_alive=True), args(require_live=True))
    assert result["ok"] is False
    assert result["reason"] == "not_live:building"


def test_complete_artifact_passes() -> None:
    mod = load_module()
    result = mod.evaluate(payload("completed"), args(require_complete=True))
    assert result["ok"] is True


def test_complete_runtime_failure_fails() -> None:
    mod = load_module()
    result = mod.evaluate(payload("completed", runtime_ok=False), args(require_complete=True))
    assert result["ok"] is False
    assert result["reason"] == "complete_artifact_not_ok"


def test_stale_artifact_fails() -> None:
    mod = load_module()
    stale = payload("completed")
    stale["artifact"]["checked_at_unix"] = 1
    stale["artifact"]["completed_at_unix"] = 1
    result = mod.evaluate(stale, args(require_complete=True, max_age_sec=10.0))
    assert result["ok"] is False
    assert result["reason"] == "artifact_stale"


def test_stale_live_or_complete_completed_artifact_fails() -> None:
    mod = load_module()
    stale = payload("completed")
    stale["artifact"]["checked_at_unix"] = 1
    stale["artifact"]["completed_at_unix"] = 1
    result = mod.evaluate(stale, args(require_live_or_complete=True, max_age_sec=10.0))
    assert result["ok"] is False
    assert result["reason"] == "artifact_stale"


def test_stale_live_or_complete_running_artifact_fails_even_with_fresh_heartbeat() -> None:
    mod = load_module()
    stale = payload("running")
    stale["artifact"]["checked_at_unix"] = 1
    stale["artifact"]["heartbeat_unix"] = 1000
    result = mod.evaluate(stale, args(require_live_or_complete=True, max_age_sec=10.0))
    assert result["ok"] is False
    assert result["reason"] == "artifact_stale"


def test_invalid_remote_json_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "not-json", "")

    mod.run_remote_powershell = fake_run
    result = mod.run_check(args())
    assert result["ok"] is False
    assert result["reason"] == "invalid_probe_json"


def test_remote_object_passes_through_evaluator() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps(payload("running", process_alive=True)), "")

    mod.run_remote_powershell = fake_run
    result = mod.run_check(args(require_live=True))
    assert result["ok"] is True


def main() -> int:
    test_live_running_process_passes()
    test_live_running_fresh_heartbeat_passes_without_process_probe()
    test_live_running_stale_heartbeat_fails_without_process_probe()
    test_default_artifact_path_expands_localappdata()
    test_live_completed_artifact_fails()
    test_live_or_complete_running_fresh_heartbeat_passes()
    test_live_or_complete_completed_artifact_passes()
    test_live_or_complete_completed_runtime_failure_fails()
    test_live_or_complete_building_artifact_fails()
    test_probe_schema_mismatch_fails_closed()
    test_artifact_resource_mismatch_fails_closed()
    test_artifact_schema_mismatch_fails_closed()
    test_live_building_artifact_fails()
    test_complete_artifact_passes()
    test_complete_runtime_failure_fails()
    test_stale_artifact_fails()
    test_stale_live_or_complete_completed_artifact_fails()
    test_stale_live_or_complete_running_artifact_fails_even_with_fresh_heartbeat()
    test_invalid_remote_json_fails_closed()
    test_remote_object_passes_through_evaluator()
    print("Windows CUDA smoke artifact selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
