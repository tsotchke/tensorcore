#!/usr/bin/env python3
"""Selftests for scripts/start_windows_cuda_smoke.py."""

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
SCRIPT = ROOT / "scripts" / "start_windows_cuda_smoke.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("start_windows_cuda_smoke_under_test", str(SCRIPT))
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
        "duration_sec": 45,
        "start_wait_sec": 20,
        "timeout_sec": 40.0,
        "foreground": False,
        "recover_foreground_timeout": False,
        "json": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_start_payload_passes_through() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        assert target == "jack-blupc"
        assert "jack-blupc:cuda3060" in script
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "resource": "jack-blupc:cuda3060",
            "state": "running",
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args())
    assert payload["ok"] is True
    assert payload["state"] == "running"


def test_default_artifact_path_expands_localappdata() -> None:
    mod = load_module()
    script = mod.render_parent(args(), "unit-token")
    assert "$ArtifactDir = Join-Path $env:LOCALAPPDATA 'tensorcore'" in script
    assert "$ArtifactPath = '$env:LOCALAPPDATA" not in script
    assert "Remove-Item -Force -LiteralPath $ArtifactPath" in script
    assert "$candidate.token -eq $Token" in script
    assert "schtasks.exe /Create" in script
    assert "/RU $env:USERNAME /NP" not in script
    assert "/TR $TaskCommand" in script
    assert "schtasks.exe /Run /TN $TaskName" in script
    assert "$TaskScriptPath = Join-Path $env:TEMP" in script
    assert "Set-Content -LiteralPath $TaskScriptPath" in script
    assert "scheduled_task_name = $TaskName" in script


def test_remote_upload_uses_scp_not_ssh_stdin() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_powershell)
    assert '"scp"' in source
    assert "In.ReadToEnd" not in source
    assert "input=script" not in source


def test_child_runtime_wait_is_bounded() -> None:
    mod = load_module()
    assert "[System.Diagnostics.ProcessStartInfo]::new()" in mod.CHILD_SCRIPT
    assert "[System.Diagnostics.Process]::new()" in mod.CHILD_SCRIPT
    assert "Start-Process -FilePath $exe" not in mod.CHILD_SCRIPT
    assert "-PassThru" not in mod.CHILD_SCRIPT
    assert "-Wait -PassThru" not in mod.CHILD_SCRIPT
    assert "ReadToEndAsync()" in mod.CHILD_SCRIPT
    assert "$runProc.WaitForExit($runtimeTimeoutMs)" in mod.CHILD_SCRIPT
    assert "$runProc.Kill()" in mod.CHILD_SCRIPT
    assert "$stdoutLooksOk" in mod.CHILD_SCRIPT
    assert "cuda_pid = $cudaPid" in mod.CHILD_SCRIPT
    assert "runtime_timeout_sec = $runtimeTimeoutSec" in mod.CHILD_SCRIPT
    assert "Finish-Smoke" in mod.CHILD_SCRIPT
    assert "schtasks.exe /Delete /F /TN $TaskName" in mod.CHILD_SCRIPT
    assert "heartbeat_unix = UnixNow" in mod.CHILD_SCRIPT
    assert "runtime_timeout" in mod.CHILD_SCRIPT


def test_foreground_mode_waits_for_completed_artifact() -> None:
    mod = load_module()
    script = mod.render_parent(args(foreground=True), "unit-token")
    assert "launch_mode = 'foreground'" in script
    assert "$payload.state -eq 'completed'" in script
    assert "exit 0\n\n$InvokeLine" in script


def test_scheduled_mode_accepts_running_or_completed_artifact() -> None:
    mod = load_module()
    script = mod.render_parent(args(), "unit-token")
    assert "$startOk = (" in script
    assert "$payload.state -eq 'running' -or" in script
    assert "$payload.state -eq 'completed' -and $payload.ok -eq $true" in script
    assert "ok = $startOk" in script


def test_invalid_start_json_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "not-json", "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args())
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_start_json"


def test_invalid_start_schema_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "wrong", "resource": "jack-blupc:cuda3060"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args())
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_start_schema"


def test_start_resource_mismatch_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "resource": "other:cuda"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args())
    assert payload["ok"] is False
    assert payload["reason"] == "start_resource_mismatch"


def test_start_timeout_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["ssh"], timeout)

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args())
    assert payload["ok"] is False
    assert payload["reason"] == "start_timeout"
    assert payload["token"].startswith("tensorcore_windows_cuda_smoke_")


def test_foreground_timeout_recovers_completed_artifact() -> None:
    mod = load_module()
    calls = []

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(script)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(["ssh"], timeout)
        assert "foreground_recovered_after_timeout" in script
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "reason": "foreground_recovered",
            "resource": "jack-blupc:cuda3060",
            "state": "completed",
            "payload": {"state": "completed", "runtime_ok": True},
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args(foreground=True, recover_foreground_timeout=True))
    assert payload["ok"] is True
    assert payload["reason"] == "foreground_recovered"
    assert len(calls) == 2


def test_foreground_timeout_does_not_recover_by_default() -> None:
    mod = load_module()
    calls = []

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(script)
        raise subprocess.TimeoutExpired(["ssh"], timeout)

    mod.run_remote_powershell = fake_run
    payload = mod.run_start(args(foreground=True))
    assert payload["ok"] is False
    assert payload["reason"] == "start_timeout"
    assert len(calls) == 1


def main() -> int:
    test_start_payload_passes_through()
    test_default_artifact_path_expands_localappdata()
    test_remote_upload_uses_scp_not_ssh_stdin()
    test_child_runtime_wait_is_bounded()
    test_foreground_mode_waits_for_completed_artifact()
    test_scheduled_mode_accepts_running_or_completed_artifact()
    test_invalid_start_json_fails_closed()
    test_invalid_start_schema_fails_closed()
    test_start_resource_mismatch_fails_closed()
    test_start_timeout_fails_closed()
    test_foreground_timeout_recovers_completed_artifact()
    test_foreground_timeout_does_not_recover_by_default()
    print("Windows CUDA smoke starter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
