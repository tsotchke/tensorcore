#!/usr/bin/env python3
"""Selftests for scripts/check_windows_persistent_launch.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_windows_persistent_launch.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("check_windows_persistent_launch_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_creates_and_deletes_noop_task() -> None:
    mod = load_module()
    script = mod.render_probe("jack-blupc:cuda3060", "unit")
    assert "schtasks.exe /Create" in script
    assert "/RU $env:USERNAME /NP" not in script
    assert "/TR $TaskCommand" in script
    assert "cmd.exe /c echo ok" in script
    assert "schtasks.exe /Run /TN $TaskName" in script
    assert "marker_written = $markerExists" in script
    assert "schtasks.exe /Delete" in script
    assert "TensorcorePersistentLaunchPreflight_unit" in script


def test_success_payload_passes_through() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "jack-blupc", "--resource", "jack-blupc:cuda3060", "--json"])

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "reason": "ok", "resource": "jack-blupc:cuda3060"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is True
    assert payload["reason"] == "ok"


def test_create_failure_is_payload_failure() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "jack-blupc", "--resource", "jack-blupc:cuda3060"])

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": mod.SCHEMA,
            "ok": False,
            "reason": "scheduled_task_create_failed",
            "resource": "jack-blupc:cuda3060",
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "scheduled_task_create_failed"


def test_invalid_json_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "jack-blupc", "--resource", "jack-blupc:cuda3060"])

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "not-json", "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_probe_json"


def test_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "jack-blupc", "--resource", "jack-blupc:cuda3060"])

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "wrong", "ok": True, "resource": "jack-blupc:cuda3060"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_probe_schema"


def test_resource_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "jack-blupc", "--resource", "jack-blupc:cuda3060"])

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "resource": "other:cuda"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "probe_resource_mismatch"


def main() -> int:
    test_probe_creates_and_deletes_noop_task()
    test_success_payload_passes_through()
    test_create_failure_is_payload_failure()
    test_invalid_json_fails_closed()
    test_schema_mismatch_fails_closed()
    test_resource_mismatch_fails_closed()
    print("Windows persistent launch selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
