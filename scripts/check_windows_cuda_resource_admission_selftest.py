#!/usr/bin/env python3
"""Selftests for scripts/check_windows_cuda_resource_admission.py."""

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
SCRIPT = ROOT / "scripts" / "check_windows_cuda_resource_admission.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("windows_cuda_admission_under_test", str(SCRIPT))
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
        "allowed_process_max_memory_mib": 64,
        "require_toolchain": True,
        "disallow_opaque_wddm": False,
        "timeout_sec": 5.0,
        "json": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_success_payload_passes_through() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        assert target == "jack-blupc"
        assert "jack-blupc:cuda3060" in script
        assert "$requireToolchain = $true" in script
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "reason": "ok_opaque_wddm_rows_no_visible_cuda_processes",
            "resource": "jack-blupc:cuda3060",
            "driver_ok": True,
            "toolchain_ok": True,
            "admission_ok": True,
            "blocked": [],
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is True
    assert payload["toolchain_ok"] is True


def test_remote_upload_uses_scp_not_ssh_stdin() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_powershell)
    assert '"scp"' in source
    assert "In.ReadToEnd" not in source
    assert "input=script" not in source


def test_invalid_json_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "not-json", "")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_probe_json"


def test_remote_failure_reports_tail() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 255, "", "ssh failed")

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is False
    assert payload["reason"] == "ssh_or_powershell_failed"
    assert "ssh failed" in payload["stderr_tail"]


def test_timeout_fails_closed() -> None:
    mod = load_module()

    def fake_run(target: str, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["scp"], timeout)

    mod.run_remote_powershell = fake_run
    payload = mod.run_probe(args())
    assert payload["ok"] is False
    assert payload["reason"] == "probe_timeout"


def main() -> int:
    test_success_payload_passes_through()
    test_remote_upload_uses_scp_not_ssh_stdin()
    test_invalid_json_fails_closed()
    test_remote_failure_reports_tail()
    test_timeout_fails_closed()
    print("Windows CUDA resource admission selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
