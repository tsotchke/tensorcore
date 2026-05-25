#!/usr/bin/env python3
"""Fixture tests for the Windows CUDA probe evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_windows_cuda_probe_evidence.py"
TEST_HEAD = "abc123"


def evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_cuda_probe.evidence.v1",
        "schema_version": 1,
        "runtime_status": "driver_only",
        "checked_at_unix": 123,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "ref": "master",
        "repo": "src/tensorcore",
        "remote_url": "https://github.com/tsotchke/tensorcore.git",
        "host": {
            "computer_name": "DESKTOP-JACK-BLUPC",
            "user": "tsotchke",
            "os": "Microsoft Windows 11 Pro",
        },
        "nvidia_smi": {
            "found": True,
            "path": "C:/Windows/System32/nvidia-smi.exe",
            "query_rc": 0,
            "stderr_tail": "",
        },
        "device_count": 1,
        "devices": [{
            "name": "NVIDIA GeForce RTX 3060",
            "driver_version": "560.94",
            "memory_total_mib": 12288,
            "compute_capability": "8.6",
        }],
        "cuda_toolkit": {
            "nvcc_found": False,
            "nvcc_path": None,
            "nvcc_version": "",
            "cuda_path": None,
            "toolkit_dirs": [],
        },
        "admission": {
            "ok": True,
            "reason": "ok",
            "allowed_process_max_memory_mib": 64,
            "compute_app_count": 0,
            "blocked": [],
        },
    }


def write_json(directory: pathlib.Path, name: str, data: dict[str, Any]) -> pathlib.Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checker(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_passes(path: pathlib.Path, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(path: pathlib.Path, needle: str, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        good = write_json(directory, "windows-cuda.json", evidence())
        assert_passes(good)
        assert_passes(
            good,
            "--git-head", TEST_HEAD,
            "--require-clean-head",
            "--require-driver",
            "--require-admission-clear",
        )
        assert_fails(good, "--require-toolchain needs nvcc on PATH", "--require-toolchain")
        assert_fails(good, "--require-ready needs ready evidence", "--require-ready")

        ready = copy.deepcopy(evidence())
        ready["runtime_status"] = "ready"
        ready["cuda_toolkit"]["nvcc_found"] = True
        ready["cuda_toolkit"]["nvcc_path"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
        ready_path = write_json(directory, "ready.json", ready)
        assert_passes(ready_path, "--require-ready", "--require-toolchain")

        dirty = copy.deepcopy(evidence())
        dirty["git_dirty"] = True
        dirty_path = write_json(directory, "dirty.json", dirty)
        assert_fails(
            dirty_path,
            "Windows CUDA evidence must be from a clean git tree",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

        stale = copy.deepcopy(evidence())
        stale["git_head"] = "stale"
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails(
            stale_path,
            "Windows CUDA evidence git_head mismatch",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

        blocked = copy.deepcopy(evidence())
        blocked["runtime_status"] = "admission_blocked"
        blocked["admission"]["ok"] = False
        blocked["admission"]["reason"] = "blocked_cuda_compute_apps"
        blocked["admission"]["blocked"] = [{"pid": 1234, "process_name": "python.exe"}]
        blocked_path = write_json(directory, "blocked.json", blocked)
        assert_passes(blocked_path, "--require-driver")
        assert_fails(
            blocked_path,
            "--require-admission-clear needs admission.ok=true",
            "--require-admission-clear",
        )

        blocked_without_driver = copy.deepcopy(blocked)
        blocked_without_driver["nvidia_smi"]["found"] = False
        blocked_without_driver["device_count"] = 0
        blocked_without_driver["devices"] = []
        blocked_without_driver_path = write_json(
            directory, "blocked-without-driver.json", blocked_without_driver
        )
        assert_fails(
            blocked_without_driver_path,
            "admission_blocked evidence must include at least one CUDA device",
        )

        unavailable = copy.deepcopy(evidence())
        unavailable["runtime_status"] = "unavailable"
        unavailable["nvidia_smi"]["found"] = False
        unavailable["device_count"] = 0
        unavailable["devices"] = []
        unavailable_path = write_json(directory, "unavailable.json", unavailable)
        assert_fails(
            unavailable_path,
            "--require-driver needs nvidia_smi and at least one CUDA device",
            "--require-driver",
        )

    print("Windows CUDA probe evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
