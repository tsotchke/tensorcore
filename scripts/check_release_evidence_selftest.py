#!/usr/bin/env python3
"""Fixture tests for the release evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_release_evidence.py"


def base_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.release_smoke.runtime_evidence.v1",
        "meta": {"format": 3, "source": "tensorcore_release_smoke"},
        "status": "passed",
        "run": {"phase": "complete", "exit_status": "0"},
        "files": {
            "lib/core/device.mm": {},
            "python/tensorcore/__init__.py": {},
        },
        "summary": {
            "tests_passed": True,
            "wheel_tag_inspected": True,
            "installed_wheel_smoke_passed": True,
            "cmake_consumers_passed": True,
            "pkg_config_consumer_passed": True,
            "packaging_and_consumers_passed": True,
            "public_headers_passed": True,
            "python_ffi_surface_passed": True,
            "python_constants_passed": True,
            "python_abi_layout_passed": True,
            "public_core_paths_passed": False,
            "public_integration_runtime_passed": False,
            "autotune_cache_passed": False,
            "gemm_128_tile_passed": False,
            "gemm_async_passed": False,
            "metal4_tensorops_compile_passed": False,
            "metal4_tensorops_runtime_passed": False,
        },
        "checks": {
            "tests": {
                "status": "passed",
                "passed": True,
                "mode": "paravirtual_safe_subset",
                "gpu_device_available": False,
            },
            "wheel_tag": {
                "status": "passed",
                "inspected": True,
            },
            "installed_wheel_smoke": {
                "status": "passed",
                "passed": True,
            },
            "consumers": {
                "cmake": {"status": "passed", "passed": True},
                "pkg_config": {"status": "passed", "passed": True},
            },
            "packaging_and_consumers": {
                "runtime_status": "passed",
                "runtime_covered": True,
            },
            "public_headers": {"status": "passed", "passed": True},
            "python_ffi_surface": {"status": "passed", "passed": True},
            "python_constants": {"status": "passed", "passed": True},
            "python_abi_layout": {"status": "passed", "passed": True},
            "public_integration": {
                "runtime_status": "skipped_no_gpu",
                "runtime_covered": False,
            },
            "autotune_cache": {"status": "skipped_no_gpu", "passed": False},
            "gemm_env_variants": {
                "use_128_tile": {"status": "skipped_paravirtual_gpu", "passed": False},
                "use_async": {"status": "skipped_paravirtual_gpu", "passed": False},
            },
            "metal4_tensorops": {
                "compile_status": "skipped_sdk_too_old",
                "runtime_compile_status": "skipped_sdk_too_old",
                "runtime_status": "skipped_no_m5",
                "runtime_covered": False,
            },
            "public_core_paths": {
                "runtime_status": "skipped_no_gpu",
                "runtime_covered": False,
                "required_files": [
                    "lib/core/device.mm",
                    "lib/ops/gemm.mm",
                    "python/tensorcore/__init__.py",
                ],
                "missing_files": [],
                "uncovered_files": ["lib/ops/gemm.mm"],
            },
        },
    }


def run_checker(evidence: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(evidence, handle)
        path = pathlib.Path(handle.name)
    try:
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_passes(evidence: dict[str, Any]) -> None:
    result = run_checker(evidence)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(evidence: dict[str, Any], needle: str) -> None:
    result = run_checker(evidence)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    good = base_evidence()
    assert_passes(good)

    stale = copy.deepcopy(good)
    stale["checks"]["public_core_paths"]["missing_files"] = ["lib/ops/gemm.mm"]
    assert_fails(stale, "no-GPU/paravirtual evidence must not report")

    drifted = copy.deepcopy(good)
    drifted["checks"]["public_core_paths"]["uncovered_files"] = []
    assert_fails(drifted, "uncovered_files must match")

    gpu_missing = copy.deepcopy(good)
    gpu_missing["checks"]["tests"]["gpu_device_available"] = True
    gpu_missing["checks"]["tests"]["mode"] = "full"
    gpu_missing["checks"]["public_core_paths"]["runtime_status"] = "failed"
    gpu_missing["checks"]["public_core_paths"]["missing_files"] = []
    assert_fails(gpu_missing, "GPU evidence with incomplete public core coverage")

    print("release evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
