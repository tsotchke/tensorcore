#!/usr/bin/env python3
"""Fixture tests for the HIP smoke evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_hip_smoke_evidence.py"
TEST_HEAD = "abc123"


def toolchain_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "tools": {"hipcc": {"path": "/opt/chipstar/bin/hipcc"}},
        "cmake_packages": {
            "hip": ["/opt/chipstar/lib/cmake/hip/hip-config.cmake"],
            "hipblas": ["/opt/chipstar/lib/cmake/hipblas/hipblas-config.cmake"],
        },
        "runtime": {
            "opencl_library": "libOpenCL.so.1",
            "opencl_devices": [{
                "platform": "chipStar",
                "name": "SPIR-V GPU",
                "type": "GPU",
                "il_version": "SPIR-V_1.2",
                "extensions": "cl_khr_il_program",
                "spirv_capable": True,
            }],
            "gpu_spirv_device": True,
        },
        "readiness": {
            "hip_runtime_config": True,
            "hipcc": True,
            "spirv_translator": True,
            "opencl_or_level_zero": True,
            "gpu_spirv_runtime": True,
            "hipblas_config": True,
            "status": "ready_for_hip_gemm",
            "missing": [],
        },
    }


def hip_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "runtime_status": "passed",
        "hip_build_enabled": True,
        "hip_gemm_enabled": True,
        "device_count": 1,
        "backend": "hip",
        "kernel": "hipblas_sgemm_staged",
        "gemm_kernels": {
            "hip_gemm_sgemm": {
                "status": "passed",
                "backend": "hip",
                "kernel": "hipblas_sgemm_staged",
            },
            "hip_gemm_hgemm": {
                "status": "passed",
                "backend": "hip",
                "kernel": "hipblas_hgemm_staged",
            },
        },
        "fallback_backend": "portable_cpu",
        "files": {
            "lib/hip/gemm.cpp": {
                "executed_lines": [124, 168],
                "functions": {
                    "hip_gemm_sgemm": {"start_line": 124, "executed_lines": [124]},
                    "hip_gemm_hgemm": {"start_line": 168, "executed_lines": [168]},
                },
            },
        },
        "summary": {
            "required_functions": [
                "lib/hip/gemm.cpp:hip_gemm_hgemm",
                "lib/hip/gemm.cpp:hip_gemm_sgemm",
            ],
            "covered_functions": [
                "lib/hip/gemm.cpp:hip_gemm_hgemm",
                "lib/hip/gemm.cpp:hip_gemm_sgemm",
            ],
            "missing_functions": [],
        },
        "toolchain": toolchain_evidence(),
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
        good = write_json(directory, "hip.json", hip_evidence())
        assert_passes(good)
        assert_passes(
            good,
            "--git-head", TEST_HEAD,
            "--require-hip",
            "--require-hip-build",
            "--require-clean-head",
            "--require-toolchain",
            "--require-ready-toolchain",
            "--require-hip-gemm-sgemm",
            "--require-hip-gemm-hgemm",
        )

        no_toolchain = copy.deepcopy(hip_evidence())
        no_toolchain.pop("toolchain")
        no_toolchain_path = write_json(directory, "no-toolchain.json", no_toolchain)
        assert_passes(no_toolchain_path)
        assert_fails(
            no_toolchain_path,
            "toolchain evidence must be an object",
            "--require-toolchain",
        )

        missing = copy.deepcopy(hip_evidence())
        missing["toolchain"]["readiness"].update({
            "status": "missing_requirements",
            "missing": ["hipcc"],
        })
        missing_path = write_json(directory, "missing-toolchain.json", missing)
        assert_fails(
            missing_path,
            "--require-ready-toolchain needs ready_for_hip_gemm",
            "--require-ready-toolchain",
        )

        no_hgemm = copy.deepcopy(hip_evidence())
        no_hgemm["gemm_kernels"]["hip_gemm_hgemm"]["status"] = "skipped_unsupported"
        no_hgemm_path = write_json(directory, "no-hgemm.json", no_hgemm)
        assert_fails(
            no_hgemm_path,
            "hip_gemm_hgemm.status must be passed",
            "--require-hip-gemm-hgemm",
        )

        missing_coverage = copy.deepcopy(hip_evidence())
        del missing_coverage["files"]["lib/hip/gemm.cpp"]["functions"]["hip_gemm_hgemm"]
        missing_coverage_path = write_json(directory, "missing-coverage.json", missing_coverage)
        assert_fails(
            missing_coverage_path,
            "summary.covered_functions",
        )

    print("HIP smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
