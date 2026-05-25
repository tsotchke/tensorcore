#!/usr/bin/env python3
"""Fixture tests for the HIP toolchain evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_hip_toolchain_evidence.py"
TEST_HEAD = "abc123"


def evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "platform": {"system": "Linux", "machine": "x86_64", "release": "test"},
        "environment": {
            "TC_HIP_PREFIX": "/opt/chipstar",
            "CHIPSTAR_HOME": None,
            "HIP_PATH": None,
            "ROCM_PATH": None,
            "CMAKE_PREFIX_PATH": "/opt/chipstar",
        },
        "prefixes": ["/opt/chipstar"],
        "tools": {
            "cmake": {"path": "/usr/bin/cmake", "available": True},
            "hipcc": {"path": "/opt/chipstar/bin/hipcc", "available": True},
            "clang": {"path": "/opt/chipstar/bin/clang", "available": True},
            "clang++": {"path": "/opt/chipstar/bin/clang++", "available": True},
            "llvm-spirv": {"path": "/opt/chipstar/bin/llvm-spirv", "available": True},
            "spirv-val": {"path": "/opt/chipstar/bin/spirv-val", "available": True},
            "clinfo": {"path": "/usr/bin/clinfo", "available": True},
        },
        "cmake_packages": {
            "hip": ["/opt/chipstar/lib/cmake/hip/hip-config.cmake"],
            "hipblas": ["/opt/chipstar/lib/cmake/hipblas/hipblas-config.cmake"],
        },
        "runtime": {
            "opencl_library": "libOpenCL.so.1",
            "opencl_icd_files": ["/etc/OpenCL/vendors/chipStar.icd"],
            "level_zero_library": None,
            "clinfo_available": True,
            "clinfo_devices": "Platform #0: chipStar",
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
        "path_hints": [
            "export TC_HIP_PREFIX=/opt/chipstar",
            "export PATH=/opt/chipstar/bin:$PATH",
        ],
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
        good = write_json(directory, "hip-toolchain.json", evidence())
        assert_passes(good)
        assert_passes(
            good,
            "--git-head", TEST_HEAD,
            "--require-clean-head",
            "--require-build-toolchain",
            "--require-spirv-runtime",
            "--require-hipblas",
            "--require-ready",
        )

        missing_runtime = copy.deepcopy(evidence())
        missing_runtime["readiness"].update({
            "opencl_or_level_zero": False,
            "gpu_spirv_runtime": False,
            "status": "missing_requirements",
            "missing": ["OpenCL or Level Zero runtime"],
        })
        missing_runtime_path = write_json(directory, "missing-runtime.json", missing_runtime)
        assert_fails(
            missing_runtime_path,
            "--require-spirv-runtime needs readiness.gpu_spirv_runtime=true",
            "--require-spirv-runtime",
        )

        gpu_missing = copy.deepcopy(evidence())
        gpu_missing["runtime"]["gpu_spirv_device"] = False
        gpu_missing["readiness"].update({
            "gpu_spirv_runtime": False,
            "status": "missing_requirements",
            "missing": ["SPIR-V-capable GPU OpenCL or Level Zero runtime"],
        })
        gpu_missing_path = write_json(directory, "missing-gpu-spirv.json", gpu_missing)
        assert_fails(
            gpu_missing_path,
            "--require-spirv-runtime needs readiness.gpu_spirv_runtime=true",
            "--require-spirv-runtime",
        )

        runtime_only = copy.deepcopy(evidence())
        runtime_only["cmake_packages"]["hipblas"] = []
        runtime_only["readiness"].update({
            "hipblas_config": False,
            "status": "runtime_only_no_hipblas",
            "missing": ["hipBLAS CMake config"],
        })
        runtime_only_path = write_json(directory, "runtime-only.json", runtime_only)
        assert_passes(
            runtime_only_path,
            "--require-build-toolchain",
            "--require-spirv-runtime",
        )
        assert_fails(
            runtime_only_path,
            "--require-hipblas needs readiness.hipblas_config=true",
            "--require-hipblas",
        )

        dirty = copy.deepcopy(evidence())
        dirty["git_dirty"] = True
        dirty_path = write_json(directory, "dirty.json", dirty)
        assert_fails(
            dirty_path,
            "HIP toolchain evidence must be from a clean git tree",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

        stale = copy.deepcopy(evidence())
        stale["git_head"] = "stale"
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails(
            stale_path,
            "HIP toolchain evidence git_head mismatch",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

    print("HIP toolchain evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
