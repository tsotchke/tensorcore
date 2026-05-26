#!/usr/bin/env python3
"""Fixture tests for the operational evidence bundle checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import check_live_mesh_training_evidence_selftest as live_mesh_fixture
import check_release_evidence_selftest as release_fixture
from check_cuda_smoke_evidence import EXPECTED_TRAINING_KERNELS


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_operational_evidence.py"
TEST_HEAD = "abc123"


def live_mesh_evidence() -> dict[str, Any]:
    evidence = live_mesh_fixture.base_evidence()
    evidence["meta"]["git_head"] = TEST_HEAD
    evidence["meta"]["git_dirty"] = False
    return evidence


def release_evidence() -> dict[str, Any]:
    evidence = release_fixture.base_evidence()
    evidence["meta"]["git_head"] = TEST_HEAD
    evidence["meta"]["git_dirty"] = False
    return evidence


def sdk26_evidence() -> dict[str, Any]:
    evidence = release_evidence()
    evidence["checks"]["metal4_tensorops"]["compile_status"] = "compiled"
    evidence["checks"]["metal4_tensorops"]["runtime_compile_status"] = "compiled"
    evidence["summary"]["metal4_tensorops_compile_passed"] = True
    return evidence


def cuda_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "runtime_status": "passed",
        "cuda_build_enabled": True,
        "device_count": 1,
        "backend": "cuda",
        "f32_kernel": "cublas_sgemm_managed",
        "f16_kernel": "cublas_gemmex_fp16_tensorop_managed",
        "fallback_backend": "portable_cpu",
        "training_kernels": {
            op: {"backend": "cuda", "kernel": kernel}
            for op, kernel in EXPECTED_TRAINING_KERNELS.items()
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
        "fallback_backend": "portable_cpu",
    }


def hip_toolchain_evidence() -> dict[str, Any]:
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
            "hipcc": {"path": "/opt/chipstar/bin/hipcc", "available": True},
            "llvm-spirv": {"path": "/opt/chipstar/bin/llvm-spirv", "available": True},
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
        "path_hints": ["export TC_HIP_PREFIX=/opt/chipstar"],
    }


def hip_runtime_unavailable_evidence() -> dict[str, Any]:
    evidence = hip_evidence()
    evidence.update({
        "runtime_status": "skipped_runtime_unavailable",
        "hip_build_enabled": True,
        "hip_gemm_enabled": False,
        "device_count": 0,
        "backend": None,
        "kernel": "none",
        "fallback_backend": None,
        "init_status": -4,
        "init_status_string": "operation unsupported on this GPU family",
    })
    return evidence


def hip_not_built_evidence() -> dict[str, Any]:
    evidence = hip_runtime_unavailable_evidence()
    evidence.update({
        "runtime_status": "skipped_not_built",
        "hip_build_enabled": False,
    })
    return evidence


def pytorch_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "require_pytorch": False,
        "require_pytorch_backend": False,
        "runtime_status": "passed",
        "message": "tensorcore PyTorch bridge smoke OK",
        "torch_version": "2.11.0",
        "tensorcore_lib_dir": "/tmp/tensorcore",
        "backend_state": {
            "backend_name": "tensorcore",
            "privateuse1_backend_name": "tensorcore",
            "extension_privateuse1_backend_name": "tensorcore",
            "registered": True,
            "torch_module_registered": True,
            "generated_tensor_methods": True,
            "is_available": True,
            "device_count": 1,
            "current_device": 0,
            "supports_device_allocation": True,
            "allocator_status": "available",
            "factory_kernels": True,
            "storage_kernels": True,
            "matmul_extension_loaded": True,
            "matmul_dispatch_probe": {"eligible": True, "reason": "eligible"},
            "default_matmul_enabled": False,
            "last_backend_name": "portable_cpu",
            "amp_supported_dtypes": ["torch.float32", "torch.bfloat16"],
        },
        "backend_report": "tensorcore PyTorch backend: registered=True allocation=available",
        "matmul": {
            "fp32_eligibility_reason": "eligible",
            "fp32_backend": "portable_cpu",
            "bf16_checked": True,
            "noncontiguous_checked": True,
            "degenerate_checked": True,
            "error_paths_checked": True,
            "default_matmul_dispatch_checked": True,
            "autograd_fallback_checked": True,
            "privateuse1_matmul_checked": True,
            "device_roundtrip_checked": True,
        },
        "direct_device_allocation": {
            "available": True,
            "error": None,
        },
    }


def windows_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_host_smoke.evidence.v1",
        "schema_version": 1,
        "runtime_status": "passed",
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
        "update": {"reset": False},
        "bootstrap": {
            "ran": True,
            "install_requested": False,
            "skip_python": False,
        },
    }


def windows_cuda_evidence() -> dict[str, Any]:
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


def windows_cuda_ready_evidence() -> dict[str, Any]:
    data = windows_cuda_evidence()
    data["runtime_status"] = "ready"
    data["cuda_toolkit"]["nvcc_found"] = True
    data["cuda_toolkit"]["nvcc_path"] = "C:/Users/tsotchke/src/cuda-redist-12.6/toolkit/bin/nvcc.exe"
    data["cuda_toolkit"]["cuda_path"] = "C:/Users/tsotchke/src/cuda-redist-12.6/toolkit"
    data["build_smoke"] = {
        "ran": True,
        "ok": True,
        "build_dir": "C:/Users/tsotchke/src/tensorcore/build-windows-cuda-smoke",
        "reason": None,
        "rc": 0,
        "tests_total": 17,
        "tests_passed": 11,
        "tests_failed": 0,
        "tests_skipped": 6,
        "cuda_gemm_passed": True,
    }
    return data


def windows_cuda_smoke_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_cuda_scheduled_smoke.evidence.v1",
        "schema_version": 1,
        "checked_at_unix": 4102444800,
        "phase": "completed",
        "resource": "jack-blupc:cuda3060",
        "job": "jack-cuda3060-smoke",
        "driver_visible": True,
        "toolchain_found": True,
        "wddm_admission_ok": True,
        "build_smoke_passed": True,
        "runtime_smoke_passed": True,
        "scheduler_lease_held": True,
        "worker_identity_recorded": True,
        "lease_id": "lease-jack",
        "smoke_artifact": {
            "schema": "tensorcore.windows_cuda_smoke.v1",
            "ok": True,
            "resource": "jack-blupc:cuda3060",
            "state": "completed",
            "build_ok": True,
            "runtime_ok": True,
        },
        "worker_identity": {
            "schema": "tensorcore.mesh_worker_identity.v1",
            "ok": True,
            "resource": "jack-blupc:cuda3060",
            "worker_host": "DESKTOP-JACK-BLUPC",
            "worker_pid": 1234,
        },
    }


def mesh_preflights_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.mesh_resource_preflights.v1",
        "ok": True,
        "checked_at_unix": 4102444800,
        "jobs_checked": 3,
        "missing_job_ids": [],
        "results": [
            {
                "job": "georefine-m2-cosbox",
                "resource": "cosbox:cuda3090",
                "ok": True,
                "reason": "preflight_ok",
                "rc": 0,
                "json": {
                    "schema": "tensorcore.georefine_qwen_cr025.start.v1",
                    "ok": True,
                    "resource": "cosbox:cuda3090",
                    "reason": "preflight_ok",
                },
            },
            {
                "job": "old-donkey-precompute-chain",
                "resource": "old-donkey:cuda3050",
                "ok": True,
                "reason": "preflight_ok",
                "rc": 0,
                "json": {
                    "schema": "tensorcore.qllm_olddonkey_precompute_chain.start.v1",
                    "ok": True,
                    "resource": "old-donkey:cuda3050",
                    "reason": "preflight_ok",
                },
            },
            {
                "job": "jack-cuda3060-smoke",
                "resource": "jack-blupc:cuda3060",
                "ok": True,
                "reason": "preflight_ok",
                "rc": 0,
                "json": {
                    "schema": "tensorcore.windows_persistent_launch.v1",
                    "ok": True,
                    "resource": "jack-blupc:cuda3060",
                    "reason": "preflight_ok",
                },
            },
        ],
    }


def mesh_default_preflights_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.mesh_resource_preflights.v1",
        "ok": True,
        "checked_at_unix": 4102444800,
        "jobs_checked": 0,
        "missing_job_ids": [],
        "skipped_default_job_ids": ["jack-cuda3060-smoke"],
        "results": [],
    }


def mesh_preflights_with_jack_blocker_evidence() -> dict[str, Any]:
    data = mesh_preflights_evidence()
    data["ok"] = False
    for row in data["results"]:
        if row["job"] == "jack-cuda3060-smoke":
            row["ok"] = False
            row["reason"] = "scheduled_task_did_not_run"
            row["rc"] = 1
            row["json"] = {
                "schema": "tensorcore.windows_persistent_launch.v1",
                "ok": False,
                "resource": "jack-blupc:cuda3060",
                "reason": "scheduled_task_did_not_run",
            }
    return data


def write_json(directory: pathlib.Path, name: str, data: dict[str, Any]) -> pathlib.Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_passes(*args: str) -> None:
    result = run_checker(*args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(needle: str, *args: str) -> None:
    result = run_checker(*args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        live_path = write_json(directory, "live.json", live_mesh_evidence())
        release_path = write_json(directory, "release.json", release_evidence())
        sdk26_path = write_json(directory, "sdk26.json", sdk26_evidence())
        cuda_path = write_json(directory, "cuda.json", cuda_evidence())
        hip_path = write_json(directory, "hip.json", hip_evidence())
        hip_toolchain_path = write_json(
            directory, "hip-toolchain.json", hip_toolchain_evidence()
        )
        pytorch_path = write_json(directory, "pytorch.json", pytorch_evidence())
        windows_path = write_json(directory, "windows.json", windows_evidence())
        windows_cuda_path = write_json(
            directory, "windows-cuda.json", windows_cuda_ready_evidence()
        )
        windows_cuda_smoke_path = write_json(
            directory, "windows-cuda-smoke.json", windows_cuda_smoke_evidence()
        )
        mesh_preflights_path = write_json(
            directory, "mesh-preflights.json", mesh_preflights_evidence()
        )
        mesh_default_preflights_path = write_json(
            directory, "mesh-default-preflights.json", mesh_default_preflights_evidence()
        )
        mesh_jack_blocked_path = write_json(
            directory,
            "mesh-jack-blocked.json",
            mesh_preflights_with_jack_blocker_evidence(),
        )

        assert_passes(
            "--release", str(release_path),
            "--sdk26", str(sdk26_path),
            "--cuda", str(cuda_path),
            "--hip", str(hip_path),
            "--hip-toolchain", str(hip_toolchain_path),
            "--pytorch", str(pytorch_path),
            "--windows", str(windows_path),
            "--windows-cuda", str(windows_cuda_path),
            "--windows-cuda-smoke", str(windows_cuda_smoke_path),
            "--mesh-preflights", str(mesh_preflights_path),
            "--live-mesh", str(live_path),
            "--git-head", TEST_HEAD,
            "--require-release",
            "--require-sdk26",
            "--require-cuda",
            "--require-hip",
            "--require-hip-build",
            "--require-hip-toolchain",
            "--require-hip-spirv-runtime",
            "--require-ready-hip-toolchain",
            "--require-pytorch",
            "--require-pytorch-backend-allocation",
            "--require-windows",
            "--require-windows-python",
            "--require-windows-cuda-driver",
            "--require-windows-cuda-toolchain",
            "--require-windows-cuda-admission-clear",
            "--require-windows-cuda-ready",
            "--require-windows-cuda-build-smoke",
            "--require-windows-cuda-scheduled-smoke",
            "--windows-cuda-smoke-max-age-sec", "86400",
            "--require-mesh-preflights",
            "--require-mesh-preflights-pass",
            "--mesh-preflights-max-age-sec", "86400",
            "--mesh-preflight-job", "georefine-m2-cosbox",
            "--mesh-preflight-job", "old-donkey-precompute-chain",
            "--mesh-preflight-job", "jack-cuda3060-smoke",
            "--require-live-mesh",
            "--require-release-clean-head",
            "--require-sdk26-clean-head",
            "--require-cuda-clean-head",
            "--require-hip-clean-head",
            "--require-hip-toolchain-clean-head",
            "--require-pytorch-clean-head",
            "--require-windows-clean-head",
            "--require-windows-cuda-clean-head",
            "--require-live-clean-head",
            "--min-live-outer-steps", "5",
            "--require-direct-ring",
            "--require-checkpoint",
            "--require-cuda-rank3",
            "--require-explicit-backends",
            "--require-no-backend-fallback",
        )

        assert_passes(
            "--mesh-preflights", str(mesh_default_preflights_path),
            "--mesh-preflight-skipped-default-job", "jack-cuda3060-smoke",
            "--mesh-preflights-max-age-sec", "86400",
        )
        assert_passes(
            "--mesh-preflights", str(mesh_jack_blocked_path),
            "--mesh-preflight-job", "jack-cuda3060-smoke",
            "--mesh-preflight-allowed-failure",
            "jack-cuda3060-smoke:scheduled_task_did_not_run",
            "--mesh-preflights-max-age-sec", "86400",
        )
        assert_fails(
            "allowed failure reason mismatch",
            "--mesh-preflights", str(mesh_jack_blocked_path),
            "--mesh-preflight-allowed-failure",
            "jack-cuda3060-smoke:wrong_reason",
        )
        assert_fails(
            "required skipped default jobs missing",
            "--mesh-preflights", str(mesh_default_preflights_path),
            "--mesh-preflight-skipped-default-job", "old-donkey-precompute-chain",
        )

        assert_fails("no evidence paths were provided")
        assert_fails("--live-mesh evidence is required", "--require-live-mesh")
        assert_fails("--windows evidence is required", "--require-windows")
        assert_fails(
            "--windows-cuda evidence is required",
            "--require-windows-cuda-driver",
        )
        assert_fails(
            "--windows-cuda-smoke evidence is required",
            "--require-windows-cuda-scheduled-smoke",
        )
        assert_fails(
            "--mesh-preflights evidence is required",
            "--require-mesh-preflights",
        )
        assert_fails(
            "--mesh-preflights evidence is required",
            "--mesh-preflight-skipped-default-job", "jack-cuda3060-smoke",
        )
        assert_fails(
            "--mesh-preflights evidence is required",
            "--mesh-preflight-allowed-failure", "jack-cuda3060-smoke:scheduled_task_did_not_run",
        )
        assert_fails("--hip-toolchain evidence is required", "--require-hip-toolchain")
        assert_fails(
            "--hip-toolchain evidence is required",
            "--require-hip-spirv-runtime",
        )
        assert_fails(
            "expected git head is unavailable",
            "--live-mesh", str(live_path),
            "--git-head", "",
            "--require-live-clean-head",
        )

        dirty = copy.deepcopy(live_mesh_evidence())
        dirty["meta"]["git_dirty"] = True
        dirty_path = write_json(directory, "dirty.fixture", dirty)
        assert_fails(
            "live mesh evidence must be from a clean git tree",
            "--live-mesh", str(dirty_path),
            "--git-head", TEST_HEAD,
            "--require-live-clean-head",
        )

        stale = copy.deepcopy(live_mesh_evidence())
        stale["meta"]["git_head"] = "stale"
        stale_path = write_json(directory, "stale.fixture", stale)
        assert_fails(
            "live mesh evidence git_head mismatch",
            "--live-mesh", str(stale_path),
            "--git-head", TEST_HEAD,
            "--require-live-clean-head",
        )

        stale_release = release_evidence()
        stale_release["meta"]["git_head"] = "stale"
        stale_release_path = write_json(directory, "stale-release.fixture", stale_release)
        assert_fails(
            "release evidence git_head mismatch",
            "--release", str(stale_release_path),
            "--git-head", TEST_HEAD,
            "--require-release-clean-head",
        )

        dirty_pytorch = pytorch_evidence()
        dirty_pytorch["git_dirty"] = True
        dirty_pytorch_path = write_json(directory, "dirty-pytorch.fixture", dirty_pytorch)
        assert_fails(
            "PyTorch evidence must be from a clean git tree",
            "--pytorch", str(dirty_pytorch_path),
            "--git-head", TEST_HEAD,
            "--require-pytorch-clean-head",
        )

        stale_windows = windows_evidence()
        stale_windows["git_head"] = "stale"
        stale_windows_path = write_json(directory, "stale-windows.fixture", stale_windows)
        assert_fails(
            "Windows evidence git_head mismatch",
            "--windows", str(stale_windows_path),
            "--git-head", TEST_HEAD,
            "--require-windows-clean-head",
        )

        dirty_windows = windows_evidence()
        dirty_windows["git_dirty"] = True
        dirty_windows_path = write_json(directory, "dirty-windows.fixture", dirty_windows)
        assert_fails(
            "Windows evidence must be from a clean git tree",
            "--windows", str(dirty_windows_path),
            "--git-head", TEST_HEAD,
            "--require-windows-clean-head",
        )

        stale_windows_cuda = windows_cuda_evidence()
        stale_windows_cuda["git_head"] = "stale"
        stale_windows_cuda_path = write_json(
            directory, "stale-windows-cuda.fixture", stale_windows_cuda
        )
        assert_fails(
            "Windows CUDA evidence git_head mismatch",
            "--windows-cuda", str(stale_windows_cuda_path),
            "--git-head", TEST_HEAD,
            "--require-windows-cuda-clean-head",
        )

        missing_windows_cuda_toolchain = windows_cuda_evidence()
        missing_windows_cuda_toolchain_path = write_json(
            directory,
            "missing-windows-cuda-toolchain.fixture",
            missing_windows_cuda_toolchain,
        )
        assert_fails(
            "--require-toolchain needs nvcc on PATH",
            "--windows-cuda", str(missing_windows_cuda_toolchain_path),
            "--require-windows-cuda-toolchain",
        )

        stale_cuda = cuda_evidence()
        stale_cuda["git_head"] = "stale"
        stale_cuda_path = write_json(directory, "stale-cuda.fixture", stale_cuda)
        assert_fails(
            "CUDA evidence git_head mismatch",
            "--cuda", str(stale_cuda_path),
            "--git-head", TEST_HEAD,
            "--require-cuda-clean-head",
        )

        dirty_cuda = cuda_evidence()
        dirty_cuda["git_dirty"] = True
        dirty_cuda_path = write_json(directory, "dirty-cuda.fixture", dirty_cuda)
        assert_fails(
            "CUDA evidence must be from a clean git tree",
            "--cuda", str(dirty_cuda_path),
            "--git-head", TEST_HEAD,
            "--require-cuda-clean-head",
        )

        stale_hip = hip_evidence()
        stale_hip["git_head"] = "stale"
        stale_hip_path = write_json(directory, "stale-hip.fixture", stale_hip)
        assert_fails(
            "HIP evidence git_head mismatch",
            "--hip", str(stale_hip_path),
            "--git-head", TEST_HEAD,
            "--require-hip-clean-head",
        )

        dirty_hip = hip_evidence()
        dirty_hip["git_dirty"] = True
        dirty_hip_path = write_json(directory, "dirty-hip.fixture", dirty_hip)
        assert_fails(
            "HIP evidence must be from a clean git tree",
            "--hip", str(dirty_hip_path),
            "--git-head", TEST_HEAD,
            "--require-hip-clean-head",
        )

        stale_hip_toolchain = hip_toolchain_evidence()
        stale_hip_toolchain["git_head"] = "stale"
        stale_hip_toolchain_path = write_json(
            directory, "stale-hip-toolchain.fixture", stale_hip_toolchain
        )
        assert_fails(
            "HIP toolchain evidence git_head mismatch",
            "--hip-toolchain", str(stale_hip_toolchain_path),
            "--git-head", TEST_HEAD,
            "--require-hip-toolchain-clean-head",
        )

        missing_spirv = hip_toolchain_evidence()
        missing_spirv["runtime"]["gpu_spirv_device"] = False
        missing_spirv["readiness"].update({
            "gpu_spirv_runtime": False,
            "status": "missing_requirements",
            "missing": ["SPIR-V-capable GPU OpenCL or Level Zero runtime"],
        })
        missing_spirv_path = write_json(
            directory, "missing-hip-spirv-runtime.fixture", missing_spirv
        )
        assert_fails(
            "--require-spirv-runtime needs readiness.gpu_spirv_runtime=true",
            "--hip-toolchain", str(missing_spirv_path),
            "--require-hip-spirv-runtime",
        )

        hip_runtime_unavailable_path = write_json(
            directory,
            "hip-runtime-unavailable.fixture",
            hip_runtime_unavailable_evidence(),
        )
        assert_passes(
            "--hip", str(hip_runtime_unavailable_path),
            "--git-head", TEST_HEAD,
            "--require-hip-build",
            "--require-hip-clean-head",
        )
        assert_fails(
            "--require-hip needs passed evidence",
            "--hip", str(hip_runtime_unavailable_path),
            "--git-head", TEST_HEAD,
            "--require-hip",
        )

        hip_not_built_path = write_json(
            directory,
            "hip-not-built.fixture",
            hip_not_built_evidence(),
        )
        assert_fails(
            "--require-hip-build needs hip_build_enabled=true",
            "--hip", str(hip_not_built_path),
            "--git-head", TEST_HEAD,
            "--require-hip-build",
        )

        brokered = copy.deepcopy(live_mesh_evidence())
        brokered["summary"]["direct_ring_ranks"] = 3
        brokered["ranks"][2]["direct_ring"]["enabled"] = False
        brokered_path = write_json(directory, "brokered.fixture", brokered)
        assert_fails(
            "all ranks must report direct_ring=enabled",
            "--live-mesh", str(brokered_path),
            "--min-live-outer-steps", "5",
            "--require-direct-ring",
        )

        fallback_live = copy.deepcopy(live_mesh_evidence())
        fallback_live["summary"]["cuda_ranks"] = []
        fallback_live["summary"]["all_requested_cuda_ranks_used"] = False
        fallback_live["summary"]["backend_fallback_ranks"] = [3]
        fallback_live["summary"]["rank_backend_summary"][3]["observed_backends"] = [
            "portable_cpu"
        ]
        fallback_live["summary"]["rank_backend_summary"][3]["cuda_fallback"] = True
        for outer in fallback_live["ranks"][3]["outer"]:
            outer["backend"] = "portable_cpu"
        fallback_live_path = write_json(directory, "fallback-live.fixture", fallback_live)
        assert_fails(
            "requested CUDA ranks fell back to another backend",
            "--live-mesh", str(fallback_live_path),
            "--min-live-outer-steps", "5",
            "--require-explicit-backends",
            "--require-no-backend-fallback",
        )

        bad_cuda = copy.deepcopy(cuda_evidence())
        bad_cuda["training_kernels"]["adamw_step_fp16"]["kernel"] = "cpu_fallback"
        bad_cuda_path = write_json(directory, "bad-cuda.fixture", bad_cuda)
        assert_fails(
            "adamw_step_fp16 kernel must be cuda_adamw_step_fp16",
            "--cuda", str(bad_cuda_path),
            "--require-cuda",
        )

    print("operational evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
