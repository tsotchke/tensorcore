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
        pytorch_path = write_json(directory, "pytorch.json", pytorch_evidence())

        assert_passes(
            "--release", str(release_path),
            "--sdk26", str(sdk26_path),
            "--cuda", str(cuda_path),
            "--hip", str(hip_path),
            "--pytorch", str(pytorch_path),
            "--live-mesh", str(live_path),
            "--git-head", TEST_HEAD,
            "--require-release",
            "--require-sdk26",
            "--require-cuda",
            "--require-hip",
            "--require-hip-build",
            "--require-pytorch",
            "--require-pytorch-backend-allocation",
            "--require-live-mesh",
            "--require-release-clean-head",
            "--require-sdk26-clean-head",
            "--require-cuda-clean-head",
            "--require-hip-clean-head",
            "--require-pytorch-clean-head",
            "--require-live-clean-head",
            "--min-live-outer-steps", "5",
            "--require-direct-ring",
            "--require-checkpoint",
            "--require-cuda-rank3",
        )

        assert_fails("no evidence paths were provided")
        assert_fails("--live-mesh evidence is required", "--require-live-mesh")
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
