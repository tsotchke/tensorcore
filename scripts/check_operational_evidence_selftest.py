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
from check_cuda_smoke_evidence import EXPECTED_TRAINING_KERNELS


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_operational_evidence.py"
TEST_HEAD = "abc123"


def live_mesh_evidence() -> dict[str, Any]:
    evidence = live_mesh_fixture.base_evidence()
    evidence["meta"]["git_head"] = TEST_HEAD
    evidence["meta"]["git_dirty"] = False
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
        cuda_path = write_json(directory, "cuda.json", cuda_evidence())
        pytorch_path = write_json(directory, "pytorch.json", pytorch_evidence())

        assert_passes(
            "--cuda", str(cuda_path),
            "--pytorch", str(pytorch_path),
            "--live-mesh", str(live_path),
            "--git-head", TEST_HEAD,
            "--require-cuda",
            "--require-pytorch",
            "--require-pytorch-backend-allocation",
            "--require-live-mesh",
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
        dirty_path = write_json(directory, "dirty.json", dirty)
        assert_fails(
            "live mesh evidence must be from a clean git tree",
            "--live-mesh", str(dirty_path),
            "--git-head", TEST_HEAD,
            "--require-live-clean-head",
        )

        stale = copy.deepcopy(live_mesh_evidence())
        stale["meta"]["git_head"] = "stale"
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails(
            "live mesh evidence git_head mismatch",
            "--live-mesh", str(stale_path),
            "--git-head", TEST_HEAD,
            "--require-live-clean-head",
        )

        brokered = copy.deepcopy(live_mesh_evidence())
        brokered["summary"]["direct_ring_ranks"] = 3
        brokered["ranks"][2]["direct_ring"]["enabled"] = False
        brokered_path = write_json(directory, "brokered.json", brokered)
        assert_fails(
            "all ranks must report direct_ring=enabled",
            "--live-mesh", str(brokered_path),
            "--min-live-outer-steps", "5",
            "--require-direct-ring",
        )

        bad_cuda = copy.deepcopy(cuda_evidence())
        bad_cuda["training_kernels"]["adamw_step_fp16"]["kernel"] = "cpu_fallback"
        bad_cuda_path = write_json(directory, "bad-cuda.json", bad_cuda)
        assert_fails(
            "adamw_step_fp16 kernel must be cuda_adamw_step_fp16",
            "--cuda", str(bad_cuda_path),
            "--require-cuda",
        )

    print("operational evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
