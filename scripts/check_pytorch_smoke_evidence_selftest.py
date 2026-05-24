#!/usr/bin/env python3
"""Fixture tests for the PyTorch smoke evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_pytorch_smoke_evidence.py"


def passed_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": "abc123",
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
            "matmul_dispatch_probe": {
                "eligible": True,
                "reason": "eligible",
            },
            "default_matmul_enabled": False,
            "last_backend_name": "portable_cpu",
            "amp_supported_dtypes": ["torch.float32", "torch.bfloat16"],
        },
        "backend_report": "tensorcore PyTorch backend: registered=True",
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


def skipped_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git_head": "abc123",
        "git_dirty": False,
        "require_pytorch": False,
        "require_pytorch_backend": False,
        "runtime_status": "skipped_torch_unavailable",
        "message": "PyTorch is not importable",
        "torch_version": None,
        "tensorcore_lib_dir": "/tmp/tensorcore",
        "backend_state": None,
        "backend_report": None,
        "matmul": {},
        "direct_device_allocation": {
            "available": False,
            "error": None,
        },
    }


def run_checker(
    evidence: dict[str, Any],
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(evidence, handle)
        path = pathlib.Path(handle.name)
    try:
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), *extra_args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_passes(evidence: dict[str, Any], *extra_args: str) -> None:
    result = run_checker(evidence, *extra_args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(evidence: dict[str, Any], needle: str, *extra_args: str) -> None:
    result = run_checker(evidence, *extra_args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    good = passed_evidence()
    assert_passes(good)
    assert_passes(good, "--require-pytorch")
    assert_passes(good, "--require-backend-allocation")

    skipped = skipped_evidence()
    assert_passes(skipped)
    assert_fails(skipped, "--require-pytorch needs passed evidence", "--require-pytorch")

    bad_registration = copy.deepcopy(good)
    bad_registration["backend_state"]["registered"] = False
    assert_fails(bad_registration, "backend_state.registered must be true")

    bad_matmul = copy.deepcopy(good)
    bad_matmul["matmul"]["bf16_checked"] = False
    assert_fails(bad_matmul, "bf16_checked must be true")

    bad_allocation = copy.deepcopy(good)
    bad_allocation["direct_device_allocation"]["available"] = False
    assert_fails(bad_allocation, "available allocator_status requires allocation evidence")

    bad_roundtrip = copy.deepcopy(good)
    bad_roundtrip["matmul"]["device_roundtrip_checked"] = False
    assert_fails(bad_roundtrip, "device_roundtrip_checked must be true")
    assert_fails(
        skipped,
        "--require-backend-allocation needs tensorcore device allocation",
        "--require-backend-allocation",
    )

    print("PyTorch smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
