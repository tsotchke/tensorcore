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
        "files": {
            "bindings/pytorch/tensorcore_torch/__init__.py": {
                "executed_lines": [25],
                "functions": {
                    "_privateuse1_backend_name": {"start_line": 25, "executed_lines": [25]},
                    "_ensure_privateuse1_name": {"start_line": 32, "executed_lines": [32]},
                    "_device_index": {"start_line": 50, "executed_lines": [50]},
                    "_check_device": {"start_line": 58, "executed_lines": [58]},
                    "_torch_backend_module": {"start_line": 65, "executed_lines": [65]},
                    "_torch_backend_module_registered": {"start_line": 72, "executed_lines": [72]},
                    "_new_backend_module": {"start_line": 77, "executed_lines": [77]},
                    "_ensure_generated_methods": {"start_line": 160, "executed_lines": [160]},
                    "_ensure_torch_backend_module": {"start_line": 169, "executed_lines": [169]},
                    "pytorch_backend_registered": {"start_line": 222, "executed_lines": [222]},
                    "pytorch_backend_state": {"start_line": 227, "executed_lines": [227]},
                    "pytorch_backend_report": {"start_line": 302, "executed_lines": [302]},
                },
            },
            "bindings/pytorch/tensorcore_torch_ext.cpp": {
                "executed_lines": [110],
                "functions": {
                    "register_tensorcore_allocator": {"start_line": 110, "executed_lines": [110]},
                    "is_tensorcore_device": {"start_line": 117, "executed_lines": [117]},
                    "is_host_accessible_device_pair": {"start_line": 121, "executed_lines": [121]},
                    "tc_matmul_eligibility_reason": {"start_line": 126, "executed_lines": [126]},
                    "is_tc_matmul_eligible": {"start_line": 140, "executed_lines": [140]},
                    "tc_matmul_eligibility": {"start_line": 177, "executed_lines": [177]},
                    "tc_matmul_fp32": {"start_line": 190, "executed_lines": [190]},
                    "tc_matmul_bf16": {"start_line": 321, "executed_lines": [321]},
                    "tc_last_backend_name": {"start_line": 326, "executed_lines": [326]},
                    "tc_matmul_dispatch": {"start_line": 331, "executed_lines": [331]},
                    "tc_matmul_autograd_cpu": {"start_line": 340, "executed_lines": [340]},
                    "tc_matmul_privateuse1": {"start_line": 352, "executed_lines": [352]},
                    "tc_empty_memory_format": {"start_line": 356, "executed_lines": [356]},
                    "tc_empty_strided": {"start_line": 383, "executed_lines": [383]},
                    "tc_to_tensorcore": {"start_line": 414, "executed_lines": [414]},
                    "tc_to_cpu": {"start_line": 429, "executed_lines": [429]},
                    "tc_set_default_matmul": {"start_line": 440, "executed_lines": [440]},
                    "tc_default_matmul_enabled": {"start_line": 445, "executed_lines": [445]},
                    "tc_privateuse1_backend_name": {"start_line": 449, "executed_lines": [449]},
                },
            },
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

    missing_coverage = copy.deepcopy(good)
    del missing_coverage["files"]["bindings/pytorch/tensorcore_torch_ext.cpp"]["functions"][
        "tc_to_tensorcore"
    ]
    assert_fails(missing_coverage, "missing function coverage")

    assert_fails(
        skipped,
        "--require-backend-allocation needs tensorcore device allocation",
        "--require-backend-allocation",
    )

    print("PyTorch smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
