#!/usr/bin/env python3
"""Fixture tests for the CUDA smoke evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

from check_cuda_smoke_evidence import EXPECTED_TRAINING_KERNELS


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_cuda_smoke_evidence.py"
TEST_HEAD = "abc123"


def cuda_evidence() -> dict[str, Any]:
    training = {
        op: {"backend": "cuda", "kernel": kernel}
        for op, kernel in EXPECTED_TRAINING_KERNELS.items()
    }
    return {
        "schema_version": 1,
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "runtime_status": "passed",
        "cuda_build_enabled": True,
        "require_cuda": True,
        "device_count": 1,
        "device": {
            "device_name": "RTX test",
            "supports_bf16": True,
            "supports_int8_tensor_core": True,
        },
        "backend": "cuda",
        "f32_kernel": "cublas_sgemm_managed",
        "f16_kernel": "cublas_gemmex_fp16_tensorop_managed",
        "gemm_kernels": {
            "cuda_gemm_sgemm": {
                "status": "passed",
                "backend": "cuda",
                "kernel": "cublas_sgemm_managed",
            },
            "cuda_gemm_hgemm": {
                "status": "passed",
                "backend": "cuda",
                "kernel": "cublas_gemmex_fp16_tensorop_managed",
            },
            "cuda_gemm_bf16": {
                "status": "passed",
                "backend": "cuda",
                "kernel": "cublas_gemmex_bf16_tensorop_managed",
            },
            "cuda_gemm_i8": {
                "status": "passed",
                "backend": "cuda",
                "kernel": "cublas_gemmex_i8_tensorop_managed",
            },
        },
        "fallback_backend": "portable_cpu",
        "training_kernels": training,
        "files": {
            "lib/cuda/gemm.cpp": {
                "executed_lines": [116, 190, 290, 374],
                "functions": {
                    "cuda_gemm_sgemm": {"start_line": 116, "executed_lines": [116]},
                    "cuda_gemm_hgemm": {"start_line": 190, "executed_lines": [190]},
                    "cuda_gemm_bf16": {"start_line": 290, "executed_lines": [290]},
                    "cuda_gemm_i8": {"start_line": 374, "executed_lines": [374]},
                },
            },
            "lib/cuda/training.cu": {
                "executed_lines": [128, 156],
                "functions": {
                    "adamw_step_fp32_kernel": {"start_line": 128, "executed_lines": [128]},
                    "adamw_step_fp16_kernel": {"start_line": 156, "executed_lines": [156]},
                },
            },
        },
        "summary": {
            "required_functions": [
                "lib/cuda/gemm.cpp:cuda_gemm_bf16",
                "lib/cuda/gemm.cpp:cuda_gemm_hgemm",
                "lib/cuda/gemm.cpp:cuda_gemm_i8",
                "lib/cuda/gemm.cpp:cuda_gemm_sgemm",
                "lib/cuda/training.cu:adamw_step_fp16_kernel",
                "lib/cuda/training.cu:adamw_step_fp32_kernel",
            ],
            "covered_functions": [
                "lib/cuda/gemm.cpp:cuda_gemm_bf16",
                "lib/cuda/gemm.cpp:cuda_gemm_hgemm",
                "lib/cuda/gemm.cpp:cuda_gemm_i8",
                "lib/cuda/gemm.cpp:cuda_gemm_sgemm",
                "lib/cuda/training.cu:adamw_step_fp16_kernel",
                "lib/cuda/training.cu:adamw_step_fp32_kernel",
            ],
            "missing_functions": [],
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
        good = write_json(directory, "cuda.json", cuda_evidence())
        assert_passes(good)
        assert_passes(
            good,
            "--git-head", TEST_HEAD,
            "--require-cuda",
            "--require-clean-head",
            "--require-cuda-gemm-sgemm",
            "--require-cuda-gemm-hgemm",
            "--require-cuda-gemm-bf16",
            "--require-cuda-gemm-i8",
            "--require-cuda-adamw",
        )

        no_bf16 = copy.deepcopy(cuda_evidence())
        no_bf16["gemm_kernels"]["cuda_gemm_bf16"]["status"] = "skipped_unsupported"
        no_bf16_path = write_json(directory, "no-bf16.json", no_bf16)
        assert_fails(
            no_bf16_path,
            "cuda_gemm_bf16.status must be passed",
            "--require-cuda-gemm-bf16",
        )

        no_adamw = copy.deepcopy(cuda_evidence())
        del no_adamw["training_kernels"]["adamw_step_fp16"]
        no_adamw_path = write_json(directory, "no-adamw.json", no_adamw)
        assert_fails(no_adamw_path, "missing training kernel evidence", "--require-cuda-adamw")

        dirty = copy.deepcopy(cuda_evidence())
        dirty["git_dirty"] = True
        dirty_path = write_json(directory, "dirty.json", dirty)
        assert_fails(dirty_path, "clean git tree", "--git-head", TEST_HEAD, "--require-clean-head")

    print("CUDA smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
