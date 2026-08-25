#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/ci_cuda_smoke.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_STATUSES = {
    "passed",
    "skipped_not_built",
    "skipped_runtime_unavailable",
}

EXPECTED_TRAINING_KERNELS = {
    "rmsnorm_forward": "cuda_rmsnorm_forward",
    "rmsnorm_backward": "cuda_rmsnorm_backward",
    "layernorm_forward": "cuda_layernorm_forward",
    "layernorm_backward": "cuda_layernorm_backward",
    "swiglu_forward": "cuda_swiglu_forward",
    "swiglu_backward": "cuda_swiglu_backward",
    "softmax_forward": "cuda_softmax_forward",
    "softmax_backward": "cuda_softmax_backward",
    "rope_forward": "cuda_rope_forward",
    "rope_backward": "cuda_rope_backward",
    "adamw_step_fp32": "cuda_adamw_step_fp32",
    "adamw_step_fp16": "cuda_adamw_step_fp16",
}
EXPECTED_GEMM_KERNELS = {
    "cuda_gemm_sgemm": "cublas_sgemm_managed",
    "cuda_gemm_hgemm": "cublas_gemmex_fp16_tensorop_managed",
    "cuda_gemm_bf16": "cublas_gemmex_bf16_tensorop_managed",
    "cuda_gemm_i8": "cublas_gemmex_i8_tensorop_managed",
}
REQUIRED_FUNCTIONS = {
    "lib/cuda/gemm.cpp": {
        "cuda_gemm_sgemm",
        "cuda_gemm_hgemm",
        "cuda_gemm_bf16",
        "cuda_gemm_i8",
    },
    "lib/cuda/training.cu": {
        "adamw_step_fp32_kernel",
        "adamw_step_fp16_kernel",
        "block_reduce_sum_f32",
    },
}


def fail(message: str) -> int:
    print(f"CUDA smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def covered_functions(data: dict) -> dict[str, set[str]]:
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    covered: dict[str, set[str]] = {}
    for rel_path, entry in files.items():
        if not isinstance(entry, dict):
            continue
        functions = entry.get("functions")
        if isinstance(functions, dict):
            covered[str(rel_path)] = {str(name) for name in functions}
    return covered


def derived_covered_functions(data: dict) -> list[str]:
    covered = covered_functions(data)
    return sorted(f"{path}:{name}" for path, names in covered.items() for name in names)


def require_gemm_kernel(evidence: dict, name: str) -> str | None:
    gemm = evidence.get("gemm_kernels")
    if not isinstance(gemm, dict):
        return "passed evidence must include gemm_kernels"
    item = gemm.get(name)
    if not isinstance(item, dict):
        return f"missing CUDA GEMM kernel evidence for {name}"
    expected = EXPECTED_GEMM_KERNELS[name]
    if item.get("status") != "passed":
        return f"{name}.status must be passed, got {item.get('status')!r}"
    if item.get("backend") != "cuda":
        return f"{name}.backend must be cuda"
    if item.get("kernel") != expected:
        return f"{name}.kernel must be {expected}, got {item.get('kernel')!r}"
    return None


def check_summary(evidence: dict, required_functions: list[str]) -> str | None:
    summary = evidence.get("summary")
    if not isinstance(summary, dict):
        return "passed evidence must include summary"
    covered = derived_covered_functions(evidence)
    missing = sorted(set(required_functions) - set(covered))
    if summary.get("covered_functions") != covered:
        return "summary.covered_functions must match files coverage"
    if summary.get("missing_functions") != sorted(
        set(summary.get("required_functions") or []) - set(covered)
    ):
        return "summary.missing_functions must match derived missing functions"
    if missing:
        return f"CUDA evidence is missing function coverage: {missing!r}"
    return None


def cuda_required_functions(evidence: dict) -> list[str]:
    required = [
        "lib/cuda/gemm.cpp:cuda_gemm_sgemm",
        "lib/cuda/gemm.cpp:cuda_gemm_hgemm",
        "lib/cuda/training.cu:adamw_step_fp16_kernel",
        "lib/cuda/training.cu:adamw_step_fp32_kernel",
        "lib/cuda/training.cu:block_reduce_sum_f32",
    ]
    device = evidence.get("device")
    if isinstance(device, dict) and device.get("supports_bf16"):
        required.append("lib/cuda/gemm.cpp:cuda_gemm_bf16")
    if isinstance(device, dict) and device.get("supports_int8_tensor_core"):
        required.append("lib/cuda/gemm.cpp:cuda_gemm_i8")
    return sorted(required)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-clean-head", action="store_true")
    parser.add_argument("--require-cuda-gemm-sgemm", action="store_true")
    parser.add_argument("--require-cuda-gemm-hgemm", action="store_true")
    parser.add_argument("--require-cuda-gemm-bf16", action="store_true")
    parser.add_argument("--require-cuda-gemm-i8", action="store_true")
    parser.add_argument("--require-cuda-adamw", action="store_true")
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")

    if evidence.get("schema_version") != 1:
        return fail("schema_version must be 1")
    status = evidence.get("runtime_status")
    if status not in VALID_STATUSES:
        return fail(f"unexpected runtime_status={status!r}")

    if args.require_cuda and status != "passed":
        return fail(f"--require-cuda needs passed evidence, got {status}")

    if args.require_clean_head:
        if not args.git_head:
            return fail("expected git head is unavailable for CUDA evidence check")
        if evidence.get("git_dirty") is not False:
            return fail("CUDA evidence must be from a clean git tree")
        if evidence.get("git_head") != args.git_head:
            return fail(
                "CUDA evidence git_head mismatch: "
                f"{evidence.get('git_head')!r} != {args.git_head!r}"
            )

    if status == "passed":
        if evidence.get("cuda_build_enabled") is not True:
            return fail("passed evidence must have cuda_build_enabled=true")
        if int(evidence.get("device_count") or 0) <= 0:
            return fail("passed evidence must report at least one device")
        if evidence.get("backend") != "cuda":
            return fail("passed evidence must report backend=cuda")
        if evidence.get("f32_kernel") != "cublas_sgemm_managed":
            return fail("passed evidence must report cublas_sgemm_managed")
        if evidence.get("f16_kernel") != "cublas_gemmex_fp16_tensorop_managed":
            return fail("passed evidence must report fp16 tensor-op cuBLAS")
        if evidence.get("fallback_backend") in (None, "cuda"):
            return fail("passed evidence must prove non-CUDA fallback when disabled")
        for name in ("cuda_gemm_sgemm", "cuda_gemm_hgemm"):
            error = require_gemm_kernel(evidence, name)
            if error:
                return fail(error)
        device = evidence.get("device")
        if isinstance(device, dict) and device.get("supports_bf16"):
            error = require_gemm_kernel(evidence, "cuda_gemm_bf16")
            if error:
                return fail(error)
        if isinstance(device, dict) and device.get("supports_int8_tensor_core"):
            error = require_gemm_kernel(evidence, "cuda_gemm_i8")
            if error:
                return fail(error)

        training = evidence.get("training_kernels")
        if not isinstance(training, dict):
            return fail("passed evidence must include training_kernels")
        for op, kernel in EXPECTED_TRAINING_KERNELS.items():
            got = training.get(op)
            if not isinstance(got, dict):
                return fail(f"missing training kernel evidence for {op}")
            if got.get("backend") != "cuda":
                return fail(f"{op} backend must be cuda")
            if got.get("kernel") != kernel:
                return fail(f"{op} kernel must be {kernel}, got {got.get('kernel')!r}")
        error = check_summary(evidence, cuda_required_functions(evidence))
        if error:
            return fail(error)

    for flag, name in (
        (args.require_cuda_gemm_sgemm, "cuda_gemm_sgemm"),
        (args.require_cuda_gemm_hgemm, "cuda_gemm_hgemm"),
        (args.require_cuda_gemm_bf16, "cuda_gemm_bf16"),
        (args.require_cuda_gemm_i8, "cuda_gemm_i8"),
    ):
        if flag:
            error = require_gemm_kernel(evidence, name)
            if error:
                return fail(error)

    if args.require_cuda_adamw:
        training = evidence.get("training_kernels")
        if not isinstance(training, dict):
            return fail("--require-cuda-adamw needs training_kernels")
        for op in ("adamw_step_fp32", "adamw_step_fp16"):
            expected = EXPECTED_TRAINING_KERNELS[op]
            got = training.get(op)
            if not isinstance(got, dict):
                return fail(f"missing training kernel evidence for {op}")
            if got.get("backend") != "cuda" or got.get("kernel") != expected:
                return fail(f"{op} must use {expected}")

    if status == "skipped_not_built" and evidence.get("cuda_build_enabled"):
        return fail("skipped_not_built cannot have cuda_build_enabled=true")

    print(
        "CUDA smoke evidence OK: "
        f"status={status} build={bool(evidence.get('cuda_build_enabled'))} "
        f"devices={int(evidence.get('device_count') or 0)} "
        f"training_ops={len(evidence.get('training_kernels') or {})} "
        f"covered={len(derived_covered_functions(evidence))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
