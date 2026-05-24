#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/ci_cuda_smoke.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


VALID_STATUSES = {
    "passed",
    "skipped_not_built",
    "skipped_runtime_unavailable",
}

EXPECTED_TRAINING_KERNELS = {
    "rmsnorm_forward": "cuda_rmsnorm_forward",
    "rmsnorm_backward": "cuda_rmsnorm_backward",
    "swiglu_forward": "cuda_swiglu_forward",
    "swiglu_backward": "cuda_swiglu_backward",
    "softmax_forward": "cuda_softmax_forward",
    "softmax_backward": "cuda_softmax_backward",
    "adamw_step_fp32": "cuda_adamw_step_fp32",
    "adamw_step_fp16": "cuda_adamw_step_fp16",
}


def fail(message: str) -> int:
    print(f"CUDA smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--require-cuda", action="store_true")
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

    if status == "skipped_not_built" and evidence.get("cuda_build_enabled"):
        return fail("skipped_not_built cannot have cuda_build_enabled=true")

    print(
        "CUDA smoke evidence OK: "
        f"status={status} build={bool(evidence.get('cuda_build_enabled'))} "
        f"devices={int(evidence.get('device_count') or 0)} "
        f"training_ops={len(evidence.get('training_kernels') or {})}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
