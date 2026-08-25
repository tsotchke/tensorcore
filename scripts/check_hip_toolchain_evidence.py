#!/usr/bin/env python3
"""Validate evidence from scripts/probe_hip_toolchain.py."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_STATUSES = {
    "ready_for_hip_gemm",
    "runtime_only_no_hipblas",
    "missing_requirements",
}
VALID_DIAGNOSTIC_CLASSES = {
    "ready",
    "runtime_only_no_hipblas",
    "no_hip_rocm",
    "diagnostic_blocked",
}


def fail(message: str) -> int:
    print(f"HIP toolchain evidence invalid: {message}", file=sys.stderr)
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


def require_bool(readiness: dict[str, Any], name: str, flag: str) -> str | None:
    if readiness.get(name) is not True:
        return f"{flag} needs readiness.{name}=true"
    return None


def check_diagnostic_class(readiness: dict[str, Any], status: str) -> str | None:
    diagnostic_class = readiness.get("diagnostic_class")
    if diagnostic_class is None:
        return None
    if diagnostic_class not in VALID_DIAGNOSTIC_CLASSES:
        return (
            "readiness.diagnostic_class must be one of "
            f"{sorted(VALID_DIAGNOSTIC_CLASSES)!r}, got {diagnostic_class!r}"
        )
    install_markers = readiness.get("install_markers")
    if not isinstance(install_markers, list):
        return "readiness.install_markers must be a list when diagnostic_class is present"
    if any(not isinstance(item, str) for item in install_markers):
        return "readiness.install_markers entries must be strings"

    if diagnostic_class == "ready" and status != "ready_for_hip_gemm":
        return "readiness.diagnostic_class=ready requires status=ready_for_hip_gemm"
    if (
        diagnostic_class == "runtime_only_no_hipblas"
        and status != "runtime_only_no_hipblas"
    ):
        return (
            "readiness.diagnostic_class=runtime_only_no_hipblas requires "
            "status=runtime_only_no_hipblas"
        )
    if diagnostic_class == "no_hip_rocm":
        if status != "missing_requirements":
            return "readiness.diagnostic_class=no_hip_rocm requires missing_requirements"
        if install_markers:
            return "readiness.diagnostic_class=no_hip_rocm requires no install_markers"
        for name in ("hip_runtime_config", "hipcc", "hipblas_config"):
            if readiness.get(name) is not False:
                return f"readiness.diagnostic_class=no_hip_rocm requires {name}=false"
    if diagnostic_class == "diagnostic_blocked":
        if status != "missing_requirements":
            return "readiness.diagnostic_class=diagnostic_blocked requires missing_requirements"
        if not install_markers:
            return "readiness.diagnostic_class=diagnostic_blocked requires install_markers"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-clean-head", action="store_true")
    parser.add_argument("--require-build-toolchain", action="store_true")
    parser.add_argument("--require-spirv-runtime", action="store_true")
    parser.add_argument("--require-hipblas", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--require-diagnostic-class",
        choices=sorted(VALID_DIAGNOSTIC_CLASSES),
        help="Require the HIP/ROCm absence/blocker classification emitted by the probe.",
    )
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")

    if evidence.get("schema_version") != 1:
        return fail("schema_version must be 1")
    readiness = evidence.get("readiness")
    if not isinstance(readiness, dict):
        return fail("readiness must be an object")
    status = readiness.get("status")
    if status not in VALID_STATUSES:
        return fail(f"unexpected readiness.status={status!r}")
    missing = readiness.get("missing")
    if not isinstance(missing, list):
        return fail("readiness.missing must be a list")
    error = check_diagnostic_class(readiness, status)
    if error:
        return fail(error)
    diagnostic_class = readiness.get("diagnostic_class")
    if args.require_diagnostic_class and diagnostic_class != args.require_diagnostic_class:
        return fail(
            "--require-diagnostic-class needs "
            f"{args.require_diagnostic_class}, got {diagnostic_class!r}"
        )

    tools = evidence.get("tools")
    if not isinstance(tools, dict):
        return fail("tools must be an object")
    packages = evidence.get("cmake_packages")
    if not isinstance(packages, dict):
        return fail("cmake_packages must be an object")
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict):
        return fail("runtime must be an object")

    if args.require_clean_head:
        if not args.git_head:
            return fail("expected git head is unavailable for HIP toolchain evidence check")
        if evidence.get("git_dirty") is not False:
            return fail("HIP toolchain evidence must be from a clean git tree")
        if evidence.get("git_head") != args.git_head:
            return fail(
                "HIP toolchain evidence git_head mismatch: "
                f"{evidence.get('git_head')!r} != {args.git_head!r}"
            )

    errors: list[str] = []
    if args.require_build_toolchain:
        for name in ("hip_runtime_config", "hipcc"):
            error = require_bool(readiness, name, "--require-build-toolchain")
            if error:
                errors.append(error)
    if args.require_spirv_runtime:
        for name in ("spirv_translator", "gpu_spirv_runtime"):
            error = require_bool(readiness, name, "--require-spirv-runtime")
            if error:
                errors.append(error)
    if args.require_hipblas:
        error = require_bool(readiness, "hipblas_config", "--require-hipblas")
        if error:
            errors.append(error)
    if args.require_ready and status != "ready_for_hip_gemm":
        errors.append(f"--require-ready needs ready_for_hip_gemm, got {status}")
    if args.require_ready:
        for name in (
            "hip_runtime_config", "hipcc", "spirv_translator",
            "gpu_spirv_runtime", "hipblas_config",
        ):
            error = require_bool(readiness, name, "--require-ready")
            if error:
                errors.append(error)
    if errors:
        return fail("; ".join(errors))

    print(
        "HIP toolchain evidence OK: "
        f"status={status} diagnostic={diagnostic_class or 'unspecified'} "
        f"missing={','.join(str(item) for item in missing) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
