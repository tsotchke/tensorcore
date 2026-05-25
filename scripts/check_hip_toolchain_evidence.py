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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-clean-head", action="store_true")
    parser.add_argument("--require-build-toolchain", action="store_true")
    parser.add_argument("--require-spirv-runtime", action="store_true")
    parser.add_argument("--require-hipblas", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
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
        f"status={status} missing={','.join(str(item) for item in missing) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
