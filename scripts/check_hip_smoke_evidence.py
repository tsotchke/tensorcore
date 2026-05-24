#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/ci_hip_smoke.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_STATUSES = {
    "passed",
    "runtime_only_no_hipblas",
    "skipped_not_built",
    "skipped_runtime_unavailable",
}


def fail(message: str) -> int:
    print(f"HIP smoke evidence invalid: {message}", file=sys.stderr)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-hip", action="store_true")
    parser.add_argument(
        "--require-hip-build",
        action="store_true",
        help="Require that TC_ENABLE_HIP found the chipStar/HIP runtime target.",
    )
    parser.add_argument("--require-clean-head", action="store_true")
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

    if args.require_hip and status != "passed":
        return fail(f"--require-hip needs passed evidence, got {status}")

    if args.require_hip_build:
        if evidence.get("hip_build_enabled") is not True:
            return fail("--require-hip-build needs hip_build_enabled=true")
        if status == "skipped_not_built":
            return fail("--require-hip-build cannot accept skipped_not_built evidence")

    if args.require_clean_head:
        if not args.git_head:
            return fail("expected git head is unavailable for HIP evidence check")
        if evidence.get("git_dirty") is not False:
            return fail("HIP evidence must be from a clean git tree")
        if evidence.get("git_head") != args.git_head:
            return fail(
                "HIP evidence git_head mismatch: "
                f"{evidence.get('git_head')!r} != {args.git_head!r}"
            )

    if status == "passed":
        if evidence.get("hip_build_enabled") is not True:
            return fail("passed evidence must have hip_build_enabled=true")
        if evidence.get("hip_gemm_enabled") is not True:
            return fail("passed evidence must have hip_gemm_enabled=true")
        if int(evidence.get("device_count") or 0) <= 0:
            return fail("passed evidence must report at least one device")
        if evidence.get("backend") != "hip":
            return fail("passed evidence must report backend=hip")
        if evidence.get("kernel") != "hipblas_sgemm_staged":
            return fail("passed evidence must report hipblas_sgemm_staged")
        if evidence.get("fallback_backend") in (None, "hip"):
            return fail("passed evidence must prove non-HIP fallback when disabled")

    if status == "skipped_not_built" and evidence.get("hip_build_enabled"):
        return fail("skipped_not_built cannot have hip_build_enabled=true")

    if status == "runtime_only_no_hipblas":
        if evidence.get("hip_build_enabled") is not True:
            return fail("runtime_only_no_hipblas requires hip_build_enabled=true")
        if evidence.get("hip_gemm_enabled") is True:
            return fail("runtime_only_no_hipblas cannot have hip_gemm_enabled=true")
        if int(evidence.get("device_count") or 0) <= 0:
            return fail("runtime_only_no_hipblas must report a HIP device")

    print(
        "HIP smoke evidence OK: "
        f"status={status} build={bool(evidence.get('hip_build_enabled'))} "
        f"gemm={bool(evidence.get('hip_gemm_enabled'))} "
        f"devices={int(evidence.get('device_count') or 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
