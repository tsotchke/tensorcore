#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/run_windows_cuda_probe.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.windows_cuda_probe.evidence.v1"
VALID_STATUSES = {"ready", "driver_only", "admission_blocked", "unavailable"}


def fail(message: str) -> int:
    print(f"Windows CUDA probe evidence invalid: {message}", file=sys.stderr)
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


def require_dict(value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-driver", action="store_true")
    parser.add_argument("--require-toolchain", action="store_true")
    parser.add_argument("--require-admission-clear", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--require-build-smoke", action="store_true")
    parser.add_argument("--require-clean-head", action="store_true")
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")

    if evidence.get("schema") != SCHEMA:
        return fail(f"schema must be {SCHEMA!r}")
    if evidence.get("schema_version") != 1:
        return fail("schema_version must be 1")

    status = evidence.get("runtime_status")
    if status not in VALID_STATUSES:
        return fail(f"unexpected runtime_status={status!r}")

    if args.require_clean_head:
        if not args.git_head:
            return fail("expected git head is unavailable for Windows CUDA evidence check")
        if evidence.get("git_dirty") is not False:
            return fail("Windows CUDA evidence must be from a clean git tree")
        if evidence.get("git_head") != args.git_head:
            return fail(
                "Windows CUDA evidence git_head mismatch: "
                f"{evidence.get('git_head')!r} != {args.git_head!r}"
            )

    host = require_dict(evidence.get("host"), "host")
    if host is None:
        return fail("host must be an object")
    if "Windows" not in str(host.get("os") or ""):
        return fail("host.os must identify Windows")

    nvidia = require_dict(evidence.get("nvidia_smi"), "nvidia_smi")
    if nvidia is None:
        return fail("nvidia_smi must be an object")
    toolkit = require_dict(evidence.get("cuda_toolkit"), "cuda_toolkit")
    if toolkit is None:
        return fail("cuda_toolkit must be an object")
    admission = require_dict(evidence.get("admission"), "admission")
    if admission is None:
        return fail("admission must be an object")
    devices = evidence.get("devices")
    if not isinstance(devices, list):
        return fail("devices must be a list")

    try:
        device_count = int(evidence.get("device_count") or 0)
    except (TypeError, ValueError):
        return fail("device_count must be an integer")
    if device_count != len(devices):
        return fail("device_count must match devices length")

    driver_ok = bool(nvidia.get("found")) and device_count > 0
    toolchain_ok = bool(toolkit.get("nvcc_found"))
    admission_ok = admission.get("ok") is True
    blocked = admission.get("blocked", [])
    if not isinstance(blocked, list):
        return fail("admission.blocked must be a list")
    if admission_ok and blocked:
        return fail("clear admission evidence must not include blocked CUDA processes")
    if admission.get("reason") == "ok_opaque_wddm_rows_no_visible_cuda_processes":
        visible = nvidia.get("visible_processes", [])
        ignored = admission.get("ignored_opaque_wddm", [])
        if not isinstance(visible, list) or visible:
            return fail("opaque WDDM admission needs an empty visible_processes list")
        if not isinstance(ignored, list) or not ignored:
            return fail("opaque WDDM admission needs ignored_opaque_wddm rows")

    if status == "ready" and not (driver_ok and toolchain_ok and admission_ok):
        return fail("ready evidence must include driver, toolchain, and clear admission")
    if status == "driver_only" and not driver_ok:
        return fail("driver_only evidence must include at least one CUDA device")
    if status == "admission_blocked" and not driver_ok:
        return fail("admission_blocked evidence must include at least one CUDA device")
    if status == "admission_blocked" and admission_ok:
        return fail("admission_blocked evidence must have admission.ok=false")
    if status == "unavailable" and driver_ok:
        return fail("unavailable evidence cannot report a CUDA driver/device")

    build_smoke = evidence.get("build_smoke", {"ran": False})
    if not isinstance(build_smoke, dict):
        return fail("build_smoke must be an object when present")
    build_smoke_ok = build_smoke.get("ok") is True
    if build_smoke.get("ran"):
        if build_smoke_ok:
            if status != "ready":
                return fail("passing build_smoke requires runtime_status=ready")
            if build_smoke.get("rc") != 0:
                return fail("passing build_smoke must have rc=0")
            if int(build_smoke.get("tests_failed") or 0) != 0:
                return fail("passing build_smoke must have tests_failed=0")
            if int(build_smoke.get("tests_total") or 0) <= 0:
                return fail("passing build_smoke must report tests_total")
            if build_smoke.get("cuda_gemm_passed") is not True:
                return fail("passing build_smoke must include test_cuda_gemm")
        elif not build_smoke.get("reason"):
            return fail("failed build_smoke must include reason")

    if args.require_driver and not driver_ok:
        return fail("--require-driver needs nvidia_smi and at least one CUDA device")
    if args.require_toolchain and not toolchain_ok:
        return fail("--require-toolchain needs nvcc on PATH")
    if args.require_admission_clear and not admission_ok:
        return fail("--require-admission-clear needs admission.ok=true")
    if args.require_ready and status != "ready":
        return fail(f"--require-ready needs ready evidence, got {status}")
    if args.require_build_smoke and not build_smoke_ok:
        return fail("--require-build-smoke needs passing CUDA configure/build/CTest evidence")

    print(
        "Windows CUDA probe evidence OK: "
        f"status={status} devices={device_count} "
        f"toolchain={toolchain_ok} admission={admission_ok} "
        f"build_smoke={build_smoke_ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
