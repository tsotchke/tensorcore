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
VALID_TOOLCHAIN_STATUSES = {
    "ready_for_hip_gemm",
    "runtime_only_no_hipblas",
    "missing_requirements",
}
VALID_TOOLCHAIN_DIAGNOSTIC_CLASSES = {
    "ready",
    "runtime_only_no_hipblas",
    "no_hip_rocm",
    "diagnostic_blocked",
}
EXPECTED_GEMM_KERNELS = {
    "hip_gemm_sgemm": "hipblas_sgemm_staged",
    "hip_gemm_hgemm": "hipblas_hgemm_staged",
}
REQUIRED_FUNCTIONS = {
    "lib/hip/gemm.cpp": {
        "hip_gemm_sgemm",
        "hip_gemm_hgemm",
    },
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


def expected_required_functions() -> list[str]:
    return sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)


def derived_covered_functions(data: dict) -> list[str]:
    covered = covered_functions(data)
    return sorted(f"{path}:{name}" for path, names in covered.items() for name in names)


def missing_required_functions(data: dict) -> list[str]:
    required = expected_required_functions()
    covered = derived_covered_functions(data)
    return sorted(set(required) - set(covered))


def require_gemm_kernel(evidence: dict, name: str) -> str | None:
    gemm = evidence.get("gemm_kernels")
    if not isinstance(gemm, dict):
        return "passed evidence must include gemm_kernels"
    item = gemm.get(name)
    if not isinstance(item, dict):
        return f"missing HIP GEMM kernel evidence for {name}"
    expected = EXPECTED_GEMM_KERNELS[name]
    if item.get("status") != "passed":
        return f"{name}.status must be passed, got {item.get('status')!r}"
    if item.get("backend") != "hip":
        return f"{name}.backend must be hip"
    if item.get("kernel") != expected:
        return f"{name}.kernel must be {expected}, got {item.get('kernel')!r}"
    return None


def check_toolchain_diagnostic_class(readiness: dict, status: str) -> str | None:
    diagnostic_class = readiness.get("diagnostic_class")
    if diagnostic_class is None:
        return None
    if diagnostic_class not in VALID_TOOLCHAIN_DIAGNOSTIC_CLASSES:
        return (
            "toolchain.readiness.diagnostic_class must be one of "
            f"{sorted(VALID_TOOLCHAIN_DIAGNOSTIC_CLASSES)!r}, got {diagnostic_class!r}"
        )
    install_markers = readiness.get("install_markers")
    if not isinstance(install_markers, list):
        return (
            "toolchain.readiness.install_markers must be a list when "
            "diagnostic_class is present"
        )
    if any(not isinstance(item, str) for item in install_markers):
        return "toolchain.readiness.install_markers entries must be strings"

    if diagnostic_class == "ready" and status != "ready_for_hip_gemm":
        return "toolchain.readiness.diagnostic_class=ready requires status=ready_for_hip_gemm"
    if (
        diagnostic_class == "runtime_only_no_hipblas"
        and status != "runtime_only_no_hipblas"
    ):
        return (
            "toolchain.readiness.diagnostic_class=runtime_only_no_hipblas requires "
            "status=runtime_only_no_hipblas"
        )
    if diagnostic_class == "no_hip_rocm":
        if status != "missing_requirements":
            return "toolchain.readiness.diagnostic_class=no_hip_rocm requires missing_requirements"
        if install_markers:
            return "toolchain.readiness.diagnostic_class=no_hip_rocm requires no install_markers"
        for name in ("hip_runtime_config", "hipcc", "hipblas_config"):
            if readiness.get(name) is not False:
                return (
                    "toolchain.readiness.diagnostic_class=no_hip_rocm requires "
                    f"{name}=false"
                )
    if diagnostic_class == "diagnostic_blocked":
        if status != "missing_requirements":
            return (
                "toolchain.readiness.diagnostic_class=diagnostic_blocked "
                "requires missing_requirements"
            )
        if not install_markers:
            return (
                "toolchain.readiness.diagnostic_class=diagnostic_blocked "
                "requires install_markers"
            )
    return None


def check_summary(evidence: dict) -> str | None:
    summary = evidence.get("summary")
    if not isinstance(summary, dict):
        return "passed evidence must include summary"
    required = expected_required_functions()
    covered = derived_covered_functions(evidence)
    missing = sorted(set(required) - set(covered))
    if summary.get("required_functions") != required:
        return "summary.required_functions must match checker required functions"
    if summary.get("covered_functions") != covered:
        return "summary.covered_functions must match files coverage"
    if summary.get("missing_functions") != missing:
        return "summary.missing_functions must match derived missing functions"
    if missing:
        return f"HIP evidence is missing function coverage: {missing!r}"
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
    parser.add_argument(
        "--require-toolchain",
        action="store_true",
        help="Require embedded chipStar/OpenCL/SPIR-V toolchain evidence.",
    )
    parser.add_argument(
        "--require-ready-toolchain",
        action="store_true",
        help="Require embedded toolchain evidence ready for hipBLAS GEMM.",
    )
    parser.add_argument(
        "--require-toolchain-diagnostic-class",
        choices=sorted(VALID_TOOLCHAIN_DIAGNOSTIC_CLASSES),
        help="Require the embedded HIP/ROCm absence/blocker classification.",
    )
    parser.add_argument("--require-hip-gemm-sgemm", action="store_true")
    parser.add_argument("--require-hip-gemm-hgemm", action="store_true")
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

    toolchain = evidence.get("toolchain")
    toolchain_diagnostic_class = None
    if (
        args.require_toolchain
        or args.require_ready_toolchain
        or args.require_toolchain_diagnostic_class
        or toolchain is not None
    ):
        if not isinstance(toolchain, dict):
            return fail("toolchain evidence must be an object")
        if toolchain.get("schema_version") != 1:
            return fail("toolchain.schema_version must be 1")
        readiness = toolchain.get("readiness")
        if not isinstance(readiness, dict):
            return fail("toolchain.readiness must be an object")
        toolchain_status = readiness.get("status")
        if toolchain_status not in VALID_TOOLCHAIN_STATUSES:
            return fail(f"unexpected toolchain readiness.status={toolchain_status!r}")
        error = check_toolchain_diagnostic_class(readiness, toolchain_status)
        if error:
            return fail(error)
        toolchain_diagnostic_class = readiness.get("diagnostic_class")
        if (
            args.require_toolchain_diagnostic_class
            and toolchain_diagnostic_class != args.require_toolchain_diagnostic_class
        ):
            return fail(
                "--require-toolchain-diagnostic-class needs "
                f"{args.require_toolchain_diagnostic_class}, "
                f"got {toolchain_diagnostic_class!r}"
            )
        if args.require_ready_toolchain and toolchain_status != "ready_for_hip_gemm":
            return fail(
                "--require-ready-toolchain needs ready_for_hip_gemm, "
                f"got {toolchain_status}"
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
        for name in EXPECTED_GEMM_KERNELS:
            error = require_gemm_kernel(evidence, name)
            if error:
                return fail(error)
        error = check_summary(evidence)
        if error:
            return fail(error)

    if args.require_hip_gemm_sgemm:
        error = require_gemm_kernel(evidence, "hip_gemm_sgemm")
        if error:
            return fail(error)

    if args.require_hip_gemm_hgemm:
        error = require_gemm_kernel(evidence, "hip_gemm_hgemm")
        if error:
            return fail(error)

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
        f"devices={int(evidence.get('device_count') or 0)} "
        f"covered={len(derived_covered_functions(evidence))} "
        f"toolchain_diagnostic={toolchain_diagnostic_class or 'unspecified'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
