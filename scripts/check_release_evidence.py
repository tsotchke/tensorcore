#!/usr/bin/env python3
"""Validate release smoke runtime evidence as a public integration contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.release_smoke.runtime_evidence.v1"
FORMAT_VERSION = 3
MISSING = object()

BASE_REQUIRED_TRUE = (
    ("summary.tests_passed", "summary must record passing tests"),
    ("summary.wheel_tag_inspected", "summary must record wheel tag inspection"),
    ("summary.installed_wheel_smoke_passed", "summary must record installed wheel smoke"),
    ("summary.cmake_consumers_passed", "summary must record CMake consumer coverage"),
    ("summary.pkg_config_consumer_passed", "summary must record pkg-config consumer coverage"),
    ("summary.packaging_and_consumers_passed", "summary must record package/consumer coverage"),
    ("summary.public_headers_passed", "summary must record public header coverage"),
    ("summary.python_ffi_surface_passed", "summary must record Python FFI coverage"),
    ("summary.python_constants_passed", "summary must record Python constant coverage"),
    ("summary.python_abi_layout_passed", "summary must record Python ABI coverage"),
    ("checks.tests.passed", "tests check must pass"),
    ("checks.wheel_tag.inspected", "wheel tag check must pass"),
    ("checks.installed_wheel_smoke.passed", "installed wheel smoke must pass"),
    ("checks.consumers.cmake.passed", "CMake consumer check must pass"),
    ("checks.consumers.pkg_config.passed", "pkg-config consumer check must pass"),
    ("checks.packaging_and_consumers.runtime_covered", "package/consumer runtime coverage must pass"),
    ("checks.public_headers.passed", "public headers check must pass"),
    ("checks.python_ffi_surface.passed", "Python FFI surface check must pass"),
    ("checks.python_constants.passed", "Python constants check must pass"),
    ("checks.python_abi_layout.passed", "Python ABI layout check must pass"),
)

GPU_REQUIRED_TRUE = (
    ("summary.public_core_paths_passed", "summary must record public core path coverage"),
    ("summary.public_integration_runtime_passed", "summary must record public integration coverage"),
    ("summary.autotune_cache_passed", "summary must record autotune cache coverage"),
    ("summary.gemm_128_tile_passed", "summary must record 128-tile GEMM coverage"),
    ("summary.gemm_async_passed", "summary must record async GEMM coverage"),
    ("checks.public_core_paths.runtime_covered", "public core paths must be covered on GPU runs"),
    ("checks.public_integration.runtime_covered", "public integration must be covered on GPU runs"),
    ("checks.autotune_cache.passed", "autotune cache must pass on GPU runs"),
    ("checks.gemm_env_variants.use_128_tile.passed", "128-tile GEMM variant must pass on GPU runs"),
    ("checks.gemm_env_variants.use_async.passed", "async GEMM variant must pass on GPU runs"),
)

METAL4_REQUIRED_TRUE = (
    ("summary.metal4_tensorops_compile_passed", "summary must record Metal 4 TensorOps compilation"),
    ("summary.metal4_tensorops_runtime_passed", "summary must record Metal 4 TensorOps runtime"),
    ("checks.metal4_tensorops.runtime_covered", "Metal 4 TensorOps runtime must be covered"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tensorcore release_smoke runtime evidence."
    )
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument(
        "--require-clean-head",
        action="store_true",
        help="Require evidence from the expected clean git head.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require a production Apple GPU full-test run.",
    )
    parser.add_argument(
        "--require-metal4-tensorops",
        action="store_true",
        help="Require SDK26+ Metal 4 TensorOps compile and M5+ runtime coverage.",
    )
    parser.add_argument(
        "--require-metal4-compile",
        action="store_true",
        help="Require SDK26+ Metal 4 TensorOps compile evidence without M5 runtime coverage.",
    )
    return parser.parse_args()


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


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read release evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"release evidence is not valid JSON: {exc}") from exc


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def require_equal(errors: list[str], data: Any, path: str, expected: Any, message: str) -> None:
    actual = get_path(data, path)
    if actual != expected:
        errors.append(f"{message}: expected {path}={expected!r}, got {actual!r}")


def require_true(errors: list[str], data: Any, path: str, message: str) -> None:
    require_equal(errors, data, path, True, message)


def require_status(
    errors: list[str],
    data: Any,
    path: str,
    expected: str,
    message: str,
) -> None:
    require_equal(errors, data, path, expected, message)


def collect_failed_statuses(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    failed: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else key
            if key.endswith("status") and isinstance(child, str) and child.startswith("failed"):
                failed.append((child_path, child))
            failed.extend(collect_failed_statuses(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failed.extend(collect_failed_statuses(child, f"{prefix}[{index}]"))
    return failed


def check_required_true(errors: list[str], data: Any, checks: tuple[tuple[str, str], ...]) -> None:
    for path, message in checks:
        require_true(errors, data, path, message)


def check_no_gpu_consistency(errors: list[str], data: Any) -> None:
    skipped_paths = (
        ("checks.public_core_paths.runtime_status", "public core path status"),
        ("checks.public_integration.runtime_status", "public integration status"),
        ("checks.autotune_cache.status", "autotune cache status"),
        ("checks.gemm_env_variants.use_128_tile.status", "128-tile GEMM status"),
        ("checks.gemm_env_variants.use_async.status", "async GEMM status"),
    )
    for path, label in skipped_paths:
        actual = get_path(data, path)
        if actual not in ("skipped_no_gpu", "skipped_paravirtual_gpu"):
            errors.append(f"{label} must be skipped on no-GPU evidence, got {actual!r}")


def require_string_list(errors: list[str], data: Any, path: str) -> list[str] | None:
    actual = get_path(data, path)
    if not isinstance(actual, list):
        errors.append(f"{path} must be a list, got {actual!r}")
        return None
    non_strings = [item for item in actual if not isinstance(item, str)]
    if non_strings:
        errors.append(f"{path} must contain only strings, got {non_strings!r}")
        return None
    return sorted(actual)


def check_public_core_files(errors: list[str], data: Any) -> None:
    public_core = get_path(data, "checks.public_core_paths")
    files = get_path(data, "files")
    if not isinstance(public_core, dict) or not isinstance(files, dict):
        errors.append("checks.public_core_paths and files must be objects")
        return

    required = require_string_list(errors, data, "checks.public_core_paths.required_files")
    missing = require_string_list(errors, data, "checks.public_core_paths.missing_files")
    uncovered = require_string_list(errors, data, "checks.public_core_paths.uncovered_files")
    if required is None or missing is None or uncovered is None:
        return

    runtime_covered = public_core.get("runtime_covered")
    gpu_available = get_path(data, "checks.tests.gpu_device_available")
    if not isinstance(runtime_covered, bool):
        errors.append(f"checks.public_core_paths.runtime_covered must be boolean, got {runtime_covered!r}")
        return
    if not isinstance(gpu_available, bool):
        errors.append(f"checks.tests.gpu_device_available must be boolean, got {gpu_available!r}")
        return

    computed_uncovered = sorted(path for path in required if path not in files)
    if uncovered != computed_uncovered:
        errors.append(
            "checks.public_core_paths.uncovered_files must match required files absent from "
            f"files: expected {computed_uncovered!r}, got {uncovered!r}"
        )

    if runtime_covered:
        if missing:
            errors.append(
                "public core paths are marked covered but missing_files is not empty: "
                f"{missing!r}"
            )
        if uncovered:
            errors.append(
                "public core paths are marked covered but uncovered_files is not empty: "
                f"{uncovered!r}"
            )
    elif gpu_available:
        if missing != computed_uncovered:
            errors.append(
                "GPU evidence with incomplete public core coverage must record true missing "
                f"files in missing_files: expected {computed_uncovered!r}, got {missing!r}"
            )
    elif missing:
        errors.append(
            "no-GPU/paravirtual evidence must not report runtime coverage gaps as "
            f"missing files: {missing!r}"
        )


def check_metal4_consistency(
    errors: list[str],
    data: Any,
    require_metal4_runtime: bool,
    require_metal4_compile: bool,
) -> None:
    compile_status = get_path(data, "checks.metal4_tensorops.compile_status")
    runtime_compile_status = get_path(data, "checks.metal4_tensorops.runtime_compile_status")
    runtime_status = get_path(data, "checks.metal4_tensorops.runtime_status")
    runtime_covered = get_path(data, "checks.metal4_tensorops.runtime_covered")
    summary_compile = get_path(data, "summary.metal4_tensorops_compile_passed")
    summary_runtime = get_path(data, "summary.metal4_tensorops_runtime_passed")

    if runtime_compile_status != compile_status:
        errors.append(
            "Metal 4 TensorOps compile status mismatch: "
            f"compile_status={compile_status!r}, runtime_compile_status={runtime_compile_status!r}"
        )

    if runtime_covered is True and runtime_status != "passed":
        errors.append(
            "Metal 4 TensorOps runtime is marked covered but runtime_status "
            f"is {runtime_status!r}"
        )
    if summary_compile != (compile_status == "compiled"):
        errors.append(
            "summary.metal4_tensorops_compile_passed does not match compile_status "
            f"{compile_status!r}"
        )
    if summary_runtime != (runtime_status == "passed"):
        errors.append(
            "summary.metal4_tensorops_runtime_passed does not match runtime_status "
            f"{runtime_status!r}"
        )

    if require_metal4_compile or require_metal4_runtime:
        require_true(
            errors,
            data,
            "summary.metal4_tensorops_compile_passed",
            "summary must record Metal 4 TensorOps compilation",
        )
        require_status(
            errors,
            data,
            "checks.metal4_tensorops.compile_status",
            "compiled",
            "Metal 4 TensorOps sources must compile with SDK26+",
        )

    if require_metal4_runtime:
        check_required_true(errors, data, METAL4_REQUIRED_TRUE)
        require_status(
            errors,
            data,
            "checks.metal4_tensorops.runtime_status",
            "passed",
            "Metal 4 TensorOps runtime probe must pass on M5+",
        )


def check_git_provenance(errors: list[str], data: Any, expected_head: str | None) -> None:
    if not expected_head:
        errors.append("expected git head is unavailable for release evidence check")
        return
    if get_path(data, "meta.git_dirty") is not False:
        errors.append("release evidence must be from a clean git tree")
    actual_head = get_path(data, "meta.git_head")
    if actual_head != expected_head:
        errors.append(
            "release evidence git_head mismatch: "
            f"{actual_head!r} != {expected_head!r}"
        )


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    if not isinstance(data, dict):
        print("release evidence root must be a JSON object", file=sys.stderr)
        return 1

    require_equal(errors, data, "schema", SCHEMA, "schema mismatch")
    require_equal(errors, data, "meta.format", FORMAT_VERSION, "evidence format mismatch")
    require_equal(errors, data, "meta.source", "tensorcore_release_smoke", "evidence source mismatch")
    require_equal(errors, data, "status", "passed", "release smoke status must pass")
    require_equal(errors, data, "run.phase", "complete", "release smoke must reach complete phase")
    if get_path(data, "run.exit_status") not in ("0", 0):
        errors.append(f"release smoke exit status must be 0, got {get_path(data, 'run.exit_status')!r}")

    check_required_true(errors, data, BASE_REQUIRED_TRUE)

    failed_statuses = collect_failed_statuses(data)
    for path, status in failed_statuses:
        errors.append(f"unexpected failed status at {path}: {status!r}")

    require_gpu = args.require_gpu or args.require_metal4_tensorops
    gpu_available = get_path(data, "checks.tests.gpu_device_available")
    if require_gpu:
        require_true(
            errors,
            data,
            "checks.tests.gpu_device_available",
            "hardware evidence must come from a production Apple GPU",
        )
        require_equal(
            errors,
            data,
            "checks.tests.mode",
            "full",
            "hardware evidence must run the full test suite",
        )
        check_required_true(errors, data, GPU_REQUIRED_TRUE)
    elif gpu_available is True:
        check_required_true(errors, data, GPU_REQUIRED_TRUE)
    elif gpu_available is False:
        check_no_gpu_consistency(errors, data)
    else:
        errors.append(f"checks.tests.gpu_device_available must be boolean, got {gpu_available!r}")

    check_public_core_files(errors, data)
    check_metal4_consistency(
        errors,
        data,
        args.require_metal4_tensorops,
        args.require_metal4_compile,
    )
    if args.require_clean_head:
        check_git_provenance(errors, data, args.git_head)

    if errors:
        print("release evidence validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    metal4 = get_path(data, "checks.metal4_tensorops")
    print(
        "release evidence OK: "
        f"gpu={gpu_available} "
        f"metal4={metal4.get('compile_status')}/{metal4.get('runtime_status')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
