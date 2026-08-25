#!/usr/bin/env python3
"""Validate JSON evidence from scripts/run_sparse_24_runtime_evidence.py."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.sparse_24_runtime_evidence.v1"
FORMAT_VERSION = 1
VALID_STATUSES = {"passed", "failed", "blocked"}
VALID_CHECK_STATUSES = {"passed", "failed", "blocked"}
REQUIRED_CHECKS = {
    "sparse_24",
    "sparse_24_k_axis_contract",
}
REQUIRED_FUNCTIONS = {
    "lib/ops/sparse_gemm_cpu.cpp": {
        "read_elem",
        "write_elem",
        "supported_dtype",
        "tc_sparse_24_prune",
        "tc_sparse_24_check",
        "tc_sparse_24_gemm",
        "tc_sparse_24_available",
    },
    "tests/test_sparse_24.c": {
        "main",
    },
}
REQUIRED_HARDWARE_FUNCTIONS = {
    "lib/cuda/sparse_gemm.cpp": {
        "tc_sparse_24_gemm",
        "tc_sparse_24_available",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-clean-head", action="store_true")
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
        raise SystemExit(f"could not read sparse 2:4 evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"sparse 2:4 evidence is not valid JSON: {exc}") from exc


def fail(errors: list[str]) -> int:
    print("Sparse 2:4 evidence invalid:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def covered_functions(data: dict[str, Any]) -> dict[str, set[str]]:
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


def flatten_required(functions: dict[str, set[str]]) -> list[str]:
    return sorted(f"{path}:{name}" for path, names in functions.items() for name in names)


def expected_required_functions() -> list[str]:
    return flatten_required(REQUIRED_FUNCTIONS)


def expected_required_hardware_functions() -> list[str]:
    return flatten_required(REQUIRED_HARDWARE_FUNCTIONS)


def derived_covered_functions(data: dict[str, Any]) -> list[str]:
    covered = covered_functions(data)
    return sorted(f"{path}:{name}" for path, names in covered.items() for name in names)


def check_required_functions(errors: list[str], data: dict[str, Any]) -> None:
    covered = covered_functions(data)
    missing: list[str] = []
    for rel_path, names in REQUIRED_FUNCTIONS.items():
        present = covered.get(rel_path, set())
        for name in names:
            if name not in present:
                missing.append(f"{rel_path}:{name}")
    if missing:
        errors.append(f"sparse 2:4 evidence is missing function coverage: {sorted(missing)!r}")


def check_required_hardware_functions(errors: list[str], data: dict[str, Any]) -> None:
    covered = covered_functions(data)
    missing: list[str] = []
    for rel_path, names in REQUIRED_HARDWARE_FUNCTIONS.items():
        present = covered.get(rel_path, set())
        for name in names:
            if name not in present:
                missing.append(f"{rel_path}:{name}")
    if missing:
        errors.append(f"sparse 2:4 hardware evidence is missing function coverage: {sorted(missing)!r}")


def check_summary_contract(errors: list[str], data: dict[str, Any]) -> None:
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return
    required = expected_required_functions()
    required_hardware = expected_required_hardware_functions()
    covered = derived_covered_functions(data)
    missing = sorted(set(required) - set(covered))
    missing_hardware = sorted(set(required_hardware) - set(covered))
    if summary.get("required_functions") != required:
        errors.append(
            "summary.required_functions must match checker required functions: "
            f"{summary.get('required_functions')!r} != {required!r}"
        )
    if summary.get("covered_functions") != covered:
        errors.append(
            "summary.covered_functions must match files coverage: "
            f"{summary.get('covered_functions')!r} != {covered!r}"
        )
    if summary.get("missing_functions") != missing:
        errors.append(
            "summary.missing_functions must match derived missing functions: "
            f"{summary.get('missing_functions')!r} != {missing!r}"
        )
    if summary.get("required_hardware_functions") != required_hardware:
        errors.append(
            "summary.required_hardware_functions must match checker hardware functions: "
            f"{summary.get('required_hardware_functions')!r} != {required_hardware!r}"
        )
    if summary.get("missing_hardware_functions") != missing_hardware:
        errors.append(
            "summary.missing_hardware_functions must match derived hardware missing functions: "
            f"{summary.get('missing_hardware_functions')!r} != {missing_hardware!r}"
        )


def check_clean_head(errors: list[str], data: dict[str, Any], expected_head: str | None) -> None:
    if not expected_head:
        errors.append("expected git head is unavailable")
        return
    meta = data.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        return
    if meta.get("git_dirty") is not False:
        errors.append("sparse 2:4 evidence must be from a clean git tree")
    if meta.get("git_head") != expected_head:
        errors.append(
            "sparse 2:4 evidence git_head mismatch: "
            f"{meta.get('git_head')!r} != {expected_head!r}"
        )


def check_checks(errors: list[str], checks: Any, require_pass: bool) -> None:
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
        return
    missing = sorted(REQUIRED_CHECKS - set(checks))
    if missing:
        errors.append(f"checks missing required entries: {missing!r}")
    for name, item in checks.items():
        if not isinstance(item, dict):
            errors.append(f"checks.{name} must be an object")
            continue
        status = item.get("status")
        if status not in VALID_CHECK_STATUSES:
            errors.append(
                f"checks.{name}.status must be one of {sorted(VALID_CHECK_STATUSES)!r}, got {status!r}"
            )
        runtime_status = item.get("runtime_status")
        if runtime_status is not None and runtime_status not in VALID_CHECK_STATUSES:
            errors.append(
                f"checks.{name}.runtime_status must be one of {sorted(VALID_CHECK_STATUSES)!r}, "
                f"got {runtime_status!r}"
            )
        if require_pass and name in REQUIRED_CHECKS and status != "passed":
            errors.append(f"checks.{name}.status must be passed, got {status!r}")


def check_status_consistency(errors: list[str], data: dict[str, Any]) -> None:
    status = data.get("status")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return
    blocked = summary.get("blocked_reasons")
    failures = summary.get("failure_reasons")
    hardware_blocked = summary.get("hardware_blocked_reasons")
    if not isinstance(blocked, list):
        errors.append("summary.blocked_reasons must be a list")
    if not isinstance(failures, list):
        errors.append("summary.failure_reasons must be a list")
    if not isinstance(hardware_blocked, list):
        errors.append("summary.hardware_blocked_reasons must be a list")
    if status == "passed":
        if blocked:
            errors.append(f"passed evidence must not include blocked_reasons={blocked!r}")
        if failures:
            errors.append(f"passed evidence must not include failure_reasons={failures!r}")
    elif status == "blocked":
        if not blocked:
            errors.append("blocked evidence requires summary.blocked_reasons")
        if failures:
            errors.append(f"blocked evidence must not include failure_reasons={failures!r}")
    elif status == "failed":
        if not failures:
            errors.append("failed evidence requires summary.failure_reasons")


def check_hardware_requirement(errors: list[str], data: dict[str, Any]) -> None:
    summary = data.get("summary")
    checks = data.get("checks")
    if not isinstance(summary, dict) or not isinstance(checks, dict):
        return
    hardware = checks.get("sparse_24_hardware")
    if not isinstance(hardware, dict):
        errors.append("checks.sparse_24_hardware must be present for --require-hardware")
        return
    if summary.get("hardware_accelerated") is not True:
        errors.append("--require-hardware needs summary.hardware_accelerated=true")
    if summary.get("hardware_path") != "cusparseLt":
        errors.append(f"--require-hardware needs summary.hardware_path='cusparseLt', got {summary.get('hardware_path')!r}")
    if hardware.get("status") != "passed":
        errors.append(f"--require-hardware needs checks.sparse_24_hardware.status=passed, got {hardware.get('status')!r}")
    check_required_hardware_functions(errors, data)


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    if not isinstance(data, dict):
        return fail(["sparse 2:4 evidence root must be a JSON object"])
    meta = data.get("meta")
    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if not isinstance(meta, dict) or meta.get("format") != FORMAT_VERSION:
        errors.append(f"meta.format must be {FORMAT_VERSION}")
    if not isinstance(meta, dict) or meta.get("source") != "tensorcore_sparse_24_probe":
        errors.append("meta.source must be tensorcore_sparse_24_probe")
    if data.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)!r}, got {data.get('status')!r}")

    check_checks(errors, data.get("checks"), args.require_pass)
    check_status_consistency(errors, data)
    check_summary_contract(errors, data)
    trace = data.get("trace")
    if not isinstance(trace, list):
        errors.append("trace must be a list")
    elif data.get("status") in {"passed", "failed"} and not trace:
        errors.append("passed/failed evidence must include command traces")
    if data.get("status") == "passed":
        check_required_functions(errors, data)
    if args.require_pass and data.get("status") != "passed":
        errors.append(f"--require-pass needs passed evidence, got {data.get('status')!r}")
    if args.require_hardware:
        check_hardware_requirement(errors, data)
    if args.require_clean_head:
        check_clean_head(errors, data, args.git_head)
    if errors:
        return fail(errors)
    covered_count = sum(len(names) for names in covered_functions(data).values())
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    print(
        "Sparse 2:4 evidence OK: "
        f"status={data.get('status')} hardware={summary.get('hardware_path')} "
        f"covered_functions={covered_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
