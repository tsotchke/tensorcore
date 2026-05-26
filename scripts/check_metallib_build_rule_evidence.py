#!/usr/bin/env python3
"""Validate JSON evidence from scripts/run_metallib_build_rule_evidence.py."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.metallib_build_rule_evidence.v1"
FORMAT_VERSION = 1
VALID_STATUSES = {"passed", "failed", "blocked"}
VALID_CHECK_STATUSES = {"passed", "failed", "blocked", "skipped"}
VALID_BLOCKED_REASONS = {
    "cmake_missing",
    "non_apple_platform",
    "xcrun_missing",
    "metal_compiler_missing",
    "metallib_linker_missing",
}
REQUIRED_CHECKS = {
    "cmake_available",
    "probe_project",
    "configure_rule",
    "build_metallib",
}
REQUIRED_FUNCTIONS = {
    "cmake/compile_metallib.cmake": {"tc_compile_metallib"},
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--require-pass", action="store_true")
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
        raise SystemExit(f"could not read metallib build-rule evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"metallib build-rule evidence is not valid JSON: {exc}") from exc


def fail(errors: list[str]) -> int:
    print("metallib build-rule evidence invalid:", file=sys.stderr)
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
        if not isinstance(functions, dict):
            continue
        covered[str(rel_path)] = {str(name) for name in functions}
    return covered


def check_required_functions(errors: list[str], data: dict[str, Any]) -> None:
    covered = covered_functions(data)
    missing: list[str] = []
    for rel_path, names in REQUIRED_FUNCTIONS.items():
        present = covered.get(rel_path, set())
        for name in names:
            if name not in present:
                missing.append(f"{rel_path}:{name}")
    if missing:
        errors.append(f"metallib build-rule evidence is missing function coverage: {sorted(missing)!r}")


def check_clean_head(errors: list[str], data: dict[str, Any], expected_head: str | None) -> None:
    if not expected_head:
        errors.append("expected git head is unavailable")
        return
    meta = data.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")
        return
    if meta.get("git_dirty") is not False:
        errors.append("metallib build-rule evidence must be from a clean git tree")
    if meta.get("git_head") != expected_head:
        errors.append(
            "metallib build-rule evidence git_head mismatch: "
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
        if require_pass and name in REQUIRED_CHECKS and status != "passed":
            errors.append(f"checks.{name}.status must be passed, got {status!r}")


def check_status_consistency(errors: list[str], data: dict[str, Any]) -> None:
    status = data.get("status")
    checks = data.get("checks")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return
    if not isinstance(checks, dict):
        return

    blocked_reason = summary.get("blocked_reason")
    failure_reason = summary.get("failure_reason")
    if status == "passed":
        if blocked_reason not in (None, ""):
            errors.append(f"passed evidence must not include blocked_reason={blocked_reason!r}")
        if failure_reason not in (None, ""):
            errors.append(f"passed evidence must not include failure_reason={failure_reason!r}")
        if checks.get("configure_rule", {}).get("status") != "passed":
            errors.append("passed evidence requires checks.configure_rule.status=passed")
        if checks.get("build_metallib", {}).get("status") != "passed":
            errors.append("passed evidence requires checks.build_metallib.status=passed")
        build = checks.get("build_metallib", {})
        if not isinstance(build.get("output_size"), int) or build.get("output_size", 0) <= 0:
            errors.append("passed evidence requires positive checks.build_metallib.output_size")
        if not isinstance(build.get("artifact_hash"), str) or not SHA256_RE.match(build["artifact_hash"]):
            errors.append("passed evidence requires checks.build_metallib.artifact_hash sha256")
        if summary.get("artifact_hash") != build.get("artifact_hash"):
            errors.append("passed evidence requires summary.artifact_hash to match build_metallib")
        if summary.get("backend") != "metal":
            errors.append(f"passed evidence requires summary.backend='metal', got {summary.get('backend')!r}")
    elif status == "blocked":
        if blocked_reason not in VALID_BLOCKED_REASONS:
            errors.append(
                f"blocked evidence requires blocked_reason in {sorted(VALID_BLOCKED_REASONS)!r}, "
                f"got {blocked_reason!r}"
            )
        if blocked_reason == "cmake_missing":
            if checks.get("cmake_available", {}).get("status") != "blocked":
                errors.append("cmake_missing evidence requires checks.cmake_available.status=blocked")
        elif checks.get("configure_rule", {}).get("status") != "blocked" and checks.get(
            "build_metallib", {}
        ).get("status") != "blocked":
            errors.append("blocked evidence requires a blocked configure_rule or build_metallib check")
        if failure_reason not in (None, ""):
            errors.append(f"blocked evidence must not include failure_reason={failure_reason!r}")
        if blocked_reason == "non_apple_platform" and summary.get("backend") != "unsupported":
            errors.append("non_apple_platform evidence requires summary.backend='unsupported'")
    elif status == "failed":
        if not failure_reason:
            errors.append("failed evidence requires summary.failure_reason")
        if not summary.get("error"):
            errors.append("failed evidence requires summary.error")


def coverage_required(data: dict[str, Any]) -> bool:
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return True
    return not (data.get("status") == "blocked" and summary.get("blocked_reason") == "cmake_missing")


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    if not isinstance(data, dict):
        return fail(["metallib build-rule evidence root must be a JSON object"])

    meta = data.get("meta")
    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if not isinstance(meta, dict) or meta.get("format") != FORMAT_VERSION:
        errors.append(f"meta.format must be {FORMAT_VERSION}")
    if not isinstance(meta, dict) or meta.get("source") != "tensorcore_metallib_build_rule_probe":
        errors.append("meta.source must be tensorcore_metallib_build_rule_probe")
    if data.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)!r}, got {data.get('status')!r}")

    check_checks(errors, data.get("checks"), args.require_pass)
    check_status_consistency(errors, data)

    trace = data.get("trace")
    if not isinstance(trace, list):
        errors.append("trace must be a list")
    elif data.get("status") in {"passed", "failed"} and not trace:
        errors.append("passed/failed evidence must include at least one command trace")

    if coverage_required(data):
        check_required_functions(errors, data)
    summary = data.get("summary")
    if coverage_required(data) and isinstance(summary, dict) and summary.get("missing_functions") not in ([], None):
        errors.append(f"summary.missing_functions must be empty, got {summary.get('missing_functions')!r}")

    if args.require_pass and data.get("status") != "passed":
        errors.append(f"--require-pass needs passed evidence, got {data.get('status')!r}")

    if args.require_clean_head:
        check_clean_head(errors, data, args.git_head)

    if errors:
        return fail(errors)

    covered_count = sum(len(names) for names in covered_functions(data).values())
    reason = ""
    if isinstance(summary, dict):
        reason = summary.get("blocked_reason") or summary.get("failure_reason") or "ok"
    print(
        "metallib build-rule evidence OK: "
        f"status={data.get('status')} reason={reason} covered_functions={covered_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
