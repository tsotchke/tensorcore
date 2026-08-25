#!/usr/bin/env python3
"""Validate an M5 TensorOps runner preflight evidence artifact."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.m5_tensorops_runner_preflight.v1"
FORMAT_VERSION = 1
VALID_STATUSES = {"ready", "candidate", "blocked"}
CHECK_STATUSES = {"passed", "blocked", "skipped", "unknown"}
REQUIRED_CHECKS = {
    "host_platform",
    "xcode",
    "sdk26",
    "display_gpu",
    "tensorops_runtime_probe",
}
CANDIDATE_REQUIRED_CHECKS = ("host_platform", "xcode", "sdk26", "display_gpu")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--require-blocked-check", action="append", default=[])
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-clean-head", action="store_true")
    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read M5 TensorOps runner preflight {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"M5 TensorOps runner preflight is not valid JSON: {exc}") from exc


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def fail(errors: list[str]) -> int:
    print("M5 TensorOps runner preflight invalid:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def check_clean_head(errors: list[str], data: dict[str, Any], expected_head: str | None) -> None:
    if not expected_head:
        errors.append("expected git head is unavailable")
        return
    if get_path(data, "meta.git_dirty") is not False:
        errors.append("M5 TensorOps runner preflight must be from a clean git tree")
    actual = get_path(data, "meta.git_head")
    if actual != expected_head:
        errors.append(f"M5 TensorOps runner preflight git_head mismatch: {actual!r} != {expected_head!r}")


def check_summary_consistency(errors: list[str], data: dict[str, Any]) -> None:
    status = data.get("status")
    checks = data.get("checks")
    summary = data.get("summary")
    if not isinstance(checks, dict) or not isinstance(summary, dict):
        return

    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, list):
        errors.append("diagnostics must be a list")
        diagnostics = []

    blocked_checks = sorted(
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "blocked"
    )
    if summary.get("blocked_checks") != blocked_checks:
        errors.append(
            "summary.blocked_checks must match blocked checks: "
            f"{summary.get('blocked_checks')!r} != {blocked_checks!r}"
        )
    if summary.get("ready_for_m5_tensorops_runtime") != (status == "ready"):
        errors.append("summary.ready_for_m5_tensorops_runtime does not match status")
    if summary.get("candidate_host") != (status in {"ready", "candidate"}):
        errors.append("summary.candidate_host does not match status")
    diagnostic_classes = [
        str(item.get("diagnostic_class"))
        for item in diagnostics
        if isinstance(item, dict) and item.get("diagnostic_class")
    ]
    class_counts = {
        diagnostic_class: diagnostic_classes.count(diagnostic_class)
        for diagnostic_class in sorted(set(diagnostic_classes))
    }
    if summary.get("diagnostic_class_counts") != class_counts:
        errors.append(
            "summary.diagnostic_class_counts must match diagnostics: "
            f"{summary.get('diagnostic_class_counts')!r} != {class_counts!r}"
        )
    if summary.get("environment_unavailable") != ("environment_unavailable" in class_counts):
        errors.append("summary.environment_unavailable does not match diagnostics")
    if summary.get("source_failed") != ("source_failed" in class_counts):
        errors.append("summary.source_failed does not match diagnostics")


def check_ready_contract(errors: list[str], data: dict[str, Any]) -> None:
    checks = data.get("checks")
    if not isinstance(checks, dict):
        return
    for name in ("host_platform", "xcode", "sdk26", "display_gpu", "tensorops_runtime_probe"):
        item = checks.get(name)
        if not isinstance(item, dict) or item.get("status") != "passed":
            errors.append(f"checks.{name}.status must be passed for ready evidence")
    runtime_status = get_path(data, "checks.tensorops_runtime_probe.runtime_status")
    if runtime_status != "passed":
        errors.append(f"checks.tensorops_runtime_probe.runtime_status must be passed, got {runtime_status!r}")


def check_candidate_contract(errors: list[str], data: dict[str, Any]) -> None:
    checks = data.get("checks")
    if not isinstance(checks, dict):
        return
    for name in CANDIDATE_REQUIRED_CHECKS:
        item = checks.get(name)
        if not isinstance(item, dict) or item.get("status") != "passed":
            errors.append(f"checks.{name}.status must be passed for candidate evidence")
    runtime_status = get_path(data, "checks.tensorops_runtime_probe.status")
    if runtime_status not in {"skipped", "unknown"}:
        errors.append(
            "checks.tensorops_runtime_probe.status must be skipped or unknown for candidate "
            f"evidence, got {runtime_status!r}"
        )


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    if not isinstance(data, dict):
        return fail(["M5 TensorOps runner preflight root must be a JSON object"])

    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if get_path(data, "meta.format") != FORMAT_VERSION:
        errors.append(f"meta.format must be {FORMAT_VERSION}")
    if get_path(data, "meta.source") != "tensorcore_m5_tensorops_runner_preflight":
        errors.append("meta.source must be tensorcore_m5_tensorops_runner_preflight")
    if data.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)!r}, got {data.get('status')!r}")

    checks = data.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
    else:
        missing = sorted(REQUIRED_CHECKS - set(checks))
        if missing:
            errors.append(f"checks missing required entries: {missing!r}")
        for name, item in checks.items():
            if not isinstance(item, dict):
                errors.append(f"checks.{name} must be an object")
                continue
            if item.get("status") not in CHECK_STATUSES:
                errors.append(
                    f"checks.{name}.status must be one of {sorted(CHECK_STATUSES)!r}, "
                    f"got {item.get('status')!r}"
                )

    if not isinstance(data.get("summary"), dict):
        errors.append("summary must be an object")
    check_summary_consistency(errors, data)
    if data.get("status") == "candidate":
        check_candidate_contract(errors, data)

    if args.require_ready:
        if data.get("status") != "ready":
            errors.append(f"--require-ready needs ready evidence, got {data.get('status')!r}")
        check_ready_contract(errors, data)
    if args.require_candidate and data.get("status") not in {"ready", "candidate"}:
        errors.append(f"--require-candidate needs ready or candidate evidence, got {data.get('status')!r}")
    if args.require_blocked_check:
        blocked = set(get_path(data, "summary.blocked_checks") or [])
        for check_name in args.require_blocked_check:
            if check_name not in blocked:
                errors.append(f"required blocked check {check_name!r} is absent from summary.blocked_checks")
    if args.require_clean_head:
        check_clean_head(errors, data, args.git_head)

    if errors:
        return fail(errors)

    blocked_checks = ",".join(get_path(data, "summary.blocked_checks") or [])
    print(
        "M5 TensorOps runner preflight OK: "
        f"status={data.get('status')} blocked_checks={blocked_checks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
