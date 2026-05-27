#!/usr/bin/env python3
"""Fixture tests for the M5 TensorOps runner preflight checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_m5_tensorops_runner_preflight.py"


def check(status: str, **extra: Any) -> dict[str, Any]:
    item = {"status": status}
    item.update(extra)
    return item


def evidence(status: str = "ready") -> dict[str, Any]:
    checks = {
        "host_platform": check("passed"),
        "xcode": check("passed"),
        "sdk26": check("passed", sdk_version="26.0", minimum="26.0"),
        "display_gpu": check("passed", m5_name_candidate=True, device_names=["Apple M5"]),
        "tensorops_runtime_probe": check("passed", runtime_status="passed"),
    }
    if status == "candidate":
        checks["tensorops_runtime_probe"] = check("skipped", reason="test_tensorops_runtime_missing")
    elif status == "blocked":
        checks["sdk26"] = check("blocked", sdk_version="15.2", minimum="26.0")
        checks["display_gpu"] = check("blocked", m5_name_candidate=False, device_names=["Apple M2 Ultra"])
        checks["tensorops_runtime_probe"] = check("skipped", reason="test_tensorops_runtime_missing")

    blocked_checks = sorted(name for name, item in checks.items() if item["status"] == "blocked")
    diagnostics = []
    for name, item in sorted(checks.items()):
        if item["status"] == "blocked":
            diagnostics.append(
                {
                    "id": f"m5_tensorops_preflight_diagnostic.{name}",
                    "name": name,
                    "status": "blocked",
                    "message": f"{name} is blocked",
                    "reason": f"{name} is blocked",
                    "diagnostic_class": "environment_unavailable",
                    "check_status": "blocked",
                }
            )
        elif item["status"] == "skipped":
            diagnostics.append(
                {
                    "id": f"m5_tensorops_preflight_diagnostic.{name}",
                    "name": name,
                    "status": "skipped",
                    "message": f"{name} is skipped",
                    "reason": f"{name} is skipped",
                    "diagnostic_class": "artifact_missing",
                    "check_status": "skipped",
                }
            )
    diagnostic_class_counts = {
        diagnostic_class: sum(1 for item in diagnostics if item["diagnostic_class"] == diagnostic_class)
        for diagnostic_class in sorted({item["diagnostic_class"] for item in diagnostics})
    }
    return {
        "schema": "tensorcore.m5_tensorops_runner_preflight.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_m5_tensorops_runner_preflight",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": status,
        "checks": checks,
        "diagnostics": diagnostics,
        "summary": {
            "ready_for_m5_tensorops_runtime": status == "ready",
            "candidate_host": status in {"ready", "candidate"},
            "blocked_checks": blocked_checks,
            "diagnostic_class_counts": diagnostic_class_counts,
            "environment_unavailable": "environment_unavailable" in diagnostic_class_counts,
            "source_failed": "source_failed" in diagnostic_class_counts,
        },
    }


def run_checker(payload: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        path = pathlib.Path(handle.name)
    try:
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_passes(payload: dict[str, Any], *args: str) -> None:
    result = run_checker(payload, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(payload: dict[str, Any], needle: str, *args: str) -> None:
    result = run_checker(payload, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    ready = evidence("ready")
    assert_passes(ready)
    assert_passes(ready, "--require-ready", "--git-head", "abc123", "--require-clean-head")

    candidate = evidence("candidate")
    assert_passes(candidate, "--require-candidate")
    assert_fails(candidate, "--require-ready needs ready evidence", "--require-ready")

    blocked = evidence("blocked")
    assert_passes(blocked)
    assert_passes(blocked, "--require-blocked-check", "sdk26")
    assert_fails(blocked, "required blocked check", "--require-blocked-check", "host_platform")

    inconsistent = copy.deepcopy(ready)
    inconsistent["summary"]["blocked_checks"] = ["sdk26"]
    assert_fails(inconsistent, "summary.blocked_checks")

    dirty = copy.deepcopy(ready)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(ready)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("M5 TensorOps runner preflight checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
