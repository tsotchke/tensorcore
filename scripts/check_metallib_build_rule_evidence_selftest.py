#!/usr/bin/env python3
"""Fixture tests for the metallib build-rule evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_metallib_build_rule_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "cmake/compile_metallib.cmake": {
            "executed_lines": [22, 31, 43, 53, 64, 127],
            "functions": {
                "tc_compile_metallib": {"start_line": 22, "executed_lines": [22]},
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.metallib_build_rule_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_metallib_build_rule_probe",
            "git_head": "abc123",
            "git_dirty": False,
            "host_system": "Darwin",
            "host_machine": "arm64",
        },
        "status": "passed",
        "paths": {
            "work_dir": "/tmp/metallib-evidence",
            "source_dir": "/tmp/metallib-evidence/source",
            "build_dir": "/tmp/metallib-evidence/build",
            "evidence": "/tmp/metallib-evidence/metallib_build_rule_evidence.json",
        },
        "toolchain": {"cmake": "/usr/bin/cmake", "xcrun": "/usr/bin/xcrun"},
        "checks": {
            "cmake_available": {"status": "passed", "path": "/usr/bin/cmake"},
            "probe_project": {"status": "passed"},
            "configure_rule": {"status": "passed", "trace": "configure_rule"},
            "build_metallib": {
                "status": "passed",
                "trace": "build_metallib",
                "output": "/tmp/metallib-evidence/build/probe.metallib",
                "output_size": 1024,
                "artifact_hash": "a" * 64,
            },
        },
        "trace": [
            {"name": "configure_rule", "cmd": ["cmake"], "cwd": "/repo", "rc": 0},
            {"name": "build_metallib", "cmd": ["cmake"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "backend": "metal",
            "artifact_hash": "a" * 64,
            "error": None,
            "blocked_reason": None,
            "failure_reason": None,
            "required_functions": ["cmake/compile_metallib.cmake:tc_compile_metallib"],
            "covered_functions": ["cmake/compile_metallib.cmake:tc_compile_metallib"],
            "missing_functions": [],
        },
    }


def blocked_evidence(reason: str = "non_apple_platform") -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["meta"]["host_system"] = "Linux" if reason == "non_apple_platform" else "Darwin"
    if reason == "cmake_missing":
        evidence["checks"]["cmake_available"] = {
            "status": "blocked",
            "path": "cmake",
            "blocked_reason": reason,
        }
        evidence["checks"]["probe_project"] = {"status": "skipped"}
        evidence["checks"]["configure_rule"] = {"status": "skipped"}
        evidence["checks"]["build_metallib"] = {"status": "skipped"}
        evidence["trace"] = []
        evidence["files"] = {}
        evidence["summary"]["covered_functions"] = []
        evidence["summary"]["missing_functions"] = [
            "cmake/compile_metallib.cmake:tc_compile_metallib",
        ]
    else:
        evidence["checks"]["configure_rule"] = {
            "status": "blocked",
            "blocked_reason": reason,
            "trace": "configure_rule",
        }
        evidence["checks"]["build_metallib"] = {
            "status": "skipped",
            "reason": "configure_blocked",
        }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["backend"] = "unsupported" if reason == "non_apple_platform" else "unknown"
    evidence["summary"]["artifact_hash"] = None
    evidence["summary"]["error"] = reason
    evidence["summary"]["blocked_reason"] = reason
    return evidence


def run_checker(evidence: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(evidence, handle)
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


def assert_passes(evidence: dict[str, Any], *args: str) -> None:
    result = run_checker(evidence, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(evidence: dict[str, Any], needle: str, *args: str) -> None:
    result = run_checker(evidence, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    passed = passed_evidence()
    assert_passes(passed)
    assert_passes(passed, "--require-pass", "--git-head", "abc123", "--require-clean-head")

    blocked_non_apple = blocked_evidence("non_apple_platform")
    assert_passes(blocked_non_apple)
    assert_fails(blocked_non_apple, "--require-pass needs passed evidence", "--require-pass")

    blocked_xcrun = blocked_evidence("xcrun_missing")
    assert_passes(blocked_xcrun)

    blocked_cmake = blocked_evidence("cmake_missing")
    assert_passes(blocked_cmake)

    missing = copy.deepcopy(passed)
    del missing["files"]["cmake/compile_metallib.cmake"]["functions"]["tc_compile_metallib"]
    assert_fails(missing, "missing function coverage")

    no_hash = copy.deepcopy(passed)
    no_hash["checks"]["build_metallib"]["artifact_hash"] = "not-a-sha"
    assert_fails(no_hash, "artifact_hash sha256", "--require-pass")

    bad_blocked = blocked_evidence("unknown")
    assert_fails(bad_blocked, "blocked evidence requires blocked_reason")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("metallib build-rule evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
