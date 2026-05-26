#!/usr/bin/env python3
"""Fixture tests for the Python packaging evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_python_packaging_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "setup.py": {
            "executed_lines": [82, 156, 162],
            "functions": {
                "_run_tool": {"start_line": 82, "executed_lines": [82]},
                "build_py_with_native_artifacts.run": {
                    "start_line": 156,
                    "executed_lines": [156],
                },
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.python_packaging_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_python_packaging_probe",
            "git_head": "abc123",
            "git_dirty": False,
            "host_system": "Darwin",
            "host_machine": "arm64",
        },
        "status": "passed",
        "run": {"exit_status": "0", "platform_tag": "macosx_15_0_arm64"},
        "project": {"root": "/repo"},
        "python": {"executable": "/usr/bin/python3"},
        "paths": {
            "work_dir": "/tmp/tensorcore-python-packaging",
            "build_base": "/tmp/tensorcore-python-packaging/build-base",
            "bdist_dir": "/tmp/tensorcore-python-packaging/bdist",
            "dist_dir": "/tmp/tensorcore-python-packaging/dist",
            "evidence": "/tmp/tensorcore-python-packaging/python_packaging_evidence.json",
        },
        "checks": {
            "native_artifacts": {
                "status": "passed",
                "required": ["libtensorcore.dylib", "tensorcore.metallib"],
                "found": {
                    "libtensorcore.dylib": "/repo/build/libtensorcore.dylib",
                    "tensorcore.metallib": "/repo/build/tensorcore.metallib",
                },
                "missing": [],
            },
            "run_tool_lipo": {
                "status": "passed",
                "trace": "run_tool_lipo",
                "arches": ["arm64"],
            },
            "build_py_native_copy": {
                "status": "passed",
                "trace": "build_py_native_copy",
                "package_dir": "/tmp/tensorcore-python-packaging/build-base/lib/tensorcore",
                "copied": {
                    "libtensorcore.dylib": {"path": "/tmp/libtensorcore.dylib", "size": 8, "sha256": "a" * 64},
                    "tensorcore.metallib": {"path": "/tmp/tensorcore.metallib", "size": 8, "sha256": "b" * 64},
                },
                "missing": [],
            },
            "bdist_wheel_native_artifacts": {
                "status": "passed",
                "trace": "bdist_wheel_native_artifacts",
                "platform_tag": "macosx_15_0_arm64",
                "wheel": "/tmp/tensorcore_apple-0.1.22-py3-none-macosx_15_0_arm64.whl",
                "wheel_size": 42,
                "wheel_sha256": "c" * 64,
                "missing": [],
            },
        },
        "trace": [
            {"name": "run_tool_lipo", "cmd": ["python"], "cwd": "/repo", "rc": 0},
            {"name": "build_py_native_copy", "cmd": ["python"], "cwd": "/repo", "rc": 0},
            {"name": "bdist_wheel_native_artifacts", "cmd": ["python"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "blocked_reason": None,
            "failure_reason": None,
            "required_functions": [
                "setup.py:_run_tool",
                "setup.py:build_py_with_native_artifacts.run",
            ],
            "covered_functions": [
                "setup.py:_run_tool",
                "setup.py:build_py_with_native_artifacts.run",
            ],
            "missing_functions": [],
        },
    }


def blocked_evidence(reason: str = "native_artifacts_missing") -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["native_artifacts"] = {
        "status": "blocked" if reason == "native_artifacts_missing" else "passed",
        "required": ["libtensorcore.dylib", "tensorcore.metallib"],
        "found": {},
        "missing": ["libtensorcore.dylib"],
    }
    evidence["checks"]["run_tool_lipo"] = {"status": "blocked", "blocked_reason": reason}
    evidence["checks"]["build_py_native_copy"] = {"status": "skipped"}
    evidence["checks"]["bdist_wheel_native_artifacts"] = {"status": "skipped"}
    evidence["trace"] = []
    evidence["files"] = {}
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reason"] = reason
    evidence["summary"]["covered_functions"] = []
    evidence["summary"]["missing_functions"] = [
        "setup.py:_run_tool",
        "setup.py:build_py_with_native_artifacts.run",
    ]
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

    blocked = blocked_evidence()
    assert_passes(blocked)
    assert_fails(blocked, "--require-pass needs passed evidence", "--require-pass")

    no_hash = copy.deepcopy(passed)
    no_hash["checks"]["bdist_wheel_native_artifacts"]["wheel_sha256"] = "not-a-sha"
    assert_fails(no_hash, "wheel_sha256", "--require-pass")

    missing_function = copy.deepcopy(passed)
    del missing_function["files"]["setup.py"]["functions"]["_run_tool"]
    assert_fails(missing_function, "missing function coverage", "--require-pass")

    bad_blocked = blocked_evidence("unknown")
    assert_fails(bad_blocked, "blocked evidence requires blocked_reason")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("python packaging evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
