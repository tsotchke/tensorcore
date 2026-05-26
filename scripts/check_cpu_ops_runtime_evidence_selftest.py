#!/usr/bin/env python3
"""Fixture tests for the CPU ops runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_cpu_ops_runtime_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "lib/ops/gemm_cpu.cpp": {
            "executed_lines": [419],
            "functions": {
                "gemm_compute": {"start_line": 419, "executed_lines": [419]},
            },
        },
        "lib/ops/conv2d_cpu.cpp": {
            "executed_lines": [44],
            "functions": {
                "direct_sgemm_f32": {"start_line": 44, "executed_lines": [44]},
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.cpu_ops_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_cpu_ops_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "paths": {
            "build_dir": "/repo/build-portable-cpu-current",
            "evidence": "/repo/build/cpu_ops_runtime_evidence.json",
        },
        "checks": {
            "portable_cpu": {"status": "passed", "binary": "/repo/build/tests/test_portable_cpu", "trace": "portable_cpu"},
            "conv2d": {"status": "passed", "binary": "/repo/build/tests/test_conv2d", "trace": "conv2d"},
        },
        "trace": [
            {"name": "portable_cpu", "cmd": ["test_portable_cpu"], "cwd": "/repo", "rc": 0},
            {"name": "conv2d", "cmd": ["test_conv2d"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "blocked_reasons": [],
            "failure_reasons": [],
            "required_functions": [
                "lib/ops/conv2d_cpu.cpp:direct_sgemm_f32",
                "lib/ops/gemm_cpu.cpp:gemm_compute",
            ],
            "covered_functions": [
                "lib/ops/conv2d_cpu.cpp:direct_sgemm_f32",
                "lib/ops/gemm_cpu.cpp:gemm_compute",
            ],
            "missing_functions": [],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["portable_cpu"] = {
        "status": "blocked",
        "blocked_reason": "test_binary_missing",
        "binary": None,
    }
    evidence["files"] = {
        "lib/ops/conv2d_cpu.cpp": coverage()["lib/ops/conv2d_cpu.cpp"],
    }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["portable_cpu:test_binary_missing"]
    evidence["summary"]["covered_functions"] = [
        "lib/ops/conv2d_cpu.cpp:direct_sgemm_f32",
    ]
    evidence["summary"]["missing_functions"] = [
        "lib/ops/gemm_cpu.cpp:gemm_compute",
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

    missing_function = copy.deepcopy(passed)
    del missing_function["files"]["lib/ops/gemm_cpu.cpp"]["functions"]["gemm_compute"]
    missing_function["summary"]["covered_functions"] = [
        "lib/ops/conv2d_cpu.cpp:direct_sgemm_f32"
    ]
    missing_function["summary"]["missing_functions"] = [
        "lib/ops/gemm_cpu.cpp:gemm_compute"
    ]
    assert_fails(missing_function, "missing function coverage", "--require-pass")

    stale_summary = copy.deepcopy(passed)
    stale_summary["summary"]["covered_functions"] = []
    assert_fails(stale_summary, "summary.covered_functions must match files coverage")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("CPU ops evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
