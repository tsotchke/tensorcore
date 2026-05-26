#!/usr/bin/env python3
"""Fixture tests for the Metal ops runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_metal_ops_runtime_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "lib/ops/attention.mm": {
            "executed_lines": [205],
            "functions": {
                "encode_forward": {"start_line": 205, "executed_lines": [205]},
            },
        },
        "lib/ops/conv.mm": {
            "executed_lines": [46],
            "functions": {
                "conv_bytes": {"start_line": 46, "executed_lines": [46]},
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.metal_ops_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_metal_ops_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "paths": {
            "build_dir": "/repo/build",
            "evidence": "/repo/build/metal_ops_runtime_evidence.json",
        },
        "checks": {
            "attention_correctness": {"status": "passed", "binary": "/repo/build/tests/test_attention_correctness", "trace": "attention_correctness"},
            "conv2d": {"status": "passed", "binary": "/repo/build/tests/test_conv2d", "trace": "conv2d"},
            "async_copy_shader": {
                "status": "blocked",
                "blocked_reason": "shader_line_execution_trace_unavailable",
                "compiled_async_kernel": True,
            },
        },
        "trace": [
            {"name": "attention_correctness", "cmd": ["test_attention_correctness"], "cwd": "/repo", "rc": 0},
            {"name": "conv2d", "cmd": ["test_conv2d"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "blocked_reasons": [],
            "failure_reasons": [],
            "optional_blocked_reasons": ["async_copy_shader:shader_line_execution_trace_unavailable"],
            "required_functions": [
                "lib/ops/attention.mm:encode_forward",
                "lib/ops/conv.mm:conv_bytes",
            ],
            "covered_functions": [
                "lib/ops/attention.mm:encode_forward",
                "lib/ops/conv.mm:conv_bytes",
            ],
            "missing_functions": [],
            "optional_missing_functions": [
                "kernels/metal/metal_simdgroup_event.h:async_copy",
            ],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["attention_correctness"] = {
        "status": "blocked",
        "blocked_reason": "test_binary_missing",
        "binary": None,
    }
    evidence["files"] = {
        "lib/ops/conv.mm": coverage()["lib/ops/conv.mm"],
    }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["attention_correctness:test_binary_missing"]
    evidence["summary"]["covered_functions"] = ["lib/ops/conv.mm:conv_bytes"]
    evidence["summary"]["missing_functions"] = ["lib/ops/attention.mm:encode_forward"]
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
    del missing_function["files"]["lib/ops/attention.mm"]["functions"]["encode_forward"]
    missing_function["summary"]["covered_functions"] = ["lib/ops/conv.mm:conv_bytes"]
    missing_function["summary"]["missing_functions"] = ["lib/ops/attention.mm:encode_forward"]
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

    print("Metal ops evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
