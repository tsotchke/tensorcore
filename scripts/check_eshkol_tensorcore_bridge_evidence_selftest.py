#!/usr/bin/env python3
"""Fixture tests for the Eshkol tensorcore bridge evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_eshkol_tensorcore_bridge_evidence.py"


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.eshkol_bridge_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_eshkol_bridge_smoke",
            "git_head": "abc123",
            "git_dirty": False,
            "eshkol_run": "/opt/eshkol/bin/eshkol-run",
        },
        "status": "passed",
        "checks": {
            "eshkol_run_available": {"status": "passed"},
            "hello_tensorcore_compile": {"status": "passed"},
            "hello_tensorcore_runtime": {"status": "passed"},
            "tensorcore_bridge_smoke_compile": {"status": "passed"},
            "tensorcore_bridge_smoke_runtime": {"status": "passed"},
            "source_module_load": {"status": "passed"},
            "bridge_builtin_resolution": {
                "status": "passed",
                "missing_builtins": [],
                "missing_public_wrappers": [],
            },
        },
        "files": {
            "eshkol/hello_tensorcore.esk": {
                "executed_lines": [14],
                "functions": {
                    "main": {"start_line": 14, "executed_lines": [14]},
                },
            },
            "eshkol/tensorcore.esk": {
                "executed_lines": [40, 44, 48, 61, 65, 68, 85, 91, 95, 99, 107, 134],
                "functions": {
                    "tc-init": {"start_line": 40, "executed_lines": [40]},
                    "tc-shutdown": {"start_line": 44, "executed_lines": [44]},
                    "tc-device-info": {"start_line": 48, "executed_lines": [48]},
                    "tc-buffer-alloc": {"start_line": 61, "executed_lines": [61]},
                    "tc-buffer-free": {"start_line": 65, "executed_lines": [65]},
                    "tc-buffer-map": {"start_line": 68, "executed_lines": [68]},
                    "tc-gemm": {"start_line": 85, "executed_lines": [85]},
                    "tc-gemm-fp16": {"start_line": 91, "executed_lines": [91]},
                    "tc-gemm-fp32": {"start_line": 95, "executed_lines": [95]},
                    "tc-gemm-bf16": {"start_line": 99, "executed_lines": [99]},
                    "tc-attention-forward": {"start_line": 107, "executed_lines": [107]},
                    "tc-last-backend": {"start_line": 134, "executed_lines": [134]},
                },
            },
        },
        "summary": {
            "required_functions": [],
            "covered_functions": [],
            "missing_functions": [],
            "missing_builtins": [],
            "missing_public_wrappers": [],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["bridge_builtin_resolution"] = {
        "status": "blocked",
        "missing_builtins": ["__tc-init"],
        "missing_public_wrappers": [],
    }
    evidence["checks"]["hello_tensorcore_compile"]["status"] = "failed"
    evidence["checks"]["hello_tensorcore_runtime"]["status"] = "skipped_compile_failed"
    evidence["checks"]["tensorcore_bridge_smoke_compile"]["status"] = "failed"
    evidence["checks"]["tensorcore_bridge_smoke_runtime"]["status"] = "skipped_compile_failed"
    evidence["files"] = {}
    evidence["summary"]["missing_functions"] = [
        "eshkol/hello_tensorcore.esk:main",
        "eshkol/tensorcore.esk:tc-device-info",
    ]
    evidence["summary"]["missing_builtins"] = ["__tc-init"]
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

    missing = copy.deepcopy(passed)
    del missing["files"]["eshkol/tensorcore.esk"]["functions"]["tc-buffer-map"]
    assert_fails(missing, "missing function coverage", "--require-pass")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("Eshkol tensorcore bridge evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
