#!/usr/bin/env python3
"""Fixture tests for the fallback runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_fallback_runtime_evidence.py"


def good_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.fallback_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_fallback_runtime_smoke",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "checks": {
            "accelerate_f32": {"status": "passed", "label_seen": True},
            "mps_f32": {"status": "passed", "label_seen": True},
            "mps_bf16": {"status": "passed", "label_seen": True},
            "mps_i8": {"status": "passed", "label_seen": True},
        },
        "files": {
            "lib/fallback/accelerate_gemm.c": {
                "executed_lines": [17],
                "functions": {
                    "tc_accelerate_gemm_f32": {"start_line": 17, "executed_lines": [17]},
                },
            },
            "lib/fallback/mps_gemm.mm": {
                "executed_lines": [19, 37, 41, 48, 52, 56, 63, 137, 208],
                "functions": {
                    "to_mps_dtype": {"start_line": 19, "executed_lines": [19]},
                    "bf16_to_f32": {"start_line": 37, "executed_lines": [37]},
                    "f32_to_bf16": {"start_line": 41, "executed_lines": [41]},
                    "effective_lda": {"start_line": 48, "executed_lines": [48]},
                    "effective_ldb": {"start_line": 52, "executed_lines": [52]},
                    "effective_ldc": {"start_line": 56, "executed_lines": [56]},
                    "bf16_via_fp32": {"start_line": 63, "executed_lines": [63]},
                    "i8_via_fp32": {"start_line": 137, "executed_lines": [137]},
                    "tc_mps_gemm": {"start_line": 208, "executed_lines": [208]},
                },
            },
        },
        "summary": {
            "checks_passed": True,
            "missing_functions": [],
        },
    }


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
    good = good_evidence()
    assert_passes(good)
    assert_passes(good, "--require-pass", "--git-head", "abc123", "--require-clean-head")

    failed = copy.deepcopy(good)
    failed["status"] = "failed"
    failed["checks"]["mps_i8"]["status"] = "failed"
    assert_fails(failed, "--require-pass needs passed evidence", "--require-pass")

    missing = copy.deepcopy(good)
    del missing["files"]["lib/fallback/mps_gemm.mm"]["functions"]["effective_ldc"]
    assert_fails(missing, "missing function coverage")

    dirty = copy.deepcopy(good)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(good)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("fallback runtime evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
