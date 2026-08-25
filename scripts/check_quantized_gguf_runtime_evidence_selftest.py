#!/usr/bin/env python3
"""Fixture tests for the Quantized/GGUF runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_quantized_gguf_runtime_evidence.py"


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.quantized_gguf_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_quantized_gguf_runtime_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "checks": {
            "quantized": {"status": "passed", "binary": "/repo/build/tests/test_quantized", "trace": "quantized"},
            "gguf": {"status": "passed", "binary": "/repo/build/tests/test_gguf", "trace": "gguf"},
        },
        "trace": [
            {"name": "quantized", "cmd": ["/repo/build/tests/test_quantized"], "rc": 0},
            {"name": "gguf", "cmd": ["/repo/build/tests/test_gguf"], "rc": 0},
        ],
        "files": {
            "lib/ops/quantized.mm": {
                "executed_lines": [143],
                "functions": {
                    "gemv_quant_encode": {"start_line": 143, "executed_lines": [143]},
                },
            },
            "lib/io/gguf.c": {
                "executed_lines": [766],
                "functions": {
                    "gguf_quantized_matrix_info_common": {"start_line": 766, "executed_lines": [766]},
                },
            },
        },
        "summary": {
            "checks_passed": True,
            "blocked_reasons": [],
            "failure_reasons": [],
            "required_functions": [
                "lib/io/gguf.c:gguf_quantized_matrix_info_common",
                "lib/ops/quantized.mm:gemv_quant_encode",
            ],
            "covered_functions": [
                "lib/io/gguf.c:gguf_quantized_matrix_info_common",
                "lib/ops/quantized.mm:gemv_quant_encode",
            ],
            "missing_functions": [],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["gguf"] = {
        "status": "blocked",
        "binary": "/repo/build/tests/test_gguf",
        "trace": "gguf",
        "blocked_reason": "metal_device_unavailable",
    }
    evidence["files"] = {
        "lib/ops/quantized.mm": evidence["files"]["lib/ops/quantized.mm"],
    }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["gguf:metal_device_unavailable"]
    evidence["summary"]["covered_functions"] = ["lib/ops/quantized.mm:gemv_quant_encode"]
    evidence["summary"]["missing_functions"] = ["lib/io/gguf.c:gguf_quantized_matrix_info_common"]
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
    assert_fails(blocked, "checks.gguf.status must be passed", "--require-pass")

    missing = copy.deepcopy(passed)
    del missing["files"]["lib/ops/quantized.mm"]["functions"]["gemv_quant_encode"]
    assert_fails(missing, "missing function coverage", "--require-pass")

    dirty = copy.deepcopy(passed)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(passed)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("Quantized/GGUF runtime evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
