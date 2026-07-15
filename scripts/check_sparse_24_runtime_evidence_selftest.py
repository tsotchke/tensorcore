#!/usr/bin/env python3
"""Fixture tests for the sparse 2:4 runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_sparse_24_runtime_evidence.py"


def local_coverage() -> dict[str, Any]:
    return {
        "lib/ops/sparse_gemm_cpu.cpp": {
            "executed_lines": [34, 52, 63, 60, 98, 141, 166],
            "functions": {
                "read_elem": {"start_line": 34, "executed_lines": [34]},
                "write_elem": {"start_line": 52, "executed_lines": [52]},
                "supported_dtype": {"start_line": 63, "executed_lines": [63]},
                "tc_sparse_24_prune": {"start_line": 60, "executed_lines": [60]},
                "tc_sparse_24_check": {"start_line": 98, "executed_lines": [98]},
                "tc_sparse_24_gemm": {"start_line": 141, "executed_lines": [141]},
                "tc_sparse_24_available": {"start_line": 166, "executed_lines": [166]},
            },
        },
        "tests/test_sparse_24.c": {
            "executed_lines": [25],
            "functions": {
                "main": {"start_line": 25, "executed_lines": [25]},
            },
        },
    }


def hardware_coverage() -> dict[str, Any]:
    coverage = local_coverage()
    coverage["lib/cuda/sparse_gemm.cpp"] = {
        "executed_lines": [82, 174],
        "functions": {
            "tc_sparse_24_gemm": {"start_line": 82, "executed_lines": [82]},
            "tc_sparse_24_available": {"start_line": 174, "executed_lines": [174]},
        },
    }
    return coverage


def covered_functions(files: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for path, entry in files.items():
        for name in sorted((entry.get("functions") or {}).keys()):
            out.append(f"{path}:{name}")
    return sorted(out)


REQUIRED = [
    "lib/ops/sparse_gemm_cpu.cpp:read_elem",
    "lib/ops/sparse_gemm_cpu.cpp:supported_dtype",
    "lib/ops/sparse_gemm_cpu.cpp:tc_sparse_24_available",
    "lib/ops/sparse_gemm_cpu.cpp:tc_sparse_24_check",
    "lib/ops/sparse_gemm_cpu.cpp:tc_sparse_24_gemm",
    "lib/ops/sparse_gemm_cpu.cpp:tc_sparse_24_prune",
    "lib/ops/sparse_gemm_cpu.cpp:write_elem",
    "tests/test_sparse_24.c:main",
]
REQUIRED_HARDWARE = [
    "lib/cuda/sparse_gemm.cpp:tc_sparse_24_available",
    "lib/cuda/sparse_gemm.cpp:tc_sparse_24_gemm",
]


def passed_fallback_evidence() -> dict[str, Any]:
    files = local_coverage()
    covered = covered_functions(files)
    return {
        "schema": "tensorcore.sparse_24_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_sparse_24_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "paths": {
            "build_dir": "/repo/build",
            "evidence": "/repo/build/sparse_24_runtime_evidence.json",
        },
        "build_features": {
            "cuda": False,
            "cusparseLt": False,
        },
        "checks": {
            "sparse_24": {
                "status": "passed",
                "runtime_status": "passed",
                "binary": "/repo/build/tests/test_sparse_24",
                "trace": "sparse_24",
            },
            "sparse_24_k_axis_contract": {
                "status": "passed",
                "runtime_status": "passed",
                "binary": "/repo/build/tests/test_sparse_24",
                "trace": "sparse_24",
                "layout": "linear_weight_N_by_K",
            },
            "sparse_24_hardware": {
                "status": "blocked",
                "runtime_status": "blocked",
                "backend": "dense_fallback",
                "provider": "dense_fallback",
                "hardware_path": "dense_fallback",
                "blocked_reason": "cusparseLt_unavailable",
            },
        },
        "trace": [
            {"name": "sparse_24", "cmd": ["test_sparse_24"], "cwd": "/repo", "rc": 0},
        ],
        "files": files,
        "summary": {
            "checks_passed": True,
            "hardware_accelerated": False,
            "hardware_path": "dense_fallback",
            "blocked_reasons": [],
            "failure_reasons": [],
            "hardware_blocked_reasons": ["sparse_24_hardware:cusparseLt_unavailable"],
            "required_functions": REQUIRED,
            "covered_functions": covered,
            "missing_functions": [],
            "required_hardware_functions": REQUIRED_HARDWARE,
            "missing_hardware_functions": REQUIRED_HARDWARE,
        },
    }


def passed_hardware_evidence() -> dict[str, Any]:
    evidence = passed_fallback_evidence()
    files = hardware_coverage()
    evidence["build_features"] = {"cuda": True, "cusparseLt": True}
    evidence["checks"]["sparse_24_hardware"] = {
        "status": "passed",
        "runtime_status": "passed",
        "backend": "cusparseLt",
        "provider": "cusparseLt",
        "hardware_path": "cusparseLt",
        "blocked_reason": None,
    }
    evidence["files"] = files
    evidence["summary"]["hardware_accelerated"] = True
    evidence["summary"]["hardware_path"] = "cusparseLt"
    evidence["summary"]["hardware_blocked_reasons"] = []
    evidence["summary"]["covered_functions"] = covered_functions(files)
    evidence["summary"]["missing_hardware_functions"] = []
    return evidence


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_fallback_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["sparse_24"] = {
        "status": "blocked",
        "runtime_status": "blocked",
        "blocked_reason": "test_binary_missing",
        "binary": "/repo/build/tests/test_sparse_24",
    }
    evidence["checks"]["sparse_24_k_axis_contract"]["status"] = "blocked"
    evidence["checks"]["sparse_24_k_axis_contract"]["runtime_status"] = "blocked"
    evidence["trace"] = []
    evidence["files"] = {}
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["sparse_24:test_binary_missing"]
    evidence["summary"]["covered_functions"] = []
    evidence["summary"]["missing_functions"] = REQUIRED
    evidence["summary"]["missing_hardware_functions"] = REQUIRED_HARDWARE
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
    fallback = passed_fallback_evidence()
    assert_passes(fallback)
    assert_passes(fallback, "--require-pass", "--git-head", "abc123", "--require-clean-head")
    assert_fails(fallback, "--require-hardware needs summary.hardware_accelerated=true", "--require-hardware")

    hardware = passed_hardware_evidence()
    assert_passes(hardware, "--require-pass", "--require-hardware")

    blocked = blocked_evidence()
    assert_passes(blocked)
    assert_fails(blocked, "--require-pass needs passed evidence", "--require-pass")

    missing_function = copy.deepcopy(fallback)
    del missing_function["files"]["lib/ops/sparse_gemm_cpu.cpp"]["functions"]["tc_sparse_24_gemm"]
    missing_function["summary"]["covered_functions"] = [
        item for item in fallback["summary"]["covered_functions"] if not item.endswith(":tc_sparse_24_gemm")
    ]
    missing_function["summary"]["missing_functions"] = [
        "lib/ops/sparse_gemm_cpu.cpp:tc_sparse_24_gemm"
    ]
    assert_fails(missing_function, "missing function coverage", "--require-pass")

    stale_summary = copy.deepcopy(fallback)
    stale_summary["summary"]["covered_functions"] = []
    assert_fails(stale_summary, "summary.covered_functions must match files coverage")

    dirty = copy.deepcopy(fallback)
    dirty["meta"]["git_dirty"] = True
    assert_fails(dirty, "clean git tree", "--git-head", "abc123", "--require-clean-head")

    stale = copy.deepcopy(fallback)
    stale["meta"]["git_head"] = "stale"
    assert_fails(stale, "git_head mismatch", "--git-head", "abc123", "--require-clean-head")

    print("Sparse 2:4 evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
