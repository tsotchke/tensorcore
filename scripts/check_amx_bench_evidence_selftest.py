#!/usr/bin/env python3
"""Fixture tests for the AMX/bench runtime evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_amx_bench_evidence.py"


def coverage() -> dict[str, Any]:
    return {
        "lib/ops/gemm_cpu_amx.cpp": {
            "executed_lines": [394, 440, 520, 591, 623, 715, 840, 853, 876],
            "functions": {
                "amx_worker_local": {"start_line": 394, "executed_lines": [394]},
                "amx_process_tile_strip": {"start_line": 440, "executed_lines": [440]},
                "amx_worker_thread_entry": {"start_line": 520, "executed_lines": [520]},
                "amx_pool_dispatch_pair": {"start_line": 591, "executed_lines": [591]},
                "tc_amx_gemm_f32_core": {"start_line": 623, "executed_lines": [623]},
                "tc_amx_gemm_f32": {"start_line": 715, "executed_lines": [715]},
                "tc_amx_gemm_f32_available": {"start_line": 840, "executed_lines": [840]},
                "tc_amx_isa_version": {"start_line": 853, "executed_lines": [853]},
                "tc_amx_cluster_count": {"start_line": 876, "executed_lines": [876]},
            },
        },
        "bench/bench_gemm.c": {
            "executed_lines": [27, 33, 38, 46, 54, 69, 100, 116, 148, 156],
            "functions": {
                "now_seconds": {"start_line": 27, "executed_lines": [27]},
                "cmp_double": {"start_line": 33, "executed_lines": [33]},
                "trim_token": {"start_line": 38, "executed_lines": [38]},
                "only_spaces": {"start_line": 46, "executed_lines": [46]},
                "env_int": {"start_line": 54, "executed_lines": [54]},
                "parse_sizes": {"start_line": 69, "executed_lines": [69]},
                "parse_dtype_token": {"start_line": 100, "executed_lines": [100]},
                "parse_dtypes": {"start_line": 116, "executed_lines": [116]},
                "print_throughput": {"start_line": 148, "executed_lines": [148]},
                "bench_one": {"start_line": 156, "executed_lines": [156]},
            },
        },
        "bench/bench_attention.c": {
            "executed_lines": [17, 23, 28, 44],
            "functions": {
                "now_seconds": {"start_line": 17, "executed_lines": [17]},
                "cmp_double": {"start_line": 23, "executed_lines": [23]},
                "env_int": {"start_line": 28, "executed_lines": [28]},
                "bench_one": {"start_line": 44, "executed_lines": [44]},
            },
        },
    }


def passed_evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.amx_bench_runtime_evidence.v1",
        "meta": {
            "format": 1,
            "source": "tensorcore_amx_bench_probe",
            "git_head": "abc123",
            "git_dirty": False,
        },
        "status": "passed",
        "paths": {
            "build_dir": "/repo/build",
            "portable_build_dir": "/repo/build-portable-cpu-current",
            "evidence": "/repo/build/amx_bench_evidence.json",
        },
        "checks": {
            "amx_gemm": {
                "status": "passed",
                "binary": "/repo/build-portable-cpu-current/tests/test_amx_gemm",
                "trace": "amx_gemm",
            },
            "amx_probe": {
                "status": "passed",
                "binary": "/repo/build-portable-cpu-current/tests/test_amx_probe",
                "trace": "amx_probe",
            },
            "bench_gemm": {
                "status": "passed",
                "binary": "/repo/build/bench/bench_gemm",
                "trace": "bench_gemm",
            },
        },
        "optional_checks": {
            "bench_attention": {
                "status": "passed",
                "binary": "/repo/build/bench/bench_attention",
                "trace": "bench_attention",
            },
            "tensorops_layout": {
                "status": "skipped",
                "skip_reason": "skipped_no_metal4_sdk",
                "metal4_sdk_compiled": False,
                "tensorops_m5_source_compiled": False,
            },
        },
        "trace": [
            {"name": "amx_gemm", "cmd": ["test_amx_gemm"], "cwd": "/repo", "rc": 0},
            {"name": "amx_probe", "cmd": ["test_amx_probe"], "cwd": "/repo", "rc": 0},
            {"name": "bench_gemm", "cmd": ["bench_gemm"], "cwd": "/repo", "rc": 0},
            {"name": "bench_attention", "cmd": ["bench_attention"], "cwd": "/repo", "rc": 0},
        ],
        "files": coverage(),
        "summary": {
            "checks_passed": True,
            "blocked_reasons": [],
            "failure_reasons": [],
            "optional_skipped_reasons": ["tensorops_layout:skipped_no_metal4_sdk"],
            "required_functions": sorted([
                "bench/bench_gemm.c:bench_one",
                "bench/bench_gemm.c:cmp_double",
                "bench/bench_gemm.c:env_int",
                "bench/bench_gemm.c:now_seconds",
                "bench/bench_gemm.c:only_spaces",
                "bench/bench_gemm.c:parse_dtype_token",
                "bench/bench_gemm.c:parse_dtypes",
                "bench/bench_gemm.c:parse_sizes",
                "bench/bench_gemm.c:print_throughput",
                "bench/bench_gemm.c:trim_token",
                "lib/ops/gemm_cpu_amx.cpp:amx_process_tile_strip",
                "lib/ops/gemm_cpu_amx.cpp:amx_pool_dispatch_pair",
                "lib/ops/gemm_cpu_amx.cpp:amx_worker_local",
                "lib/ops/gemm_cpu_amx.cpp:amx_worker_thread_entry",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_cluster_count",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_available",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_core",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_isa_version",
            ]),
            "covered_functions": sorted([
                "bench/bench_attention.c:bench_one",
                "bench/bench_attention.c:cmp_double",
                "bench/bench_attention.c:env_int",
                "bench/bench_attention.c:now_seconds",
                "bench/bench_gemm.c:bench_one",
                "bench/bench_gemm.c:cmp_double",
                "bench/bench_gemm.c:env_int",
                "bench/bench_gemm.c:now_seconds",
                "bench/bench_gemm.c:only_spaces",
                "bench/bench_gemm.c:parse_dtype_token",
                "bench/bench_gemm.c:parse_dtypes",
                "bench/bench_gemm.c:parse_sizes",
                "bench/bench_gemm.c:print_throughput",
                "bench/bench_gemm.c:trim_token",
                "lib/ops/gemm_cpu_amx.cpp:amx_process_tile_strip",
                "lib/ops/gemm_cpu_amx.cpp:amx_pool_dispatch_pair",
                "lib/ops/gemm_cpu_amx.cpp:amx_worker_local",
                "lib/ops/gemm_cpu_amx.cpp:amx_worker_thread_entry",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_cluster_count",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_available",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_core",
                "lib/ops/gemm_cpu_amx.cpp:tc_amx_isa_version",
            ]),
            "missing_functions": [],
            "optional_missing_functions": [
                "lib/ops/gemm.mm:gemm_uses_default_layout",
                "lib/tensorops/tensorops_m5.mm:uses_default_layout",
            ],
        },
    }


def blocked_evidence() -> dict[str, Any]:
    evidence = passed_evidence()
    evidence["status"] = "blocked"
    evidence["checks"]["amx_gemm"] = {
        "status": "blocked",
        "blocked_reason": "test_binary_missing",
        "binary": None,
    }
    amx_probe_coverage = {
        "executed_lines": [840, 853, 876],
        "functions": {
            "tc_amx_gemm_f32_available": {"start_line": 840, "executed_lines": [840]},
            "tc_amx_isa_version": {"start_line": 853, "executed_lines": [853]},
            "tc_amx_cluster_count": {"start_line": 876, "executed_lines": [876]},
        },
    }
    evidence["files"] = {
        "bench/bench_attention.c": coverage()["bench/bench_attention.c"],
        "bench/bench_gemm.c": coverage()["bench/bench_gemm.c"],
        "lib/ops/gemm_cpu_amx.cpp": amx_probe_coverage,
    }
    evidence["summary"]["checks_passed"] = False
    evidence["summary"]["blocked_reasons"] = ["amx_gemm:test_binary_missing"]
    evidence["summary"]["covered_functions"] = sorted([
        "bench/bench_attention.c:bench_one",
        "bench/bench_attention.c:cmp_double",
        "bench/bench_attention.c:env_int",
        "bench/bench_attention.c:now_seconds",
        "bench/bench_gemm.c:bench_one",
        "bench/bench_gemm.c:cmp_double",
        "bench/bench_gemm.c:env_int",
        "bench/bench_gemm.c:now_seconds",
        "bench/bench_gemm.c:only_spaces",
        "bench/bench_gemm.c:parse_dtype_token",
        "bench/bench_gemm.c:parse_dtypes",
        "bench/bench_gemm.c:parse_sizes",
        "bench/bench_gemm.c:print_throughput",
        "bench/bench_gemm.c:trim_token",
        "lib/ops/gemm_cpu_amx.cpp:tc_amx_cluster_count",
        "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_available",
        "lib/ops/gemm_cpu_amx.cpp:tc_amx_isa_version",
    ])
    evidence["summary"]["missing_functions"] = sorted([
        "lib/ops/gemm_cpu_amx.cpp:amx_process_tile_strip",
        "lib/ops/gemm_cpu_amx.cpp:amx_pool_dispatch_pair",
        "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32",
        "lib/ops/gemm_cpu_amx.cpp:tc_amx_gemm_f32_core",
        "lib/ops/gemm_cpu_amx.cpp:amx_worker_local",
        "lib/ops/gemm_cpu_amx.cpp:amx_worker_thread_entry",
    ])
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
    del missing_function["files"]["lib/ops/gemm_cpu_amx.cpp"]["functions"]["amx_worker_local"]
    missing_function["summary"]["covered_functions"].remove(
        "lib/ops/gemm_cpu_amx.cpp:amx_worker_local"
    )
    missing_function["summary"]["missing_functions"] = [
        "lib/ops/gemm_cpu_amx.cpp:amx_worker_local"
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

    print("AMX/bench evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
