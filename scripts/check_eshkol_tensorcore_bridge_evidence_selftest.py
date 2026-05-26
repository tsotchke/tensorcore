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
                "executed_lines": [
                    45,
                    49,
                    53,
                    57,
                    71,
                    75,
                    78,
                    88,
                    95,
                    101,
                    105,
                    109,
                    117,
                    144,
                    149,
                    153,
                    157,
                ],
                "functions": {
                    "tc-init": {"start_line": 45, "executed_lines": [45]},
                    "tc-shutdown": {"start_line": 49, "executed_lines": [49]},
                    "tc-device-name": {"start_line": 53, "executed_lines": [53]},
                    "tc-device-info": {"start_line": 57, "executed_lines": [57]},
                    "tc-buffer-alloc": {"start_line": 71, "executed_lines": [71]},
                    "tc-buffer-free": {"start_line": 75, "executed_lines": [75]},
                    "tc-buffer-map": {"start_line": 78, "executed_lines": [78]},
                    "tc-dtype-code": {"start_line": 88, "executed_lines": [88]},
                    "tc-gemm": {"start_line": 95, "executed_lines": [95]},
                    "tc-gemm-fp16": {"start_line": 101, "executed_lines": [101]},
                    "tc-gemm-fp32": {"start_line": 105, "executed_lines": [105]},
                    "tc-gemm-bf16": {"start_line": 109, "executed_lines": [109]},
                    "tc-attention-forward": {"start_line": 117, "executed_lines": [117]},
                    "tc-last-backend": {"start_line": 144, "executed_lines": [144]},
                    "tc-last-backend-name": {"start_line": 149, "executed_lines": [149]},
                    "tc-version": {"start_line": 153, "executed_lines": [153]},
                    "tc-status-string": {"start_line": 157, "executed_lines": [157]},
                },
            },
            "lib/c_api/eshkol_bridge.c": {
                "executed_lines": [
                    17,
                    21,
                    25,
                    35,
                    44,
                    48,
                    54,
                    63,
                    69,
                    75,
                    81,
                    87,
                    93,
                    102,
                    106,
                    115,
                    153,
                    188,
                    192,
                    196,
                    200,
                ],
                "functions": {
                    "bool_to_i32": {"start_line": 17, "executed_lines": [17]},
                    "normalize_status": {"start_line": 21, "executed_lines": [21]},
                    "dtype_from_eshkol": {"start_line": 25, "executed_lines": [25]},
                    "tc_eshkol_init": {"start_line": 35, "executed_lines": [35]},
                    "tc_eshkol_shutdown": {"start_line": 44, "executed_lines": [44]},
                    "get_device_info": {"start_line": 48, "executed_lines": [48]},
                    "tc_eshkol_device_name": {"start_line": 54, "executed_lines": [54]},
                    "tc_eshkol_device_family": {"start_line": 63, "executed_lines": [63]},
                    "tc_eshkol_device_unified_memory": {"start_line": 69, "executed_lines": [69]},
                    "tc_eshkol_device_supports_bf16": {"start_line": 75, "executed_lines": [75]},
                    "tc_eshkol_device_supports_i8": {"start_line": 81, "executed_lines": [81]},
                    "tc_eshkol_device_supports_tensorops_m5": {
                        "start_line": 87,
                        "executed_lines": [87],
                    },
                    "tc_eshkol_buffer_alloc": {"start_line": 93, "executed_lines": [93]},
                    "tc_eshkol_buffer_free": {"start_line": 102, "executed_lines": [102]},
                    "tc_eshkol_buffer_map": {"start_line": 106, "executed_lines": [106]},
                    "tc_eshkol_gemm": {"start_line": 115, "executed_lines": [115]},
                    "tc_eshkol_attention_forward": {"start_line": 153, "executed_lines": [153]},
                    "tc_eshkol_last_backend": {"start_line": 188, "executed_lines": [188]},
                    "tc_eshkol_last_backend_code": {"start_line": 192, "executed_lines": [192]},
                    "tc_eshkol_version": {"start_line": 196, "executed_lines": [196]},
                    "tc_eshkol_status_string": {"start_line": 200, "executed_lines": [200]},
                },
            },
            "lib/core/status.c": {
                "executed_lines": [3],
                "functions": {
                    "tc_status_string": {"start_line": 3, "executed_lines": [3]},
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
