#!/usr/bin/env python3
"""Fixture tests for qLLM TensorCore dispatch evidence validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_qllm_tensorcore_dispatch_evidence.py"


def good() -> dict:
    digest = "a" * 64
    return {
        "schema": "qllm.tensorcore_dispatch.runtime.v1",
        "runtime_status": "passed",
        "runtime_available": True,
        "runtime_library_identity_matches": True,
        "capability_abi_version": 1,
        "available_backend_mask": 8,
        "gemm_capabilities": {
            dtype: {"bit": 1 << index, "known": True, "available": True}
            for index, dtype in enumerate(("f32", "f16", "bf16", "i8"))
        },
        "auto_dispatch": {
            "passed": True,
            "execution_backend": 2,
            "backend_name": "mps",
            "tensorcore_calls": 1,
            "fallback_calls": 0,
        },
        "linked_smoke": {"status": "passed", "returncode": 0},
        "qllm_library_sha256": digest,
        "tensorcore_library_sha256": digest,
        "linked_smoke_sha256": digest,
        "qllm_git_head": "qllm-head",
        "qllm_git_dirty": False,
        "tensorcore_git_head": "tensorcore-head",
        "tensorcore_git_dirty": False,
    }


def run(payload: dict, *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), *args],
            text=True, capture_output=True, check=False,
        )


def main() -> int:
    payload = good()
    assert run(
        payload,
        "--tensorcore-git-head", "tensorcore-head",
        "--qllm-git-head", "qllm-head",
        "--require-clean-heads",
    ).returncode == 0

    unavailable = copy.deepcopy(payload)
    unavailable["gemm_capabilities"]["bf16"]["available"] = False
    result = run(unavailable)
    assert result.returncode != 0 and "bf16 GEMM capability" in result.stderr

    fallback = copy.deepcopy(payload)
    fallback["auto_dispatch"]["fallback_calls"] = 1
    result = run(fallback)
    assert result.returncode != 0 and "unexpectedly used qLLM fallback" in result.stderr

    dirty = copy.deepcopy(payload)
    dirty["qllm_git_dirty"] = True
    result = run(
        dirty,
        "--tensorcore-git-head", "tensorcore-head",
        "--qllm-git-head", "qllm-head",
        "--require-clean-heads",
    )
    assert result.returncode != 0 and "qLLM evidence source must be clean" in result.stderr

    print("qLLM TensorCore dispatch evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
