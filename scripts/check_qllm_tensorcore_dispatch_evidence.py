#!/usr/bin/env python3
"""Validate qLLM-owned TensorCore dispatch runtime evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


SCHEMA = "qllm.tensorcore_dispatch.runtime.v1"
GEMM_DTYPES = ("f32", "f16", "bf16", "i8")


def fail(message: str) -> int:
    print(f"qLLM TensorCore dispatch evidence invalid: {message}", file=sys.stderr)
    return 1


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tensorcore-git-head", default=git_head())
    parser.add_argument("--qllm-git-head")
    parser.add_argument("--require-clean-heads", action="store_true")
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")
    if evidence.get("schema") != SCHEMA:
        return fail(f"schema must be {SCHEMA}")
    if evidence.get("runtime_status") != "passed":
        return fail("runtime_status must be passed")
    if evidence.get("runtime_available") is not True:
        return fail("qLLM did not report the TensorCore runtime available")
    if evidence.get("runtime_library_identity_matches") is not True:
        return fail("qLLM did not load the supplied TensorCore library artifact")
    if evidence.get("capability_abi_version") != 1:
        return fail("qLLM did not consume TensorCore capability ABI v1")
    if int(evidence.get("available_backend_mask") or 0) == 0:
        return fail("available_backend_mask must name at least one runtime backend")

    capabilities = evidence.get("gemm_capabilities")
    if not isinstance(capabilities, dict):
        return fail("gemm_capabilities must be an object")
    for dtype in GEMM_DTYPES:
        row = capabilities.get(dtype)
        if not isinstance(row, dict) or row.get("known") is not True:
            return fail(f"{dtype} GEMM capability is not known")
        if row.get("available") is not True:
            return fail(f"{dtype} GEMM capability is not available")

    auto = evidence.get("auto_dispatch")
    if not isinstance(auto, dict) or auto.get("passed") is not True:
        return fail("AUTO dispatch probe did not pass")
    if auto.get("execution_backend") != 2:
        return fail("AUTO dispatch did not report qLLM TensorCore execution")
    if int(auto.get("tensorcore_calls") or 0) != 1:
        return fail("AUTO dispatch must prove exactly one TensorCore call")
    if int(auto.get("fallback_calls") or 0) != 0:
        return fail("AUTO dispatch unexpectedly used qLLM fallback")
    if auto.get("backend_name") in (None, "", "none", "qllm-fallback"):
        return fail("AUTO dispatch lacks an explicit native backend")

    linked = evidence.get("linked_smoke")
    if not isinstance(linked, dict) or linked.get("status") != "passed":
        return fail("linked four-dtype smoke did not pass")
    for key in (
        "qllm_library_sha256", "tensorcore_library_sha256",
        "linked_smoke_sha256",
    ):
        if not valid_sha256(evidence.get(key)):
            return fail(f"{key} must be a SHA-256 digest")

    if args.tensorcore_git_head and evidence.get("tensorcore_git_head") != args.tensorcore_git_head:
        return fail("TensorCore git head does not match the expected revision")
    if args.qllm_git_head and evidence.get("qllm_git_head") != args.qllm_git_head:
        return fail("qLLM git head does not match the expected revision")
    if args.require_clean_heads:
        if evidence.get("tensorcore_git_dirty") is not False:
            return fail("TensorCore evidence source must be clean")
        if evidence.get("qllm_git_dirty") is not False:
            return fail("qLLM evidence source must be clean")

    print(
        "qLLM TensorCore dispatch evidence OK: "
        f"backend={auto.get('backend_name')} dtypes={','.join(GEMM_DTYPES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
