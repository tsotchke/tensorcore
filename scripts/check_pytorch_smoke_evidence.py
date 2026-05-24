#!/usr/bin/env python3
"""Validate machine-readable evidence from scripts/ci_pytorch_smoke.sh."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


VALID_STATUSES = {
    "passed",
    "skipped_torch_unavailable",
    "skipped_native_lib_missing",
}


def fail(message: str) -> int:
    print(f"PyTorch smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--require-pytorch", action="store_true")
    parser.add_argument("--require-backend-allocation", action="store_true")
    args = parser.parse_args()

    try:
        evidence = json.loads(args.path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"could not read JSON: {exc}")

    if evidence.get("schema_version") != 1:
        return fail("schema_version must be 1")
    status = evidence.get("runtime_status")
    if status not in VALID_STATUSES:
        return fail(f"unexpected runtime_status={status!r}")

    if args.require_pytorch and status != "passed":
        return fail(f"--require-pytorch needs passed evidence, got {status}")

    allocation = evidence.get("direct_device_allocation")
    if not isinstance(allocation, dict):
        return fail("direct_device_allocation must be an object")
    if args.require_backend_allocation and allocation.get("available") is not True:
        return fail("--require-backend-allocation needs tensorcore device allocation")

    if status == "passed":
        if not evidence.get("torch_version"):
            return fail("passed evidence must include torch_version")
        state = evidence.get("backend_state")
        if not isinstance(state, dict):
            return fail("passed evidence must include backend_state")
        if state.get("backend_name") != "tensorcore":
            return fail("backend_state.backend_name must be tensorcore")
        if state.get("privateuse1_backend_name") != "tensorcore":
            return fail("PrivateUse1 backend name must be tensorcore")
        if state.get("registered") is not True:
            return fail("backend_state.registered must be true")
        if state.get("torch_module_registered") is not True:
            return fail("torch.tensorcore module must be registered")
        if state.get("generated_tensor_methods") is not True:
            return fail("PrivateUse1 tensor helper methods must be generated")
        if state.get("matmul_extension_loaded") is not True:
            return fail("matmul extension must be loaded")
        probe = state.get("matmul_dispatch_probe")
        if not isinstance(probe, dict) or probe.get("reason") != "eligible":
            return fail("matmul dispatch probe must report eligible")

        matmul = evidence.get("matmul")
        if not isinstance(matmul, dict):
            return fail("passed evidence must include matmul object")
        if matmul.get("fp32_eligibility_reason") != "eligible":
            return fail("fp32 matmul eligibility must be eligible")
        if not matmul.get("fp32_backend"):
            return fail("fp32_backend must be populated")
        for key in (
            "bf16_checked",
            "noncontiguous_checked",
            "degenerate_checked",
            "error_paths_checked",
            "default_matmul_dispatch_checked",
            "autograd_fallback_checked",
        ):
            if matmul.get(key) is not True:
                return fail(f"{key} must be true")

        state_allocation = bool(state.get("supports_device_allocation"))
        evidence_allocation = bool(allocation.get("available"))
        if evidence_allocation and not state_allocation:
            return fail("allocation evidence contradicts backend_state")
        if state.get("allocator_status") == "available" and not evidence_allocation:
            return fail("available allocator_status requires allocation evidence")

    print(
        "PyTorch smoke evidence OK: "
        f"status={status} torch={evidence.get('torch_version') or 'none'} "
        f"allocation={bool(allocation.get('available'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
