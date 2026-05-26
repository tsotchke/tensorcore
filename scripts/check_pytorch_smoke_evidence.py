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

REQUIRED_FUNCTIONS = {
    "bindings/pytorch/tensorcore_torch/__init__.py": {
        "_privateuse1_backend_name",
        "_ensure_privateuse1_name",
        "_device_index",
        "_check_device",
        "_torch_backend_module",
        "_torch_backend_module_registered",
        "_new_backend_module",
        "_ensure_generated_methods",
        "_ensure_torch_backend_module",
        "pytorch_backend_registered",
        "pytorch_backend_state",
        "pytorch_backend_report",
    },
    "bindings/pytorch/tensorcore_torch_ext.cpp": {
        "register_tensorcore_allocator",
        "is_tensorcore_device",
        "is_host_accessible_device_pair",
        "tc_matmul_eligibility_reason",
        "is_tc_matmul_eligible",
        "tc_matmul_eligibility",
        "tc_matmul_fp32",
        "tc_matmul_bf16",
        "tc_last_backend_name",
        "tc_matmul_dispatch",
        "tc_matmul_autograd_cpu",
        "tc_matmul_privateuse1",
        "tc_empty_memory_format",
        "tc_empty_strided",
        "tc_to_tensorcore",
        "tc_to_cpu",
        "tc_set_default_matmul",
        "tc_default_matmul_enabled",
        "tc_privateuse1_backend_name",
    },
}


def fail(message: str) -> int:
    print(f"PyTorch smoke evidence invalid: {message}", file=sys.stderr)
    return 1


def covered_functions(evidence: dict) -> dict[str, set[str]]:
    files = evidence.get("files")
    if not isinstance(files, dict):
        return {}
    covered: dict[str, set[str]] = {}
    for rel_path, entry in files.items():
        if not isinstance(entry, dict):
            continue
        functions = entry.get("functions")
        if not isinstance(functions, dict):
            continue
        covered[str(rel_path)] = {str(name) for name in functions}
    return covered


def check_function_coverage(evidence: dict) -> str | None:
    covered = covered_functions(evidence)
    missing: list[str] = []
    for rel_path, names in REQUIRED_FUNCTIONS.items():
        present = covered.get(rel_path, set())
        for name in names:
            if name not in present:
                missing.append(f"{rel_path}:{name}")
    if missing:
        return f"passed evidence is missing function coverage: {sorted(missing)!r}"
    return None


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
        if state_allocation:
            for key in ("privateuse1_matmul_checked", "device_roundtrip_checked"):
                if matmul.get(key) is not True:
                    return fail(f"{key} must be true when allocation is available")
        coverage_error = check_function_coverage(evidence)
        if coverage_error is not None:
            return fail(coverage_error)

    print(
        "PyTorch smoke evidence OK: "
        f"status={status} torch={evidence.get('torch_version') or 'none'} "
        f"allocation={bool(allocation.get('available'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
