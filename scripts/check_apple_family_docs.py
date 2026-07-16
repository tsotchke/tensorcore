#!/usr/bin/env python3
"""Enforce the public Apple-family and Metal matrix-element contract."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "apple_family_runtime.json"
TEXT_SUFFIXES = {".c", ".cc", ".cff", ".cpp", ".h", ".md", ".metal", ".mm", ".py", ".yml", ".yaml"}
FORBIDDEN = {
    "M4 mapped to Apple10": re.compile(r"(?:M4\s*(?:\([^)]*)?\bApple10\b|Apple10\s*\|\s*M4)", re.I),
    "M5 mapped to Apple11": re.compile(r"(?:M5\s*(?:\([^)]*)?\bApple11\b|Apple11\s*\|\s*M5|Apple11/M5)", re.I),
    "integer Apple simdgroup claim": re.compile(
        r"(?:(?:TC_DTYPE_I8|int8|i8)[^\n]{0,60}"
        r"(?:simdgroup_matrix\s+(?:on\s+)?Apple10|native\s+(?:on\s+)?Apple10|Apple10\+\s+native)|"
        r"simdgroup_matrix[^\n]{0,80}(?:8×8\s+i8|fp16/bf16/fp32/int8|int8\s+variant|i8\s+variant))",
        re.I,
    ),
    "unbounded FP32-widen exactness claim": re.compile(r"exact.{0,40}K\s*(?:=|≤|<=)\s*2\^16", re.I),
}


def tracked_files() -> list[pathlib.Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def scan_forbidden(errors: list[str]) -> None:
    self_path = pathlib.Path(__file__).resolve()
    for path in tracked_files():
        if path.resolve() == self_path or (path.suffix.lower() not in TEXT_SUFFIXES and path.name != "ROADMAP"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in FORBIDDEN.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{path.relative_to(ROOT)}:{line}: {label}")


def check_policy(errors: list[str]) -> None:
    try:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not load policy: {exc}")
        return
    expected = [
        ("Apple7", 1007, 7, ["M1"]),
        ("Apple8", 1008, 8, ["M2"]),
        ("Apple9", 1009, 9, ["M3", "M4"]),
        ("Apple10", 1010, 10, ["M5"]),
    ]
    actual = [
        (row.get("mtl_family"), row.get("mtl_raw_value"), row.get("tensorcore_family"), row.get("apple_silicon"))
        for row in policy.get("families", [])
    ]
    if actual != expected:
        errors.append(f"family policy mismatch: {actual!r} != {expected!r}")
    reserved = policy.get("reserved_public_abi_values")
    if not isinstance(reserved, list) or len(reserved) != 1 or reserved[0].get("tensorcore_family") != 11 or reserved[0].get("status") != "reserved":
        errors.append("Apple11 must be the sole reserved public ABI family value")
    matrix = policy.get("simdgroup_matrix", {})
    if matrix.get("public_element_types") != ["half", "bfloat", "float"]:
        errors.append("public simdgroup_matrix types must be half, bfloat, float")
    if matrix.get("integer_element_types_publicly_supported") is not False:
        errors.append("integer simdgroup_matrix support must remain false")
    tensorops = policy.get("tensorops", {})
    required_tensorops = {
        "minimum_chip": "M5",
        "required_mtl_family": "Apple10",
        "minimum_sdk": "26.0",
    }
    for key, expected_value in required_tensorops.items():
        if tensorops.get(key) != expected_value:
            errors.append(f"tensorops.{key} must be {expected_value!r}")
    evidence = policy.get("evidence", {})
    if evidence.get("m4_reserved_resource") != "enki:metal_m4_tsotchke_chan":
        errors.append("M4 policy must identify the reserved enki resource")
    if evidence.get("m4_reserved_resource_required_owner_prefix") != "tsotchke-chan":
        errors.append("reserved M4 evidence owner prefix must be tsotchke-chan")
    if evidence.get("allow_independent_m4_resource_evidence") is not True:
        errors.append("independently supplied M4 evidence must be allowed")
    if evidence.get("m4_independent_resource_requires_owner") is not True:
        errors.append("independent M4 evidence must require an accountable owner")


def check_sources(errors: list[str]) -> None:
    kernel = (ROOT / "kernels" / "metal" / "gemm_simdgroup.metal").read_text(encoding="utf-8")
    for token in ("simdgroup_matrix<char", "simdgroup_matrix<int", "tc_gemm_i8_i32"):
        if token in kernel:
            errors.append(f"unsupported integer Metal kernel token remains: {token}")
    gemm = (ROOT / "lib" / "ops" / "gemm.mm").read_text(encoding="utf-8")
    required = (
        "Public MSL simdgroup_matrix element types are half, bfloat, and float.",
        "d->a_dtype == TC_DTYPE_I8",
        "*err = TC_ERR_UNSUPPORTED_DTYPE",
    )
    for token in required:
        if token not in gemm:
            errors.append(f"lib/ops/gemm.mm missing integer fallback contract: {token}")
    family = (ROOT / "lib" / "core" / "apple_family.h").read_text(encoding="utf-8")
    for token in (
        "TC_MTL_GPU_FAMILY_APPLE10_RAW = 1010",
        "return family == TC_FAMILY_APPLE9 || family == TC_FAMILY_APPLE10;",
        "return family == TC_FAMILY_APPLE10;",
    ):
        if token not in family:
            errors.append(f"lib/core/apple_family.h missing policy token: {token}")


def check_contributor_intake(errors: list[str]) -> None:
    intake = (ROOT / "scripts" / "intake_apple_family_runtime_evidence.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "APPLE_FAMILY_EVIDENCE_HANDOFF",
        "evidence SHA-256 digest mismatch",
        "duplicate object key in evidence JSON",
        "evidence_checker.validate",
    ):
        if token not in intake:
            errors.append(f"Apple evidence intake verifier missing contract token: {token}")
    contribution = (ROOT / "docs" / "hardware_evidence_contribution.md").read_text(
        encoding="utf-8"
    )
    if "intake_apple_family_runtime_evidence.py" not in contribution:
        errors.append("hardware evidence contribution guide must use the intake verifier")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    if "intake_apple_family_runtime_evidence_selftest.py" not in workflow:
        errors.append("portable CI must run the Apple evidence intake selftest")


def main() -> int:
    errors: list[str] = []
    check_policy(errors)
    check_sources(errors)
    check_contributor_intake(errors)
    scan_forbidden(errors)
    if errors:
        print("Apple family documentation contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Apple family documentation contract OK: M4=Apple9 M5=Apple10 Apple11=reserved i8=MPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
