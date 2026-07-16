#!/usr/bin/env python3
"""Validate physical Apple-family runtime evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.apple_family_runtime_evidence.v1"
EXPECTED = {
    "M1": (7, False, False, False),
    "M2": (8, False, False, False),
    "M3": (9, True, False, False),
    "M4": (9, True, False, False),
    "M5": (10, True, False, True),
}
TESTS = ("test_device", "test_gemm_bf16", "test_gemm_i8", "test_tensorops_runtime")
BUILD_TRACE = "build_runtime_tests"
DEVICE_RE = re.compile(
    r'device="([^"]+)" family=Apple(\d+).*?bf16_sg=(yes|no) '
    r'i8_sg=(yes|no) tensorops_m5=(yes|no)'
)


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-clean-head", action="store_true")
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-chip", choices=sorted(EXPECTED))
    return parser.parse_args()


def version_tuple(value: Any) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value or ""))
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def owner_authorized(owner: Any) -> bool:
    return owner == "tsotchke-chan" or (isinstance(owner, str) and owner.startswith("tsotchke-chan:"))


def trace_output(attempt: dict[str, Any]) -> str:
    return "\n".join((str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))))


def validate(data: Any, args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["evidence root must be an object"]
    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    meta = data.get("meta")
    if not isinstance(meta, dict) or meta.get("format") != 1:
        errors.append("meta.format must be 1")
        meta = {}
    if meta.get("source") != "tensorcore_apple_family_probe":
        errors.append("meta.source must be tensorcore_apple_family_probe")
    host = data.get("host")
    if not isinstance(host, dict):
        errors.append("host must be an object")
        host = {}
    if host.get("system") != "Darwin" or host.get("machine") not in {"arm64", "aarch64"}:
        errors.append("evidence must come from physical Darwin/arm64 Apple silicon")
    chip = host.get("chip")
    if chip not in EXPECTED:
        errors.append(f"unsupported or missing chip class: {chip!r}")
    if args.require_chip and chip != args.require_chip:
        errors.append(f"required chip {args.require_chip}, got {chip!r}")

    reservation = data.get("reservation")
    if not isinstance(reservation, dict):
        errors.append("reservation must be an object")
        reservation = {}
    if chip == "M4":
        if reservation.get("resource") != "enki:metal_m4_tsotchke_chan":
            errors.append("M4 evidence must identify the reserved enki resource")
        if not owner_authorized(reservation.get("authority_owner")):
            errors.append("M4 evidence authority owner must use the tsotchke-chan prefix")
        if reservation.get("authorized") is not True:
            errors.append("M4 evidence reservation must be authorized")

    device = data.get("device")
    if not isinstance(device, dict):
        errors.append("device must be an object")
        device = {}
    if chip in EXPECTED:
        family, bf16, i8, tensorops = EXPECTED[chip]
        for field, expected in (
            ("chip", chip),
            ("family", family),
            ("bf16_simdgroup", bf16),
            ("i8_simdgroup", i8),
            ("tensorops_m5", tensorops),
        ):
            if device.get(field) != expected:
                errors.append(f"device.{field} must be {expected!r} for {chip}, got {device.get(field)!r}")
        if chip == "M5" and version_tuple(host.get("sdk_version")) < (26, 0):
            errors.append("M5 TensorOps evidence requires SDK 26.0 or newer")

    build = data.get("build")
    if not isinstance(build, dict):
        errors.append("build must be an object")
        build = {}
    if build.get("source_root_matches_repo") is not True:
        errors.append("CMake build directory must be configured from the evidence repository")
    binary_hashes = build.get("binary_sha256")
    if not isinstance(binary_hashes, dict) or set(binary_hashes) != set(TESTS):
        errors.append("build.binary_sha256 must contain exactly the four runtime tests")
    elif any(not isinstance(binary_hashes[name], str) or not re.fullmatch(r"[0-9a-f]{64}", binary_hashes[name]) for name in TESTS):
        errors.append("every runtime test must include a SHA-256 binary digest")

    trace = data.get("trace")
    if not isinstance(trace, list):
        errors.append("trace must be a list")
        trace = []
    attempts = {item.get("name"): item for item in trace if isinstance(item, dict)}
    required_traces = (BUILD_TRACE, *TESTS)
    if set(attempts) != set(required_traces):
        errors.append(f"trace must contain exactly {list(required_traces)!r}")
    build_attempt = attempts.get(BUILD_TRACE)
    if isinstance(build_attempt, dict):
        argv = build_attempt.get("cmd")
        if not isinstance(argv, list) or argv[:2] != ["cmake", "--build"] or "--target" not in argv:
            errors.append("build trace must use cmake --build with explicit test targets")
        elif any(name not in argv for name in TESTS):
            errors.append("build trace must target all four runtime tests")
    for name in required_traces:
        attempt = attempts.get(name)
        if not isinstance(attempt, dict):
            continue
        if attempt.get("rc") != 0:
            errors.append(f"{name} must exit zero")
        digest = attempt.get("output_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{name} must include output_sha256")

    outputs = {name: trace_output(item) for name, item in attempts.items() if name in TESTS}
    all_device_records: set[tuple[str, int, bool, bool, bool]] = set()
    for text in outputs.values():
        for match in DEVICE_RE.finditer(text):
            all_device_records.add(
                (
                    match.group(1), int(match.group(2)), match.group(3) == "yes",
                    match.group(4) == "yes", match.group(5) == "yes",
                )
            )
    expected_record = (
        device.get("name"), device.get("family"), device.get("bf16_simdgroup"),
        device.get("i8_simdgroup"), device.get("tensorops_m5"),
    )
    if all_device_records != {expected_record}:
        errors.append("runtime trace device records must be consistent with device summary")

    device_text = outputs.get("test_device", "")
    if "family policy" not in device_text or "verified" not in device_text:
        errors.append("test_device trace lacks verified family policy marker")
    bf16_text = outputs.get("test_gemm_bf16", "")
    if "mps_bf16_sw_fallback" not in bf16_text or "padding=OK  OK" not in bf16_text:
        errors.append("BF16 trace lacks correctness and padded-layout markers")
    if chip in {"M1", "M2"} and "lacks bf16 simdgroup_matrix" not in bf16_text:
        errors.append("pre-Apple9 BF16 trace must prove fallback selection")
    if chip in {"M3", "M4", "M5"} and "supports bf16 simdgroup_matrix" not in bf16_text:
        errors.append("Apple9+ BF16 trace must prove simdgroup capability selection")
    i8_text = outputs.get("test_gemm_i8", "")
    if (
        "public MPS i8 fallback" not in i8_text
        or "mps_i8_sw_fallback" not in i8_text
        or "errors=0/12" not in i8_text
        or "padding=OK  OK" not in i8_text
    ):
        errors.append("integer trace lacks bit-exact public MPS fallback markers")
    if re.search(r"backend=(?!mps\b)\S+", i8_text):
        errors.append("integer trace selected a non-MPS backend")
    tensor_text = outputs.get("test_tensorops_runtime", "")
    tensor_status = "passed" if chip == "M5" else "skipped_no_m5"
    if f"tensorops_runtime_status={tensor_status}" not in tensor_text:
        errors.append(f"TensorOps trace must report {tensor_status} for {chip}")
    if chip == "M5" and "backend=tensorops_m5" not in tensor_text:
        errors.append("M5 TensorOps trace must select tensorops_m5")

    checks = data.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(TESTS):
        errors.append("checks must contain exactly the four Apple runtime tests")
    else:
        for name in TESTS:
            if not isinstance(checks.get(name), dict):
                errors.append(f"checks.{name} must be an object")
            elif checks[name].get("status") != "passed":
                errors.append(f"checks.{name}.status must be passed")
    if args.require_clean_head:
        if meta.get("git_dirty") is not False:
            errors.append("runtime evidence must come from a clean tracked git tree")
        if not args.git_head or meta.get("git_head") != args.git_head:
            errors.append(f"runtime evidence git_head mismatch: {meta.get('git_head')!r} != {args.git_head!r}")
    if args.require_pass and data.get("status") != "passed":
        errors.append(f"--require-pass needs status=passed, got {data.get('status')!r}")
    if data.get("status") == "passed" and errors:
        errors.append("evidence status is passed but its derived contract is invalid")
    return errors


def main() -> int:
    args = parse_args()
    try:
        data = json.loads(args.evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read Apple family runtime evidence: {exc}", file=sys.stderr)
        return 1
    errors = validate(data, args)
    if errors:
        print("Apple family runtime evidence invalid:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Apple family runtime evidence OK: status={data.get('status')} chip={data['host']['chip']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
