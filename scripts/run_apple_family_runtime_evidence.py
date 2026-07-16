#!/usr/bin/env python3
"""Collect physical Apple-family runtime evidence for TensorCore."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "apple_family_runtime.json"
SCHEMA = "tensorcore.apple_family_runtime_evidence.v1"
FORMAT_VERSION = 1
TESTS = ("test_device", "test_gemm_bf16", "test_gemm_i8", "test_tensorops_runtime")
BUILD_TRACE = "build_runtime_tests"
RESERVED_M4_RESOURCE = "enki:metal_m4_tsotchke_chan"
CHIP_RE = re.compile(r"\bApple M([1-5])(?:\s|\b)")
DEVICE_RE = re.compile(
    r'device="([^"]+)" family=Apple(\d+).*?bf16_sg=(yes|no) '
    r'i8_sg=(yes|no) tensorops_m5=(yes|no)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "apple_family_runtime_evidence.json",
    )
    parser.add_argument(
        "--hardware-resource",
        default=os.environ.get("TC_HARDWARE_RESOURCE", ""),
        help=(
            "Stable operator resource identifier. Required for M4 so the collector can "
            "distinguish the reserved enki resource from independently supplied hardware."
        ),
    )
    parser.add_argument("--authority-owner", default=os.environ.get("TC_AUTHORITY_OWNER", ""))
    parser.add_argument("--expected-chip", choices=[f"M{i}" for i in range(1, 6)])
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def command(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        digest = hashlib.sha256((stdout + "\0" + stderr).encode()).hexdigest()
        return {
            "cmd": cmd,
            "rc": proc.returncode,
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-20000:],
            "output_sha256": digest,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"cmd": cmd, "rc": None, "stdout_tail": "", "stderr_tail": str(exc)}


def git_value(*args: str) -> str | None:
    attempt = command(["git", *args])
    return attempt["stdout_tail"].strip() if attempt.get("rc") == 0 else None


def tracked_dirty() -> bool | None:
    unstaged = command(["git", "diff", "--quiet"])
    staged = command(["git", "diff", "--cached", "--quiet"])
    if unstaged.get("rc") is None or staged.get("rc") is None:
        return None
    return unstaged["rc"] != 0 or staged["rc"] != 0


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def chip_from_text(text: str) -> str | None:
    match = CHIP_RE.search(text)
    return f"M{match.group(1)}" if match else None


def host_chip(timeout: float) -> tuple[str | None, dict[str, Any]]:
    attempt = command(["system_profiler", "SPHardwareDataType", "-json"], timeout=timeout)
    # Do not retain system_profiler output: it contains host serial identifiers.
    chip = chip_from_text(attempt.get("stdout_tail", ""))
    safe_attempt = {
        "cmd": attempt["cmd"],
        "rc": attempt["rc"],
        "output_sha256": attempt.get("output_sha256"),
    }
    return chip, safe_attempt


def owner_authorized(owner: str) -> bool:
    return owner == "tsotchke-chan" or owner.startswith("tsotchke-chan:")


def resource_scope(resource: str) -> str:
    if resource == RESERVED_M4_RESOURCE:
        return "reserved"
    return "independent" if resource else "unspecified"


def resource_authorized(chip: str | None, resource: str, owner: str) -> bool:
    if chip != "M4":
        return True
    scope = resource_scope(resource)
    if scope == "reserved":
        return owner_authorized(owner)
    return scope == "independent" and bool(owner.strip())


def write_evidence(path: pathlib.Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output(attempt: dict[str, Any]) -> str:
    return "\n".join((str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))))


def cmake_source_root(build_dir: pathlib.Path) -> pathlib.Path | None:
    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
    for line in lines:
        if line.startswith(prefix):
            return pathlib.Path(line[len(prefix):]).resolve()
    return None


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_record(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    records: set[tuple[str, int, bool, bool, bool]] = set()
    for attempt in trace:
        for match in DEVICE_RE.finditer(output(attempt)):
            records.add(
                (
                    match.group(1),
                    int(match.group(2)),
                    match.group(3) == "yes",
                    match.group(4) == "yes",
                    match.group(5) == "yes",
                )
            )
    if len(records) != 1:
        return None
    name, family, bf16, i8, tensorops = records.pop()
    return {
        "name": name,
        "chip": chip_from_text(name),
        "family": family,
        "bf16_simdgroup": bf16,
        "i8_simdgroup": i8,
        "tensorops_m5": tensorops,
    }


def test_checks(trace: list[dict[str, Any]], chip: str | None) -> dict[str, Any]:
    attempts = {str(item.get("name")): item for item in trace}
    checks: dict[str, Any] = {}
    for test_name in TESTS:
        attempt = attempts.get(test_name, {})
        text = output(attempt)
        passed = attempt.get("rc") == 0
        if test_name == "test_device":
            passed = passed and "family policy" in text and "verified" in text
        elif test_name == "test_gemm_bf16":
            passed = passed and "mps_bf16_sw_fallback" in text and "padding=OK  OK" in text
            if chip in {"M1", "M2"}:
                passed = passed and "lacks bf16 simdgroup_matrix" in text
            else:
                passed = passed and "supports bf16 simdgroup_matrix" in text
        elif test_name == "test_gemm_i8":
            passed = (
                passed
                and "public MPS i8 fallback" in text
                and "mps_i8_sw_fallback" in text
                and "errors=0/12" in text
                and "padding=OK  OK" in text
                and not re.search(r"backend=(?!mps\b)\S+", text)
            )
        elif test_name == "test_tensorops_runtime":
            expected = "passed" if chip == "M5" else "skipped_no_m5"
            passed = passed and f"tensorops_runtime_status={expected}" in text
            if chip == "M5":
                passed = passed and "backend=tensorops_m5" in text
        checks[test_name] = {"status": "passed" if passed else "failed"}
    return checks


def base_evidence(args: argparse.Namespace, chip: str | None, host_attempt: dict[str, Any]) -> dict[str, Any]:
    sdk_attempt = command(["xcrun", "--show-sdk-version"], timeout=args.timeout_sec)
    sdk = sdk_attempt.get("stdout_tail", "").strip() if sdk_attempt.get("rc") == 0 else ""
    hardware_resource = args.hardware_resource.strip()
    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_apple_family_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": tracked_dirty(),
            "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "chip": chip,
            "sdk_version": sdk,
            "identity_probe": host_attempt,
        },
        "reservation": {
            "resource": hardware_resource or None,
            "scope": resource_scope(hardware_resource),
            "authority_owner": args.authority_owner,
            "authorized": resource_authorized(chip, hardware_resource, args.authority_owner),
        },
        "build": {
            "directory": str(args.build_dir.resolve()),
            "source_root_matches_repo": False,
            "binary_sha256": {},
        },
        "device": None,
        "checks": {},
        "trace": [],
        "status": "blocked",
        "summary": {"blocked_reasons": [], "failure_reasons": []},
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    chip, host_attempt = host_chip(args.timeout_sec)
    evidence = base_evidence(args, chip, host_attempt)
    blocked = evidence["summary"]["blocked_reasons"]
    failures = evidence["summary"]["failure_reasons"]

    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        blocked.append("physical_apple_silicon_host_required")
    if chip is None:
        blocked.append("apple_chip_identity_unavailable")
    if args.expected_chip and chip != args.expected_chip:
        failures.append(f"expected_chip_mismatch:{args.expected_chip}:{chip or 'unknown'}")
    if chip == "M4":
        reservation = evidence["reservation"]
        if reservation["scope"] == "unspecified":
            blocked.append("m4_resource_provenance_required")
        elif reservation["scope"] == "independent" and not reservation["authority_owner"].strip():
            blocked.append("independent_m4_owner_required")
        elif not reservation["authorized"]:
            blocked.append("reserved_m4_owner_not_authorized")

    # Reservation failures must stop before any TensorCore GPU binary executes.
    if blocked or failures:
        evidence["status"] = "failed" if failures else "blocked"
        return evidence

    build_dir = args.build_dir.resolve()
    source_root = cmake_source_root(build_dir)
    evidence["build"]["source_root_matches_repo"] = source_root == ROOT.resolve()
    if not evidence["build"]["source_root_matches_repo"]:
        blocked.append("cmake_build_source_root_mismatch")
        evidence["status"] = "blocked"
        return evidence

    build_attempt = command(
        [
            "cmake", "--build", str(build_dir), "--target", *TESTS, "--parallel",
        ],
        timeout=args.timeout_sec,
    )
    build_attempt["name"] = BUILD_TRACE
    evidence["trace"].append(build_attempt)
    if build_attempt.get("rc") != 0:
        failures.append("runtime_test_build_failed")
        evidence["status"] = "failed"
        return evidence

    env = os.environ.copy()
    metallib = build_dir / "tensorcore.metallib"
    if metallib.exists():
        env["TC_METALLIB"] = str(metallib)
    for name in TESTS:
        binary = build_dir / "tests" / name
        if not binary.exists():
            blocked.append(f"test_binary_missing:{name}")
            continue
        evidence["build"]["binary_sha256"][name] = file_sha256(binary)
        attempt = command([str(binary)], env=env, timeout=args.timeout_sec)
        attempt["name"] = name
        evidence["trace"].append(attempt)

    evidence["device"] = device_record(evidence["trace"])
    evidence["checks"] = test_checks(evidence["trace"], chip)
    if evidence["device"] is None:
        failures.append("inconsistent_or_missing_device_record")
    if chip == "M5" and version_tuple(evidence["host"]["sdk_version"]) < (26, 0):
        failures.append("m5_requires_sdk26")
    for name, check in evidence["checks"].items():
        if check["status"] != "passed":
            failures.append(f"runtime_check_failed:{name}")
    if evidence["meta"]["git_dirty"] is not False:
        failures.append("tracked_git_tree_not_clean")
    evidence["status"] = "failed" if failures else ("blocked" if blocked else "passed")
    return evidence


def main() -> int:
    args = parse_args()
    evidence = collect(args)
    write_evidence(args.evidence_path, evidence)
    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        print(
            "Apple family runtime evidence "
            f"status={evidence['status']} chip={evidence['host']['chip'] or 'unknown'} "
            f"path={args.evidence_path}"
        )
    return 1 if args.require_pass and evidence["status"] != "passed" else 0


if __name__ == "__main__":
    sys.exit(main())
