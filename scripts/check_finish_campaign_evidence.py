#!/usr/bin/env python3
"""Join exact-revision release receipts into ICC finish-campaign gate events."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE_NAMES = (
    "public-contract-parity",
    "native-correctness-and-sanitizers",
    "bindings-and-packaging",
    "verification-integrity",
    "representative-hardware",
    "icc-production-readiness",
)


def current_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tracked_dirty() -> bool | None:
    try:
        for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if subprocess.run(
                ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                return True
        return False
    except Exception:
        return None


def load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def exact_clean(errors: list[str], evidence: dict[str, Any], label: str, head: str) -> None:
    evidence_head = get(evidence, "meta.git_head") or evidence.get("git_head")
    evidence_dirty = get(evidence, "meta.git_dirty")
    if evidence_dirty is None:
        evidence_dirty = evidence.get("git_dirty")
    require(errors, evidence_head == head, f"{label} is not bound to the requested git head")
    require(errors, evidence_dirty is False, f"{label} is not from a clean tracked tree")


def grade_release(release: dict[str, Any], head: str) -> dict[str, list[str]]:
    common: list[str] = []
    exact_clean(common, release, "release smoke evidence", head)
    require(common, release.get("schema") == "tensorcore.release_smoke.runtime_evidence.v1", "release smoke schema mismatch")
    require(common, release.get("status") == "passed", "release smoke did not pass")
    require(common, get(release, "run.phase") == "complete", "release smoke did not complete")

    public = list(common)
    for path in (
        "checks.public_exports.passed",
        "checks.public_headers.passed",
        "checks.python_ffi_surface.passed",
        "checks.python_constants.passed",
        "checks.python_abi_layout.passed",
    ):
        require(public, get(release, path) is True, f"release smoke lacks passing {path}")

    native = list(common)
    require(native, get(release, "checks.tests.passed") is True, "native test suite did not pass")
    require(native, get(release, "checks.tests.gpu_device_available") is True, "representative Metal GPU was not exercised")

    packaging = list(common)
    for path in (
        "checks.artifact_privacy.passed",
        "checks.installed_wheel_smoke.passed",
        "checks.consumers.cmake.passed",
        "checks.consumers.pkg_config.passed",
        "checks.packaging_and_consumers.runtime_covered",
    ):
        require(packaging, get(release, path) is True, f"release smoke lacks passing {path}")
    return {"public": public, "native": native, "packaging": packaging}


def grade_sanitizers(evidence: dict[str, Any], head: str) -> list[str]:
    errors: list[str] = []
    exact_clean(errors, evidence, "sanitizer evidence", head)
    require(errors, evidence.get("schema") == "tensorcore.release_sanitizers.v1", "sanitizer schema mismatch")
    require(errors, evidence.get("status") == "passed", "sanitizer suite did not pass")
    require(errors, get(evidence, "checks.address.enabled") is True, "AddressSanitizer was not enabled")
    require(errors, get(evidence, "checks.undefined.enabled") is True, "UndefinedBehaviorSanitizer was not enabled")
    require(errors, get(evidence, "checks.ctest.passed") is True, "sanitized CTest suite did not pass")
    return errors


def grade_verification(evidence: dict[str, Any], head: str) -> dict[str, list[str]]:
    errors: list[str] = []
    exact_clean(errors, evidence, "verification evidence", head)
    require(errors, evidence.get("schema") == "tensorcore.release_verification.v1", "verification schema mismatch")
    require(errors, evidence.get("status") == "passed", "verification suite did not pass")
    for name in (
        "documentation",
        "version_consistency",
        "source_privacy",
        "history_privacy",
        "campaign",
        "script_selftests",
        "adversarial_evals",
    ):
        require(errors, get(evidence, f"checks.{name}.passed") is True, f"verification check {name} did not pass")

    icc = list(errors)
    require(icc, get(evidence, "checks.icc_precommit.passed") is True, "ICC pre-commit gate did not pass")
    require(icc, get(evidence, "checks.architecture_model.passed") is True, "ICC architecture model did not pass")
    return {"verification": errors, "icc": icc}


def grade_cuda(evidence: dict[str, Any], label: str, head: str, capability: str) -> list[str]:
    errors: list[str] = []
    exact_clean(errors, evidence, label, head)
    require(errors, evidence.get("runtime_status") == "passed", f"{label} runtime did not pass")
    require(errors, evidence.get("backend") == "cuda", f"{label} did not use CUDA")
    require(errors, evidence.get("cuda_build_enabled") is True, f"{label} CUDA build was not enabled")
    require(errors, get(evidence, "device.compute_capability") == capability, f"{label} compute capability mismatch")
    for name in ("cuda_gemm_sgemm", "cuda_gemm_hgemm", "cuda_gemm_i8"):
        require(errors, get(evidence, f"gemm_kernels.{name}.status") == "passed", f"{label} {name} did not pass")
    training = evidence.get("training_kernels")
    require(errors, isinstance(training, dict) and bool(training), f"{label} lacks training-kernel evidence")
    if isinstance(training, dict):
        require(errors, all(isinstance(v, dict) and v.get("backend") == "cuda" for v in training.values()), f"{label} contains a non-CUDA training result")
    return errors


def grade_hardware(
    release_errors: list[str], windows: dict[str, Any], xavier: dict[str, Any],
    blackwell: dict[str, Any], head: str,
) -> list[str]:
    errors = list(release_errors)
    exact_clean(errors, windows, "Windows CPU evidence", head)
    require(errors, windows.get("schema") == "tensorcore.windows_host_smoke.evidence.v2", "Windows evidence schema mismatch")
    require(errors, windows.get("runtime_status") == "passed", "Windows CPU smoke did not pass")
    require(errors, get(windows, "host.platform") == "windows", "Windows evidence platform mismatch")
    require(errors, windows.get("checkout") == "configured", "Windows evidence exposes or omits checkout provenance")
    require(errors, windows.get("source") == "configured", "Windows evidence exposes or omits source provenance")
    errors.extend(grade_cuda(xavier, "Jetson CUDA evidence", head, "7.2"))
    errors.extend(grade_cuda(blackwell, "Blackwell CUDA evidence", head, "12.0"))
    require(errors, get(blackwell, "gemm_kernels.cuda_gemm_bf16.status") == "passed", "Blackwell BF16 GEMM did not pass")
    return errors


def event(name: str, errors: list[str], head: str, timestamp: float) -> dict[str, Any]:
    return {
        "kind": "tensorcore_finish_gate",
        "name": name,
        "value": "PASS" if not errors else "FAIL",
        "status": "PASS" if not errors else "FAIL",
        "git_head": head,
        "timestamp": timestamp,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--sanitizer-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--verification-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--windows-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--xavier-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--blackwell-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--git-head", default=current_head())
    parser.add_argument("--require-clean-head", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.git_head:
        print("finish campaign evidence invalid: git head is unavailable", file=sys.stderr)
        return 2
    if args.require_clean_head and tracked_dirty() is not False:
        print("finish campaign evidence invalid: tracked worktree is not clean", file=sys.stderr)
        return 1

    try:
        release = load_json(args.release_evidence, "release smoke evidence")
        sanitizer = load_json(args.sanitizer_evidence, "sanitizer evidence")
        verification = load_json(args.verification_evidence, "verification evidence")
        windows = load_json(args.windows_evidence, "Windows evidence")
        xavier = load_json(args.xavier_evidence, "Jetson evidence")
        blackwell = load_json(args.blackwell_evidence, "Blackwell evidence")
    except ValueError as exc:
        print(f"finish campaign evidence invalid: {exc}", file=sys.stderr)
        return 1

    release_grades = grade_release(release, args.git_head)
    verification_grades = grade_verification(verification, args.git_head)
    grades = {
        "public-contract-parity": release_grades["public"],
        "native-correctness-and-sanitizers": release_grades["native"] + grade_sanitizers(sanitizer, args.git_head),
        "bindings-and-packaging": release_grades["packaging"],
        "verification-integrity": verification_grades["verification"],
        "representative-hardware": grade_hardware(release_grades["native"], windows, xavier, blackwell, args.git_head),
        "icc-production-readiness": verification_grades["icc"],
    }

    now = time.time()
    events = [event(name, grades[name], args.git_head, now) for name in GATE_NAMES]
    architecture_ok = not verification_grades["icc"]
    architecture_event = {
        "event_kind": "architecture_model",
        "kind": "architecture_model",
        "name": "tensorcore",
        "repo": "tensorcore",
        "schema": "icc.architecture_model_verify.v1",
        "status": "PASS" if architecture_ok else "FAIL",
        "timestamp": now,
        "target": "tensorcore",
        "git_head": args.git_head,
        "value": {
            "overall_ok": architecture_ok,
            "failed_invariants": [] if architecture_ok else [
                {"id": "release-verification", "kind": "evidence", "severity": "critical"}
            ],
        },
    }
    summary = {
        "kind": "tensorcore_finish_campaign",
        "name": "tensorcore-production-release-closure-20260824",
        "value": "PASS" if all(not grades[name] for name in GATE_NAMES) else "FAIL",
        "status": "PASS" if all(not grades[name] for name in GATE_NAMES) else "FAIL",
        "git_head": args.git_head,
        "timestamp": now,
        "generated_at": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "gate_count": len(events),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(f".{args.output.name}.tmp")
    temp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*events, architecture_event, summary]),
        encoding="utf-8",
    )
    temp.replace(args.output)

    failures = [row for row in events if row["value"] != "PASS"]
    if failures:
        for row in failures:
            print(f"finish gate FAIL: {row['name']}", file=sys.stderr)
            for message in row["errors"]:
                print(f"  - {message}", file=sys.stderr)
        return 1
    print(f"finish campaign evidence OK: head={args.git_head[:12]} gates={len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
