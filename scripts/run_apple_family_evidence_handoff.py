#!/usr/bin/env python3
"""Produce verifier-ready M4/M5 evidence on borrowed Apple hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import re
import shlex
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_apple_family_runtime_evidence as collector  # noqa: E402


SCHEMA = "tensorcore.apple_family_evidence_handoff.v1"
HEAD_RE = re.compile(r"[0-9a-f]{40}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", choices=("M4", "M5"), required=True)
    parser.add_argument("--expected-head", required=True, help="Exact 40-character TensorCore commit SHA")
    parser.add_argument("--hardware-resource", required=True, help="Non-secret provider or physical asset id")
    parser.add_argument("--authority-owner", required=True, help="Non-secret accountable owner id")
    parser.add_argument("--build-dir", type=pathlib.Path)
    parser.add_argument("--evidence-path", type=pathlib.Path)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact command plan without probing hardware")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def authority_error(chip: str, resource: str, owner: str) -> str | None:
    if not resource.strip():
        return "hardware resource id must not be empty"
    if not owner.strip():
        return "authority owner id must not be empty"
    if chip == "M4" and not collector.resource_authorized(chip, resource.strip(), owner.strip()):
        if collector.resource_scope(resource.strip()) == "reserved":
            return "reserved enki M4 handoff requires a tsotchke-chan authority owner"
        return "M4 hardware authority is not valid"
    return None


def command_plan(
    *,
    chip: str,
    head: str,
    resource: str,
    owner: str,
    build_dir: pathlib.Path,
    evidence_path: pathlib.Path,
    timeout_sec: float,
) -> list[list[str]]:
    return [
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTC_BUILD_TESTS=ON",
            "-DTC_BUILD_BENCH=OFF",
            "-DTC_BUILD_EXAMPLES=OFF",
            "-DTC_ENABLE_TENSOROPS=ON",
        ],
        [
            sys.executable,
            str(SCRIPTS / "run_apple_family_runtime_evidence.py"),
            "--build-dir",
            str(build_dir),
            "--evidence-path",
            str(evidence_path),
            "--expected-chip",
            chip,
            "--hardware-resource",
            resource,
            "--authority-owner",
            owner,
            "--timeout-sec",
            str(timeout_sec),
            "--require-pass",
        ],
        [
            sys.executable,
            str(SCRIPTS / "check_apple_family_runtime_evidence.py"),
            str(evidence_path),
            "--git-head",
            head,
            "--require-chip",
            chip,
            "--require-clean-head",
            "--require-pass",
        ],
    ]


def print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if report["status"] == "planned":
        print("Apple family evidence handoff plan:")
        for command in report["commands"]:
            print(f"  {command}")
        return
    print(
        "APPLE_FAMILY_EVIDENCE_HANDOFF "
        f"chip={report['chip']} head={report['git_head']} "
        f"sha256={report['evidence_sha256']} path={report['evidence_path']}"
    )


def fail(message: str) -> int:
    print(f"Apple family evidence handoff blocked: {message}", file=sys.stderr)
    return 1


def run_command(argv: list[str], timeout_sec: float) -> int:
    print(f"[tensorcore-handoff] {shlex.join(argv)}", flush=True)
    try:
        return subprocess.run(argv, cwd=ROOT, timeout=timeout_sec, check=False).returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[tensorcore-handoff] command failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    resource = args.hardware_resource.strip()
    owner = args.authority_owner.strip()
    if not HEAD_RE.fullmatch(args.expected_head):
        return fail("--expected-head must be a full lowercase 40-character git SHA")
    problem = authority_error(args.chip, resource, owner)
    if problem:
        return fail(problem)

    build_dir = (args.build_dir or ROOT / f"build-apple-family-handoff-{args.chip.lower()}").resolve()
    evidence_path = (
        args.evidence_path or build_dir / "apple_family_runtime_evidence.json"
    ).resolve()
    commands = command_plan(
        chip=args.chip,
        head=args.expected_head,
        resource=resource,
        owner=owner,
        build_dir=build_dir,
        evidence_path=evidence_path,
        timeout_sec=args.timeout_sec,
    )
    if args.dry_run:
        print_report(
            {
                "schema": SCHEMA,
                "status": "planned",
                "chip": args.chip,
                "git_head": args.expected_head,
                "hardware_resource": resource,
                "authority_owner": owner,
                "evidence_path": str(evidence_path),
                "commands": [shlex.join(command) for command in commands],
            },
            args.json,
        )
        return 0

    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return fail("physical Darwin/arm64 Apple silicon is required")
    head = collector.git_value("rev-parse", "HEAD")
    if head != args.expected_head:
        return fail(f"checkout head {head!r} does not match expected {args.expected_head!r}")
    if collector.tracked_dirty() is not False:
        return fail("tracked git tree must be clean before configuration")

    host_chip, _safe_identity = collector.host_chip(min(args.timeout_sec, 60.0))
    if host_chip != args.chip:
        return fail(f"physical host chip {host_chip!r} does not match required {args.chip}")
    if args.chip == "M5":
        sdk_attempt = collector.command(["xcrun", "--show-sdk-version"], timeout=min(args.timeout_sec, 60.0))
        sdk = sdk_attempt.get("stdout_tail", "").strip() if sdk_attempt.get("rc") == 0 else ""
        if collector.version_tuple(sdk) < (26, 0):
            return fail(f"M5 handoff requires SDK 26.0 or newer, got {sdk or 'unavailable'}")

    for command in commands:
        if run_command(command, args.timeout_sec) != 0:
            return 1
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    print_report(
        {
            "schema": SCHEMA,
            "status": "passed",
            "chip": args.chip,
            "git_head": head,
            "hardware_resource": resource,
            "authority_owner": owner,
            "evidence_path": str(evidence_path),
            "evidence_sha256": digest,
        },
        args.json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
