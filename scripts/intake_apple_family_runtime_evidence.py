#!/usr/bin/env python3
"""Verify a returned Apple-family handoff record and runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import pathlib
import re
import sys
from typing import Any


SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_apple_family_runtime_evidence as evidence_checker  # noqa: E402


SCHEMA = "tensorcore.apple_family_evidence_intake.v1"
PREFIX = "APPLE_FAMILY_EVIDENCE_HANDOFF "
HEAD_RE = re.compile(r"[0-9a-f]{40}")
RECORD_RE = re.compile(
    r"^APPLE_FAMILY_EVIDENCE_HANDOFF "
    r"chip=(M4|M5) "
    r"head=([0-9a-f]{40}) "
    r"sha256=([0-9a-f]{64}) "
    r"path=(.+)$"
)
MAX_EVIDENCE_BYTES = 10 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024


class IntakeError(ValueError):
    """A contributor artifact failed transport or evidence validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--handoff-record", help="Returned APPLE_FAMILY_EVIDENCE_HANDOFF line")
    group.add_argument(
        "--handoff-record-file",
        type=pathlib.Path,
        help="Text file or log containing exactly one returned handoff line",
    )
    parser.add_argument("--expected-head", required=True, help="Requested full 40-character commit SHA")
    parser.add_argument("--require-chip", choices=("M4", "M5"), required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntakeError(f"duplicate object key in evidence JSON: {key!r}")
        result[key] = value
    return result


def bounded_bytes(path: pathlib.Path, limit: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise IntakeError(f"could not inspect {label}: {exc}") from exc
    if size > limit:
        raise IntakeError(f"{label} exceeds the {limit}-byte intake limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IntakeError(f"could not read {label}: {exc}") from exc


def extract_record(text: str) -> dict[str, str]:
    lines = [line.rstrip("\r") for line in text.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise IntakeError(f"handoff input must contain exactly one {PREFIX.strip()} line")
    match = RECORD_RE.fullmatch(lines[0])
    if not match:
        raise IntakeError("handoff record has invalid fields or ordering")
    return {
        "chip": match.group(1),
        "head": match.group(2),
        "sha256": match.group(3),
        "path": match.group(4),
    }


def decode_evidence(raw: bytes) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntakeError(f"evidence is not UTF-8 JSON: {exc}") from exc
    try:
        data = json.loads(decoded, object_pairs_hook=unique_object)
    except IntakeError:
        raise
    except json.JSONDecodeError as exc:
        raise IntakeError(f"evidence is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntakeError("evidence JSON root must be an object")
    return data


def validate_intake(
    *,
    raw: bytes,
    record: dict[str, str],
    expected_head: str,
    required_chip: str,
) -> dict[str, Any]:
    if not HEAD_RE.fullmatch(expected_head):
        raise IntakeError("--expected-head must be a full lowercase 40-character git SHA")
    if record["chip"] != required_chip:
        raise IntakeError(
            f"record chip {record['chip']} does not match required {required_chip}"
        )
    if record["head"] != expected_head:
        raise IntakeError(
            f"record head {record['head']} does not match expected {expected_head}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(record["sha256"], digest):
        raise IntakeError(
            f"evidence SHA-256 digest mismatch: record={record['sha256']} actual={digest}"
        )

    data = decode_evidence(raw)
    checker_args = argparse.Namespace(
        require_pass=True,
        require_clean_head=True,
        git_head=expected_head,
        require_chip=required_chip,
    )
    errors = evidence_checker.validate(data, checker_args)
    if errors:
        raise IntakeError("evidence contract invalid:\n  - " + "\n  - ".join(errors))

    reservation = data.get("reservation", {})
    return {
        "schema": SCHEMA,
        "status": "accepted",
        "chip": required_chip,
        "git_head": expected_head,
        "evidence_sha256": digest,
        "transport_path": record["path"],
        "hardware_resource": reservation.get("resource"),
        "authority_owner": reservation.get("authority_owner"),
    }


def print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "APPLE_FAMILY_EVIDENCE_ACCEPTED "
        f"chip={report['chip']} head={report['git_head']} "
        f"sha256={report['evidence_sha256']}"
    )


def main() -> int:
    args = parse_args()
    try:
        raw = bounded_bytes(args.evidence, MAX_EVIDENCE_BYTES, "evidence artifact")
        if args.handoff_record_file:
            record_raw = bounded_bytes(
                args.handoff_record_file, MAX_RECORD_BYTES, "handoff record file"
            )
            try:
                record_text = record_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise IntakeError(f"handoff record file is not UTF-8 text: {exc}") from exc
        else:
            record_text = args.handoff_record
        record = extract_record(record_text)
        report = validate_intake(
            raw=raw,
            record=record,
            expected_head=args.expected_head,
            required_chip=args.require_chip,
        )
        report["evidence_path"] = str(args.evidence.resolve())
    except IntakeError as exc:
        print(f"Apple family evidence intake rejected: {exc}", file=sys.stderr)
        return 1
    print_report(report, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
