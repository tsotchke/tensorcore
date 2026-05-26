#!/usr/bin/env python3
"""Write hardware-evidence self-hosted runner preflight diagnostics."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


SCHEMA = "tensorcore.hardware_runner_preflight.v1"
DEFAULT_REQUIRED_LABELS = ["self-hosted", "macOS", "ARM64"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-json", type=pathlib.Path, required=True)
    parser.add_argument("--api-error", type=pathlib.Path, required=True)
    parser.add_argument("--api-rc", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--require-metal4-tensorops", default="false")
    parser.add_argument(
        "--required-label",
        action="append",
        default=[],
        help="Required self-hosted runner label; may be repeated.",
    )
    return parser.parse_args()


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_runner_payload(api_json: pathlib.Path) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(read_text(api_json))
    except json.JSONDecodeError as exc:
        return [], f"could not parse runner API response: {exc}"
    runners = payload.get("runners")
    if not isinstance(runners, list):
        return [], "runner API response did not contain a runners list"
    return [item for item in runners if isinstance(item, dict)], ""


def labels_for(runner: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for item in runner.get("labels") or []:
        if isinstance(item, dict) and item.get("name"):
            labels.add(str(item["name"]))
    return labels


def build_evidence(
    *,
    api_json: pathlib.Path,
    api_error: pathlib.Path,
    api_rc: int,
    repository: str,
    require_metal4_tensorops: str,
    required_labels: list[str],
) -> dict[str, Any]:
    api_note = read_text(api_error).strip()
    runners: list[dict[str, Any]] = []
    if api_rc == 0:
        runners, parse_error = load_runner_payload(api_json)
        if parse_error:
            api_rc = 1
            api_note = parse_error

    required = set(required_labels)
    matching: list[dict[str, Any]] = []
    online_matching: list[dict[str, Any]] = []
    for runner in runners:
        labels = labels_for(runner)
        if not required.issubset(labels):
            continue
        item = {
            "name": runner.get("name"),
            "status": runner.get("status"),
            "busy": runner.get("busy"),
            "labels": sorted(labels),
        }
        matching.append(item)
        if runner.get("status") == "online":
            online_matching.append(item)

    if api_rc != 0:
        status = "runner_api_unavailable"
    elif online_matching:
        status = "matching_runner_online"
    elif matching:
        status = "matching_runner_offline"
    else:
        status = "blocked_no_matching_runner"

    return {
        "schema": SCHEMA,
        "status": status,
        "repository": repository,
        "required_labels": required_labels,
        "require_metal4_tensorops": require_metal4_tensorops,
        "runner_api_rc": api_rc,
        "runner_api_error": api_note,
        "registered_runner_count": len(runners),
        "matching_runner_count": len(matching),
        "online_matching_runner_count": len(online_matching),
        "matching_runners": matching,
    }


def append_summary(path: pathlib.Path, evidence: dict[str, Any]) -> None:
    required = ", ".join(evidence["required_labels"])
    note = str(evidence.get("runner_api_error") or "")[:240]
    lines = [
        "### Hardware Evidence runner preflight",
        "",
        f"- Status: `{evidence['status']}`",
        f"- Required labels: `{required}`",
        f"- require_metal4_tensorops: `{evidence['require_metal4_tensorops']}`",
        f"- Registered runners visible to this job: `{evidence['registered_runner_count']}`",
        f"- Matching runners: `{evidence['matching_runner_count']}`",
        f"- Online matching runners: `{evidence['online_matching_runner_count']}`",
    ]
    if note:
        lines.append(f"- Runner API note: `{note}`")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    required_labels = args.required_label or list(DEFAULT_REQUIRED_LABELS)
    evidence = build_evidence(
        api_json=args.api_json,
        api_error=args.api_error,
        api_rc=args.api_rc,
        repository=args.repository,
        require_metal4_tensorops=args.require_metal4_tensorops,
        required_labels=required_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary:
        append_summary(args.summary, evidence)
    print(
        "hardware runner preflight: "
        f"status={evidence['status']} online_matching={evidence['online_matching_runner_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
