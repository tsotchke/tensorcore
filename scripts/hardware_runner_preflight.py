#!/usr/bin/env python3
"""Write hardware-evidence self-hosted runner preflight diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any


SCHEMA = "tensorcore.hardware_runner_preflight.v1"
FORMAT_VERSION = 1
DEFAULT_REQUIRED_LABELS = ["self-hosted", "macOS", "ARM64"]
METAL4_TENSOROPS_REQUIRED_LABELS = ["m5", "sdk26", "metal4-tensorops"]
TOKEN_UNAVAILABLE = "token_unavailable"
RUNNER_ABSENT = "runner_absent"
RUNNER_OFFLINE = "runner_offline"
RUNNER_ONLINE = "runner_online"
MAX_LABEL_CANDIDATES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-json", type=pathlib.Path, required=True)
    parser.add_argument("--api-error", type=pathlib.Path, required=True)
    parser.add_argument("--api-rc", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--require-metal4-tensorops", default="false")
    parser.add_argument("--head-sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", ""))
    parser.add_argument("--workflow", default=os.environ.get("GITHUB_WORKFLOW", ""))
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


def label_candidates_for(
    runners: list[dict[str, Any]],
    required_labels: list[str],
) -> list[dict[str, Any]]:
    required = set(required_labels)
    candidates: list[dict[str, Any]] = []
    for runner in runners:
        labels = labels_for(runner)
        matched = sorted(required & labels)
        if not matched:
            continue
        candidates.append(
            {
                "name": runner.get("name"),
                "status": runner.get("status"),
                "busy": runner.get("busy"),
                "matched_required_labels": matched,
                "missing_required_labels": sorted(required - labels),
                "labels": sorted(labels),
            }
        )
    candidates.sort(
        key=lambda item: (
            -len(item["matched_required_labels"]),
            str(item.get("name") or ""),
        )
    )
    return candidates[:MAX_LABEL_CANDIDATES]


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_required_labels(require_metal4_tensorops: str) -> list[str]:
    labels = list(DEFAULT_REQUIRED_LABELS)
    if truthy(require_metal4_tensorops):
        labels.extend(METAL4_TENSOROPS_REQUIRED_LABELS)
    return labels


def build_evidence(
    *,
    api_json: pathlib.Path,
    api_error: pathlib.Path,
    api_rc: int,
    repository: str,
    require_metal4_tensorops: str,
    required_labels: list[str],
    head_sha: str = "",
    run_id: str = "",
    run_attempt: str = "",
    workflow: str = "",
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

    label_candidates = label_candidates_for(runners, required_labels)
    diagnostics = diagnostics_for_status(
        status=status,
        api_note=api_note,
        required_labels=required_labels,
        matching=matching,
        online_matching=online_matching,
        label_candidates=label_candidates,
    )

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_hardware_runner_preflight",
            "head_sha": head_sha,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "workflow": workflow,
        },
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
        "label_candidate_runners": label_candidates,
        "diagnostics": diagnostics,
    }


def diagnostics_for_status(
    *,
    status: str,
    api_note: str,
    required_labels: list[str],
    matching: list[dict[str, Any]],
    online_matching: list[dict[str, Any]],
    label_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    labels = ", ".join(required_labels)
    if status == "runner_api_unavailable":
        return [
            {
                "id": "hardware_runner_preflight.runner_api",
                "diagnostic_class": TOKEN_UNAVAILABLE,
                "status": "failed",
                "message": api_note or "GitHub runner API was unavailable",
                "recommended_action": (
                    "Add or fix the TC_RUNNER_READ_TOKEN repository secret with permission "
                    "to list Actions runners, then redispatch hardware-evidence.yml."
                ),
            }
        ]
    if status == "blocked_no_matching_runner":
        candidate_hint = ""
        if label_candidates:
            parts = []
            for item in label_candidates[:3]:
                name = str(item.get("name") or "<unnamed>")
                missing = ", ".join(str(label) for label in item.get("missing_required_labels") or [])
                parts.append(f"{name} missing [{missing}]")
            candidate_hint = " Closest visible runner(s): " + "; ".join(parts) + "."
        return [
            {
                "id": "hardware_runner_preflight.matching_runner",
                "diagnostic_class": RUNNER_ABSENT,
                "status": "failed",
                "message": f"No runner matched required labels: {labels}",
                "recommended_action": (
                    "Register an M5/SDK26 macOS ARM64 self-hosted runner with the required "
                    "labels, or fix the labels on the closest visible runner, then redispatch "
                    f"hardware-evidence.yml.{candidate_hint}"
                ),
            }
        ]
    if status == "matching_runner_offline":
        names = ", ".join(str(item.get("name") or "<unnamed>") for item in matching)
        return [
            {
                "id": "hardware_runner_preflight.runner_online",
                "diagnostic_class": RUNNER_OFFLINE,
                "status": "failed",
                "message": f"Matching runner(s) are offline: {names}",
                "recommended_action": (
                    "Start the matching self-hosted runner service on the M5/SDK26 host, "
                    "then redispatch hardware-evidence.yml."
                ),
            }
        ]
    if status == "matching_runner_online":
        names = ", ".join(str(item.get("name") or "<unnamed>") for item in online_matching)
        next_job = (
            "m5-tensorops-release-smoke"
            if "metal4-tensorops" in required_labels
            else "apple-gpu-release-smoke"
        )
        return [
            {
                "id": "hardware_runner_preflight.runner_online",
                "diagnostic_class": RUNNER_ONLINE,
                "status": "passed",
                "message": f"Online matching runner(s): {names}",
                "recommended_action": (
                    f"Wait for {next_job} to run and upload hardware evidence."
                ),
            }
        ]
    return []


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
    candidates = evidence.get("label_candidate_runners")
    if isinstance(candidates, list) and candidates:
        lines.append("- Closest visible runners:")
        for item in candidates[:MAX_LABEL_CANDIDATES]:
            if not isinstance(item, dict):
                continue
            missing = ", ".join(str(label) for label in item.get("missing_required_labels") or [])
            matched = ", ".join(str(label) for label in item.get("matched_required_labels") or [])
            lines.append(
                f"  - `{item.get('name') or '<unnamed>'}` status=`{item.get('status')}` "
                f"matched=`{matched}` missing=`{missing}`"
            )
    if note:
        lines.append(f"- Runner API note: `{note}`")
    diagnostics = evidence.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            action = str(item.get("recommended_action") or "")
            if action:
                lines.append(f"- Recommended action: {action}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    required_labels = args.required_label or default_required_labels(args.require_metal4_tensorops)
    evidence = build_evidence(
        api_json=args.api_json,
        api_error=args.api_error,
        api_rc=args.api_rc,
        repository=args.repository,
        require_metal4_tensorops=args.require_metal4_tensorops,
        required_labels=required_labels,
        head_sha=args.head_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        workflow=args.workflow,
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
