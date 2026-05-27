#!/usr/bin/env python3
"""Validate hardware-evidence self-hosted runner preflight artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


SCHEMA = "tensorcore.hardware_runner_preflight.v1"
FORMAT_VERSION = 1
VALID_STATUSES = {
    "runner_api_unavailable",
    "matching_runner_online",
    "matching_runner_offline",
    "blocked_no_matching_runner",
}
VALID_DIAGNOSTIC_CLASSES = {
    "token_unavailable",
    "runner_absent",
    "runner_offline",
    "runner_online",
}
METAL4_TENSOROPS_REQUIRED_LABELS = {"m5", "sdk26", "metal4-tensorops"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--require-runner-api", action="store_true")
    parser.add_argument("--require-online-runner", action="store_true")
    parser.add_argument("--require-metal4-tensorops", action="store_true")
    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read hardware runner preflight {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"hardware runner preflight is not valid JSON: {exc}") from exc


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def fail(errors: list[str]) -> int:
    print("hardware runner preflight invalid:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def require_int(errors: list[str], data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if not isinstance(value, int):
        errors.append(f"{key} must be an integer, got {value!r}")
        return None
    if value < 0:
        errors.append(f"{key} must be non-negative, got {value!r}")
        return None
    return value


def validate_counts(errors: list[str], data: dict[str, Any]) -> None:
    matching = data.get("matching_runners")
    if not isinstance(matching, list):
        errors.append("matching_runners must be a list")
        return
    matching_count = require_int(errors, data, "matching_runner_count")
    online_count = require_int(errors, data, "online_matching_runner_count")
    registered_count = require_int(errors, data, "registered_runner_count")
    if matching_count is None or online_count is None or registered_count is None:
        return
    if matching_count != len(matching):
        errors.append(
            "matching_runner_count must match matching_runners length: "
            f"{matching_count!r} != {len(matching)!r}"
        )
    computed_online = sum(
        1 for item in matching if isinstance(item, dict) and item.get("status") == "online"
    )
    if online_count != computed_online:
        errors.append(
            "online_matching_runner_count must match online matching runners: "
            f"{online_count!r} != {computed_online!r}"
        )
    if matching_count > registered_count:
        errors.append("matching_runner_count cannot exceed registered_runner_count")


def validate_matching_runner_labels(errors: list[str], data: dict[str, Any]) -> None:
    required = data.get("required_labels")
    matching = data.get("matching_runners")
    if not isinstance(required, list) or not isinstance(matching, list):
        return
    required_set = {str(item) for item in required}
    for index, item in enumerate(matching):
        if not isinstance(item, dict):
            continue
        labels = item.get("labels")
        if not isinstance(labels, list):
            errors.append(f"matching_runners[{index}].labels must be a list")
            continue
        missing = sorted(required_set - {str(label) for label in labels})
        if missing:
            errors.append(f"matching_runners[{index}] missing required labels: {missing!r}")


def validate_label_candidate_runners(errors: list[str], data: dict[str, Any]) -> None:
    candidates = data.get("label_candidate_runners")
    if candidates is None:
        return
    if not isinstance(candidates, list):
        errors.append("label_candidate_runners must be a list")
        return
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            errors.append(f"label_candidate_runners[{index}] must be an object")
            continue
        for key in ("matched_required_labels", "missing_required_labels", "labels"):
            value = item.get(key)
            if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
                errors.append(f"label_candidate_runners[{index}].{key} must be a list of strings")
        if item.get("status") is not None and not isinstance(item.get("status"), str):
            errors.append(f"label_candidate_runners[{index}].status must be a string when present")
        if item.get("busy") is not None and not isinstance(item.get("busy"), bool):
            errors.append(f"label_candidate_runners[{index}].busy must be a boolean when present")


def validate_status_consistency(errors: list[str], data: dict[str, Any]) -> None:
    status = data.get("status")
    api_rc = data.get("runner_api_rc")
    matching_count = data.get("matching_runner_count")
    online_count = data.get("online_matching_runner_count")
    if status == "runner_api_unavailable" and api_rc == 0:
        errors.append("runner_api_unavailable requires non-zero runner_api_rc")
    if status == "matching_runner_online" and online_count == 0:
        errors.append("matching_runner_online requires online_matching_runner_count > 0")
    if status == "matching_runner_offline" and (matching_count == 0 or online_count != 0):
        errors.append("matching_runner_offline requires matching runners and zero online runners")
    if status == "blocked_no_matching_runner" and matching_count != 0:
        errors.append("blocked_no_matching_runner requires matching_runner_count == 0")


def validate_diagnostics(errors: list[str], data: dict[str, Any]) -> None:
    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, list):
        errors.append("diagnostics must be a list")
        return
    if not diagnostics:
        errors.append("diagnostics must not be empty")
        return
    classes: set[str] = set()
    for index, item in enumerate(diagnostics):
        if not isinstance(item, dict):
            errors.append(f"diagnostics[{index}] must be an object")
            continue
        diagnostic_class = item.get("diagnostic_class")
        if diagnostic_class not in VALID_DIAGNOSTIC_CLASSES:
            errors.append(
                f"diagnostics[{index}].diagnostic_class must be one of "
                f"{sorted(VALID_DIAGNOSTIC_CLASSES)!r}, got {diagnostic_class!r}"
            )
        else:
            classes.add(str(diagnostic_class))
        if item.get("status") not in {"passed", "failed"}:
            errors.append(f"diagnostics[{index}].status must be passed or failed")
        for key in ("id", "message", "recommended_action"):
            if not isinstance(item.get(key), str) or not item.get(key):
                errors.append(f"diagnostics[{index}].{key} must be a non-empty string")

    expected_by_status = {
        "runner_api_unavailable": "token_unavailable",
        "blocked_no_matching_runner": "runner_absent",
        "matching_runner_offline": "runner_offline",
        "matching_runner_online": "runner_online",
    }
    expected = expected_by_status.get(str(data.get("status")))
    if expected and expected not in classes:
        errors.append(f"diagnostics must include diagnostic_class={expected!r}")


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    if not isinstance(data, dict):
        return fail(["hardware runner preflight root must be a JSON object"])

    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if get_path(data, "meta.format") != FORMAT_VERSION:
        errors.append(f"meta.format must be {FORMAT_VERSION}")
    if get_path(data, "meta.source") != "tensorcore_hardware_runner_preflight":
        errors.append("meta.source must be tensorcore_hardware_runner_preflight")
    if data.get("status") not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)!r}, got {data.get('status')!r}")
    if not isinstance(data.get("repository"), str) or not data.get("repository"):
        errors.append("repository must be a non-empty string")
    if not isinstance(data.get("required_labels"), list) or not all(
        isinstance(item, str) for item in data.get("required_labels", [])
    ):
        errors.append("required_labels must be a list of strings")
    if data.get("require_metal4_tensorops") not in {"true", "false", True, False}:
        errors.append("require_metal4_tensorops must be true/false")

    validate_counts(errors, data)
    validate_matching_runner_labels(errors, data)
    validate_label_candidate_runners(errors, data)
    validate_status_consistency(errors, data)
    validate_diagnostics(errors, data)

    if args.expected_head and get_path(data, "meta.head_sha") != args.expected_head:
        errors.append(
            "hardware runner preflight head mismatch: "
            f"{get_path(data, 'meta.head_sha')!r} != {args.expected_head!r}"
        )
    if args.require_runner_api and data.get("runner_api_rc") != 0:
        errors.append(f"runner API must be available, got rc={data.get('runner_api_rc')!r}")
    if args.require_online_runner and data.get("status") != "matching_runner_online":
        errors.append(f"online matching runner required, got status={data.get('status')!r}")
    if args.require_metal4_tensorops and data.get("require_metal4_tensorops") not in {"true", True}:
        errors.append("preflight must be from require_metal4_tensorops=true workflow input")
    if data.get("require_metal4_tensorops") in {"true", True} and isinstance(
        data.get("required_labels"), list
    ):
        labels = {str(item) for item in data["required_labels"]}
        missing = sorted(METAL4_TENSOROPS_REQUIRED_LABELS - labels)
        if missing:
            errors.append(f"Metal 4 TensorOps preflight missing required runner labels: {missing!r}")

    if errors:
        return fail(errors)

    print(
        "hardware runner preflight OK: "
        f"status={data.get('status')} online_matching={data.get('online_matching_runner_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
