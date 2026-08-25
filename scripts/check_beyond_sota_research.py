#!/usr/bin/env python3
"""Validate the checked-in Beyond-SOTA research ledger and freshness contract."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import urllib.parse
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "configs" / "beyond_sota_research.json"
ALLOWED_STATUSES = {"open", "blocked", "verified", "retired"}
PRIMARY_HOSTS = {
    "arxiv.org",
    "developer.apple.com",
    "developer.nvidia.com",
    "docs.nvidia.com",
    "docs.vllm.ai",
    "github.com",
    "kubernetes.io",
    "rocm.docs.amd.com",
    "spiffe.io",
    "tailscale.com",
}
PRIMARY_GITHUB_OWNERS = {"deepseek-ai", "mlcommons"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=pathlib.Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="Date used for deterministic freshness checks (YYYY-MM-DD).",
    )
    return parser.parse_args()


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def parse_date(errors: list[str], value: Any, label: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        errors.append(f"{label} must be an ISO date")
        return None


def is_primary_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in PRIMARY_HOSTS:
        return False
    if host != "github.com":
        return True
    parts = [part for part in parsed.path.split("/") if part]
    return bool(parts and parts[0].lower() in PRIMARY_GITHUB_OWNERS)


def validate_ledger(path: pathlib.Path, as_of: dt.date) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load ledger: {exc}"]

    require(errors, payload.get("schema_version") == 1, "schema_version must be 1")
    refresh_days = payload.get("refresh_interval_days")
    require(
        errors,
        isinstance(refresh_days, int) and 1 <= refresh_days <= 90,
        "refresh_interval_days must be between 1 and 90",
    )
    if not isinstance(refresh_days, int):
        refresh_days = 0

    assessed_at = parse_date(errors, payload.get("assessed_at"), "assessed_at")
    next_due = parse_date(errors, payload.get("next_refresh_due"), "next_refresh_due")
    if assessed_at is not None:
        require(errors, assessed_at <= as_of, "assessed_at cannot be in the future")
        require(
            errors,
            (as_of - assessed_at).days <= refresh_days,
            f"research ledger is stale by {(as_of - assessed_at).days - refresh_days} days",
        )
    if assessed_at is not None and next_due is not None:
        require(errors, next_due > assessed_at, "next_refresh_due must follow assessed_at")
        require(
            errors,
            (next_due - assessed_at).days <= refresh_days,
            "next_refresh_due exceeds refresh_interval_days",
        )

    policy = payload.get("claims_policy") or {}
    for key in (
        "primary_sources_only",
        "same_model_hardware_quality_required",
        "end_to_end_required_for_system_claims",
        "microbenchmarks_must_be_labeled",
        "projected_results_are_not_evidence",
    ):
        require(errors, policy.get(key) is True, f"claims_policy.{key} must be true")
    require(
        errors,
        isinstance(policy.get("minimum_repetitions"), int)
        and policy["minimum_repetitions"] >= 5,
        "claims_policy.minimum_repetitions must be at least 5",
    )
    require(
        errors,
        set(policy.get("required_percentiles") or []) >= {"p50", "p95", "p99"},
        "claims_policy.required_percentiles must include p50, p95, and p99",
    )

    required_axes = set(payload.get("required_axes") or [])
    require(errors, len(required_axes) >= 8, "required_axes must cover at least eight axes")

    source_ids: set[str] = set()
    source_axes: set[str] = set()
    sources = payload.get("sources") or []
    require(errors, len(sources) >= 10, "ledger must contain at least ten primary sources")
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        source_id = str(source.get("id") or "")
        require(errors, bool(source_id), f"{label}.id is required")
        require(errors, source_id not in source_ids, f"duplicate source id {source_id!r}")
        source_ids.add(source_id)
        axes = set(source.get("axes") or [])
        require(errors, bool(axes), f"{label}.axes must not be empty")
        source_axes.update(axes)
        require(errors, source.get("primary") is True, f"{label} must be primary")
        url = str(source.get("url") or "")
        require(errors, is_primary_url(url), f"{label}.url is not an approved primary source")
        for key in ("title", "claim", "tensorcore_implication"):
            require(errors, bool(str(source.get(key) or "").strip()), f"{label}.{key} is required")
        verified = parse_date(errors, source.get("last_verified"), f"{label}.last_verified")
        if verified is not None:
            require(errors, verified <= as_of, f"{label}.last_verified cannot be in the future")
            require(
                errors,
                (as_of - verified).days <= refresh_days,
                f"{label} is stale",
            )

    require(
        errors,
        required_axes <= source_axes,
        f"required axes missing sources: {sorted(required_axes - source_axes)}",
    )

    gate_ids: set[str] = set()
    gate_axes: set[str] = set()
    gates = payload.get("gates") or []
    require(errors, len(gates) >= 8, "ledger must contain at least eight falsifiable gates")
    for index, gate in enumerate(gates):
        label = f"gates[{index}]"
        gate_id = str(gate.get("id") or "")
        require(errors, bool(gate_id), f"{label}.id is required")
        require(errors, gate_id not in gate_ids, f"duplicate gate id {gate_id!r}")
        gate_ids.add(gate_id)
        axes = set(gate.get("axes") or [])
        require(errors, bool(axes), f"{label}.axes must not be empty")
        gate_axes.update(axes)
        refs = set(gate.get("source_ids") or [])
        require(errors, bool(refs), f"{label}.source_ids must not be empty")
        require(errors, refs <= source_ids, f"{label} references unknown sources: {sorted(refs - source_ids)}")
        for key in ("metric", "comparator", "pass_condition", "evidence", "task_id"):
            require(errors, bool(str(gate.get(key) or "").strip()), f"{label}.{key} is required")
        status = gate.get("status")
        require(errors, status in ALLOWED_STATUSES, f"{label}.status must be one of {sorted(ALLOWED_STATUSES)}")
        if status == "blocked":
            require(errors, bool(str(gate.get("blocker") or "").strip()), f"{label}.blocker is required")

    require(
        errors,
        required_axes <= gate_axes,
        f"required axes missing gates: {sorted(required_axes - gate_axes)}",
    )
    return errors


def main() -> int:
    args = parse_args()
    errors = validate_ledger(args.ledger, args.as_of)
    if errors:
        for error in errors:
            print(f"beyond-SOTA research error: {error}")
        return 1
    print(f"beyond-SOTA research ledger OK: {args.ledger} (as of {args.as_of})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
