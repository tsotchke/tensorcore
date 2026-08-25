#!/usr/bin/env python3
"""Validate a sustained production window from the scheduler reconciliation journal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


EVENT_SCHEMA = "tensorcore.scheduler_reconciliation_event.v1"
EVIDENCE_SCHEMA = "tensorcore.scheduler_reconciler.evidence.v1"


def fail(message: str) -> int:
    print(f"scheduler reconciler evidence invalid: {message}", file=sys.stderr)
    return 1


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def read_journal(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"journal line {line_number} is not an object")
        rows.append(value)
    return rows


def eligible_resources(inventory: dict) -> set[str]:
    resources = inventory.get("resources")
    if not isinstance(resources, list):
        raise ValueError("inventory.resources must be a list")
    return {
        str(row["id"])
        for row in resources
        if isinstance(row, dict)
        and row.get("id")
        and row.get("status", "active") == "active"
        and row.get("general_queue_eligible", True) is not False
        and row.get("control_plane") == "tensorcore_scheduler"
    }


def scheduled_resources(jobs: dict) -> set[str]:
    rows = jobs.get("jobs")
    if not isinstance(rows, list):
        raise ValueError("jobs.jobs must be a list")
    return {
        str(row["resource"])
        for row in rows
        if isinstance(row, dict) and row.get("resource") and row.get("enabled", True) is not False
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-head", default=current_git_head())
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--max-age-sec", type=float, default=120.0)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--require-clean-head", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_consecutive < 1:
        return fail("--min-consecutive must be >= 1")
    if args.max_age_sec <= 0:
        return fail("--max-age-sec must be > 0")
    try:
        rows = read_journal(args.journal.expanduser().resolve())
        inventory = read_object(args.inventory.expanduser().resolve())
        jobs = read_object(args.jobs.expanduser().resolve())
    except Exception as exc:
        return fail(f"could not load evidence inputs: {exc}")
    if len(rows) < args.min_consecutive:
        return fail(
            f"journal has {len(rows)} events; need {args.min_consecutive} consecutive events"
        )
    window = rows[-args.min_consecutive:]
    if any(row.get("schema") != EVENT_SCHEMA for row in window):
        return fail(f"every journal event must use schema {EVENT_SCHEMA}")
    starts = {row.get("scheduler_started_at_unix") for row in window}
    if len(starts) != 1:
        return fail("the proof window crosses a scheduler restart")
    iterations = [row.get("iteration") for row in window]
    if any(not isinstance(value, int) for value in iterations):
        return fail("journal iterations must be integers")
    if iterations != list(range(iterations[0], iterations[0] + len(iterations))):
        return fail("journal iterations are not consecutive")
    checked = [row.get("checked_at_unix") for row in window]
    if any(not isinstance(value, (int, float)) for value in checked):
        return fail("journal checked_at_unix values must be numeric")
    if checked != sorted(checked):
        return fail("journal timestamps are not monotonic")
    age = max(0.0, time.time() - float(checked[-1]))
    if age > args.max_age_sec:
        return fail(f"latest reconciliation event is stale ({age:.3f}s)")
    if any(row.get("runtime_status") != "passed" for row in window):
        return fail("a reconciliation event in the proof window failed")
    if args.require_live and any(row.get("runtime_scheduler_mode") != "live" for row in window):
        return fail("the proof window contains a dry-run reconciliation")
    if any(row.get("runtime_topology_status") != "passed" for row in window):
        return fail("signed topology admission did not pass throughout the window")
    if any(row.get("runtime_queue_integrity_status") != "passed" for row in window):
        return fail("queue integrity did not pass throughout the window")
    if any(
        row.get("runtime_gpu_reconciliation_status") not in {"passed", "not_required"}
        for row in window
    ):
        return fail("GPU reconciliation did not pass throughout the window")

    heads = {row.get("source_git_head") for row in window}
    if len(heads) != 1 or None in heads:
        return fail("proof window does not bind one source revision")
    source_head = next(iter(heads))
    if args.git_head and source_head != args.git_head:
        return fail("journal source revision does not match --git-head")
    if args.require_clean_head and any(row.get("source_git_dirty") is not False for row in window):
        return fail("journal source revision was not clean")

    topology_digests = {row.get("topology_snapshot_sha256") for row in window}
    if len(topology_digests) != 1:
        return fail("proof window spans multiple topology snapshots")
    topology_digest = next(iter(topology_digests))
    if not isinstance(topology_digest, str) or re.fullmatch(r"[0-9a-f]{64}", topology_digest) is None:
        return fail("proof window lacks a topology SHA-256")

    eligible = eligible_resources(inventory)
    scheduled = scheduled_resources(jobs)
    inventory_rows = inventory.get("resources") or []
    if any(int(row.get("topology_resource_count") or -1) != len(inventory_rows) for row in window):
        return fail("topology resource count does not cover the scheduler inventory")
    missing = sorted(eligible - scheduled)
    if missing:
        return fail(f"eligible scheduler resources lack desired-state rows: {len(missing)}")
    expected_resources = sorted(scheduled)
    expected_digest = canonical_sha256(expected_resources)
    if any(row.get("result_resource_set_sha256") != expected_digest for row in window):
        return fail("reconciliation results do not cover the desired-state resource set")
    if any(int(row.get("error_count") or 0) != 0 for row in window):
        return fail("proof window contains scheduler errors")

    actions: dict[str, int] = {}
    for row in window:
        for action, count in (row.get("action_counts") or {}).items():
            actions[str(action)] = actions.get(str(action), 0) + int(count)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "runtime_status": "passed",
        "source_git_head": source_head,
        "source_git_dirty": any(row.get("source_git_dirty") is not False for row in window),
        "topology_snapshot_sha256": topology_digest,
        "eligible_resource_count": len(eligible),
        "scheduled_resource_count": len(scheduled),
        "window_event_count": len(window),
        "window_first_iteration": iterations[0],
        "window_last_iteration": iterations[-1],
        "window_age_sec": age,
        "action_counts": dict(sorted(actions.items())),
        "checks": {
            "tensorcore_scheduler_reconciler": {
                "runtime_status": "passed",
                "runtime_scheduler_mode": "live" if args.require_live else "validated",
                "runtime_source_status": "clean" if evidence_clean(window) else "dirty",
                "runtime_topology_status": "passed",
                "runtime_queue_integrity_status": "passed",
                "runtime_gpu_reconciliation_status": "passed",
                "runtime_resource_coverage_status": "passed",
                "runtime_sustained_window_status": "passed",
            }
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "scheduler reconciler evidence OK: "
        f"events={len(window)} resources={len(scheduled)} age={age:.3f}s"
    )
    return 0


def evidence_clean(window: list[dict]) -> bool:
    return all(row.get("source_git_dirty") is False for row in window)


if __name__ == "__main__":
    raise SystemExit(main())
