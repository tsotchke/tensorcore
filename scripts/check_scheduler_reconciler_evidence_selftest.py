#!/usr/bin/env python3
"""Fixture tests for the scheduler reconciliation evidence checker."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_scheduler_reconciler_evidence.py"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixtures(root: Path) -> tuple[Path, Path, Path]:
    resources = ["cpu:one", "gpu:one"]
    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({
        "resources": [
            {"id": resource, "status": "active", "general_queue_eligible": True,
             "control_plane": "tensorcore_scheduler"}
            for resource in resources
        ]
    }))
    jobs = root / "jobs.json"
    jobs.write_text(json.dumps({
        "jobs": [
            {"id": f"job-{index}", "resource": resource, "enabled": True}
            for index, resource in enumerate(resources)
        ]
    }))
    journal = root / "journal.jsonl"
    now = time.time()
    rows = [
        {
            "schema": "tensorcore.scheduler_reconciliation_event.v1",
            "checked_at_unix": now - (2 - index),
            "scheduler_started_at_unix": now - 10,
            "iteration": index + 7,
            "runtime_status": "passed",
            "runtime_scheduler_mode": "live",
            "runtime_topology_status": "passed",
            "runtime_queue_integrity_status": "passed",
            "runtime_gpu_reconciliation_status": "passed",
            "source_git_head": "head",
            "source_git_dirty": False,
            "topology_snapshot_sha256": "a" * 64,
            "topology_resource_count": 2,
            "result_resource_set_sha256": canonical_sha256(resources),
            "error_count": 0,
            "action_counts": {"idle_no_candidate": 2},
        }
        for index in range(3)
    ]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return journal, inventory, jobs


def run(root: Path, journal: Path, inventory: Path, jobs: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(CHECKER),
        "--journal", str(journal),
        "--inventory", str(inventory),
        "--jobs", str(jobs),
        "--output", str(root / "evidence.json"),
        "--git-head", "head",
        "--min-consecutive", "3",
        "--max-age-sec", "30",
        "--require-live",
        "--require-clean-head",
    ], text=True, capture_output=True, check=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        journal, inventory, jobs = fixtures(root)
        assert run(root, journal, inventory, jobs).returncode == 0

        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        restarted = copy.deepcopy(rows)
        restarted[-1]["scheduler_started_at_unix"] += 1
        journal.write_text("".join(json.dumps(row) + "\n" for row in restarted))
        result = run(root, journal, inventory, jobs)
        assert result.returncode != 0 and "crosses a scheduler restart" in result.stderr

        failed = copy.deepcopy(rows)
        failed[-1]["runtime_queue_integrity_status"] = "failed"
        journal.write_text("".join(json.dumps(row) + "\n" for row in failed))
        result = run(root, journal, inventory, jobs)
        assert result.returncode != 0 and "queue integrity" in result.stderr

    print("scheduler reconciler evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
