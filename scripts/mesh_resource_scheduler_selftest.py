#!/usr/bin/env python3
"""Selftests for scripts/mesh_resource_scheduler.py."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import subprocess
import tempfile
from contextlib import redirect_stdout
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "mesh_resource_scheduler.py"


def load_scheduler() -> ModuleType:
    candidates = [
        SCHEDULER,
        pathlib.Path(__file__).with_name("mesh-resource-scheduler"),
        pathlib.Path(__file__).with_name("mesh_resource_scheduler.py"),
    ]
    path = next((item for item in candidates if item.exists()), SCHEDULER)
    loader = importlib.machinery.SourceFileLoader(
        "mesh_resource_scheduler_under_test",
        str(path),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jobs(directory: pathlib.Path, jobs: list[dict[str, Any]]) -> pathlib.Path:
    path = directory / "jobs.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tensorcore.mesh_resource_jobs.v1",
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return path


def args_for(path: pathlib.Path, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        arbiter_cmd="arbiter",
        jobs_json=str(path),
        state_json=None,
        timeout_sec=1.0,
        probe_timeout_sec=1.0,
        start_timeout_sec=1.0,
        dry_run=dry_run,
        json=False,
        pretty_json=False,
        loop=False,
        interval_sec=0.0,
        max_iterations=0,
    )


def job(
    job_id: str,
    *,
    priority: int,
    owner: str | None = None,
    desired_state: str = "running",
) -> dict[str, Any]:
    return {
        "id": job_id,
        "sync_id": job_id,
        "resource": "cosbox:cuda3090",
        "owner": owner or f"{job_id}:cosbox",
        "priority": priority,
        "desired_state": desired_state,
        "ttl_sec": 60,
        "probe_cmd": ["probe", job_id],
        "start_cmd": ["start", job_id],
    }


class FakeRuntime:
    def __init__(
        self,
        *,
        leases: list[dict[str, Any]] | None = None,
        live: dict[str, bool] | None = None,
        start_rc: dict[str, int] | None = None,
    ) -> None:
        self.leases = list(leases or [])
        self.live = dict(live or {})
        self.start_rc = dict(start_rc or {})
        self.events: list[tuple[str, str | None]] = []
        self.next_lease = 1

    def run_json(self, argv: list[str], *, timeout: float) -> dict:
        op = argv[1]
        self.events.append((op, argv[2] if len(argv) > 2 else None))
        if op == "status":
            return {"leases": list(self.leases)}
        if op == "claim":
            resource = argv[2]
            owner = argv[argv.index("--owner") + 1]
            metadata = json.loads(argv[argv.index("--metadata-json") + 1])
            lease_id = f"lease-new-{self.next_lease}"
            self.next_lease += 1
            self.leases.append(
                {
                    "id": lease_id,
                    "resource": resource,
                    "owner": owner,
                    "metadata": metadata,
                }
            )
            return {"ok": True, "lease_id": lease_id}
        if op == "heartbeat":
            return {"ok": True, "lease_id": argv[2]}
        if op == "release":
            lease_id = argv[2]
            self.leases = [lease for lease in self.leases if lease.get("id") != lease_id]
            return {"ok": True, "lease_id": lease_id}
        raise AssertionError(f"unexpected arbiter argv: {argv!r}")

    def run_capture(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        op = argv[0]
        ident = argv[1] if len(argv) > 1 else ""
        self.events.append((op, ident))
        if op == "probe":
            rc = 0 if self.live.get(ident, False) else 1
            return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")
        if op == "start":
            rc = self.start_rc.get(ident, 0)
            return subprocess.CompletedProcess(argv, rc, stdout=f"started {ident}", stderr="")
        raise AssertionError(f"unexpected command argv: {argv!r}")


def run_case(
    jobs: list[dict[str, Any]],
    runtime: FakeRuntime,
    *,
    dry_run: bool = False,
) -> dict:
    scheduler = load_scheduler()
    scheduler.run_json = runtime.run_json
    scheduler.run_capture = runtime.run_capture
    with tempfile.TemporaryDirectory() as tmp:
        path = write_jobs(pathlib.Path(tmp), jobs)
        return scheduler.schedule_once(args_for(path, dry_run=dry_run))


def assert_event_order(runtime: FakeRuntime, expected: list[tuple[str, str | None]]) -> None:
    if runtime.events != expected:
        raise AssertionError(f"expected events {expected!r}, got {runtime.events!r}")


def test_idle_claims_highest_priority() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": False, "georefine-m2": False})
    result = run_case(
        [
            job("georefine-m2", priority=10),
            job("qllm-phase1", priority=50),
        ],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "claimed_and_launched"
    assert result["results"][0]["job"] == "qllm-phase1"
    assert_event_order(
        runtime,
        [
            ("probe", "georefine-m2"),
            ("probe", "qllm-phase1"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
        ],
    )


def test_live_holder_is_heartbeated() -> None:
    runtime = FakeRuntime(
        leases=[
            {
                "id": "lease-qllm",
                "resource": "cosbox:cuda3090",
                "owner": "qllm-phase1:cosbox",
                "metadata": {"sync_job_id": "qllm-phase1"},
            }
        ],
        live={"qllm-phase1": True, "georefine-m2": False},
    )
    result = run_case(
        [job("qllm-phase1", priority=50), job("georefine-m2", priority=10)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "heartbeated_live_holder"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
            ("heartbeat", "lease-qllm"),
        ],
    )


def test_live_holder_without_lease_is_adopted() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": True, "georefine-m2": False})
    result = run_case(
        [job("qllm-phase1", priority=50), job("georefine-m2", priority=10)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "adopted_live_holder"
    assert result["results"][0]["job"] == "qllm-phase1"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
        ],
    )


def test_stale_known_lease_is_released_before_launch() -> None:
    runtime = FakeRuntime(
        leases=[
            {
                "id": "lease-geo",
                "resource": "cosbox:cuda3090",
                "owner": "georefine-m2:old-pid",
                "metadata": {"sync_job_id": "georefine-m2"},
            }
        ],
        live={"qllm-phase1": False, "georefine-m2": False},
    )
    result = run_case(
        [job("qllm-phase1", priority=50), job("georefine-m2", priority=10)],
        runtime,
    )
    assert result["ok"] is True
    assert [row["action"] for row in result["results"]] == [
        "released_stale_lease",
        "claimed_and_launched",
    ]
    assert result["results"][1]["job"] == "qllm-phase1"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
            ("release", "lease-geo"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
        ],
    )


def test_unknown_lease_blocks_launch() -> None:
    runtime = FakeRuntime(
        leases=[
            {
                "id": "lease-other",
                "resource": "cosbox:cuda3090",
                "owner": "unknown-agent",
                "metadata": {"sync_job_id": "unknown"},
            }
        ],
        live={"qllm-phase1": False, "georefine-m2": False},
    )
    result = run_case(
        [job("qllm-phase1", priority=50), job("georefine-m2", priority=10)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "resource_busy_unknown_lease"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
        ],
    )


def test_known_lease_with_unknown_liveness_blocks_live_adoption() -> None:
    georefine = job("georefine-m2", priority=10)
    georefine.pop("probe_cmd")
    runtime = FakeRuntime(
        leases=[
            {
                "id": "lease-geo",
                "resource": "cosbox:cuda3090",
                "owner": "georefine-m2:cosbox",
                "metadata": {"sync_job_id": "georefine-m2"},
            }
        ],
        live={"qllm-phase1": True},
    )
    result = run_case([job("qllm-phase1", priority=50), georefine], runtime)
    assert result["ok"] is True
    assert (
        result["results"][0]["action"]
        == "live_holder_blocked_by_known_lease_unknown_liveness"
    )
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("status", "--json"),
        ],
    )


def test_paused_live_job_still_holds_resource() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": False, "georefine-m2": True})
    result = run_case(
        [
            job("qllm-phase1", priority=50),
            job("georefine-m2", priority=10, desired_state="paused"),
        ],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "adopted_live_holder"
    assert result["results"][0]["job"] == "georefine-m2"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
        ],
    )


def test_multiple_live_holders_is_an_error() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": True, "georefine-m2": True})
    result = run_case(
        [job("qllm-phase1", priority=50), job("georefine-m2", priority=10)],
        runtime,
    )
    assert result["ok"] is False
    assert "multiple live holders" in result["errors"][0]["error"]
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("status", "--json"),
        ],
    )


def test_failed_launch_releases_claimed_lease() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False},
        start_rc={"qllm-phase1": 7},
    )
    result = run_case([job("qllm-phase1", priority=50)], runtime)
    assert result["ok"] is False
    assert result["results"][0]["action"] == "claimed_and_launched"
    assert result["results"][0]["release_after_failed_start"]["ok"] is True
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
            ("release", "lease-new-1"),
        ],
    )


def test_loop_pretty_json_emits_json() -> None:
    scheduler = load_scheduler()
    runtime = FakeRuntime(live={"qllm-phase1": False})
    scheduler.run_json = runtime.run_json
    scheduler.run_capture = runtime.run_capture
    with tempfile.TemporaryDirectory() as tmp:
        path = write_jobs(pathlib.Path(tmp), [job("qllm-phase1", priority=50)])
        args = args_for(path)
        args.loop = True
        args.pretty_json = True
        args.max_iterations = 1
        out = io.StringIO()
        with redirect_stdout(out):
            rc = scheduler.run_loop(args)
    payload = json.loads(out.getvalue())
    assert rc == 0
    assert payload["iteration"] == 1
    assert payload["results"][0]["action"] == "claimed_and_launched"
    assert out.getvalue().startswith("{\n")


def main() -> int:
    test_idle_claims_highest_priority()
    test_live_holder_is_heartbeated()
    test_live_holder_without_lease_is_adopted()
    test_stale_known_lease_is_released_before_launch()
    test_unknown_lease_blocks_launch()
    test_known_lease_with_unknown_liveness_blocks_live_adoption()
    test_paused_live_job_still_holds_resource()
    test_multiple_live_holders_is_an_error()
    test_failed_launch_releases_claimed_lease()
    test_loop_pretty_json_emits_json()
    print("mesh resource scheduler selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
