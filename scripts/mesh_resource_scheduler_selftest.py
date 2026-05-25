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
        inventory_json=None,
        state_json=None,
        timeout_sec=1.0,
        probe_timeout_sec=1.0,
        admission_timeout_sec=1.0,
        start_timeout_sec=1.0,
        post_start_timeout_sec=1.0,
        post_start_interval_sec=0.0,
        worker_identity_timeout_sec=1.0,
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
    completion_cmd: bool = False,
    admission_cmd: bool = False,
    post_start_probe_cmd: bool = False,
    worker_identity_cmd: bool = False,
    resource: str = "cosbox:cuda3090",
    resource_class: str | None = "generic",
) -> dict[str, Any]:
    out = {
        "id": job_id,
        "sync_id": job_id,
        "resource": resource,
        "owner": owner or f"{job_id}:cosbox",
        "priority": priority,
        "desired_state": desired_state,
        "ttl_sec": 60,
        "probe_cmd": ["probe", job_id],
        "start_cmd": ["start", job_id],
    }
    if resource_class is not None:
        out["resource_class"] = resource_class
    if completion_cmd:
        out["completion_cmd"] = ["complete", job_id]
    if admission_cmd:
        out["admission_cmd"] = ["admit", job_id]
    if post_start_probe_cmd:
        out["post_start_probe_cmd"] = ["post", job_id]
    if worker_identity_cmd:
        out["worker_identity_cmd"] = ["identity", job_id]
    return out


class FakeRuntime:
    def __init__(
        self,
        *,
        leases: list[dict[str, Any]] | None = None,
        live: dict[str, bool] | None = None,
        complete: dict[str, bool | None] | None = None,
        admitted: dict[str, bool | None] | None = None,
        start_rc: dict[str, int] | None = None,
        post_start: dict[str, bool | None] | None = None,
        identity: dict[str, bool | None] | None = None,
    ) -> None:
        self.leases = list(leases or [])
        self.live = dict(live or {})
        self.complete = dict(complete or {})
        self.admitted = dict(admitted or {})
        self.start_rc = dict(start_rc or {})
        self.post_start = dict(post_start or {})
        self.identity = dict(identity or {})
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
        if op == "complete":
            state = self.complete.get(ident, False)
            if state is None:
                raise subprocess.TimeoutExpired(argv, timeout)
            rc = 0 if state else 1
            stdout = (
                f"{ident} final_heldout_ppl=12.3 final_stored_size_bytes=456"
                if rc == 0 else ""
            )
            return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")
        if op == "admit":
            state = self.admitted.get(ident, True)
            if state is None:
                raise subprocess.TimeoutExpired(argv, timeout)
            rc = 0 if state else 1
            stdout = f"{ident} admission ok" if rc == 0 else ""
            stderr = "" if rc == 0 else f"{ident} admission failed"
            return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)
        if op == "post":
            state = self.post_start.get(ident, True)
            if state is None:
                raise subprocess.TimeoutExpired(argv, timeout)
            rc = 0 if state else 1
            stdout = f"{ident} post-start ok" if rc == 0 else ""
            stderr = "" if rc == 0 else f"{ident} post-start failed"
            return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)
        if op == "identity":
            state = self.identity.get(ident, True)
            if state is None:
                raise subprocess.TimeoutExpired(argv, timeout)
            rc = 0 if state else 1
            payload = {
                "worker_host": "cosbox",
                "worker_pid": 1234,
                "worker_systemd_unit": f"{ident}.service",
                "worker_cgroup": f"/user.slice/{ident}.service",
                "cuda_pids": [1234],
            }
            stdout = json.dumps(payload) if rc == 0 else ""
            stderr = "" if rc == 0 else f"{ident} identity failed"
            return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)
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


def test_incomplete_high_priority_job_runs_before_lower_priority_job() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False, "georefine-m2": False},
        complete={"georefine-m2": False},
    )
    result = run_case(
        [
            job("georefine-m2", priority=100, completion_cmd=True),
            job("qllm-phase1", priority=50),
        ],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "claimed_and_launched"
    assert result["results"][0]["job"] == "georefine-m2"
    assert result["results"][0]["completion"]["complete"] is False
    assert_event_order(
        runtime,
        [
            ("probe", "georefine-m2"),
            ("probe", "qllm-phase1"),
            ("complete", "georefine-m2"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "georefine-m2"),
        ],
    )


def test_completed_high_priority_job_is_not_relaunched() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False, "georefine-m2": False},
        complete={"georefine-m2": True},
    )
    result = run_case(
        [
            job("georefine-m2", priority=100, completion_cmd=True),
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
            ("complete", "georefine-m2"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
        ],
    )


def test_completed_stale_lease_is_released_before_next_job() -> None:
    runtime = FakeRuntime(
        leases=[
            {
                "id": "lease-geo",
                "resource": "cosbox:cuda3090",
                "owner": "georefine-m2:cosbox",
                "metadata": {"sync_job_id": "georefine-m2"},
            }
        ],
        live={"qllm-phase1": False, "georefine-m2": False},
        complete={"georefine-m2": True},
    )
    result = run_case(
        [
            job("georefine-m2", priority=100, completion_cmd=True),
            job("qllm-phase1", priority=50),
        ],
        runtime,
    )
    assert result["ok"] is True
    assert [row["action"] for row in result["results"]] == [
        "released_completed_lease",
        "claimed_and_launched",
    ]
    assert result["results"][1]["job"] == "qllm-phase1"
    assert_event_order(
        runtime,
        [
            ("probe", "georefine-m2"),
            ("probe", "qllm-phase1"),
            ("complete", "georefine-m2"),
            ("status", "--json"),
            ("release", "lease-geo"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
        ],
    )


def test_unknown_completion_does_not_relaunch_job() -> None:
    runtime = FakeRuntime(
        live={"georefine-m2": False},
        complete={"georefine-m2": None},
    )
    result = run_case(
        [job("georefine-m2", priority=100, completion_cmd=True)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "idle_completion_unknown"
    assert result["results"][0]["jobs"] == ["georefine-m2"]
    assert_event_order(
        runtime,
        [
            ("probe", "georefine-m2"),
            ("complete", "georefine-m2"),
            ("status", "--json"),
        ],
    )


def test_admission_failure_skips_blocked_high_priority_job() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False, "georefine-m2": False},
        admitted={"qllm-phase1": False, "georefine-m2": True},
    )
    result = run_case(
        [
            job("qllm-phase1", priority=100, admission_cmd=True),
            job("georefine-m2", priority=50, admission_cmd=True),
        ],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "claimed_and_launched"
    assert result["results"][0]["job"] == "georefine-m2"
    assert result["results"][0]["admission"]["admitted"] is True
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("probe", "georefine-m2"),
            ("admit", "qllm-phase1"),
            ("admit", "georefine-m2"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "georefine-m2"),
        ],
    )


def test_admission_failure_blocks_when_no_admitted_candidate() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False},
        admitted={"qllm-phase1": False},
    )
    result = run_case(
        [job("qllm-phase1", priority=100, admission_cmd=True)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "idle_admission_blocked"
    assert result["results"][0]["jobs"] == ["qllm-phase1"]
    assert result["results"][0]["admissions"]["qllm-phase1"]["admitted"] is False
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("admit", "qllm-phase1"),
            ("status", "--json"),
        ],
    )


def test_admission_timeout_blocks_launch_for_that_pass() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False},
        admitted={"qllm-phase1": None},
    )
    result = run_case(
        [job("qllm-phase1", priority=100, admission_cmd=True)],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "idle_admission_blocked"
    admission = result["results"][0]["admissions"]["qllm-phase1"]
    assert admission["admitted"] is None
    assert admission["reason"] == "admission_timeout"
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("admit", "qllm-phase1"),
            ("status", "--json"),
        ],
    )


def test_cuda_job_requires_admission_post_start_and_identity() -> None:
    scheduler = load_scheduler()
    with tempfile.TemporaryDirectory() as tmp:
        bad_job = job("qllm-phase1", priority=50, resource_class=None)
        path = write_jobs(pathlib.Path(tmp), [bad_job])
        try:
            scheduler.schedule_once(args_for(path))
        except ValueError as exc:
            assert "requires admission_cmd" in str(exc)
        else:
            raise AssertionError("cuda job without admission_cmd was accepted")


def test_job_schema_rejects_string_boolean() -> None:
    scheduler = load_scheduler()
    with tempfile.TemporaryDirectory() as tmp:
        bad_job = job("qllm-phase1", priority=50)
        bad_job["enabled"] = "false"
        path = write_jobs(pathlib.Path(tmp), [bad_job])
        try:
            scheduler.schedule_once(args_for(path))
        except ValueError as exc:
            assert "must be a JSON boolean" in str(exc)
        else:
            raise AssertionError("string boolean was accepted")


def test_inventory_rejects_reserved_resource_for_unlisted_owner() -> None:
    scheduler = load_scheduler()
    jobs = [job("assistant-metal", priority=1, resource="enki:metal_m4_tsotchke_chan")]
    inventory = {
        "enki:metal_m4_tsotchke_chan": {
            "id": "enki:metal_m4_tsotchke_chan",
            "general_queue_eligible": False,
            "reserved_for": ["tsotchke-chan", "tsotchke-chan:*"],
            "status": "reserved",
        }
    }
    try:
        scheduler.validate_jobs_against_inventory(jobs, inventory)
    except ValueError as exc:
        assert "reserved resource" in str(exc)
    else:
        raise AssertionError("reserved resource accepted an unlisted owner")


def test_inventory_allows_reserved_resource_owner_prefix() -> None:
    scheduler = load_scheduler()
    jobs = [
        job(
            "tsotchke-chan-metal",
            priority=1,
            owner="tsotchke-chan:interactive",
            resource="enki:metal_m4_tsotchke_chan",
        )
    ]
    inventory = {
        "enki:metal_m4_tsotchke_chan": {
            "id": "enki:metal_m4_tsotchke_chan",
            "general_queue_eligible": False,
            "reserved_for": ["tsotchke-chan", "tsotchke-chan:*"],
            "status": "reserved",
        }
    }
    scheduler.validate_jobs_against_inventory(jobs, inventory)


def test_inventory_blocks_running_job_on_blocked_resource() -> None:
    scheduler = load_scheduler()
    jobs = [job("jack-cuda", priority=1, resource="jack-blupc:cuda3060")]
    inventory = {
        "jack-blupc:cuda3060": {
            "id": "jack-blupc:cuda3060",
            "general_queue_eligible": False,
            "status": "blocked",
            "blocked_reason": "ssh unavailable",
        }
    }
    try:
        scheduler.validate_jobs_against_inventory(jobs, inventory)
    except ValueError as exc:
        assert "blocked resource" in str(exc)
    else:
        raise AssertionError("blocked resource accepted a running job")


def test_inventory_rejects_bad_resource_rows() -> None:
    scheduler = load_scheduler()
    bad_rows = [
        {
            "cosbox:cuda3090": {
                "id": "cosbox:cuda3090",
                "capacity": 0,
            },
            "needle": "capacity must be a positive integer",
        },
        {
            "cosbox:cuda3090": {
                "id": "cosbox:cuda3090",
                "status": "offline",
            },
            "needle": "status must be one of",
        },
        {
            "cosbox:cuda3090": {
                "id": "cosbox:cuda3090",
                "status": "blocked",
            },
            "needle": "status=blocked requires blocked_reason",
        },
        {
            "cosbox:cuda3090": {
                "id": "cosbox:cuda3090",
                "general_queue_eligible": "false",
            },
            "needle": "general_queue_eligible must be a JSON boolean",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for index, case in enumerate(bad_rows):
            path = pathlib.Path(tmp) / f"bad-inventory-{index}.json"
            path.write_text(
                json.dumps({
                    "schema": "tensorcore.mesh_resources.v1",
                    "resources": [case["cosbox:cuda3090"]],
                }),
                encoding="utf-8",
            )
            try:
                scheduler.load_inventory(str(path))
            except ValueError as exc:
                assert case["needle"] in str(exc)
            else:
                raise AssertionError(f"bad inventory row {index} was accepted")


def test_inventory_blocks_non_general_resource_without_allowlist() -> None:
    scheduler = load_scheduler()
    jobs = [job("metal", priority=1, resource="atlas:private_metal")]
    inventory = {
        "atlas:private_metal": {
            "id": "atlas:private_metal",
            "general_queue_eligible": False,
            "status": "reserved",
        }
    }
    try:
        scheduler.validate_jobs_against_inventory(jobs, inventory)
    except ValueError as exc:
        assert "with no reserved_for allow-list" in str(exc)
    else:
        raise AssertionError("non-general resource without allow-list accepted a job")


def test_cuda_launch_runs_post_start_and_identity() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": False})
    result = run_case(
        [
            job(
                "qllm-phase1",
                priority=100,
                resource_class="cuda_exclusive",
                admission_cmd=True,
                post_start_probe_cmd=True,
                worker_identity_cmd=True,
            )
        ],
        runtime,
    )
    assert result["ok"] is True
    row = result["results"][0]
    assert row["action"] == "claimed_and_launched"
    assert row["post_start"]["verified"] is True
    assert row["worker_identity"]["ok"] is True
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("admit", "qllm-phase1"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
            ("post", "qllm-phase1"),
            ("identity", "qllm-phase1"),
        ],
    )


def test_cuda_post_start_failure_releases_claimed_lease() -> None:
    runtime = FakeRuntime(
        live={"qllm-phase1": False},
        post_start={"qllm-phase1": False},
    )
    result = run_case(
        [
            job(
                "qllm-phase1",
                priority=100,
                resource_class="cuda_exclusive",
                admission_cmd=True,
                post_start_probe_cmd=True,
                worker_identity_cmd=True,
            )
        ],
        runtime,
    )
    assert result["ok"] is False
    row = result["results"][0]
    assert row["post_start"]["verified"] is False
    assert row["release_after_failed_start"]["ok"] is True
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("admit", "qllm-phase1"),
            ("status", "--json"),
            ("claim", "cosbox:cuda3090"),
            ("start", "qllm-phase1"),
            ("post", "qllm-phase1"),
            ("release", "lease-new-1"),
        ],
    )


def test_cuda_live_adoption_records_worker_identity_in_claim() -> None:
    runtime = FakeRuntime(live={"qllm-phase1": True})
    result = run_case(
        [
            job(
                "qllm-phase1",
                priority=100,
                resource_class="cuda_exclusive",
                admission_cmd=True,
                post_start_probe_cmd=True,
                worker_identity_cmd=True,
            )
        ],
        runtime,
    )
    assert result["ok"] is True
    assert result["results"][0]["action"] == "adopted_live_holder"
    assert runtime.leases[0]["metadata"]["worker_identity"]["worker_host"] == "cosbox"
    assert runtime.leases[0]["metadata"]["worker_identity_pending"] is False
    assert_event_order(
        runtime,
        [
            ("probe", "qllm-phase1"),
            ("admit", "qllm-phase1"),
            ("status", "--json"),
            ("identity", "qllm-phase1"),
            ("claim", "cosbox:cuda3090"),
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
    test_incomplete_high_priority_job_runs_before_lower_priority_job()
    test_completed_high_priority_job_is_not_relaunched()
    test_completed_stale_lease_is_released_before_next_job()
    test_unknown_completion_does_not_relaunch_job()
    test_admission_failure_skips_blocked_high_priority_job()
    test_admission_failure_blocks_when_no_admitted_candidate()
    test_admission_timeout_blocks_launch_for_that_pass()
    test_cuda_job_requires_admission_post_start_and_identity()
    test_job_schema_rejects_string_boolean()
    test_inventory_rejects_reserved_resource_for_unlisted_owner()
    test_inventory_allows_reserved_resource_owner_prefix()
    test_inventory_blocks_running_job_on_blocked_resource()
    test_inventory_rejects_bad_resource_rows()
    test_inventory_blocks_non_general_resource_without_allowlist()
    test_cuda_launch_runs_post_start_and_identity()
    test_cuda_post_start_failure_releases_claimed_lease()
    test_cuda_live_adoption_records_worker_identity_in_claim()
    test_loop_pretty_json_emits_json()
    print("mesh resource scheduler selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
