#!/usr/bin/env python3
"""Coordinate shared mesh resources through the Tsotchke arbiter.

This is a control-plane scheduler, not a workload-specific launcher. It reads
a small jobs file, probes known work, reconciles stale leases, and only starts a
job after it has claimed the requested resource. The policy is intentionally
non-destructive: live work is never killed for priority alone, and unknown
leases are treated as busy resources.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ARBITER_CMD = "~/.tsotchke/bin/tsotchke-arbiter"
SCHEMA = "tensorcore.mesh_resource_jobs.v1"


def command(value: Any) -> list[str]:
    if isinstance(value, list):
        argv = [str(part) for part in value]
    elif isinstance(value, str):
        argv = shlex.split(value)
    else:
        return []
    if argv and argv[0].startswith("~"):
        argv[0] = str(Path(argv[0]).expanduser())
    return argv


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_json(argv: list[str], *, timeout: float) -> dict:
    proc = run_capture(argv, timeout=timeout)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"{argv!r} failed rc={proc.returncode}: {detail}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{argv!r} did not return JSON: {proc.stdout[:240]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{argv!r} returned non-object JSON")
    return data


def write_json(path: str, payload: dict) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


def load_jobs(path: str) -> list[dict]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        jobs = raw.get("jobs")
    else:
        jobs = raw
    if not isinstance(jobs, list):
        raise SystemExit("--jobs-json must contain a list or an object with jobs")
    return [normalize_job(job) for job in jobs]


def normalize_job(job: Any) -> dict:
    if not isinstance(job, dict):
        raise SystemExit("--jobs-json contains a non-object job")
    out = dict(job)
    missing = [key for key in ("id", "resource", "owner") if not out.get(key)]
    if missing:
        raise ValueError(f"job missing required field(s): {', '.join(missing)}")
    out["id"] = str(out["id"])
    out["sync_id"] = str(out.get("sync_id") or out["id"])
    out["resource"] = str(out["resource"])
    out["owner"] = str(out["owner"])
    out["priority"] = int(out.get("priority", 0))
    out["enabled"] = bool(out.get("enabled", True))
    out["ttl_sec"] = float(out.get("ttl_sec", 900.0))
    out["desired_state"] = str(out.get("desired_state", "running"))

    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.setdefault("surface", "tensorcore_mesh_scheduler")
    metadata["sync_job_id"] = out["sync_id"]
    metadata["job_id"] = out["id"]
    out["metadata"] = metadata
    return out


def probe_job(job: dict, *, timeout: float) -> dict:
    probe_cmd = command(job.get("probe_cmd"))
    if not probe_cmd:
        return {"live": None, "reason": "missing_probe_cmd", "rc": None}
    try:
        proc = run_capture(probe_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"live": None, "reason": "probe_timeout", "rc": None}
    return {
        "live": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "probe_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-240:],
        "stderr_tail": proc.stderr.strip()[-240:],
    }


def complete_job(job: dict, *, timeout: float) -> dict:
    completion_cmd = command(job.get("completion_cmd"))
    if not completion_cmd:
        return {"complete": False, "reason": "missing_completion_cmd", "rc": None}
    try:
        proc = run_capture(completion_cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"complete": None, "reason": "completion_timeout", "rc": None}
    return {
        "complete": proc.returncode == 0,
        "reason": "ok" if proc.returncode == 0 else "completion_failed",
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def lease_metadata(row: dict) -> dict:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def lease_sync_id(row: dict) -> str | None:
    metadata = lease_metadata(row)
    value = (
        row.get("sync_job_id")
        or row.get("sync_id")
        or metadata.get("sync_job_id")
        or metadata.get("sync_id")
    )
    return str(value) if value else None


def dedupe_leases(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str | int]] = set()
    for row in rows:
        lease_id = row.get("id")
        key = ("id", str(lease_id)) if lease_id else ("object", id(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def leases_for_resource(status: dict, resource: str) -> list[dict]:
    return [
        row
        for row in status.get("leases") or []
        if isinstance(row, dict) and row.get("resource") == resource
    ]


def job_for_lease(lease: dict, jobs: list[dict]) -> dict | None:
    sync_id = lease_sync_id(lease)
    for job in jobs:
        if lease.get("resource") != job["resource"]:
            continue
        if sync_id and sync_id == job["sync_id"]:
            return job
        if lease.get("owner") == job["owner"]:
            return job
    return None


def matching_leases(status: dict, job: dict) -> list[dict]:
    logical = []
    owner = []
    for row in leases_for_resource(status, job["resource"]):
        if lease_sync_id(row) == job["sync_id"]:
            logical.append(row)
        if row.get("owner") == job["owner"]:
            owner.append(row)
    if logical:
        return dedupe_leases(logical + owner)
    return dedupe_leases(owner)


def claim_job(job: dict, *, arbiter_cmd: list[str], timeout: float) -> dict:
    return run_json(
        arbiter_cmd
        + [
            "claim",
            job["resource"],
            "--owner",
            job["owner"],
            "--ttl-sec",
            str(job["ttl_sec"]),
            "--metadata-json",
            json.dumps(job["metadata"], sort_keys=True),
            "--json",
        ],
        timeout=timeout,
    )


def heartbeat_lease(lease: dict, job: dict, *, arbiter_cmd: list[str], timeout: float) -> dict:
    return run_json(
        arbiter_cmd
        + ["heartbeat", str(lease["id"]), "--ttl-sec", str(job["ttl_sec"]), "--json"],
        timeout=timeout,
    )


def release_lease(lease: dict, *, arbiter_cmd: list[str], timeout: float) -> dict:
    return run_json(
        arbiter_cmd + ["release", str(lease["id"]), "--json"],
        timeout=timeout,
    )


def release_many(
    leases: list[dict],
    *,
    arbiter_cmd: list[str],
    timeout: float,
) -> tuple[list[dict], bool]:
    payloads = []
    ok = True
    for lease in leases:
        payload = release_lease(lease, arbiter_cmd=arbiter_cmd, timeout=timeout)
        payloads.append(payload)
        ok = ok and bool(payload.get("ok"))
    return payloads, ok


def start_job(job: dict, *, timeout: float) -> dict:
    start_cmd = command(job.get("start_cmd"))
    if not start_cmd:
        raise RuntimeError(f"job {job['id']} has no start_cmd")
    proc = run_capture(start_cmd, timeout=timeout)
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def enabled_running_jobs(jobs: list[dict], resource: str) -> list[dict]:
    return [
        job
        for job in jobs
        if job["resource"] == resource
        and job["enabled"]
        and job["desired_state"] == "running"
    ]


def enabled_jobs(jobs: list[dict], resource: str) -> list[dict]:
    return [
        job
        for job in jobs
        if job["resource"] == resource and job["enabled"]
    ]


def choose_candidate(
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    resource: str,
) -> dict | None:
    candidates = [
        job
        for job in enabled_running_jobs(jobs, resource)
        if probes[job["id"]]["live"] is False
        and completions[job["id"]]["complete"] is False
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item["priority"], item["id"]))[0]


def reconcile_stale_leases(
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    resource: str,
    status: dict,
    *,
    arbiter_cmd: list[str],
    timeout: float,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    results = []
    errors = []
    for job in jobs:
        if job["resource"] != resource:
            continue
        job_probe = probes[job["id"]]
        if job_probe["live"] is not False:
            continue
        completion = completions[job["id"]]
        leases = matching_leases(status, job)
        if not leases:
            continue
        if completion["complete"] is True:
            action = (
                "would_release_completed_lease"
                if dry_run
                else "released_completed_lease"
            )
        else:
            action = "would_release_stale_lease" if dry_run else "released_stale_lease"
        result = {
            "resource": resource,
            "job": job["id"],
            "action": action,
            "lease_ids": [lease.get("id") for lease in leases],
            "probe": job_probe,
            "completion": completion,
            "ok": True,
        }
        if not dry_run:
            try:
                payloads, ok = release_many(
                    leases,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                result["arbiter"] = payloads
                result["ok"] = ok
            except Exception as exc:
                result["ok"] = False
                errors.append({"resource": resource, "job": job["id"], "error": str(exc)})
        results.append(result)
    return results, errors


def resource_leases_after_reconcile(
    leases: list[dict],
    stale_results: list[dict],
    dry_run: bool,
) -> list[dict]:
    if dry_run:
        return leases
    released = {
        str(lease_id)
        for row in stale_results
        for lease_id in row.get("lease_ids", [])
        if lease_id is not None and row.get("ok")
    }
    return [lease for lease in leases if str(lease.get("id")) not in released]


def handle_live_holders(
    live_jobs: list[dict],
    known_leases: list[dict],
    unknown_leases: list[dict],
    probes: dict[str, dict],
    *,
    arbiter_cmd: list[str],
    timeout: float,
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    results = []
    errors = []
    if len(live_jobs) > 1:
        return [], [{
            "resource": live_jobs[0]["resource"] if live_jobs else "-",
            "error": "multiple live holders detected; refusing destructive arbitration",
            "jobs": [job["id"] for job in live_jobs],
        }]

    job = live_jobs[0]
    matches = [
        lease
        for lease in known_leases
        if job_for_lease(lease, [job]) is not None
    ]
    current = matches[0] if matches else None
    stale = matches[1:] if matches else []
    other_known = [
        lease
        for lease in known_leases
        if all(lease.get("id") != match.get("id") for match in matches)
    ]

    if current:
        action = "would_heartbeat_live_holder" if dry_run else "heartbeated_live_holder"
        result = {
            "resource": job["resource"],
            "job": job["id"],
            "action": action,
            "lease_id": current.get("id"),
            "stale_lease_ids": [lease.get("id") for lease in stale],
            "probe": probes[job["id"]],
            "ok": not other_known,
        }
        if other_known:
            result["action"] = "live_holder_conflicting_known_lease"
            result["other_lease_ids"] = [lease.get("id") for lease in other_known]
        if not dry_run:
            try:
                releases, releases_ok = release_many(
                    stale,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                payload = heartbeat_lease(
                    current,
                    job,
                    arbiter_cmd=arbiter_cmd,
                    timeout=timeout,
                )
                result["arbiter"] = {"release": releases, "heartbeat": payload}
                result["ok"] = (
                    not other_known
                    and releases_ok
                    and bool(payload.get("ok"))
                )
            except Exception as exc:
                result["ok"] = False
                errors.append({"resource": job["resource"], "job": job["id"], "error": str(exc)})
        results.append(result)
        return results, errors

    if other_known:
        results.append({
            "resource": job["resource"],
            "job": job["id"],
            "action": "live_holder_blocked_by_known_lease_unknown_liveness",
            "lease_ids": [lease.get("id") for lease in other_known],
            "ok": True,
        })
        return results, errors

    if unknown_leases:
        results.append({
            "resource": job["resource"],
            "job": job["id"],
            "action": "live_holder_blocked_by_unknown_lease",
            "lease_ids": [lease.get("id") for lease in unknown_leases],
            "ok": True,
        })
        return results, errors

    action = "would_adopt_live_holder" if dry_run else "adopted_live_holder"
    result = {
        "resource": job["resource"],
        "job": job["id"],
        "action": action,
        "probe": probes[job["id"]],
        "ok": True,
    }
    if not dry_run:
        try:
            payload = claim_job(job, arbiter_cmd=arbiter_cmd, timeout=timeout)
            result["arbiter"] = payload
            result["ok"] = bool(payload.get("ok"))
            result["lease_id"] = payload.get("lease_id") or payload.get("id")
        except Exception as exc:
            result["ok"] = False
            errors.append({"resource": job["resource"], "job": job["id"], "error": str(exc)})
    results.append(result)
    return results, errors


def schedule_resource(
    resource: str,
    jobs: list[dict],
    probes: dict[str, dict],
    completions: dict[str, dict],
    status: dict,
    args: argparse.Namespace,
    *,
    arbiter_cmd: list[str],
) -> tuple[list[dict], list[dict]]:
    results, errors = reconcile_stale_leases(
        jobs,
        probes,
        completions,
        resource,
        status,
        arbiter_cmd=arbiter_cmd,
        timeout=args.timeout_sec,
        dry_run=args.dry_run,
    )
    leases = resource_leases_after_reconcile(
        leases_for_resource(status, resource),
        results,
        args.dry_run,
    )
    known_leases = [lease for lease in leases if job_for_lease(lease, jobs) is not None]
    unknown_leases = [lease for lease in leases if job_for_lease(lease, jobs) is None]
    live_jobs = [
        job
        for job in enabled_jobs(jobs, resource)
        if probes[job["id"]]["live"] is True
    ]

    if live_jobs:
        live_results, live_errors = handle_live_holders(
            live_jobs,
            known_leases,
            unknown_leases,
            probes,
            arbiter_cmd=arbiter_cmd,
            timeout=args.timeout_sec,
            dry_run=args.dry_run,
        )
        results.extend(live_results)
        errors.extend(live_errors)
        return results, errors

    if unknown_leases:
        results.append({
            "resource": resource,
            "action": "resource_busy_unknown_lease",
            "lease_ids": [lease.get("id") for lease in unknown_leases],
            "ok": True,
        })
        return results, errors

    if known_leases:
        results.append({
            "resource": resource,
            "action": "resource_busy_known_lease_unknown_liveness",
            "lease_ids": [lease.get("id") for lease in known_leases],
            "ok": True,
        })
        return results, errors

    candidate = choose_candidate(jobs, probes, completions, resource)
    if candidate is None:
        completed_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if completions[job["id"]]["complete"] is True
        ]
        if completed_jobs:
            results.append({
                "resource": resource,
                "action": "idle_completed_jobs",
                "jobs": completed_jobs,
                "ok": True,
            })
            return results, errors
        unknown_completion_jobs = [
            job["id"]
            for job in enabled_running_jobs(jobs, resource)
            if probes[job["id"]]["live"] is False
            and completions[job["id"]]["complete"] is None
        ]
        if unknown_completion_jobs:
            results.append({
                "resource": resource,
                "action": "idle_completion_unknown",
                "jobs": unknown_completion_jobs,
                "ok": True,
            })
            return results, errors
        results.append({"resource": resource, "action": "idle_no_candidate", "ok": True})
        return results, errors

    if args.dry_run:
        results.append({
            "resource": resource,
            "job": candidate["id"],
            "action": "would_claim_and_launch",
            "probe": probes[candidate["id"]],
            "completion": completions[candidate["id"]],
            "ok": True,
        })
        return results, errors

    claim_payload = claim_job(candidate, arbiter_cmd=arbiter_cmd, timeout=args.timeout_sec)
    result = {
        "resource": resource,
        "job": candidate["id"],
        "action": "claimed_and_launched",
        "arbiter": claim_payload,
        "completion": completions[candidate["id"]],
        "ok": bool(claim_payload.get("ok")),
    }
    if not claim_payload.get("ok"):
        results.append(result)
        return results, errors

    try:
        start_payload = start_job(candidate, timeout=args.start_timeout_sec)
        result["start"] = start_payload
        result["ok"] = bool(start_payload.get("ok"))
    except Exception as exc:
        result["ok"] = False
        result["start_error"] = str(exc)

    if not result["ok"]:
        lease_id = claim_payload.get("lease_id") or claim_payload.get("id")
        if lease_id:
            try:
                result["release_after_failed_start"] = release_lease(
                    {"id": lease_id},
                    arbiter_cmd=arbiter_cmd,
                    timeout=args.timeout_sec,
                )
            except Exception as exc:
                errors.append({
                    "resource": resource,
                    "job": candidate["id"],
                    "error": f"failed to release lease after failed start: {exc}",
                })
    results.append(result)
    return results, errors


def schedule_once(args: argparse.Namespace) -> dict:
    arbiter_cmd = command(args.arbiter_cmd)
    if not arbiter_cmd:
        raise SystemExit("--arbiter-cmd resolved to an empty command")
    jobs = load_jobs(args.jobs_json)
    probes = {job["id"]: probe_job(job, timeout=args.probe_timeout_sec) for job in jobs}
    completions = {
        job["id"]: complete_job(job, timeout=args.probe_timeout_sec)
        for job in jobs
    }
    status = run_json(arbiter_cmd + ["status", "--json"], timeout=args.timeout_sec)
    resources = sorted({job["resource"] for job in jobs})
    results = []
    errors = []
    for resource in resources:
        try:
            resource_results, resource_errors = schedule_resource(
                resource,
                jobs,
                probes,
                completions,
                status,
                args,
                arbiter_cmd=arbiter_cmd,
            )
            results.extend(resource_results)
            errors.extend(resource_errors)
        except Exception as exc:
            errors.append({"resource": resource, "error": str(exc)})
    return {
        "schema": "tensorcore.mesh_resource_scheduler.result.v1",
        "ok": not errors and all(row.get("ok", True) for row in results),
        "checked_at_unix": time.time(),
        "dry_run": args.dry_run,
        "results": results,
        "errors": errors,
    }


def emit_text(payload: dict) -> None:
    if "iteration" in payload:
        print(f"== iteration {payload['iteration']} ==")
    for row in payload["results"]:
        print(
            f"{row.get('resource', '-')}: {row.get('action')} "
            f"job={row.get('job', '-')} ok={row.get('ok', True)}"
        )
    for row in payload["errors"]:
        print(f"ERROR {row}", file=sys.stderr)


def emit_json(payload: dict, *, pretty: bool = False) -> None:
    if pretty:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    else:
        json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_loop(args: argparse.Namespace) -> int:
    iteration = 0
    all_ok = True
    try:
        while True:
            iteration += 1
            try:
                payload = schedule_once(args)
            except Exception as exc:
                payload = {
                    "schema": "tensorcore.mesh_resource_scheduler.result.v1",
                    "ok": False,
                    "checked_at_unix": time.time(),
                    "dry_run": args.dry_run,
                    "results": [],
                    "errors": [{"error": str(exc)}],
                }
            payload["iteration"] = iteration
            all_ok = all_ok and bool(payload.get("ok"))
            if args.state_json:
                write_json(args.state_json, payload)
            if args.json or args.pretty_json:
                emit_json(payload, pretty=args.pretty_json)
            else:
                emit_text(payload)
            if args.max_iterations > 0 and iteration >= args.max_iterations:
                break
            time.sleep(max(0.0, args.interval_sec))
    except KeyboardInterrupt:
        return 130
    return 0 if all_ok else 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arbiter-cmd", default=DEFAULT_ARBITER_CMD)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--state-json")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--probe-timeout-sec", type=float, default=10.0)
    parser.add_argument("--start-timeout-sec", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_iterations < 0:
        parser.error("--max-iterations must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.loop:
        return run_loop(args)
    try:
        payload = schedule_once(args)
    except Exception as exc:
        payload = {
            "schema": "tensorcore.mesh_resource_scheduler.result.v1",
            "ok": False,
            "checked_at_unix": time.time(),
            "dry_run": args.dry_run,
            "results": [],
            "errors": [{"error": str(exc)}],
        }
    if args.state_json:
        write_json(args.state_json, payload)
    if args.json or args.pretty_json:
        emit_json(payload, pretty=args.pretty_json)
    else:
        emit_text(payload)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
