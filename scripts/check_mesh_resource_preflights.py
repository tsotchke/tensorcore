#!/usr/bin/env python3
"""Run non-launching preflight commands from mesh_resource_jobs.json."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any


SCHEMA = "tensorcore.mesh_resource_preflights.v1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_JOBS = ROOT / "configs" / "mesh_resource_jobs.json"


def resolve_part(value: str, *, executable: bool = False) -> str:
    if executable and value.startswith("~"):
        return str(pathlib.Path(value).expanduser())
    if value.startswith(("scripts/", "configs/")):
        path = ROOT / value
        if path.exists():
            return str(path)
    return value


def command(value: Any) -> list[str]:
    if isinstance(value, list):
        return [resolve_part(str(part), executable=index == 0) for index, part in enumerate(value)]
    if isinstance(value, str):
        return [resolve_part(part, executable=index == 0) for index, part in enumerate(shlex.split(value))]
    return []


def load_jobs(path: pathlib.Path) -> list[dict[str, Any]]:
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    rows = raw.get("jobs") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("jobs file must contain a jobs list")
    return [row for row in rows if isinstance(row, dict)]


def parse_stdout_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def preflight_payload_failure(job: dict[str, Any], payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return "invalid_preflight_json"
    if not payload.get("schema"):
        return "invalid_preflight_schema"
    expected_resource = job.get("resource")
    if expected_resource and payload.get("resource") != expected_resource:
        return "preflight_resource_mismatch"
    if payload.get("ok") is not True:
        return str(payload.get("reason") or "preflight_failed")
    return None


def run_preflight(job: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    argv = command(job.get("preflight_cmd"))
    if not argv:
        return {
            "job": job.get("id"),
            "resource": job.get("resource"),
            "ok": False,
            "reason": "missing_preflight_cmd",
        }
    try:
        proc = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "job": job.get("id"),
            "resource": job.get("resource"),
            "ok": False,
            "reason": "preflight_timeout",
            "cmd": argv,
        }
    payload = parse_stdout_json(proc.stdout)
    payload_failure = preflight_payload_failure(job, payload)
    if payload_failure:
        reason = payload_failure
    elif proc.returncode != 0:
        reason = "preflight_nonzero_exit"
    elif isinstance(payload, dict) and payload.get("reason"):
        reason = str(payload.get("reason"))
    else:
        reason = "ok"
    return {
        "job": job.get("id"),
        "resource": job.get("resource"),
        "ok": proc.returncode == 0 and payload_failure is None,
        "reason": reason,
        "rc": proc.returncode,
        "cmd": argv,
        "json": payload,
        "stdout_tail": proc.stdout.strip()[-1000:],
        "stderr_tail": proc.stderr.strip()[-1000:],
    }


def preflight_default_enabled(job: dict[str, Any]) -> bool:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    return metadata.get("preflight_default", True) is not False


def select_jobs(jobs: list[dict[str, Any]], ids: list[str], include_running: bool) -> list[dict[str, Any]]:
    if ids:
        wanted = set(ids)
        return [job for job in jobs if job.get("id") in wanted]
    return [
        job for job in jobs
        if job.get("preflight_cmd") and (include_running or job.get("desired_state") == "paused")
        and preflight_default_enabled(job)
    ]


def skipped_default_job_ids(jobs: list[dict[str, Any]], ids: list[str], include_running: bool) -> list[str]:
    if ids:
        return []
    return [
        str(job.get("id"))
        for job in jobs
        if job.get("id")
        and job.get("preflight_cmd")
        and (include_running or job.get("desired_state") == "paused")
        and not preflight_default_enabled(job)
    ]


def missing_job_ids(jobs: list[dict[str, Any]], ids: list[str]) -> list[str]:
    existing = {str(job.get("id")) for job in jobs if job.get("id")}
    return [job_id for job_id in ids if job_id not in existing]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=pathlib.Path, default=DEFAULT_JOBS)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    all_jobs = load_jobs(args.jobs_json)
    missing = missing_job_ids(all_jobs, args.job_id)
    jobs = select_jobs(all_jobs, args.job_id, args.include_running)
    skipped_default = skipped_default_job_ids(all_jobs, args.job_id, args.include_running)
    results = [run_preflight(job, timeout=args.timeout_sec) for job in jobs]
    payload = {
        "schema": SCHEMA,
        "ok": not missing and all(result.get("ok") is True for result in results),
        "checked_at_unix": int(time.time()),
        "jobs_checked": len(results),
        "missing_job_ids": missing,
        "skipped_default_job_ids": skipped_default,
        "results": results,
    }
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"mesh resource preflights: ok={payload['ok']} jobs_checked={len(results)}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
