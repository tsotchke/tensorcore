#!/usr/bin/env python3
"""Validate mesh resource preflight evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any


SCHEMA = "tensorcore.mesh_resource_preflights.v1"


def fail(message: str) -> int:
    print(f"mesh resource preflight evidence invalid: {message}", file=sys.stderr)
    return 1


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"mesh resource preflight evidence invalid: could not read JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print("mesh resource preflight evidence invalid: evidence must be a JSON object", file=sys.stderr)
        return None
    return payload


def result_by_job(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in payload.get("results") or []:
        if isinstance(row, dict) and row.get("job"):
            rows[str(row["job"])] = row
    return rows


def parse_job_reason_specs(specs: list[str], *, flag: str) -> tuple[dict[str, str], str | None]:
    parsed: dict[str, str] = {}
    for spec in specs:
        job_id, sep, reason = spec.partition(":")
        if not sep or not job_id.strip() or not reason.strip():
            return {}, f"{flag} values must use job_id:reason"
        parsed[job_id.strip()] = reason.strip()
    return parsed, None


def validate_string_list(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key, [])
    if not isinstance(value, list):
        return f"{key} must be a list"
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return f"{key} must contain non-empty strings"
    return None


def validate_result_rows(rows: list[Any]) -> str | None:
    seen_jobs: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return f"result {index} must be an object"
        job = str(row.get("job") or "").strip()
        resource = str(row.get("resource") or "").strip()
        if not job:
            return f"result {index} job is required"
        if job in seen_jobs:
            return f"duplicate result job {job!r}"
        seen_jobs.add(job)
        if not resource:
            return f"result {index} resource is required"
        if not isinstance(row.get("ok"), bool):
            return f"result {index} ok must be boolean"
        if not str(row.get("reason") or "").strip():
            return f"result {index} reason is required"
        nested = row.get("json")
        if nested is not None and not isinstance(nested, dict):
            return f"result {index} json must be an object when present"
        if row.get("ok") is True and not isinstance(nested, dict):
            return f"result {index} json is required when ok=true"
        if row.get("ok") is True:
            try:
                rc = int(row.get("rc"))
            except (TypeError, ValueError):
                return f"result {index} rc must be 0 when ok=true"
            if rc != 0:
                return f"result {index} rc must be 0 when ok=true"
        if isinstance(nested, dict):
            if not str(nested.get("schema") or "").strip():
                return f"result {index} json.schema is required"
            nested_resource = nested.get("resource")
            if nested_resource is not None and nested_resource != resource:
                return f"result {index} json.resource must match result resource"
            if row.get("ok") is True and nested.get("ok") is not True:
                return f"result {index} json.ok must be true when result ok=true"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--max-age-sec", type=float, default=0.0)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-job", action="append", default=[])
    parser.add_argument("--require-skipped-default-job", action="append", default=[])
    parser.add_argument("--allow-failure", action="append", default=[], help="Require job_id to fail with reason")
    args = parser.parse_args(argv)

    payload = load_json(args.path)
    if payload is None:
        return 1
    if payload.get("schema") != SCHEMA:
        return fail(f"schema must be {SCHEMA!r}")
    if not isinstance(payload.get("results"), list):
        return fail("results must be a list")
    row_error = validate_result_rows(payload["results"])
    if row_error:
        return fail(row_error)
    try:
        jobs_checked = int(payload.get("jobs_checked"))
    except (TypeError, ValueError):
        return fail("jobs_checked must be numeric")
    if jobs_checked != len(payload["results"]):
        return fail("jobs_checked must match results length")
    for key in ("missing_job_ids", "skipped_default_job_ids"):
        list_error = validate_string_list(payload, key)
        if list_error:
            return fail(list_error)
    if payload.get("missing_job_ids"):
        return fail("missing_job_ids must be empty")
    if not isinstance(payload.get("ok"), bool):
        return fail("top-level ok must be boolean")
    expected_ok = all(row.get("ok") is True for row in payload["results"])
    if payload.get("ok") is not expected_ok:
        return fail("top-level ok must match result rows")
    if args.max_age_sec > 0:
        try:
            age = time.time() - float(payload.get("checked_at_unix") or 0.0)
        except (TypeError, ValueError):
            return fail("checked_at_unix must be numeric when --max-age-sec is used")
        if age > args.max_age_sec:
            return fail(f"evidence is stale: age_sec={age:.3f} max_age_sec={args.max_age_sec}")
    rows = result_by_job(payload)
    missing_jobs = [job_id for job_id in args.require_job if job_id not in rows]
    if missing_jobs:
        return fail(f"required jobs missing: {', '.join(missing_jobs)}")
    allowed_failures, allowed_error = parse_job_reason_specs(args.allow_failure, flag="--allow-failure")
    if allowed_error:
        return fail(allowed_error)
    missing_allowed = [job_id for job_id in allowed_failures if job_id not in rows]
    if missing_allowed:
        return fail(f"allowed failure jobs missing: {', '.join(missing_allowed)}")
    for job_id, expected_reason in allowed_failures.items():
        row = rows[job_id]
        actual_reason = str(row.get("reason") or "")
        if row.get("ok") is True:
            return fail(f"allowed failure job passed unexpectedly: {job_id}")
        if actual_reason != expected_reason:
            return fail(
                f"allowed failure reason mismatch for {job_id}: "
                f"expected {expected_reason!r}, got {actual_reason!r}"
            )
    skipped_default_jobs = set(payload.get("skipped_default_job_ids") or [])
    missing_skipped = [
        job_id for job_id in args.require_skipped_default_job
        if job_id not in skipped_default_jobs
    ]
    if missing_skipped:
        return fail(f"required skipped default jobs missing: {', '.join(missing_skipped)}")
    if args.require_pass:
        if jobs_checked <= 0:
            return fail("at least one preflight result is required when --require-pass is used")
        failed = [row for row in payload["results"] if not isinstance(row, dict) or row.get("ok") is not True]
        if failed:
            labels = [
                f"{row.get('job')}:{row.get('reason')}"
                for row in failed
                if isinstance(row, dict)
            ]
            return fail("preflights failed: " + ", ".join(labels))
        if payload.get("ok") is not True:
            return fail("top-level ok must be true when --require-pass is used")
    print(f"mesh resource preflight evidence OK: jobs_checked={jobs_checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
