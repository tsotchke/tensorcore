#!/usr/bin/env python3
"""Check whether a GeoRefine Qwen run is live."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


SCHEMA = "tensorcore.georefine_qwen_live.v1"


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def proc_cmdline(pid: int) -> str:
    try:
        raw = (pathlib.Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def status_live(status: dict[str, Any], run_dir: str) -> tuple[bool, list[int]]:
    if status.get("state") != "running":
        return False, []
    matched: list[int] = []
    for key in ("compressor_pid", "supervisor_pid"):
        pid = status.get(key)
        if not isinstance(pid, int) or pid <= 1:
            continue
        cmdline = proc_cmdline(pid)
        if run_dir in cmdline:
            matched.append(pid)
    return bool(matched), matched


def pgrep_matches(pattern: str, run_dir: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return []
    current_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1] if len(parts) > 1 else ""
        if pid == current_pid or "check_georefine_qwen_live.py" in cmdline:
            continue
        if run_dir and run_dir not in cmdline:
            continue
        matches.append({"pid": pid, "cmdline": cmdline})
    return matches


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = str(args.run_dir)
    status_path = pathlib.Path(args.status_file) if args.status_file else args.run_dir / "m2_supervisor_status.json"
    status = read_json(status_path)
    status_ok = False
    status_pids: list[int] = []
    if status is not None:
        status_ok, status_pids = status_live(status, run_dir)
    matches = [] if status_ok else pgrep_matches(args.match_regex, run_dir)
    ok = status_ok or bool(matches)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "reason": "ok" if ok else "not_live",
        "run_dir": run_dir,
        "status_path": str(status_path),
        "status_state": status.get("state") if isinstance(status, dict) else None,
        "status_pids": status_pids,
        "pgrep_matches": matches,
        "checked_at_unix": int(time.time()),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--status-file", default="")
    parser.add_argument("--match-regex", default="experiments.georefine.m2_compress")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{args.run_dir}: ok={payload.get('ok')} reason={payload.get('reason')}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
