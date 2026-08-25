#!/usr/bin/env python3
"""Emit worker-host identity for a mesh-scheduled job."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_key_values(stdout: str) -> dict[str, str]:
    out = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def systemd_unit_status(unit: str, timeout: float) -> dict[str, Any]:
    try:
        proc = run_capture(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "MainPID",
                "-p",
                "ControlGroup",
            ],
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"unit": unit, "ok": False, "reason": "systemctl_not_found", "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"unit": unit, "ok": False, "reason": "systemctl_timeout"}
    payload: dict[str, Any] = {
        "unit": unit,
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
    }
    if proc.returncode == 0:
        payload.update(parse_key_values(proc.stdout))
        try:
            payload["MainPID"] = int(payload.get("MainPID", "0"))
        except ValueError:
            payload["MainPID"] = 0
    else:
        payload["stderr_tail"] = proc.stderr.strip()[-1000:]
    return payload


def process_rows(timeout: float) -> list[dict[str, Any]]:
    proc = run_capture(
        ["ps", "-eo", "pid=,ppid=,pgid=,sid=,stat=,etime=,args="],
        timeout=timeout,
    )
    rows = []
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, pgid, sid, stat, etime, args = parts
        try:
            rows.append({
                "pid": int(pid),
                "ppid": int(ppid),
                "pgid": int(pgid),
                "sid": int(sid),
                "stat": stat,
                "etime": etime,
                "args": args,
            })
        except ValueError:
            continue
    return rows


def parse_cuda_apps(stdout: str) -> list[dict[str, Any]]:
    apps = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        parts = [part.strip() for part in row]
        raw = ", ".join(parts)
        if len(parts) < 3:
            apps.append({"raw": raw, "parse_error": "expected pid, process_name, used_gpu_memory"})
            continue
        pid_s, name, memory_s = parts
        try:
            pid = int(pid_s)
        except ValueError:
            pid = None
        try:
            memory = int(memory_s)
        except ValueError:
            memory = None
        apps.append({
            "pid": pid,
            "process_name": name,
            "used_memory_mib": memory,
            "raw": raw,
        })
    return apps


def cuda_apps(nvidia_smi: str, timeout: float) -> dict[str, Any]:
    try:
        proc = run_capture(
            [
                nvidia_smi,
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "reason": "nvidia_smi_not_found", "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "nvidia_smi_timeout"}
    payload: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "apps": parse_cuda_apps(proc.stdout) if proc.returncode == 0 else [],
    }
    if proc.returncode != 0:
        payload["stderr_tail"] = proc.stderr.strip()[-1000:]
    return payload


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    patterns = [re.compile(pattern) for pattern in args.match_regex]
    self_pid = os.getpid()
    rows = process_rows(args.timeout_sec)
    matched = [
        row
        for row in rows
        if row["pid"] != self_pid
        and any(pattern.search(row["args"]) for pattern in patterns)
    ] if patterns else []
    unit = systemd_unit_status(args.unit, args.timeout_sec) if args.unit else None
    cuda = cuda_apps(args.nvidia_smi, args.timeout_sec)
    cuda_pids = [
        app.get("pid")
        for app in cuda.get("apps", [])
        if isinstance(app.get("pid"), int)
    ]
    matched_pids = [row["pid"] for row in matched]
    worker_candidate_pids = set(matched_pids)
    worker_pid = None
    if unit and isinstance(unit.get("MainPID"), int) and unit.get("MainPID", 0) > 0:
        worker_pid = unit["MainPID"]
        worker_candidate_pids.add(worker_pid)
    elif matched_pids:
        worker_pid = matched_pids[0]
    elif cuda_pids:
        worker_pid = cuda_pids[0]
    matched_cuda_pids = sorted(set(cuda_pids).intersection(worker_candidate_pids))
    ok = True
    reasons = []
    if args.require_active_unit:
        if not unit or unit.get("ActiveState") != "active":
            ok = False
            reasons.append("unit_not_active")
    if args.require_matching_process and not matched:
        ok = False
        reasons.append("no_matching_process")
    if args.require_cuda and not cuda_pids:
        ok = False
        reasons.append("no_cuda_process")
    if args.require_matched_cuda and not matched_cuda_pids:
        ok = False
        reasons.append("no_matched_cuda_process")
    return {
        "schema": "tensorcore.mesh_worker_identity.v1",
        "ok": ok,
        "reason": "ok" if ok else ",".join(reasons),
        "checked_at_unix": time.time(),
        "worker_host": socket.gethostname(),
        "resource": args.resource,
        "worker_pid": worker_pid,
        "worker_pids": matched_pids,
        "worker_systemd_unit": args.unit,
        "worker_cgroup": unit.get("ControlGroup") if unit else None,
        "systemd": unit,
        "cuda_pids": cuda_pids,
        "matched_cuda_pids": matched_cuda_pids,
        "cuda": cuda,
        "matched_processes": matched,
        "artifact_dir": args.artifact_dir,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", default="cuda")
    parser.add_argument("--unit")
    parser.add_argument("--match-regex", action="append", default=[])
    parser.add_argument("--artifact-dir")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--require-active-unit", action="store_true")
    parser.add_argument("--require-matching-process", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-matched-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
