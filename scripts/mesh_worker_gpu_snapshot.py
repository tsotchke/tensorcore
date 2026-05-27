#!/usr/bin/env python3
"""Emit local or SSH-polled GPU process state for scheduler reconciliation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "tensorcore.mesh_worker_gpu_snapshot.v1"


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def nvidia_smi_command(args: argparse.Namespace, nvidia_smi_args: list[str]) -> list[str]:
    if args.ssh_host:
        return ["ssh", args.ssh_host, args.nvidia_smi, *nvidia_smi_args]
    return [args.nvidia_smi, *nvidia_smi_args]


def parse_compute_apps(stdout: str) -> list[dict[str, Any]]:
    apps: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        parts = [part.strip() for part in row]
        raw = ", ".join(parts)
        if len(parts) < 3:
            apps.append({"raw": raw, "parse_error": "expected pid, process_name, used_gpu_memory"})
            continue
        pid_s, process_name, memory_s = parts[:3]
        try:
            pid = int(pid_s)
        except ValueError:
            pid = None
        try:
            used_memory_mib = int(memory_s)
        except ValueError:
            used_memory_mib = None
        apps.append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": used_memory_mib,
                "raw": raw,
            }
        )
    return apps


def parse_gpu_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        parts = [part.strip() for part in row]
        raw = ", ".join(parts)
        if len(parts) < 8:
            rows.append({"raw": raw, "parse_error": "expected gpu query columns"})
            continue
        index, uuid, pci_bus_id, name, total, used, free, util = parts[:8]
        def as_int(value: str) -> int | None:
            try:
                return int(value)
            except ValueError:
                return None
        rows.append(
            {
                "index": as_int(index),
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "name": name,
                "memory_total_mib": as_int(total),
                "memory_used_mib": as_int(used),
                "memory_free_mib": as_int(free),
                "utilization_gpu_pct": as_int(util),
                "raw": raw,
            }
        )
    return rows


def query_compute_apps(args: argparse.Namespace) -> dict[str, Any]:
    try:
        proc = run_capture(
            nvidia_smi_command(
                args,
                [
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
            ),
            timeout=args.timeout_sec,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "reason": "nvidia_smi_not_found", "error": str(exc), "apps": []}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "nvidia_smi_timeout", "apps": []}
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "reason": "ok" if proc.returncode == 0 else "nvidia_smi_failed",
        "stderr_tail": proc.stderr.strip()[-1000:] if proc.returncode != 0 else "",
        "apps": parse_compute_apps(proc.stdout) if proc.returncode == 0 else [],
    }


def query_gpus(args: argparse.Namespace) -> dict[str, Any]:
    try:
        proc = run_capture(
            nvidia_smi_command(
                args,
                [
                    "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
            ),
            timeout=args.timeout_sec,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "reason": "nvidia_smi_not_found", "error": str(exc), "gpus": []}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "nvidia_smi_timeout", "gpus": []}
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "reason": "ok" if proc.returncode == 0 else "nvidia_smi_failed",
        "stderr_tail": proc.stderr.strip()[-1000:] if proc.returncode != 0 else "",
        "gpus": parse_gpu_rows(proc.stdout) if proc.returncode == 0 else [],
    }


def proc_cmdline(pid: int) -> str:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def enrich_apps(apps: list[dict[str, Any]], *, include_cmdline: bool) -> list[dict[str, Any]]:
    enriched = []
    for app in apps:
        row = dict(app)
        pid = row.get("pid")
        if include_cmdline and isinstance(pid, int):
            row["cmdline"] = proc_cmdline(pid)
        enriched.append(row)
    return enriched


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    gpu_query = query_gpus(args)
    compute_query = query_compute_apps(args)
    apps = enrich_apps(compute_query.get("apps", []), include_cmdline=not args.ssh_host)
    ok = bool(gpu_query.get("ok")) and bool(compute_query.get("ok"))
    poller_host = socket.gethostname()
    worker_host = args.ssh_host or poller_host
    reasons = []
    if not gpu_query.get("ok"):
        reasons.append(str(gpu_query.get("reason") or "gpu_query_failed"))
    if not compute_query.get("ok"):
        reasons.append(str(compute_query.get("reason") or "compute_query_failed"))
    return {
        "schema": SCHEMA,
        "ok": ok,
        "reason": "ok" if ok else ",".join(reasons),
        "checked_at_unix": time.time(),
        "worker_host": worker_host,
        "poller_host": poller_host,
        "ssh_host": args.ssh_host,
        "resource": args.resource,
        "nvidia_smi": args.nvidia_smi,
        "gpus": gpu_query.get("gpus", []),
        "cuda_apps": apps,
        "cuda_pids": [app.get("pid") for app in apps if isinstance(app.get("pid"), int)],
        "compute_app_count": len(apps),
        "gpu_query": {k: v for k, v in gpu_query.items() if k != "gpus"},
        "compute_query": {k: v for k, v in compute_query.items() if k != "apps"},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", default="cuda")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{payload['resource']}: {payload['compute_app_count']} CUDA compute app(s)")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
