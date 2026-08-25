#!/usr/bin/env python3
"""Emit JSON identity for a CUDA worker process on a mesh host."""

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


def run_capture(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def read_cgroup(path: Path = Path("/proc/self/cgroup")) -> str:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    if not rows:
        return ""
    last = rows[-1].split(":", 2)
    return last[-1] if last else ""


def systemd_unit_info(unit: str, *, timeout: float) -> dict[str, Any]:
    active = run_capture(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        timeout=timeout,
    )
    if active.returncode != 0:
        return {
            "ok": False,
            "unit": unit,
            "reason": "systemd_unit_inactive",
            "rc": active.returncode,
            "stderr_tail": active.stderr.strip()[-500:],
        }
    proc = run_capture(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=MainPID",
            "--property=SubState",
            "--property=Names",
            "--no-pager",
        ],
        timeout=timeout,
    )
    info: dict[str, Any] = {"ok": proc.returncode == 0, "unit": unit, "rc": proc.returncode}
    if proc.returncode != 0:
        info["stderr_tail"] = proc.stderr.strip()[-500:]
        return info
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "MainPID":
            try:
                info["main_pid"] = int(value)
            except ValueError:
                info["main_pid"] = 0
        elif key == "SubState":
            info["substate"] = value
        elif key == "Names":
            info["names"] = value
    return info


def cuda_processes(nvidia_smi: str, *, timeout: float) -> list[dict[str, Any]]:
    proc = run_capture(
        [
            nvidia_smi,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "nvidia-smi failed")
    processes: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 2:
            continue
        item: dict[str, Any] = {
            "pid": int(row[0].strip()),
            "process_name": row[1].strip(),
        }
        if len(row) >= 3:
            try:
                item["used_gpu_memory_mb"] = int(row[2].strip())
            except ValueError:
                item["used_gpu_memory_mb"] = row[2].strip()
        processes.append(item)
    return processes


def filter_processes(processes: list[dict[str, Any]], needle: str | None) -> list[dict[str, Any]]:
    if not needle:
        return processes
    lower = needle.lower()
    return [
        proc
        for proc in processes
        if lower in str(proc.get("process_name", "")).lower()
        or lower == str(proc.get("pid", ""))
    ]


def build_identity(args: argparse.Namespace) -> dict[str, Any]:
    unit_info: dict[str, Any] = {}
    if args.unit:
        unit_info = systemd_unit_info(args.unit, timeout=args.timeout_sec)
        if not unit_info.get("ok"):
            raise RuntimeError(json.dumps(unit_info, sort_keys=True))

    processes = cuda_processes(args.nvidia_smi, timeout=args.timeout_sec)
    matched = filter_processes(processes, args.process_substring)
    if args.require_cuda_process and not matched:
        raise RuntimeError("no matching CUDA process found")

    payload: dict[str, Any] = {
        "schema": "tensorcore.mesh_cuda_worker_identity.v1",
        "checked_at_unix": time.time(),
        "worker_host": socket.gethostname(),
        "worker_pid": os.getpid(),
        "worker_cgroup": read_cgroup(),
        "cuda_processes": matched,
        "cuda_pids": [proc["pid"] for proc in matched],
    }
    if args.unit:
        payload["worker_systemd_unit"] = args.unit
        payload["worker_systemd"] = unit_info
        if "main_pid" in unit_info:
            payload["worker_systemd_main_pid"] = unit_info["main_pid"]
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", help="systemd --user unit that should be active")
    parser.add_argument("--process-substring", help="CUDA process name substring or exact pid")
    parser.add_argument("--require-cuda-process", action="store_true")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be > 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        payload = build_identity(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
