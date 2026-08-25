#!/usr/bin/env python3
"""Host-local CUDA admission gate for exclusive mesh jobs.

Run this on the worker host immediately before launching an exclusive CUDA
workload. It refuses admission when unmanaged CUDA compute applications are
present, unless an explicit allowlist pattern permits them.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import time
from typing import Any


def parse_compute_apps(stdout: str) -> list[dict[str, Any]]:
    apps = []
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        parts = [part.strip() for part in row]
        raw = ", ".join(parts)
        if len(parts) < 3:
            apps.append({"raw": raw, "parse_error": "expected pid, process_name, used_gpu_memory"})
            continue
        pid_s, process_name, memory_s = parts
        try:
            pid = int(pid_s)
        except ValueError:
            pid = None
        try:
            used_memory_mib = int(memory_s)
        except ValueError:
            used_memory_mib = None
        apps.append({
            "pid": pid,
            "process_name": process_name,
            "used_memory_mib": used_memory_mib,
            "raw": raw,
        })
    return apps


def query_compute_apps(nvidia_smi: str, timeout: float) -> tuple[int, str, str]:
    proc = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def classify_apps(
    apps: list[dict[str, Any]],
    *,
    allow_patterns: list[re.Pattern[str]],
    allowed_process_max_memory_mib: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = []
    blocked = []
    for app in apps:
        process_name = str(app.get("process_name") or app.get("raw") or "")
        used_memory = app.get("used_memory_mib")
        parse_ok = app.get("pid") is not None and isinstance(used_memory, int)
        pattern_ok = any(pattern.search(process_name) for pattern in allow_patterns)
        memory_ok = isinstance(used_memory, int) and used_memory <= allowed_process_max_memory_mib
        if parse_ok and pattern_ok and memory_ok:
            allowed.append(app)
        else:
            blocked.append(app)
    return allowed, blocked


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        rc, stdout, stderr = query_compute_apps(args.nvidia_smi, args.timeout_sec)
    except FileNotFoundError as exc:
        return {
            "schema": "tensorcore.cuda_resource_admission.v1",
            "ok": False,
            "resource": args.resource,
            "checked_at_unix": time.time(),
            "reason": "nvidia_smi_not_found",
            "error": str(exc),
        }
    except subprocess.TimeoutExpired:
        return {
            "schema": "tensorcore.cuda_resource_admission.v1",
            "ok": False,
            "resource": args.resource,
            "checked_at_unix": time.time(),
            "reason": "nvidia_smi_timeout",
        }
    if rc != 0:
        return {
            "schema": "tensorcore.cuda_resource_admission.v1",
            "ok": False,
            "resource": args.resource,
            "checked_at_unix": time.time(),
            "reason": "nvidia_smi_failed",
            "rc": rc,
            "stderr_tail": stderr.strip()[-1000:],
            "stdout_tail": stdout.strip()[-1000:],
        }
    patterns = [re.compile(pattern) for pattern in args.allow_process_regex]
    apps = parse_compute_apps(stdout)
    allowed, blocked = classify_apps(
        apps,
        allow_patterns=patterns,
        allowed_process_max_memory_mib=args.allowed_process_max_memory_mib,
    )
    return {
        "schema": "tensorcore.cuda_resource_admission.v1",
        "ok": not blocked,
        "resource": args.resource,
        "checked_at_unix": time.time(),
        "reason": "ok" if not blocked else "blocked_cuda_compute_apps",
        "allowed_process_max_memory_mib": args.allowed_process_max_memory_mib,
        "allowed": allowed,
        "blocked": blocked,
        "compute_app_count": len(apps),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", default="cuda")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--allow-process-regex",
        action="append",
        default=[],
        help="Allow matching process names only if they are below the memory cap.",
    )
    parser.add_argument("--allowed-process-max-memory-mib", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be > 0")
    if args.allowed_process_max_memory_mib < 0:
        parser.error("--allowed-process-max-memory-mib must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = build_payload(args)
    if args.json or not payload["ok"]:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif payload["ok"]:
        print(f"{payload['resource']}: CUDA admission ok")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
