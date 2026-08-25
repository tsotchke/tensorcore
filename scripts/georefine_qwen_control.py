#!/usr/bin/env python3
"""Tensorcore status and live-control surface for GeoRefine Qwen runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any


STATUS_SCHEMA = "tensorcore.georefine_qwen_control.status.v1"
SET_SCHEMA = "tensorcore.georefine_qwen_control.set.v1"
KNOBS_SCHEMA = "tensorcore.georefine_qwen_control.knobs.v1"

DEFAULT_RECONCILER_SCRIPT = (
    "/srv/tensorcore/.local/share/georefine_substrate_tools/20260526/scripts/"
    "georefine_live_control_reconciler.py"
)
DEFAULT_PYTHON_BIN = "/srv/tensorcore/work/venv/bin/python"
DEFAULT_ACTOR = "tensorcore-georefine-reconciler"
DEFAULT_SCHEDULER_STATE_JSON = "artifacts/mesh/georefine_chat_surgery_scheduler_state.json"

LIVE_PARAM_SPECS: dict[str, dict[str, Any]] = {
    "m2_target_kl": {
        "kind": "float",
        "minimum": 0.0,
        "maximum": None,
        "category": "m2_quality",
        "description": "Target calibration KL for M2 refinement.",
        "affects": ["quality", "rank_growth", "runtime"],
    },
    "m2_target_size_ratio": {
        "kind": "float",
        "minimum": 0.0,
        "maximum": 1.0,
        "category": "m2_size",
        "description": "Maximum projected artifact byte ratio.",
        "affects": ["size_gate", "runtime"],
    },
    "m2_target_kl_max_iterations": {
        "kind": "int",
        "minimum": 1,
        "maximum": 64,
        "category": "m2_refinement",
        "description": "Maximum target-KL refinement iterations.",
        "affects": ["quality", "runtime"],
    },
    "m2_target_kl_layers_per_iter": {
        "kind": "int",
        "minimum": 1,
        "maximum": 256,
        "category": "m2_refinement",
        "description": "Number of worst layers to rank-grow per iteration.",
        "affects": ["quality", "size_ratio", "runtime"],
    },
    "m2_target_kl_rank_growth_factor": {
        "kind": "float",
        "minimum": 1.000001,
        "maximum": 16.0,
        "category": "m2_refinement",
        "description": "Multiplicative rank growth for selected layers.",
        "affects": ["quality", "size_ratio"],
    },
    "m2_target_kl_kd_steps": {
        "kind": "int",
        "minimum": 0,
        "maximum": 100000,
        "category": "m2_kd",
        "description": "KD recovery steps for the current or next M2 KD block.",
        "affects": ["quality", "runtime"],
    },
    "m2_target_kl_kd_lr": {
        "kind": "float",
        "minimum": 1e-8,
        "maximum": 1.0,
        "category": "m2_kd",
        "description": "KD learning rate for the current or next M2 KD block.",
        "affects": ["quality", "stability"],
    },
    "m2_target_kl_kd_temperature": {
        "kind": "float",
        "minimum": 1e-6,
        "maximum": 100.0,
        "category": "m2_kd",
        "description": "KD softmax temperature.",
        "affects": ["quality", "stability"],
    },
    "m2_target_kl_kd_chunk_size": {
        "kind": "int",
        "minimum": 1,
        "maximum": 4096,
        "category": "m2_kd",
        "description": "KD chunk size for memory/runtime tradeoffs.",
        "affects": ["memory", "runtime"],
    },
    "m2_target_kl_kd_hidden_state_weight": {
        "kind": "float",
        "minimum": 0.0,
        "maximum": 10.0,
        "category": "m2_kd",
        "description": "Auxiliary hidden-state matching weight.",
        "affects": ["quality", "stability"],
    },
    "m2_target_kl_post_kd_refit": {
        "kind": "bool",
        "category": "m2_refinement",
        "description": "Run the post-KD CC-ASVD folding/refit step after KD blocks.",
        "affects": ["quality", "folding", "runtime"],
    },
    "m2_target_kl_kd_only": {
        "kind": "bool",
        "category": "m2_refinement",
        "description": "Skip rank regrowth and use KD-only recovery.",
        "affects": ["quality", "rank_growth", "size_ratio", "runtime"],
    },
    "m2_abort_on_size_gate": {
        "kind": "bool",
        "category": "m2_gates",
        "description": "Abort before expensive work when size gate fails.",
        "affects": ["runtime", "artifact_writes"],
    },
    "m2_abort_on_quality_gate": {
        "kind": "bool",
        "category": "m2_gates",
        "description": "Abort before storage writes when quality gate fails.",
        "affects": ["runtime", "artifact_writes"],
    },
    "m2_pause": {
        "kind": "bool",
        "category": "m2_control",
        "description": "Pause at the next safe M2 checkpoint.",
        "affects": ["runtime"],
    },
    "m2_abort": {
        "kind": "bool",
        "category": "m2_control",
        "description": "Abort cleanly at the next safe M2 checkpoint.",
        "affects": ["runtime", "artifact_writes"],
    },
}

REMOTE_STATUS_CODE = r'''
import csv, io, json, math, pathlib, re, subprocess, sys, time
run = pathlib.Path(sys.argv[1])
max_age = float(sys.argv[2])
KD_RE = re.compile(r"\[M2/KD\] step (?P<step>\d+)/(?P<total>\d+): loss=(?P<loss>\S+)\s+chunk_kl=(?P<chunk_kl>\S+)\s+lr=(?P<lr>\S+).*?ETA (?P<eta>[^)]+)\)")
CHAT_RE = re.compile(r"\[M2/chat\] (?P<phase>[^:]+): passed=(?P<passed>True|False) blockers=(?P<blockers>\[.*\])")
def rj(name):
    try:
        data = json.loads((run / name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) else None
def kv(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = int(v) if k == "MainPID" and v.isdigit() else v
    return out
def cap(argv, timeout=5):
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
def systemd(unit):
    if not unit:
        return None
    try:
        p = cap(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "ControlGroup", "-p", "Result"])
    except Exception:
        return None
    d = kv(p.stdout); d.update({"unit": unit, "ok": p.returncode == 0, "rc": p.returncode})
    return d
def cuda(pid):
    try:
        p = cap(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    except Exception:
        return {"ok": False, "reason": "nvidia_smi_unavailable", "apps": [], "matched_compressor": []}
    apps = []
    if p.returncode == 0:
        for row in csv.reader(io.StringIO(p.stdout)):
            if len(row) < 3:
                continue
            ps, name, mem = [x.strip() for x in row[:3]]
            try: ipid = int(ps)
            except Exception: ipid = None
            try: imem = int(mem)
            except Exception: imem = None
            apps.append({"pid": ipid, "process_name": name, "used_memory_mib": imem})
    return {"ok": p.returncode == 0, "rc": p.returncode, "apps": apps, "matched_compressor": [a for a in apps if pid is not None and a.get("pid") == pid]}
def progress():
    try:
        lines = (run / "m2_supervisor_child.log").read_text(encoding="utf-8", errors="replace").splitlines()[-1200:]
    except Exception:
        lines = []
    kd = chat = None
    for line in lines:
        m = KD_RE.search(line)
        if m:
            kd = {"step": int(m.group("step")), "total": int(m.group("total")), "loss": float(m.group("loss")), "chunk_kl": float(m.group("chunk_kl")), "lr": float(m.group("lr")), "eta": m.group("eta").strip(), "line": line}
            kd["fraction"] = kd["step"] / kd["total"] if kd["total"] else None
        c = CHAT_RE.search(line)
        if c:
            chat = {"phase": c.group("phase"), "passed": c.group("passed") == "True", "blockers": c.group("blockers"), "line": line}
    return {"latest_kd": kd, "latest_chat": chat}
intent = rj("run_intent.json")
sup = rj("m2_supervisor_status.json")
rec = rj("georefine_live_control_reconciler_state.json")
ctrl = rj("m2_live_control.json")
cert = rj("m2_certificate.json")
lease = str(intent.get("lease_id") or "")
unit = f"tensorcore-georefine-{lease}.service" if lease else ""
comp = sup.get("compressor_pid") if isinstance(sup.get("compressor_pid"), int) else None
sd = systemd(unit)
cu = cuda(comp)
last = rec.get("last_result") if isinstance(rec.get("last_result"), dict) else {}
updated = num(sup.get("updated_at"))
age = None if updated is None else max(0.0, time.time() - updated)
health = {"run_state": sup.get("state"), "heartbeat_age_seconds": age, "lease_id": lease or None, "lease_verification": last.get("lease_verification"), "systemd_active": (sd.get("ActiveState") == "active" if isinstance(sd, dict) else None), "cuda_matched": bool(cu.get("matched_compressor")), "reconciler_ok": last.get("ok")}
ok = sup.get("state") == "running" and (age is None or age < max_age) and bool(cu.get("matched_compressor")) and health["systemd_active"] is not False and health["reconciler_ok"] is not False
payload = {"schema": "tensorcore.georefine_qwen_control.status.v1", "ok": ok, "reason": "ok" if ok else "not_healthy", "run_dir": str(run), "resource": intent.get("resource"), "target": intent.get("target"), "health": health, "progress": progress(), "completion": {"certificate_present": bool(cert), "completed": cert.get("completed") is True if cert else False, "current_stage": cert.get("current_stage") if cert else None, "size_ratio": (cert.get("size_ratio") or cert.get("final_size_ratio")) if cert else None, "quality_gate": cert.get("quality_gate") if cert else None}, "adjustable": {"current_control": {"set_params": ctrl.get("set_params"), "commands": ctrl.get("commands"), "proposal_reason": ctrl.get("proposal_reason"), "proposal_digest": ctrl.get("proposal_digest"), "writer_actor": ctrl.get("writer_actor"), "expires_at": ctrl.get("expires_at")}, "proposal_dir": str(run / "proposals"), "control_path": str(run / "m2_live_control.json")}, "pids": {"supervisor": sup.get("supervisor_pid"), "compressor": comp, "live_agent": sup.get("live_agent_pid")}, "systemd": sd, "cuda": cu}
print(json.dumps(payload, sort_keys=True))
sys.exit(0 if ok else 1)
'''

KD_RE = re.compile(
    r"\[M2/KD\] step (?P<step>\d+)/(?P<total>\d+): "
    r"loss=(?P<loss>\S+)\s+chunk_kl=(?P<chunk_kl>\S+)\s+"
    r"lr=(?P<lr>\S+).*?ETA (?P<eta>[^)]+)\)"
)
CHAT_RE = re.compile(
    r"\[M2/chat\] (?P<phase>[^:]+): passed=(?P<passed>True|False) "
    r"blockers=(?P<blockers>\[.*\])"
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def nested_dict(root: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = root
    for key in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def discover_active_run(state_json: Path, *, job_id: str = "") -> dict[str, Any] | None:
    state = read_json(state_json.expanduser())
    if not state:
        return None
    rows = state.get("results")
    if not isinstance(rows, list):
        return None
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if job_id and str(row.get("job") or "") != job_id:
            continue
        identity = nested_dict(row, ("worker_identity", "identity"))
        run_dir = identity.get("artifact_dir")
        if not isinstance(run_dir, str) or not run_dir:
            continue
        resource = str(row.get("resource") or identity.get("resource") or "")
        worker_host = identity.get("worker_host")
        if not isinstance(worker_host, str) or not worker_host:
            worker_host = resource.split(":", 1)[0] if ":" in resource else ""
        candidates.append(
            {
                "run_dir": run_dir,
                "target": worker_host,
                "resource": resource or None,
                "job": row.get("job"),
                "lease_id": row.get("lease_id"),
                "source": str(state_json),
                "action": row.get("action"),
                "ok": row.get("ok"),
            }
        )
    healthy = [
        row
        for row in candidates
        if row.get("ok") is True
        and str(row.get("action") or "").startswith(("heartbeated_", "started_"))
    ]
    if healthy:
        return healthy[0]
    return candidates[0] if candidates else None


def resolve_run_context(args: argparse.Namespace) -> None:
    if getattr(args, "cmd", "") not in {"status", "set"}:
        return
    if getattr(args, "run_dir", ""):
        return
    state_json = Path(getattr(args, "state_json", DEFAULT_SCHEDULER_STATE_JSON))
    discovered = discover_active_run(state_json, job_id=getattr(args, "job_id", ""))
    if discovered is None:
        raise ValueError(
            "no active GeoRefine Qwen run discovered; pass --run-dir or refresh "
            f"{state_json}"
        )
    args.run_dir = discovered["run_dir"]
    args.discovered_run = discovered
    if not getattr(args, "target", "") and discovered.get("target"):
        args.target = str(discovered["target"])


def tail_lines(path: Path, limit: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def latest_progress_from_lines(lines: list[str]) -> dict[str, Any]:
    latest_kd: dict[str, Any] | None = None
    latest_chat: dict[str, Any] | None = None
    for line in lines:
        kd = KD_RE.search(line)
        if kd:
            latest_kd = {
                "step": int(kd.group("step")),
                "total": int(kd.group("total")),
                "loss": parse_float(kd.group("loss")),
                "chunk_kl": parse_float(kd.group("chunk_kl")),
                "lr": parse_float(kd.group("lr")),
                "eta": kd.group("eta").strip(),
                "line": line,
            }
            if latest_kd["total"] > 0:
                latest_kd["fraction"] = latest_kd["step"] / latest_kd["total"]
        chat = CHAT_RE.search(line)
        if chat:
            try:
                blockers = json.loads(chat.group("blockers").replace("'", '"'))
            except json.JSONDecodeError:
                blockers = chat.group("blockers")
            latest_chat = {
                "phase": chat.group("phase"),
                "passed": chat.group("passed") == "True",
                "blockers": blockers,
                "line": line,
            }
    return {
        "latest_kd": latest_kd,
        "latest_chat": latest_chat,
    }


def latest_progress(log_path: Path) -> dict[str, Any]:
    return latest_progress_from_lines(tail_lines(log_path, 1200))


def parse_key_values(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "MainPID":
            try:
                out[key] = int(value)
                continue
            except ValueError:
                pass
        out[key] = value
    return out


def run_capture(argv: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def systemd_status(unit: str) -> dict[str, Any] | None:
    if not unit:
        return None
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
                "-p",
                "Result",
            ],
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    payload = parse_key_values(proc.stdout)
    payload.update({"unit": unit, "ok": proc.returncode == 0, "rc": proc.returncode})
    if proc.returncode != 0:
        payload["stderr_tail"] = proc.stderr.strip()[-1000:]
    return payload


def parse_cuda_apps(stdout: str) -> list[dict[str, Any]]:
    apps: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(stdout)):
        if len(row) < 3:
            continue
        pid_s, name, memory_s = [part.strip() for part in row[:3]]
        try:
            pid = int(pid_s)
        except ValueError:
            pid = None
        try:
            memory = int(memory_s)
        except ValueError:
            memory = None
        apps.append(
            {
                "pid": pid,
                "process_name": name,
                "used_memory_mib": memory,
            }
        )
    return apps


def cuda_status(compressor_pid: int | None) -> dict[str, Any]:
    try:
        proc = run_capture(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"ok": False, "reason": "nvidia_smi_unavailable"}
    apps = parse_cuda_apps(proc.stdout) if proc.returncode == 0 else []
    matched = [
        app
        for app in apps
        if compressor_pid is not None and app.get("pid") == compressor_pid
    ]
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "apps": apps,
        "matched_compressor": matched,
    }


def completion_summary(run_dir: Path) -> dict[str, Any]:
    cert = read_json(run_dir / "m2_certificate.json")
    if not cert:
        return {"completed": False, "certificate_present": False}
    quality_gate = cert.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_summary: dict[str, Any] | None = {
            "passed": quality_gate.get("passed"),
            "blockers": quality_gate.get("blockers"),
            "selected_quality_signal": quality_gate.get("selected_quality_signal"),
        }
    else:
        quality_summary = None
    return {
        "certificate_present": True,
        "completed": cert.get("completed") is True,
        "current_stage": cert.get("current_stage"),
        "size_ratio": cert.get("size_ratio") or cert.get("final_size_ratio"),
        "quality_gate": quality_summary,
        "target_kl": (
            cert.get("m2_target_kl_achievement", {}).get("post_storage_kl_mean")
            if isinstance(cert.get("m2_target_kl_achievement"), dict)
            else None
        ),
    }


def status_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser()
    intent = read_json(run_dir / "run_intent.json") or {}
    supervisor = read_json(run_dir / "m2_supervisor_status.json") or {}
    reconciler = read_json(run_dir / "georefine_live_control_reconciler_state.json") or {}
    control = read_json(run_dir / "m2_live_control.json") or {}
    lease_id = str(intent.get("lease_id") or "")
    unit = f"tensorcore-georefine-{lease_id}.service" if lease_id else ""
    compressor_pid = supervisor.get("compressor_pid")
    compressor_pid = compressor_pid if isinstance(compressor_pid, int) else None
    systemd = systemd_status(unit)
    cuda = cuda_status(compressor_pid)
    progress = latest_progress(run_dir / "m2_supervisor_child.log")
    now = time.time()
    updated_at = number(supervisor.get("updated_at"))
    heartbeat_age = None if updated_at is None else max(0.0, now - updated_at)
    reconciler_result = reconciler.get("last_result")
    if not isinstance(reconciler_result, dict):
        reconciler_result = {}
    health = {
        "run_state": supervisor.get("state"),
        "heartbeat_age_seconds": heartbeat_age,
        "lease_id": lease_id or None,
        "lease_verification": reconciler_result.get("lease_verification"),
        "systemd_active": (
            systemd.get("ActiveState") == "active" if isinstance(systemd, dict) else None
        ),
        "cuda_matched": bool(cuda.get("matched_compressor")),
        "reconciler_ok": reconciler_result.get("ok"),
    }
    ok = (
        supervisor.get("state") == "running"
        and (heartbeat_age is None or heartbeat_age < args.max_heartbeat_age_sec)
        and bool(cuda.get("matched_compressor"))
        and (health["systemd_active"] is not False)
        and (health["reconciler_ok"] is not False)
    )
    return {
        "schema": STATUS_SCHEMA,
        "ok": ok,
        "reason": "ok" if ok else "not_healthy",
        "run_dir": str(run_dir),
        "resource": intent.get("resource"),
        "target": intent.get("target"),
        "health": health,
        "progress": progress,
        "completion": completion_summary(run_dir),
        "adjustable": {
            "knobs": LIVE_PARAM_SPECS,
            "current_control": {
                "set_params": control.get("set_params"),
                "commands": control.get("commands"),
                "proposal_reason": control.get("proposal_reason"),
                "proposal_digest": control.get("proposal_digest"),
                "writer_actor": control.get("writer_actor"),
                "expires_at": control.get("expires_at"),
            },
            "proposal_dir": str(run_dir / "proposals"),
            "control_path": str(run_dir / "m2_live_control.json"),
        },
        "pids": {
            "supervisor": supervisor.get("supervisor_pid"),
            "compressor": compressor_pid,
            "live_agent": supervisor.get("live_agent_pid"),
            "reconciler": (
                intent.get("reconciler_sidecar", {}).get("pid")
                if isinstance(intent.get("reconciler_sidecar"), dict)
                else None
            ),
        },
        "systemd": systemd,
        "cuda": cuda,
    }


def parse_bool(value: str) -> bool:
    lower = value.strip().lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean, got {value!r}")


def parse_param_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected name=value, got {raw!r}")
    name, text = raw.split("=", 1)
    name = name.strip()
    if name not in LIVE_PARAM_SPECS:
        raise ValueError(f"unknown live-control parameter {name!r}")
    spec = LIVE_PARAM_SPECS[name]
    kind = spec["kind"]
    try:
        if kind == "bool":
            value: Any = parse_bool(text)
        elif kind == "int":
            value = int(text)
        elif kind == "float":
            value = float(text)
        else:
            value = text
    except ValueError as exc:
        raise ValueError(f"invalid value for {name}: {exc}") from exc
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}")
    return name, value


def set_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser()
    params, commands, proposal_json = build_adjustment(args)
    propose_cmd = [
        args.python_bin,
        args.reconciler_script,
        "propose",
        "--artifact-dir",
        str(run_dir),
        "--actor",
        args.actor,
        "--reason",
        args.reason,
        "--proposal-json",
        json.dumps(proposal_json, sort_keys=True),
    ]
    return run_set_commands(args, run_dir, params, commands, propose_cmd)


def build_adjustment(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    params: dict[str, Any] = {}
    for raw in args.set or []:
        name, value = parse_param_assignment(raw)
        params[name] = value
    commands: dict[str, bool] = {}
    if args.pause:
        commands["pause"] = True
    if args.resume:
        commands["pause"] = False
    if args.abort:
        commands["abort"] = True
    if not params and not commands:
        raise ValueError("no adjustment requested; pass --set, --pause, --resume, or --abort")
    now = time.time()
    proposal_json: dict[str, Any] = {
        "set_params": params,
        "agent_decision": {
            "action": "tensorcore_live_adjustment",
            "timestamp": now,
            "rationale": [args.reason],
        },
        "expires_at": now + args.expires_sec,
    }
    if commands:
        proposal_json["commands"] = commands
    return params, commands, proposal_json


def run_set_commands(
    args: argparse.Namespace,
    run_dir: Path,
    params: dict[str, Any],
    commands: dict[str, bool],
    propose_cmd: list[str],
    *,
    target: str = "",
) -> dict[str, Any]:
    def capture(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if target:
            return run_capture(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, shlex.join(argv)],
                timeout=args.timeout_sec,
            )
        return run_capture(argv, timeout=args.timeout_sec)

    propose = capture(propose_cmd)
    payload: dict[str, Any] = {
        "schema": SET_SCHEMA,
        "ok": propose.returncode == 0,
        "proposal": {
            "rc": propose.returncode,
            "stdout": propose.stdout.strip(),
            "stderr": propose.stderr.strip(),
        },
        "set_params": params,
        "commands": commands,
    }
    if propose.returncode == 0:
        try:
            payload["proposal"]["json"] = json.loads(propose.stdout)
        except json.JSONDecodeError:
            pass
    if propose.returncode != 0 or not args.apply_now:
        return payload
    reconcile_cmd = [
        args.python_bin,
        args.reconciler_script,
        "reconcile",
        "--artifact-dir",
        str(run_dir),
        "--reconciler-actor",
        DEFAULT_ACTOR,
        "--require-active-lease",
        "--arbiter-status-json",
        str(run_dir / "tensorcore_harness_arbiter_status.json"),
    ]
    reconcile = capture(reconcile_cmd)
    payload["reconcile"] = {
        "rc": reconcile.returncode,
        "stdout": reconcile.stdout.strip(),
        "stderr": reconcile.stderr.strip(),
    }
    if reconcile.returncode == 0:
        try:
            payload["reconcile"]["json"] = json.loads(reconcile.stdout)
        except json.JSONDecodeError:
            pass
    payload["ok"] = payload["ok"] and reconcile.returncode == 0
    return payload


def split_sections(stdout: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in stdout.splitlines():
        if line.startswith("__TC_SECTION_") and line.endswith("__"):
            current = line.removeprefix("__TC_SECTION_").removesuffix("__").lower()
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def json_section(sections: dict[str, str], key: str) -> dict[str, Any]:
    raw = sections.get(key, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def remote_status_payload(target: str, args: argparse.Namespace) -> tuple[int, dict[str, Any] | None, str]:
    run_q = shlex.quote(args.run_dir)
    remote_cmd = (
        f"run={run_q}; "
        "lease=$(python3 -c 'import json,sys; "
        "print(json.load(open(sys.argv[1])).get(\"lease_id\", \"\"))' "
        "\"$run/run_intent.json\" 2>/dev/null || true); "
        "unit=\"tensorcore-georefine-${lease}.service\"; "
        "section(){ printf '\\n__TC_SECTION_%s__\\n' \"$1\"; }; "
        "section intent; cat \"$run/run_intent.json\" 2>/dev/null || true; "
        "section supervisor; cat \"$run/m2_supervisor_status.json\" 2>/dev/null || true; "
        "section reconciler; cat \"$run/georefine_live_control_reconciler_state.json\" 2>/dev/null || true; "
        "section control; cat \"$run/m2_live_control.json\" 2>/dev/null || true; "
        "section certificate; cat \"$run/m2_certificate.json\" 2>/dev/null || true; "
        "section log; tail -1200 \"$run/m2_supervisor_child.log\" 2>/dev/null || true; "
        "section systemd; systemctl --user show \"$unit\" -p ActiveState -p SubState -p MainPID -p ControlGroup -p Result 2>/dev/null || true; "
        "section cuda; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true"
    )
    proc = run_capture(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, remote_cmd],
        timeout=120.0,
    )
    if proc.returncode != 0:
        return proc.returncode, None, proc.stderr or proc.stdout
    sections = split_sections(proc.stdout)
    intent = json_section(sections, "intent")
    supervisor = json_section(sections, "supervisor")
    reconciler = json_section(sections, "reconciler")
    control = json_section(sections, "control")
    cert = json_section(sections, "certificate")
    systemd = parse_key_values(sections.get("systemd", ""))
    if systemd:
        systemd.update({"unit": f"tensorcore-georefine-{intent.get('lease_id')}.service", "ok": True, "rc": 0})
    compressor_pid = supervisor.get("compressor_pid")
    compressor_pid = compressor_pid if isinstance(compressor_pid, int) else None
    cuda = {
        "ok": bool(sections.get("cuda", "")),
        "apps": parse_cuda_apps(sections.get("cuda", "")),
    }
    cuda["matched_compressor"] = [
        app for app in cuda["apps"] if compressor_pid is not None and app.get("pid") == compressor_pid
    ]
    last = reconciler.get("last_result")
    if not isinstance(last, dict):
        last = {}
    now = time.time()
    updated_at = number(supervisor.get("updated_at"))
    heartbeat_age = None if updated_at is None else max(0.0, now - updated_at)
    health = {
        "run_state": supervisor.get("state"),
        "heartbeat_age_seconds": heartbeat_age,
        "lease_id": intent.get("lease_id"),
        "lease_verification": last.get("lease_verification"),
        "systemd_active": systemd.get("ActiveState") == "active" if systemd else None,
        "cuda_matched": bool(cuda["matched_compressor"]),
        "reconciler_ok": last.get("ok"),
    }
    ok = (
        supervisor.get("state") == "running"
        and (heartbeat_age is None or heartbeat_age < args.max_heartbeat_age_sec)
        and bool(cuda["matched_compressor"])
        and health["systemd_active"] is not False
        and health["reconciler_ok"] is not False
    )
    payload = {
        "schema": STATUS_SCHEMA,
        "ok": ok,
        "reason": "ok" if ok else "not_healthy",
        "run_dir": args.run_dir,
        "resource": intent.get("resource"),
        "target": intent.get("target"),
        "health": health,
        "progress": latest_progress_from_lines(sections.get("log", "").splitlines()),
        "completion": {
            "certificate_present": bool(cert),
            "completed": cert.get("completed") is True if cert else False,
            "current_stage": cert.get("current_stage") if cert else None,
            "size_ratio": (cert.get("size_ratio") or cert.get("final_size_ratio")) if cert else None,
            "quality_gate": (
                {
                    "passed": cert.get("quality_gate", {}).get("passed"),
                    "blockers": cert.get("quality_gate", {}).get("blockers"),
                    "selected_quality_signal": cert.get("quality_gate", {}).get("selected_quality_signal"),
                }
                if isinstance(cert.get("quality_gate") if cert else None, dict)
                else None
            ),
        },
        "adjustable": {
            "knobs": LIVE_PARAM_SPECS,
            "current_control": {
                "set_params": control.get("set_params"),
                "commands": control.get("commands"),
                "proposal_reason": control.get("proposal_reason"),
                "proposal_digest": control.get("proposal_digest"),
                "writer_actor": control.get("writer_actor"),
                "expires_at": control.get("expires_at"),
            },
            "proposal_dir": str(Path(args.run_dir) / "proposals"),
            "control_path": str(Path(args.run_dir) / "m2_live_control.json"),
        },
        "pids": {
            "supervisor": supervisor.get("supervisor_pid"),
            "compressor": compressor_pid,
            "live_agent": supervisor.get("live_agent_pid"),
        },
        "systemd": systemd or None,
        "cuda": cuda,
    }
    return 0 if ok else 1, payload, proc.stderr


def remote_set_payload(target: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    params, commands, proposal_json = build_adjustment(args)
    propose_cmd = [
        args.python_bin,
        args.reconciler_script,
        "propose",
        "--artifact-dir",
        str(run_dir),
        "--actor",
        args.actor,
        "--reason",
        args.reason,
        "--proposal-json",
        json.dumps(proposal_json, sort_keys=True),
    ]
    return run_set_commands(
        args,
        run_dir,
        params,
        commands,
        propose_cmd,
        target=target,
    )


def remote_self(target: str, argv: list[str]) -> int:
    src = Path(__file__).resolve()
    remote_path = f"/tmp/tensorcore-georefine-qwen-control-{os.getpid()}-{uuid.uuid4().hex}.py"
    remote_argv = [part for part in argv if part != "--local"]
    upload = run_capture(["scp", "-q", str(src), f"{target}:{remote_path}"], timeout=30.0)
    if upload.returncode != 0:
        bootstrap = (
            "import json, os, runpy, sys; "
            "argv=json.loads(sys.argv[1]); "
            "path=f'/tmp/tensorcore-georefine-qwen-control-{os.getpid()}.py'; "
            "open(path, 'w', encoding='utf-8').write(sys.stdin.read()); "
            "sys.argv=[path]+argv; "
            "\ntry:\n runpy.run_path(path, run_name='__main__')\n"
            "finally:\n"
            "  try: os.unlink(path)\n"
            "  except OSError: pass\n"
        )
        proc = subprocess.run(
            ["ssh", target, "python3", "-c", bootstrap, json.dumps(["--local", *remote_argv])],
            input=src.read_text(encoding="utf-8"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120.0,
            check=False,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    try:
        cmd = ["ssh", target, "python3", remote_path, "--local", *remote_argv]
        proc = run_capture(cmd, timeout=120.0)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    finally:
        run_capture(["ssh", target, "rm", "-f", remote_path], timeout=10.0)


def print_human_status(payload: dict[str, Any]) -> None:
    health = payload.get("health", {})
    progress = payload.get("progress", {})
    latest_kd = progress.get("latest_kd") if isinstance(progress, dict) else None
    print(f"ok={payload.get('ok')} reason={payload.get('reason')}")
    print(
        "run="
        f"{health.get('run_state')} lease={health.get('lease_id')} "
        f"systemd_active={health.get('systemd_active')} "
        f"cuda_matched={health.get('cuda_matched')}"
    )
    if isinstance(latest_kd, dict):
        print(
            "kd="
            f"{latest_kd.get('step')}/{latest_kd.get('total')} "
            f"chunk_kl={latest_kd.get('chunk_kl')} eta={latest_kd.get('eta')}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="")
    parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="show scheduler-facing run health and knobs")
    status.add_argument("--run-dir")
    status.add_argument("--state-json", default=os.environ.get("TC_GEOREFINE_SCHEDULER_STATE_JSON", DEFAULT_SCHEDULER_STATE_JSON))
    status.add_argument("--job-id", default="")
    status.add_argument("--max-heartbeat-age-sec", type=float, default=120.0)
    status.add_argument("--json", action="store_true")

    set_cmd = sub.add_parser("set", help="write a safe live-control proposal")
    set_cmd.add_argument("--run-dir")
    set_cmd.add_argument("--state-json", default=os.environ.get("TC_GEOREFINE_SCHEDULER_STATE_JSON", DEFAULT_SCHEDULER_STATE_JSON))
    set_cmd.add_argument("--job-id", default="")
    set_cmd.add_argument("--actor", default=os.environ.get("TC_GEOREFINE_CONTROL_ACTOR", DEFAULT_ACTOR))
    set_cmd.add_argument("--reason", required=True)
    set_cmd.add_argument("--set", action="append", default=[])
    set_cmd.add_argument("--pause", action="store_true")
    set_cmd.add_argument("--resume", action="store_true")
    set_cmd.add_argument("--abort", action="store_true")
    set_cmd.add_argument("--expires-sec", type=float, default=600.0)
    set_cmd.add_argument("--apply-now", action=argparse.BooleanOptionalAction, default=True)
    set_cmd.add_argument("--python-bin", default=os.environ.get("TC_GEOREFINE_PYTHON_BIN", DEFAULT_PYTHON_BIN))
    set_cmd.add_argument("--reconciler-script", default=os.environ.get("TC_GEOREFINE_RECONCILER_SCRIPT", DEFAULT_RECONCILER_SCRIPT))
    set_cmd.add_argument("--timeout-sec", type=float, default=20.0)
    set_cmd.add_argument("--json", action="store_true")

    sub.add_parser("knobs", help="list adjustable live-control parameters")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv or sys.argv[1:])
    args = parse_args(raw_argv)
    try:
        resolve_run_context(args)
        if args.target and not args.local and args.cmd != "knobs":
            if args.cmd == "status":
                rc, payload, error = remote_status_payload(args.target, args)
                if payload is None:
                    if args.json:
                        json.dump({"ok": False, "error": error}, sys.stdout, sort_keys=True)
                        sys.stdout.write("\n")
                    else:
                        print(error, file=sys.stderr)
                    return rc or 1
                if args.json:
                    json.dump(payload, sys.stdout, sort_keys=True)
                    sys.stdout.write("\n")
                else:
                    print_human_status(payload)
                return 0 if payload.get("ok") else 1
            if args.cmd == "set":
                payload = remote_set_payload(args.target, args)
                if args.json:
                    json.dump(payload, sys.stdout, sort_keys=True)
                    sys.stdout.write("\n")
                else:
                    print(f"ok={payload.get('ok')} set={payload.get('set_params')} commands={payload.get('commands')}")
                return 0 if payload.get("ok") else 1
        if args.cmd == "status":
            payload = status_payload(args)
            if args.json:
                json.dump(payload, sys.stdout, sort_keys=True)
                sys.stdout.write("\n")
            else:
                print_human_status(payload)
            return 0 if payload.get("ok") else 1
        if args.cmd == "set":
            payload = set_payload(args)
            if args.json:
                json.dump(payload, sys.stdout, sort_keys=True)
                sys.stdout.write("\n")
            else:
                print(f"ok={payload.get('ok')} set={payload.get('set_params')} commands={payload.get('commands')}")
            return 0 if payload.get("ok") else 1
        if args.cmd == "knobs":
            payload = {"schema": KNOBS_SCHEMA, "ok": True, "knobs": LIVE_PARAM_SPECS}
            json.dump(payload, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
    except (ValueError, subprocess.TimeoutExpired) as exc:
        if getattr(args, "json", False):
            json.dump({"ok": False, "error": str(exc)}, sys.stdout, sort_keys=True)
            sys.stdout.write("\n")
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError(f"unhandled command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
