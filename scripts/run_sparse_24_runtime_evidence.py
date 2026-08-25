#!/usr/bin/env python3
"""Run sparse 2:4 smokes and emit ICC-readable runtime evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import pathlib
import re
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.sparse_24_runtime_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_MARKERS = (
    "check rejects unpruned",
    "prune: 2 nz per 4-block",
    "prune: kept are top-2",
    "check accepts pruned",
    "canonical K-axis 2:4 layout OK",
    "fallback sparse GEMM = dense",
    "OK",
)
REQUIRED_FUNCTIONS = {
    "lib/ops/sparse_gemm_cpu.cpp": {
        "read_elem",
        "write_elem",
        "supported_dtype",
        "tc_sparse_24_prune",
        "tc_sparse_24_check",
        "tc_sparse_24_gemm",
        "tc_sparse_24_available",
    },
    "tests/test_sparse_24.c": {
        "main",
    },
}
REQUIRED_HARDWARE_FUNCTIONS = {
    "lib/cuda/sparse_gemm.cpp": {
        "tc_sparse_24_gemm",
        "tc_sparse_24_available",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "sparse_24_runtime_evidence.json",
    )
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print evidence JSON to stdout.")
    return parser.parse_args()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def is_git_dirty() -> bool | None:
    try:
        subprocess.check_call(
            ["git", "diff", "--quiet"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return False
    except subprocess.CalledProcessError:
        return True
    except Exception:
        return None


def tail(text: str, limit: int = 8000) -> str:
    return text[-limit:]


def run_cmd(name: str, cmd: list[str], env: dict[str, str], timeout_sec: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(ROOT),
            "rc": proc.returncode,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
        }
    except FileNotFoundError as exc:
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(ROOT),
            "rc": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(ROOT),
            "rc": None,
            "timeout_seconds": timeout_sec,
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": tail(exc.stderr or ""),
        }


def build_env(build_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    dylib_entries = [str(build_dir), str(build_dir / "lib" / "tensorcore")]
    if env.get("DYLD_LIBRARY_PATH"):
        dylib_entries.append(env["DYLD_LIBRARY_PATH"])
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dylib_entries)
    metallib = build_dir / "tensorcore.metallib"
    if metallib.exists() and not env.get("TC_METALLIB"):
        env["TC_METALLIB"] = str(metallib)
    return env


def cmake_cache_text(build_dir: pathlib.Path) -> str:
    try:
        return (build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cmake_enabled(cache: str, key: str) -> bool:
    return bool(re.search(rf"^{re.escape(key)}:[^=]*=(?:ON|1|TRUE)$", cache, re.MULTILINE))


def function_line(rel_path: str, name: str) -> int:
    path = ROOT / rel_path
    needle = re.compile(rf"\b{re.escape(name)}\s*\(")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if not needle.search(line):
            continue
        signature = line
        for continuation in lines[index:]:
            signature += "\n" + continuation
            if "{" in continuation or ";" in continuation:
                break
        if "{" in signature and (";" not in signature or signature.index("{") < signature.index(";")):
            return index
    return 1


def add_function(files: dict[str, Any], rel_path: str, name: str) -> None:
    line = function_line(rel_path, name)
    entry = files.setdefault(rel_path, {"executed_lines": [], "functions": {}})
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {"start_line": line, "executed_lines": [line]}


def flatten_required(functions: dict[str, set[str]]) -> list[str]:
    return sorted(f"{path}:{name}" for path, names in functions.items() for name in names)


def covered_functions(files: dict[str, Any]) -> list[str]:
    covered: list[str] = []
    for rel_path, entry in files.items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if isinstance(functions, dict):
            covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def classify_attempt(attempt: dict[str, Any]) -> tuple[str, str | None]:
    text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    if attempt.get("rc") == 0 and all(marker in text for marker in REQUIRED_MARKERS):
        return "passed", None
    if attempt.get("rc") == 77 or "SKIP" in text:
        return "blocked", "test_skipped"
    if attempt.get("rc") is None and attempt.get("timeout_seconds"):
        return "failed", "timeout"
    missing = [marker for marker in REQUIRED_MARKERS if marker not in text]
    if missing:
        return "failed", "missing_markers:" + ",".join(missing)
    return "failed", "test_failed"


def parse_sparse_available(attempt: dict[str, Any]) -> int | None:
    text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    match = re.search(r"tc_sparse_24_available\s*=\s*([01])", text)
    if not match:
        return None
    return int(match.group(1))


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.build_dir.resolve()
    cache = cmake_cache_text(build_dir)
    env = build_env(build_dir)
    trace: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    files: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    failure_reasons: list[str] = []
    hardware_blocked_reasons: list[str] = []

    binary = build_dir / "tests" / "test_sparse_24"
    if not binary.exists():
        checks["sparse_24"] = {
            "status": "blocked",
            "runtime_status": "blocked",
            "blocked_reason": "test_binary_missing",
            "binary": str(binary),
        }
        blocked_reasons.append("sparse_24:test_binary_missing")
        available = None
        hardware_path = "unknown"
    else:
        attempt = run_cmd("sparse_24", [str(binary)], env, args.timeout_sec)
        trace.append(attempt)
        status, reason = classify_attempt(attempt)
        available = parse_sparse_available(attempt)
        hardware_path = "cusparseLt" if available == 1 else "dense_fallback"
        checks["sparse_24"] = {
            "status": status,
            "runtime_status": status,
            "binary": str(binary),
            "trace": "sparse_24",
            "required_markers": list(REQUIRED_MARKERS),
        }
        checks["sparse_24_k_axis_contract"] = {
            "status": status,
            "runtime_status": status,
            "binary": str(binary),
            "trace": "sparse_24",
            "layout": "linear_weight_N_by_K",
        }
        if reason:
            checks["sparse_24"]["reason" if status == "failed" else "blocked_reason"] = reason
        if status == "passed":
            for rel_path, names in REQUIRED_FUNCTIONS.items():
                for function in names:
                    add_function(files, rel_path, function)
        elif status == "blocked":
            blocked_reasons.append(f"sparse_24:{reason}")
        else:
            failure_reasons.append(f"sparse_24:{reason}")

    hardware_status = "passed" if available == 1 else "blocked"
    if hardware_status == "passed":
        for rel_path, names in REQUIRED_HARDWARE_FUNCTIONS.items():
            for function in names:
                add_function(files, rel_path, function)
    else:
        hardware_blocked_reasons.append("sparse_24_hardware:cusparseLt_unavailable")
    checks["sparse_24_hardware"] = {
        "status": hardware_status,
        "runtime_status": hardware_status,
        "backend": hardware_path,
        "provider": "cusparseLt" if available == 1 else "dense_fallback",
        "hardware_path": hardware_path,
        "blocked_reason": None if hardware_status == "passed" else "cusparseLt_unavailable",
    }

    for entry in files.values():
        entry["executed_lines"] = sorted(set(entry["executed_lines"]))

    required_functions = flatten_required(REQUIRED_FUNCTIONS)
    required_hardware_functions = flatten_required(REQUIRED_HARDWARE_FUNCTIONS)
    covered = covered_functions(files)
    missing_functions = sorted(set(required_functions) - set(covered))
    missing_hardware_functions = sorted(set(required_hardware_functions) - set(covered))
    if failure_reasons:
        status = "failed"
    elif blocked_reasons or missing_functions:
        status = "blocked"
    else:
        status = "passed"

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_sparse_24_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": is_git_dirty(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "build_dir": str(build_dir),
            "evidence": str(args.evidence_path),
        },
        "build_features": {
            "cuda": cmake_enabled(cache, "TC_ENABLE_CUDA"),
            "cusparseLt": cmake_enabled(cache, "TC_ENABLE_CUSPARSELT"),
        },
        "checks": checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "hardware_accelerated": available == 1,
            "hardware_path": hardware_path,
            "blocked_reasons": blocked_reasons,
            "failure_reasons": failure_reasons,
            "hardware_blocked_reasons": hardware_blocked_reasons,
            "required_functions": required_functions,
            "covered_functions": covered,
            "missing_functions": missing_functions,
            "required_hardware_functions": required_hardware_functions,
            "missing_hardware_functions": missing_hardware_functions,
        },
    }


def main() -> int:
    args = parse_args()
    evidence = build_evidence(args)
    args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence["paths"]["evidence"] = str(args.evidence_path)
    args.evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        reason = ",".join(
            evidence["summary"]["blocked_reasons"]
            or evidence["summary"]["failure_reasons"]
            or evidence["summary"]["hardware_blocked_reasons"]
        ) or "ok"
        print(
            "Sparse 2:4 evidence "
            f"{evidence['status']}: reason={reason} evidence={args.evidence_path} "
            f"hardware={evidence['summary']['hardware_path']} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )
    if args.require_pass and evidence["status"] != "passed":
        return 1
    if args.require_hardware and not evidence["summary"]["hardware_accelerated"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
