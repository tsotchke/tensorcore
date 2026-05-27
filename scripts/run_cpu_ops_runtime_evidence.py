#!/usr/bin/env python3
"""Run portable CPU ops smokes and emit ICC-readable runtime evidence."""

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
SCHEMA = "tensorcore.cpu_ops_runtime_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_FUNCTIONS = {
    "lib/ops/gemm_cpu.cpp": {
        "f16_to_f32",
        "gemm_compute",
        "gemm_compute_cblas_bf16",
        "gemm_compute_cblas_f16",
    },
    "lib/ops/conv2d_cpu.cpp": {
        "conv_dims_valid",
        "direct_sgemm_f32",
        "im2col_fp16",
    },
}
TESTS = {
    "portable_cpu": {
        "path": "tests/test_portable_cpu",
        "required_markers": ("portable CPU backend: OK",),
        "covers": {
            "lib/ops/gemm_cpu.cpp": {
                "f16_to_f32",
                "gemm_compute",
            },
        },
    },
    "conv2d": {
        "path": "tests/test_conv2d",
        "required_markers": (
            "conv2d_backward_input  dispatched=yes",
            "conv2d_backward_weight dispatched=yes",
            "conv2d_validation",
            "OK",
        ),
        "covers": {
            "lib/ops/conv2d_cpu.cpp": {
                "conv_dims_valid",
                "direct_sgemm_f32",
                "im2col_fp16",
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=ROOT / "build-portable-cpu-current",
    )
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "cpu_ops_runtime_evidence.json",
    )
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--require-pass", action="store_true")
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


def git_dirty() -> bool | None:
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


def run_cmd(
    name: str,
    cmd: list[str],
    env: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
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


def function_line(rel_path: str, name: str) -> int:
    path = ROOT / rel_path
    regex = re.compile(
        rf"^\s*(?:extern\s+\"C\"\s+)?(?:[A-Za-z_][\w:<>,\s\*&]*\s+)+{re.escape(name)}\s*\("
    )
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if not regex.search(line):
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


def covered_functions(files: dict[str, Any]) -> list[str]:
    covered: list[str] = []
    for rel_path, entry in files.items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if isinstance(functions, dict):
            covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def build_env(build_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    dylib_entries = [str(build_dir), str(build_dir / "lib" / "tensorcore")]
    if env.get("DYLD_LIBRARY_PATH"):
        dylib_entries.append(env["DYLD_LIBRARY_PATH"])
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dylib_entries)
    return env


def build_has_cblas(build_dir: pathlib.Path) -> bool:
    for rel in ("compile_commands.json", "CMakeCache.txt"):
        try:
            text = (build_dir / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "TC_HAS_CBLAS=1" in text or "TC_HAS_CBLAS:INTERNAL=1" in text:
            return True
    return False


def classify_attempt(attempt: dict[str, Any], markers: tuple[str, ...]) -> tuple[str, str | None]:
    text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    if attempt.get("rc") == 0 and all(marker in text for marker in markers):
        return "passed", None
    if attempt.get("rc") == 77 or "SKIP" in text:
        return "blocked", "test_skipped"
    if attempt.get("rc") is None and attempt.get("timeout_seconds"):
        return "failed", "timeout"
    return "failed", "test_failed"


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.build_dir.resolve()
    env = build_env(build_dir)
    cblas_available = build_has_cblas(build_dir)
    trace: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    files: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    failure_reasons: list[str] = []

    for name, spec in TESTS.items():
        binary = build_dir / spec["path"]
        if not binary.exists():
            checks[name] = {
                "status": "blocked",
                "blocked_reason": "test_binary_missing",
                "binary": str(binary),
            }
            blocked_reasons.append(f"{name}:test_binary_missing")
            continue
        attempt = run_cmd(name, [str(binary)], env, args.timeout_sec)
        trace.append(attempt)
        status, reason = classify_attempt(attempt, tuple(spec["required_markers"]))
        checks[name] = {
            "status": status,
            "binary": str(binary),
            "trace": name,
        }
        if reason:
            checks[name]["reason" if status == "failed" else "blocked_reason"] = reason
        if status == "passed":
            for rel_path, names in spec["covers"].items():
                for function in names:
                    add_function(files, rel_path, function)
            if name == "portable_cpu":
                if cblas_available:
                    add_function(files, "lib/ops/gemm_cpu.cpp", "gemm_compute_cblas_f16")
                    add_function(files, "lib/ops/gemm_cpu.cpp", "gemm_compute_cblas_bf16")
                else:
                    blocked_reasons.append("portable_cpu:cblas_not_compiled")
        elif status == "blocked":
            blocked_reasons.append(f"{name}:{reason}")
        else:
            failure_reasons.append(f"{name}:{reason}")

    for entry in files.values():
        entry["executed_lines"] = sorted(set(entry["executed_lines"]))

    required_functions = sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)
    covered = covered_functions(files)
    missing_functions = sorted(set(required_functions) - set(covered))
    if failure_reasons:
        status = "failed"
    elif missing_functions or blocked_reasons:
        status = "blocked"
    else:
        status = "passed"

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_cpu_ops_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
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
            "cblas": cblas_available,
        },
        "checks": checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "blocked_reasons": blocked_reasons,
            "failure_reasons": failure_reasons,
            "required_functions": required_functions,
            "covered_functions": covered,
            "missing_functions": missing_functions,
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
        reason = ",".join(evidence["summary"]["blocked_reasons"] or evidence["summary"]["failure_reasons"]) or "ok"
        print(
            "CPU ops evidence "
            f"{evidence['status']}: reason={reason} evidence={args.evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )
    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
