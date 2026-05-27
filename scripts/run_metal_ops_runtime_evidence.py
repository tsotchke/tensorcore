#!/usr/bin/env python3
"""Run Metal attention/Conv2D smokes and emit ICC-readable runtime evidence."""

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
SCHEMA = "tensorcore.metal_ops_runtime_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_FUNCTIONS = {
    "kernels/metal/fused_norm_gemv.metal": {
        "tg_sum32",
    },
    "lib/ops/attention.mm": {
        "encode_forward",
    },
    "lib/ops/conv.mm": {
        "conv_bytes",
    },
    "lib/ops/gemm.mm": {
        "batched_matrix_bytes",
    },
}
OPTIONAL_FUNCTIONS = {
    "kernels/metal/metal_simdgroup_event.h": {
        "async_copy",
        "async_copy_clamp_mode",
        "tc",
        "wait",
    },
}
TESTS = {
    "attention_correctness": {
        "path": "tests/test_attention_correctness",
        "required_markers": (
            "trace op=tc_attention_forward status=ok backend=simdgroup_matrix",
            "flash_attention",
            "OK",
        ),
        "covers": {
            "lib/ops/attention.mm": {
                "encode_forward",
            },
        },
    },
    "conv2d": {
        "path": "tests/test_conv2d",
        "required_markers": (
            "trace op=tc_conv2d_forward status=ok backend=metal_compute",
            "trace op=tc_conv2d_backward_input status=ok backend=metal_compute",
            "trace op=tc_conv2d_backward_weight status=ok backend=metal_compute",
            "conv2d_validation",
            "OK",
        ),
        "covers": {
            "lib/ops/conv.mm": {
                "conv_bytes",
            },
        },
    },
    "gemm_batched": {
        "path": "tests/test_gemm_f16",
        "required_markers": (
            "trace op=tc_gemm_batched status=ok backend=simdgroup_matrix",
            "batched batch=3",
            "batched transpose/padded",
            "OK",
        ),
        "covers": {
            "lib/ops/gemm.mm": {
                "batched_matrix_bytes",
            },
        },
    },
    "fused_norm_gemv": {
        "path": "tests/test_fused_norm_gemv",
        "required_markers": (
            "trace op=tc_fused_rmsnorm_gemv status=ok backend=metal_compute",
            "trace op=tc_fused_layernorm_gemv status=ok backend=metal_compute",
            "fused_rmsnorm_gemv",
            "fused_layernorm_gemv",
            "OK",
        ),
        "covers": {
            "kernels/metal/fused_norm_gemv.metal": {
                "tg_sum32",
            },
        },
    },
}

ASYNC_KERNEL_RE = re.compile(r"kernel=(tc_gemm_(?:f16|bf16)_f32_async(?:_128|_db)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "metal_ops_runtime_evidence.json",
    )
    parser.add_argument("--timeout-sec", type=float, default=120.0)
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


def tail(text: str, limit: int = 10000) -> str:
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
    namespace_regex = re.compile(rf"^\s*namespace\s+{re.escape(name)}\b")
    enum_regex = re.compile(rf"^\s*enum\s+(?:class\s+)?{re.escape(name)}\b")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if namespace_regex.search(line) or enum_regex.search(line):
            return index
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
    env["TC_TRACE"] = "1"
    metallib = build_dir / "tensorcore.metallib"
    if metallib.exists():
        env.setdefault("TC_METALLIB", str(metallib))
    dylib_entries = [str(build_dir), str(build_dir / "lib" / "tensorcore")]
    if env.get("DYLD_LIBRARY_PATH"):
        dylib_entries.append(env["DYLD_LIBRARY_PATH"])
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dylib_entries)
    return env


def classify_attempt(attempt: dict[str, Any], markers: tuple[str, ...]) -> tuple[str, str | None]:
    text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    if attempt.get("rc") == 0 and all(marker in text for marker in markers):
        return "passed", None
    if "no Metal device available" in text or "SKIP" in text:
        return "blocked", "metal_device_unavailable"
    if attempt.get("rc") is None and attempt.get("timeout_seconds"):
        return "failed", "timeout"
    return "failed", "test_failed"


def async_copy_shader_check(build_dir: pathlib.Path, trace: list[dict[str, Any]]) -> dict[str, Any]:
    metallib = build_dir / "tensorcore.metallib"
    if not metallib.exists():
        return {
            "status": "blocked",
            "blocked_reason": "metallib_missing",
            "metallib": str(metallib),
        }
    try:
        blob = metallib.read_bytes()
    except OSError as exc:
        return {
            "status": "blocked",
            "blocked_reason": "metallib_unreadable",
            "metallib": str(metallib),
            "error": str(exc),
        }
    compiled_async = b"tc_gemm_f16_f32_async" in blob or b"air.simdgroup_async_copy_2d" in blob
    trace_text = "\n".join(
        "\n".join([str(item.get("stdout_tail", "")), str(item.get("stderr_tail", ""))])
        for item in trace
    )
    runtime_kernels = sorted(set(ASYNC_KERNEL_RE.findall(trace_text)))
    if runtime_kernels:
        return {
            "status": "passed",
            "metallib": str(metallib),
            "compiled_async_kernel": compiled_async,
            "runtime_kernels": runtime_kernels,
            "reason": "A traced GEMM smoke selected an async-copy Metal kernel at runtime.",
        }
    return {
        "status": "blocked",
        "blocked_reason": "shader_line_execution_trace_unavailable",
        "metallib": str(metallib),
        "compiled_async_kernel": compiled_async,
        "reason": (
            "Async GEMM shader is present, but current host traces report public "
            "ops/backends rather than Metal shader function-line execution."
        ),
    }


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.build_dir.resolve()
    env = build_env(build_dir)
    trace: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    files: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    failure_reasons: list[str] = []
    optional_blocked_reasons: list[str] = []

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
        elif status == "blocked":
            blocked_reasons.append(f"{name}:{reason}")
        else:
            failure_reasons.append(f"{name}:{reason}")

    async_check = async_copy_shader_check(build_dir, trace)
    checks["async_copy_shader"] = async_check
    if async_check["status"] == "blocked":
        optional_blocked_reasons.append(f"async_copy_shader:{async_check.get('blocked_reason')}")
    elif async_check["status"] == "failed":
        failure_reasons.append(f"async_copy_shader:{async_check.get('reason')}")
    elif async_check["status"] == "passed":
        for rel_path, names in OPTIONAL_FUNCTIONS.items():
            for function in names:
                add_function(files, rel_path, function)

    for entry in files.values():
        entry["executed_lines"] = sorted(set(entry["executed_lines"]))

    required_functions = sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)
    optional_functions = sorted(f"{path}:{name}" for path, names in OPTIONAL_FUNCTIONS.items() for name in names)
    covered = covered_functions(files)
    missing_functions = sorted(set(required_functions) - set(covered))
    optional_missing_functions = sorted(set(optional_functions) - set(covered))
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
            "source": "tensorcore_metal_ops_probe",
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
        "checks": checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "blocked_reasons": blocked_reasons,
            "failure_reasons": failure_reasons,
            "optional_blocked_reasons": optional_blocked_reasons,
            "required_functions": required_functions,
            "covered_functions": covered,
            "missing_functions": missing_functions,
            "optional_missing_functions": optional_missing_functions,
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
            or evidence["summary"]["optional_blocked_reasons"]
        ) or "ok"
        print(
            "Metal ops evidence "
            f"{evidence['status']}: reason={reason} evidence={args.evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )
    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
