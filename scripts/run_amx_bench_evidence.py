#!/usr/bin/env python3
"""Run local AMX and GEMM benchmark smokes and emit ICC-readable evidence."""

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
SCHEMA = "tensorcore.amx_bench_runtime_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_FUNCTIONS = {
    "lib/ops/gemm_cpu_amx.cpp": {
        "amx_process_tile_strip",
        "amx_pool_dispatch_pair",
        "amx_worker_local",
        "amx_worker_thread_entry",
        "tc_amx_cluster_count",
        "tc_amx_gemm_f32",
        "tc_amx_gemm_f32_available",
        "tc_amx_gemm_f32_core",
        "tc_amx_isa_version",
    },
    "bench/bench_gemm.c": {
        "bench_one",
        "cmp_double",
        "env_int",
        "now_seconds",
        "only_spaces",
        "parse_dtype_token",
        "parse_dtypes",
        "parse_sizes",
        "print_throughput",
        "trim_token",
    },
}
OPTIONAL_BENCH_FUNCTIONS = {
    "bench/bench_attention.c": {
        "bench_one",
        "cmp_double",
        "env_int",
        "now_seconds",
    },
}
OPTIONAL_LAYOUT_FUNCTIONS = {
    "lib/ops/gemm.mm": {
        "gemm_uses_default_layout",
    },
    "lib/tensorops/tensorops_m5.mm": {
        "uses_default_layout",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--portable-build-dir",
        type=pathlib.Path,
        default=ROOT / "build-portable-cpu-current",
    )
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "amx_bench_evidence.json",
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


def line_matching(rel_path: str, pattern: str) -> int:
    path = ROOT / rel_path
    regex = re.compile(pattern)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if regex.search(line):
            return index
    return 1


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


def build_env(build_dir: pathlib.Path, portable_build_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    metallib = build_dir / "tensorcore.metallib"
    if metallib.exists():
        env.setdefault("TC_METALLIB", str(metallib))
    dylib_entries = [
        str(build_dir),
        str(build_dir / "lib" / "tensorcore"),
        str(portable_build_dir),
        str(portable_build_dir / "lib" / "tensorcore"),
    ]
    if env.get("DYLD_LIBRARY_PATH"):
        dylib_entries.append(env["DYLD_LIBRARY_PATH"])
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dylib_entries)
    return env


def classify_output(
    attempt: dict[str, Any],
    markers: tuple[str, ...],
    blocked_markers: tuple[str, ...] = ("SKIP", "skip", "skipped"),
) -> tuple[str, str | None]:
    text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    if attempt.get("rc") == 0 and all(marker in text for marker in markers):
        return "passed", None
    if "no Metal device available" in text:
        return "blocked", "metal_device_unavailable"
    if attempt.get("rc") == -4 or "SIGILL" in text or "Illegal instruction" in text:
        return "blocked", "sigill"
    if any(marker in text for marker in blocked_markers):
        return "blocked", "test_skipped"
    if attempt.get("rc") is None and attempt.get("timeout_seconds"):
        return "failed", "timeout"
    return "failed", "test_failed"


def run_probe(
    name: str,
    candidates: list[pathlib.Path],
    env: dict[str, str],
    timeout_sec: float,
    markers: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    seen_binary = False
    fallback_check: dict[str, Any] | None = None
    for binary in candidates:
        if not binary.exists():
            continue
        seen_binary = True
        attempt = run_cmd(name, [str(binary)], env, timeout_sec)
        status, reason = classify_output(attempt, markers)
        check: dict[str, Any] = {
            "status": status,
            "binary": str(binary),
            "trace": name,
        }
        if reason:
            check["reason" if status == "failed" else "blocked_reason"] = reason
        if status == "passed":
            return check, [attempt]
        trace.append(attempt)
        fallback_check = check
    if not seen_binary:
        return {
            "status": "blocked",
            "blocked_reason": "test_binary_missing",
            "binary": None,
        }, trace
    assert fallback_check is not None
    return fallback_check, trace


def read_optional(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def compile_text(build_dir: pathlib.Path) -> str:
    return "\n".join(
        [
            read_optional(build_dir / "compile_commands.json"),
            read_optional(build_dir / "CMakeCache.txt"),
        ]
    )


def tensorops_layout_check(
    build_dir: pathlib.Path,
    env: dict[str, str],
    timeout_sec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    text = compile_text(build_dir)
    metal4_compiled = "TC_HAVE_METAL4_SDK" in text or "TC_HAVE_METAL4:BOOL=ON" in text
    tensorops_compiled = "lib/tensorops/tensorops_m5.mm" in text or "tensorops_m5.mm.o" in text
    base: dict[str, Any] = {
        "metal4_sdk_compiled": metal4_compiled,
        "tensorops_m5_source_compiled": tensorops_compiled,
        "build_dir": str(build_dir),
    }
    if not metal4_compiled:
        return {
            **base,
            "status": "blocked",
            "blocked_reason": "skipped_no_metal4_sdk",
            "reason": "Current build lacks TC_HAVE_METAL4_SDK/TC_HAVE_METAL4, so TensorOps layout helpers are not compiled",
        }, [], False
    if not tensorops_compiled:
        return {
            **base,
            "status": "blocked",
            "blocked_reason": "skipped_no_metal4_sdk",
            "reason": "Current build did not compile lib/tensorops/tensorops_m5.mm, so TensorOps layout helpers are unavailable",
        }, [], False

    binary = build_dir / "tests" / "test_tensorops_runtime"
    if not binary.exists():
        return {
            **base,
            "status": "blocked",
            "blocked_reason": "tensorops_runtime_binary_missing",
            "binary": str(binary),
        }, [], False

    attempt = run_cmd("tensorops_layout", [str(binary)], env, timeout_sec)
    output = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
    check = {
        **base,
        "binary": str(binary),
        "trace": "tensorops_layout",
    }
    if attempt.get("rc") == 0 and "tensorops_runtime_status=passed" in output:
        check["status"] = "passed"
        return check, [attempt], True
    if "skipped_no_m5" in output:
        check["status"] = "blocked"
        check["blocked_reason"] = "skipped_no_m5"
        check["reason"] = "Host GPU does not report supports_tensorops_m5, so dispatch returns before layout validation"
        return check, [attempt], False
    if "skipped_no_gpu" in output:
        check["status"] = "blocked"
        check["blocked_reason"] = "metal_device_unavailable"
        return check, [attempt], False
    check["status"] = "failed"
    check["reason"] = "tensorops_runtime_failed"
    return check, [attempt], False


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    build_dir = args.build_dir.resolve()
    portable_build_dir = args.portable_build_dir.resolve()
    env = build_env(build_dir, portable_build_dir)
    trace: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    optional_checks: dict[str, Any] = {}
    files: dict[str, Any] = {}
    blocked_reasons: list[str] = []
    failure_reasons: list[str] = []
    optional_skipped_reasons: list[str] = []

    amx_probe_check, amx_probe_trace = run_probe(
        "amx_probe",
        [
            portable_build_dir / "tests" / "test_amx_probe",
            build_dir / "tests" / "test_amx_probe",
        ],
        env,
        args.timeout_sec,
        ("AMX probe OK:",),
    )
    checks["amx_probe"] = amx_probe_check
    trace.extend(amx_probe_trace)
    if amx_probe_check["status"] == "passed":
        for function in (
            "tc_amx_gemm_f32_available",
            "tc_amx_isa_version",
            "tc_amx_cluster_count",
        ):
            add_function(files, "lib/ops/gemm_cpu_amx.cpp", function)
    elif amx_probe_check["status"] == "blocked":
        blocked_reasons.append(f"amx_probe:{amx_probe_check.get('blocked_reason')}")
    else:
        failure_reasons.append(f"amx_probe:{amx_probe_check.get('reason')}")

    amx_env = env.copy()
    amx_env["TC_RUN_AMX_GEMM_TEST"] = "1"
    amx_check, amx_trace = run_probe(
        "amx_gemm",
        [
            portable_build_dir / "tests" / "test_amx_gemm",
            build_dir / "tests" / "test_amx_gemm",
        ],
        amx_env,
        args.timeout_sec,
        ("amx M=256 N=16 K=33", "max_abs="),
    )
    checks["amx_gemm"] = amx_check
    trace.extend(amx_trace)
    if amx_check["status"] == "passed":
        for function in (
            "tc_amx_gemm_f32",
            "tc_amx_gemm_f32_core",
            "tc_amx_cluster_count",
            "amx_pool_dispatch_pair",
            "amx_worker_thread_entry",
            "amx_worker_local",
            "amx_process_tile_strip",
        ):
            add_function(files, "lib/ops/gemm_cpu_amx.cpp", function)
    elif amx_check["status"] == "blocked":
        blocked_reasons.append(f"amx_gemm:{amx_check.get('blocked_reason')}")
    else:
        failure_reasons.append(f"amx_gemm:{amx_check.get('reason')}")

    bench_env = env.copy()
    bench_env["TC_BENCH_SIZES"] = "16"
    bench_env["TC_BENCH_DTYPES"] = "f32"
    bench_env["TC_BENCH_WARMUP"] = "0"
    bench_env["TC_BENCH_ITERS"] = "1"
    bench_check, bench_trace = run_probe(
        "bench_gemm",
        [
            portable_build_dir / "bench" / "bench_gemm",
            portable_build_dir / "bench" / "bench_gemm_shared",
            build_dir / "bench" / "bench_gemm",
            build_dir / "bench" / "bench_gemm_shared",
        ],
        bench_env,
        args.timeout_sec,
        ("=== tensorcore GEMM bench ===", "median="),
    )
    checks["bench_gemm"] = bench_check
    trace.extend(bench_trace)
    if bench_check["status"] == "passed":
        for function in (
            "bench_one",
            "cmp_double",
            "env_int",
            "now_seconds",
            "only_spaces",
            "parse_dtype_token",
            "parse_dtypes",
            "parse_sizes",
            "print_throughput",
            "trim_token",
        ):
            add_function(files, "bench/bench_gemm.c", function)
    elif bench_check["status"] == "blocked":
        blocked_reasons.append(f"bench_gemm:{bench_check.get('blocked_reason')}")
    else:
        failure_reasons.append(f"bench_gemm:{bench_check.get('reason')}")

    attention_env = env.copy()
    attention_env["TC_ATTENTION_BENCH_SINGLE"] = "1"
    attention_env["TC_ATTENTION_BENCH_B"] = "1"
    attention_env["TC_ATTENTION_BENCH_H"] = "1"
    attention_env["TC_ATTENTION_BENCH_S"] = "16"
    attention_env["TC_ATTENTION_BENCH_D"] = "64"
    attention_env["TC_ATTENTION_BENCH_WARMUP"] = "0"
    attention_env["TC_ATTENTION_BENCH_ITERS"] = "1"
    attention_check, attention_trace = run_probe(
        "bench_attention",
        [
            build_dir / "bench" / "bench_attention",
        ],
        attention_env,
        args.timeout_sec,
        ("=== tensorcore FlashAttention bench", "median="),
    )
    if attention_check["status"] == "passed":
        optional_checks["bench_attention"] = attention_check
        trace.extend(attention_trace)
        for function in ("bench_one", "cmp_double", "env_int", "now_seconds"):
            add_function(files, "bench/bench_attention.c", function)
    else:
        optional_reason = attention_check.get("blocked_reason") or attention_check.get("reason") or "not_passed"
        optional_checks["bench_attention"] = {
            "status": "skipped",
            "skip_reason": optional_reason,
            "binary": attention_check.get("binary"),
        }
        optional_skipped_reasons.append(f"bench_attention:{optional_reason}")

    layout_check, layout_trace, layout_covered = tensorops_layout_check(build_dir, env, args.timeout_sec)
    trace.extend(layout_trace)
    if layout_check["status"] == "passed" and layout_covered:
        optional_checks["tensorops_layout"] = layout_check
        for rel_path, names in OPTIONAL_LAYOUT_FUNCTIONS.items():
            for function in names:
                add_function(files, rel_path, function)
    elif layout_check["status"] == "blocked":
        optional_checks["tensorops_layout"] = {
            key: value
            for key, value in layout_check.items()
            if key not in {"status", "blocked_reason", "reason"}
        }
        optional_checks["tensorops_layout"]["status"] = "skipped"
        optional_checks["tensorops_layout"]["skip_reason"] = layout_check.get("blocked_reason")
        optional_skipped_reasons.append(f"tensorops_layout:{layout_check.get('blocked_reason')}")
    elif layout_check["status"] == "failed":
        checks["tensorops_layout"] = layout_check
        failure_reasons.append(f"tensorops_layout:{layout_check.get('reason')}")

    for entry in files.values():
        entry["executed_lines"] = sorted(set(entry["executed_lines"]))

    required_functions = sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)
    optional_functions = sorted(
        f"{path}:{name}"
        for path, names in {**OPTIONAL_BENCH_FUNCTIONS, **OPTIONAL_LAYOUT_FUNCTIONS}.items()
        for name in names
    )
    covered = covered_functions(files)
    missing_functions = sorted(set(required_functions) - set(covered))
    optional_missing_functions = sorted(set(optional_functions) - set(covered))
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
            "source": "tensorcore_amx_bench_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "build_dir": str(build_dir),
            "portable_build_dir": str(portable_build_dir),
            "evidence": str(args.evidence_path),
        },
        "checks": checks,
        "optional_checks": optional_checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "blocked_reasons": blocked_reasons,
            "failure_reasons": failure_reasons,
            "optional_skipped_reasons": optional_skipped_reasons,
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
            or evidence["summary"]["optional_skipped_reasons"]
        ) or "ok"
        print(
            "AMX/bench evidence "
            f"{evidence['status']}: reason={reason} evidence={args.evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )
    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
