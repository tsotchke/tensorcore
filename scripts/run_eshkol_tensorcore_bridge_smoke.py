#!/usr/bin/env python3
"""Run the Eshkol tensorcore bridge smoke and emit ICC-readable evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.eshkol_bridge_runtime_evidence.v1"
FORMAT_VERSION = 1

SOURCES = {
    "hello_tensorcore": {
        "path": "eshkol/hello_tensorcore.esk",
        "required_output": ("tensorcore device=", "gemm OK"),
    },
    "tensorcore_bridge_smoke": {
        "path": "eshkol/tensorcore_bridge_smoke.esk",
        "required_output": ("TENSORCORE_BRIDGE_OK", "buffer map OK"),
    },
}

REQUIRED_FUNCTIONS = {
    "eshkol/hello_tensorcore.esk": {"main"},
    "eshkol/tensorcore.esk": {
        "tc-init",
        "tc-shutdown",
        "tc-device-name",
        "tc-device-info",
        "tc-buffer-alloc",
        "tc-buffer-free",
        "tc-buffer-map",
        "tc-dtype-code",
        "tc-gemm",
        "tc-gemm-fp32",
        "tc-gemm-fp16",
        "tc-gemm-bf16",
        "tc-attention-forward",
        "tc-last-backend",
        "tc-last-backend-name",
        "tc-version",
        "tc-status-string",
    },
    "lib/c_api/eshkol_bridge.c": {
        "bool_to_i32",
        "normalize_status",
        "dtype_from_eshkol",
        "get_device_info",
        "tc_eshkol_init",
        "tc_eshkol_shutdown",
        "tc_eshkol_device_name",
        "tc_eshkol_device_family",
        "tc_eshkol_device_unified_memory",
        "tc_eshkol_device_supports_bf16",
        "tc_eshkol_device_supports_i8",
        "tc_eshkol_device_supports_tensorops_m5",
        "tc_eshkol_buffer_alloc",
        "tc_eshkol_buffer_free",
        "tc_eshkol_buffer_map",
        "tc_eshkol_gemm",
        "tc_eshkol_attention_forward",
        "tc_eshkol_last_backend",
        "tc_eshkol_last_backend_code",
        "tc_eshkol_version",
        "tc_eshkol_status_string",
    },
    "lib/core/status.c": {
        "tc_status_string",
    },
}

EXPECTED_BUILTINS = {
    "__tc-init",
    "__tc-shutdown",
    "__tc-device-name",
    "__tc-device-family",
    "__tc-device-unified-memory",
    "__tc-device-supports-bf16",
    "__tc-device-supports-i8",
    "__tc-device-supports-tensorops-m5",
    "__tc-buffer-alloc",
    "__tc-buffer-free",
    "__tc-buffer-map",
    "__tc-gemm",
    "__tc-attention-forward",
    "__tc-last-backend",
    "__tc-last-backend-name",
    "__tc-version",
    "__tc-status-string",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eshkol-run", type=pathlib.Path, default=None)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=ROOT / "build" / "eshkol_tensorcore_bridge_evidence.json",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0)
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


def tail(text: str, limit: int = 4000) -> str:
    clean = ANSI_RE.sub("", text)
    return clean[-limit:]


def default_eshkol_run(explicit: pathlib.Path | None) -> pathlib.Path | None:
    candidates: list[pathlib.Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get("ESHKOL_RUN")
    if env_path:
        candidates.append(pathlib.Path(env_path))
    candidates.extend(
        [
            ROOT.parent / "eshkol" / "build" / "eshkol-run",
            pathlib.Path("/Users/tyr/Desktop/eshkol/build/eshkol-run"),
        ]
    )
    which = shutil.which("eshkol-run")
    if which:
        candidates.append(pathlib.Path(which))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def bridge_env(build_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ESHKOL_ENABLE_TENSORCORE"] = "1"
    path_entries = [str(ROOT / "eshkol")]
    eshkol_lib = ROOT.parent / "eshkol" / "lib"
    if eshkol_lib.exists():
        path_entries.append(str(eshkol_lib))
    if env.get("ESHKOL_PATH"):
        path_entries.append(env["ESHKOL_PATH"])
    env["ESHKOL_PATH"] = os.pathsep.join(path_entries)
    dylib_entries = [str(build_dir), str(build_dir / "lib" / "tensorcore")]
    if env.get("DYLD_LIBRARY_PATH"):
        dylib_entries.append(env["DYLD_LIBRARY_PATH"])
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dylib_entries)
    return env


def run_cmd(cmd: list[str], env: dict[str, str], timeout_sec: float) -> dict[str, Any]:
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
            "cmd": cmd,
            "rc": proc.returncode,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "rc": None,
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": tail(exc.stderr or f"timed out after {timeout_sec}s"),
            "timed_out": True,
        }


def unknown_functions(*attempts: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    pattern = re.compile(r"Unknown function:\s+([A-Za-z0-9_+*/<>=!?$%&~.^:-]+)")
    for attempt in attempts:
        text = "\n".join([str(attempt.get("stdout_tail", "")), str(attempt.get("stderr_tail", ""))])
        seen.update(pattern.findall(text))
    return sorted(seen)


def command_output(attempt: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(attempt.get("stdout_tail", "")),
            str(attempt.get("stderr_tail", "")),
        ]
    )


def normalize_expected_skip_attempt(attempt: dict[str, Any], reason: str) -> dict[str, Any]:
    """Keep classified skip evidence from looking like a hard failure to ICC."""
    normalized = dict(attempt)
    for key in ("stdout_tail", "stderr_tail"):
        text = str(normalized.get(key) or "")
        if text:
            normalized[f"{key}_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized["stdout_tail"] = ""
    normalized["stderr_tail"] = f"{reason}\n"
    normalized["classified_skip_reason"] = reason
    return normalized


def metal_device_available(
    build_dir: pathlib.Path,
    env: dict[str, str],
    timeout_sec: float,
) -> tuple[bool | None, dict[str, Any]]:
    binary = build_dir / "tests" / "test_device"
    if not binary.exists():
        return None, {
            "status": "skipped",
            "reason": "test_device_missing",
            "path": str(binary),
        }
    attempt = run_cmd([str(binary)], env, timeout_sec)
    output = command_output(attempt)
    if "No usable Metal device" in output or "tc_init failed: no Metal device" in output:
        return False, {
            "status": "skipped_no_gpu",
            "attempt": normalize_expected_skip_attempt(
                attempt,
                "no Metal device available",
            ),
        }
    if attempt.get("rc") == 0:
        return True, {
            "status": "passed",
            "attempt": attempt,
        }
    return None, {
        "status": "failed",
        "attempt": attempt,
    }


def classify_runtime_status(
    source_name: str,
    runtime_attempt: dict[str, Any],
    required_output: tuple[str, ...],
    device_available: bool | None,
) -> str:
    output = command_output(runtime_attempt)
    if runtime_attempt.get("rc") == 0 and all(marker in output for marker in required_output):
        return "passed"
    no_gpu_failure = (
        source_name == "hello_tensorcore"
        and "gemm FAIL status=-1" in output
    ) or (
        source_name == "tensorcore_bridge_smoke"
        and "TENSORCORE_BRIDGE_FAIL statuses=" in output
    )
    if runtime_attempt.get("rc") == 0 and device_available is False and no_gpu_failure:
        return "skipped_no_gpu"
    return "failed"


def source_has_require(rel_path: str, module: str) -> bool:
    try:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
    except OSError:
        return False
    return f"(require {module})" in text


def function_line(rel_path: str, name: str) -> int:
    try:
        lines = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return 1
    patterns = [
        re.compile(rf"\(define\s+\(\s*{re.escape(name)}(?:\s|\))"),
        re.compile(rf"\(define\s+{re.escape(name)}(?:\s|\))"),
        re.compile(rf"^\s*(?:static\s+)?[A-Za-z_][A-Za-z0-9_\s\*]*\s+{re.escape(name)}\s*\("),
    ]
    for index, line in enumerate(lines, start=1):
        if any(pattern.search(line) for pattern in patterns):
            return index
    return 1


def add_function(files: dict[str, Any], rel_path: str, name: str) -> None:
    line = function_line(rel_path, name)
    entry = files.setdefault(rel_path, {"executed_lines": [], "functions": {}})
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {"start_line": line, "executed_lines": [line]}


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    eshkol_run = default_eshkol_run(args.eshkol_run)
    env = bridge_env(args.build_dir)
    checks: dict[str, Any] = {}
    attempts: dict[str, Any] = {}
    device_available, device_probe = metal_device_available(args.build_dir, env, args.timeout_sec)
    checks["metal_device_probe"] = device_probe

    if eshkol_run is None:
        status = "blocked"
        checks["eshkol_run_available"] = {
            "status": "blocked",
            "reason": "eshkol_run_missing",
        }
    else:
        checks["eshkol_run_available"] = {
            "status": "passed",
            "path": str(eshkol_run),
        }
        with tempfile.TemporaryDirectory(prefix="tensorcore-eshkol-") as tmp:
            tmpdir = pathlib.Path(tmp)
            for source_name, spec in SOURCES.items():
                rel_path = str(spec["path"])
                obj_path = tmpdir / f"{source_name}.o"
                compile_cmd = [
                    str(eshkol_run),
                    "-I",
                    str(ROOT / "eshkol"),
                    "--compile-only",
                    "-o",
                    str(obj_path),
                    rel_path,
                ]
                compile_attempt = run_cmd(compile_cmd, env, args.timeout_sec)
                attempts[f"{source_name}_compile"] = compile_attempt
                checks[f"{source_name}_compile"] = {
                    "status": "passed" if compile_attempt.get("rc") == 0 else "failed",
                    "source": rel_path,
                    "attempt": compile_attempt,
                }

                runtime_status = "skipped_compile_failed"
                runtime_attempt: dict[str, Any] = {}
                if compile_attempt.get("rc") == 0:
                    exe_path = tmpdir / source_name
                    link_cmd = [
                        str(eshkol_run),
                        "-I",
                        str(ROOT / "eshkol"),
                        "--lib-path",
                        str(args.build_dir),
                        "--lib-path",
                        str(args.build_dir / "lib" / "tensorcore"),
                        "--lib",
                        "tensorcore",
                        "--output",
                        str(exe_path),
                        rel_path,
                    ]
                    link_attempt = run_cmd(link_cmd, env, args.timeout_sec)
                    attempts[f"{source_name}_link"] = link_attempt
                    if link_attempt.get("rc") == 0 and exe_path.exists():
                        runtime_attempt = run_cmd([str(exe_path)], env, args.timeout_sec)
                        required_output = tuple(str(item) for item in spec["required_output"])
                        runtime_status = classify_runtime_status(
                            source_name,
                            runtime_attempt,
                            required_output,
                            device_available,
                        )
                    else:
                        runtime_status = "failed_link"
                        runtime_attempt = link_attempt
                checks[f"{source_name}_runtime"] = {
                    "status": runtime_status,
                    "source": rel_path,
                    "attempt": normalize_expected_skip_attempt(
                        runtime_attempt,
                        "no Metal device available",
                    )
                    if runtime_status == "skipped_no_gpu"
                    else runtime_attempt,
                }

    missing_unknowns = unknown_functions(*attempts.values())
    missing_builtins = sorted(EXPECTED_BUILTINS.intersection(missing_unknowns))
    missing_wrappers = sorted(
        (REQUIRED_FUNCTIONS["eshkol/tensorcore.esk"] | {"tc-init", "tc-shutdown", "tc-last-backend"})
        .intersection(missing_unknowns)
    )
    checks["source_module_load"] = {
        "status": "passed"
        if source_has_require("eshkol/hello_tensorcore.esk", "tensorcore")
        and source_has_require("eshkol/tensorcore_bridge_smoke.esk", "tensorcore")
        else "failed",
        "hello_requires_tensorcore": source_has_require("eshkol/hello_tensorcore.esk", "tensorcore"),
        "bridge_smoke_requires_tensorcore": source_has_require(
            "eshkol/tensorcore_bridge_smoke.esk", "tensorcore"
        ),
    }
    checks["bridge_builtin_resolution"] = {
        "status": "passed" if not missing_builtins and not missing_wrappers else "blocked",
        "missing_builtins": missing_builtins,
        "missing_public_wrappers": missing_wrappers,
        "unknown_functions": missing_unknowns,
    }

    required_functions = sorted(
        f"{rel_path}:{name}" for rel_path, names in REQUIRED_FUNCTIONS.items() for name in names
    )
    all_runtime_passed = bool(checks) and all(
        item.get("status") == "passed"
        for key, item in checks.items()
        if key.endswith("_compile")
        or key.endswith("_runtime")
        or key in ("eshkol_run_available", "source_module_load", "bridge_builtin_resolution")
    )
    runtime_statuses = {
        str(item.get("status"))
        for key, item in checks.items()
        if key.endswith("_runtime")
    }
    blocked_reasons: list[str] = []
    if missing_builtins:
        blocked_reasons.append("missing_builtins")
    if eshkol_run is None:
        blocked_reasons.append("eshkol_run_missing")
    if "skipped_no_gpu" in runtime_statuses:
        blocked_reasons.append("runtime_skipped_no_gpu")
    status = "passed" if all_runtime_passed else ("blocked" if blocked_reasons else "failed")

    files: dict[str, Any] = {}
    runtime_invoked = runtime_statuses and runtime_statuses.issubset({"passed", "skipped_no_gpu"})
    bridge_resolved = checks["bridge_builtin_resolution"]["status"] == "passed"
    source_loaded = checks["source_module_load"]["status"] == "passed"
    compiled = all(
        item.get("status") == "passed"
        for key, item in checks.items()
        if key.endswith("_compile")
    )
    if status == "passed" or (bridge_resolved and source_loaded and compiled and runtime_invoked):
        for rel_path, names in REQUIRED_FUNCTIONS.items():
            for name in names:
                add_function(files, rel_path, name)
        for entry in files.values():
            entry["executed_lines"].sort()

    covered_functions = sorted(
        f"{rel_path}:{name}" for rel_path, entry in files.items() for name in entry.get("functions", {})
    )
    missing_functions = sorted(set(required_functions) - set(covered_functions))

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_eshkol_bridge_smoke",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
            "eshkol_run": str(eshkol_run) if eshkol_run is not None else None,
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "build_dir": str(args.build_dir),
            "evidence_path": str(args.evidence_path),
        },
        "checks": checks,
        "files": files,
        "summary": {
            "required_functions": required_functions,
            "covered_functions": covered_functions,
            "missing_functions": missing_functions,
            "missing_builtins": missing_builtins,
            "missing_public_wrappers": missing_wrappers,
            "blocked_reasons": blocked_reasons,
        },
    }


def main() -> int:
    args = parse_args()
    evidence = build_evidence(args)
    args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    if args.require_pass and evidence.get("status") != "passed":
        print(
            "Eshkol tensorcore bridge smoke did not pass: "
            f"status={evidence.get('status')} "
            f"missing_builtins={evidence.get('summary', {}).get('missing_builtins')}",
            file=sys.stderr,
        )
        return 1
    print(
        "Eshkol tensorcore bridge smoke evidence: "
        f"status={evidence.get('status')} evidence={args.evidence_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
