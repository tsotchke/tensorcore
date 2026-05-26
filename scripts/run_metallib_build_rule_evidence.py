#!/usr/bin/env python3
"""Probe cmake/compile_metallib.cmake and emit ICC-readable JSON evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.metallib_build_rule_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_FUNCTIONS = {
    "cmake/compile_metallib.cmake": {"tc_compile_metallib"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "build" / "metallib-build-rule-evidence",
        help="Directory for the generated probe project and build tree.",
    )
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=None,
        help="Default: $WORK_DIR/metallib_build_rule_evidence.json",
    )
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
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
    return text[-limit:]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    name: str,
    cmd: list[str],
    cwd: pathlib.Path,
    timeout: int = 300,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "rc": proc.returncode,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
        }
    except FileNotFoundError as exc:
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "rc": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "rc": None,
            "timeout_seconds": timeout,
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": tail(exc.stderr or ""),
        }


def cmake_escape(value: pathlib.Path | str) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"')


def write_probe_project(source_dir: pathlib.Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    cmake_dir = cmake_escape(ROOT / "cmake")
    (source_dir / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(tensorcore_metallib_rule_probe LANGUAGES NONE)",
                f'list(APPEND CMAKE_MODULE_PATH "{cmake_dir}")',
                "include(compile_metallib)",
                "",
                "tc_compile_metallib(",
                "    TARGET tensorcore_metallib_probe",
                '    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/probe.metal"',
                '    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe.metallib"',
                "    STD metal3.0",
                '    FLAGS -gline-tables-only "-fmodules-cache-path=${CMAKE_CURRENT_BINARY_DIR}/clang-module-cache"',
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "probe.metal").write_text(
        "\n".join(
            [
                "#include <metal_stdlib>",
                "using namespace metal;",
                "",
                "kernel void tc_metallib_rule_probe(device float *out [[buffer(0)]],",
                "                                      uint id [[thread_position_in_grid]]) {",
                "    out[id] = 0.0f;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    return line_matching(rel_path, rf"^\s*function\s*\(\s*{re.escape(name)}\b")


def add_function(files: dict[str, Any], rel_path: str, name: str, extra_lines: list[int]) -> None:
    line = function_line(rel_path, name)
    executed = sorted({line, *extra_lines})
    files[rel_path] = {
        "executed_lines": executed,
        "functions": {
            name: {
                "start_line": line,
                "executed_lines": [line],
            },
        },
    }


def coverage_for(reason: str | None) -> dict[str, Any]:
    rel_path = "cmake/compile_metallib.cmake"
    if reason == "cmake_missing":
        return {}
    extra_lines: list[int] = []
    if reason == "non_apple_platform":
        extra_lines.append(line_matching(rel_path, r"^\s*if\s*\(\s*NOT APPLE\s*\)"))
    elif reason == "xcrun_missing":
        extra_lines.append(line_matching(rel_path, r"^\s*if\s*\(\s*NOT TC_XCRUN_EXECUTABLE\s*\)"))
    else:
        extra_lines.extend(
            [
                line_matching(rel_path, r"-find metal"),
                line_matching(rel_path, r"-find metallib"),
                line_matching(rel_path, r"^\s*add_custom_target\s*\("),
            ]
        )
    files: dict[str, Any] = {}
    add_function(files, rel_path, "tc_compile_metallib", extra_lines)
    return files


def covered_functions(files: dict[str, Any]) -> list[str]:
    covered: list[str] = []
    for rel_path, entry in files.items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if not isinstance(functions, dict):
            continue
        covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def classify_configure(text: str) -> str | None:
    if "tc_compile_metallib requires Apple platform" in text:
        return "non_apple_platform"
    if "tc_compile_metallib: xcrun was not found" in text:
        return "xcrun_missing"
    if "could not locate the Metal compiler" in text:
        return "metal_compiler_missing"
    if "could not locate the metallib linker" in text:
        return "metallib_linker_missing"
    return None


def classify_build(text: str) -> str | None:
    if "xcrun" in text and ("not found" in text or "unable to find utility" in text):
        return "xcrun_missing"
    return None


def build_evidence(work_dir: pathlib.Path, cmake: str) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    source_dir = work_dir / "source"
    build_dir = work_dir / "build"
    evidence_path = work_dir / "metallib_build_rule_evidence.json"
    trace: list[dict[str, Any]] = []
    checks: dict[str, Any] = {
        "cmake_available": {
            "status": "passed" if shutil.which(cmake) or pathlib.Path(cmake).exists() else "blocked",
            "path": cmake,
        },
        "probe_project": {"status": "skipped"},
        "configure_rule": {"status": "skipped"},
        "build_metallib": {"status": "skipped"},
    }
    blocked_reason: str | None = None
    failure_reason: str | None = None
    artifact_hash: str | None = None

    if checks["cmake_available"]["status"] == "blocked":
        blocked_reason = "cmake_missing"
        checks["cmake_available"]["blocked_reason"] = blocked_reason
    else:
        write_probe_project(source_dir)
        checks["probe_project"] = {
            "status": "passed",
            "source_dir": str(source_dir),
            "cmakelists": str(source_dir / "CMakeLists.txt"),
            "metal_source": str(source_dir / "probe.metal"),
        }

        configure = run_command(
            "configure_rule",
            [cmake, "-S", str(source_dir), "-B", str(build_dir)],
            cwd=ROOT,
        )
        trace.append(configure)
        configure_text = "\n".join([configure.get("stdout_tail", ""), configure.get("stderr_tail", "")])
        if configure.get("rc") != 0:
            blocked_reason = classify_configure(configure_text)
            if blocked_reason is not None:
                checks["configure_rule"] = {
                    "status": "blocked",
                    "blocked_reason": blocked_reason,
                    "trace": "configure_rule",
                }
                checks["build_metallib"] = {
                    "status": "skipped",
                    "reason": "configure_blocked",
                }
            else:
                failure_reason = "configure_failed"
                checks["configure_rule"] = {
                    "status": "failed",
                    "reason": failure_reason,
                    "trace": "configure_rule",
                }
                checks["build_metallib"] = {
                    "status": "skipped",
                    "reason": "configure_failed",
                }
        else:
            checks["configure_rule"] = {
                "status": "passed",
                "trace": "configure_rule",
            }
            output = build_dir / "probe.metallib"
            build = run_command(
                "build_metallib",
                [cmake, "--build", str(build_dir), "--target", "tensorcore_metallib_probe", "--parallel"],
                cwd=ROOT,
            )
            trace.append(build)
            build_text = "\n".join([build.get("stdout_tail", ""), build.get("stderr_tail", "")])
            if build.get("rc") == 0 and output.exists() and output.stat().st_size > 0:
                artifact_hash = sha256_file(output)
                checks["build_metallib"] = {
                    "status": "passed",
                    "trace": "build_metallib",
                    "output": str(output),
                    "output_size": output.stat().st_size,
                    "artifact_hash": artifact_hash,
                }
            else:
                blocked_reason = classify_build(build_text)
                if blocked_reason is not None:
                    checks["build_metallib"] = {
                        "status": "blocked",
                        "blocked_reason": blocked_reason,
                        "trace": "build_metallib",
                        "output": str(output),
                        "output_exists": output.exists(),
                    }
                else:
                    failure_reason = "build_failed" if build.get("rc") != 0 else "metallib_output_missing"
                    checks["build_metallib"] = {
                        "status": "failed",
                        "reason": failure_reason,
                        "trace": "build_metallib",
                        "output": str(output),
                        "output_exists": output.exists(),
                    }

    status = "blocked" if blocked_reason else "failed" if failure_reason else "passed"
    backend = "metal" if status == "passed" else "unsupported" if blocked_reason == "non_apple_platform" else "unknown"
    error = blocked_reason or failure_reason
    files = coverage_for(blocked_reason)
    required_functions = sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)
    covered = covered_functions(files)
    missing = sorted(set(required_functions) - set(covered))

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_metallib_build_rule_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
            "host_system": platform.system(),
            "host_machine": platform.machine(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "work_dir": str(work_dir),
            "source_dir": str(source_dir),
            "build_dir": str(build_dir),
            "evidence": str(evidence_path),
        },
        "toolchain": {
            "cmake": cmake,
            "xcrun": shutil.which("xcrun"),
        },
        "checks": checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "backend": backend,
            "artifact_hash": artifact_hash,
            "error": error,
            "blocked_reason": blocked_reason,
            "failure_reason": failure_reason,
            "required_functions": required_functions,
            "covered_functions": covered,
            "missing_functions": missing,
        },
    }


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    evidence_path = args.evidence_path or (work_dir / "metallib_build_rule_evidence.json")
    evidence = build_evidence(work_dir, args.cmake)

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence["paths"]["evidence"] = str(evidence_path)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        reason = evidence["summary"]["blocked_reason"] or evidence["summary"]["failure_reason"] or "ok"
        print(
            "metallib build-rule evidence "
            f"{evidence['status']}: reason={reason} evidence={evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )

    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
