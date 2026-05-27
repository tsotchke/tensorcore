#!/usr/bin/env python3
"""Emit local host readiness diagnostics for M5 TensorOps runtime evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.m5_tensorops_runner_preflight.v1"
FORMAT_VERSION = 1
M5_NAME_RE = re.compile(r"\b(?:Apple\s+)?M(?:[5-9]|\d{2,})(?:\b|[^0-9])", re.IGNORECASE)
ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
ARTIFACT_MISSING = "artifact_missing"
SOURCE_FAILED = "source_failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "build" / "m5_tensorops_runner_preflight.json",
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build-m5-tensorops")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print evidence JSON to stdout.")
    return parser.parse_args()


def version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in re.findall(r"\d+", text):
        parts.append(int(item))
    return tuple(parts or [0])


def version_at_least(text: str, minimum: tuple[int, ...]) -> bool:
    parsed = version_tuple(text)
    width = max(len(parsed), len(minimum))
    return parsed + (0,) * (width - len(parsed)) >= minimum + (0,) * (width - len(minimum))


def run_cmd(cmd: list[str], timeout_sec: float) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        return {
            "cmd": cmd,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "rc": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "rc": None,
            "timeout_seconds": timeout_sec,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }


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


def collect_string_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            out.extend(collect_string_values(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(collect_string_values(child))
    return out


def display_device_names(system_profiler_json: str) -> list[str]:
    try:
        payload = json.loads(system_profiler_json)
    except json.JSONDecodeError:
        return []
    names: set[str] = set()
    interesting_keys = {
        "_name",
        "sppci_model",
        "sppci_chipset_model",
        "spdisplays_device_name",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in interesting_keys and isinstance(child, str):
                    names.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return sorted(names)


def m5_candidate_from_names(names: list[str]) -> bool:
    return any(M5_NAME_RE.search(name) for name in names)


def runtime_status_from_output(text: str) -> str | None:
    match = re.search(r"tensorops_runtime_status=([A-Za-z0-9_+-]+)", text)
    return match.group(1) if match else None


def host_platform_check() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    passed = system == "Darwin" and machine in {"arm64", "aarch64"}
    return {
        "status": "passed" if passed else "blocked",
        "system": system,
        "machine": machine,
        **(
            {}
            if passed
            else {
                "diagnostic_class": ENVIRONMENT_UNAVAILABLE,
                "diagnostic_message": f"host is {system}/{machine}; M5 TensorOps requires Darwin/arm64",
            }
        ),
    }


def xcode_check(timeout_sec: float) -> dict[str, Any]:
    attempt = run_cmd(["xcodebuild", "-version"], timeout_sec)
    status = "passed" if attempt.get("rc") == 0 else "blocked"
    result: dict[str, Any] = {"status": status, "attempt": attempt}
    if status == "blocked":
        result["diagnostic_class"] = ENVIRONMENT_UNAVAILABLE
        result["diagnostic_message"] = "xcodebuild -version did not complete; Xcode is unavailable"
    return result


def sdk26_check(timeout_sec: float) -> dict[str, Any]:
    attempt = run_cmd(["xcrun", "--show-sdk-version"], timeout_sec)
    sdk_version = str(attempt.get("stdout_tail") or "").strip()
    status = "passed" if attempt.get("rc") == 0 and version_at_least(sdk_version, (26, 0)) else "blocked"
    result: dict[str, Any] = {
        "status": status,
        "sdk_version": sdk_version,
        "minimum": "26.0",
        "attempt": attempt,
    }
    if status == "blocked":
        result["diagnostic_class"] = ENVIRONMENT_UNAVAILABLE
        result["diagnostic_message"] = f"SDK {sdk_version or 'unknown'} is below the SDK 26.0 TensorOps requirement"
    return result


def display_check(timeout_sec: float) -> dict[str, Any]:
    attempt = run_cmd(["system_profiler", "SPDisplaysDataType", "-json"], timeout_sec)
    names = display_device_names(str(attempt.get("stdout_tail") or ""))
    is_m5 = m5_candidate_from_names(names)
    if attempt.get("rc") != 0:
        status = "blocked"
    elif is_m5:
        status = "passed"
    else:
        status = "blocked"
    result: dict[str, Any] = {
        "status": status,
        "m5_name_candidate": is_m5,
        "device_names": names,
        "attempt": attempt,
    }
    if status == "blocked":
        if attempt.get("rc") != 0:
            message = "system_profiler SPDisplaysDataType -json did not complete; cannot prove M5 display GPU"
        else:
            display_names = ", ".join(names) if names else "none reported"
            message = f"display GPU is not M5 or newer ({display_names})"
        result["diagnostic_class"] = ENVIRONMENT_UNAVAILABLE
        result["diagnostic_message"] = message
    return result


def runtime_probe_check(build_dir: pathlib.Path, timeout_sec: float) -> dict[str, Any]:
    binary = build_dir / "tests" / "test_tensorops_runtime"
    if not binary.exists():
        return {
            "status": "skipped",
            "reason": "test_tensorops_runtime_missing",
            "path": str(binary),
            "diagnostic_class": ARTIFACT_MISSING,
            "diagnostic_message": "test_tensorops_runtime is not built yet; run the M5 build before requiring readiness",
        }
    attempt = run_cmd([str(binary)], timeout_sec)
    output = "\n".join([str(attempt.get("stdout_tail") or ""), str(attempt.get("stderr_tail") or "")])
    runtime_status = runtime_status_from_output(output)
    status = "passed" if attempt.get("rc") == 0 and runtime_status == "passed" else "blocked"
    result: dict[str, Any] = {
        "status": status,
        "runtime_status": runtime_status,
        "path": str(binary),
        "attempt": attempt,
    }
    if status == "blocked":
        if runtime_status in {"skipped_no_m5", "skipped_no_gpu", "skipped_sdk_too_old"}:
            result["diagnostic_class"] = ENVIRONMENT_UNAVAILABLE
            result["diagnostic_message"] = f"runtime probe was environment-skipped with {runtime_status}"
        elif attempt.get("rc") not in (0, None):
            result["diagnostic_class"] = SOURCE_FAILED
            result["diagnostic_message"] = f"runtime probe exited with {attempt.get('rc')}"
        else:
            result["diagnostic_class"] = SOURCE_FAILED
            result["diagnostic_message"] = "runtime probe did not emit tensorops_runtime_status=passed"
    return result


def diagnostics_for_checks(checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name, check in sorted(checks.items()):
        status = check.get("status")
        diagnostic_class = check.get("diagnostic_class")
        if status not in {"blocked", "skipped"} or not diagnostic_class:
            continue
        message = str(check.get("diagnostic_message") or check.get("reason") or f"{name} is not ready")
        diagnostics.append(
            {
                "id": f"m5_tensorops_preflight_diagnostic.{name}",
                "name": name,
                "status": "blocked" if status == "blocked" else "skipped",
                "message": message,
                "reason": message,
                "diagnostic_class": diagnostic_class,
                "check_status": status,
            }
        )
    return diagnostics


def overall_status(checks: dict[str, dict[str, Any]]) -> str:
    required = ["host_platform", "xcode", "sdk26", "display_gpu"]
    if any(checks.get(name, {}).get("status") != "passed" for name in required):
        return "blocked"
    runtime = checks.get("tensorops_runtime_probe", {})
    if runtime.get("status") == "passed":
        return "ready"
    if runtime.get("status") == "blocked":
        return "blocked"
    return "candidate"


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "host_platform": host_platform_check(),
        "xcode": xcode_check(args.timeout_sec),
        "sdk26": sdk26_check(args.timeout_sec),
        "display_gpu": display_check(args.timeout_sec),
        "tensorops_runtime_probe": runtime_probe_check(args.build_dir, args.timeout_sec),
    }
    status = overall_status(checks)
    diagnostics = diagnostics_for_checks(checks)
    diagnostic_class_counts = {
        diagnostic_class: sum(1 for item in diagnostics if item.get("diagnostic_class") == diagnostic_class)
        for diagnostic_class in sorted({str(item.get("diagnostic_class")) for item in diagnostics})
    }
    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_m5_tensorops_runner_preflight",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "build_dir": str(args.build_dir),
            "output": str(args.output),
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "summary": {
            "ready_for_m5_tensorops_runtime": status == "ready",
            "candidate_host": status in {"ready", "candidate"},
            "blocked_checks": sorted(
                name for name, check in checks.items() if check.get("status") == "blocked"
            ),
            "diagnostic_class_counts": diagnostic_class_counts,
            "environment_unavailable": ENVIRONMENT_UNAVAILABLE in diagnostic_class_counts,
            "source_failed": SOURCE_FAILED in diagnostic_class_counts,
        },
    }


def main() -> int:
    args = parse_args()
    evidence = build_evidence(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    print(
        "M5 TensorOps runner preflight: "
        f"status={evidence['status']} output={args.output}"
    )
    if args.require_ready and evidence["status"] != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
