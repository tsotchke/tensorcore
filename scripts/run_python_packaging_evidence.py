#!/usr/bin/env python3
"""Probe setup.py native-artifact packaging paths and emit ICC-readable evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.python_packaging_evidence.v1"
FORMAT_VERSION = 1
REQUIRED_FUNCTIONS = {
    "setup.py": {"_run_tool", "build_py_with_native_artifacts.run"},
}
NATIVE_LIBRARY_NAMES = {
    "darwin": ("libtensorcore.dylib",),
    "linux": ("libtensorcore.so",),
    "windows": ("tensorcore.dll", "libtensorcore.dll"),
}
OPTIONAL_ARTIFACTS = ("tensorcore.metallib",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "build" / "python-packaging-evidence",
        help="Directory for temporary packaging outputs.",
    )
    parser.add_argument(
        "--native-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing native artifacts. Defaults to build/ then build/lib/.",
    )
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=None,
        help="Default: $WORK_DIR/python_packaging_evidence.json",
    )
    parser.add_argument("--timeout-sec", type=float, default=180.0)
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


def tail(text: str, limit: int = 6000) -> str:
    return text[-limit:]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_key() -> str:
    system = platform.system().lower()
    if system.startswith("darwin"):
        return "darwin"
    if system.startswith("linux"):
        return "linux"
    if system.startswith("windows"):
        return "windows"
    return system


def artifact_dirs(explicit: pathlib.Path | None) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        for env_name in ("TENSORCORE_NATIVE_DIR", "TENSORCORE_LIB", "TC_METALLIB"):
            value = os.environ.get(env_name)
            if value:
                path = pathlib.Path(value).expanduser()
                candidates.append(path.parent if path.is_file() else path)
        candidates.extend([ROOT / "build", ROOT / "build" / "lib", ROOT / "build-portable-cpu-current"])
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def required_artifacts() -> list[str]:
    libs = NATIVE_LIBRARY_NAMES.get(host_key(), ())
    required = list(libs)
    if host_key() == "darwin":
        required.extend(OPTIONAL_ARTIFACTS)
    return required


def find_artifacts(explicit: pathlib.Path | None) -> dict[str, pathlib.Path]:
    found: dict[str, pathlib.Path] = {}
    names = set(name for names in NATIVE_LIBRARY_NAMES.values() for name in names)
    names.update(OPTIONAL_ARTIFACTS)
    for directory in artifact_dirs(explicit):
        for name in names:
            if name in found:
                continue
            candidate = directory / name
            if candidate.exists():
                found[name] = candidate
    return found


def native_dir_for(found: dict[str, pathlib.Path], work_dir: pathlib.Path) -> pathlib.Path:
    native_dir = work_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    for path in found.values():
        shutil.copy2(path, native_dir / path.name)
    return native_dir


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


def write_run_tool_probe(path: pathlib.Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import pathlib",
                "import runpy",
                "import setuptools",
                "import sys",
                "",
                "root = pathlib.Path(sys.argv[1])",
                "dylib = pathlib.Path(sys.argv[2])",
                "original_setup = setuptools.setup",
                "setuptools.setup = lambda *args, **kwargs: None",
                "try:",
                "    ns = runpy.run_path(str(root / 'setup.py'))",
                "finally:",
                "    setuptools.setup = original_setup",
                "print(ns['_run_tool'](['lipo', '-archs', str(dylib)]).strip())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def setup_env(native_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env["TENSORCORE_NATIVE_DIR"] = str(native_dir)
    env["TENSORCORE_REQUIRE_METALLIB"] = "1" if host_key() == "darwin" else "0"
    return env


def wheel_platform_tag() -> str | None:
    if host_key() != "darwin":
        return None
    arch = platform.machine() or "arm64"
    return f"macosx_15_0_{arch}"


def line_matching(rel_path: str, pattern: str, start: int = 1) -> int:
    path = ROOT / rel_path
    regex = re.compile(pattern)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1
    for index, line in enumerate(lines, start=1):
        if index >= start and regex.search(line):
            return index
    return 1


def function_line(rel_path: str, name: str) -> int:
    return line_matching(rel_path, rf"^\s*def\s+{re.escape(name)}\s*\(")


def class_method_line(rel_path: str, class_name: str, method_name: str) -> int:
    class_line = line_matching(rel_path, rf"^\s*class\s+{re.escape(class_name)}\b")
    return line_matching(rel_path, rf"^\s+def\s+{re.escape(method_name)}\s*\(", class_line)


def add_function(files: dict[str, Any], rel_path: str, name: str, start_line: int) -> None:
    entry = files.setdefault(rel_path, {"executed_lines": [], "functions": {}})
    if start_line not in entry["executed_lines"]:
        entry["executed_lines"].append(start_line)
    entry["functions"][name] = {"start_line": start_line, "executed_lines": [start_line]}


def coverage_for(run_tool_passed: bool, build_py_passed: bool) -> dict[str, Any]:
    files: dict[str, Any] = {}
    if run_tool_passed:
        add_function(files, "setup.py", "_run_tool", function_line("setup.py", "_run_tool"))
    if build_py_passed:
        add_function(
            files,
            "setup.py",
            "build_py_with_native_artifacts.run",
            class_method_line("setup.py", "build_py_with_native_artifacts", "run"),
        )
        copy_line = line_matching("setup.py", r"shutil\.copy2")
        files["setup.py"]["executed_lines"].append(copy_line)
    for entry in files.values():
        entry["executed_lines"] = sorted(set(entry["executed_lines"]))
    return files


def covered_functions(files: dict[str, Any]) -> list[str]:
    covered: list[str] = []
    for rel_path, entry in files.items():
        functions = entry.get("functions") if isinstance(entry, dict) else None
        if isinstance(functions, dict):
            covered.extend(f"{rel_path}:{name}" for name in functions)
    return sorted(covered)


def inspect_build_py_outputs(build_base: pathlib.Path, names: list[str]) -> dict[str, Any]:
    package_dir = build_base / "lib" / "tensorcore"
    copied: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        candidate = package_dir / name
        if candidate.exists():
            copied[name] = {
                "path": str(candidate),
                "size": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        else:
            missing.append(name)
    return {
        "status": "passed" if not missing else "failed",
        "package_dir": str(package_dir),
        "copied": copied,
        "missing": missing,
    }


def inspect_wheel(dist_dir: pathlib.Path, names: list[str]) -> dict[str, Any]:
    wheels = sorted(dist_dir.glob("tensorcore_apple-*.whl"))
    if not wheels:
        return {"status": "failed", "reason": "wheel_missing", "dist_dir": str(dist_dir)}
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        archive_names = set(archive.namelist())
    missing: list[str] = []
    for name in names:
        suffix = f"tensorcore/{name}"
        if not any(item == suffix or item.endswith(f"/purelib/{suffix}") for item in archive_names):
            missing.append(name)
    return {
        "status": "passed" if not missing else "failed",
        "wheel": str(wheel),
        "wheel_size": wheel.stat().st_size,
        "wheel_sha256": sha256_file(wheel),
        "missing": missing,
    }


def build_evidence(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    build_base = work_dir / "build-base"
    bdist_dir = work_dir / "bdist"
    dist_dir = work_dir / "dist"
    evidence_path = args.evidence_path or (work_dir / "python_packaging_evidence.json")
    found = find_artifacts(args.native_dir)
    required = required_artifacts()
    missing_required = [name for name in required if name not in found]
    trace: list[dict[str, Any]] = []
    platform_tag = wheel_platform_tag()
    checks: dict[str, Any] = {
        "native_artifacts": {
            "status": "passed" if not missing_required else "blocked",
            "required": required,
            "found": {name: str(path) for name, path in sorted(found.items())},
            "missing": missing_required,
        },
        "run_tool_lipo": {"status": "skipped"},
        "build_py_native_copy": {"status": "skipped"},
        "bdist_wheel_native_artifacts": {"status": "skipped"},
    }
    blocked_reason: str | None = "native_artifacts_missing" if missing_required else None
    failure_reason: str | None = None
    run_tool_passed = False
    build_py_passed = False

    if blocked_reason is None:
        native_dir = native_dir_for(found, work_dir)
        env = setup_env(native_dir)
        artifact_names = required

        if host_key() == "darwin":
            lipo = shutil.which("lipo")
            if not lipo:
                blocked_reason = "lipo_missing"
                checks["run_tool_lipo"] = {"status": "blocked", "blocked_reason": blocked_reason}
            else:
                probe = work_dir / "run_tool_probe.py"
                write_run_tool_probe(probe)
                lipo_attempt = run_cmd(
                    "run_tool_lipo",
                    [sys.executable, str(probe), str(ROOT), str(native_dir / "libtensorcore.dylib")],
                    env,
                    args.timeout_sec,
                )
                trace.append(lipo_attempt)
                if lipo_attempt.get("rc") == 0:
                    run_tool_passed = True
                    checks["run_tool_lipo"] = {
                        "status": "passed",
                        "trace": "run_tool_lipo",
                        "arches": lipo_attempt.get("stdout_tail", "").split(),
                    }
                else:
                    failure_reason = "run_tool_lipo_failed"
                    checks["run_tool_lipo"] = {
                        "status": "failed",
                        "trace": "run_tool_lipo",
                        "reason": failure_reason,
                    }
        else:
            blocked_reason = "non_macos_lipo_validation"
            checks["run_tool_lipo"] = {"status": "blocked", "blocked_reason": blocked_reason}

        build_attempt = run_cmd(
            "build_py_native_copy",
            [
                sys.executable,
                "setup.py",
                "build",
                "--build-base",
                str(build_base),
            ],
            env,
            args.timeout_sec,
        )
        trace.append(build_attempt)
        copied = inspect_build_py_outputs(build_base, artifact_names)
        if build_attempt.get("rc") == 0 and copied["status"] == "passed":
            build_py_passed = True
            checks["build_py_native_copy"] = {
                "status": "passed",
                "trace": "build_py_native_copy",
                **copied,
            }
        elif failure_reason is None:
            failure_reason = "build_py_native_copy_failed"
            checks["build_py_native_copy"] = {
                "status": "failed",
                "trace": "build_py_native_copy",
                "reason": failure_reason,
                **copied,
            }

        wheel_cmd = [
            sys.executable,
            "setup.py",
            "build",
            "--build-base",
            str(build_base),
            "bdist_wheel",
            "--bdist-dir",
            str(bdist_dir),
            "--dist-dir",
            str(dist_dir),
        ]
        if platform_tag:
            wheel_cmd.extend(["--plat-name", platform_tag])
        wheel_attempt = run_cmd(
            "bdist_wheel_native_artifacts",
            wheel_cmd,
            env,
            args.timeout_sec,
        )
        trace.append(wheel_attempt)
        wheel = inspect_wheel(dist_dir, artifact_names)
        if wheel_attempt.get("rc") == 0 and wheel["status"] == "passed":
            checks["bdist_wheel_native_artifacts"] = {
                "status": "passed",
                "trace": "bdist_wheel_native_artifacts",
                "platform_tag": platform_tag,
                **wheel,
            }
        elif failure_reason is None:
            failure_reason = "bdist_wheel_native_artifacts_failed"
            checks["bdist_wheel_native_artifacts"] = {
                "status": "failed",
                "trace": "bdist_wheel_native_artifacts",
                "reason": failure_reason,
                "platform_tag": platform_tag,
                **wheel,
            }

    if failure_reason:
        status = "failed"
    elif blocked_reason and not (run_tool_passed and build_py_passed):
        status = "blocked"
    else:
        status = "passed"

    files = coverage_for(run_tool_passed, build_py_passed)
    required_functions = sorted(f"{path}:{name}" for path, names in REQUIRED_FUNCTIONS.items() for name in names)
    covered = covered_functions(files)
    missing_functions = sorted(set(required_functions) - set(covered))

    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_python_packaging_probe",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
            "host_system": platform.system(),
            "host_machine": platform.machine(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "run": {
            "exit_status": "0" if status == "passed" else status,
            "platform_tag": platform_tag,
        },
        "project": {"root": str(ROOT)},
        "python": {"executable": sys.executable},
        "paths": {
            "work_dir": str(work_dir),
            "build_base": str(build_base),
            "bdist_dir": str(bdist_dir),
            "dist_dir": str(dist_dir),
            "evidence": str(evidence_path),
        },
        "checks": checks,
        "trace": trace,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "blocked_reason": blocked_reason if status == "blocked" else None,
            "failure_reason": failure_reason,
            "required_functions": required_functions,
            "covered_functions": covered,
            "missing_functions": missing_functions,
        },
    }


def main() -> int:
    args = parse_args()
    evidence_path = args.evidence_path or (args.work_dir.resolve() / "python_packaging_evidence.json")
    evidence = build_evidence(args)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence["paths"]["evidence"] = str(evidence_path)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        reason = evidence["summary"]["blocked_reason"] or evidence["summary"]["failure_reason"] or "ok"
        print(
            "python packaging evidence "
            f"{evidence['status']}: reason={reason} evidence={evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )
    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
