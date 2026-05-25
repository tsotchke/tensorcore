#!/usr/bin/env python3
"""Probe chipStar/HIP + OpenCL/SPIR-V toolchain readiness.

The probe is intentionally diagnostic: it never installs anything and it does
not require tensorcore to build. It records the tools and CMake packages a host
needs before scripts/ci_hip_smoke.sh can graduate from "not built" to real
HIP/chipStar runtime evidence.
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION_TIMEOUT_SEC = 8


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


def split_env_paths(value: str | None) -> list[str]:
    if not value:
        return []
    sep = ";" if ";" in value else os.pathsep
    return [part for part in value.split(sep) if part]


def normalize_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        try:
            resolved = str(pathlib.Path(item).expanduser().resolve())
        except Exception:
            resolved = item
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def candidate_prefixes() -> list[str]:
    prefixes: list[str] = []
    for name in ("TC_HIP_PREFIX", "CHIPSTAR_HOME", "HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(name)
        if value:
            prefixes.append(value)
    prefixes.extend(split_env_paths(os.environ.get("CMAKE_PREFIX_PATH")))
    home = os.environ.get("HOME")
    if home:
        prefixes.append(str(pathlib.Path(home) / "chipstar-install"))
    prefixes.extend(["/opt/chipstar", "/opt/rocm", "/usr/local", "/usr"])
    return normalize_paths(prefixes)


def find_tool(name: str, prefixes: list[str]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    suffixes = [("bin", name), ("llvm", "bin", name)]
    for prefix in prefixes:
        root = pathlib.Path(prefix)
        for parts in suffixes:
            candidate = root.joinpath(*parts)
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def command_version(path: str | None, args: list[str]) -> dict[str, Any]:
    if not path:
        return {
            "path": None,
            "available": False,
            "returncode": None,
            "version": None,
        }
    try:
        result = subprocess.run(
            [path, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=VERSION_TIMEOUT_SEC,
        )
        output = "\n".join(result.stdout.strip().splitlines()[:12])
        return {
            "path": path,
            "available": result.returncode == 0,
            "returncode": result.returncode,
            "version": output,
        }
    except subprocess.TimeoutExpired:
        return {
            "path": path,
            "available": False,
            "returncode": None,
            "version": "version command timed out",
        }
    except OSError as exc:
        return {
            "path": path,
            "available": False,
            "returncode": None,
            "version": str(exc),
        }


def collect_tools(prefixes: list[str]) -> dict[str, Any]:
    commands = {
        "cmake": ["--version"],
        "hipcc": ["--version"],
        "clang": ["--version"],
        "clang++": ["--version"],
        "llvm-spirv": ["--version"],
        "spirv-val": ["--version"],
        "clinfo": ["-l"],
    }
    return {
        name: command_version(find_tool(name, prefixes), args)
        for name, args in commands.items()
    }


def cmake_search_roots(prefix: pathlib.Path) -> list[pathlib.Path]:
    roots = [
        prefix / "lib" / "cmake",
        prefix / "lib64" / "cmake",
        prefix / "share" / "cmake",
        prefix / "share",
    ]
    return [root for root in roots if root.is_dir()]


def find_cmake_configs(prefixes: list[str], needles: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    package_terms = {
        needle.replace("config", "").replace("-config", "").replace("-", "")
        for needle in needles
    }
    for prefix in prefixes:
        root = pathlib.Path(prefix)
        if not root.is_dir():
            continue
        for search_root in cmake_search_roots(root):
            try:
                children = list(search_root.iterdir())
            except OSError:
                continue
            candidates: list[pathlib.Path] = []
            for child in children:
                name = child.name.lower().replace("-", "")
                if any(term and term in name for term in package_terms):
                    candidates.append(child)
            matches: list[pathlib.Path] = []
            for candidate in candidates:
                if candidate.is_file():
                    matches.append(candidate)
                    continue
                try:
                    matches.extend(candidate.rglob("*Config.cmake"))
                    matches.extend(candidate.rglob("*-config.cmake"))
                except OSError:
                    continue
            for path in matches:
                lowered = path.name.lower()
                if any(needle in lowered for needle in needles):
                    resolved = str(path.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(resolved)
    return found


def collect_cmake_packages(prefixes: list[str]) -> dict[str, list[str]]:
    return {
        "hip": find_cmake_configs(prefixes, ("hipconfig", "hip-config")),
        "hipblas": find_cmake_configs(prefixes, ("hipblasconfig", "hipblas-config")),
    }


def collect_runtime(tools: dict[str, Any]) -> dict[str, Any]:
    icd_candidates = [
        pathlib.Path("/etc/OpenCL/vendors"),
        pathlib.Path("/usr/share/OpenCL/vendors"),
        pathlib.Path("/usr/local/etc/OpenCL/vendors"),
    ]
    icd_files: list[str] = []
    for directory in icd_candidates:
        if not directory.is_dir():
            continue
        try:
            icd_files.extend(str(path) for path in sorted(directory.glob("*.icd")))
        except OSError:
            pass

    opencl_library = ctypes.util.find_library("OpenCL")
    level_zero_library = (
        ctypes.util.find_library("ze_loader")
        or ctypes.util.find_library("ze_loader.so")
        or ctypes.util.find_library("ze_loader.dll")
    )

    clinfo = tools.get("clinfo", {})
    return {
        "opencl_library": opencl_library,
        "opencl_icd_files": icd_files,
        "level_zero_library": level_zero_library,
        "clinfo_available": bool(clinfo.get("path")),
        "clinfo_devices": clinfo.get("version") if clinfo.get("available") else None,
    }


def path_hints(prefixes: list[str]) -> list[str]:
    hints: list[str] = []
    preferred = os.environ.get("TC_HIP_PREFIX")
    if not preferred:
        for prefix in prefixes:
            root = pathlib.Path(prefix)
            if (root / "bin" / "hipcc").exists() or "chipstar" in root.name.lower():
                preferred = prefix
                break
    if preferred:
        hints.append(f"export TC_HIP_PREFIX={preferred}")
        hints.append(f"export PATH={preferred}/bin:$PATH")
        hints.append(f"export CMAKE_PREFIX_PATH={preferred}:$CMAKE_PREFIX_PATH")
        lib = pathlib.Path(preferred) / "lib"
        if lib.exists():
            hints.append(f"export LD_LIBRARY_PATH={lib}:$LD_LIBRARY_PATH")
    else:
        hints.append("set TC_HIP_PREFIX to the chipStar install prefix")
    return hints


def collect_evidence(root: str | pathlib.Path = ROOT) -> dict[str, Any]:
    del root  # Keep the callable stable for scripts that pass TC_ROOT.
    prefixes = candidate_prefixes()
    tools = collect_tools(prefixes)
    packages = collect_cmake_packages(prefixes)
    runtime = collect_runtime(tools)

    readiness = {
        "hip_runtime_config": bool(packages["hip"]),
        "hipcc": bool(tools["hipcc"].get("path")),
        "spirv_translator": bool(tools["llvm-spirv"].get("path")),
        "opencl_or_level_zero": bool(
            runtime["opencl_library"]
            or runtime["opencl_icd_files"]
            or runtime["level_zero_library"]
        ),
        "hipblas_config": bool(packages["hipblas"]),
    }
    missing: list[str] = []
    if not readiness["hip_runtime_config"]:
        missing.append("hip CMake config")
    if not readiness["hipcc"]:
        missing.append("hipcc")
    if not readiness["spirv_translator"]:
        missing.append("llvm-spirv")
    if not readiness["opencl_or_level_zero"]:
        missing.append("OpenCL or Level Zero runtime")
    if not readiness["hipblas_config"]:
        missing.append("hipBLAS CMake config")

    if not missing:
        status = "ready_for_hip_gemm"
    elif all(readiness[key] for key in ("hip_runtime_config", "hipcc", "spirv_translator", "opencl_or_level_zero")):
        status = "runtime_only_no_hipblas"
    else:
        status = "missing_requirements"

    return {
        "schema_version": 1,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--short")),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "environment": {
            "TC_HIP_PREFIX": os.environ.get("TC_HIP_PREFIX"),
            "CHIPSTAR_HOME": os.environ.get("CHIPSTAR_HOME"),
            "HIP_PATH": os.environ.get("HIP_PATH"),
            "ROCM_PATH": os.environ.get("ROCM_PATH"),
            "CMAKE_PREFIX_PATH": os.environ.get("CMAKE_PREFIX_PATH"),
        },
        "prefixes": prefixes,
        "tools": tools,
        "cmake_packages": packages,
        "runtime": runtime,
        "readiness": {
            **readiness,
            "status": status,
            "missing": missing,
        },
        "path_hints": path_hints(prefixes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=pathlib.Path, help="Write evidence JSON here")
    parser.add_argument("--require-build-toolchain", action="store_true")
    parser.add_argument("--require-spirv-runtime", action="store_true")
    parser.add_argument("--require-hipblas", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    evidence = collect_evidence()
    if args.json:
        args.json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readiness = evidence["readiness"]
    print(
        "HIP toolchain probe: "
        f"status={readiness['status']} missing={','.join(readiness['missing']) or 'none'}"
    )
    for hint in evidence["path_hints"]:
        print(f"hint: {hint}")

    errors: list[str] = []
    if args.require_build_toolchain and not (
        readiness["hip_runtime_config"] and readiness["hipcc"]
    ):
        errors.append("--require-build-toolchain needs hip CMake config and hipcc")
    if args.require_spirv_runtime and not (
        readiness["spirv_translator"] and readiness["opencl_or_level_zero"]
    ):
        errors.append("--require-spirv-runtime needs llvm-spirv and OpenCL/Level Zero")
    if args.require_hipblas and not readiness["hipblas_config"]:
        errors.append("--require-hipblas needs hipBLAS CMake config")
    if args.require_ready and readiness["status"] != "ready_for_hip_gemm":
        errors.append("--require-ready needs ready_for_hip_gemm")
    if errors:
        for error in errors:
            print(f"HIP toolchain probe failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
