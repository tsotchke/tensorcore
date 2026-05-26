#!/usr/bin/env python3
"""Run focused fallback GEMM smokes and emit ICC-readable runtime evidence."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "tensorcore.fallback_runtime_evidence.v1"
FORMAT_VERSION = 1

CHECKS = {
    "accelerate_f32": {
        "test": "test_gemm_f32",
        "label": "accelerate_f32_fallback",
        "functions": {
            "lib/fallback/accelerate_gemm.c": ["tc_accelerate_gemm_f32"],
        },
    },
    "mps_f32": {
        "test": "test_gemm_f32",
        "label": "mps_f32_fallback",
        "functions": {
            "lib/fallback/mps_gemm.mm": [
                "to_mps_dtype",
                "effective_lda",
                "effective_ldb",
                "effective_ldc",
                "tc_mps_gemm",
            ],
        },
    },
    "mps_bf16": {
        "test": "test_gemm_bf16",
        "label": "mps_bf16_sw_fallback",
        "functions": {
            "lib/fallback/mps_gemm.mm": [
                "bf16_to_f32",
                "f32_to_bf16",
                "effective_lda",
                "effective_ldb",
                "effective_ldc",
                "bf16_via_fp32",
                "tc_mps_gemm",
            ],
        },
    },
    "mps_i8": {
        "test": "test_gemm_i8",
        "label": "mps_i8_sw_fallback",
        "functions": {
            "lib/fallback/mps_gemm.mm": [
                "effective_lda",
                "effective_ldb",
                "effective_ldc",
                "i8_via_fp32",
                "tc_mps_gemm",
            ],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument(
        "--evidence-path",
        type=pathlib.Path,
        default=None,
        help="Default: $BUILD_DIR/fallback_runtime_evidence.json",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero unless all focused fallback checks pass.",
    )
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


def run_test(build_dir: pathlib.Path, name: str) -> dict[str, Any]:
    exe = build_dir / "tests" / name
    if not exe.exists():
        return {
            "status": "missing",
            "cmd": [str(exe)],
            "rc": None,
            "stdout_tail": "",
            "stderr_tail": f"{exe} does not exist",
        }

    env = os.environ.copy()
    metallib = build_dir / "tensorcore.metallib"
    if metallib.exists():
        env.setdefault("TC_METALLIB", str(metallib))

    proc = subprocess.run(
        [str(exe)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "cmd": [str(exe)],
        "rc": proc.returncode,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def function_line(rel_path: str, name: str) -> int:
    path = ROOT / rel_path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 1

    c_ref = re.compile(rf"\b{re.escape(name)}\s*\(")
    control_prefixes = ("if", "for", "while", "switch", "return")
    for index, line in enumerate(lines, start=1):
        if not c_ref.search(line):
            continue
        stripped = line.strip()
        prefix = stripped.split(name, 1)[0].strip()
        if "=" in prefix or prefix.startswith(control_prefixes):
            continue
        for lookahead in lines[index - 1 : min(len(lines), index + 12)]:
            if "{" in lookahead:
                return index
            if ";" in lookahead:
                break
    return 1


def add_function(files: dict[str, Any], rel_path: str, name: str) -> None:
    line = function_line(rel_path, name)
    entry = files.setdefault(rel_path, {"executed_lines": [], "functions": {}})
    if line not in entry["executed_lines"]:
        entry["executed_lines"].append(line)
    entry["functions"][name] = {
        "start_line": line,
        "executed_lines": [line],
    }


def build_evidence(build_dir: pathlib.Path) -> dict[str, Any]:
    tests = {
        name: run_test(build_dir, name)
        for name in sorted({str(spec["test"]) for spec in CHECKS.values()})
    }
    files: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    for check_name, spec in CHECKS.items():
        test_name = str(spec["test"])
        test = tests[test_name]
        output = "\n".join([test.get("stdout_tail", ""), test.get("stderr_tail", "")])
        label_seen = str(spec["label"]) in output
        status = "passed" if test["status"] == "passed" and label_seen else test["status"]
        if test["status"] == "passed" and not label_seen:
            status = "failed_label_missing"

        checks[check_name] = {
            "status": status,
            "test": test_name,
            "label": spec["label"],
            "label_seen": label_seen,
        }
        if status == "passed":
            for rel_path, names in spec["functions"].items():
                for name in names:
                    add_function(files, rel_path, name)

    for entry in files.values():
        entry["executed_lines"].sort()

    required_functions = sorted(
        {
            f"{rel_path}:{name}"
            for spec in CHECKS.values()
            for rel_path, names in spec["functions"].items()
            for name in names
        }
    )
    covered_functions = sorted(
        f"{rel_path}:{name}"
        for rel_path, entry in files.items()
        for name in entry.get("functions", {})
    )
    missing_functions = sorted(set(required_functions) - set(covered_functions))
    status = "passed" if all(item["status"] == "passed" for item in checks.values()) else "failed"

    metallib = build_dir / "tensorcore.metallib"
    return {
        "schema": SCHEMA,
        "meta": {
            "format": FORMAT_VERSION,
            "source": "tensorcore_fallback_runtime_smoke",
            "git_head": git_value("rev-parse", "HEAD"),
            "git_dirty": git_dirty(),
        },
        "status": status,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "paths": {
            "build_dir": str(build_dir),
            "metallib": str(metallib) if metallib.exists() else None,
        },
        "checks": checks,
        "tests": tests,
        "files": files,
        "summary": {
            "checks_passed": status == "passed",
            "required_functions": required_functions,
            "covered_functions": covered_functions,
            "missing_functions": missing_functions,
        },
    }


def main() -> int:
    args = parse_args()
    build_dir = args.build_dir.resolve()
    evidence_path = args.evidence_path or (build_dir / "fallback_runtime_evidence.json")
    evidence = build_evidence(build_dir)

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        print(
            "fallback runtime smoke "
            f"{evidence['status']}: evidence={evidence_path} "
            f"covered={len(evidence['summary']['covered_functions'])}/"
            f"{len(evidence['summary']['required_functions'])}"
        )

    if args.require_pass and evidence["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
