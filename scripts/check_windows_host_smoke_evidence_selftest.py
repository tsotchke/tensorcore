#!/usr/bin/env python3
"""Fixture tests for the Windows host smoke evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_windows_host_smoke_evidence.py"
TEST_HEAD = "abc123"


def evidence() -> dict[str, Any]:
    return {
        "schema": "tensorcore.windows_host_smoke.evidence.v1",
        "schema_version": 1,
        "runtime_status": "passed",
        "git_head": TEST_HEAD,
        "git_dirty": False,
        "ref": "master",
        "repo": "src/tensorcore",
        "remote_url": "https://github.com/tsotchke/tensorcore.git",
        "host": {
            "computer_name": "DESKTOP-JACK-BLUPC",
            "user": "tsotchke",
            "os": "Microsoft Windows 11 Pro",
        },
        "update": {"reset": False},
        "bootstrap": {
            "ran": True,
            "install_requested": False,
            "skip_python": False,
        },
    }


def write_json(directory: pathlib.Path, name: str, data: dict[str, Any]) -> pathlib.Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checker(path: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_passes(path: pathlib.Path, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(path: pathlib.Path, needle: str, *args: str) -> None:
    result = run_checker(path, *args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        good = write_json(directory, "windows.json", evidence())
        assert_passes(good)
        assert_passes(
            good,
            "--git-head", TEST_HEAD,
            "--require-windows",
            "--require-clean-head",
            "--require-python",
        )

        no_smoke = copy.deepcopy(evidence())
        no_smoke["runtime_status"] = "skipped_no_smoke"
        no_smoke["bootstrap"]["ran"] = False
        no_smoke_path = write_json(directory, "no-smoke.json", no_smoke)
        assert_passes(no_smoke_path)
        assert_fails(no_smoke_path, "--require-windows needs passed evidence", "--require-windows")

        dirty = copy.deepcopy(evidence())
        dirty["git_dirty"] = True
        dirty_path = write_json(directory, "dirty.json", dirty)
        assert_fails(
            dirty_path,
            "Windows evidence must be from a clean git tree",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

        stale = copy.deepcopy(evidence())
        stale["git_head"] = "stale"
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails(
            stale_path,
            "Windows evidence git_head mismatch",
            "--git-head", TEST_HEAD,
            "--require-clean-head",
        )

        skipped_python = copy.deepcopy(evidence())
        skipped_python["bootstrap"]["skip_python"] = True
        skipped_python_path = write_json(directory, "skipped-python.json", skipped_python)
        assert_fails(
            skipped_python_path,
            "--require-python cannot accept skip_python=true",
            "--require-python",
        )

        non_windows = copy.deepcopy(evidence())
        non_windows["host"]["os"] = "Darwin"
        non_windows_path = write_json(directory, "non-windows.json", non_windows)
        assert_fails(non_windows_path, "host.os must identify Windows")

    print("Windows host smoke evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
