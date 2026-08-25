#!/usr/bin/env python3
"""Selftests for remote_tsotchke_arbiter.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import os
import pathlib
from contextlib import redirect_stderr
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    path = ROOT / "scripts" / "remote_tsotchke_arbiter.py"
    loader = importlib.machinery.SourceFileLoader("remote_tsotchke_arbiter_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_env(values: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in values.items():
        previous[key] = os.environ.get(key)
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_remote_argv_quotes_json_for_single_remote_shell_argument() -> None:
    mod = load_script()
    previous = with_env(
        {
            "TC_REMOTE_ARBITER_HOST": "arbiter.example",
            "TC_REMOTE_ARBITER_BIN": "/opt/computer_mesh/tsotchke/bin/tsotchke-arbiter",
            "TC_REMOTE_ARBITER_KEY": "",
        }
    )
    try:
        argv = mod.build_ssh_argv(
            [
                "claim",
                "cosbox:cuda3090",
                "--metadata-json",
                '{"job_id":"geo refine","resource":"cosbox:cuda3090"}',
            ]
        )
    finally:
        restore_env(previous)
    assert argv[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    assert argv[-2] == "arbiter.example"
    remote_cmd = argv[-1]
    assert remote_cmd.startswith("/opt/computer_mesh/tsotchke/bin/tsotchke-arbiter claim")
    assert "'{\"job_id\":\"geo refine\",\"resource\":\"cosbox:cuda3090\"}'" in remote_cmd


def test_remote_host_and_binary_are_required() -> None:
    mod = load_script()
    previous = with_env(
        {
            "TC_REMOTE_ARBITER_HOST": "",
            "TC_REMOTE_ARBITER_BIN": "",
            "TC_REMOTE_ARBITER_KEY": "",
        }
    )
    try:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = mod.main(["status", "--json"])
    finally:
        restore_env(previous)
    assert result == 2
    assert "TC_REMOTE_ARBITER_HOST must be set" in stderr.getvalue()


def main() -> int:
    test_remote_argv_quotes_json_for_single_remote_shell_argument()
    test_remote_host_and_binary_are_required()
    print("remote tsotchke arbiter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
