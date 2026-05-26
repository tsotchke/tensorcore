#!/usr/bin/env python3
"""Selftests for scripts/check_mesh_git_access.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_mesh_git_access.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("check_mesh_git_access_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_parses_head_and_ref() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "git@github.com:Tsotchke-Corporation/GeoRefineInternal.git",
        "--resource",
        "cosbox:cuda3090",
        "--ref",
        "HEAD",
        "--json",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "abc123\tHEAD\n", "")

    mod.subprocess.run = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is True
    assert payload["reason"] == "ok"
    assert payload["resource"] == "cosbox:cuda3090"
    assert payload["head"] == "abc123"
    assert payload["matched_ref"] == "HEAD"


def test_publickey_denied_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "old-donkey",
        "--repo-url",
        "git@github.com:Tsotchke-Corporation/semiclassical_qllm.git",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 128, "", "git@github.com: Permission denied (publickey).\n")

    mod.subprocess.run = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_publickey_denied"


def test_https_prompt_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/Tsotchke-Corporation/GeoRefineInternal.git",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 128, "", "fatal: could not read Username for 'https://github.com'\n")

    mod.subprocess.run = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_credentials_required"


def test_timeout_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "git@example.com:repo.git",
        "--resource",
        "cosbox:cuda3090",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["ssh"], 1.0)

    mod.subprocess.run = fake_run
    payload = mod.run_check(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_access_timeout"
    assert payload["resource"] == "cosbox:cuda3090"


def main() -> int:
    test_success_parses_head_and_ref()
    test_publickey_denied_is_specific()
    test_https_prompt_is_specific()
    test_timeout_fails_closed()
    print("mesh git access selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
