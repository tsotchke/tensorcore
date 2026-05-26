#!/usr/bin/env python3
"""Selftests for scripts/mesh_deploy_git_checkout.py."""

from __future__ import annotations

import inspect
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mesh_deploy_git_checkout.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("mesh_deploy_git_checkout_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_script_clones_and_fast_forwards() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
        "--ref",
        "master",
        "--require-clean",
        "--print-script",
    ])
    script = mod.render_remote_script(args)
    assert "git clone --filter=blob:none --branch \"$ref\" \"$repo_url\" \"$repo_dir\"" in script
    assert "emit false clone_failed" in script
    assert "emit false dirty_checkout" in script
    assert "git -C \"$repo_dir\" pull --ff-only origin \"$ref\"" in script
    assert "require_clean=1" in script
    assert "repo_dir=\"$HOME/${repo_dir#\\~/}\"" in script


def test_remote_runner_closes_stdin_and_masks_cleanup_timeout() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_script) + inspect.getsource(mod.run_remote_inline_script)
    assert "stdin=subprocess.DEVNULL" in source
    assert "except subprocess.TimeoutExpired" in source


def test_success_payload_passes_through() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
        "--json",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "repo_dir": "/home/tyr/src/tensorcore",
            "repo_url": "https://github.com/tsotchke/tensorcore.git",
            "head": "abc123",
        }
        return subprocess.CompletedProcess([], 0, "remote log\n" + json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is True
    assert payload["target"] == "cosbox"
    assert payload["head"] == "abc123"


def test_scp_failure_falls_back_to_chunked_ssh() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
        "--json",
    ])
    calls: list[object] = []

    def fake_run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if isinstance(cmd, list) and cmd and cmd[0] == "scp":
            return subprocess.CompletedProcess(cmd, 255, "", "scp closed")
        if isinstance(cmd, list) and len(cmd) >= 3 and "base64 -d" in str(cmd[2]):
            payload = {
                "schema": mod.SCHEMA,
                "ok": True,
                "repo_dir": "/home/tyr/src/tensorcore",
                "repo_url": "https://github.com/tsotchke/tensorcore.git",
                "head": "abc123",
            }
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is True
    assert any(isinstance(cmd, list) and cmd and cmd[0] == "scp" for cmd in calls)
    assert any(isinstance(cmd, list) and len(cmd) >= 3 and "printf '%s'" in str(cmd[2]) for cmd in calls)


def test_remote_failure_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": mod.SCHEMA,
            "ok": False,
            "reason": "dirty_checkout",
            "repo_url": "https://github.com/tsotchke/tensorcore.git",
        }
        return subprocess.CompletedProcess([], 3, json.dumps(payload), "dirty tree")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is False
    assert payload["reason"] == "dirty_checkout"
    assert "dirty tree" in payload["stderr_tail"]


def test_clone_publickey_failure_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "git@github.com:Tsotchke-Corporation/private.git",
        "--repo-dir",
        "~/src/private",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            json.dumps({
                "schema": mod.SCHEMA,
                "ok": False,
                "reason": "clone_failed",
                "repo_url": "git@github.com:Tsotchke-Corporation/private.git",
            }),
            "git@github.com: Permission denied (publickey).",
        )

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_publickey_denied"


def test_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "wrong", "ok": True, "repo_url": "https://github.com/tsotchke/tensorcore.git"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_deploy_schema"


def test_repo_url_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "https://github.com/tsotchke/tensorcore.git",
        "--repo-dir",
        "~/src/tensorcore",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "repo_url": "https://example.com/other.git"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is False
    assert payload["reason"] == "deploy_repo_url_mismatch"


def test_publickey_stderr_without_payload_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-url",
        "git@github.com:Tsotchke-Corporation/private.git",
        "--repo-dir",
        "~/src/private",
    ])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "git@github.com: Permission denied (publickey).")

    mod.subprocess.run = fake_run
    payload = mod.run_deploy(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_publickey_denied"


def main() -> int:
    test_remote_script_clones_and_fast_forwards()
    test_remote_runner_closes_stdin_and_masks_cleanup_timeout()
    test_success_payload_passes_through()
    test_scp_failure_falls_back_to_chunked_ssh()
    test_remote_failure_fails_closed()
    test_clone_publickey_failure_is_specific()
    test_schema_mismatch_fails_closed()
    test_repo_url_mismatch_fails_closed()
    test_publickey_stderr_without_payload_is_specific()
    print("mesh git deploy selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
