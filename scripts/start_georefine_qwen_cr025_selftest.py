#!/usr/bin/env python3
"""Selftests for scripts/start_georefine_qwen_cr025.py."""

from __future__ import annotations

import inspect
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_georefine_qwen_cr025.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("start_georefine_qwen_cr025_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_script_clones_clean_checkout_and_runs_supervisor() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--repo-dir",
        "~/src/georefine",
        "--python-bin",
        "bin/python",
        "--run-dir",
        "/runs/georefine/qwen",
        "--print-script",
    ])
    script = mod.render_remote_script(args)
    assert "git clone --filter=blob:none --branch \"$ref\" \"$repo_url\" \"$repo_dir\"" in script
    assert "git -C \"$repo_dir\" pull --ff-only origin \"$ref\"" in script
    assert "dirty_checkout" in script
    assert "experiments.georefine.m2_supervised_run" in script
    assert "experiments.georefine.m2_compress" in script
    assert "--model Qwen/Qwen3.5-0.8B" in script
    assert "--compression-ratio 0.25" in script
    assert "--embedding-rank 1024" in script
    assert "--target-kl-kd-steps 2048" in script
    assert "--fail-on-verdict LOSSY" in script
    assert "--trust-remote-code" in script


def test_preflight_mode_checks_environment_without_launching() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "cosbox",
        "--preflight-only",
        "--print-script",
    ])
    script = mod.render_remote_script(args)
    assert "preflight_only=1" in script
    assert "python_env_not_ready" in script
    assert "calibration_text_missing" in script
    assert "eval_text_missing" in script
    assert "import experiments.georefine.m2_compress" in script
    assert "import experiments.georefine.m2_supervised_run" in script
    assert "emit true preflight_ok" in script


def test_remote_runner_closes_stdin_and_masks_cleanup_timeout() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_script) + inspect.getsource(mod.run_remote_inline_script)
    assert "stdin=subprocess.DEVNULL" in source
    assert "except subprocess.TimeoutExpired" in source


def test_success_payload_passes_through() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "reason": "started",
            "resource": "cosbox:cuda3090",
            "pid": 1234,
        }
        return subprocess.CompletedProcess([], 0, "remote log\n" + json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is True
    assert payload["reason"] == "started"
    assert payload["target"] == "cosbox"


def test_scp_failure_falls_back_to_chunked_ssh() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--preflight-only", "--json"])
    calls: list[object] = []

    def fake_run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if isinstance(cmd, list) and cmd and cmd[0] == "scp":
            return subprocess.CompletedProcess(cmd, 255, "", "scp closed")
        if isinstance(cmd, list) and len(cmd) >= 3 and "base64 -d" in str(cmd[2]):
            payload = {
                "schema": mod.SCHEMA,
                "ok": True,
                "reason": "preflight_ok",
                "resource": "cosbox:cuda3090",
                "pid": 0,
            }
            return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is True
    assert payload["reason"] == "preflight_ok"
    assert any(isinstance(cmd, list) and cmd and cmd[0] == "scp" for cmd in calls)
    assert any(isinstance(cmd, list) and len(cmd) >= 3 and "printf '%s'" in str(cmd[2]) for cmd in calls)


def test_remote_failure_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            '{"schema":"tensorcore.georefine_qwen_cr025.start.v1","ok":false,"reason":"dirty_checkout","resource":"cosbox:cuda3090"}',
            "dirty tree",
        )

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "dirty_checkout"
    assert "dirty tree" in payload["stderr_tail"]


def test_clone_publickey_failure_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            '{"schema":"tensorcore.georefine_qwen_cr025.start.v1","ok":false,"reason":"clone_failed","resource":"cosbox:cuda3090"}',
            "git@github.com: Permission denied (publickey).",
        )

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_publickey_denied"


def test_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "wrong", "ok": True, "resource": "cosbox:cuda3090"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_start_schema"


def test_resource_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "resource": "old-donkey:cuda3050"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "start_resource_mismatch"


def test_timeout_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["ssh"], 1.0)

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "start_timeout"
    assert payload["resource"] == "cosbox:cuda3090"


def test_remote_transport_failure_carries_resource() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "cosbox", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 255, "", "Connection closed")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "remote_start_failed"
    assert payload["resource"] == "cosbox:cuda3090"


def main() -> int:
    test_remote_script_clones_clean_checkout_and_runs_supervisor()
    test_preflight_mode_checks_environment_without_launching()
    test_remote_runner_closes_stdin_and_masks_cleanup_timeout()
    test_success_payload_passes_through()
    test_scp_failure_falls_back_to_chunked_ssh()
    test_remote_failure_fails_closed()
    test_clone_publickey_failure_is_specific()
    test_schema_mismatch_fails_closed()
    test_resource_mismatch_fails_closed()
    test_timeout_fails_closed()
    test_remote_transport_failure_carries_resource()
    print("GeoRefine Qwen CR025 starter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
