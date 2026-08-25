#!/usr/bin/env python3
"""Selftests for scripts/start_qllm_olddonkey_precompute_chain.py."""

from __future__ import annotations

import inspect
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_qllm_olddonkey_precompute_chain.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("start_qllm_chain_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_script_clones_and_launches_tmux_chain() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "old-donkey",
        "--repo-dir",
        "/data/qllm/semiclassical_qllm",
        "--python-bin",
        "/data/venv/qllm/bin/python",
        "--shards",
        "ds07_shard02",
        "ds07_shard03",
        "--print-script",
    ])
    script = mod.render_remote_script(args)
    assert "git clone --filter=blob:none --branch \"$ref\" \"$repo_url\" \"$repo_dir\"" in script
    assert "git -C \"$repo_dir\" pull --ff-only origin \"$ref\"" in script
    assert "dirty_checkout" in script
    assert "tmux new-session -d -s \"$session\"" in script
    assert "scripts/precompute_teacher_logits.py" in script
    assert "--teacher \"$teacher\"" in script
    assert "--batch \"$batch\"" in script
    assert "ds07_shard02 ds07_shard03" in script
    assert "/data/qllm/runs/start_olddonkey_precompute_chain.sh" not in script
    assert "/data/qllm/runs/precompute_olddonkey.sh" not in script


def test_preflight_mode_checks_environment_without_launching() -> None:
    mod = load_module()
    args = mod.parse_args([
        "--target",
        "old-donkey",
        "--preflight-only",
        "--print-script",
    ])
    script = mod.render_remote_script(args)
    assert "preflight_only=1" in script
    assert "precompute_script_missing" in script
    assert "nvidia_smi_not_found" in script
    assert "shard_missing:$name" in script
    assert "import torch" in script
    assert "import transformers" in script
    assert script.index("git clone --filter=blob:none --branch") < script.index(
        "precompute_teacher_logits.py.*--shard $shard_dir/"
    )
    assert "emit true preflight_ok" in script


def test_remote_runner_closes_stdin_and_masks_cleanup_timeout() -> None:
    mod = load_module()
    source = inspect.getsource(mod.run_remote_script) + inspect.getsource(mod.run_remote_inline_script)
    assert "stdin=subprocess.DEVNULL" in source
    assert "except subprocess.TimeoutExpired" in source


def test_success_payload_passes_through() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": mod.SCHEMA,
            "ok": True,
            "reason": "started_pending",
            "resource": "old-donkey:cuda3050",
            "session": "qllm-precompute-chain",
        }
        return subprocess.CompletedProcess([], 0, "remote log\n" + json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is True
    assert payload["target"] == "old-donkey"
    assert payload["session"] == "qllm-precompute-chain"


def test_scp_failure_falls_back_to_chunked_ssh() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--preflight-only", "--json"])
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
                "resource": "old-donkey:cuda3050",
                "session": "qllm-precompute-chain",
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
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            '{"schema":"tensorcore.qllm_olddonkey_precompute_chain.start.v1","ok":false,"reason":"tmux_not_found","resource":"old-donkey:cuda3050"}',
            "missing tmux",
        )

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "tmux_not_found"
    assert "missing tmux" in payload["stderr_tail"]


def test_clone_publickey_failure_is_specific() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            1,
            '{"schema":"tensorcore.qllm_olddonkey_precompute_chain.start.v1","ok":false,"reason":"clone_failed","resource":"old-donkey:cuda3050"}',
            "git@github.com: Permission denied (publickey).",
        )

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "git_publickey_denied"


def test_schema_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "wrong", "ok": True, "resource": "old-donkey:cuda3050"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "invalid_start_schema"


def test_resource_mismatch_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": mod.SCHEMA, "ok": True, "resource": "cosbox:cuda3090"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "start_resource_mismatch"


def test_timeout_fails_closed() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["ssh"], 1.0)

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "start_timeout"
    assert payload["resource"] == "old-donkey:cuda3050"


def test_remote_transport_failure_carries_resource() -> None:
    mod = load_module()
    args = mod.parse_args(["--target", "old-donkey", "--json"])

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 255, "", "Connection closed")

    mod.subprocess.run = fake_run
    payload = mod.run_start(args)
    assert payload["ok"] is False
    assert payload["reason"] == "remote_start_failed"
    assert payload["resource"] == "old-donkey:cuda3050"


def main() -> int:
    test_remote_script_clones_and_launches_tmux_chain()
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
    print("qLLM old-donkey precompute chain starter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
