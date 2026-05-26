#!/usr/bin/env python3
"""Selftests for scripts/check_mesh_resource_preflights.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_mesh_resource_preflights.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("check_mesh_resource_preflights_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jobs(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "jobs.json"
    path.write_text(
        json.dumps({
            "schema": "tensorcore.mesh_resource_jobs.v1",
            "jobs": [
                {
                    "id": "paused-good",
                    "resource": "cosbox:cuda3090",
                    "desired_state": "paused",
                    "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
                },
                {
                    "id": "running-skip",
                    "resource": "old-donkey:cuda3050",
                    "desired_state": "running",
                    "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
                },
                {
                    "id": "paused-explicit-only",
                    "resource": "jack-blupc:cuda3060",
                    "desired_state": "paused",
                    "preflight_cmd": ["python3", "scripts/check_windows_persistent_launch.py", "--json"],
                    "metadata": {"preflight_default": False},
                },
            ],
        }),
        encoding="utf-8",
    )
    return path


def test_selects_paused_preflights_by_default() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = mod.load_jobs(write_jobs(pathlib.Path(tmp)))
    selected = mod.select_jobs(jobs, [], False)
    assert [job["id"] for job in selected] == ["paused-good"]


def test_explicit_preflight_selection_includes_default_opt_out() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = mod.load_jobs(write_jobs(pathlib.Path(tmp)))
    selected = mod.select_jobs(jobs, ["paused-explicit-only"], False)
    assert [job["id"] for job in selected] == ["paused-explicit-only"]


def test_skipped_default_job_ids_reports_default_opt_out() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = mod.load_jobs(write_jobs(pathlib.Path(tmp)))
    assert mod.skipped_default_job_ids(jobs, [], False) == ["paused-explicit-only"]
    assert mod.skipped_default_job_ids(jobs, ["paused-explicit-only"], False) == []


def test_checked_in_default_preflights_include_launchable_paused_rows() -> None:
    mod = load_module()
    jobs = mod.load_jobs(mod.DEFAULT_JOBS)
    selected = {job["id"] for job in mod.select_jobs(jobs, [], False)}
    assert {
        "georefine-m2-cosbox",
        "old-donkey-precompute-chain",
        "jack-cuda3060-smoke",
    }.issubset(selected)


def test_missing_requested_job_ids_fail_closed() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = mod.load_jobs(write_jobs(pathlib.Path(tmp)))
    assert mod.missing_job_ids(jobs, ["paused-good", "missing-job"]) == ["missing-job"]


def test_run_preflight_passes_json_payload() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": "unit.preflight.v1",
            "ok": True,
            "reason": "preflight_ok",
            "resource": "cosbox:cuda3090",
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-good",
            "resource": "cosbox:cuda3090",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is True
    assert result["reason"] == "preflight_ok"
    assert result["json"]["ok"] is True
    assert pathlib.Path(result["cmd"][1]).is_absolute()


def test_command_resolution_preserves_remote_tilde_arguments() -> None:
    mod = load_module()
    argv = mod.command([
        "python3",
        "scripts/start_georefine_qwen_cr025.py",
        "--repo-dir",
        "~/src/georefine",
    ])
    assert pathlib.Path(argv[1]).is_absolute()
    assert argv[3] == "~/src/georefine"


def test_run_preflight_fails_with_payload_reason() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": "unit.preflight.v1",
            "ok": False,
            "reason": "git_publickey_denied",
            "resource": "old-donkey:cuda3050",
        }
        return subprocess.CompletedProcess([], 1, json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-bad",
            "resource": "old-donkey:cuda3050",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "git_publickey_denied"


def test_run_preflight_nonzero_exit_with_ok_payload_is_explicit() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "schema": "unit.preflight.v1",
            "ok": True,
            "reason": "preflight_ok",
            "resource": "cosbox:cuda3090",
        }
        return subprocess.CompletedProcess([], 1, json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-bad",
            "resource": "cosbox:cuda3090",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "preflight_nonzero_exit"


def test_run_preflight_requires_json_payload() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "plain text success\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-bad",
            "resource": "cosbox:cuda3090",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "invalid_preflight_json"


def test_run_preflight_requires_schema() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"ok": True, "resource": "cosbox:cuda3090"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-bad",
            "resource": "cosbox:cuda3090",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "invalid_preflight_schema"


def test_run_preflight_rejects_resource_mismatch() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"schema": "unit.preflight.v1", "ok": True, "resource": "other:cuda"}
        return subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {
            "id": "paused-bad",
            "resource": "cosbox:cuda3090",
            "preflight_cmd": ["python3", "scripts/check_mesh_git_access.py", "--json"],
        },
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "preflight_resource_mismatch"


def test_timeout_fails_closed() -> None:
    mod = load_module()

    def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["python3"], 1.0)

    mod.subprocess.run = fake_run
    result = mod.run_preflight(
        {"id": "slow", "resource": "cosbox:cuda3090", "preflight_cmd": ["python3", "slow.py"]},
        timeout=1.0,
    )
    assert result["ok"] is False
    assert result["reason"] == "preflight_timeout"


def main() -> int:
    test_selects_paused_preflights_by_default()
    test_explicit_preflight_selection_includes_default_opt_out()
    test_skipped_default_job_ids_reports_default_opt_out()
    test_checked_in_default_preflights_include_launchable_paused_rows()
    test_missing_requested_job_ids_fail_closed()
    test_run_preflight_passes_json_payload()
    test_command_resolution_preserves_remote_tilde_arguments()
    test_run_preflight_fails_with_payload_reason()
    test_run_preflight_nonzero_exit_with_ok_payload_is_explicit()
    test_run_preflight_requires_json_payload()
    test_run_preflight_requires_schema()
    test_run_preflight_rejects_resource_mismatch()
    test_timeout_fails_closed()
    print("mesh resource preflights selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
