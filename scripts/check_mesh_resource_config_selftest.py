#!/usr/bin/env python3
"""Selftests for mesh resource config validators."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jobs_reject_private_remote_wrappers() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_checked_in_command(
        errors,
        "job 'bad'",
        "start_cmd",
        ["ssh", "cosbox", "cd /tmp && /home/tyr/.local/bin/start-georefine"],
    )
    assert any("private wrapper path" in error for error in errors)


def test_jobs_reject_host_local_run_scripts() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_checked_in_command(
        errors,
        "job 'bad'",
        "start_cmd",
        ["ssh", "old-donkey", "/data/qllm/runs/start_olddonkey_precompute_chain.sh"],
    )
    assert any("host-local script" in error for error in errors)


def test_jobs_reject_missing_repo_paths_inside_remote_strings() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_checked_in_command(
        errors,
        "job 'bad'",
        "admission_cmd",
        ["ssh", "cosbox", "cd ~/src/tensorcore && python3 scripts/does_not_exist.py --json"],
    )
    assert any("missing repo path" in error for error in errors)


def test_jobs_allow_repo_local_helpers() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_checked_in_command(
        errors,
        "job 'good'",
        "admission_cmd",
        ["python3", "scripts/check_windows_cuda_resource_admission.py", "--json"],
    )
    assert errors == []


def test_paused_launchable_jobs_require_preflight() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "paused-launchable",
            "desired_state": "paused",
            "start_cmd": ["python3", "scripts/start_windows_cuda_smoke.py"],
            "metadata": {"scheduler_pause_reason": "waiting for durable launch path"},
        },
    )
    assert any("requires preflight_cmd" in error for error in errors)


def test_paused_jobs_require_pause_reason() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "paused-lane",
            "desired_state": "paused",
            "metadata": {},
        },
    )
    assert any("scheduler_pause_reason" in error for error in errors)


def test_running_jobs_reject_host_local_systemd_starts() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "systemd-hidden-launcher",
            "desired_state": "running",
            "start_cmd": ["ssh", "cosbox", "systemctl --user start qllm-phase1.service"],
            "metadata": {},
        },
    )
    assert any("host-local systemd unit" in error for error in errors)


def test_running_jobs_reject_legacy_georefine_direct_starter() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "legacy-georefine",
            "desired_state": "running",
            "start_cmd": ["python3", "scripts/start_georefine_qwen_cr025.py", "--json"],
            "metadata": {},
        },
    )
    assert any("legacy direct GeoRefine starter" in error for error in errors)


def test_tensorcore_job_v1_georefine_contract_requires_rank_probe_starter() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "bad-georefine-v1",
            "desired_state": "running",
            "start_cmd": ["python3", "scripts/start_georefine_qwen_cr025.py", "--json"],
            "metadata": {
                "scheduler_contract": "tensorcore_job_v1_cuda_exclusive_trusted_artifact",
            },
        },
    )
    assert any("start_georefine_qwen_rank_probe.py" in error for error in errors)


def qllm_phase1_cached_job(**overrides: object) -> dict:
    row = {
        "id": "qllm-phase1",
        "desired_state": "paused",
        "start_cmd": [
            "python3",
            "./scripts/start_qllm_phase1_cached.py",
            "--target",
            "cosbox",
            "--json",
        ],
        "preflight_cmd": [
            "python3",
            "./scripts/start_qllm_phase1_cached.py",
            "--target",
            "cosbox",
            "--preflight-only",
            "--json",
        ],
        "probe_cmd": [
            "ssh",
            "cosbox",
            "cd /home/tyr/projects/semiclassical_qllm && python3 scripts/check_qllm_training_run.py /home/tyr/.local/share/qllm_trusted/runs/phase1 --require-live --json",
        ],
        "post_start_probe_cmd": [
            "ssh",
            "cosbox",
            "cd /home/tyr/projects/semiclassical_qllm && python3 scripts/check_qllm_training_run.py /home/tyr/.local/share/qllm_trusted/runs/phase1 --require-live --json",
        ],
        "completion_cmd": [
            "ssh",
            "cosbox",
            "cd /home/tyr/projects/semiclassical_qllm && python3 scripts/check_qllm_training_run.py /home/tyr/.local/share/qllm_trusted/runs/phase1 --require-complete --json",
        ],
        "worker_identity_cmd": [
            "ssh",
            "cosbox",
            "cd ~/src/tensorcore && python3 scripts/mesh_worker_identity.py --resource cosbox:cuda3090 --require-matched-cuda --json",
        ],
        "artifact_root": "/home/tyr/.local/share/qllm_trusted/evidence",
        "metadata": {
            "scheduler_contract": "tensorcore_job_v1_qllm_phase1_cached",
            "scheduler_pause_reason": "operator-started proof run",
            "trusted_evidence_root": "/home/tyr/.local/share/qllm_trusted/evidence",
        },
    }
    row.update(overrides)
    return row


def test_tensorcore_job_v1_qllm_contract_requires_phase1_starter() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        qllm_phase1_cached_job(start_cmd=["python3", "scripts/start_windows_cuda_smoke.py", "--json"]),
    )
    assert any("start_qllm_phase1_cached.py" in error for error in errors)


def test_tensorcore_job_v1_qllm_contract_requires_completion_checker() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        qllm_phase1_cached_job(completion_cmd=["python3", "scripts/check_windows_cuda_smoke_artifact.py", "--json"]),
    )
    assert any("check_qllm_training_run.py" in error for error in errors)


def test_tensorcore_job_v1_qllm_contract_rejects_bytehole_evidence_root() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    row = qllm_phase1_cached_job(artifact_root="/home/tyr/bytehole/qllm/evidence")
    row["metadata"]["trusted_evidence_root"] = "/home/tyr/bytehole/qllm/evidence"
    jobs.validate_job_policy(errors, row)
    assert any("trusted evidence root" in error for error in errors)


def test_tensorcore_job_v1_qllm_contract_passes_policy() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(errors, qllm_phase1_cached_job())
    assert errors == []


def test_cuda_inventory_requires_gpu_reconciliation_policy() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_gpu_reconciliation_policy(
        errors,
        "cosbox:cuda3090",
        {
            "id": "cosbox:cuda3090",
            "backend": "cuda",
            "status": "active",
            "control_plane": "tensorcore_scheduler",
        },
    )
    assert any("requires gpu_reconciliation policy" in error for error in errors)


def test_cuda_inventory_accepts_enabled_gpu_reconciliation_policy() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_gpu_reconciliation_policy(
        errors,
        "cosbox:cuda3090",
        {
            "id": "cosbox:cuda3090",
            "backend": "cuda",
            "status": "active",
            "control_plane": "tensorcore_scheduler",
            "gpu_reconciliation": {
                "enabled": True,
                "poll_host": "cosbox",
                "allow_process_regex": ["steamwebhelper$"],
                "allowed_process_max_memory_mib": 64,
            },
        },
    )
    assert errors == []


def test_cuda_inventory_disabled_gpu_reconciliation_requires_reason() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_gpu_reconciliation_policy(
        errors,
        "jack-blupc:cuda3060",
        {
            "id": "jack-blupc:cuda3060",
            "backend": "cuda",
            "status": "active",
            "control_plane": "tensorcore_scheduler",
            "gpu_reconciliation": {"enabled": False},
        },
    )
    assert any("requires reason" in error for error in errors)


def test_preflight_commands_must_emit_json() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "paused-preflight-no-json",
            "desired_state": "paused",
            "preflight_cmd": ["python3", "scripts/check_windows_persistent_launch.py"],
            "metadata": {"scheduler_pause_reason": "waiting for durable launch path"},
        },
    )
    assert any("must emit JSON" in error for error in errors)


def test_git_access_preflight_requires_resource() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "paused-git-preflight",
            "desired_state": "paused",
            "preflight_cmd": [
                "python3",
                "scripts/check_mesh_git_access.py",
                "--target",
                "cosbox",
                "--repo-url",
                "git@example.com:repo.git",
                "--json",
            ],
            "metadata": {"scheduler_pause_reason": "waiting for git access"},
        },
    )
    assert any("must pass --resource" in error for error in errors)


def test_paused_launchable_job_with_preflight_passes_policy() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            "id": "paused-good",
            "desired_state": "paused",
            "start_cmd": ["python3", "scripts/start_windows_cuda_smoke.py"],
            "preflight_cmd": ["python3", "scripts/check_windows_persistent_launch.py", "--json"],
            "metadata": {"scheduler_pause_reason": "waiting for durable launch path"},
        },
    )
    assert errors == []


def windows_scheduled_smoke_job(**overrides: object) -> dict:
    row = {
        "id": "windows-smoke",
        "desired_state": "paused",
        "start_cmd": [
            "python3",
            "scripts/start_windows_cuda_smoke.py",
            "--duration-sec",
            "3",
            "--json",
        ],
        "preflight_cmd": ["python3", "scripts/check_windows_persistent_launch.py", "--json"],
        "probe_cmd": [
            "python3",
            "scripts/check_windows_cuda_smoke_artifact.py",
            "--require-live",
            "--json",
        ],
        "post_start_probe_cmd": [
            "python3",
            "scripts/check_windows_cuda_smoke_artifact.py",
            "--require-live-or-complete",
            "--json",
        ],
        "completion_cmd": [
            "python3",
            "scripts/check_windows_cuda_smoke_artifact.py",
            "--require-complete",
            "--json",
        ],
        "metadata": {
            "scheduler_contract": "windows_cuda_scheduled_smoke",
            "scheduler_pause_reason": "waiting for durable launch path",
        },
    }
    row.update(overrides)
    return row


def test_windows_scheduled_smoke_rejects_foreground_start() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
            "start_cmd": [
                "python3",
                "scripts/start_windows_cuda_smoke.py",
                "--duration-sec",
                "3",
                "--foreground",
            ],
        },
    )
    assert any("must not use --foreground" in error for error in errors)


def test_windows_scheduled_smoke_requires_persistent_preflight() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
            "preflight_cmd": ["python3", "scripts/check_windows_cuda_resource_admission.py", "--json"],
        },
    )
    assert any("check_windows_persistent_launch.py" in error for error in errors)


def test_windows_scheduled_smoke_requires_live_or_complete_post_start() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
            "post_start_probe_cmd": [
                "python3",
                "scripts/check_windows_cuda_smoke_artifact.py",
                "--require-complete",
                "--json",
            ],
        },
    )
    assert any("post_start_probe_cmd must accept live or completed" in error for error in errors)


def test_windows_scheduled_smoke_probe_must_not_accept_completed_artifacts() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
            "probe_cmd": [
                "python3",
                "scripts/check_windows_cuda_smoke_artifact.py",
                "--require-live-or-complete",
                "--json",
            ],
        },
    )
    assert any("must not treat completed artifacts as live" in error for error in errors)


def test_windows_scheduled_smoke_rejects_long_duration() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
            "start_cmd": [
                "python3",
                "scripts/start_windows_cuda_smoke.py",
                "--duration-sec",
                "30",
            ],
        },
    )
    assert any("duration must be <=" in error for error in errors)


def test_windows_scheduled_smoke_contract_passes_policy() -> None:
    jobs = load_script("check_mesh_resource_jobs_under_test", ROOT / "scripts" / "check_mesh_resource_jobs.py")
    errors: list[str] = []
    jobs.validate_job_policy(
        errors,
        {
            **windows_scheduled_smoke_job(),
        },
    )
    assert errors == []


def test_inventory_rejects_missing_repo_paths() -> None:
    inventory = load_script(
        "check_mesh_resource_inventory_under_test",
        ROOT / "scripts" / "check_mesh_resource_inventory.py",
    )
    errors: list[str] = []
    inventory.validate_checked_in_command(
        errors,
        "node:gpu",
        "cuda_probe_cmd",
        ["python3", "scripts/does_not_exist.py"],
    )
    assert any("missing repo path" in error for error in errors)


def main() -> int:
    test_jobs_reject_private_remote_wrappers()
    test_jobs_reject_host_local_run_scripts()
    test_jobs_reject_missing_repo_paths_inside_remote_strings()
    test_jobs_allow_repo_local_helpers()
    test_paused_launchable_jobs_require_preflight()
    test_paused_jobs_require_pause_reason()
    test_running_jobs_reject_host_local_systemd_starts()
    test_running_jobs_reject_legacy_georefine_direct_starter()
    test_tensorcore_job_v1_georefine_contract_requires_rank_probe_starter()
    test_tensorcore_job_v1_qllm_contract_requires_phase1_starter()
    test_tensorcore_job_v1_qllm_contract_requires_completion_checker()
    test_tensorcore_job_v1_qllm_contract_rejects_bytehole_evidence_root()
    test_tensorcore_job_v1_qllm_contract_passes_policy()
    test_cuda_inventory_requires_gpu_reconciliation_policy()
    test_cuda_inventory_accepts_enabled_gpu_reconciliation_policy()
    test_cuda_inventory_disabled_gpu_reconciliation_requires_reason()
    test_preflight_commands_must_emit_json()
    test_git_access_preflight_requires_resource()
    test_paused_launchable_job_with_preflight_passes_policy()
    test_windows_scheduled_smoke_rejects_foreground_start()
    test_windows_scheduled_smoke_requires_persistent_preflight()
    test_windows_scheduled_smoke_requires_live_or_complete_post_start()
    test_windows_scheduled_smoke_probe_must_not_accept_completed_artifacts()
    test_windows_scheduled_smoke_rejects_long_duration()
    test_windows_scheduled_smoke_contract_passes_policy()
    test_inventory_rejects_missing_repo_paths()
    print("mesh resource config validator selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
