#!/usr/bin/env python3
"""Selftests for scripts/georefine_qwen_control.py."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import pathlib
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "georefine_qwen_control.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("georefine_qwen_control_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_summarizes_live_run() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "run_intent.json").write_text(
            json.dumps(
                {
                    "lease_id": "lease123",
                    "resource": "cosbox:cuda3090",
                    "target": "qwen-test",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "m2_supervisor_status.json").write_text(
            json.dumps(
                {
                    "state": "running",
                    "updated_at": 2000.0,
                    "supervisor_pid": 10,
                    "compressor_pid": 20,
                    "live_agent_pid": 30,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "georefine_live_control_reconciler_state.json").write_text(
            json.dumps({"last_result": {"ok": True, "lease_verification": {"ok": True}}}),
            encoding="utf-8",
        )
        (run_dir / "m2_live_control.json").write_text(
            json.dumps({"set_params": {"m2_target_kl_kd_lr": 7e-6}, "commands": {}}),
            encoding="utf-8",
        )
        (run_dir / "m2_supervisor_child.log").write_text(
            "[M2/KD] step 204/4096: loss=1.0e+01  chunk_kl=1.0e-02  lr=7.00e-06  (chunk 1/1, ETA 129.3m)\n",
            encoding="utf-8",
        )
        mod.time.time = lambda: 2010.0
        mod.systemd_status = lambda unit: {"ActiveState": "active", "unit": unit}
        mod.cuda_status = lambda pid: {"ok": True, "matched_compressor": [{"pid": pid}], "apps": []}
        payload = mod.status_payload(
            argparse.Namespace(run_dir=run_dir, max_heartbeat_age_sec=120.0)
        )
    assert payload["ok"] is True
    assert payload["health"]["systemd_active"] is True
    assert payload["progress"]["latest_kd"]["step"] == 204
    assert payload["adjustable"]["current_control"]["set_params"]["m2_target_kl_kd_lr"] == 7e-6


def test_param_assignment_validation() -> None:
    mod = load_module()
    assert mod.parse_param_assignment("m2_target_kl_kd_steps=123") == (
        "m2_target_kl_kd_steps",
        123,
    )
    assert mod.parse_param_assignment("m2_target_kl_kd_only=true") == (
        "m2_target_kl_kd_only",
        True,
    )
    try:
        mod.parse_param_assignment("m2_target_kl_kd_chunk_size=999999")
    except ValueError as exc:
        assert "must be <=" in str(exc)
    else:
        raise AssertionError("expected out-of-range assignment to fail")


def test_discovers_active_run_from_scheduler_state() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        state_json = pathlib.Path(tmp) / "state.json"
        state_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "action": "heartbeated_live_holder",
                            "job": "qwen-job",
                            "lease_id": "lease123",
                            "ok": True,
                            "resource": "cosbox:cuda3090",
                            "worker_identity": {
                                "identity": {
                                    "artifact_dir": "/remote/qwen-run",
                                    "worker_host": "cosbox",
                                }
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        discovered = mod.discover_active_run(state_json)
    assert discovered is not None
    assert discovered["run_dir"] == "/remote/qwen-run"
    assert discovered["target"] == "cosbox"
    assert discovered["lease_id"] == "lease123"


def main() -> int:
    test_status_summarizes_live_run()
    test_param_assignment_validation()
    test_discovers_active_run_from_scheduler_state()
    print("GeoRefine Qwen control selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
