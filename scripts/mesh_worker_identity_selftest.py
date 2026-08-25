#!/usr/bin/env python3
"""Selftests for scripts/mesh_worker_identity.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mesh_worker_identity.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("mesh_worker_identity_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_csv_parse() -> None:
    mod = load_module()
    apps = mod.parse_cuda_apps("1234, /opt/train.py, 8192\n")
    assert apps == [{
        "pid": 1234,
        "process_name": "/opt/train.py",
        "used_memory_mib": 8192,
        "raw": "1234, /opt/train.py, 8192",
    }]


def test_matching_process_and_cuda_requirement_pass() -> None:
    mod = load_module()
    mod.process_rows = lambda timeout: [{
        "pid": 1234,
        "ppid": 1,
        "pgid": 1234,
        "sid": 1234,
        "stat": "Sl",
        "etime": "00:01",
        "args": "python train_qllm.py",
    }]
    mod.cuda_apps = lambda nvidia_smi, timeout: {
        "ok": True,
        "rc": 0,
        "apps": [{"pid": 1234, "process_name": "python", "used_memory_mib": 8192}],
    }
    args = mod.parse_args([
        "--resource",
        "cosbox:cuda3090",
        "--match-regex",
        "train_qllm",
        "--require-matching-process",
        "--require-cuda",
        "--require-matched-cuda",
    ])
    payload = mod.build_payload(args)
    assert payload["ok"] is True
    assert payload["worker_pid"] == 1234
    assert payload["cuda_pids"] == [1234]
    assert payload["matched_cuda_pids"] == [1234]


def test_matched_cuda_requirement_blocks_unrelated_cuda() -> None:
    mod = load_module()
    mod.process_rows = lambda timeout: [{
        "pid": 1234,
        "ppid": 1,
        "pgid": 1234,
        "sid": 1234,
        "stat": "Sl",
        "etime": "00:01",
        "args": "python train_qllm.py",
    }]
    mod.cuda_apps = lambda nvidia_smi, timeout: {
        "ok": True,
        "rc": 0,
        "apps": [{"pid": 9999, "process_name": "steamwebhelper", "used_memory_mib": 9}],
    }
    args = mod.parse_args([
        "--match-regex",
        "train_qllm",
        "--require-matching-process",
        "--require-matched-cuda",
    ])
    payload = mod.build_payload(args)
    assert payload["ok"] is False
    assert payload["reason"] == "no_matched_cuda_process"


def test_required_unit_failure_blocks() -> None:
    mod = load_module()
    mod.process_rows = lambda timeout: []
    mod.cuda_apps = lambda nvidia_smi, timeout: {"ok": True, "rc": 0, "apps": []}
    mod.systemd_unit_status = lambda unit, timeout: {
        "unit": unit,
        "ok": True,
        "ActiveState": "inactive",
        "MainPID": 0,
    }
    args = mod.parse_args(["--unit", "qllm.service", "--require-active-unit"])
    payload = mod.build_payload(args)
    assert payload["ok"] is False
    assert payload["reason"] == "unit_not_active"


def main() -> int:
    test_cuda_csv_parse()
    test_matching_process_and_cuda_requirement_pass()
    test_matched_cuda_requirement_blocks_unrelated_cuda()
    test_required_unit_failure_blocks()
    print("mesh worker identity selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
