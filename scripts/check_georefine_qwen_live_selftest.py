#!/usr/bin/env python3
"""Selftests for scripts/check_georefine_qwen_live.py."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import pathlib
import subprocess
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_georefine_qwen_live.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("check_georefine_qwen_live_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(run_dir: pathlib.Path, **kwargs: object) -> argparse.Namespace:
    defaults = {
        "run_dir": run_dir,
        "status_file": "",
        "match_regex": "experiments.georefine.m2_compress",
        "json": True,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_status_pid_with_run_dir_passes() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run"
        run_dir.mkdir()
        status = run_dir / "m2_supervisor_status.json"
        status.write_text(json.dumps({"state": "running", "compressor_pid": 123}), encoding="utf-8")
        mod.proc_cmdline = lambda pid: f"python -m experiments.georefine.m2_compress --output-dir {run_dir}"
        payload = mod.build_payload(args(run_dir))
    assert payload["ok"] is True
    assert payload["status_pids"] == [123]


def test_non_running_status_falls_back_to_pgrep() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "m2_supervisor_status.json").write_text(
            json.dumps({"state": "starting", "compressor_pid": 123}),
            encoding="utf-8",
        )

        def fake_run(*cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, f"456 python -m experiments.georefine.m2_compress --output-dir {run_dir}\n", "")

        mod.subprocess.run = fake_run
        payload = mod.build_payload(args(run_dir))
    assert payload["ok"] is True
    assert payload["pgrep_matches"][0]["pid"] == 456


def test_wrong_run_dir_fails_closed() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "m2_supervisor_status.json").write_text(
            json.dumps({"state": "running", "compressor_pid": 123}),
            encoding="utf-8",
        )
        mod.proc_cmdline = lambda pid: "python -m experiments.georefine.m2_compress --output-dir /other"
        mod.pgrep_matches = lambda pattern, run_dir: []
        payload = mod.build_payload(args(run_dir))
    assert payload["ok"] is False
    assert payload["reason"] == "not_live"


def main() -> int:
    test_status_pid_with_run_dir_passes()
    test_non_running_status_falls_back_to_pgrep()
    test_wrong_run_dir_fails_closed()
    print("georefine qwen live checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
