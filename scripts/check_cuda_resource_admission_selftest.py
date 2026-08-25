#!/usr/bin/env python3
"""Selftests for scripts/check_cuda_resource_admission.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cuda_resource_admission.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(
        "check_cuda_resource_admission_under_test",
        str(SCRIPT),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_nvidia_smi(directory: pathlib.Path, stdout: str, rc: int = 0) -> pathlib.Path:
    path = directory / "nvidia-smi"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"raise SystemExit({rc})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_empty_gpu_is_admitted() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        smi = fake_nvidia_smi(pathlib.Path(tmp), "")
        args = mod.parse_args(["--nvidia-smi", str(smi), "--json"])
        payload = mod.build_payload(args)
    assert payload["ok"] is True
    assert payload["compute_app_count"] == 0


def test_unmanaged_cuda_process_blocks() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        smi = fake_nvidia_smi(pathlib.Path(tmp), "1234, /opt/train.py, 1024\n")
        args = mod.parse_args(["--nvidia-smi", str(smi), "--json"])
        payload = mod.build_payload(args)
    assert payload["ok"] is False
    assert payload["reason"] == "blocked_cuda_compute_apps"
    assert payload["blocked"][0]["pid"] == 1234


def test_explicit_small_allowlist_is_admitted() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        smi = fake_nvidia_smi(pathlib.Path(tmp), "99, /usr/bin/steamwebhelper, 9\n")
        args = mod.parse_args(
            [
                "--nvidia-smi",
                str(smi),
                "--allow-process-regex",
                "steamwebhelper$",
                "--allowed-process-max-memory-mib",
                "16",
                "--json",
            ]
        )
        payload = mod.build_payload(args)
    assert payload["ok"] is True
    assert payload["allowed"][0]["pid"] == 99


def test_allowlist_memory_cap_still_blocks() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        smi = fake_nvidia_smi(pathlib.Path(tmp), "99, /usr/bin/steamwebhelper, 128\n")
        args = mod.parse_args(
            [
                "--nvidia-smi",
                str(smi),
                "--allow-process-regex",
                "steamwebhelper$",
                "--allowed-process-max-memory-mib",
                "16",
                "--json",
            ]
        )
        payload = mod.build_payload(args)
    assert payload["ok"] is False
    assert payload["blocked"][0]["used_memory_mib"] == 128


def main() -> int:
    test_empty_gpu_is_admitted()
    test_unmanaged_cuda_process_blocks()
    test_explicit_small_allowlist_is_admitted()
    test_allowlist_memory_cap_still_blocks()
    print("cuda resource admission selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
