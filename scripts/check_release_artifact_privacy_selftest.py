#!/usr/bin/env python3
"""Focused tests for binary/archive release privacy detection."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release_artifact_privacy",
    ROOT / "scripts" / "check_release_artifact_privacy.py",
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def main() -> int:
    private_mac = b"prefix /Users/" + b"private-user/project suffix"
    private_linux = b"prefix /home/" + b"private-user/project suffix"
    private_windows = b"prefix C:/Users/" + b"private-user/project suffix"
    assert MOD.findings_for_bytes("mac", private_mac) == ["mac: macos_home"]
    assert MOD.findings_for_bytes("linux", private_linux) == ["linux: linux_home"]
    assert MOD.findings_for_bytes("windows", private_windows) == ["windows: windows_home"]
    assert MOD.findings_for_bytes("network", b"tcp://192.168." + b"1.4:9000") == [
        "network: private_network_address"
    ]
    assert not MOD.findings_for_bytes(
        "public",
        b"tensorcore.metallib kernels/metal/gemm.metal tcp://192.0.2.10:9000",
    )
    print("release artifact privacy selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
