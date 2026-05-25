#!/usr/bin/env python3
"""Fixture tests for scripts/probe_hip_toolchain.py."""

from __future__ import annotations

import pathlib
import tempfile

import probe_hip_toolchain


TEST_HEAD = "abc123"
VALID_STATUSES = {
    "ready_for_hip_gemm",
    "runtime_only_no_hipblas",
    "missing_requirements",
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / ".tensorcore_source_head").write_text(TEST_HEAD + "\n", encoding="utf-8")
        (root / ".tensorcore_source_dirty").write_text("0\n", encoding="utf-8")
        evidence = probe_hip_toolchain.collect_evidence(root)
        if evidence["git_head"] != TEST_HEAD:
            raise AssertionError(f"git_head marker was not honored: {evidence['git_head']!r}")
        if evidence["git_dirty"] is not False:
            raise AssertionError(f"git_dirty marker was not honored: {evidence['git_dirty']!r}")
        if evidence["readiness"]["status"] not in VALID_STATUSES:
            raise AssertionError(f"unexpected readiness: {evidence['readiness']!r}")

        prefix = root / "prefix"
        bindir = prefix / "bin"
        bindir.mkdir(parents=True)
        versioned = bindir / "llvm-spirv-19"
        versioned.write_text("#!/bin/sh\nprintf 'llvm-spirv test\\n'\n", encoding="utf-8")
        versioned.chmod(0o755)
        got = probe_hip_toolchain.find_tool("llvm-spirv", [str(prefix)])
        if got != str(versioned):
            raise AssertionError(f"versioned llvm-spirv was not found: {got!r}")

        clinfo = """
  Platform Name                                   NVIDIA CUDA
  Device Name                                     NVIDIA GeForce RTX 3090
  Device Type                                     GPU
    IL version                                    (n/a)
  Device Extensions                               cl_khr_fp64
  Platform Name                                   Level Zero
  Device Name                                     Intel Arc
  Device Type                                     GPU
    IL version                                    SPIR-V_1.2
  Device Extensions                               cl_khr_il_program
"""
        devices = probe_hip_toolchain.parse_clinfo_devices(clinfo)
        if not probe_hip_toolchain.has_gpu_spirv_device(devices):
            raise AssertionError(f"SPIR-V GPU device was not detected: {devices!r}")
    print("HIP toolchain probe selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
