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
    print("HIP toolchain probe selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
