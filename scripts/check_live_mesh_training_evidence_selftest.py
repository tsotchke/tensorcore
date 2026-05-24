#!/usr/bin/env python3
"""Fixture tests for the live mesh training evidence checker."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_live_mesh_training_evidence.py"


def base_evidence() -> dict[str, Any]:
    ranks = []
    for rank in range(4):
        backend = "cuda" if rank == 3 else "portable_cpu"
        ranks.append({
            "rank": rank,
            "ok": True,
            "first_loss": 4.0 - 0.1 * rank,
            "last_loss": 1.0 - 0.05 * rank,
            "outer_steps_completed": 5,
            "direct_ring": {
                "enabled": True,
                "next_rank": (rank + 1) % 4,
                "next": f"100.0.0.{rank + 1}:6000{rank}",
                "timeout_ms": 10000,
            },
            "routes": [
                {"route": "ring", "elements": 2048},
                {"route": "ring", "elements": 64},
                {"route": "ring", "elements": 2048},
                {"route": "ring", "elements": 64},
                {"route": "ring", "elements": 2048},
            ],
            "outer": [
                {"step": step, "total": 5, "loss": 2.0 / step, "bytes": 8448, "backend": backend}
                for step in range(1, 6)
            ],
            "checkpoint": {
                "discards": 40,
                "realizes": 40,
                "peak_discarded": 512,
                "final_discarded": 0,
            },
        })
    return {
        "schema": "tensorcore.live_mesh_training.evidence.v1",
        "meta": {"format": 1, "source": "selftest"},
        "status": "passed",
        "run": {
            "world": 4,
            "outer_steps": 5,
            "checkpoint_enabled": True,
            "ring_enabled": True,
            "rank3_cuda_requested": True,
        },
        "summary": {
            "passed": True,
            "all_ranks_passed": True,
            "all_requested_outer_steps": True,
            "loss_decreased": True,
            "direct_ring_ranks": 4,
            "ring_route_events": 20,
            "checkpoint_ranks": 4,
            "cuda_ranks": [3],
        },
        "ranks": ranks,
    }


def run_checker(evidence: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(evidence, handle)
        path = pathlib.Path(handle.name)
    try:
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        path.unlink(missing_ok=True)


def assert_passes(evidence: dict[str, Any]) -> None:
    result = run_checker(
        evidence,
        "--min-outer-steps", "5",
        "--require-direct-ring",
        "--require-checkpoint",
        "--require-cuda-rank3",
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(evidence: dict[str, Any], needle: str) -> None:
    result = run_checker(
        evidence,
        "--min-outer-steps", "5",
        "--require-direct-ring",
        "--require-checkpoint",
        "--require-cuda-rank3",
    )
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    good = base_evidence()
    assert_passes(good)

    no_cuda = copy.deepcopy(good)
    no_cuda["summary"]["cuda_ranks"] = []
    no_cuda["ranks"][3]["outer"][0]["backend"] = "portable_cpu"
    assert_fails(no_cuda, "rank 3 must report CUDA backend")

    brokered = copy.deepcopy(good)
    brokered["summary"]["direct_ring_ranks"] = 3
    brokered["ranks"][2]["direct_ring"]["enabled"] = False
    assert_fails(brokered, "all ranks must report direct_ring=enabled")

    leaked_checkpoint = copy.deepcopy(good)
    leaked_checkpoint["ranks"][1]["checkpoint"]["final_discarded"] = 512
    assert_fails(leaked_checkpoint, "leaked discarded checkpoint bytes")

    stalled = copy.deepcopy(good)
    stalled["summary"]["loss_decreased"] = False
    stalled["ranks"][0]["last_loss"] = stalled["ranks"][0]["first_loss"]
    assert_fails(stalled, "all ranks must decrease loss")

    print("live mesh training evidence checker selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
