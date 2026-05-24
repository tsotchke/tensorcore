#!/usr/bin/env python3
"""Validate live mesh_training_demo evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


SCHEMA = "tensorcore.live_mesh_training.evidence.v1"
FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tensorcore live mesh training evidence."
    )
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--min-outer-steps", type=int, default=1)
    parser.add_argument("--require-direct-ring", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-cuda-rank3", action="store_true")
    return parser.parse_args()


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read live mesh evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"live mesh evidence is not valid JSON: {exc}") from exc


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    args = parse_args()
    data = load_json(args.evidence)
    errors: list[str] = []

    require(errors, data.get("schema") == SCHEMA, "schema mismatch")
    require(errors, data.get("meta", {}).get("format") == FORMAT_VERSION, "format mismatch")
    require(errors, data.get("status") == "passed", "run status must be passed")

    run = data.get("run", {})
    summary = data.get("summary", {})
    ranks = data.get("ranks", [])
    require(errors, run.get("world") == args.world, f"run.world must be {args.world}")
    require(errors, isinstance(ranks, list) and len(ranks) == args.world,
            f"ranks must contain {args.world} entries")

    require(errors, summary.get("passed") is True, "summary.passed must be true")
    require(errors, summary.get("all_ranks_passed") is True,
            "all ranks must report mesh_training_demo OK")
    require(errors, summary.get("all_requested_outer_steps") is True,
            "all ranks must finish requested outer steps")
    require(errors, summary.get("loss_decreased") is True,
            "all ranks must decrease loss")

    expected_route_events = args.world * args.min_outer_steps
    if args.require_direct_ring:
        require(errors, summary.get("direct_ring_ranks") == args.world,
                "all ranks must report direct_ring=enabled")
        require(errors, summary.get("ring_route_events", 0) >= expected_route_events,
                f"expected at least {expected_route_events} ring route events")

    if args.require_checkpoint:
        require(errors, summary.get("checkpoint_ranks") == args.world,
                "all ranks must report checkpoint counters")

    if args.require_cuda_rank3:
        require(errors, 3 in summary.get("cuda_ranks", []),
                "rank 3 must report CUDA backend")

    if isinstance(ranks, list):
        for item in ranks:
            if not isinstance(item, dict):
                errors.append(f"rank entry must be object, got {item!r}")
                continue
            rank = item.get("rank")
            require(errors, item.get("ok") is True, f"rank {rank} did not pass")
            require(errors, item.get("outer_steps_completed", 0) >= args.min_outer_steps,
                    f"rank {rank} completed too few outer steps")
            require(errors, item.get("last_loss", 0.0) < item.get("first_loss", 0.0),
                    f"rank {rank} did not decrease loss")
            if args.require_direct_ring:
                require(errors, item.get("direct_ring", {}).get("enabled") is True,
                        f"rank {rank} did not enable direct ring")
                ring_routes = [
                    route for route in item.get("routes", [])
                    if isinstance(route, dict) and route.get("route") == "ring"
                ]
                require(errors, len(ring_routes) >= args.min_outer_steps,
                        f"rank {rank} has too few ring route events")
            if args.require_checkpoint:
                checkpoint = item.get("checkpoint", {})
                require(errors, checkpoint.get("discards", 0) > 0,
                        f"rank {rank} did not discard checkpointed activations")
                require(errors, checkpoint.get("realizes", 0) > 0,
                        f"rank {rank} did not realize checkpointed activations")
                require(errors, checkpoint.get("final_discarded") == 0,
                        f"rank {rank} leaked discarded checkpoint bytes")
            if args.require_cuda_rank3 and rank == 3:
                backends = {
                    outer.get("backend")
                    for outer in item.get("outer", [])
                    if isinstance(outer, dict)
                }
                require(errors, "cuda" in backends, "rank 3 outer steps did not use CUDA")

    if errors:
        for error in errors:
            print(f"live mesh evidence error: {error}")
        return 1

    cuda = ",".join(str(rank) for rank in summary.get("cuda_ranks", [])) or "none"
    print(
        "live mesh evidence OK: "
        f"world={run.get('world')} outer_steps={run.get('outer_steps')} "
        f"ring_routes={summary.get('ring_route_events')} cuda_ranks={cuda}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
