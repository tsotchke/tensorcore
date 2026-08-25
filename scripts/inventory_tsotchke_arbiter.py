#!/usr/bin/env python3
"""Run tsotchke-arbiter with TensorCore's inventory as its resource registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


INVENTORY_SCHEMA = "tensorcore.mesh_resources.v1"


def load_registry(path: Path) -> dict[str, dict]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"inventory schema must be {INVENTORY_SCHEMA}")
    rows = payload.get("resources")
    if not isinstance(rows, list):
        raise ValueError("inventory.resources must be a list")
    registry: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("inventory contains an invalid resource row")
        resource = row["id"]
        if resource in registry:
            raise ValueError(f"inventory repeats resource {resource!r}")
        capacity = row.get("capacity", 1)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"resource {resource!r} has invalid capacity")
        registry[resource] = {
            "capacity": capacity,
            "class": str(row.get("class") or "generic"),
            "node": str(row.get("node") or resource.split(":", 1)[0]),
            "description": "TensorCore authoritative scheduler inventory",
        }
    return registry


def command_name(argv: list[str]) -> tuple[str | None, str | None]:
    positional = [part for part in argv if not part.startswith("-")]
    command = positional[0] if positional else None
    resource = positional[1] if command == "claim" and len(positional) > 1 else None
    return command, resource


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", required=True, type=Path)
    parser.add_argument("--delegate", required=True)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, delegated = parse_args(argv or sys.argv[1:])
    try:
        registry = load_registry(args.inventory_json)
    except Exception as exc:
        print(f"inventory arbiter configuration error: {exc}", file=sys.stderr)
        return 2
    command, resource = command_name(delegated)
    if command == "claim" and resource not in registry:
        payload = {
            "ok": False,
            "resource": resource,
            "error": "resource_not_in_tensorcore_inventory",
        }
        if "--json" in delegated:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(payload["error"], file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["TSOTCHKE_RESOURCE_CAPACITIES"] = json.dumps(
        registry, sort_keys=True, separators=(",", ":"),
    )
    try:
        proc = subprocess.run([args.delegate, *delegated], env=env, check=False)
    except FileNotFoundError:
        print("configured arbiter delegate was not found", file=sys.stderr)
        return 2
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
