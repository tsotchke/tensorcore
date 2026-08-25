#!/usr/bin/env python3
"""Run the Tsotchke arbiter with Tensorcore mesh inventory resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "tensorcore.mesh_resources.v1"
STATUSES = {"active", "reserved", "blocked"}


def load_inventory_capacities(path: str) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError(f"inventory schema must be {SCHEMA}")
    resources = raw.get("resources")
    if not isinstance(resources, list):
        raise ValueError("inventory resources must be a list")
    capacities: dict[str, dict[str, Any]] = {}
    for row in resources:
        if not isinstance(row, dict):
            raise ValueError("inventory resources contains a non-object row")
        resource_id = row.get("id")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ValueError("inventory resource id must be a non-empty string")
        if resource_id in capacities:
            raise ValueError(f"inventory has duplicate resource {resource_id!r}")
        capacity = row.get("capacity", 1)
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"resource {resource_id!r} capacity must be a positive integer")
        status = row.get("status", "active")
        if status not in STATUSES:
            raise ValueError(f"resource {resource_id!r} status must be one of {sorted(STATUSES)!r}")
        general = row.get("general_queue_eligible", True)
        if not isinstance(general, bool):
            raise ValueError(f"resource {resource_id!r} general_queue_eligible must be a JSON boolean")
        capacities[resource_id] = {
            "capacity": capacity,
            "class": row.get("class"),
            "node": row.get("node"),
            "backend": row.get("backend"),
            "status": status,
            "general_queue_eligible": general,
            "description": row.get("description"),
        }
    return capacities


def merged_capacities(path: str, existing_json: str | None) -> dict[str, dict[str, Any]]:
    existing: dict[str, Any] = {}
    if existing_json:
        try:
            parsed = json.loads(existing_json)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            existing = parsed
    merged = {str(key): value for key, value in existing.items() if isinstance(value, dict)}
    merged.update(load_inventory_capacities(path))
    return merged


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--arbiter-cmd", required=True)
    parser.add_argument("--print-env-json", action="store_true")
    parser.add_argument("arbiter_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.arbiter_args and args.arbiter_args[0] == "--":
        args.arbiter_args = args.arbiter_args[1:]
    if not args.print_env_json and not args.arbiter_args:
        parser.error("arbiter arguments are required unless --print-env-json is set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    capacities = merged_capacities(
        args.inventory_json,
        os.environ.get("TSOTCHKE_RESOURCE_CAPACITIES"),
    )
    encoded = json.dumps(capacities, sort_keys=True)
    if args.print_env_json:
        print(encoded)
        return 0
    env = os.environ.copy()
    env["TSOTCHKE_RESOURCE_CAPACITIES"] = encoded
    proc = subprocess.run(
        [str(Path(args.arbiter_cmd).expanduser()), *args.arbiter_args],
        env=env,
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
