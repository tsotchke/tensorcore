#!/usr/bin/env python3
"""Validate the checked-in Tensorcore mesh resource inventory."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"


def load_module(name: str, path: pathlib.Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inventory",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_INVENTORY,
    )
    return parser.parse_args()


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_inventory(path: pathlib.Path) -> list[str]:
    scheduler = load_module(
        "mesh_resource_scheduler_for_inventory_check",
        ROOT / "scripts" / "mesh_resource_scheduler.py",
    )
    arbiter_inventory = load_module(
        "mesh_arbiter_with_inventory_for_inventory_check",
        ROOT / "scripts" / "mesh_arbiter_with_inventory.py",
    )
    resources: dict[str, dict[str, Any]] = scheduler.load_inventory(str(path))
    capacities: dict[str, dict[str, Any]] = arbiter_inventory.load_inventory_capacities(str(path))
    errors: list[str] = []

    require(errors, bool(resources), "inventory must contain at least one resource")
    require(errors, set(resources) == set(capacities), "arbiter capacity rows must match inventory ids")
    require(
        errors,
        any(row.get("backend") == "cuda" for row in resources.values()),
        "inventory must contain at least one CUDA resource",
    )
    require(
        errors,
        any(row.get("status", "active") == "active" for row in resources.values()),
        "inventory must contain at least one active resource",
    )

    for resource_id, row in resources.items():
        capacity = capacities.get(resource_id, {})
        require(errors, capacity.get("capacity") == row.get("capacity", 1),
                f"{resource_id}: arbiter capacity mismatch")
        require(errors, capacity.get("status") == row.get("status", "active"),
                f"{resource_id}: arbiter status mismatch")
        require(errors, capacity.get("general_queue_eligible") == row.get("general_queue_eligible", True),
                f"{resource_id}: arbiter general_queue_eligible mismatch")
        if row.get("status") == "blocked":
            require(errors, bool(str(row.get("blocked_reason") or "").strip()),
                    f"{resource_id}: blocked resource must have blocked_reason")
        if row.get("general_queue_eligible") is False and row.get("status") != "blocked":
            require(errors, bool(row.get("reserved_for")),
                    f"{resource_id}: non-general resource must have reserved_for or be blocked")

    return errors


def main() -> int:
    args = parse_args()
    errors = validate_inventory(args.inventory)
    if errors:
        for error in errors:
            print(f"mesh resource inventory error: {error}")
        return 1
    print(f"mesh resource inventory OK: {args.inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
