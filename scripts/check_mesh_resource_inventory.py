#!/usr/bin/env python3
"""Validate the checked-in Tensorcore mesh resource inventory."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
import shlex
from types import ModuleType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "configs" / "mesh_resources.json"
CONTROL_PLANES = {"tensorcore_scheduler", "direct_lease", "reserved", "blocked"}
PRIVATE_COMMAND_PATTERNS = (
    "~/.tsotchke/bin",
    "/.tsotchke/bin",
    "/Users/",
    "/home/tyr/.local/bin/",
    "/.local/bin/",
)
HOST_LOCAL_SCRIPT_PREFIXES = (
    "/data/qllm/runs/",
    "/home/tyr/bytehole/qllm/runs/",
)
REPO_RELATIVE_PREFIXES = ("scripts/", "configs/")
SCHEDULER_CANDIDATES = [
    ROOT / "scripts" / "mesh_resource_scheduler.py",
    pathlib.Path(__file__).with_name("mesh-resource-scheduler"),
    pathlib.Path(__file__).with_name("mesh_resource_scheduler.py"),
]
ARBITER_INVENTORY_CANDIDATES = [
    ROOT / "scripts" / "mesh_arbiter_with_inventory.py",
    pathlib.Path(__file__).with_name("mesh-arbiter-with-inventory"),
    pathlib.Path(__file__).with_name("mesh_arbiter_with_inventory.py"),
]


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


def raw_command(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(part) for part in value]
    if isinstance(value, str):
        return shlex.split(value)
    return []


def command_parts(value: Any) -> list[str]:
    parts = [str(value)]
    argv = raw_command(value)
    parts.extend(argv)
    for item in argv:
        if " " not in item and "&&" not in item and ";" not in item:
            continue
        try:
            parts.extend(shlex.split(item))
        except ValueError:
            pass
    return parts


def validate_checked_in_command(
    errors: list[str],
    resource_id: str,
    field: str,
    value: Any,
) -> None:
    parts = command_parts(value)
    if len(parts) <= 1:
        return
    for part in parts:
        if any(pattern in part for pattern in PRIVATE_COMMAND_PATTERNS):
            errors.append(
                f"{resource_id}: {field} must use git-checkout commands, not private wrapper path {part!r}"
            )
        if part.endswith(".sh") and part.startswith(HOST_LOCAL_SCRIPT_PREFIXES):
            errors.append(
                f"{resource_id}: {field} must launch repo-owned scripts, not host-local script {part!r}"
            )
        if part.startswith(REPO_RELATIVE_PREFIXES) and not (ROOT / part).exists():
            errors.append(f"{resource_id}: {field} references missing repo path {part!r}")


def validate_inventory(path: pathlib.Path) -> list[str]:
    scheduler_path = next((item for item in SCHEDULER_CANDIDATES if item.exists()), SCHEDULER_CANDIDATES[0])
    arbiter_inventory_path = next(
        (item for item in ARBITER_INVENTORY_CANDIDATES if item.exists()),
        ARBITER_INVENTORY_CANDIDATES[0],
    )
    scheduler = load_module(
        "mesh_resource_scheduler_for_inventory_check",
        scheduler_path,
    )
    arbiter_inventory = load_module(
        "mesh_arbiter_with_inventory_for_inventory_check",
        arbiter_inventory_path,
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
        control_plane = row.get("control_plane")
        require(errors, control_plane in CONTROL_PLANES,
                f"{resource_id}: control_plane must be one of {sorted(CONTROL_PLANES)!r}")
        if "cuda_probe_cmd" in row:
            validate_checked_in_command(errors, resource_id, "cuda_probe_cmd", row["cuda_probe_cmd"])
        if row.get("status") == "blocked":
            require(errors, control_plane == "blocked",
                    f"{resource_id}: blocked resources must use control_plane=blocked")
        if row.get("status") == "reserved":
            require(errors, control_plane == "reserved",
                    f"{resource_id}: reserved resources must use control_plane=reserved")

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
