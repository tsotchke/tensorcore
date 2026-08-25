#!/usr/bin/env python3
"""Selftests for scripts/mesh_arbiter_with_inventory.py."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import pathlib
import tempfile
from types import ModuleType


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_CANDIDATES = [
    ROOT / "scripts" / "mesh_arbiter_with_inventory.py",
    pathlib.Path(__file__).with_name("mesh-arbiter-with-inventory"),
    pathlib.Path(__file__).with_name("mesh_arbiter_with_inventory.py"),
]


def load_module() -> ModuleType:
    path = next((item for item in SCRIPT_CANDIDATES if item.exists()), SCRIPT_CANDIDATES[0])
    loader = importlib.machinery.SourceFileLoader("mesh_arbiter_with_inventory_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_inventory(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "resources.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tensorcore.mesh_resources.v1",
                "resources": [
                    {
                        "id": "atlas:metal_m2ultra",
                        "node": "atlas",
                        "class": "metal-large-unified",
                        "capacity": 1,
                        "description": "Atlas Metal",
                    },
                    {
                        "id": "enki:metal_m4_tsotchke_chan",
                        "node": "enki",
                        "class": "metal-reserved",
                        "capacity": 1,
                        "description": "Reserved M4 Metal",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_inventory_converts_to_arbiter_capacities() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_inventory(pathlib.Path(tmp))
        capacities = mod.load_inventory_capacities(str(path))
    assert capacities["atlas:metal_m2ultra"]["class"] == "metal-large-unified"
    assert capacities["enki:metal_m4_tsotchke_chan"]["node"] == "enki"
    assert capacities["atlas:metal_m2ultra"]["backend"] is None
    assert capacities["atlas:metal_m2ultra"]["status"] == "active"
    assert capacities["atlas:metal_m2ultra"]["general_queue_eligible"] is True


def test_inventory_overrides_existing_env_rows() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = write_inventory(pathlib.Path(tmp))
        merged = mod.merged_capacities(
            str(path),
            json.dumps({"atlas:metal_m2ultra": {"capacity": 99}, "custom": {"capacity": 2}}),
        )
    assert merged["atlas:metal_m2ultra"]["capacity"] == 1
    assert merged["custom"]["capacity"] == 2


def test_inventory_rejects_duplicate_resources() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "resources.json"
        path.write_text(
            json.dumps({
                "schema": "tensorcore.mesh_resources.v1",
                "resources": [
                    {"id": "cosbox:cuda3090", "capacity": 1},
                    {"id": "cosbox:cuda3090", "capacity": 1},
                ],
            }),
            encoding="utf-8",
        )
        try:
            mod.load_inventory_capacities(str(path))
        except ValueError as exc:
            assert "duplicate resource" in str(exc)
        else:
            raise AssertionError("duplicate inventory resource was accepted")


def test_inventory_rejects_bad_status() -> None:
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "resources.json"
        path.write_text(
            json.dumps({
                "schema": "tensorcore.mesh_resources.v1",
                "resources": [{"id": "cosbox:cuda3090", "status": "offline"}],
            }),
            encoding="utf-8",
        )
        try:
            mod.load_inventory_capacities(str(path))
        except ValueError as exc:
            assert "status must be one of" in str(exc)
        else:
            raise AssertionError("bad inventory status was accepted")


def main() -> int:
    test_inventory_converts_to_arbiter_capacities()
    test_inventory_overrides_existing_env_rows()
    test_inventory_rejects_duplicate_resources()
    test_inventory_rejects_bad_status()
    print("mesh arbiter inventory selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
