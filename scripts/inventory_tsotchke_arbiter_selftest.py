#!/usr/bin/env python3
"""Selftests for the TensorCore inventory to tsotchke-arbiter adapter."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inventory_tsotchke_arbiter.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("inventory_arbiter_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_module()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "inventory.json"
        path.write_text(json.dumps({
            "schema": "tensorcore.mesh_resources.v1",
            "resources": [
                {"id": "node:gpu", "node": "node", "class": "cuda-training", "capacity": 2},
            ],
        }))
        registry = mod.load_registry(path)
    assert registry == {
        "node:gpu": {
            "capacity": 2,
            "class": "cuda-training",
            "node": "node",
            "description": "TensorCore authoritative scheduler inventory",
        }
    }
    assert mod.command_name(["status", "--json"]) == ("status", None)
    assert mod.command_name(["claim", "node:gpu", "--owner", "test"]) == (
        "claim", "node:gpu",
    )
    print("inventory tsotchke arbiter selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
