#!/usr/bin/env python3
"""Selftests for scheduler-scoped topology declaration projection."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "project_scheduler_topology_declarations.py"


def load_module() -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("scheduler_topology_projection", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load_module()
    payload = mod.project(
        {"schema": "computer_mesh.nodes.v1", "nodes": [
            {"name": "worker", "tailscale_name": "worker-mesh", "roles": []},
            {"name": "stale-cloud", "cloud": {"instance": "gone"}, "roles": []},
        ]},
        {"resources": [{"id": "worker:cuda", "node": "worker"}]},
        {"instances": [
            {"name": "new-gpu", "status": "TERMINATED", "guestAccelerators": [
                {"acceleratorType": "nvidia-example", "acceleratorCount": 1}
            ]}
        ]},
    )
    names = [row["name"] for row in payload["nodes"]]
    assert names == ["new-gpu", "worker"]
    assert payload["nodes"][0]["lifecycle"] == "retired"
    assert payload["projection"]["synthesized_cloud_accelerator_count"] == 1
    print("scheduler topology declaration projection selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
