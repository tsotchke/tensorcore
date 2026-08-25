#!/usr/bin/env python3
"""Check the public topology-authority documentation and deployment contract."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


REQUIRED_TEXT = {
    "docs/topology_authority.md": (
        "tensorcore.topology_snapshot.v1",
        "tensorcore.public_topology_snapshot.v1",
        "wrap_topology_observation.py tailscale",
        "wrap_topology_observation.py cloud",
        "--tailscale-profile mesh=",
        "--tailscale-profile personal=",
        "--cloud-inventory gcp=",
        "--require-clean --require-signature",
        "TC_TOPOLOGY_SIGNING_KEY",
        "TC_TOPOLOGY_SNAPSHOT",
        "--allow-unreconciled-topology",
        "win11",
        "CPU-only",
        "no Windows CUDA capacity may be inferred",
    ),
    "docs/mesh_resource_scheduler.md": (
        "TC_TOPOLOGY_SNAPSHOT",
        "--topology-snapshot",
        "--allow-unreconciled-topology",
    ),
    "configs/tensorcore-scheduler.env.example": (
        "TC_TOPOLOGY_SNAPSHOT=",
        "TC_TOPOLOGY_MAX_AGE_SEC=300",
        "TC_TOPOLOGY_SIGNING_KEY=",
    ),
    ".github/workflows/ci.yml": (
        "topology-authority-portable:",
        "os: [ubuntu-latest, macos-14, windows-latest]",
        "python scripts/topology_authority_selftest.py",
        "python scripts/wrap_topology_observation_selftest.py",
        "python3 scripts/topology_scheduler_binding_selftest.py",
        "python3 scripts/check_topology_authority_sprint.py",
    ),
}


def main() -> int:
    errors: list[str] = []
    for relative, needles in REQUIRED_TEXT.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing required topology contract file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{relative} is missing required text {needle!r}")
    for relative in (
        "configs/topology_authority.json",
        "scripts/topology_authority.py",
        "scripts/wrap_topology_observation.py",
        "scripts/topology_scheduler_binding_selftest.py",
        "scripts/check_topology_authority_sprint.py",
    ):
        if not (ROOT / relative).is_file():
            errors.append(f"documented topology path does not exist: {relative}")
    if errors:
        for error in errors:
            print(f"topology documentation error: {error}", file=sys.stderr)
        return 1
    print("topology authority documentation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
