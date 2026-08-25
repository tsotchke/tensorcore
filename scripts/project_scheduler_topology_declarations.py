#!/usr/bin/env python3
"""Project private mesh declarations to the scheduler and live accelerator scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def read_object(path: Path) -> dict:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def aliases(row: dict) -> set[str]:
    values = {row.get("name"), row.get("ssh_alias"), row.get("tailscale_name")}
    cloud = row.get("cloud") if isinstance(row.get("cloud"), dict) else {}
    values.add(cloud.get("instance"))
    values.update(row.get("extra_aliases") or [])
    return {str(value).strip().lower() for value in values if str(value or "").strip()}


def cloud_accelerators(row: dict) -> list[str]:
    values = row.get("guestAccelerators") or row.get("accelerators") or []
    out = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, dict):
            raw = value.get("acceleratorType") or value.get("type")
        else:
            raw = value
        if raw:
            out.append(str(raw).split("/")[-1].lower())
    return sorted(set(out))


def project(nodes: dict, inventory: dict, cloud: dict) -> dict:
    source_rows = nodes.get("nodes")
    resources = inventory.get("resources")
    instances = cloud.get("instances")
    if not isinstance(source_rows, list):
        raise ValueError("nodes.nodes must be a list")
    if not isinstance(resources, list):
        raise ValueError("inventory.resources must be a list")
    if not isinstance(instances, list):
        raise ValueError("cloud.instances must be a list")

    required = {
        str(row.get("node") or str(row.get("id") or "").split(":", 1)[0]).lower()
        for row in resources
        if isinstance(row, dict)
        and row.get("id")
        and row.get("status", "active") in {"active", "reserved"}
    }
    selected: dict[str, dict] = {}
    alias_to_name: dict[str, str] = {}
    for row in source_rows:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        name = str(row["name"]).lower()
        for alias in aliases(row):
            alias_to_name[alias] = name
        if name in required or aliases(row).intersection(required):
            selected[name] = row
    missing = sorted(required - set(alias_to_name) - set(selected))
    if missing:
        raise ValueError(f"scheduler nodes missing from declarations: {len(missing)}")
    for required_name in required:
        source_name = alias_to_name.get(required_name, required_name)
        if source_name not in selected:
            selected[source_name] = next(
                row for row in source_rows
                if isinstance(row, dict) and str(row.get("name") or "").lower() == source_name
            )

    synthesized = []
    for instance in instances:
        if not isinstance(instance, dict) or not cloud_accelerators(instance):
            continue
        name = str(instance.get("name") or instance.get("id") or "").split("/")[-1].lower()
        if not name:
            continue
        declared_name = alias_to_name.get(name)
        if declared_name:
            selected.setdefault(
                declared_name,
                next(row for row in source_rows if str(row.get("name") or "").lower() == declared_name),
            )
            continue
        status = str(instance.get("status") or "unknown").lower()
        retired = status in {"terminated", "stopped", "deleted", "deallocated", "suspended"}
        row = {
            "name": name,
            "os": "linux",
            "arch": "unknown",
            "lifecycle": "retired" if retired else "active",
            "roles": ["cloud-accelerator", *( ["retired"] if retired else [] )],
            "capabilities": {
                "compute": {
                    "class": "cloud-accelerator",
                    "accelerators": ["cuda", "nvidia"],
                    "gpu_count": sum(
                        int(value.get("acceleratorCount") or 1)
                        for value in (instance.get("guestAccelerators") or [])
                        if isinstance(value, dict)
                    ) or 1,
                }
            },
        }
        selected[name] = row
        synthesized.append(name)
    return {
        "schema": str(nodes.get("schema") or "computer_mesh.nodes.v1"),
        "nodes": [selected[name] for name in sorted(selected)],
        "projection": {
            "schema": "tensorcore.scheduler_topology_declarations.v1",
            "scheduler_node_count": len(required),
            "selected_node_count": len(selected),
            "synthesized_cloud_accelerator_count": len(synthesized),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-json", required=True, type=Path)
    parser.add_argument("--scheduler-inventory", required=True, type=Path)
    parser.add_argument("--cloud-observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = project(
            read_object(args.nodes_json),
            read_object(args.scheduler_inventory),
            read_object(args.cloud_observation),
        )
    except Exception as exc:
        print(f"scheduler topology declaration projection failed: {exc}", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, **payload["projection"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
