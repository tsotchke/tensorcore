#!/usr/bin/env python3
"""Wrap private vendor JSON in timestamped topology observation envelopes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TAILSCALE_SCHEMA = "tensorcore.tailscale_profile_observation.v1"
CLOUD_SCHEMA = "tensorcore.cloud_inventory_observation.v1"


def timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("--observed-at must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def wrap_tailscale(args: argparse.Namespace) -> dict[str, Any]:
    status = read_json(args.status_json)
    if not isinstance(status, dict):
        raise ValueError("--status-json must contain the object emitted by tailscale status --json")
    if status.get("schema") == TAILSCALE_SCHEMA and isinstance(status.get("status"), dict):
        status = status["status"]
    return {
        "schema": TAILSCALE_SCHEMA,
        "profile_id": args.profile_id,
        "observed_at": timestamp(args.observed_at),
        "status": status,
    }


def cloud_instances(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("instances", "items", "inventory"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("--instances-json must contain a list or an object with instances/items/inventory")


def wrap_cloud(args: argparse.Namespace) -> dict[str, Any]:
    instances = cloud_instances(read_json(args.instances_json))
    return {
        "schema": CLOUD_SCHEMA,
        "inventory_id": args.inventory_id,
        "provider": args.provider.strip().lower(),
        "observed_at": timestamp(args.observed_at),
        "instances": instances,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tailscale = subparsers.add_parser("tailscale", help="wrap tailscale status --json")
    tailscale.add_argument("--profile-id", required=True)
    tailscale.add_argument("--status-json", required=True)
    tailscale.add_argument("--observed-at")
    tailscale.add_argument("--output", required=True)
    cloud = subparsers.add_parser("cloud", help="wrap a cloud instance inventory list")
    cloud.add_argument("--inventory-id", required=True)
    cloud.add_argument("--provider", required=True)
    cloud.add_argument("--instances-json", required=True)
    cloud.add_argument("--observed-at")
    cloud.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = wrap_tailscale(args) if args.command == "tailscale" else wrap_cloud(args)
        write_json(args.output, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"topology observation error: {exc}", file=sys.stderr)
        return 1
    count = len(payload.get("instances") or [])
    if args.command == "tailscale":
        peers = payload["status"].get("Peer")
        count = (len(peers) if isinstance(peers, (dict, list)) else 0) + int(
            isinstance(payload["status"].get("Self"), dict)
        )
    print(
        json.dumps(
            {
                "ok": True,
                "schema": payload["schema"],
                "source_id": payload.get("profile_id") or payload.get("inventory_id"),
                "observed_at": payload["observed_at"],
                "item_count": count,
                "output": str(Path(args.output).expanduser()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
