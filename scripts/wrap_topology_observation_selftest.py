#!/usr/bin/env python3
"""Selftests for wrap_topology_observation.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wrap_topology_observation.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tensorcore-observation-") as raw:
        directory = pathlib.Path(raw)
        tailscale_raw = directory / "tailscale-raw.json"
        cloud_raw = directory / "cloud-raw.json"
        tailscale_output = directory / "tailscale.json"
        cloud_output = directory / "cloud.json"
        tailscale_raw.write_text(
            json.dumps(
                {
                    "BackendState": "Running",
                    "Self": {"HostName": "alpha", "Online": True},
                    "Peer": {"peer": {"HostName": "beta", "Online": False}},
                }
            ),
            encoding="utf-8",
        )
        cloud_raw.write_text(
            json.dumps([{"name": "alpha-cloud", "status": "RUNNING"}]),
            encoding="utf-8",
        )
        tailscale = run(
            "tailscale",
            "--profile-id",
            "mesh",
            "--status-json",
            str(tailscale_raw),
            "--observed-at",
            "2026-07-16T12:00:00-04:00",
            "--output",
            str(tailscale_output),
        )
        assert tailscale.returncode == 0, (tailscale.stdout, tailscale.stderr)
        tailscale_payload = json.loads(tailscale_output.read_text(encoding="utf-8"))
        assert tailscale_payload["schema"] == "tensorcore.tailscale_profile_observation.v1"
        assert tailscale_payload["profile_id"] == "mesh"
        assert tailscale_payload["observed_at"] == "2026-07-16T16:00:00Z"
        assert json.loads(tailscale.stdout)["item_count"] == 2

        cloud = run(
            "cloud",
            "--inventory-id",
            "gcp",
            "--provider",
            "GCP",
            "--instances-json",
            str(cloud_raw),
            "--observed-at",
            "2026-07-16T16:00:00Z",
            "--output",
            str(cloud_output),
        )
        assert cloud.returncode == 0, (cloud.stdout, cloud.stderr)
        cloud_payload = json.loads(cloud_output.read_text(encoding="utf-8"))
        assert cloud_payload["schema"] == "tensorcore.cloud_inventory_observation.v1"
        assert cloud_payload["provider"] == "gcp"
        assert cloud_payload["instances"] == [{"name": "alpha-cloud", "status": "RUNNING"}]

        invalid = directory / "invalid.json"
        invalid.write_text("[]", encoding="utf-8")
        rejected = run(
            "tailscale",
            "--profile-id",
            "mesh",
            "--status-json",
            str(invalid),
            "--output",
            str(directory / "rejected.json"),
        )
        assert rejected.returncode == 1
        assert "must contain the object" in rejected.stderr

    print("topology observation wrapper selftest OK: tailscale, cloud, invalid input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
