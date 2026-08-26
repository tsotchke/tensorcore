#!/usr/bin/env python3
"""Run and record mesh ordering, reclamation, topology, and reconnect proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


SCHEMA = "tensorcore.mesh_topology_scale.evidence.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"/(?:Users|home)/[^/\s\"']+", "<home>", text)
    return text[-4000:]


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_dirty(root: Path) -> bool | None:
    try:
        for argv in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            if subprocess.run(argv, cwd=root, check=False).returncode != 0:
                return True
        return False
    except Exception:
        return None


def run(binary: Path, markers: list[str], env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [str(binary)], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, check=False,
    )
    combined = proc.stdout + "\n" + proc.stderr
    missing = [marker for marker in markers if marker not in combined]
    return {
        "status": "passed" if proc.returncode == 0 and not missing else "failed",
        "returncode": proc.returncode,
        "missing_markers": missing,
        "stdout_tail": public_text(proc.stdout),
        "stderr_tail": public_text(proc.stderr),
        "binary_sha256": sha256_file(binary),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.source_root.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    specs = {
        "ordered_and_tagged": (
            build / "tests" / "test_mesh_collective",
            ["bounded snapshot reclamation", "tagged concurrent ordering", "ALL PASS"],
        ),
        "decentralized_three_rank": (
            build / "tests" / "test_mesh_collective_three_rank",
            ["three-rank decentralized AllReduce: OK"],
        ),
        "legacy_reconnect": (
            build / "tests" / "test_remote_tensor_adversarial",
            ["in-place peer reconnect:      OK", "OK"],
        ),
        "authenticated_reconnect": (
            build / "tests" / "test_transport_auth",
            ["in-place reconnect: OK"],
        ),
    }
    for binary, _markers in specs.values():
        if not binary.is_file():
            raise SystemExit(f"required proof binary is missing: {binary.name}")
    env = os.environ.copy()
    env["TC_METALLIB"] = str(build / "tensorcore.metallib")
    results = {
        name: run(binary, markers, env)
        for name, (binary, markers) in specs.items()
    }
    passed = all(row["status"] == "passed" for row in results.values())
    source_dirty = git_dirty(root)
    checks = {
        "runtime_status": "passed" if passed else "failed",
        "runtime_source_status": "clean" if source_dirty is False else "dirty",
        "runtime_peer_reconnect_status": results["legacy_reconnect"]["status"],
        "runtime_authenticated_reconnect_status": results["authenticated_reconnect"]["status"],
        "runtime_bounded_reclamation_status": results["ordered_and_tagged"]["status"],
        "runtime_tagged_ordering_status": results["ordered_and_tagged"]["status"],
        "runtime_decentralized_topology_status": results["decentralized_three_rank"]["status"],
    }
    payload = {
        "schema": SCHEMA,
        "runtime_status": "passed" if passed else "failed",
        "checked_at_unix": time.time(),
        "source_git_head": git_value(root, "rev-parse", "HEAD"),
        "source_git_dirty": source_dirty,
        "results": results,
        "checks": {"tensorcore_distributed_topology_scale": checks},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"runtime_status={payload['runtime_status']} checks={len(results)}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
