#!/usr/bin/env python3
"""Validate a Tensorcore operational evidence bundle."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"evidence is not valid JSON at {path}: {exc}") from exc


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def run_checker(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)


def require_path(name: str, path: pathlib.Path | None) -> pathlib.Path:
    if path is None:
        raise SystemExit(f"{name} evidence is required")
    return path.expanduser().resolve()


def require_clean_head(
    label: str,
    data: Any,
    expected_head: str | None,
    head_path: str,
    dirty_path: str,
) -> None:
    if not expected_head:
        raise SystemExit(f"expected git head is unavailable for {label} evidence check")
    actual_dirty = get_path(data, dirty_path)
    if actual_dirty is not False:
        raise SystemExit(f"{label} evidence must be from a clean git tree")
    actual_head = get_path(data, head_path)
    if actual_head != expected_head:
        raise SystemExit(
            f"{label} evidence git_head mismatch: {actual_head!r} != {expected_head!r}"
        )


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def normalize_optional_paths(args: argparse.Namespace) -> None:
    for name in ("release", "sdk26", "cuda", "hip", "pytorch", "live_mesh"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=pathlib.Path)
    parser.add_argument("--sdk26", type=pathlib.Path)
    parser.add_argument("--cuda", type=pathlib.Path)
    parser.add_argument("--hip", type=pathlib.Path)
    parser.add_argument("--pytorch", type=pathlib.Path)
    parser.add_argument("--live-mesh", type=pathlib.Path)
    parser.add_argument("--git-head", default=git_head())
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--require-sdk26", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-hip", action="store_true")
    parser.add_argument("--require-pytorch", action="store_true")
    parser.add_argument("--require-pytorch-backend-allocation", action="store_true")
    parser.add_argument("--require-live-mesh", action="store_true")
    parser.add_argument("--require-release-clean-head", action="store_true")
    parser.add_argument("--require-sdk26-clean-head", action="store_true")
    parser.add_argument("--require-cuda-clean-head", action="store_true")
    parser.add_argument("--require-hip-clean-head", action="store_true")
    parser.add_argument("--require-pytorch-clean-head", action="store_true")
    parser.add_argument("--require-live-clean-head", action="store_true")
    parser.add_argument("--min-live-outer-steps", type=int, default=1)
    parser.add_argument("--require-direct-ring", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-cuda-rank3", action="store_true")
    parser.add_argument("--require-local-only", action="store_true")
    parser.add_argument("--require-rank1-source-prepare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked: list[str] = []

    if args.require_release:
        args.release = require_path("--release", args.release)
    if args.require_sdk26:
        args.sdk26 = require_path("--sdk26", args.sdk26)
    if args.require_cuda:
        args.cuda = require_path("--cuda", args.cuda)
    if args.require_hip:
        args.hip = require_path("--hip", args.hip)
    if args.require_pytorch:
        args.pytorch = require_path("--pytorch", args.pytorch)
    if args.require_live_mesh:
        args.live_mesh = require_path("--live-mesh", args.live_mesh)
    normalize_optional_paths(args)

    if args.release is not None:
        cmd = ["scripts/check_release_evidence.py", str(args.release)]
        if args.require_release_clean_head:
            cmd.extend(["--git-head", args.git_head or "", "--require-clean-head"])
        run_checker(cmd)
        checked.append("release")

    if args.sdk26 is not None:
        cmd = [
            "scripts/check_release_evidence.py",
            str(args.sdk26),
            "--require-metal4-compile",
        ]
        if args.require_sdk26_clean_head:
            cmd.extend(["--git-head", args.git_head or "", "--require-clean-head"])
        run_checker(cmd)
        checked.append("sdk26")

    if args.cuda is not None:
        cmd = ["scripts/check_cuda_smoke_evidence.py", str(args.cuda)]
        if args.require_cuda:
            cmd.append("--require-cuda")
        if args.require_cuda_clean_head:
            cmd.extend(["--git-head", args.git_head or "", "--require-clean-head"])
        run_checker(cmd)
        checked.append("cuda")

    if args.hip is not None:
        cmd = ["scripts/check_hip_smoke_evidence.py", str(args.hip)]
        if args.require_hip:
            cmd.append("--require-hip")
        if args.require_hip_clean_head:
            cmd.extend(["--git-head", args.git_head or "", "--require-clean-head"])
        run_checker(cmd)
        checked.append("hip")

    if args.pytorch is not None:
        cmd = ["scripts/check_pytorch_smoke_evidence.py", str(args.pytorch)]
        if args.require_pytorch:
            cmd.append("--require-pytorch")
        if args.require_pytorch_backend_allocation:
            cmd.append("--require-backend-allocation")
        run_checker(cmd)
        if args.require_pytorch_clean_head:
            require_clean_head(
                "PyTorch",
                load_json(args.pytorch),
                args.git_head,
                "git_head",
                "git_dirty",
            )
        checked.append("pytorch")

    if args.live_mesh is not None:
        cmd = [
            "scripts/check_live_mesh_training_evidence.py",
            str(args.live_mesh),
            "--min-outer-steps",
            str(args.min_live_outer_steps),
        ]
        if args.require_direct_ring:
            cmd.append("--require-direct-ring")
        if args.require_checkpoint:
            cmd.append("--require-checkpoint")
        if args.require_cuda_rank3:
            cmd.append("--require-cuda-rank3")
        if args.require_local_only:
            cmd.append("--require-local-only")
        if args.require_rank1_source_prepare:
            cmd.append("--require-rank1-source-prepare")
        run_checker(cmd)

        live = load_json(args.live_mesh)
        if args.require_live_clean_head:
            require_clean_head(
                "live mesh",
                live,
                args.git_head,
                "meta.git_head",
                "meta.git_dirty",
            )
        checked.append("live_mesh")

    if not checked:
        raise SystemExit("no evidence paths were provided")

    print("operational evidence OK: " + ", ".join(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
