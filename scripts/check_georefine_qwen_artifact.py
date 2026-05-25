#!/usr/bin/env python3
"""Validate that a GeoRefine Qwen run is a completed 90% artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def first_number(data: dict[str, Any], *keys: str) -> tuple[str | None, float | None]:
    for key in keys:
        value = number(data.get(key))
        if value is not None:
            return key, value
    return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-size-ratio", type=float, default=0.10)
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"georefine artifact incomplete: {message}")
    return 1


def main() -> int:
    args = parse_args()
    cert_path = args.run_dir / "m2_certificate.json"
    if not cert_path.exists():
        return fail(f"missing {cert_path}")
    try:
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f"could not read certificate: {exc}")
    if not isinstance(cert, dict):
        return fail("certificate is not a JSON object")
    if cert.get("completed") is not True:
        return fail("certificate completed is not true")

    ppl_key, heldout_ppl = first_number(
        cert,
        "ppl_compressed_eval",
        "final_heldout_ppl",
        "final_held_out_ppl",
        "heldout_ppl",
        "held_out_ppl",
        "eval_ppl",
        "ppl",
    )
    if heldout_ppl is None or heldout_ppl <= 0.0:
        return fail("missing positive final held-out PPL")

    size_key, stored_size = first_number(
        cert,
        "size_compressed_bytes",
        "final_stored_size_bytes",
        "stored_size_bytes",
    )
    if stored_size is None or stored_size <= 0.0:
        return fail("missing positive final stored size")

    _, original_size = first_number(cert, "size_original_bytes")
    ratio_key, size_ratio = first_number(
        cert,
        "size_ratio",
        "final_size_ratio",
        "stored_size_ratio",
    )
    if size_ratio is None and original_size and original_size > 0.0:
        ratio_key = "computed_size_ratio"
        size_ratio = stored_size / original_size
    if size_ratio is None:
        return fail("missing final size ratio")
    if size_ratio > args.max_size_ratio:
        return fail(
            f"size ratio {size_ratio:.6f} exceeds max {args.max_size_ratio:.6f}"
        )

    payload = {
        "artifact": str(args.run_dir),
        "completed": True,
        "heldout_ppl": heldout_ppl,
        "heldout_ppl_key": ppl_key,
        "stored_size_bytes": int(stored_size),
        "stored_size_key": size_key,
        "size_ratio": size_ratio,
        "size_ratio_key": ratio_key,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
