#!/usr/bin/env python3
"""Validate that a GeoRefine Qwen run is a completed trusted artifact."""

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


def target_kl_value(cert: dict[str, Any]) -> tuple[str | None, float | None]:
    achievement = cert.get("m2_target_kl_achievement")
    if isinstance(achievement, dict):
        for key in ("post_storage_kl_mean", "best_achieved"):
            value = number(achievement.get(key))
            if value is not None:
                return f"m2_target_kl_achievement.{key}", value
    return first_number(cert, "m2_kl_mean", "final_kl_mean", "kl_mean")


def chat_latest_from_cert_or_sidecar(run_dir: Path, cert: dict[str, Any]) -> dict[str, Any] | None:
    chat = cert.get("chat_verification")
    if isinstance(chat, dict) and isinstance(chat.get("latest"), dict):
        return chat["latest"]
    sidecar = run_dir / "m2_chat_verification.json"
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("latest"), dict):
        return payload["latest"]
    return None


def chat_lossless_passed(run_dir: Path, cert: dict[str, Any]) -> bool:
    latest = chat_latest_from_cert_or_sidecar(run_dir, cert)
    if latest is None:
        return False
    probes = latest.get("probes")
    return (
        latest.get("status") == "ok"
        and latest.get("passed") is True
        and not latest.get("blockers")
        and isinstance(probes, dict)
        and bool(probes)
        and all(
            isinstance(probe, dict) and probe.get("passed") is True
            for probe in probes.values()
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-size-ratio", type=float, default=0.10)
    parser.add_argument("--max-ppl-delta", type=float)
    parser.add_argument("--max-target-kl", type=float)
    parser.add_argument(
        "--require-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    if args.require_quality_gate:
        quality_gate = cert.get("quality_gate")
        if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
            return fail("certificate quality_gate did not pass")
    if not chat_lossless_passed(args.run_dir, cert):
        return fail("missing or failed base-vs-artifact chat verification")

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
    _, baseline_ppl = first_number(
        cert,
        "ppl_baseline_eval",
        "baseline_heldout_ppl",
        "baseline_held_out_ppl",
        "baseline_eval_ppl",
    )
    ppl_delta_key, ppl_delta = first_number(
        cert,
        "ppl_delta_fraction_eval",
        "final_heldout_ppl_delta_fraction",
        "heldout_ppl_delta_fraction",
        "held_out_ppl_delta_fraction",
        "eval_ppl_delta_fraction",
    )
    if ppl_delta is None and heldout_ppl and baseline_ppl and baseline_ppl > 0.0:
        ppl_delta_key = "computed_ppl_delta_fraction"
        ppl_delta = (heldout_ppl - baseline_ppl) / baseline_ppl
    if args.max_ppl_delta is not None:
        if ppl_delta is None:
            return fail("missing held-out PPL delta")
        if ppl_delta > args.max_ppl_delta:
            return fail(
                f"PPL delta {ppl_delta:.6f} exceeds max {args.max_ppl_delta:.6f}"
            )

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
    target_kl_key, target_kl = target_kl_value(cert)
    if args.max_target_kl is not None:
        if target_kl is None:
            return fail("missing target KL")
        if target_kl > args.max_target_kl:
            return fail(
                f"target KL {target_kl:.6f} exceeds max {args.max_target_kl:.6f}"
            )

    payload = {
        "artifact": str(args.run_dir),
        "chat_lossless": True,
        "completed": True,
        "heldout_ppl": heldout_ppl,
        "heldout_ppl_key": ppl_key,
        "ppl_delta": ppl_delta,
        "ppl_delta_key": ppl_delta_key,
        "stored_size_bytes": int(stored_size),
        "stored_size_key": size_key,
        "size_ratio": size_ratio,
        "size_ratio_key": ratio_key,
        "target_kl": target_kl,
        "target_kl_key": target_kl_key,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
