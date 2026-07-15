#!/usr/bin/env python3
"""Self-tests for the full-capability campaign validator."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from check_full_capability_campaign import DEFAULT_CAMPAIGN, DEFAULT_FRONTIER, load_json, validate


def expect_error(campaign: dict, frontier: dict, needle: str) -> None:
    errors = validate(campaign, frontier)
    if not any(needle in error for error in errors):
        raise AssertionError(f"expected {needle!r} in {errors!r}")


def main() -> int:
    campaign = load_json(DEFAULT_CAMPAIGN)
    frontier = load_json(DEFAULT_FRONTIER)
    errors = validate(campaign, frontier)
    if errors:
        raise AssertionError(f"valid fixture failed: {errors}")

    invalid_owner = copy.deepcopy(campaign)
    adapter = next(row for row in invalid_owner["tasks"] if row["kind"] == "consumer_adapter")
    adapter["owner_repo"] = "tensorcore"
    expect_error(invalid_owner, frontier, "consumer adapters may not be owned by tensorcore")

    invalid_dependency = copy.deepcopy(campaign)
    invalid_dependency["tasks"][0]["depends_on"] = ["missing-task"]
    expect_error(invalid_dependency, frontier, "unknown dependency")

    invalid_gate = copy.deepcopy(campaign)
    invalid_gate["research_gate_ownership"].pop()
    expect_error(invalid_gate, frontier, "frontier gates without owners")

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "campaign.json"
        path.write_text(json.dumps(campaign), encoding="utf-8")
        if load_json(path)["campaign_id"] != campaign["campaign_id"]:
            raise AssertionError("round-trip load failed")

    print("full capability campaign selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
