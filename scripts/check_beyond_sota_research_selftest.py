#!/usr/bin/env python3
"""Self-test the Beyond-SOTA research ledger validator."""

from __future__ import annotations

import copy
import datetime as dt
import json
import pathlib
import tempfile

import check_beyond_sota_research as checker


ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "beyond_sota_research.json"
AS_OF = dt.date(2026, 7, 14)


def validate_payload(payload: dict[str, object]) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "ledger.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return checker.validate_ledger(path, AS_OF)


def main() -> int:
    baseline = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert not validate_payload(baseline)

    non_primary = copy.deepcopy(baseline)
    non_primary["sources"][0]["url"] = "https://example.com/summary"
    assert any("approved primary source" in item for item in validate_payload(non_primary))

    missing_evidence = copy.deepcopy(baseline)
    missing_evidence["gates"][0]["evidence"] = ""
    assert any("evidence is required" in item for item in validate_payload(missing_evidence))

    stale = copy.deepcopy(baseline)
    stale["assessed_at"] = "2026-01-01"
    assert any("ledger is stale" in item for item in validate_payload(stale))

    missing_axis = copy.deepcopy(baseline)
    missing_axis["required_axes"].append("new_frontier")
    errors = validate_payload(missing_axis)
    assert any("missing sources" in item for item in errors)
    assert any("missing gates" in item for item in errors)

    print("beyond-SOTA research checker self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
