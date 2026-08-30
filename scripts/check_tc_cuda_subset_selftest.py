#!/usr/bin/env python3
"""Regression selftest for the tc-cuda v1 subset validator.

`scripts/check_tc_cuda_subset.py` is the fail-closed gate that keeps
`docs/tc-cuda/subset.v1.json` self-consistent. This selftest proves two things:

  1. The committed real manifest passes the validator (exit 0).
  2. Each representative fail-closed mutation is REJECTED (exit 1) — i.e. the
     validator never silently accepts a duplicate, unknown, or mis-declared
     construct.

It uses only the Python stdlib and writes mutated copies into a temporary
directory (the committed manifest is never modified). It exits 0 only when the
real manifest passes and every mutation is rejected.

Usage:
    python3 scripts/check_tc_cuda_subset_selftest.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "check_tc_cuda_subset.py"
MANIFEST = ROOT / "docs" / "tc-cuda" / "subset.v1.json"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("check_tc_cuda_subset", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_validator(path: pathlib.Path) -> int:
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode


def first_supported(data: dict) -> dict:
    return data["supported"][0]


def first_unsupported(data: dict) -> dict:
    return data["unsupported"][0]


def build_mutations(base: dict) -> list[tuple[str, dict]]:
    """Return (label, mutated-manifest) pairs, each exercising one fail-closed rule."""
    mutations: list[tuple[str, dict]] = []

    # 1. Unknown top-level key.
    m = copy.deepcopy(base)
    m["bogus_top_level"] = "should be rejected"
    mutations.append(("unknown top-level key", m))

    # 2. Wrong schema.
    m = copy.deepcopy(base)
    m["schema"] = "tensorcore.tc-cuda.subset.v2"
    mutations.append(("wrong schema", m))

    # 3. Wrong version.
    m = copy.deepcopy(base)
    m["version"] = "9.9.9"
    mutations.append(("wrong version", m))

    # 4. Incorrect total (mismatched with category_counts sum).
    m = copy.deepcopy(base)
    m["total"] = base["total"] + 1
    mutations.append(("incorrect total", m))

    # 5. Incorrect category_counts (a declared count no longer matches entries).
    m = copy.deepcopy(base)
    m["category_counts"] = copy.deepcopy(base["category_counts"])
    m["category_counts"]["A"] = base["category_counts"]["A"] + 1
    mutations.append(("incorrect category_counts", m))

    # 6. Duplicate supported id (a second entry reuses the first supported id).
    m = copy.deepcopy(base)
    dup = copy.deepcopy(first_supported(m))
    m["supported"].append(dup)
    mutations.append(("duplicate supported id", m))

    # 7. Supported/unsupported overlap (an unsupported entry reuses a supported id).
    m = copy.deepcopy(base)
    overlap = {
        "id": first_supported(m)["id"],
        "name": "cross-section duplicate",
        "status": "unsupported",
        "reason": "should be rejected as a duplicate id",
    }
    m["unsupported"].append(overlap)
    mutations.append(("supported/unsupported id overlap", m))

    # 8. Invalid status on a supported entry.
    m = copy.deepcopy(base)
    first_supported(m)["status"] = "maybe"
    mutations.append(("invalid supported status", m))

    # 9. Unknown entry key on a supported entry.
    m = copy.deepcopy(base)
    first_supported(m)["extra_key"] = "should be rejected"
    mutations.append(("unknown supported entry key", m))

    # 10. Missing required non-string value (name not a string).
    m = copy.deepcopy(base)
    first_supported(m)["name"] = 12345
    mutations.append(("non-string supported name", m))

    # 11. Missing required string value (semantics absent).
    m = copy.deepcopy(base)
    del first_supported(m)["semantics"]
    mutations.append(("missing supported semantics", m))

    # 12. Unknown category letter on a supported entry.
    m = copy.deepcopy(base)
    first_supported(m)["id"] = "Z1"
    mutations.append(("unknown supported category", m))

    # 13. Category membership mismatch (supported id in a category not in category_counts).
    m = copy.deepcopy(base)
    m["category_counts"] = {k: v for k, v in base["category_counts"].items() if k != "A"}
    # total must still be a positive int so per-category comparison is exercised.
    mutations.append(("category membership mismatch", m))

    # 14. Unsupported entry with a non-U id.
    m = copy.deepcopy(base)
    first_unsupported(m)["id"] = "A99"
    mutations.append(("unsupported entry with non-U id", m))

    return mutations


def main() -> int:
    if not VALIDATOR.is_file():
        print(f"FAIL: validator not found at {VALIDATOR}", file=sys.stderr)
        return 1
    if not MANIFEST.is_file():
        print(f"FAIL: manifest not found at {MANIFEST}", file=sys.stderr)
        return 1

    failures: list[str] = []

    # The real committed manifest must pass.
    real_rc = run_validator(MANIFEST)
    if real_rc != 0:
        failures.append(f"real manifest should PASS (exit 0), got exit {real_rc}")
    else:
        print("ok: real manifest passes the validator")

    base = load_real_manifest()
    mutations = build_mutations(base)

    with tempfile.TemporaryDirectory(prefix="tc-cuda-subset-selftest-") as tmp:
        tmp_root = pathlib.Path(tmp)
        for index, (label, mutated) in enumerate(mutations):
            path = tmp_root / f"mutation_{index:02d}.json"
            path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
            rc = run_validator(path)
            if rc == 0:
                failures.append(f"mutation {label!r} should be REJECTED (exit 1), got exit 0")
            else:
                print(f"ok: rejected {label!r}")

    if failures:
        print("tc-cuda subset validator selftest FAILED:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"tc-cuda subset validator selftest OK: real manifest passed, "
          f"{len(mutations)} mutations rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
