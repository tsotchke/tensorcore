#!/usr/bin/env python3
"""Fail-closed validator for the tc-cuda v1 CUDA subset authority.

`docs/tc-cuda/subset.v1.json` is the sole machine-readable authority for the
tc-cuda v1 subset. The compiler accept-list and all subset documentation are
generated from it, so this validator is the gate that keeps that single
artifact self-consistent. It is fail-closed: any duplicate id, unknown
category, wrong status, missing required string, or unknown key is a
validation failure (exit 1), never a silent acceptance.

Usage:
    python3 scripts/check_tc_cuda_subset.py [path-to-subset.json]

Defaults to docs/tc-cuda/subset.v1.json relative to the repository root.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "tc-cuda" / "subset.v1.json"

SCHEMA = "tensorcore.tc-cuda.subset.v1"
VERSION = "1.0.0"

# Top-level keys the manifest is allowed to carry. Anything else is an
# unknown key and a validation failure (fail-closed).
ALLOWED_TOP_LEVEL = {
    "schema",
    "version",
    "design_ref",
    "total",
    "category_counts",
    "policy",
    "supported",
    "unsupported",
}

# Per-entry keys. Supported entries carry `semantics`; unsupported entries
# carry `reason`. Both carry the common id/name/status.
COMMON_ENTRY_KEYS = {"id", "name", "status"}
ALLOWED_SUPPORTED_KEYS = COMMON_ENTRY_KEYS | {"semantics"}
ALLOWED_UNSUPPORTED_KEYS = COMMON_ENTRY_KEYS | {"reason"}

VALID_STATUS = {"supported", "unsupported"}

# Supported categories are A-J (function forms, launch geometry, memory,
# synchronisation, atomics, half, fp32 math, intrinsics, integer, host API).
# Unsupported entries use the U prefix.
SUPPORTED_CATEGORY_LETTERS = set("ABCDEFGHIJ")
UNSUPPORTED_CATEGORY_LETTER = "U"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help="path to subset.v1.json (default: docs/tc-cuda/subset.v1.json)",
    )
    return parser.parse_args()


def load_manifest(path: pathlib.Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not read subset manifest {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"subset manifest is not valid JSON: {exc}") from exc


def fail(errors: list[str]) -> int:
    print("tc-cuda subset manifest invalid:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_top_level(errors: list[str], data: dict[str, Any]) -> None:
    unknown = sorted(set(data) - ALLOWED_TOP_LEVEL)
    if unknown:
        errors.append(f"unknown top-level key(s): {unknown!r}")

    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}, got {data.get('schema')!r}")
    if data.get("version") != VERSION:
        errors.append(f"version must be {VERSION!r}, got {data.get('version')!r}")
    if not is_nonempty_str(data.get("design_ref")):
        errors.append("design_ref must be a non-empty string")

    if not is_positive_int(data.get("total")):
        errors.append(f"total must be a positive integer, got {data.get('total')!r}")

    if not isinstance(data.get("supported"), list):
        errors.append("supported must be a list")
    if not isinstance(data.get("unsupported"), list):
        errors.append("unsupported must be a list")


def check_category_counts(errors: list[str], data: dict[str, Any]) -> dict[str, int]:
    counts = data.get("category_counts")
    if not isinstance(counts, dict):
        errors.append("category_counts must be an object")
        return {}
    total = 0
    for key, value in counts.items():
        if not is_nonempty_str(key):
            errors.append(f"category_counts key must be a non-empty string, got {key!r}")
            continue
        if not is_positive_int(value):
            errors.append(
                f"category_counts[{key!r}] must be a positive integer, got {value!r}"
            )
            continue
        total += value
    if is_positive_int(data.get("total")) and total != data["total"]:
        errors.append(
            f"category_counts sum {total} must equal total {data['total']}"
        )
    return counts if isinstance(counts, dict) else {}


def check_entry(
    errors: list[str],
    entry: Any,
    index: int,
    section: str,
    allowed_keys: set[str],
    extra_required: tuple[str, ...],
    expected_status: str,
) -> str | None:
    """Validate one entry; return its id if usable, else None."""
    label = f"{section}[{index}]"
    if not isinstance(entry, dict):
        errors.append(f"{label} must be an object, got {type(entry).__name__}")
        return None

    unknown = sorted(set(entry) - allowed_keys)
    if unknown:
        errors.append(f"{label} has unknown key(s): {unknown!r}")

    entry_id = entry.get("id")
    if not is_nonempty_str(entry_id):
        errors.append(f"{label}.id must be a non-empty string, got {entry_id!r}")
        entry_id = None

    if not is_nonempty_str(entry.get("name")):
        errors.append(f"{label}.name must be a non-empty string")

    if entry.get("status") != expected_status:
        errors.append(
            f"{label}.status must be {expected_status!r}, got {entry.get('status')!r}"
        )

    for key in extra_required:
        if not is_nonempty_str(entry.get(key)):
            errors.append(f"{label}.{key} must be a non-empty string")

    return entry_id


def check_supported(
    errors: list[str],
    data: dict[str, Any],
    counts: dict[str, int],
) -> list[str]:
    entries = data.get("supported")
    if not isinstance(entries, list):
        return []
    ids: list[str] = []
    per_category: dict[str, int] = {}
    for index, entry in enumerate(entries):
        entry_id = check_entry(
            errors, entry, index, "supported",
            ALLOWED_SUPPORTED_KEYS, ("semantics",), "supported",
        )
        if entry_id is None:
            continue
        ids.append(entry_id)
        letter = entry_id[0]
        if letter not in SUPPORTED_CATEGORY_LETTERS:
            errors.append(
                f"supported[{index}].id {entry_id!r} must start with one of "
                f"{sorted(SUPPORTED_CATEGORY_LETTERS)!r}"
            )
        else:
            if letter not in counts:
                errors.append(
                    f"supported[{index}].id {entry_id!r} category {letter!r} "
                    "is not declared in category_counts"
                )
            per_category[letter] = per_category.get(letter, 0) + 1

    for letter, seen in sorted(per_category.items()):
        declared = counts.get(letter)
        if isinstance(declared, int) and declared != seen:
            errors.append(
                f"category_counts[{letter!r}] is {declared} but {seen} supported "
                f"entries use category {letter!r}"
            )
    return ids


def check_unsupported(errors: list[str], data: dict[str, Any]) -> list[str]:
    entries = data.get("unsupported")
    if not isinstance(entries, list):
        return []
    ids: list[str] = []
    for index, entry in enumerate(entries):
        entry_id = check_entry(
            errors, entry, index, "unsupported",
            ALLOWED_UNSUPPORTED_KEYS, ("reason",), "unsupported",
        )
        if entry_id is None:
            continue
        ids.append(entry_id)
        if entry_id[0] != UNSUPPORTED_CATEGORY_LETTER:
            errors.append(
                f"unsupported[{index}].id {entry_id!r} must start with "
                f"{UNSUPPORTED_CATEGORY_LETTER!r}"
            )
    return ids


def check_duplicates(errors: list[str], supported_ids: list[str], unsupported_ids: list[str]) -> None:
    all_ids = supported_ids + unsupported_ids
    seen: dict[str, str] = {}
    for entry_id in all_ids:
        if entry_id in seen:
            errors.append(f"duplicate id {entry_id!r} (first seen in {seen[entry_id]!r})")
        else:
            seen[entry_id] = "supported" if entry_id in supported_ids else "unsupported"


def main() -> int:
    args = parse_args()
    data = load_manifest(args.manifest)
    if not isinstance(data, dict):
        return fail(["top-level manifest must be a JSON object"])

    errors: list[str] = []
    check_top_level(errors, data)
    counts = check_category_counts(errors, data)
    supported_ids = check_supported(errors, data, counts)
    unsupported_ids = check_unsupported(errors, data)
    check_duplicates(errors, supported_ids, unsupported_ids)

    if errors:
        return fail(errors)

    print(f"tc-cuda subset manifest OK: {args.manifest} ({data['total']} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
