#!/usr/bin/env python3
"""Verify that Python public constants match public C enum values."""

from __future__ import annotations

import ast
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "include" / "tensorcore"
PYTHON_BINDING = ROOT / "python" / "tensorcore" / "__init__.py"
CONSTANT_PREFIXES = (
    "TC_ERR_",
    "TC_DTYPE_",
    "TC_FAMILY_",
    "TC_BACKEND_",
    "TC_TIER_",
    "TC_DIST_",
    "TC_HIP_",
    "TC_DILOCO_",
    "TC_REDUCE_",
    "TC_QUANT_",
    "TC_GGUF_TYPE_",
)


def is_public_constant(name: str) -> bool:
    return name == "TC_OK" or name.startswith(CONSTANT_PREFIXES)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*", " ", text)


def c_constants() -> dict[str, int]:
    constants: dict[str, int] = {}
    enum_pattern = re.compile(r"typedef\s+enum\s*\{(?P<body>.*?)\}\s+\w+\s*;", re.S)
    item_pattern = re.compile(r"\b(?P<name>TC_[A-Z0-9_]+)\b(?:\s*=\s*(?P<value>-?\d+))?")

    for path in sorted(HEADER_DIR.glob("*.h")):
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for enum_match in enum_pattern.finditer(text):
            next_value = 0
            for raw_item in enum_match.group("body").split(","):
                item = raw_item.strip()
                if not item:
                    continue
                match = item_pattern.match(item)
                if not match:
                    continue
                name = match.group("name")
                value_text = match.group("value")
                value = int(value_text) if value_text is not None else next_value
                next_value = value + 1
                if is_public_constant(name):
                    constants[name] = value
    return constants


def literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    return None


def python_constants() -> dict[str, int]:
    tree = ast.parse(PYTHON_BINDING.read_text(encoding="utf-8"), filename=str(PYTHON_BINDING))
    constants: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = literal_int(node.value)
        if value is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and is_public_constant(target.id):
                constants[target.id] = value
    return constants


def emit_block(title: str, values: list[str]) -> None:
    if not values:
        return
    print(f"{title}:", file=sys.stderr)
    for value in values:
        print(f"  {value}", file=sys.stderr)


def main() -> int:
    expected = c_constants()
    actual = python_constants()

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        f"{name}: C={expected[name]} Python={actual[name]}"
        for name in set(expected) & set(actual)
        if expected[name] != actual[name]
    )

    if missing or extra or mismatched:
        emit_block("C enum constants missing from Python", missing)
        emit_block("Python constants not declared in public C enums", extra)
        emit_block("Python constants with mismatched values", mismatched)
        return 1

    print(f"python constants OK: {len(expected)} constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
