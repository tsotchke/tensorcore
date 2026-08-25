#!/usr/bin/env python3
"""Verify that Python public constants match public C enum/macro values."""

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
    "TC_CAPABILITY_",
    "TC_RUNTIME_CAPABILITIES_",
    "TC_TRANSPORT_AUTH_",
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
    macro_pattern = re.compile(
        r"^\s*#define\s+(?P<name>TC_[A-Z0-9_]+)\s+(?P<expr>[^\n]+)$",
        re.M,
    )

    header_texts: list[str] = []
    for path in sorted(HEADER_DIR.glob("*.h")):
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        text = re.sub(r"\\\s*\n", " ", text)
        header_texts.append(text)
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

    pending: list[tuple[str, str]] = []
    for text in header_texts:
        for macro_match in macro_pattern.finditer(text):
            name = macro_match.group("name")
            if is_public_constant(name):
                pending.append((name, macro_match.group("expr").strip()))

    while pending:
        deferred: list[tuple[str, str]] = []
        progress = False
        for name, expression in pending:
            value = c_int_expression(expression, constants)
            if value is None:
                deferred.append((name, expression))
                continue
            constants[name] = value
            progress = True
        if not progress:
            break
        pending = deferred
    return constants


def int_expression(node: ast.AST, constants: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = int_expression(node.operand, constants)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Invert):
            return ~operand
    if isinstance(node, ast.BinOp):
        left = int_expression(node.left, constants)
        right = int_expression(node.right, constants)
        if left is None or right is None:
            return None
        operations = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.BitOr: lambda a, b: a | b,
            ast.BitAnd: lambda a, b: a & b,
            ast.BitXor: lambda a, b: a ^ b,
            ast.LShift: lambda a, b: a << b,
            ast.RShift: lambda a, b: a >> b,
        }
        operation = operations.get(type(node.op))
        return None if operation is None else operation(left, right)
    return None


def c_int_expression(expression: str, constants: dict[str, int]) -> int | None:
    if "sizeof" in expression or "offsetof" in expression:
        return None
    normalized = expression
    wrapper = re.compile(r"\bU?INT(?:8|16|32|64)_C\(([^()]+)\)")
    while wrapper.search(normalized):
        normalized = wrapper.sub(r"(\1)", normalized)
    normalized = re.sub(
        r"\(\s*(?:size_t|unsigned|signed|int|long|uint(?:8|16|32|64)_t)\s*\)",
        "",
        normalized,
    )
    normalized = re.sub(r"(?<=\d)[uUlL]+\b", "", normalized)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return None
    return int_expression(parsed.body, constants)


def python_constants() -> dict[str, int]:
    tree = ast.parse(PYTHON_BINDING.read_text(encoding="utf-8"), filename=str(PYTHON_BINDING))
    constants: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = int_expression(node.value, constants)
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
        emit_block("C constants missing from Python", missing)
        emit_block("Python constants not declared in public C headers", extra)
        emit_block("Python constants with mismatched values", mismatched)
        return 1

    print(f"python constants OK: {len(expected)} constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
