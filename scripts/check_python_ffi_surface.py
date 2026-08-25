#!/usr/bin/env python3
"""Verify that Python ctypes bindings cover the public tensorcore ABI."""

from __future__ import annotations

import ast
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORTS_FILE = ROOT / "cmake" / "tensorcore.exports"
PYTHON_BINDING = ROOT / "python" / "tensorcore" / "__init__.py"
REQUIRED_PROTO_FIELDS = frozenset({"argtypes", "restype"})


def public_symbols() -> set[str]:
    symbols: set[str] = set()
    for raw in EXPORTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        symbols.add(line[1:] if line.startswith("_") else line)
    return symbols


def ffi_surface() -> tuple[dict[str, set[str]], set[str]]:
    tree = ast.parse(PYTHON_BINDING.read_text(encoding="utf-8"), filename=str(PYTHON_BINDING))
    prototypes: dict[str, set[str]] = {}
    references: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if (
                node.attr.startswith("tc_")
                and isinstance(node.value, ast.Name)
                and node.value.id == "_lib"
            ):
                references.add(node.attr)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                self._record_prototype_target(target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._record_prototype_target(node.target)
            self.generic_visit(node)

        @staticmethod
        def _record_prototype_target(target: ast.AST) -> None:
            if not isinstance(target, ast.Attribute):
                return
            if target.attr not in REQUIRED_PROTO_FIELDS:
                return
            symbol_attr = target.value
            if (
                isinstance(symbol_attr, ast.Attribute)
                and symbol_attr.attr.startswith("tc_")
                and isinstance(symbol_attr.value, ast.Name)
                and symbol_attr.value.id == "_lib"
            ):
                prototypes.setdefault(symbol_attr.attr, set()).add(target.attr)

    Visitor().visit(tree)
    return prototypes, references


def emit_block(title: str, values: list[str]) -> None:
    if not values:
        return
    print(f"{title}:", file=sys.stderr)
    for value in values:
        print(f"  {value}", file=sys.stderr)


def main() -> int:
    expected = public_symbols()
    prototypes, references = ffi_surface()
    bound = set(prototypes)

    missing = sorted(expected - bound)
    extra = sorted(bound - expected)
    incomplete = sorted(
        f"{symbol} missing {', '.join(sorted(REQUIRED_PROTO_FIELDS - fields))}"
        for symbol, fields in prototypes.items()
        if symbol in expected and not REQUIRED_PROTO_FIELDS.issubset(fields)
    )
    unexported_references = sorted(references - expected)
    unprototyped_references = sorted(
        symbol for symbol in references
        if symbol in expected and symbol not in bound
    )

    if missing or extra or incomplete or unexported_references or unprototyped_references:
        emit_block("public exports missing Python ctypes prototypes", missing)
        emit_block("Python ctypes prototypes not in public exports", extra)
        emit_block("incomplete Python ctypes prototypes", incomplete)
        emit_block("_lib references not in public exports", unexported_references)
        emit_block("_lib references without Python ctypes prototypes", unprototyped_references)
        return 1

    print(f"python FFI surface OK: {len(expected)} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
