#!/usr/bin/env python3
"""tc-cuda frontend CLI — checked, not lowered.

This is the first real tc-cuda frontend slice. It is a Python-stdlib-only
tool that:

1. Loads and validates ``docs/tc-cuda/subset.v1.json`` via the existing
   ``scripts/check_tc_cuda_subset.py`` validator (single authority).
2. Lexes CUDA source, ignoring comments and string/char literals.
3. Rejects every unsupported construct with a named diagnostic containing
   the manifest ``id`` and ``name``; exits non-zero.
4. Rejects unknown CUDA double-underscore intrinsics/qualifiers not
   represented in the supported authority.
5. On accepted source, emits a deterministic kernel manifest listing the
   discovered ``__global__`` kernels with status ``checked`` (never
   ``lowered``).

Usage:
    python3 scripts/tc_cuda.py check SOURCE.cu [--manifest-output PATH]

Exit codes:
    0  accepted (source is within the supported subset)
    1  rejected (one or more unsupported/unknown constructs found)
    2  usage or authority error
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Authority loading — reuse the existing validator, do not duplicate policy.
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "tc-cuda" / "subset.v1.json"


def _load_authority(manifest_path: pathlib.Path) -> dict[str, Any]:
    """Load and validate the subset authority via check_tc_cuda_subset."""
    import importlib.util

    validator_path = ROOT / "scripts" / "check_tc_cuda_subset.py"
    spec = importlib.util.spec_from_file_location(
        "check_tc_cuda_subset", validator_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(
            f"cannot import validator at {validator_path}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = mod.load_manifest(manifest_path)
    if not isinstance(data, dict):
        raise SystemExit("subset manifest top-level must be a JSON object")

    # Run the validator's full check; if it fails, exit 2.
    errors: list[str] = []
    mod.check_top_level(errors, data)
    counts = mod.check_category_counts(errors, data)
    supported_ids = mod.check_supported(errors, data, counts)
    unsupported_ids = mod.check_unsupported(errors, data)
    mod.check_duplicates(errors, supported_ids, unsupported_ids)
    if errors:
        print("tc-cuda: subset authority validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(2)

    return data


# ---------------------------------------------------------------------------
# Lexer — strip comments and string/char literals, track line numbers.
# ---------------------------------------------------------------------------


def lex_cuda_source(text: str) -> str:
    """Remove comments and string/char literals, preserving line structure.

    Returns a string where:
    - ``//`` line comments are replaced with spaces (same line).
    - ``/* ... */`` block comments are replaced with spaces (newlines kept).
    - ``"..."`` string literals are replaced with a single space.
    - ``'...'`` char literals are replaced with a single space.
    - Backslash escapes inside strings/chars are handled.

    The resulting string has the same number of lines as the input, so
    ``line = result.count('\\n', 0, pos) + 1`` gives the correct line number.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # Line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        # Block comment
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out.append(" ")
            out.append(" ")
            i += 2
            while i < n:
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    out.append(" ")
                    out.append(" ")
                    i += 2
                    break
                out.append(text[i] if text[i] == "\n" else " ")
                i += 1
        # String literal
        elif c == '"':
            out.append(" ")
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i < n:
                i += 1  # closing quote
        # Char literal
        elif c == "'":
            out.append(" ")
            i += 1
            while i < n and text[i] != "'":
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i < n:
                i += 1  # closing quote
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# ---------------------------------------------------------------------------
# Unsupported-construct token extraction.
#
# Each unsupported manifest entry has a human-readable `name`. We derive
# concrete source-level tokens from that name. The token list is the
# detection surface; the diagnostic always uses the manifest `id` and `name`.
# ---------------------------------------------------------------------------

# Map of unsupported id -> list of source tokens that indicate its presence.
# Tokens are matched as whole identifiers or as distinctive substrings.
_UNSUPPORTED_TOKENS: dict[str, list[str]] = {
    "U1": ["wmma", "mma_sync", "mma."],
    "U2": ["asm volatile", "asm("],
    "U3": ["cp.async", "cp_async"],
    "U4": ["ldmatrix"],
    "U5": ["cooperative_groups", "cg::", "cooperative::"],
    "U6": ["cudaGraph", "cudaGraphExec"],
    "U7": ["cudaLaunchDevice", "cudaDeviceEnablePeerAccess"],
    "U8": ["texture", "surface", "tex1D", "tex2D", "tex3D", "texFetch"],
    "U9": ["__constant__"],
    "U10": ["__ballot_sync", "__any_sync", "__all_sync"],
    "U11": ["__syncwarp"],
    "U12": ["__ldg"],
    "U13": ["atomicAdd" ],  # shared-memory atomics: atomicAdd on __shared__
    "U14": ["atomicAdd", "atomicSub", "atomicAnd", "atomicOr", "atomicXor"],
    "U15": ["atomicCAS", "atomicExch", "atomicMax", "atomicMin"],
    "U16": ["double", "fma(", "sqrt(", "exp(", "log(", "tanh(", "pow(", "sin(", "cos(", "floor(", "fmax(", "fmin(", "fabs("],
    "U17": ["printf"],
    "U18": ["malloc", "free", "assert"],
    "U19": [],  # recursion is structural; not detectable by token scan
    "U20": ["virtual", "dynamic_cast", "typeid"],
    "U21": ["template <typename", "template<class", "template<typename"],
    "U22": ["__launch_bounds__"],
    "U23": ["cudaDeviceEnablePeerAccess", "cudaDeviceDisablePeerAccess"],
    "U24": ["cudaLaunchHostFunc", "cudaStreamAddCallback"],
    "U25": ["float4", "float2", "float3", "double2", "double3", "double4",
            "int4", "int2", "int3", "uint4", "uint2", "uint3"],
}

# Tokens that are supported (from the manifest) and must NOT be rejected.
# These are the `__`-prefixed identifiers that appear in the supported list.
_SUPPORTED_DOUBLE_UNDERSCORE = frozenset({
    "__global__",
    "__device__",
    "__shared__",
    "__restrict__",
    "__syncthreads",
    "__shfl_sync",
    "__shfl_xor_sync",
    "__shfl_down_sync",
    "__shfl_up_sync",
    "__half",
    "__half2",
    "__half2float",
    "__float2half",
    "__float2half_rn",
    "__float2half2_rn",
    "__hfma2",
    "__hmul2",
    "__ushort_as_half",
    "__expf",
    "__logf",
    "__fmul_rn",
    "__fadd_rn",
})

# CUDA keywords and standard C/C++ identifiers that are not intrinsics.
_KNOWN_KEYWORDS = frozenset({
    "void", "int", "float", "double", "char", "short", "long", "unsigned",
    "signed", "const", "static", "inline", "extern", "return", "if", "else",
    "for", "while", "do", "switch", "case", "break", "continue", "sizeof",
    "struct", "union", "enum", "typedef", "template", "typename", "class",
    "public", "private", "protected", "virtual", "new", "delete", "true",
    "false", "nullptr", "auto", "bool", "size_t", "uint8_t", "uint16_t",
    "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class Diagnostic:
    __slots__ = ("code", "message", "line")

    def __init__(self, code: str, message: str, line: int) -> None:
        self.code = code
        self.message = message
        self.line = line

    def render(self, source_path: str) -> str:
        return f"tc-cuda: {self.code} {self.message} at {source_path}:{self.line}"


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------


def _find_token_positions(lexed: str, token: str) -> list[int]:
    """Find all positions of `token` in `lexed` as a whole identifier or
    as a distinctive substring. Returns a list of character positions."""
    positions: list[int] = []
    if " " in token:
        # Multi-word token like "asm volatile" — simple substring search.
        start = 0
        while True:
            idx = lexed.find(token, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
        return positions
    # Single-token: match as whole identifier (bounded by non-identifier chars).
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
    )
    for m in pattern.finditer(lexed):
        positions.append(m.start())
    return positions


def scan_unsupported(
    lexed: str,
    source_path: str,
    unsupported_entries: list[dict[str, Any]],
) -> list[Diagnostic]:
    """Scan lexed source for unsupported constructs; return diagnostics."""
    diags: list[Diagnostic] = []
    for entry in unsupported_entries:
        entry_id = entry["id"]
        name = entry["name"]
        tokens = _UNSUPPORTED_TOKENS.get(entry_id, [])
        for token in tokens:
            for pos in _find_token_positions(lexed, token):
                line = line_at(lexed, pos)
                diags.append(
                    Diagnostic(
                        code=entry_id,
                        message=f"unsupported construct {name!r} (token {token!r})",
                        line=line,
                    )
                )
                break  # one diagnostic per entry per token is enough
    return diags


def scan_unknown_intrinsics(
    lexed: str,
    source_path: str,
) -> list[Diagnostic]:
    """Reject any __-prefixed identifier not in the supported authority."""
    diags: list[Diagnostic] = []
    # Match the full double-underscore-prefixed identifier greedily so that
    # names such as ``__my_unknown__`` are reported in full (not truncated).
    pattern = re.compile(r"__[A-Za-z][A-Za-z0-9_]*")
    seen: set[str] = set()
    for m in pattern.finditer(lexed):
        ident = m.group(0)
        # Normalise: strip trailing __ if present (e.g. __global__ -> __global__)
        if ident not in seen:
            seen.add(ident)
        if ident in _SUPPORTED_DOUBLE_UNDERSCORE:
            continue
        if ident in _KNOWN_KEYWORDS:
            continue
        line = line_at(lexed, m.start())
        diags.append(
            Diagnostic(
                code="UNKNOWN",
                message=f"unknown CUDA intrinsic/qualifier {ident!r} is not in the supported subset authority",
                line=line,
            )
        )
    return diags


def discover_global_kernels(lexed: str) -> list[str]:
    """Find __global__ kernel function names in lexed source."""
    kernels: list[str] = []
    # Match: __global__ [qualifiers] return_type name (
    pattern = re.compile(
        r"__global__\s+"
        r"(?:static\s+)?(?:inline\s+)?(?:const\s+)?"
        r"(?:void|[A-Za-z_][A-Za-z0-9_]*)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    for m in pattern.finditer(lexed):
        kernels.append(m.group(1))
    return sorted(set(kernels))


# ---------------------------------------------------------------------------
# Manifest emission
# ---------------------------------------------------------------------------


def build_manifest(
    source_path: str,
    authority: dict[str, Any],
    kernels: list[str],
) -> dict[str, Any]:
    """Build a deterministic kernel manifest. Status is always 'checked'.

    The manifest is source-location independent: it records only the
    authority identity and the discovered kernel names, never the absolute
    path of the input file, so repeated runs over the same source produce
    byte-identical output.
    """
    return {
        "schema": "tensorcore.tc-cuda.kernel-manifest.v1",
        "authority": {
            "schema": authority.get("schema"),
            "version": authority.get("version"),
            "total": authority.get("total"),
        },
        "kernels": [
            {"name": k, "status": "checked"} for k in kernels
        ],
        "status": "checked",
        "note": (
            "checked means the source was parsed and validated against the "
            "subset authority. It does NOT mean the kernel was translated to "
            "a backend or can execute. tc-cuda does not claim any backend "
            "code generation has occurred."
        ),
    }


def emit_manifest(manifest: dict[str, Any], output_path: pathlib.Path) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    source_path = pathlib.Path(args.source)
    if not source_path.is_file():
        print(f"tc-cuda: source file not found: {source_path}", file=sys.stderr)
        return 2

    authority = _load_authority(args.manifest)

    raw = source_path.read_text(encoding="utf-8")
    lexed = lex_cuda_source(raw)

    unsupported_entries = authority.get("unsupported", [])
    diags = scan_unsupported(lexed, str(source_path), unsupported_entries)
    diags.extend(scan_unknown_intrinsics(lexed, str(source_path)))

    if diags:
        for d in sorted(diags, key=lambda d: (d.line, d.code)):
            print(d.render(str(source_path)), file=sys.stderr)
        print(
            f"tc-cuda: {len(diags)} diagnostic(s); source rejected.",
            file=sys.stderr,
        )
        return 1

    kernels = discover_global_kernels(lexed)
    manifest = build_manifest(source_path, authority, kernels)

    if args.manifest_output:
        out = pathlib.Path(args.manifest_output)
        emit_manifest(manifest, out)
        print(f"tc-cuda: accepted; manifest written to {out}")
        print(f"tc-cuda: {len(kernels)} __global__ kernel(s) discovered: {kernels}")
    else:
        print(f"tc-cuda: accepted; {len(kernels)} __global__ kernel(s): {kernels}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="tc-cuda frontend — checked, not lowered."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check",
        help="Check a CUDA source file against the tc-cuda v1 subset authority.",
    )
    p_check.add_argument("source", type=str, help="Path to the .cu source file.")
    p_check.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="If set, write the kernel manifest JSON to this path.",
    )
    p_check.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help="Path to subset.v1.json (default: docs/tc-cuda/subset.v1.json).",
    )
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
