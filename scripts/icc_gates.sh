#!/usr/bin/env bash
# icc_gates.sh — run every audit-patterns YAML under .icc/patterns/
# against this repo. Exit 1 on the first non-zero finding count.
#
# This is the "no hardcoded hardware-topology constants" enforcement
# hook, plus any future custom-pattern gates dropped into the same
# directory. Built-in production-audit presets are intentionally not
# touched here — `icc production-audit` already runs those.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ICC:-}" ]]; then
  if command -v icc >/dev/null 2>&1; then
    ICC="$(command -v icc)"
  elif [[ -x "$REPO_ROOT/../infinite_context_coder/bin/icc" ]]; then
    ICC="$REPO_ROOT/../infinite_context_coder/bin/icc"
  else
    ICC="icc"
  fi
fi
PATTERN_DIR="${PATTERN_DIR:-.icc/patterns}"
# Hardcoded (not basename-derived): this repo is ICC-registered as
# "tensorcore", but the checkout dir name isn't reliable — git
# worktrees (e.g. .../tc-gates-worktree) have a different basename
# than the main checkout, and basename-derivation would silently ask
# ICC to audit a repo that was never registered under that name.
REPO_NAME="${REPO_NAME:-tensorcore}"

if [[ ! -x "$ICC" ]]; then
  echo "icc_gates: $ICC not executable; set ICC=<path> to override" >&2
  exit 2
fi

shopt -s nullglob
patterns=("$PATTERN_DIR"/*.yaml "$PATTERN_DIR"/*.yml)
shopt -u nullglob

if [[ ${#patterns[@]} -eq 0 ]]; then
  echo "icc_gates: no pattern files in $PATTERN_DIR — nothing to check"
  exit 0
fi

tmp_json="$(mktemp -t icc_gate_json.XXXXXX)"
tmp_summary="$(mktemp -t icc_gate_sum.XXXXXX)"
trap 'rm -f "$tmp_json" "$tmp_summary"' EXIT

fail=0
for pat in "${patterns[@]}"; do
  # ICC resolves --pattern-file relative to its own repo root (where the
  # index lives), not relative to cwd or the target repo. Always pass abs.
  abs_pat="$(cd "$(dirname "$pat")" && pwd)/$(basename "$pat")"
  echo ">> icc audit-patterns --pattern-file $pat"
  # external/ holds vendored / submodule projects (quantum_geometric_tensor
  # etc.) — each has its own ICC audit story. reference/ is archived demo /
  # legacy corpora kept for citation, not release surface. build/ + .icc/ +
  # tests/fixtures legitimately host violation patterns for self-test.
  "$ICC" audit-patterns --repo "$REPO_NAME" --pattern-file "$abs_pat" \
      --exclude-path-prefix external \
      --exclude-path-prefix reference \
      --exclude-path-prefix build \
      --exclude-path-prefix .icc \
      --exclude-path-prefix tests/fixtures \
      --exclude-path-prefix archive \
      >"$tmp_json" 2>/dev/null || true
  # The tool prints a WARN to stderr (which we suppress) and JSON to stdout.
  if ! python3 - "$tmp_json" <<'PY' >"$tmp_summary"
import json, sys
with open(sys.argv[1]) as fh:
    raw = fh.read()
start = raw.find("{")
if start < 0:
    print("__no_json__")
    sys.exit(0)
d = json.loads(raw[start:])
print(int(d.get("finding_count", 0)))
for f in d.get("findings", []):
    sn = f.get("snippet")
    if isinstance(sn, dict):
        sn = sn.get("content", "")
    sn = (sn or "").strip().replace("\n", " | ")
    print(f"  {f.get('severity','?')} {f.get('path','?')}:{f.get('line','?')} [{f.get('pattern_id','?')}] {sn[:140]}")
PY
  then
    echo "FAIL: $pat -> json parse error"
    cat "$tmp_json"
    fail=1
    continue
  fi
  count="$(head -1 "$tmp_summary")"
  if [[ "$count" == "__no_json__" ]]; then
    echo "FAIL: $pat -> no JSON in output"
    cat "$tmp_json"
    fail=1
    continue
  fi
  if [[ "$count" != "0" ]]; then
    echo "FAIL: $pat -> $count finding(s)"
    tail -n +2 "$tmp_summary"
    fail=1
  else
    echo "ok ($pat)"
  fi
done

if [[ $fail -ne 0 ]]; then
  echo
  echo "icc_gates: at least one pattern reported findings"
  exit 1
fi

echo "icc_gates: all pattern files clean"
