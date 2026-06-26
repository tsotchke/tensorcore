#!/usr/bin/env python3
"""Emit ICC-compatible runtime evidence for the documentation truth gate.

Runs `icc doc-intelligence` against tensorcore, computes the
`unsupported_ratio` (unsupported / claim_count), and PASSes when the
ratio stays under a configurable ceiling (default 30%). The ceiling
is intentionally calibrated to today's state — the gate is a
ratchet: tighten the ceiling as docs improve.

Writes JSON keyed under `checks.doc_truth.*`.

Usage:
    python3 scripts/check_doc_truth_runtime.py \\
        --out build/doc_truth_runtime_evidence.json \\
        --ceiling-unsupported-ratio 0.30 \\
        --ceiling-unresolved-ratio 0.40 \\
        --ceiling-hallucination-ratio 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "build" / "doc_truth_runtime_evidence.json"))
    p.add_argument("--ceiling-unsupported-ratio", type=float, default=0.30,
                   help="PASS when unsupported / claim_count <= this. Current baseline ~0.264.")
    p.add_argument("--ceiling-unresolved-ratio", type=float, default=0.40,
                   help="PASS when unresolved_refs / claim_count <= this. Current ~0.332.")
    p.add_argument("--ceiling-hallucination-ratio", type=float, default=0.50,
                   help="PASS when high-hallucination-risk claims / claim_count <= this.")
    p.add_argument("--exclude-path-prefix", action="append", default=None,
                   help="Skip claims in docs under this prefix (repeatable). "
                        "Useful for excluding historical audit docs.")
    return p.parse_args()


def _skip(out_path: Path, reason: str) -> int:
    payload = {"checks": {"doc_truth": {"runtime_status": "skipped",
                                          "skip_reason": reason}}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"runtime_status=skipped reason={reason}")
    return 0


def find_icc():
    candidates = [
        Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc",
        shutil.which("icc") and Path(shutil.which("icc")),
    ]
    for c in candidates:
        if c and c.exists():
            return str(c)
    return None


def main():
    args = parse_args()
    out_path = Path(args.out).expanduser()
    icc = find_icc()
    if not icc:
        return _skip(out_path, "icc CLI not found")

    cmd = [icc, "doc-intelligence", "--repo", "tensorcore",
            "--format", "json", "--claim-limit", "5"]
    for prefix in (args.exclude_path_prefix or []):
        cmd += ["--exclude-path-prefix", prefix]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        payload = {"checks": {"doc_truth": {"runtime_status": "failed",
                                              "failure_reason": f"icc doc-intelligence rc={proc.returncode}",
                                              "stderr_tail": proc.stderr[-1000:]}}}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    d = json.loads(proc.stdout)
    claim_count = max(1, int(d.get("claim_count", 0)))
    sc = d.get("support_counts", {})
    unsupported = int(sc.get("unsupported", 0))
    unresolved = int(sc.get("unresolved_refs", 0))
    grounded   = int(sc.get("grounded", 0))
    risk_counts = d.get("risk_counts", {})
    high_risk  = int(risk_counts.get("high", 0))

    unsupported_ratio = unsupported / claim_count
    unresolved_ratio  = unresolved / claim_count
    hallucination_ratio = high_risk / claim_count

    # Top-10 docs by unsupported count — useful diagnostic in the JSON.
    top_docs = sorted(d.get("docs", []),
                       key=lambda x: -int(x.get("unsupported_count", 0)))[:10]
    top_doc_summary = [
        {"path": doc["path"],
          "unsupported": int(doc.get("unsupported_count", 0)),
          "grounded":    int(doc.get("grounded_count", 0)),
          "unresolved":  int(doc.get("unresolved_ref_count", 0)),
          "claims":      int(doc.get("claim_count", 0))}
        for doc in top_docs
    ]

    # PASS criteria: all three ratios under their ceilings.
    pass_unsupported = unsupported_ratio <= args.ceiling_unsupported_ratio
    pass_unresolved  = unresolved_ratio  <= args.ceiling_unresolved_ratio
    pass_halluc      = hallucination_ratio <= args.ceiling_hallucination_ratio
    overall = pass_unsupported and pass_unresolved and pass_halluc

    payload = {
        "checks": {
            "doc_truth": {
                "runtime_status": "passed" if overall else "failed",
                "claim_count": claim_count,
                "grounded": grounded,
                "unsupported": unsupported,
                "unresolved_refs": unresolved,
                "high_hallucination_risk": high_risk,
                "unsupported_ratio": round(unsupported_ratio, 4),
                "unresolved_ratio": round(unresolved_ratio, 4),
                "hallucination_ratio": round(hallucination_ratio, 4),
                "ceiling_unsupported_ratio": args.ceiling_unsupported_ratio,
                "ceiling_unresolved_ratio": args.ceiling_unresolved_ratio,
                "ceiling_hallucination_ratio": args.ceiling_hallucination_ratio,
                "ratchet_pass_unsupported": pass_unsupported,
                "ratchet_pass_unresolved": pass_unresolved,
                "ratchet_pass_hallucination": pass_halluc,
                "top_10_docs_by_unsupported": top_doc_summary,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "passed" if overall else "failed"
    print(f"runtime_status={status} unsupported_ratio={unsupported_ratio:.3f} "
          f"unresolved_ratio={unresolved_ratio:.3f} "
          f"hallucination_ratio={hallucination_ratio:.3f}")
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
