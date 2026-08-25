#!/usr/bin/env python3
"""Scenario: new-development-suggestions.

When a user asks "what should I build next?", ICC's
recommendations MUST be grounded in real repo evidence — weakness
map, dead-code, contract-gaps, doc-coverage — not hallucinated
work. This eval probes the recommendation surface by running
`weakness-map` + `capability-roadmap` and asserting:

  1. weakness-map returns ≥1 weakness entry tagged "must-have".
  2. every must-have weakness carries a "Goal" and an "Evidence"
     payload — the evidence is the grounding.
  3. capability-roadmap doesn't fabricate items (current state =
     no roadmap items, so it must return Items=0; if it ever
     returns >0 they MUST resolve to concrete task IDs).

Spec passes when the recommendation surface is artifact-grounded
(weakness with evidence) AND non-fabricated (no phantom roadmap
items).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import result, run_main  # noqa: E402

ICC = Path.home() / "Desktop" / "infinite_context_coder" / "bin" / "icc"


def go():
    if not ICC.exists():
        return result(False, skip_reason=f"icc not found at {ICC}")

    w = subprocess.run([str(ICC), "weakness-map", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=45)
    w_out = w.stdout
    must_have_count = 0
    for line in w_out.splitlines():
        if "Must-have:" in line:
            m = re.search(r"Must-have:\s*`(\d+)`", line)
            if m: must_have_count = int(m.group(1))
    # Every must-have weakness has a Goal and Evidence section.
    weaknesses_grounded = (
        must_have_count >= 1
        and w_out.count("Goal:") >= must_have_count
        and w_out.count("Evidence:") >= must_have_count
    )

    r = subprocess.run([str(ICC), "capability-roadmap", "--repo", "tensorcore"],
                          capture_output=True, text=True, timeout=45)
    r_out = r.stdout
    # Either Items: 0 (truthful: no roadmap) OR every item has a task id.
    items = 0
    for line in r_out.splitlines():
        m = re.search(r"Items:\s*`(\d+)`", line)
        if m: items = int(m.group(1))
    task_ids = set(re.findall(r"--task-id\s+([A-Za-z0-9_.:-]+)", r_out))
    roadmap_non_phantom = (
        (items == 0 and "_No roadmap items._" in r_out) or
        (items > 0 and len(task_ids) >= items)
    )

    return result(
        weaknesses_grounded and roadmap_non_phantom,
        must_have_weaknesses=must_have_count,
        weaknesses_grounded=weaknesses_grounded,
        capability_roadmap_items=items,
        capability_roadmap_non_phantom=roadmap_non_phantom,
    )


if __name__ == "__main__":
    sys.exit(run_main("new-development-suggestions", go))
