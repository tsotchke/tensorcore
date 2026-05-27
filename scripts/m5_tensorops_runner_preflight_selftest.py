#!/usr/bin/env python3
"""Fixture tests for m5_tensorops_runner_preflight.py."""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m5_tensorops_runner_preflight.py"

spec = importlib.util.spec_from_file_location("m5_tensorops_runner_preflight", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load m5_tensorops_runner_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def check(status: str) -> dict[str, Any]:
    return {"status": status}


def diagnostic_check(status: str, diagnostic_class: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "diagnostic_class": diagnostic_class,
        "diagnostic_message": message,
    }


def main() -> int:
    assert preflight.version_at_least("26.0", (26, 0))
    assert preflight.version_at_least("26.4", (26, 0))
    assert not preflight.version_at_least("15.2", (26, 0))
    assert not preflight.version_at_least("", (26, 0))

    assert preflight.m5_candidate_from_names(["Apple M5 Max"])
    assert preflight.m5_candidate_from_names(["Apple M6"])
    assert not preflight.m5_candidate_from_names(["Apple M4", "Apple Paravirtual device"])

    assert (
        preflight.runtime_status_from_output("tensorops_runtime_status=passed backend=tensorops_m5")
        == "passed"
    )
    assert (
        preflight.runtime_status_from_output("tensorops_runtime_status=skipped_no_m5 family=Apple10")
        == "skipped_no_m5"
    )
    assert preflight.runtime_status_from_output("no status here") is None

    assert (
        preflight.overall_status(
            {
                "host_platform": check("passed"),
                "xcode": check("passed"),
                "sdk26": check("passed"),
                "display_gpu": check("passed"),
                "tensorops_runtime_probe": check("passed"),
            }
        )
        == "ready"
    )
    assert (
        preflight.overall_status(
            {
                "host_platform": check("passed"),
                "xcode": check("passed"),
                "sdk26": check("passed"),
                "display_gpu": check("passed"),
                "tensorops_runtime_probe": check("skipped"),
            }
        )
        == "candidate"
    )
    assert (
        preflight.overall_status(
            {
                "host_platform": check("passed"),
                "xcode": check("passed"),
                "sdk26": check("blocked"),
                "display_gpu": check("passed"),
                "tensorops_runtime_probe": check("skipped"),
            }
        )
        == "blocked"
    )

    diagnostics = preflight.diagnostics_for_checks(
        {
            "display_gpu": diagnostic_check(
                "blocked",
                preflight.ENVIRONMENT_UNAVAILABLE,
                "display GPU is not M5 or newer",
            ),
            "tensorops_runtime_probe": diagnostic_check(
                "blocked",
                preflight.SOURCE_FAILED,
                "runtime probe exited with 1",
            ),
            "xcode": check("passed"),
        }
    )
    assert [item["name"] for item in diagnostics] == ["display_gpu", "tensorops_runtime_probe"]
    assert diagnostics[0]["diagnostic_class"] == preflight.ENVIRONMENT_UNAVAILABLE
    assert diagnostics[1]["diagnostic_class"] == preflight.SOURCE_FAILED

    missing_binary = preflight.diagnostics_for_checks(
        {
            "tensorops_runtime_probe": diagnostic_check(
                "skipped",
                preflight.ARTIFACT_MISSING,
                "test_tensorops_runtime is not built yet",
            )
        }
    )
    assert [item["diagnostic_class"] for item in missing_binary] == [preflight.ARTIFACT_MISSING]

    print("M5 TensorOps runner preflight selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
