#!/usr/bin/env python3
"""Fixture tests for hardware_runner_preflight.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hardware_runner_preflight.py"

spec = importlib.util.spec_from_file_location("hardware_runner_preflight", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load hardware_runner_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def runner(name: str, status: str, labels: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "busy": False,
        "labels": [{"name": label} for label in labels],
    }


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def build(
    tmp: pathlib.Path,
    api_rc: int,
    runners: list[dict[str, Any]] | None = None,
    *,
    require_metal4_tensorops: str = "true",
    required_labels: list[str] | None = None,
) -> dict[str, Any]:
    api_json = tmp / "runners.json"
    api_error = tmp / "runners.err"
    if runners is not None:
        write_json(api_json, {"total_count": len(runners), "runners": runners})
    else:
        api_json.write_text("", encoding="utf-8")
    api_error.write_text("permission denied" if api_rc else "", encoding="utf-8")
    return preflight.build_evidence(
        api_json=api_json,
        api_error=api_error,
        api_rc=api_rc,
        repository="owner/repo",
        require_metal4_tensorops=require_metal4_tensorops,
        required_labels=required_labels or preflight.default_required_labels(require_metal4_tensorops),
        head_sha="abc123",
        run_id="456",
        run_attempt="1",
        workflow="Hardware Evidence",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = pathlib.Path(raw_tmp)

        unavailable = build(tmp, 1, None)
        assert unavailable["status"] == "runner_api_unavailable"
        assert "permission denied" in unavailable["runner_api_error"]
        assert unavailable["meta"]["head_sha"] == "abc123"
        assert unavailable["meta"]["run_id"] == "456"
        assert unavailable["required_labels"] == [
            "self-hosted",
            "macOS",
            "ARM64",
            "m5",
            "sdk26",
            "metal4-tensorops",
        ]
        assert unavailable["diagnostics"][0]["diagnostic_class"] == "token_unavailable"

        missing = build(tmp, 0, [runner("linux", "online", ["self-hosted", "Linux", "X64"])])
        assert missing["status"] == "blocked_no_matching_runner"
        assert missing["registered_runner_count"] == 1
        assert missing["diagnostics"][0]["diagnostic_class"] == "runner_absent"
        assert missing["label_candidate_runners"][0]["name"] == "linux"
        assert missing["label_candidate_runners"][0]["matched_required_labels"] == ["self-hosted"]
        assert missing["label_candidate_runners"][0]["missing_required_labels"] == [
            "ARM64",
            "m5",
            "macOS",
            "metal4-tensorops",
            "sdk26",
        ]
        assert "Closest visible runner" in missing["diagnostics"][0]["recommended_action"]

        offline = build(
            tmp,
            0,
            [runner("m5", "offline", ["self-hosted", "macOS", "ARM64", "m5", "sdk26", "metal4-tensorops"])],
        )
        assert offline["status"] == "matching_runner_offline"
        assert offline["matching_runner_count"] == 1
        assert offline["diagnostics"][0]["diagnostic_class"] == "runner_offline"

        generic = build(
            tmp,
            0,
            [runner("m4", "online", ["self-hosted", "macOS", "ARM64"])],
            require_metal4_tensorops="false",
        )
        assert generic["status"] == "matching_runner_online"
        assert generic["required_labels"] == ["self-hosted", "macOS", "ARM64"]

        online = build(
            tmp,
            0,
            [runner("m5", "online", ["self-hosted", "macOS", "ARM64", "m5", "sdk26", "metal4-tensorops"])],
        )
        assert online["status"] == "matching_runner_online"
        assert online["online_matching_runner_count"] == 1
        assert online["diagnostics"][0]["diagnostic_class"] == "runner_online"
        assert "m5-tensorops-release-smoke" in online["diagnostics"][0]["recommended_action"]
        assert online["label_candidate_runners"][0]["missing_required_labels"] == []

        assert (
            "apple-gpu-release-smoke"
            in generic["diagnostics"][0]["recommended_action"]
        )

        summary = tmp / "summary.md"
        preflight.append_summary(summary, missing)
        text = summary.read_text(encoding="utf-8")
        assert "Hardware Evidence runner preflight" in text
        assert "blocked_no_matching_runner" in text
        assert "Recommended action" in text
        assert "Closest visible runners" in text

    print("hardware runner preflight selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
