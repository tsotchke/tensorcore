#!/usr/bin/env python3
"""Selftests for scripts/check_mesh_resource_preflight_evidence.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_mesh_resource_preflight_evidence.py"


def evidence(*, ok: bool = True, reason: str = "preflight_ok") -> dict[str, Any]:
    return {
        "schema": "tensorcore.mesh_resource_preflights.v1",
        "ok": ok,
        "checked_at_unix": int(time.time()),
        "jobs_checked": 1,
        "missing_job_ids": [],
        "skipped_default_job_ids": [],
        "results": [
            {
                "job": "georefine-m2-cosbox",
                "resource": "cosbox:cuda3090",
                "ok": ok,
                "reason": reason,
                "rc": 0 if ok else 1,
                "json": {
                    "schema": "unit.preflight.v1",
                    "ok": ok,
                    "resource": "cosbox:cuda3090",
                    "reason": reason,
                },
            }
        ],
    }


def write_json(directory: pathlib.Path, name: str, data: dict[str, Any]) -> pathlib.Path:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def assert_passes(*args: str) -> None:
    result = run_checker(*args)
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)


def assert_fails(needle: str, *args: str) -> None:
    result = run_checker(*args)
    if result.returncode == 0:
        raise AssertionError("checker unexpectedly passed")
    output = result.stderr + result.stdout
    if needle not in output:
        raise AssertionError(f"expected {needle!r} in checker output:\n{output}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        ok_path = write_json(directory, "ok.json", evidence())
        assert_passes(str(ok_path), "--require-pass", "--require-job", "georefine-m2-cosbox")

        failed_path = write_json(
            directory,
            "failed.json",
            evidence(ok=False, reason="git_publickey_denied"),
        )
        assert_passes(str(failed_path))
        assert_passes(
            str(failed_path),
            "--require-job",
            "georefine-m2-cosbox",
            "--allow-failure",
            "georefine-m2-cosbox:git_publickey_denied",
        )
        assert_fails("preflights failed", str(failed_path), "--require-pass")
        assert_fails(
            "allowed failure reason mismatch",
            str(failed_path),
            "--allow-failure",
            "georefine-m2-cosbox:wrong_reason",
        )
        assert_fails(
            "allowed failure jobs missing",
            str(failed_path),
            "--allow-failure",
            "old-donkey-precompute-chain:git_publickey_denied",
        )
        assert_fails(
            "--allow-failure values must use job_id:reason",
            str(failed_path),
            "--allow-failure",
            "georefine-m2-cosbox",
        )
        assert_fails(
            "allowed failure job passed unexpectedly",
            str(ok_path),
            "--allow-failure",
            "georefine-m2-cosbox:preflight_ok",
        )

        missing = evidence()
        missing["missing_job_ids"] = ["missing-job"]
        missing_path = write_json(directory, "missing.json", missing)
        assert_fails("missing_job_ids must be empty", str(missing_path))

        bad_skipped = evidence()
        bad_skipped["skipped_default_job_ids"] = "jack-cuda3060-smoke"
        bad_skipped_path = write_json(directory, "bad-skipped.json", bad_skipped)
        assert_fails("skipped_default_job_ids must be a list", str(bad_skipped_path))

        mismatch = evidence()
        mismatch["jobs_checked"] = 2
        mismatch_path = write_json(directory, "mismatch.json", mismatch)
        assert_fails("jobs_checked must match", str(mismatch_path))

        missing_resource = evidence()
        del missing_resource["results"][0]["resource"]
        missing_resource_path = write_json(directory, "missing-resource.json", missing_resource)
        assert_fails("resource is required", str(missing_resource_path))

        non_boolean_ok = evidence()
        non_boolean_ok["results"][0]["ok"] = "true"
        non_boolean_ok_path = write_json(directory, "non-boolean-ok.json", non_boolean_ok)
        assert_fails("ok must be boolean", str(non_boolean_ok_path))

        non_boolean_top = evidence()
        non_boolean_top["ok"] = "true"
        non_boolean_top_path = write_json(directory, "non-boolean-top.json", non_boolean_top)
        assert_fails("top-level ok must be boolean", str(non_boolean_top_path))

        inconsistent_top = evidence(ok=False, reason="git_publickey_denied")
        inconsistent_top["ok"] = True
        inconsistent_top_path = write_json(directory, "inconsistent-top.json", inconsistent_top)
        assert_fails("top-level ok must match result rows", str(inconsistent_top_path))

        duplicate = evidence()
        duplicate["jobs_checked"] = 2
        duplicate["results"].append(dict(duplicate["results"][0]))
        duplicate_path = write_json(directory, "duplicate.json", duplicate)
        assert_fails("duplicate result job", str(duplicate_path))

        bad_nested = evidence()
        bad_nested["results"][0]["json"] = "not-object"
        bad_nested_path = write_json(directory, "bad-nested.json", bad_nested)
        assert_fails("json must be an object", str(bad_nested_path))

        missing_nested = evidence()
        del missing_nested["results"][0]["json"]
        missing_nested_path = write_json(directory, "missing-nested.json", missing_nested)
        assert_fails("json is required when ok=true", str(missing_nested_path))

        nested_missing_schema = evidence()
        del nested_missing_schema["results"][0]["json"]["schema"]
        nested_missing_schema_path = write_json(directory, "nested-missing-schema.json", nested_missing_schema)
        assert_fails("json.schema is required", str(nested_missing_schema_path))

        nested_resource_mismatch = evidence()
        nested_resource_mismatch["results"][0]["json"]["resource"] = "other:cuda"
        nested_resource_mismatch_path = write_json(directory, "nested-resource-mismatch.json", nested_resource_mismatch)
        assert_fails("json.resource must match", str(nested_resource_mismatch_path))

        missing_rc = evidence()
        del missing_rc["results"][0]["rc"]
        missing_rc_path = write_json(directory, "missing-rc.json", missing_rc)
        assert_fails("rc must be 0", str(missing_rc_path))

        nonzero_rc = evidence()
        nonzero_rc["results"][0]["rc"] = 1
        nonzero_rc_path = write_json(directory, "nonzero-rc.json", nonzero_rc)
        assert_fails("rc must be 0", str(nonzero_rc_path))

        nested_not_ok = evidence()
        nested_not_ok["results"][0]["json"]["ok"] = False
        nested_not_ok_path = write_json(directory, "nested-not-ok.json", nested_not_ok)
        assert_fails("json.ok must be true", str(nested_not_ok_path))

        stale = evidence()
        stale["checked_at_unix"] = 1
        stale_path = write_json(directory, "stale.json", stale)
        assert_fails("evidence is stale", str(stale_path), "--max-age-sec", "1")

        empty = evidence()
        empty["jobs_checked"] = 0
        empty["results"] = []
        empty["skipped_default_job_ids"] = ["jack-cuda3060-smoke"]
        empty_path = write_json(directory, "empty.json", empty)
        assert_passes(str(empty_path))
        assert_passes(
            str(empty_path),
            "--require-skipped-default-job",
            "jack-cuda3060-smoke",
        )
        assert_fails("at least one preflight result", str(empty_path), "--require-pass")
        assert_fails(
            "required skipped default jobs missing",
            str(empty_path),
            "--require-skipped-default-job",
            "old-donkey-precompute-chain",
        )

        assert_fails("required jobs missing", str(ok_path), "--require-job", "old-donkey-precompute-chain")

    print("mesh resource preflight evidence selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
