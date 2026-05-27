#!/usr/bin/env python3
"""Fixture tests for fetch_m5_tensorops_runtime_evidence.py."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_m5_tensorops_runtime_evidence.py"

spec = importlib.util.spec_from_file_location("fetch_m5_tensorops_runtime_evidence", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load fetch_m5_tensorops_runtime_evidence.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["fake"], 0, stdout=stdout, stderr="")


def run_list_payload(items: list[dict[str, Any]]) -> str:
    return json.dumps(items)


def preflight_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "repo": "owner/repo",
        "expected_head": "abc123",
        "ref": None,
        "run_id": "123",
        "latest_for_head": False,
        "latest_preflight_for_head": False,
        "runner_preflight": True,
        "require_online_runner": False,
        "cancel_if_no_online_runner": True,
        "dispatch": False,
        "output_dir": pathlib.Path("out"),
        "keep_output_dir": False,
        "run_list_limit": 30,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_latest_run_id_selects_successful_matching_head() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return completed(
            run_list_payload(
                [
                    {
                        "databaseId": 101,
                        "headSha": "other",
                        "status": "completed",
                        "conclusion": "success",
                    },
                    {
                        "databaseId": 102,
                        "headSha": "abc123",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    {
                        "databaseId": 103,
                        "headSha": "abc123",
                        "status": "completed",
                        "conclusion": "success",
                    },
                ]
            )
        )

    original = fetch.run
    fetch.run = fake_run
    try:
        assert fetch.latest_run_id("owner/repo", "abc123", 10) == "103"
        assert calls[0][:5] == ["gh", "run", "list", "--repo", "owner/repo"]
    finally:
        fetch.run = original


def test_latest_run_id_fails_without_match() -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        del cmd
        return completed(
            run_list_payload(
                [
                    {
                        "databaseId": 201,
                        "headSha": "abc123",
                        "status": "in_progress",
                        "conclusion": None,
                    }
                ]
            )
        )

    original = fetch.run
    fetch.run = fake_run
    try:
        try:
            fetch.latest_run_id("owner/repo", "abc123", 10)
        except SystemExit as exc:
            assert "no successful hardware-evidence.yml run found" in str(exc)
        else:
            raise AssertionError("latest_run_id unexpectedly found a run")
    finally:
        fetch.run = original


def test_latest_run_id_can_select_unfinished_preflight_run() -> None:
    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        del cmd
        return completed(
            run_list_payload(
                [
                    {
                        "databaseId": 301,
                        "headSha": "abc123",
                        "status": "queued",
                        "conclusion": None,
                    }
                ]
            )
        )

    original = fetch.run
    fetch.run = fake_run
    try:
        assert (
            fetch.latest_run_id("owner/repo", "abc123", 10, require_success=False)
            == "301"
        )
    finally:
        fetch.run = original


def test_default_dispatch_ref_uses_current_branch() -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git_output(*args: str) -> str:
        calls.append(args)
        if args == ("symbolic-ref", "--short", "HEAD"):
            return "master"
        raise SystemExit("unexpected git call")

    original = fetch.git_output
    fetch.git_output = fake_git_output
    try:
        assert fetch.default_dispatch_ref() == "master"
        assert calls == [("symbolic-ref", "--short", "HEAD")]
    finally:
        fetch.git_output = original


def test_default_dispatch_ref_falls_back_to_origin_head() -> None:
    def fake_git_output(*args: str) -> str:
        if args == ("symbolic-ref", "--short", "HEAD"):
            raise SystemExit("detached")
        if args == ("symbolic-ref", "--short", "refs/remotes/origin/HEAD"):
            return "origin/main"
        raise SystemExit("unexpected git call")

    original = fetch.git_output
    fetch.git_output = fake_git_output
    try:
        assert fetch.default_dispatch_ref() == "main"
    finally:
        fetch.git_output = original


def test_check_dispatch_ref_fails_on_mismatch() -> None:
    original = fetch.resolve_local_ref
    fetch.resolve_local_ref = lambda ref: "actual"
    try:
        try:
            fetch.check_dispatch_ref("master", "expected")
        except SystemExit as exc:
            assert "resolves to actual" in str(exc)
        else:
            raise AssertionError("check_dispatch_ref unexpectedly passed")
    finally:
        fetch.resolve_local_ref = original


def test_cancel_run_calls_gh() -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return completed("")

    original = fetch.run
    fetch.run = fake_run
    try:
        fetch.cancel_run("owner/repo", "123")
        assert calls == [["gh", "run", "cancel", "123", "--repo", "owner/repo"]]
    finally:
        fetch.run = original


def test_cancel_decision_requires_known_no_online_runner() -> None:
    assert fetch.should_cancel_for_runner_preflight({"status": "blocked_no_matching_runner"})
    assert fetch.should_cancel_for_runner_preflight({"status": "matching_runner_offline"})
    assert not fetch.should_cancel_for_runner_preflight({"status": "matching_runner_online"})
    assert not fetch.should_cancel_for_runner_preflight({"status": "runner_api_unavailable"})


def test_main_does_not_cancel_when_runner_api_unavailable() -> None:
    cancelled: list[tuple[str, str]] = []

    def fake_download(
        repo: str,
        run_id: str,
        output_dir: pathlib.Path,
        keep: bool,
    ) -> pathlib.Path:
        del repo, run_id, output_dir, keep
        return pathlib.Path("hardware_runner_preflight.json")

    def fake_validate(
        evidence: pathlib.Path,
        expected_head: str,
        *,
        require_online_runner: bool,
    ) -> dict[str, Any]:
        assert evidence == pathlib.Path("hardware_runner_preflight.json")
        assert expected_head == "abc123"
        assert not require_online_runner
        return {"status": "runner_api_unavailable"}

    original_parse_args = fetch.parse_args
    original_download = fetch.download_runner_preflight
    original_validate = fetch.validate_runner_preflight
    original_cancel = fetch.cancel_run
    fetch.parse_args = preflight_args
    fetch.download_runner_preflight = fake_download
    fetch.validate_runner_preflight = fake_validate
    fetch.cancel_run = lambda repo, run_id: cancelled.append((repo, run_id))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert fetch.main() == 0
        assert cancelled == []
    finally:
        fetch.parse_args = original_parse_args
        fetch.download_runner_preflight = original_download
        fetch.validate_runner_preflight = original_validate
        fetch.cancel_run = original_cancel


def test_main_cancels_when_runner_is_known_offline() -> None:
    cancelled: list[tuple[str, str]] = []

    def fake_download(
        repo: str,
        run_id: str,
        output_dir: pathlib.Path,
        keep: bool,
    ) -> pathlib.Path:
        del repo, run_id, output_dir, keep
        return pathlib.Path("hardware_runner_preflight.json")

    def fake_validate(
        evidence: pathlib.Path,
        expected_head: str,
        *,
        require_online_runner: bool,
    ) -> dict[str, Any]:
        assert evidence == pathlib.Path("hardware_runner_preflight.json")
        assert expected_head == "abc123"
        assert not require_online_runner
        return {"status": "matching_runner_offline"}

    original_parse_args = fetch.parse_args
    original_download = fetch.download_runner_preflight
    original_validate = fetch.validate_runner_preflight
    original_cancel = fetch.cancel_run
    fetch.parse_args = preflight_args
    fetch.download_runner_preflight = fake_download
    fetch.validate_runner_preflight = fake_validate
    fetch.cancel_run = lambda repo, run_id: cancelled.append((repo, run_id))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert fetch.main() == 0
        assert cancelled == [("owner/repo", "123")]
    finally:
        fetch.parse_args = original_parse_args
        fetch.download_runner_preflight = original_download
        fetch.validate_runner_preflight = original_validate
        fetch.cancel_run = original_cancel


def test_main_cancels_before_require_online_failure() -> None:
    cancelled: list[tuple[str, str]] = []

    def fake_download(
        repo: str,
        run_id: str,
        output_dir: pathlib.Path,
        keep: bool,
    ) -> pathlib.Path:
        del repo, run_id, output_dir, keep
        return pathlib.Path("hardware_runner_preflight.json")

    def fake_validate(
        evidence: pathlib.Path,
        expected_head: str,
        *,
        require_online_runner: bool,
    ) -> dict[str, Any]:
        assert evidence == pathlib.Path("hardware_runner_preflight.json")
        assert expected_head == "abc123"
        assert not require_online_runner
        return {"status": "matching_runner_offline"}

    original_parse_args = fetch.parse_args
    original_download = fetch.download_runner_preflight
    original_validate = fetch.validate_runner_preflight
    original_cancel = fetch.cancel_run
    fetch.parse_args = lambda: preflight_args(require_online_runner=True)
    fetch.download_runner_preflight = fake_download
    fetch.validate_runner_preflight = fake_validate
    fetch.cancel_run = lambda repo, run_id: cancelled.append((repo, run_id))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                fetch.main()
            except SystemExit as exc:
                assert "online matching runner required" in str(exc)
            else:
                raise AssertionError("main unexpectedly accepted an offline runner")
        assert cancelled == [("owner/repo", "123")]
    finally:
        fetch.parse_args = original_parse_args
        fetch.download_runner_preflight = original_download
        fetch.validate_runner_preflight = original_validate
        fetch.cancel_run = original_cancel


def main() -> int:
    test_latest_run_id_selects_successful_matching_head()
    test_latest_run_id_fails_without_match()
    test_latest_run_id_can_select_unfinished_preflight_run()
    test_default_dispatch_ref_uses_current_branch()
    test_default_dispatch_ref_falls_back_to_origin_head()
    test_check_dispatch_ref_fails_on_mismatch()
    test_cancel_run_calls_gh()
    test_cancel_decision_requires_known_no_online_runner()
    test_main_does_not_cancel_when_runner_api_unavailable()
    test_main_cancels_when_runner_is_known_offline()
    test_main_cancels_before_require_online_failure()
    print("M5 TensorOps runtime evidence fetch selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
