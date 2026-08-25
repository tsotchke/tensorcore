#!/usr/bin/env python3
"""Focused tests for the tracked-content release privacy gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release_privacy", ROOT / "scripts" / "check_release_privacy.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def kinds(text: str) -> set[str]:
    return {kind for _, kind, _ in MOD.findings_for_text("fixture", text)}


def main() -> int:
    private_mac = "/Users/" + ("t" + "yr") + "/Desktop/project"
    private_linux = "/home/" + ("t" + "yr") + "/project"
    private_windows = "C:/Users/" + ("tsot" + "chke") + "/project"
    private_account = ("tsot" + "chke") + "@workstation.example"
    private_repo = "GeoRefine" + "Internal"

    assert "absolute_user_home" in kinds(private_mac)
    assert "absolute_user_home" in kinds(private_linux)
    assert "absolute_windows_home" in kinds(private_windows)
    assert "account_qualified_host" in kinds(private_account)
    assert "private_infrastructure_literal" in kinds(private_repo)
    assert "private_network_address" in kinds("tcp://100." + "64.1.2:9000")
    assert "private_network_address" in kinds("tcp://192.168." + "1.2:9000")

    public_fixture = "\n".join(
        (
            "$HOME/project",
            "/srv/tensorcore/project",
            "/home/example/project",
            "C:/Users/example/project",
            "builder@windows-host.example",
            "ssh://git@code-host.example/organization/repo.git",
            "tcp://192.0.2.10:9000",
            "tcp://127.0.0.1:9000",
        )
    )
    assert not MOD.findings_for_text("fixture", public_fixture)

    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
        private_path = repo / "receipt.txt"
        private_path.write_text(private_mac + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "receipt.txt"], cwd=repo, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Privacy Test",
                "-c", "user.email=privacy@example.invalid",
                "commit", "-q", "-m", "private fixture",
            ],
            cwd=repo,
            check=True,
        )
        private_path.write_text("$HOME/project\n", encoding="utf-8")
        subprocess.run(["git", "add", "receipt.txt"], cwd=repo, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Privacy Test",
                "-c", "user.email=privacy@example.invalid",
                "commit", "-q", "-m", "clean current tree",
            ],
            cwd=repo,
            check=True,
        )
        history = MOD.history_findings(repo, "HEAD")
        assert any("absolute_user_home" in finding for finding in history)

    print("release privacy selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
