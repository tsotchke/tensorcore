#!/usr/bin/env python3
"""Forward tsotchke-arbiter calls to the shared arbiter host over SSH.

The mesh scheduler passes JSON metadata as normal argv values.  Plain
``ssh host command arg...`` lets the remote login shell re-parse those
arguments, which breaks on JSON braces and quotes.  This wrapper quotes every
remote argv component before handing it to ssh.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def build_ssh_argv(argv: list[str]) -> list[str]:
    host = require_env("TC_REMOTE_ARBITER_HOST")
    remote = require_env("TC_REMOTE_ARBITER_BIN")
    key = os.environ.get("TC_REMOTE_ARBITER_KEY", "").strip()
    remote_cmd = " ".join(shlex.quote(part) for part in [remote, *argv])
    ssh_argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    if key:
        ssh_argv.extend(["-i", key, "-o", "IdentitiesOnly=yes"])
    ssh_argv.extend([host, remote_cmd])
    return ssh_argv


def main(argv: list[str]) -> int:
    try:
        ssh_argv = build_ssh_argv(argv)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    proc = subprocess.run(ssh_argv, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
