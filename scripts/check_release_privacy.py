#!/usr/bin/env python3
"""Fail when tracked release content exposes private host or network details."""

from __future__ import annotations

import argparse
import io
import ipaddress
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOME_NAMES = {"...", "cos", "example", "runner", "test", "user"}
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10." + "0.0.0/8",
        "172." + "16.0.0/12",
        "192.168." + "0.0/16",
        "100." + "64.0.0/10",
    )
)
IPV4_RE = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
UNIX_HOME_RE = re.compile(r"/(?P<kind>Users|home)/(?P<name>[A-Za-z0-9._-]+)/")
WINDOWS_HOME_RE = re.compile(
    r"[A-Za-z]:[\\/]Users[\\/](?P<name>[A-Za-z0-9._-]+)[\\/]",
    re.IGNORECASE,
)

# Build sensitive literals from fragments so this checker does not flag its
# own source while still keeping the denylist explicit and reviewable.
PRIVATE_USER_NAMES = ("t" + "yr", "tsot" + "chke")
FORBIDDEN_LITERALS = (
    "GeoRefine" + "Internal",
    "georefine-" + "rtxpro-spot",
    "georefine-" + "embed-bench",
)
def tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    )
    return [root / item.decode("utf-8") for item in raw.split(b"\0") if item]


def private_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(address in network for network in PRIVATE_NETWORKS)


def findings_for_text(path: str, text: str) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        unix = UNIX_HOME_RE.search(line)
        if unix and unix.group("name").lower() not in ALLOWED_HOME_NAMES:
            findings.append((line_number, "absolute_user_home", unix.group(0)))

        windows = WINDOWS_HOME_RE.search(line)
        if windows and windows.group("name").lower() not in ALLOWED_HOME_NAMES:
            findings.append((line_number, "absolute_windows_home", windows.group(0)))

        lowered = line.lower()
        for name in PRIVATE_USER_NAMES:
            if f"{name}@" in lowered:
                findings.append((line_number, "account_qualified_host", f"{name}@..."))

        for literal in FORBIDDEN_LITERALS:
            if literal.lower() in lowered:
                findings.append((line_number, "private_infrastructure_literal", literal))

        for match in IPV4_RE.finditer(line):
            value = match.group(0)
            if private_ip(value):
                findings.append((line_number, "private_network_address", value))
    return findings


def history_findings(root: Path, ref: str) -> list[str]:
    listing = subprocess.run(
        ["git", "rev-list", "--objects", ref],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(listing.stderr.strip() or f"git rev-list failed for {ref}")

    object_paths: dict[str, str] = {}
    object_ids: list[str] = []
    for line in listing.stdout.splitlines():
        fields = line.split(" ", 1)
        object_id = fields[0]
        if not re.fullmatch(r"[0-9a-f]{40}", object_id) or object_id in object_paths:
            continue
        object_ids.append(object_id)
        object_paths[object_id] = fields[1] if len(fields) == 2 else "<metadata>"

    batch_input = "".join(f"{object_id}\n" for object_id in object_ids)
    checked = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        text=True,
        input=batch_input,
        capture_output=True,
        check=False,
    )
    if checked.returncode != 0:
        raise RuntimeError(checked.stderr.strip() or "git cat-file --batch-check failed")

    blob_ids: list[str] = []
    for line in checked.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob":
            blob_ids.append(fields[0])

    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, stderr_bytes = process.communicate(
        "".join(f"{object_id}\n" for object_id in blob_ids).encode("ascii")
    )
    if process.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(stderr.strip() or "git cat-file --batch failed")
    stream = io.BytesIO(output)

    findings: list[str] = []
    for expected_id in blob_ids:
        header = stream.readline().decode("ascii", errors="replace").strip()
        fields = header.split()
        if len(fields) != 3 or fields[0] != expected_id or fields[1] != "blob":
            process.kill()
            raise RuntimeError(f"unexpected git cat-file header for {expected_id[:12]}")
        size = int(fields[2])
        payload = stream.read(size)
        stream.read(1)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        path = object_paths.get(expected_id, "<unknown>")
        seen_kinds: set[str] = set()
        for _line, kind, _sample in findings_for_text(path, text):
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            findings.append(f"history-blob:{expected_id[:12]}:{path}: {kind}")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also reject known private literals anywhere in --history-ref ancestry",
    )
    parser.add_argument("--history-ref", default="HEAD")
    args = parser.parse_args()
    root = args.root.resolve()

    findings: list[str] = []
    scanned = 0
    for path in tracked_files(root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        rel = path.relative_to(root).as_posix()
        for line, kind, sample in findings_for_text(rel, text):
            findings.append(f"{rel}:{line}: {kind}: {sample}")

    for finding in findings:
        print(f"PRIVATE: {finding}", file=sys.stderr)
    if args.history:
        findings.extend(history_findings(root, args.history_ref))
        for finding in findings:
            if finding.startswith("history-blob:"):
                print(f"PRIVATE: {finding}", file=sys.stderr)
    print(f"release privacy: scanned={scanned} findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
