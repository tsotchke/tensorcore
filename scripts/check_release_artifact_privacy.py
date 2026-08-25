#!/usr/bin/env python3
"""Reject release artifacts that embed absolute user paths or private networks."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
import sys
import tarfile
import zipfile


HOME_PATTERNS = (
    ("macos_home", re.compile(rb"(?<!:)/Users/[A-Za-z0-9._-]+/")),
    ("linux_home", re.compile(rb"/home/[A-Za-z0-9._-]+/")),
    ("windows_home", re.compile(rb"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/]", re.I)),
)
IPV4_RE = re.compile(rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10." + "0.0.0/8",
        "172." + "16.0.0/12",
        "192.168." + "0.0/16",
        "100." + "64.0.0/10",
    )
)


def private_ip(value: bytes) -> bool:
    try:
        address = ipaddress.ip_address(value.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return False
    return any(address in network for network in PRIVATE_NETWORKS)


def findings_for_bytes(label: str, data: bytes) -> list[str]:
    findings: list[str] = []
    for kind, pattern in HOME_PATTERNS:
        if pattern.search(data):
            findings.append(f"{label}: {kind}")
    if any(private_ip(match.group(0)) for match in IPV4_RE.finditer(data)):
        findings.append(f"{label}: private_network_address")
    return findings


def scan_zip(path: Path) -> list[str]:
    findings: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            findings.extend(findings_for_bytes(f"{path.name}:{info.filename}", archive.read(info)))
    return findings


def scan_tar(path: Path) -> list[str]:
    findings: list[str] = []
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                findings.extend(findings_for_bytes(f"{path.name}:{member.name}", extracted.read()))
    return findings


def scan_path(path: Path) -> list[str]:
    if path.is_dir():
        findings: list[str] = []
        for item in sorted(path.rglob("*")):
            if item.is_file():
                findings.extend(findings_for_bytes(str(item), item.read_bytes()))
        return findings
    if zipfile.is_zipfile(path):
        return scan_zip(path)
    if tarfile.is_tarfile(path):
        return scan_tar(path)
    return findings_for_bytes(str(path), path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    findings: list[str] = []
    for path in args.paths:
        if not path.exists():
            findings.append(f"{path}: missing artifact")
            continue
        findings.extend(scan_path(path))

    for finding in findings:
        print(f"PRIVATE ARTIFACT: {finding}", file=sys.stderr)
    print(f"release artifact privacy: artifacts={len(args.paths)} findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
