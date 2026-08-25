#!/usr/bin/env python3
"""Check local Markdown links in the public docs tree."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SKIP_PARTS = {".git", "build", ".cache", ".claude", "__pycache__"}


def iter_markdown(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in SKIP_PARTS or part.startswith("build-") for part in path.parts):
            continue
        out.append(path)
    return sorted(out)


def slugify_heading(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9 _-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def anchors_for(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text()
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slugify_heading(match.group(2))
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def split_link(raw: str) -> tuple[str, str]:
    target = raw.strip()
    target = target.split()[0]
    target = target.split("?", 1)[0]
    if "#" in target:
        file_part, anchor = target.split("#", 1)
        return file_part, anchor
    return target, ""


def should_skip(target: str) -> bool:
    lowered = target.lower()
    return (
        not target
        or lowered.startswith(("http://", "https://", "mailto:", "tel:"))
        or lowered.startswith("#")
        or lowered.startswith("/")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = iter_markdown(root)
    anchor_cache: dict[Path, set[str]] = {}
    checked = 0
    broken: list[str] = []

    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            raw = match.group(1)
            if should_skip(raw):
                continue
            file_part, anchor = split_link(raw)
            if not file_part.endswith(".md"):
                continue
            checked += 1
            target = (source.parent / file_part).resolve()
            if not target.exists():
                broken.append(f"{source.relative_to(root)}: missing {raw}")
                continue
            if root not in target.parents and target != root:
                broken.append(f"{source.relative_to(root)}: escapes repo {raw}")
                continue
            if anchor:
                anchors = anchor_cache.setdefault(target, anchors_for(target))
                if anchor.lower() not in anchors:
                    broken.append(
                        f"{source.relative_to(root)}: missing anchor {raw}"
                    )

    for item in broken:
        print(f"BROKEN: {item}", file=sys.stderr)
    print(f"{checked} local markdown links checked, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
