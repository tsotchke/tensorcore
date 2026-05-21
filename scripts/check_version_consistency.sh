#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$ROOT" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

pyproject = (root / "pyproject.toml").read_text()
cmake = (root / "CMakeLists.txt").read_text()
header = (root / "include" / "tensorcore" / "tensorcore.h").read_text()

project_match = re.search(r'(?m)^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"\s*$', pyproject)
cmake_match = re.search(
    r"project\s*\(\s*tensorcore\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
    cmake,
    re.MULTILINE,
)
macros = {
    name: value
    for name, value in re.findall(
        r"(?m)^#define\s+TENSORCORE_VERSION_(MAJOR|MINOR|PATCH)\s+([0-9]+)\s*$",
        header,
    )
}

errors = []
if not project_match:
    errors.append("pyproject.toml project.version is missing")
if not cmake_match:
    errors.append("CMakeLists.txt project(tensorcore VERSION ...) is missing")
for key in ("MAJOR", "MINOR", "PATCH"):
    if key not in macros:
        errors.append(f"include/tensorcore/tensorcore.h missing TENSORCORE_VERSION_{key}")

if errors:
    raise SystemExit("\n".join(errors))

project_version = project_match.group(1)
cmake_version = cmake_match.group(1)
header_version = ".".join(macros[key] for key in ("MAJOR", "MINOR", "PATCH"))

versions = {
    "pyproject.toml": project_version,
    "CMakeLists.txt": cmake_version,
    "include/tensorcore/tensorcore.h": header_version,
}

if len(set(versions.values())) != 1:
    lines = ["tensorcore version mismatch:"]
    lines.extend(f"  {path}: {version}" for path, version in versions.items())
    raise SystemExit("\n".join(lines))

print(f"tensorcore version OK: {project_version}")
PY
