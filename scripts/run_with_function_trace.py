#!/usr/bin/env python3
"""Run a Python script and emit stdlib function-call evidence as JSONL."""

from __future__ import annotations

import argparse
import json
import pathlib
import runpy
import sys
import threading
import traceback
from types import FrameType
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Repository-relative source file or directory to include (repeatable).",
    )
    parser.add_argument("script", type=pathlib.Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def repository_path(filename: str) -> str | None:
    try:
        return pathlib.Path(filename).resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return None


def included(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def write_events(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    prefixes = tuple(sorted({pathlib.PurePosixPath(item).as_posix().lstrip("./") for item in args.include}))
    if not prefixes:
        raise SystemExit("at least one --include path is required")

    calls: dict[tuple[str, str, int], dict[str, Any]] = {}

    def profile(frame: FrameType, event: str, arg: object) -> None:
        del arg
        if event != "call":
            return
        path = repository_path(frame.f_code.co_filename)
        if path is None or not included(path, prefixes):
            return
        name = frame.f_code.co_name
        line = int(frame.f_code.co_firstlineno)
        key = (path, name, line)
        calls.setdefault(
            key,
            {
                "kind": "function_called",
                "path": path,
                "line": line,
                "name": name,
                "value": {"trace_source": "sys.setprofile", "executed": True},
                "source_test": args.script.as_posix(),
                "confidence": 1.0,
            },
        )

    script = args.script.resolve()
    if not script.is_file():
        raise SystemExit(f"script not found: {script}")
    exit_code = 0
    previous_argv = sys.argv
    previous_profile = sys.getprofile()
    previous_thread_profile = threading.getprofile()
    sys.argv = [str(script), *args.script_args]
    sys.setprofile(profile)
    threading.setprofile(profile)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.setprofile(previous_profile)
        threading.setprofile(previous_thread_profile)
        sys.argv = previous_argv
        write_events(args.output, [calls[key] for key in sorted(calls)])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
