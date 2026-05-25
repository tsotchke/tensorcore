from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.errors import SetupError
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel
from wheel.macosx_libfile import extract_macosx_min_system_version


ROOT = Path(__file__).resolve().parent
NATIVE_LIBRARY_NAMES = (
    "libtensorcore.dylib",
    "libtensorcore.so",
    "tensorcore.dll",
    "libtensorcore.dll",
)
NATIVE_OPTIONAL_ARTIFACTS = ("tensorcore.metallib",)
MACOS_ARCH_TAGS = {
    "arm64": {"arm64"},
    "x86_64": {"x86_64"},
    "universal2": {"arm64", "x86_64"},
}


def _artifact_dirs():
    seen = set()

    def add(path):
        if not path:
            return
        p = Path(path).expanduser()
        if p.is_file():
            p = p.parent
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            yield p

    for env_name in ("TENSORCORE_NATIVE_DIR", "TENSORCORE_LIB", "TC_METALLIB"):
        yield from add(os.environ.get(env_name))

    yield from add(ROOT / "build")
    yield from add(ROOT / "build" / "lib")


def _platform_library_names():
    if sys.platform == "darwin":
        return ("libtensorcore.dylib",)
    if sys.platform.startswith("linux"):
        return ("libtensorcore.so",)
    if sys.platform.startswith("win"):
        return ("tensorcore.dll", "libtensorcore.dll")
    return ()


def _metallib_required():
    value = os.environ.get("TENSORCORE_REQUIRE_METALLIB")
    if value is not None:
        return value not in ("0", "false", "False", "no", "NO")
    return sys.platform == "darwin"


def _find_native_artifacts():
    found = {}
    for name in (*NATIVE_LIBRARY_NAMES, *NATIVE_OPTIONAL_ARTIFACTS):
        for directory in _artifact_dirs():
            candidate = directory / name
            if candidate.exists():
                found[name] = candidate
                break
    return found


def _run_tool(args):
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except FileNotFoundError as exc:
        raise SetupError(f"required native validation tool not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SetupError(f"{args[0]} failed while validating native artifact: {output}") from exc


def _dylib_arches(dylib):
    output = _run_tool(["lipo", "-archs", str(dylib)])
    archs = set(output.split())
    if not archs:
        raise SetupError(f"could not determine architectures for {dylib}")
    return archs


def _wheel_macos_version(version):
    major, minor = version[:2]
    if major > 10:
        return major, 0
    return major, minor


def _dylib_macos_version(dylib):
    version = extract_macosx_min_system_version(str(dylib))
    if version is None:
        raise SetupError(f"could not determine minimum macOS version for {dylib}")
    return _wheel_macos_version(version)


def _macos_platform_tags(platform):
    tags = []
    for tag in platform.split("."):
        match = re.fullmatch(r"macosx_(\d+)_(\d+)_(.+)", tag)
        if match:
            version = (int(match.group(1)), int(match.group(2)))
            tags.append((tag, version, match.group(3)))
    return tags


def _validate_dylib_matches_platform_tag(dylib, platform):
    tags = _macos_platform_tags(platform)
    if not tags:
        return

    dylib_arches = _dylib_arches(dylib)
    dylib_macos = _dylib_macos_version(dylib)
    for tag, tag_macos, arch_tag in tags:
        expected_arches = MACOS_ARCH_TAGS.get(arch_tag)
        if expected_arches is None:
            raise SetupError(f"cannot validate unsupported macOS wheel architecture tag: {tag}")
        if not expected_arches.issubset(dylib_arches):
            raise SetupError(
                f"{dylib} contains architectures {sorted(dylib_arches)}, "
                f"but wheel tag {tag} requires {sorted(expected_arches)}"
            )
        if dylib_macos > tag_macos:
            min_tag = f"macosx_{dylib_macos[0]}_{dylib_macos[1]}_{arch_tag}"
            raise SetupError(
                f"{dylib} requires macOS {dylib_macos[0]}.{dylib_macos[1]}, "
                f"but wheel tag {tag} advertises macOS {tag_macos[0]}.{tag_macos[1]}. "
                f"Use {min_tag} or rebuild the native library for an older deployment target."
            )


class build_py_with_native_artifacts(build_py):
    def run(self):
        super().run()
        package_dir = Path(self.build_lib) / "tensorcore"
        package_dir.mkdir(parents=True, exist_ok=True)

        for name, candidate in _find_native_artifacts().items():
            shutil.copy2(candidate, package_dir / name)
            self.announce(
                f"copied native artifact {candidate} -> {package_dir / name}",
                level=2,
            )


class bdist_wheel_with_native_artifacts(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, platform = super().get_tag()
        found = _find_native_artifacts()
        dylib = found.get("libtensorcore.dylib")
        if dylib is not None and Path(self.bdist_dir).exists():
            _validate_dylib_matches_platform_tag(dylib, platform)
        return "py3", "none", platform

    def run(self):
        found = _find_native_artifacts()
        required = []
        platform_libs = _platform_library_names()
        if platform_libs:
            if not any(name in found for name in platform_libs):
                required.append(" or ".join(platform_libs))
        elif not any(name in found for name in NATIVE_LIBRARY_NAMES):
            required.extend(NATIVE_LIBRARY_NAMES)
        if _metallib_required():
            required.extend(NATIVE_OPTIONAL_ARTIFACTS)

        missing = [name for name in required if name not in found]
        if missing:
            searched = ", ".join(str(p) for p in _artifact_dirs())
            lib_hint = (
                " or ".join(platform_libs)
                if platform_libs else
                "one of " + ", ".join(NATIVE_LIBRARY_NAMES)
            )
            raise SetupError(
                "cannot build tensorcore-apple wheel without native artifacts: "
                f"missing {', '.join(missing)}. Build/install tensorcore first "
                "or set TENSORCORE_NATIVE_DIR to a directory containing "
                f"{lib_hint}"
                f"{' and tensorcore.metallib' if _metallib_required() else ''}. "
                f"Searched: {searched}"
            )
        super().run()


setup(
    cmdclass={
        "build_py": build_py_with_native_artifacts,
        "bdist_wheel": bdist_wheel_with_native_artifacts,
    }
)
