from pathlib import Path
import os
import shutil

from setuptools import setup
from setuptools.errors import SetupError
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent
NATIVE_ARTIFACTS = ("libtensorcore.dylib", "tensorcore.metallib")


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


def _find_native_artifacts():
    found = {}
    for name in NATIVE_ARTIFACTS:
        for directory in _artifact_dirs():
            candidate = directory / name
            if candidate.exists():
                found[name] = candidate
                break
    return found


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
        return "py3", "none", platform

    def run(self):
        found = _find_native_artifacts()
        missing = [name for name in NATIVE_ARTIFACTS if name not in found]
        if missing:
            searched = ", ".join(str(p) for p in _artifact_dirs())
            raise SetupError(
                "cannot build tensorcore-apple wheel without native artifacts: "
                f"missing {', '.join(missing)}. Build/install tensorcore first "
                "or set TENSORCORE_NATIVE_DIR to a directory containing "
                f"{' and '.join(NATIVE_ARTIFACTS)}. Searched: {searched}"
            )
        super().run()


setup(
    cmdclass={
        "build_py": build_py_with_native_artifacts,
        "bdist_wheel": bdist_wheel_with_native_artifacts,
    }
)
