"""Build the tensorcore ↔ PyTorch bridge.

Run:
    cd bindings/pytorch
    python setup.py build_ext --inplace
    python -c "import tensorcore_torch; print(tensorcore_torch.matmul.__doc__)"
"""
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent.parent.parent       # …/tensorcore
INCLUDE_DIR = ROOT / "include"

# Prefer the most recently-built portable-cpu lib (this is the one with the
# AMX + NEON + CBLAS dispatch we care about on Apple Silicon). Caller can
# override via TENSORCORE_LIB_DIR if needed.
LIB_DIR = Path(os.environ.get(
    "TENSORCORE_LIB_DIR",
    str(ROOT / "build-portable-cpu-current")
))
if not (LIB_DIR / "libtensorcore.dylib").exists() and \
   not (LIB_DIR / "libtensorcore.so").exists():
    sys.stderr.write(
        f"WARN: no libtensorcore.{{dylib,so}} at {LIB_DIR}. "
        f"Build tensorcore first (cmake --build {LIB_DIR.name}).\n"
    )

setup(
    name="tensorcore_torch",
    version="0.1.0",
    description="PyTorch ↔ tensorcore bridge (aten::matmul via tc_gemm)",
    packages=["tensorcore_torch"],
    ext_modules=[
        CppExtension(
            name="tensorcore_torch._C",
            sources=["tensorcore_torch_ext.cpp"],
            include_dirs=[str(INCLUDE_DIR)],
            library_dirs=[str(LIB_DIR)],
            libraries=["tensorcore"],
            extra_compile_args=["-std=c++17", "-O3"],
            extra_link_args=[f"-Wl,-rpath,{LIB_DIR}"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
