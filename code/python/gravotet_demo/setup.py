"""Build the supplementary `gravotet` C++ extension.

Invoked from `code/python/gravotet_demo/`; C++ sources live under `code/cpp/`
with vendored Eigen headers under `code/deps/eigen/`. No SuiteSparse/CHOLMOD
dependency.
"""

from __future__ import annotations

import platform
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent  # code/python/gravotet_demo/
CODE = HERE.parent.parent               # code/
SRC = CODE / "cpp"
EIGEN = CODE / "deps" / "eigen"

if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc", "/bigobj"]
    extra_link_args: list[str] = []
else:
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC"]
    extra_link_args = []
    if platform.system() == "Darwin":
        extra_compile_args.append("-mmacosx-version-min=10.15")
        extra_link_args.append("-mmacosx-version-min=10.15")

PACKAGE_VERSION = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "gravotet",
        [
            str(SRC / "gravotet_binding.cpp"),
            str(SRC / "multigrid_solver.cpp"),
            str(SRC / "multigrid_solver_vcycle.cpp"),
        ],
        include_dirs=[str(SRC), str(EIGEN)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("VERSION_INFO", PACKAGE_VERSION)],
        language="c++",
    )
]

setup(
    name="gravotet",
    version=PACKAGE_VERSION,
    description="Minimal GravoTet MG supplementary bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
