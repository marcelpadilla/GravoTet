"""Stdlib-only dependency bootstrap for the GravoTet supplementary demos.

The package ``__init__`` imports this module and runs it *before* importing any
submodule that pulls in numpy / scipy / matplotlib / pyvista / PIL at module
load time.  Because of that ordering this module must itself depend on nothing
outside the standard library, so it can run on a clean Python interpreter on
Windows, macOS, or Linux and pip-install whatever is missing.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent

MIN_PY, MAX_PY = (3, 8), (3, 13)

# Import names probed to decide what is missing.  ``setuptools`` is required to
# *build* the C++ extension: ``setup.py`` and ``pybind11.setup_helpers`` import
# it, and on Python 3.12+ it also supplies the ``distutils`` shim that the
# stdlib no longer ships.  ``PIL`` is imported under that name but ships as the
# ``pillow`` distribution; the matching pip specs live in ``requirements.txt``.
REQUIRED = ("numpy", "scipy", "matplotlib", "pybind11", "pyvista", "PIL", "setuptools")


def ensure_python() -> None:
    v = sys.version_info[:2]
    if not (MIN_PY <= v <= MAX_PY):
        raise SystemExit(
            f"This demo is tested with Python "
            f"{MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]} "
            f"(detected {v[0]}.{v[1]})."
        )


def ensure_dependencies() -> None:
    """Install any missing third-party packages into the running interpreter.

    Probes each entry of :data:`REQUIRED`; if anything is missing, installs the
    full pinned set from ``requirements.txt`` via the current interpreter's pip
    (bootstrapping pip with ``ensurepip`` first if necessary).  Shelling out to
    ``sys.executable -m pip`` keeps this working on any OS and inside venvs.
    """
    missing = []
    for module in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return

    try:
        import pip  # noqa: F401
    except ImportError:
        print("Bootstrapping pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

    specs = [
        line.strip()
        for line in (PKG / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Installing missing Python packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *specs])

    # The packages landed in site-packages of the *running* interpreter; clear
    # the import caches so the fresh installs are importable immediately,
    # without having to re-run the script.
    importlib.invalidate_caches()
