"""Reset the GravoTet supplementary demo to a clean state.

Removes all runtime-generated files: output figures, JSON summaries, the
compiled C++ extension, and setuptools build artifacts. The source tree and
vendored deps are left untouched.

Run from the repository root or from this directory:

    python .scripts/reset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "gravotet_demo"

# Directories to remove entirely
_DIRS_TO_REMOVE = [
    ROOT / "output",
    PKG / "build",
    PKG / "__pycache__",
]

# Glob patterns for individual files to remove inside gravotet_demo/
_FILE_GLOBS = [
    "gravotet*.pyd",   # Windows compiled extension
    "gravotet*.so",    # Linux / macOS compiled extension
    "gravotet*.dylib", # macOS dylib (if any)
    "*.egg-info",      # setuptools metadata directories
]


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"  removed  {path.relative_to(ROOT)}")
    else:
        print(f"  skipped  {path.relative_to(ROOT)}  (not found)")


def _remove_file(path: Path) -> None:
    if path.is_file():
        path.unlink()
        print(f"  removed  {path.relative_to(ROOT)}")
    elif path.is_dir():  # egg-info shows up as a directory
        shutil.rmtree(path)
        print(f"  removed  {path.relative_to(ROOT)}")


def main() -> None:
    print("Resetting GravoTet supplementary demo...\n")

    print("Directories:")
    for d in _DIRS_TO_REMOVE:
        _remove_dir(d)

    print("\nBuild artifacts in gravotet_demo/:")
    found_any = False
    for pattern in _FILE_GLOBS:
        for match in PKG.glob(pattern):
            _remove_file(match)
            found_any = True
    if not found_any:
        print("  skipped  (none found)")

    print("\nDone. Repository is in a clean state.")
    print("Run  python run_demo.py  to rebuild and re-run the demo.")


if __name__ == "__main__":
    main()
