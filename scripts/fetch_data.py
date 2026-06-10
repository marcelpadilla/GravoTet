"""Assemble the large input meshes for the GravoTet supplementary demos.

The cube demo generates its mesh in C++ and needs no input file.  The spot,
sphere, and torus demos load tetrahedral meshes that exceed GitHub's 100 MB
per-file limit, so they are committed to the repository as sub-100 MB parts
under ``data_chunks/`` and concatenated back into ``data/`` here.

The demos do this automatically on first run; this script lets you pre-build
the meshes (for example before going offline or to verify integrity):

    python scripts/fetch_data.py            # assemble every mesh
    python scripts/fetch_data.py spot       # assemble a single mesh by name
    python scripts/fetch_data.py --force    # rebuild even if already present

Files already assembled with the expected size are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the demo package importable so the mesh table stays single-sourced.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "python"))

from gravotet_demo.data_assembly import (  # noqa: E402
    DATA_DIR,
    MESHES,
    assemble_mesh,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes", nargs="*", default=[],
        help=f"Mesh name(s) to assemble from {{{', '.join(MESHES)}}} (default: all).",
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the assembled file is already present.")
    args = parser.parse_args()

    unknown = [m for m in args.meshes if m not in MESHES]
    if unknown:
        raise SystemExit(
            f"Unknown mesh name(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(MESHES)}."
        )

    targets = args.meshes or list(MESHES)
    print(f"Assembling {len(targets)} mesh(es) into {DATA_DIR} ...")
    for name in targets:
        assemble_mesh(name, force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
