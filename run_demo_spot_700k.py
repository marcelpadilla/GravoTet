"""GravoTet supplementary demo: Spot at ~700k vertices.

Solves Poisson and biharmonic on the Spot tetrahedral mesh
(``data/spot.ply``, ~709k vertices).  All shared logic lives in
``gravotet_demo.run_demo``.

Usage:
    python run_demo_spot_700k.py
    python run_demo_spot_700k.py --problems poisson
    python run_demo_spot_700k.py --problems poisson biharmonic --verbose
    python run_demo_spot_700k.py --max-cycles 500
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "python"))

from gravotet_demo import DATA_DIR, make_mesh_from_ply, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Spot 700k",
        output_subdir="spot_700k",
        build_mesh=lambda gravotet: make_mesh_from_ply(
            gravotet, DATA_DIR / "spot.ply",
        ),
        description=__doc__,
    ))
