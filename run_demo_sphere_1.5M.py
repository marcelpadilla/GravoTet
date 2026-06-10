"""GravoTet supplementary demo: sphere at ~1.5M vertices.

Identical to the cube and torus drivers in everything except the input
mesh: solves Poisson and biharmonic on a TetGen-meshed unit ball
(``data/sphere_1.5M.ply``, ~1.45 M vertices).  All shared logic lives in
``gravotet_demo.run_demo``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "python"))

from gravotet_demo import DATA_DIR, make_mesh_from_ply, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Sphere 1.5M",
        output_subdir="sphere_1.5M",
        build_mesh=lambda gravotet: make_mesh_from_ply(
            gravotet, DATA_DIR / "sphere_1.5M.ply",
        ),
        description=__doc__,
    ))
