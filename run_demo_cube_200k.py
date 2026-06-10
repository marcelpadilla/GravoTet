"""GravoTet supplementary demo: cube at ~200k vertices.

Identical to the sphere and torus drivers in everything except the input
mesh: solves Poisson and biharmonic on a regular cube with 59**3 = 205,379
vertices.  All shared logic lives in ``gravotet_demo.run_demo``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "python"))

from gravotet_demo import make_cube_mesh, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Cube 200k",
        output_subdir="cube_200k",
        build_mesh=lambda gravotet: make_cube_mesh(gravotet, 59),
        description=__doc__,
    ))
