"""GravoTet supplementary demo: torus at ~2M vertices.

Identical to the cube and sphere drivers in everything except the input
mesh: solves Poisson and biharmonic on a TetGen-meshed solid torus
(``data/torus_2M.ply``, ~2.01 M vertices).  All shared logic lives in
``gravotet_demo.run_demo``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "python"))

from gravotet_demo import DATA_DIR, make_mesh_from_ply, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Torus 2M",
        output_subdir="torus_2M",
        build_mesh=lambda gravotet: make_mesh_from_ply(
            gravotet, DATA_DIR / "torus_2M.ply",
        ),
        description=__doc__,
    ))
