"""Tet-mesh inputs for the GravoTet supplementary demos.

Two entry points:

* :func:`make_cube_mesh` - regular cube, built directly by the C++
  extension via ``MultigridSolver.generate_cube_mesh(resolution)``.
* :func:`make_mesh_from_ply` - load any tet mesh (binary or ASCII PLY)
  from disk into a ``gravotet.TetrahedralMesh``.

The non-cube demos (spot, sphere, torus) load TetGen-quality tet meshes that
exceed GitHub's 100 MB per-file limit.  They are committed as sub-100 MB parts
under ``GravoTet/data_chunks/`` and concatenated back into ``GravoTet/data/``
on first use (see :mod:`gravotet_demo.data_assembly`).  The supplement itself
does not generate these meshes; it loads them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Binary PLY I/O
# ---------------------------------------------------------------------------

def _save_tet_ply(path: Path, vertices: np.ndarray, tetrahedra: np.ndarray) -> None:
    """Write a tetrahedral mesh as binary little-endian PLY.

    Tets are stored as faces with a 4-vertex list, which is the convention
    that loaders such as Blender and the main project's ``create_tet_meshes``
    pipeline use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    verts32 = np.ascontiguousarray(vertices, dtype="<f4")
    tets32 = np.ascontiguousarray(tetrahedra, dtype="<i4")

    face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", 4)])
    face_records = np.empty(len(tets32), dtype=face_dtype)
    face_records["count"] = 4
    face_records["idx"] = tets32

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts32)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(tets32)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts32.tobytes())
        f.write(face_records.tobytes())


def _load_tet_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a tetrahedral PLY (binary little-endian or ASCII)."""
    with open(path, "rb") as f:
        num_vertices = num_faces = 0
        is_binary = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"unexpected EOF reading header of {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("format"):
                is_binary = "binary" in line
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                num_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        if is_binary:
            verts32 = np.frombuffer(
                f.read(num_vertices * 12), dtype="<f4",
            ).reshape(num_vertices, 3)
            face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", 4)])
            face_bytes = num_faces * face_dtype.itemsize
            faces = np.frombuffer(f.read(face_bytes), dtype=face_dtype)
            return (verts32.astype(np.float64),
                    np.ascontiguousarray(faces["idx"], dtype=np.int32))

        rest = f.read().decode("ascii").splitlines()

    vertices = np.empty((num_vertices, 3), dtype=np.float64)
    for i in range(num_vertices):
        vertices[i] = list(map(float, rest[i].split()))

    tetrahedra = np.empty((num_faces, 4), dtype=np.int32)
    for i in range(num_faces):
        parts = rest[num_vertices + i].split()
        tetrahedra[i] = list(map(int, parts[1:5]))

    return vertices, tetrahedra


# ---------------------------------------------------------------------------
# Public mesh entry points
# ---------------------------------------------------------------------------

def make_cube_mesh(gravotet: Any, resolution: int) -> Any:
    """Return a ``TetrahedralMesh`` for a regular cube at the given resolution.

    The vertex count is ``resolution**3``.  Resolution 47 yields 103,823
    vertices; resolution 100 yields one million.
    """
    return gravotet.MultigridSolver.generate_cube_mesh(resolution)


_MISSING_PLY_HINT = (
    "\n\nThe spot, sphere, and torus meshes are committed as sub-100 MB parts "
    "under data_chunks/ and are normally assembled automatically.  If the parts "
    "are also missing, rebuild them from source meshes with:\n"
    "    python scripts/split_meshes.py\n"
    "The cube demo (run_demo_cube_200k.py) needs no input file.\n"
)


def make_mesh_from_ply(gravotet: Any, ply_path: Path,
                       verbose: bool = True) -> Any:
    """Load a tetrahedral PLY and return a ``gravotet.TetrahedralMesh``.

    If the file is a bundled mesh that has not been assembled yet, its parts
    under ``data_chunks/`` are concatenated automatically.  Raises
    :class:`FileNotFoundError` with an actionable hint if the file (and its
    parts) cannot be found.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        from .data_assembly import ensure_mesh_for_path
        ensure_mesh_for_path(ply_path, verbose=verbose)
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Required input mesh not found: {ply_path}{_MISSING_PLY_HINT}"
        )

    if verbose:
        print(f"  Loading tet mesh: {ply_path}")
    vertices, tetrahedra = _load_tet_ply(ply_path)
    if verbose:
        print(f"  {len(vertices):,} vertices, {len(tetrahedra):,} tetrahedra")

    mesh = gravotet.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    return mesh
