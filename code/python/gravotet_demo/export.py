"""Hierarchy export utilities for the supplementary demo.

Saves the multigrid hierarchy produced by the C++ solver to disk:
  - output/meshes/level_{i}.npz  -- vertex positions and connectivity per level
  - output/prolongations.npz     -- all prolongation operators in CSR format

Convention: P[i] maps level i+1 (coarse) to level i (fine), shape (N_fine, N_coarse).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_hierarchy_meshes(solver: Any, output_dir: Path) -> list[Path]:
    """Save each hierarchy level as a compressed NumPy archive.

    Each archive stores:
      - vertices:    float64 (N, 3) vertex positions
      - tetrahedra:  int32   (M, 4) tetrahedral connectivity

    Returns the list of written paths (one per level, finest first).
    """
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for level in range(len(solver.all_vertices)):
        V = np.asarray(solver.all_vertices[level], dtype=np.float64)
        T = np.asarray(solver.all_tetrahedra[level], dtype=np.int32)
        path = mesh_dir / f"level_{level}.npz"
        np.savez_compressed(path, vertices=V, tetrahedra=T)
        paths.append(path)
    return paths


def save_prolongation_matrices(solver: Any, output_dir: Path) -> Path:
    """Save all prolongation matrices as a single compressed NumPy archive.

    P[i] maps level i+1 (coarse) to level i (fine), shape (N_fine, N_coarse).
    Each matrix is stored in CSR format under keys:
      P_{i}_data, P_{i}_indices, P_{i}_indptr, P_{i}_shape

    Returns the path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "prolongations.npz"

    arrays: dict = {}
    for i, P in enumerate(solver.all_P):
        P_csr = P.tocsr() if hasattr(P, "tocsr") else P
        arrays[f"P_{i}_data"] = P_csr.data.astype(np.float64)
        arrays[f"P_{i}_indices"] = P_csr.indices.astype(np.int32)
        arrays[f"P_{i}_indptr"] = P_csr.indptr.astype(np.int32)
        arrays[f"P_{i}_shape"] = np.array(P_csr.shape, dtype=np.int32)

    np.savez_compressed(str(path), **arrays)
    return path
