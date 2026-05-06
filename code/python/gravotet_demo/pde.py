"""Assemble the PDE systems used in the supplementary demo.

The notation follows the paper: `S` is the stiffness matrix, `M` is the lumped
barycentric mass matrix, Poisson uses `S + eps * M`, and biharmonic uses
`S M^{-1} S`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

POISSON_SHIFT = 1e-3
BIHARMONIC_REGULARIZER = 0.0
POISSON_TOL = 1e-8
BIHARMONIC_TOL = 1e-5


@dataclass
class ProblemData:
    """Assembled system and metadata needed by the demo and smoke test."""

    name: str
    A: csr_matrix
    b: np.ndarray
    x_true: np.ndarray
    m_diag: np.ndarray
    fixed_idx: np.ndarray
    tolerance: float
    resolution: int
    num_vertices: int
    num_tetrahedra: int
    hierarchy_build_ms: float = 0.0


def _boundary_vertex_indices(T: np.ndarray) -> np.ndarray:
    """Return the Dirichlet boundary vertices from boundary tetrahedron faces."""
    faces = np.vstack([
        T[:, [0, 1, 2]],
        T[:, [0, 1, 3]],
        T[:, [0, 2, 3]],
        T[:, [1, 2, 3]],
    ])
    sorted_faces = np.sort(np.asarray(faces, dtype=np.int32), axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]
    return np.unique(boundary_faces.ravel())


def _assemble_stiffness_matrix(V: np.ndarray, T: np.ndarray) -> csr_matrix:
    """Assemble the tetrahedral Laplace stiffness matrix with linear elements."""
    tet_positions = np.asarray(V[T], dtype=np.float64)
    num_tets = int(len(tet_positions))
    ones = np.ones((num_tets, 4, 1), dtype=np.float64)
    affine = np.concatenate((ones, tet_positions), axis=2)

    inv_affine = np.linalg.inv(affine)
    gradients = inv_affine[:, 1:, :]

    edges = tet_positions[:, 1:, :] - tet_positions[:, :1, :]
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    local_stiffness = volumes[:, None, None] * np.einsum(
        "nij,nik->njk",
        gradients,
        gradients,
    )

    rows = np.repeat(T, 4, axis=1).ravel()
    cols = np.tile(T, (1, 4)).ravel()
    data = local_stiffness.reshape(-1)
    num_vertices = int(len(V))
    return csr_matrix(coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices)))


def _assemble_lumped_mass_diag(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Assemble the lumped barycentric mass diagonal for a tet mesh."""
    tet_positions = np.asarray(V[T], dtype=np.float64)
    edges = tet_positions[:, 1:, :] - tet_positions[:, :1, :]
    volumes = np.abs(np.linalg.det(edges)) / 6.0

    mass_diag = np.zeros(int(len(V)), dtype=np.float64)
    np.add.at(mass_diag, T.ravel(), np.repeat(volumes / 4.0, 4))
    return mass_diag


def _manufactured_solution(V: np.ndarray, cycles: float = 2.0) -> np.ndarray:
    """Manufactured solution used to define the right-hand side and BCs."""
    bbox_min = V.min(axis=0)
    bbox_size = V.max(axis=0) - bbox_min
    scale = max(float(np.max(bbox_size)), 1e-12)
    centered = V - (bbox_min + 0.5 * bbox_size)
    k = 2.0 * np.pi * cycles / scale
    return (
        np.cos(k * centered[:, 0])
        + np.cos(k * centered[:, 1])
        + np.cos(k * centered[:, 2])
    ) / 3.0


def _apply_dirichlet(A: csr_matrix, b: np.ndarray, fixed_idx: np.ndarray, fixed_values: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    """Apply symmetric Dirichlet elimination while preserving SPD structure."""
    A = csr_matrix(A)
    b = np.asarray(b, dtype=np.float64).copy()

    if fixed_idx.size == 0:
        return A, b

    rows, _cols = A.get_shape()
    n = int(rows)
    mask = np.ones(n, dtype=np.float64)
    mask[fixed_idx] = 0.0

    diag_mask = csr_matrix((mask, (np.arange(n), np.arange(n))), shape=(n, n))
    fixed_ones = np.zeros(n, dtype=np.float64)
    fixed_ones[fixed_idx] = 1.0
    fixed_diag = csr_matrix((fixed_ones, (np.arange(n), np.arange(n))), shape=(n, n))

    x_fixed = np.zeros(n, dtype=np.float64)
    x_fixed[fixed_idx] = fixed_values
    b -= A @ x_fixed
    A = diag_mask @ A @ diag_mask
    A = A + fixed_diag
    b[fixed_idx] = fixed_values
    return A.tocsr(), b


def assemble_problem(solver, problem_name: str, resolution: int) -> ProblemData:
    """Assemble the level-0 Poisson or biharmonic demo system."""
    V = np.asarray(solver.all_vertices[0], dtype=np.float64)
    T = np.asarray(solver.all_tetrahedra[0], dtype=np.int32)

    S = _assemble_stiffness_matrix(V, T)
    M_diag = _assemble_lumped_mass_diag(V, T)
    M = diags(M_diag, format="csr")
    x_true = _manufactured_solution(V)

    if problem_name == "poisson":
        A = (S + POISSON_SHIFT * M).tocsr()
        tol = POISSON_TOL
    elif problem_name == "biharmonic":
        M_inv_diag = np.where(np.abs(M_diag) > 1e-30, 1.0 / M_diag, 0.0)
        M_inv = diags(M_inv_diag, format="csr")
        A = (S @ M_inv @ S + BIHARMONIC_REGULARIZER * M).tocsr()
        tol = BIHARMONIC_TOL
    else:
        raise ValueError(f"Unsupported problem '{problem_name}'")

    b = np.asarray(A @ x_true, dtype=np.float64)
    fixed_idx = _boundary_vertex_indices(T)
    fixed_values = x_true[fixed_idx]
    A_bc, b_bc = _apply_dirichlet(A, b, fixed_idx, fixed_values)

    return ProblemData(
        name=problem_name,
        A=A_bc,
        b=b_bc,
        x_true=x_true,
        m_diag=M_diag,
        fixed_idx=fixed_idx,
        tolerance=tol,
        resolution=resolution,
        num_vertices=int(len(V)),
        num_tetrahedra=int(len(T)),
    )
