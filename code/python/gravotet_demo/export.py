"""Hierarchy export utilities for the supplementary demo.

Saves the multigrid hierarchy produced by the C++ solver to disk:
  - output/meshes/level_{i}.ply  -- ASCII PLY: vertices (x,y,z,depth) + surface triangles
  - output/meshes/level_{i}.png  -- matplotlib Phong-shaded surface render per level
  - output/prolongations.npz     -- all prolongation operators in CSR format

The PLY depth attribute is 0 for surface (boundary) vertices and 1 for interior
vertices. Surface triangles are faces of the tet mesh that appear exactly once
(i.e., are not shared between two tetrahedra).

Convention: P[i] maps level i+1 (coarse) to level i (fine), shape (N_fine, N_coarse).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """Return surface triangles: faces that appear exactly once across all tets.

    Parameters
    ----------
    tets : (M, 4) int32 array of tetrahedral vertex indices

    Returns
    -------
    tris : (K, 3) int32 array of boundary triangle vertex indices,
           with original winding order preserved for correct outward normals.
    """
    if len(tets) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    t = tets.astype(np.int32)
    # Each tet has 4 faces; each face is the 3 vertices opposite one vertex.
    # Winding chosen so normals point outward for a positively-oriented tet.
    faces = np.vstack([
        t[:, [1, 2, 3]],
        t[:, [0, 3, 2]],
        t[:, [0, 1, 3]],
        t[:, [0, 2, 1]],
    ])  # shape (4M, 3)

    # Canonical form: sort vertex indices per face so shared faces match.
    canonical = np.sort(faces, axis=1)
    dt = np.dtype([("a", np.int32), ("b", np.int32), ("c", np.int32)])
    structured = np.ascontiguousarray(canonical).view(dt).reshape(-1)
    _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1].astype(np.int32)


def _surface_depth(num_verts: int, boundary_tris: np.ndarray) -> np.ndarray:
    """Return depth attribute: 0 for surface (boundary) vertices, 1 for interior.

    Parameters
    ----------
    num_verts : total number of vertices in this level
    boundary_tris : (K, 3) int array of boundary triangle indices

    Returns
    -------
    depth : (num_verts,) int32 array
    """
    depth = np.ones(num_verts, dtype=np.int32)
    if len(boundary_tris) > 0:
        depth[np.unique(boundary_tris)] = 0
    return depth


def _write_ply(filepath: Path, verts: np.ndarray, tris: np.ndarray,
               depth: np.ndarray) -> None:
    """Write an ASCII PLY file with x,y,z,depth vertex attributes and surface triangles.

    Parameters
    ----------
    filepath : destination path (must have a .ply suffix)
    verts    : (N, 3) float64 vertex positions
    tris     : (K, 3) int32 surface triangle indices
    depth    : (N,) int32 per-vertex depth attribute (0=surface, 1=interior)
    """
    num_verts = len(verts)
    num_tris = len(tris)

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_verts}",
        "property float x",
        "property float y",
        "property float z",
        "property int depth",
    ]
    if num_tris > 0:
        header_lines += [
            f"element face {num_tris}",
            "property list uchar int vertex_indices",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    # Build vertex block via numpy for speed.
    vert_buf = io.StringIO()
    np.savetxt(vert_buf, np.column_stack([verts, depth.astype(np.float64)]),
               fmt=["%.6g", "%.6g", "%.6g", "%d"])
    vertex_block = vert_buf.getvalue()

    # Build face block.
    face_block = ""
    if num_tris > 0:
        face_buf = io.StringIO()
        np.savetxt(face_buf,
                   np.column_stack([np.full(num_tris, 3, dtype=np.int32), tris]),
                   fmt="%d")
        face_block = face_buf.getvalue()

    with open(filepath, "w", newline="\n") as f:
        f.write(header)
        f.write(vertex_block)
        if face_block:
            f.write(face_block)


def _setup_ax(ax, verts: np.ndarray, elev: float, azim: float) -> None:
    """Apply a consistent clean style and view to a 3-D matplotlib axis."""
    if len(verts) > 0:
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        c = (mn + mx) / 2.0
        r = max((mx - mn).max() * 0.55, 1e-3)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.view_init(elev=elev, azim=azim)


def _render_surface(verts: np.ndarray, tris: np.ndarray,
                    title: str, out_path: Path) -> None:
    """Render the boundary surface of the fine mesh as a Phong-shaded PNG.

    Uses matplotlib Poly3DCollection with back-face culling and ambient+diffuse
    shading. Intended for the finest hierarchy level (level 0) where boundary
    triangles form a meaningful, manifold-like surface.

    Parameters
    ----------
    verts    : (N, 3) float64 vertex positions
    tris     : (K, 3) int32 boundary triangle indices
    title    : figure title string
    out_path : destination PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ELEV, AZIM = 25.0, 45.0
    BASE_COLOR = np.array([0.27, 0.51, 0.71])  # steel blue

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if len(tris) > 0 and len(verts) > 0:
        ae, aa = np.radians(ELEV), np.radians(AZIM)
        light = np.array([np.cos(ae) * np.cos(aa),
                          np.cos(ae) * np.sin(aa),
                          np.sin(ae)])

        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        normals /= norms

        dot = normals @ light
        front = dot > 0.0

        if np.any(front):
            shading = np.clip(0.35 + 0.65 * dot[front], 0.2, 1.0)
            face_rgb = shading[:, None] * BASE_COLOR
            rgba = np.column_stack([face_rgb, np.full(len(shading), 0.95)])

            tri_verts = np.stack([v0[front], v1[front], v2[front]], axis=1)
            poly = Poly3DCollection(tri_verts)
            poly.set_facecolor(rgba.tolist())
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)

    _setup_ax(ax, verts, ELEV, AZIM)
    ax.set_title(title, fontsize=11, pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_scatter(verts: np.ndarray, exterior_mask: np.ndarray,
                    title: str, out_path: Path) -> None:
    """Render coarse hierarchy vertices as a scatter plot.

    Exterior vertices (on the fine mesh boundary) are drawn in a darker blue;
    interior vertices in a lighter, semi-transparent blue. This makes the
    boundary-aware sampling visible for coarser levels.

    Parameters
    ----------
    verts         : (N, 3) float64 vertex positions
    exterior_mask : (N,) bool — True for vertices on the original fine boundary
    title         : figure title string
    out_path      : destination PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ELEV, AZIM = 25.0, 45.0

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if len(verts) > 0:
        pt_size = max(4.0, min(40.0, 12000.0 / len(verts)))

        interior = ~exterior_mask
        if np.any(interior):
            iv = verts[interior]
            ax.scatter(iv[:, 0], iv[:, 1], iv[:, 2],
                       s=pt_size, c=[[0.55, 0.72, 0.88]], alpha=0.55,
                       edgecolors="none", depthshade=True, label="interior")
        if np.any(exterior_mask):
            ev = verts[exterior_mask]
            ax.scatter(ev[:, 0], ev[:, 1], ev[:, 2],
                       s=pt_size * 1.4, c=[[0.18, 0.42, 0.65]], alpha=0.90,
                       edgecolors="none", depthshade=True, label="boundary")

    _setup_ax(ax, verts, ELEV, AZIM)
    ax.set_title(title, fontsize=11, pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_hierarchy_meshes(solver: Any, output_dir: Path) -> list[Path]:
    """Export each hierarchy level as a PLY mesh file and a PNG render.

    For each level i the following files are written to output_dir/meshes/:
      - level_{i}.ply   ASCII PLY: all vertices with depth attribute (0=surface,
                        1=interior) and the extracted surface triangles.
      - level_{i}.png   Render: Phong-shaded surface for level 0; scatter plot
                        of coarse nodes (exterior in dark blue, interior in light
                        blue) for levels 1 and above.

    The exterior/interior classification at coarse levels is derived from the
    fine mesh boundary: a coarse vertex is considered exterior if its position
    matches a fine boundary vertex (nearest-position lookup with 6-digit rounding).

    Parameters
    ----------
    solver     : GravoTet C++ solver object exposing all_vertices / all_tetrahedra.
    output_dir : root output directory; a ``meshes/`` subdirectory is created.

    Returns
    -------
    list of Path objects for the written PLY files, one per level (finest first).
    """
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    n_levels = len(solver.all_vertices)
    ply_paths: list[Path] = []

    # Precompute fine-level surface once for use across all coarse levels.
    fine_verts = np.asarray(solver.all_vertices[0], dtype=np.float64)
    fine_tets = np.asarray(solver.all_tetrahedra[0], dtype=np.int32)
    fine_boundary_tris = _find_boundary_triangles(fine_tets)

    # Build a rounded-position set for O(1) exterior lookup at coarse levels.
    _ROUND = 6
    if len(fine_boundary_tris) > 0:
        fine_surf_idx = np.unique(fine_boundary_tris)
        fine_surface_pos = {
            tuple(np.round(fine_verts[vi], _ROUND)) for vi in fine_surf_idx
        }
    else:
        fine_surface_pos = set()

    for i in range(n_levels):
        verts = np.asarray(solver.all_vertices[i], dtype=np.float64)
        tets = np.asarray(solver.all_tetrahedra[i], dtype=np.int32)

        boundary_tris = _find_boundary_triangles(tets)
        depth = _surface_depth(len(verts), boundary_tris)

        # --- PLY ---
        ply_path = mesh_dir / f"level_{i}.ply"
        _write_ply(ply_path, verts, boundary_tris, depth)
        ply_paths.append(ply_path)

        # --- PNG render ---
        if i == 0:
            level_label = "Level 0 (fine)"
        elif i == n_levels - 1:
            level_label = f"Level {i} (coarse)"
        else:
            level_label = f"Level {i}"
        title = f"{level_label}\n{len(verts):,} vertices"
        png_path = mesh_dir / f"level_{i}.png"

        if i == 0:
            # Fine level: Phong-shaded surface mesh render.
            _render_surface(verts, boundary_tris, title, png_path)
        else:
            # Coarse levels: scatter plot with exterior/interior coloring
            # derived from the fine mesh boundary.
            exterior_mask = np.array([
                tuple(np.round(v, _ROUND)) in fine_surface_pos for v in verts
            ], dtype=bool)
            _render_scatter(verts, exterior_mask, title, png_path)

    return ply_paths


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
