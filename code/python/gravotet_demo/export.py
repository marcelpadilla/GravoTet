"""Hierarchy export utilities for the supplementary demo.

Saves the multigrid hierarchy produced by the C++ solver to disk:
  - output/meshes/level_{i}.ply  -- ASCII PLY: all vertices (depth attribute:
                                    0=surface, 1=interior) + the actual simplicial
                                    complex triangles (tet boundary faces + free
                                    triangles from allTris).  This matches the full
                                    algorithm output and loads correctly in Blender.
  - output/meshes/level_{i}.png  -- pyvista offscreen render using nearest-coarse-
                                    vertex mapping of the fine surface triangles.
                                    Proper Z-buffer rendering; no painter's-algorithm
                                    artifacts.  Hole-free at every coarsening level.
  - output/prolongations.npz     -- all prolongation operators in CSR format

Render strategy (PNG only):
  Matplotlib has no Z-buffer, so rendering the full non-manifold simplicial complex
  (which contains interior triangles) causes interior faces to bleed through the
  exterior.  Instead, the PNG renders use nearest-coarse-vertex mapping:
    1. Take the fine-level boundary triangles (clean manifold exterior).
    2. For each coarse level, snap every fine surface vertex to its nearest coarse
       vertex via KD-tree query.
    3. Remap fine boundary triangles to coarse vertex indices; remove degenerate
       triangles (two or more vertices snapping to the same coarse vertex);
       deduplicate.
  This produces a complete, hole-free exterior surface representation at every level.
  The coarsening density is clearly visible: dense at level 1, sparse at level 4.

  pyvista renders the scene offscreen with a proper Z-buffer so depth ordering
  is correct regardless of face count or camera angle.  All rendering is fully
  opaque.

PLY strategy:
  PLY files contain the actual algorithm output: tet boundary faces UNION free
  triangles (allTris).  These load and render correctly in Blender/MeshLab.

Convention: P[i] maps level i+1 (coarse) to level i (fine), shape (N_fine, N_coarse).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _find_boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """Return surface triangles: faces that appear exactly once across all tets.

    Parameters
    ----------
    tets : (M, 4) int32 array of tetrahedral vertex indices

    Returns
    -------
    tris : (K, 3) int32 array of boundary triangle vertex indices,
           with outward winding order preserved.
    """
    if len(tets) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    t = tets.astype(np.int32)
    # Outward winding for each of the 4 faces of a positively-oriented tet.
    faces = np.vstack([
        t[:, [1, 2, 3]],
        t[:, [0, 3, 2]],
        t[:, [0, 1, 3]],
        t[:, [0, 2, 1]],
    ])

    canonical = np.sort(faces, axis=1)
    dt = np.dtype([("a", np.int32), ("b", np.int32), ("c", np.int32)])
    structured = np.ascontiguousarray(canonical).view(dt).reshape(-1)
    _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1].astype(np.int32)


def _simplicial_complex_triangles(
    tets: np.ndarray, free_tris: list
) -> np.ndarray:
    """Collect all surface triangles of the simplicial complex at a coarse level.

    Used for PLY export.  Returns tet boundary faces UNION free triangles (allTris).

    Parameters
    ----------
    tets      : (M, 4) int32 tet indices at this level
    free_tris : list of [v0, v1, v2] lists from solver.all_tris[level-1]

    Returns
    -------
    (K, 3) int32 triangle array
    """
    boundary = _find_boundary_triangles(tets)
    if free_tris:
        free_arr = np.array(free_tris, dtype=np.int32).reshape(-1, 3)
    else:
        free_arr = np.zeros((0, 3), dtype=np.int32)

    if len(boundary) == 0:
        return free_arr
    if len(free_arr) == 0:
        return boundary

    combined = np.vstack([boundary, free_arr])
    canonical = np.sort(combined, axis=1)
    _, idx = np.unique(canonical, axis=0, return_index=True)
    return combined[idx].astype(np.int32)


def _surface_by_nearest_coarse(
    fine_verts: np.ndarray,
    fine_boundary_tris: np.ndarray,
    coarse_verts: np.ndarray,
    num_coarse_exterior: int = 0,
) -> np.ndarray:
    """Map fine surface triangles to nearest coarse exterior vertices for PNG rendering.

    Only exterior coarse vertices (the first `num_coarse_exterior` entries in
    coarse_verts) are considered as candidates.  This prevents fine surface
    points from snapping to interior coarse vertices that happen to be closer,
    which would create inward dents in the rendered surface.

    Parameters
    ----------
    fine_verts          : (N, 3) fine vertex positions
    fine_boundary_tris  : (F, 3) fine surface triangle vertex indices
    coarse_verts        : (C, 3) coarse vertex positions
    num_coarse_exterior : number of exterior vertices (first K in coarse_verts)
                          If 0 or >= C, all coarse verts are used.

    Returns
    -------
    (K, 3) int32 array of coarse triangle vertex indices covering the exterior
    """
    from scipy.spatial import KDTree

    if len(fine_boundary_tris) == 0 or len(coarse_verts) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Restrict to exterior coarse vertices only.
    if 0 < num_coarse_exterior < len(coarse_verts):
        candidate_verts = coarse_verts[:num_coarse_exterior]
    else:
        candidate_verts = coarse_verts

    surf_idx = np.unique(fine_boundary_tris)
    surf_pos = fine_verts[surf_idx]

    tree = KDTree(candidate_verts)
    _, nn = tree.query(surf_pos)

    # Build fine-vertex-index → coarse-vertex-index map.
    fine_to_coarse = np.full(len(fine_verts), -1, dtype=np.int32)
    fine_to_coarse[surf_idx] = nn

    # Remap fine triangles.
    coarse_tris = fine_to_coarse[fine_boundary_tris]
    valid = np.all(coarse_tris >= 0, axis=1)
    coarse_tris = coarse_tris[valid]
    if len(coarse_tris) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Remove degenerate triangles.
    v0, v1, v2 = coarse_tris[:, 0], coarse_tris[:, 1], coarse_tris[:, 2]
    valid = (v0 != v1) & (v1 != v2) & (v0 != v2)
    coarse_tris = coarse_tris[valid]

    if len(coarse_tris) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Deduplicate by canonical (sorted) vertex set.
    canonical = np.sort(coarse_tris, axis=1)
    _, idx = np.unique(canonical, axis=0, return_index=True)
    return coarse_tris[idx].astype(np.int32)


def _surface_depth(num_verts: int, boundary_tris: np.ndarray) -> np.ndarray:
    """Return depth attribute: 0 for surface (boundary) vertices, 1 for interior."""
    depth = np.ones(num_verts, dtype=np.int32)
    if len(boundary_tris) > 0:
        depth[np.unique(boundary_tris)] = 0
    return depth


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_ply(filepath: Path, verts: np.ndarray, tris: np.ndarray,
               depth: np.ndarray) -> None:
    """Write an ASCII PLY file with x,y,z,depth vertex attributes and triangles."""
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

    vert_buf = io.StringIO()
    np.savetxt(vert_buf, np.column_stack([verts, depth.astype(np.float64)]),
               fmt=["%.6g", "%.6g", "%.6g", "%d"])
    vertex_block = vert_buf.getvalue()

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


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _apply_render_rotation(verts: np.ndarray) -> np.ndarray:
    """Rotate vertices 90 degrees around the X axis for rendering.

    Many mesh datasets (including Keenan Crane's collection) use a Y-up
    coordinate convention, while pyvista's default isometric camera treats Z
    as "up".  Without correction the models appear to be lying on their backs
    when rendered from the iso viewpoint.

    Rx(+90 deg): (x, y, z) -> (x, -z, y)
    This maps the Y-up axis to Z-up so the model stands upright in pyvista.
    """
    R = np.array([[1, 0,  0],
                  [0, 0, -1],
                  [0, 1,  0]], dtype=np.float64)
    return verts @ R.T


def _render_surface(verts: np.ndarray, tris: np.ndarray,
                    title: str, out_path: Path,
                    draw_edges: bool = False) -> None:
    """Render a surface mesh to a PNG using pyvista (proper Z-buffer rendering).

    pyvista renders offscreen via VTK so depth ordering is always correct —
    no painter's-algorithm artifacts regardless of face count or camera angle.

    Parameters
    ----------
    verts      : (N, 3) float64 vertex positions
    tris       : (K, 3) int32 triangle indices
    title      : text label drawn at the top of the image
    out_path   : destination PNG path
    draw_edges : if True, draw a wireframe edge overlay on the shaded surface
    """
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=[750, 750])
    pl.background_color = "white"

    if len(tris) > 0 and len(verts) > 0:
        rotated = _apply_render_rotation(verts)
        # pyvista face format: [3, i0, i1, i2, 3, i0, i1, i2, ...]
        faces_pv = np.hstack(
            [np.full((len(tris), 1), 3, dtype=np.int_), tris]
        ).ravel()
        mesh = pv.PolyData(rotated.astype(np.float32), faces_pv)

        lw = max(0.5, min(3.0, 300.0 / max(len(tris), 1)))
        pl.add_mesh(
            mesh,
            color="#4582B5",       # steel blue
            show_edges=draw_edges,
            edge_color="#141F3F",  # dark navy
            line_width=lw,
            lighting=True,
            smooth_shading=False,
        )

    # Isometric-ish view: elevation 25 deg, azimuth 45 deg.
    pl.camera_position = "iso"
    pl.camera.azimuth = -15
    pl.camera.elevation = 10
    pl.camera.zoom(0.9)

    pl.add_text(title, position="upper_edge", font_size=10, color="black")
    pl.screenshot(str(out_path), transparent_background=False)
    pl.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _render_level_strip(solver: Any, output_dir: Path,
                        fine_boundary_tris: np.ndarray,
                        fine_verts: np.ndarray,
                        exterior_counts: list[int]) -> Path:
    """Render all hierarchy levels into a single horizontal strip PNG.

    Each level shows the exterior surface with edge overlay and vertex count
    below. All levels share the same camera angle for easy visual comparison.

    Returns the path to the generated strip PNG.
    """
    import pyvista as pv
    from PIL import Image

    n_levels = len(solver.all_vertices)

    # Collect renders for each level
    from io import BytesIO
    level_images: list[Image.Image] = []

    for i in range(n_levels):
        verts = np.asarray(solver.all_vertices[i], dtype=np.float64)

        if i == 0:
            render_tris = fine_boundary_tris
        else:
            render_tris = _surface_by_nearest_coarse(
                fine_verts, fine_boundary_tris, verts,
                num_coarse_exterior=exterior_counts[i],
            )

        # Label: level name + vertex count
        if i == 0:
            level_name = f"Level {i} (fine) — {len(verts):,} verts"
        elif i == n_levels - 1:
            level_name = f"Level {i} (coarsest) — {len(verts):,} verts"
        else:
            level_name = f"Level {i} — {len(verts):,} verts"

        # Render each level as a 750×750 image
        pl = pv.Plotter(off_screen=True, window_size=[750, 750])
        pl.background_color = "white"

        if len(render_tris) > 0 and len(verts) > 0:
            rotated = _apply_render_rotation(verts)
            faces_pv = np.hstack(
                [np.full((len(render_tris), 1), 3, dtype=np.int_),
                 render_tris]
            ).ravel()
            mesh = pv.PolyData(rotated.astype(np.float32), faces_pv)

            lw = max(0.6, min(3.0, 400.0 / max(len(render_tris), 1)))
            pl.add_mesh(
                mesh,
                color="#4582B5",
                show_edges=True,
                edge_color="#141F3F",
                line_width=lw,
                lighting=True,
                smooth_shading=False,
            )

        pl.camera_position = "iso"
        pl.camera.azimuth = -15
        pl.camera.elevation = 10
        pl.camera.zoom(0.9)

        pl.add_text(level_name, position="upper_edge",
                    font_size=10, color="black")
        # Use in-memory screenshot to avoid Windows file-lock issues
        img_arr = pl.screenshot(return_img=True)
        pl.close()

        level_images.append(Image.fromarray(img_arr))

    # --- Concatenate horizontally ---
    widths = [img.width for img in level_images]
    max_h = max(img.height for img in level_images)

    total_w = sum(widths)
    strip = Image.new("RGB", (total_w, max_h), (255, 255, 255))

    x_offset = 0
    for img in level_images:
        y_offset = (max_h - img.height) // 2
        strip.paste(img, (x_offset, y_offset))
        x_offset += img.width

    strip_path = output_dir / "hierarchy_strip.png"
    strip.save(str(strip_path), "PNG")
    return strip_path


def save_hierarchy_meshes(solver: Any, output_dir: Path) -> list[Path]:
    """Export each hierarchy level as a PLY mesh file and a PNG render.

    PLY (actual algorithm output):
      - Level 0: exterior surface of the fine manifold tet mesh.
      - Level i > 0: tet boundary faces UNION free triangles (solver.all_tris[i-1]).
      Vertices carry a depth attribute (0=surface, 1=interior).
      These files load correctly in Blender and MeshLab.

    PNG (visualization — nearest-coarse-vertex mapped surface):
      - Level 0: fine boundary surface, smooth Phong shading with edge overlay.
      - Level i > 0: fine surface triangles remapped to nearest coarse vertices
        via KD-tree (see _surface_by_nearest_coarse).  All faces rendered with
        both-side shading (abs(dot)) and fully opaque — no holes, no interior
        bleed-through.

    In addition to individual per-level PNGs, a ``hierarchy_strip.png`` is
    written showing all levels side-by-side with edge overlay on every level.

    Parameters
    ----------
    solver     : GravoTet C++ solver exposing all_vertices, all_tetrahedra,
                 all_tris, all_P, nr_points.
    output_dir : root output directory; a ``meshes/`` subdirectory is created.

    Returns
    -------
    list of Path objects for the written PLY files, one per level (finest first).
    """
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    n_levels = len(solver.all_vertices)

    # Number of exterior vertices per level.
    # all_exterior_indices[j] = exterior vertex indices at level j.
    #   j=0: fine-level exterior vertices (from preprocessExteriorMesh)
    #   j>0: coarse exterior vertices at level j (from hierarchy construction)
    # exterior_counts[i] is used in _surface_by_nearest_coarse to restrict the
    # KD-tree to the exterior coarse vertices (which FDS places first).
    ext_idx_attr = getattr(solver, "all_exterior_indices", [])
    exterior_counts: list[int] = [0] * n_levels
    for j, idx_list in enumerate(ext_idx_attr):
        coarse_level = j  # index matches level directly
        if coarse_level < n_levels:
            exterior_counts[coarse_level] = len(idx_list)

    fine_verts = np.asarray(solver.all_vertices[0], dtype=np.float64)
    fine_tets = np.asarray(solver.all_tetrahedra[0], dtype=np.int32)
    fine_boundary_tris = _find_boundary_triangles(fine_tets)

    coarse_free_tris = list(solver.all_tris) if hasattr(solver, "all_tris") else []

    ply_paths: list[Path] = []

    for i in range(n_levels):
        verts = np.asarray(solver.all_vertices[i], dtype=np.float64)
        tets = np.asarray(solver.all_tetrahedra[i], dtype=np.int32)

        # --- PLY: actual algorithm output ---
        if i == 0:
            ply_tris = fine_boundary_tris
        else:
            free = coarse_free_tris[i - 1] if i - 1 < len(coarse_free_tris) else []
            ply_tris = _simplicial_complex_triangles(tets, free)

        depth = _surface_depth(len(verts), ply_tris)
        ply_path = mesh_dir / f"level_{i}.ply"
        _write_ply(ply_path, verts, ply_tris, depth)
        ply_paths.append(ply_path)

        # --- PNG: clean exterior surface via NN mapping ---
        if i == 0:
            render_tris = fine_boundary_tris
        else:
            render_tris = _surface_by_nearest_coarse(
                fine_verts, fine_boundary_tris, verts,
                num_coarse_exterior=exterior_counts[i],
            )

        if i == 0:
            level_label = "Level 0 (fine)"
        elif i == n_levels - 1:
            level_label = f"Level {i} (coarsest)"
        else:
            level_label = f"Level {i}"
        title = f"{level_label}\n{len(verts):,} vertices"
        png_path = mesh_dir / f"level_{i}.png"

        _render_surface(verts, render_tris, title, png_path, draw_edges=True)

    # --- Strip render: all levels side-by-side with edges ---
    _render_level_strip(solver, output_dir, fine_boundary_tris, fine_verts, exterior_counts)

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

    np.savez_compressed(path, **arrays)
    return path
