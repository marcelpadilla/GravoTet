"""Maintainer tool: convert TetGen ASCII PLYs into the supplement binary PLYs.

This is used once by the supplement maintainer to produce the binary meshes
that are uploaded to Zenodo; end users never run it (they fetch the finished
meshes with ``scripts/fetch_data.py``). It needs numpy and a directory of the
ASCII tet PLYs written by the main project's ``create_tet_meshes.py`` pipeline.

The source directory is supplied explicitly, so the script carries no path to
any particular machine:

    python scripts/import_tetgen_meshes.py --src-dir /path/to/meshes_tets_final

It can also be read from the ``GRAVOTET_TETGEN_SRC`` environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "python"))

from gravotet_demo.meshes import _save_tet_ply  # noqa: E402

DST_DIR = REPO / "data"

# Source filename in the TetGen pipeline -> destination filename under data/.
WORK = [
    ("spot.ply",   "spot.ply"),
    ("sphere.ply", "sphere_1.5M.ply"),
    ("torus.ply",  "torus_2M.ply"),
]


def _read_tetgen_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the ASCII PLY format written by create_tet_meshes.py.

    Layout: header, then ``num_vertices`` lines of ``x y z``, then
    ``num_tets`` lines of ``4 v0 v1 v2 v3``.
    """
    with open(path, "rb") as f:
        # Find end of header (ASCII bytes only).
        header_chunks: list[bytes] = []
        num_vertices = num_tets = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF in {path}")
            header_chunks.append(line)
            text = line.decode("ascii").strip()
            if text.startswith("element vertex"):
                num_vertices = int(text.split()[-1])
            elif text.startswith("element face"):
                num_tets = int(text.split()[-1])
            elif text == "end_header":
                break
        body_offset = f.tell()

    print(f"  header parsed: {num_vertices:,} verts, {num_tets:,} faces")
    # Vertex block: numpy can parse it fast with loadtxt-style readers via
    # np.fromstring / np.loadtxt.  np.loadtxt with max_rows is the simplest.
    with open(path, "r") as f:
        f.seek(body_offset)
        verts = np.loadtxt(f, dtype=np.float64, max_rows=num_vertices)
        tets_raw = np.loadtxt(f, dtype=np.int64, max_rows=num_tets)

    if verts.shape != (num_vertices, 3):
        raise ValueError(f"unexpected vertex array shape {verts.shape}")
    if tets_raw.shape[1] != 5 or (tets_raw[:, 0] != 4).any():
        raise ValueError("expected each face row to start with `4 ...` (tet)")
    return verts, tets_raw[:, 1:].astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir", type=Path,
        default=os.environ.get("GRAVOTET_TETGEN_SRC"),
        help="Directory holding the TetGen ASCII PLYs (spot.ply, sphere.ply, "
             "torus.ply). Defaults to $GRAVOTET_TETGEN_SRC.",
    )
    args = parser.parse_args()

    if args.src_dir is None:
        raise SystemExit(
            "No source directory given. Pass --src-dir /path/to/meshes_tets_final "
            "or set GRAVOTET_TETGEN_SRC."
        )
    src_dir = Path(args.src_dir)
    if not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in WORK:
        src = src_dir / src_name
        dst = DST_DIR / dst_name
        if not src.exists():
            print(f"Skipping {src_name} (not found in {src_dir})")
            continue
        print(f"Loading {src} ...")
        t0 = time.perf_counter()
        pts, tets = _read_tetgen_ascii_ply(src)
        print(f"  loaded in {time.perf_counter() - t0:.1f} s: "
              f"{len(pts):,} verts, {len(tets):,} tets")

        print(f"  writing {dst} (binary PLY) ...")
        t0 = time.perf_counter()
        _save_tet_ply(dst, pts, tets)
        print(f"  wrote in {time.perf_counter() - t0:.1f} s "
              f"({dst.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
