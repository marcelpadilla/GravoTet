"""Bundled large-mesh assembly for the GravoTet supplementary demos.

The spot, sphere, and torus tetrahedral meshes exceed GitHub's 100 MB
per-file limit, so each is committed to the repository as a sequence of
sub-100 MB parts under ``data_chunks/``.  On first use the parts are
concatenated back into the original ``.ply`` under ``data/`` (which is
git-ignored and acts as a local cache).

Reassembly is automatic: when a demo needs one of these meshes and the
assembled file is missing, the parts are joined on the fly.  The same step
is also exposed explicitly through ``scripts/fetch_data.py``.  The cube demo
generates its mesh in C++ and needs none of this.

This module is the single source of truth for the mesh table, the chunk
size, and the chunk/data directory locations; both the runtime demos and the
maintainer split tool (``scripts/split_meshes.py``) import from here.
"""

from __future__ import annotations

from pathlib import Path

# code/python/gravotet_demo/data_assembly.py -> repository root is parents[3].
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
CHUNK_DIR = ROOT / "data_chunks"

# Maximum bytes per committed part.  GitHub rejects files over 100 MiB
# (104,857,600 bytes); 90 MB leaves a comfortable margin.
CHUNK_SIZE = 90_000_000

# name -> (filename in data/, exact assembled size in bytes for integrity check)
MESHES = {
    "spot":   ("spot.ply",        148_921_976),
    "sphere": ("sphere_1.5M.ply", 168_811_020),
    "torus":  ("torus_2M.ply",    230_821_897),
}

# Reverse lookup: assembled filename -> mesh name.
_BY_FILENAME = {filename: name for name, (filename, _) in MESHES.items()}

# Read/write block size used when joining parts (keeps memory flat).
_IO_BLOCK = 1 << 20  # 1 MB


def part_paths(filename: str) -> list[Path]:
    """Return the committed part files for ``filename`` in concatenation order.

    Parts are named ``<filename>.partNNN`` with zero-padded indices, so plain
    lexicographic sorting yields the correct order.
    """
    return sorted(CHUNK_DIR.glob(f"{filename}.part*"))


def assemble_mesh(name: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Concatenate the committed parts of one mesh into ``data/<filename>``.

    Returns the path to the assembled mesh.  If the assembled file already
    exists with the expected size it is reused.  Raises ``FileNotFoundError``
    when no parts are present, and ``RuntimeError`` on a size mismatch.
    """
    if name not in MESHES:
        raise KeyError(f"Unknown mesh '{name}'. Choose from: {', '.join(MESHES)}.")
    filename, expected_size = MESHES[name]
    dest = DATA_DIR / filename

    if dest.exists() and not force and dest.stat().st_size == expected_size:
        if verbose:
            print(f"  [skip] {filename} already assembled ({expected_size / 1e6:.0f} MB)")
        return dest

    parts = part_paths(filename)
    if not parts:
        raise FileNotFoundError(
            f"No parts found for {filename} under {CHUNK_DIR}. "
            f"Expected files like {filename}.part000. "
            f"Run 'python scripts/split_meshes.py' if you have the source meshes."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if verbose:
        print(f"  [join] {filename}  <-  {len(parts)} part(s)")

    written = 0
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                while True:
                    block = src.read(_IO_BLOCK)
                    if not block:
                        break
                    out.write(block)
                    written += len(block)

    if written != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Assembled {filename} has size {written} bytes "
            f"(expected {expected_size}); parts may be incomplete or corrupt."
        )
    tmp.replace(dest)
    if verbose:
        print(f"  [ok  ] {filename}  ({written / 1e6:.0f} MB)")
    return dest


def assemble_all(*, force: bool = False, verbose: bool = True) -> None:
    """Assemble every known mesh that has committed parts on disk."""
    for name in MESHES:
        assemble_mesh(name, force=force, verbose=verbose)


def ensure_mesh_for_path(ply_path: Path, *, verbose: bool = True) -> bool:
    """Ensure a bundled mesh exists at ``ply_path``, assembling it if needed.

    If the file is already present, returns True immediately.  Otherwise, if
    the path's filename matches a known bundled mesh and its parts are present,
    the mesh is assembled.  Returns True when the file exists afterwards, False
    if the path is not a known bundled mesh (or its parts are missing).
    """
    ply_path = Path(ply_path)
    if ply_path.exists():
        return True
    name = _BY_FILENAME.get(ply_path.name)
    if name is None:
        return False
    try:
        assemble_mesh(name, verbose=verbose)
    except FileNotFoundError:
        return False
    return ply_path.exists()
