"""Maintainer tool: split data/*.ply into committed sub-100 MB parts.

The large spot/sphere/torus tetrahedral meshes exceed GitHub's 100 MB
per-file limit, so they cannot be committed directly.  This script slices each
mesh in ``data/`` into ``<filename>.partNNN`` files under ``data_chunks/`` and
writes a ``manifest.json`` recording sizes and SHA-256 digests.  The parts are
committed to git; the demos (and ``scripts/fetch_data.py``) concatenate them
back into ``data/`` on first use.

Run this after refreshing or replacing the source meshes in ``data/``:

    python scripts/split_meshes.py            # split every mesh
    python scripts/split_meshes.py spot       # split one mesh by name

Mesh names and expected sizes come from ``gravotet_demo.data_assembly``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make the demo package importable so the mesh table stays single-sourced.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "python"))

from gravotet_demo.data_assembly import (  # noqa: E402
    CHUNK_DIR,
    CHUNK_SIZE,
    DATA_DIR,
    MESHES,
    part_paths,
)

_IO_BLOCK = 1 << 20  # 1 MB


def _split_one(name: str) -> dict:
    filename, expected_size = MESHES[name]
    src = DATA_DIR / filename
    if not src.exists():
        raise SystemExit(f"Source mesh not found: {src}")

    actual_size = src.stat().st_size
    if actual_size != expected_size:
        print(
            f"  [warn] {filename} size is {actual_size} bytes but the table "
            f"expects {expected_size}. Update MESHES in data_assembly.py if the "
            f"mesh changed on purpose."
        )

    # Clear any stale parts for this mesh before re-splitting.
    for old in part_paths(filename):
        old.unlink()

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    part_index = 0
    part_names: list[str] = []

    with open(src, "rb") as f:
        while True:
            block = f.read(CHUNK_SIZE)
            if not block:
                break
            part_name = f"{filename}.part{part_index:03d}"
            (CHUNK_DIR / part_name).write_bytes(block)
            digest.update(block)
            written += len(block)
            part_names.append(part_name)
            print(f"  [part] {part_name}  ({len(block) / 1e6:.0f} MB)")
            part_index += 1

    print(f"  [done] {filename}: {part_index} part(s), {written / 1e6:.0f} MB total")
    return {
        "filename": filename,
        "size": written,
        "sha256": digest.hexdigest(),
        "parts": part_names,
    }


def _write_manifest() -> None:
    """Rebuild manifest.json from whatever meshes currently have parts on disk."""
    manifest: dict[str, dict] = {}
    for name, (filename, _) in MESHES.items():
        parts = part_paths(filename)
        if not parts:
            continue
        size = sum(p.stat().st_size for p in parts)
        manifest[name] = {
            "filename": filename,
            "size": size,
            "parts": [p.name for p in parts],
        }
    (CHUNK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes", nargs="*", default=[],
        help=f"Mesh name(s) to split from {{{', '.join(MESHES)}}} (default: all).",
    )
    args = parser.parse_args()

    unknown = [m for m in args.meshes if m not in MESHES]
    if unknown:
        raise SystemExit(
            f"Unknown mesh name(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(MESHES)}."
        )

    targets = args.meshes or list(MESHES)
    print(f"Splitting {len(targets)} mesh(es) into {CHUNK_DIR} "
          f"(<= {CHUNK_SIZE / 1e6:.0f} MB per part) ...")
    digests = {}
    for name in targets:
        digests[name] = _split_one(name)
    _write_manifest()
    print("Done.")
    for name, info in digests.items():
        print(f"  {name}: sha256 {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
