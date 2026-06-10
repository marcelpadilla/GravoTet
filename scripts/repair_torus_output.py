"""Repair output/torus_2M/output.png without re-running the full demo.

Uses existing files:
  - output/torus_2M/data.json          (convergence / timing data)
  - output/torus_2M/meshes/level_*.png (already-rendered hierarchy levels)

Regenerates a correct output.png: plots on top, single hierarchy strip below.
The previous file had the strip duplicated due to an older version of the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "python"))

from gravotet_demo.plotting import save_combined_figure  # noqa: E402
from PIL import Image  # noqa: E402

OUTPUT_DIR = REPO / "output" / "torus_2M"


def main() -> int:
    summary_path = OUTPUT_DIR / "data.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    # 1. Regenerate residual / timing plots
    print("Generating combined.png ...")
    combined_path = save_combined_figure(summary, OUTPUT_DIR)
    print(f"  -> {combined_path}")

    # 2. Build hierarchy strip from the pre-existing per-level renders
    meshes_dir = OUTPUT_DIR / "meshes"
    level_pngs = sorted(
        meshes_dir.glob("level_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not level_pngs:
        print(f"ERROR: no level PNGs in {meshes_dir}")
        return 1

    print(f"Compositing {len(level_pngs)} level renders into hierarchy_strip.png ...")
    images = [Image.open(str(p)) for p in level_pngs]
    total_w = sum(img.width for img in images)
    max_h = max(img.height for img in images)

    strip = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for img in images:
        strip.paste(img, (x, (max_h - img.height) // 2))
        x += img.width
    for img in images:
        img.close()

    strip_path = OUTPUT_DIR / "hierarchy_strip.png"
    strip.save(str(strip_path), "PNG")
    strip.close()
    print(f"  -> {strip_path}")

    # 3. Concatenate: combined on top, hierarchy strip below (exactly once)
    print("Merging into output.png ...")
    combined = Image.open(str(combined_path))
    strip_img = Image.open(str(strip_path))

    target_w = combined.width
    new_h = int(strip_img.height * target_w / strip_img.width)
    strip_resized = strip_img.resize((target_w, new_h), Image.LANCZOS)

    result = Image.new("RGB", (target_w, combined.height + new_h), (255, 255, 255))
    result.paste(combined, (0, 0))
    result.paste(strip_resized, (0, combined.height))

    combined.close()
    strip_img.close()

    output_path = OUTPUT_DIR / "output.png"
    result.save(str(output_path), "PNG")

    # Remove intermediates
    combined_path.unlink(missing_ok=True)
    strip_path.unlink(missing_ok=True)

    print(f"  -> {output_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
