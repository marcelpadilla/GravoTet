"""Shared bootstrap and execution logic for the GravoTet supplementary demos.

The four driver scripts at the repository root (``run_demo_cube_200k.py``,
``run_demo_spot_700k.py``, ``run_demo_sphere_1.5M.py``, ``run_demo_torus_2M.py``)
all funnel through this module so that environment setup, extension build,
hierarchy construction, V-cycle solve, and on-disk export are written exactly
once.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import _bootstrap

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "code"
PKG = CODE / "python" / "gravotet_demo"
SRC = CODE / "cpp"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
# The Python-version and third-party-package checks live in the stdlib-only
# :mod:`_bootstrap` module, which the package __init__ runs before importing
# any submodule that needs those packages.  The extension build below stays
# here because it relies on the package already being importable.

def _ext_paths() -> list[Path]:
    return [PKG / f"gravotet{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _needs_rebuild() -> bool:
    built = [p for p in _ext_paths() if p.exists()]
    if not built:
        return True
    ext_mtime = max(p.stat().st_mtime for p in built)
    sources = [
        PKG / "setup.py",
        SRC / "gravotet_binding.cpp",
        SRC / "multigrid_solver.h",
        SRC / "multigrid_solver.cpp",
        SRC / "multigrid_solver_vcycle.cpp",
    ]
    return any(p.stat().st_mtime > ext_mtime for p in sources)


def _build_extension() -> None:
    print("Building the local GravoTet extension...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=PKG,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        print("Extension build failed. Full build log follows:\n")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(
            result.returncode, result.args,
            output=result.stdout, stderr=result.stderr,
        )
    print("Extension build finished.")


def ensure_extension() -> Any:
    """Bootstrap Python deps, (re)build the C++ extension, return the module."""
    _bootstrap.ensure_python()
    _bootstrap.ensure_dependencies()
    sys.path.insert(0, str(PKG.parent))   # for gravotet_demo package
    sys.path.insert(0, str(PKG))          # for gravotet C++ extension
    if _needs_rebuild():
        _build_extension()
    importlib.invalidate_caches()
    return importlib.import_module("gravotet")


def _prepare_output_dir(output_dir: Path) -> None:
    """Wipe and recreate `output_dir`."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

def build_hierarchy(gravotet: Any, mesh: Any, verbose: bool = False) -> tuple[Any, float]:
    """Construct the GravoTet hierarchy and return (solver, build_time_ms)."""
    from .solver import create_solver
    return create_solver(gravotet, mesh, verbose=verbose)


def _concat_hierarchy_below_plots(output_dir: Path) -> None:
    """Merge hierarchy_strip.png below combined.png into a single output.png.

    The hierarchy strip is resized to match the width of the combined figure
    and placed directly underneath it, producing a single ``output.png`` that
    contains both the residual/timing plots and the mesh hierarchy.
    Intermediate files (``combined.png`` and ``hierarchy_strip.png``) are
    deleted so only ``output.png`` remains.
    """
    from PIL import Image

    combined_path = output_dir / "combined.png"
    strip_path = output_dir / "hierarchy_strip.png"
    output_path = output_dir / "output.png"

    if not combined_path.exists() or not strip_path.exists():
        return

    combined = Image.open(str(combined_path))
    strip = Image.open(str(strip_path))

    # Resize strip to match combined width, preserving aspect ratio
    target_w = combined.width
    scale = target_w / strip.width
    new_h = int(strip.height * scale)
    strip_resized = strip.resize((target_w, new_h), Image.LANCZOS)

    # Vertically concatenate: combined on top, hierarchy strip below
    total_h = combined.height + new_h
    result = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    result.paste(combined, (0, 0))
    result.paste(strip_resized, (0, combined.height))

    combined.close()
    strip.close()

    result.save(str(output_path), "PNG")

    # Clean up intermediates — only output.png should remain
    combined_path.unlink(missing_ok=True)
    strip_path.unlink(missing_ok=True)


def run_suite(
    gravotet: Any,
    mesh: Any,
    *,
    label: str,
    output_subdir: str,
    problems: Iterable[str] = ("poisson",),
    make_figure: bool = False,
    verbose: bool = False,
    max_cycles: int = 150,
) -> dict[str, Any]:
    """Build the hierarchy, solve every requested PDE, and write the outputs.

    Parameters
    ----------
    gravotet      : imported C++ extension module
    mesh          : ``gravotet.TetrahedralMesh``
    label         : human-readable label for log lines (e.g. ``"Cube 100k"``)
    output_subdir : name of the per-suite directory under ``output/``
    problems      : iterable of ``"poisson"`` and/or ``"biharmonic"``
    make_figure   : generate ``combined_<subdir>.png`` (residual + timing)
    verbose       : forward to the C++ solver
    """
    from .pde import assemble_problem
    from .solver import solve_problem
    from .export import save_hierarchy_meshes, save_prolongation_matrices

    print(f"\n{'='*60}")
    print(f"  {label}  ({int(mesh.num_vertices()):,} vertices, "
          f"{int(mesh.num_tetrahedra()):,} tets)")
    print(f"{'='*60}")

    solver, hierarchy_build_ms = build_hierarchy(gravotet, mesh, verbose=verbose)

    summary: dict[str, Any] = {
        "label": label,
        "binding_version": gravotet.version(),
        "num_vertices": int(mesh.num_vertices()),
        "num_tetrahedra": int(mesh.num_tetrahedra()),
        "hierarchy_levels": int(solver.num_levels()),
        "hierarchy_build_ms": float(hierarchy_build_ms),
        "hierarchy_level_sizes": [
            {
                "level": i,
                "vertices": int(len(solver.all_vertices[i])),
                "tetrahedra": int(len(solver.all_tetrahedra[i])),
            }
            for i in range(len(solver.all_vertices))
        ],
        "problems": {},
    }

    for name in problems:
        print(f"  Solving {name} ...")
        t0 = time.perf_counter()
        problem = assemble_problem(solver, name, resolution=int(mesh.num_vertices()))
        problem.hierarchy_build_ms = hierarchy_build_ms
        result = solve_problem(solver, problem, max_cycles=max_cycles)
        result["assemble_and_solve_wall_s"] = time.perf_counter() - t0
        summary["problems"][name] = result

    output_dir = ROOT / "output" / output_subdir
    _prepare_output_dir(output_dir)
    save_hierarchy_meshes(solver, output_dir)
    save_prolongation_matrices(solver, output_dir)

    if make_figure:
        from .plotting import save_combined_figure
        save_combined_figure(summary, output_dir)

        # Concatenate hierarchy strip below the combined residual+timing figure
        # so the supplementary produces a single self-contained image.
        _concat_hierarchy_below_plots(output_dir)

    (output_dir / "data.json").write_text(json.dumps(summary, indent=2))

    _print_summary(label, summary)
    print(f"  Output: {output_dir}")
    return summary


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n--- {label} results ---")
    print(f"  Binding: {summary['binding_version']}")
    print(
        f"  Levels: {summary['hierarchy_levels']}  |  "
        f"hierarchy build: {summary['hierarchy_build_ms']:.1f} ms"
    )
    for lvl in summary["hierarchy_level_sizes"]:
        print(f"    Level {lvl['level']}: "
              f"{lvl['vertices']:,} verts, {lvl['tetrahedra']:,} tets")
    for name, result in summary["problems"].items():
        print(
            f"  [{name}] converged={result['converged']}  "
            f"cycles={result['num_cycles']}  "
            f"rel_res={result['relative_residual']:.3e}  "
            f"solve={result['total_time_ms']:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Argparse helper used by the driver scripts
# ---------------------------------------------------------------------------

DEFAULT_PROBLEMS = ("poisson", "biharmonic")
DEFAULT_MAX_CYCLES = 500


def parse_common_args(description: str) -> argparse.Namespace:
    """Common CLI surface shared by every driver script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable C++ hierarchy logging")
    parser.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        choices=["poisson", "biharmonic"],
        help="PDE(s) to solve (default: %(default)s)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=DEFAULT_MAX_CYCLES,
        help="V-cycle iteration cap (default: %(default)s)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Unified driver entry point
# ---------------------------------------------------------------------------

def run_demo(
    label: str,
    output_subdir: str,
    build_mesh: Callable[[Any], Any],
    description: str | None = None,
) -> int:
    """End-to-end demo entry shared by every driver script.

    A driver supplies (a) a human-readable ``label``, (b) the ``output_subdir``
    name under ``output/``, and (c) a ``build_mesh(gravotet) -> TetrahedralMesh``
    callable.  Everything else - environment bootstrap, extension build,
    hierarchy construction, Poisson + biharmonic solves, residual / timing
    figure, hierarchy mesh and prolongation export, JSON summary, log
    output - is identical across demos and lives here.
    """
    args = parse_common_args(description or label)
    gravotet = ensure_extension()
    mesh = build_mesh(gravotet)
    run_suite(
        gravotet, mesh,
        label=label,
        output_subdir=output_subdir,
        problems=args.problems,
        make_figure=True,
        verbose=args.verbose,
        max_cycles=args.max_cycles,
    )
    return 0
