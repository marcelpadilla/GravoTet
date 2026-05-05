"""GravoTet supplementary demo.

Single entry point that installs missing Python dependencies, builds the local
C++ extension, constructs the Ours hierarchy once, solves the Poisson and
biharmonic cube problems, and writes the following into `output/`:

  - combined_cube{resolution}.png  -- residual-vs-time and timing figure
  - data.json                      -- all computation metrics
  - meshes/level_{i}.npz           -- vertex positions and connectivity per hierarchy level
  - prolongations.npz              -- all prolongation matrices in CSR format

Run from the repository root:

    python run_demo.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "gravotet_demo"
SRC = ROOT / "src"
MIN_PY, MAX_PY = (3, 8), (3, 13)
REQUIRED = ("numpy", "scipy", "matplotlib", "pybind11")


def _ensure_python() -> None:
    v = sys.version_info[:2]
    if not (MIN_PY <= v <= MAX_PY):
        raise SystemExit(
            f"This supplementary demo is tested with Python "
            f"{MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]} "
            f"(detected {v[0]}.{v[1]})."
        )


def _ensure_deps() -> None:
    missing = []
    for module in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return
    try:
        import pip  # noqa: F401
    except ImportError:
        print("Bootstrapping pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    specs = [
        line.strip()
        for line in (PKG / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Installing missing Python packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *specs])


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
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    print("Extension build finished.")


def _ensure_extension() -> Any:
    sys.path.insert(0, str(PKG))
    if _needs_rebuild():
        _build_extension()
    importlib.invalidate_caches()
    return importlib.import_module("gravotet")


def run_suite(resolution: int = 50, verbose: bool = False) -> tuple[dict[str, Any], Any]:
    """Build the hierarchy once, solve both PDEs, and return (summary, solver).

    The solver is returned so callers can export the hierarchy meshes and
    prolongation matrices without rebuilding.
    """
    _ensure_python()
    _ensure_deps()
    gravotet = _ensure_extension()

    from gravotet_demo.pde import assemble_problem
    from gravotet_demo.solver import create_solver, solve_problem

    mesh = gravotet.MultigridSolver.generate_cube_mesh(resolution)
    solver, hierarchy_build_ms = create_solver(gravotet, mesh, verbose=verbose)

    summary: dict[str, Any] = {
        "resolution": resolution,
        "binding_version": gravotet.version(),
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
    for name in ("poisson", "biharmonic"):
        problem = assemble_problem(solver, name, resolution)
        problem.hierarchy_build_ms = hierarchy_build_ms
        summary["problems"][name] = solve_problem(solver, problem)
    return summary, solver


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Binding: {summary['binding_version']}")
    print(
        f"Cube{summary['resolution']} | levels={summary['hierarchy_levels']} | "
        f"hierarchy={summary['hierarchy_build_ms']:.1f} ms"
    )
    for name, result in summary["problems"].items():
        print(
            f"[{name}] converged={result['converged']} cycles={result['num_cycles']} "
            f"rel_res={result['relative_residual']:.3e} rel_err={result['relative_error']:.3e} "
            f"solve={result['total_time_ms']:.1f} ms"
        )


def _prepare_output_dir(output_dir: Path) -> None:
    """Remove stale demo-managed artifacts so each run leaves one clean output set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("combined_cube*.png", "run_demo_cube*.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()

    for name in ("data.json", "prolongations.npz"):
        path = output_dir / name
        if path.exists():
            path.unlink()

    mesh_dir = output_dir / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    summary, solver = run_suite(resolution=args.resolution, verbose=args.verbose)

    from gravotet_demo.export import save_hierarchy_meshes, save_prolongation_matrices
    from gravotet_demo.plotting import save_combined_figure

    output_dir = ROOT / "output"
    _prepare_output_dir(output_dir)
    png_path = save_combined_figure(summary, output_dir)
    save_hierarchy_meshes(solver, output_dir)
    save_prolongation_matrices(solver, output_dir)

    json_path = output_dir / "data.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    _print_summary(summary)
    print(f"Figure:         {png_path.name}")
    print(f"JSON:           {json_path.name}")
    print(f"Meshes:         output/meshes/  ({summary['hierarchy_levels']} levels)")
    print(f"Prolongations:  prolongations.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
