"""Run the supplementary cube demo.

This is the publication-facing entry point: it installs missing Python
dependencies, builds the local binding if needed, constructs the `ours_pro`
hierarchy once, solves the two PDE examples, and writes the combined figure.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MIN_PYTHON = (3, 8)
MAX_TESTED_PYTHON = (3, 13)

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pybind11": "pybind11",
}


def _requirement_specs() -> dict[str, str]:
    """Return install specs keyed by package name from `requirements.txt`."""
    specs: dict[str, str] = {}
    for raw_line in (HERE / "requirements.txt").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package_name = line.split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        specs[package_name] = line
    return specs


def _extension_candidates() -> list[Path]:
    """Return possible in-place extension module paths for `gravotet`."""
    return [HERE / f"gravotet{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _run_checked(command: list[str], *, cwd: Path, failure_message: str) -> None:
    """Run a subprocess and re-raise failures with reviewer-facing guidance."""
    try:
        subprocess.check_call(command, cwd=cwd)
    except FileNotFoundError as exc:
        raise RuntimeError(failure_message) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(failure_message) from exc


def _needs_rebuild() -> bool:
    """Rebuild when the extension is missing or older than its sources."""
    candidates = [path for path in _extension_candidates() if path.exists()]
    if not candidates:
        return True

    ext_mtime = max(path.stat().st_mtime for path in candidates)
    source_paths = [
        HERE / "setup.py",
        HERE / "src" / "gravotet_binding.cpp",
        HERE / "src" / "multigrid_solver.h",
        HERE / "src" / "multigrid_solver.cpp",
        HERE / "src" / "multigrid_solver_vcycle.cpp",
    ]
    return any(path.stat().st_mtime > ext_mtime for path in source_paths)


def ensure_supported_python() -> None:
    """Fail early on Python versions outside the tested dependency range."""
    version = sys.version_info[:2]
    if version < MIN_PYTHON or version > MAX_TESTED_PYTHON:
        min_text = ".".join(str(part) for part in MIN_PYTHON)
        max_text = ".".join(str(part) for part in MAX_TESTED_PYTHON)
        current_text = ".".join(str(part) for part in version)
        raise RuntimeError(
            "This supplementary demo is tested with Python "
            f"{min_text}-{max_text}. Detected Python {current_text}. "
            "Please rerun with a local system Python in the tested range."
        )


def ensure_python_deps() -> None:
    """Install only the missing Python requirements on first run."""
    missing = []
    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        try:
            import pip  # noqa: F401
        except ImportError:
            print("Bootstrapping pip with ensurepip...")
            _run_checked(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                cwd=HERE,
                failure_message=(
                    "Unable to bootstrap pip for the supplementary demo. "
                    "Please install pip for this Python interpreter and rerun `python run_demo.py`."
                ),
            )

        requirement_specs = _requirement_specs()
        install_specs = [requirement_specs[package_name] for package_name in missing]
        print(f"Installing missing Python packages: {', '.join(missing)}")
        _run_checked(
            [sys.executable, "-m", "pip", "install", *install_specs],
            cwd=HERE,
            failure_message=(
                "Failed to install the Python dependencies for the supplementary demo. "
                "Please check your internet connection and local Python packaging setup, "
                "then rerun `python run_demo.py`."
            ),
        )


def ensure_binding_built() -> Any:
    """Build the local pybind11 extension if it is not yet importable."""
    sys.path.insert(0, str(HERE))
    if _needs_rebuild():
        print("Building the local GravoTet extension...")
        _run_checked(
            [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
            cwd=HERE,
            failure_message=(
                "Failed to build the supplementary C++ extension. "
                "Please install a working C++17 toolchain and rerun the demo "
                "(macOS: Xcode Command Line Tools, Linux: GCC/Clang build tools, "
                "Windows: Visual Studio Build Tools)."
            ),
        )

    try:
        return importlib.import_module("gravotet")
    except ImportError:
        print("Retrying local extension build...")
        _run_checked(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=HERE,
            failure_message=(
                "The supplementary extension could not be imported or rebuilt. "
                "Please confirm that Python development headers and a C++17 compiler are available, "
                "then rerun `python run_demo.py`."
            ),
        )
        importlib.invalidate_caches()
        return importlib.import_module("gravotet")


def run_suite(resolution: int = 20, verbose: bool = False) -> dict[str, Any]:
    """Run the two supplementary problems and return a compact summary.

    The hierarchy is built once and reused for both the Poisson and biharmonic
    systems, matching the intended supplementary workflow.
    """
    ensure_supported_python()
    ensure_python_deps()
    gravotet = ensure_binding_built()

    from gravotet_demo.pde import assemble_problem
    from gravotet_demo.solver import create_solver, solve_problem

    mesh = gravotet.MultigridSolver.generate_cube_mesh(resolution)
    solver, hierarchy_build_ms = create_solver(gravotet, mesh, verbose=verbose)

    summary: dict[str, Any] = {
        "resolution": resolution,
        "binding_version": gravotet.version(),
        "hierarchy_levels": int(solver.num_levels()),
        "hierarchy_build_ms": float(hierarchy_build_ms),
        "problems": {},
    }

    for problem_name in ("poisson", "biharmonic"):
        assembled = assemble_problem(solver, problem_name, resolution)
        assembled.hierarchy_build_ms = hierarchy_build_ms
        summary["problems"][problem_name] = solve_problem(solver, assembled)

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Binding: {summary['binding_version']}")
    print(
        f"Cube resolution {summary['resolution']} | "
        f"levels={summary['hierarchy_levels']} | "
        f"hierarchy={summary['hierarchy_build_ms']:.1f} ms"
    )
    for name, result in summary["problems"].items():
        print(
            f"[{name}] converged={result['converged']} cycles={result['num_cycles']} "
            f"rel_res={result['relative_residual']:.3e} rel_err={result['relative_error']:.3e} "
            f"solve={result['total_time_ms']:.1f} ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=40, help="Cube resolution for the tetrahedral test mesh")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save a JSON summary (defaults to output/run_demo_cube<resolution>.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose C++ hierarchy logging")
    args = parser.parse_args()

    summary = run_suite(resolution=args.resolution, verbose=args.verbose)

    from gravotet_demo.plotting import save_combined_figure

    png_path = save_combined_figure(summary, HERE / "output")
    summary["figure_png"] = str(png_path)
    json_path = args.json if args.json is not None else HERE / "output" / f"run_demo_cube{args.resolution}.json"
    _print_summary(summary)
    print(f"Combined figure: {png_path.name}")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"JSON summary: {json_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
