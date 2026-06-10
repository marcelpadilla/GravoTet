"""Helpers for the GravoTet MG supplementary demos."""

from .meshes import make_cube_mesh, make_mesh_from_ply
from .pde import assemble_problem
from .plotting import save_combined_figure
from .runner import (
    DATA_DIR,
    ROOT,
    ensure_extension,
    parse_common_args,
    run_demo,
    run_suite,
)
from .solver import create_solver, solve_problem

__all__ = [
    "DATA_DIR",
    "ROOT",
    "assemble_problem",
    "create_solver",
    "ensure_extension",
    "make_cube_mesh",
    "make_mesh_from_ply",
    "parse_common_args",
    "run_demo",
    "run_suite",
    "save_combined_figure",
    "solve_problem",
]
