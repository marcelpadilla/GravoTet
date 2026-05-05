"""Thin Python wrapper around the supplementary C++ hierarchy and V-cycle.

This module keeps the Python side small: build the Ours hierarchy once,
assemble each PDE separately, and solve with the same V-cycle interface used in
the paper experiments.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .pde import ProblemData


def create_solver(gravotet: Any, mesh: Any, verbose: bool = False):
    """Construct the Ours hierarchy once for the supplied mesh."""
    solver = gravotet.MultigridSolver()
    solver.verbose = verbose
    solver.feature_preserve = True
    solver.min_verts = 200
    solver.max_levels = 25
    solver.use_dense_coarse_solver = True

    start = time.perf_counter()
    solver.construct_prolongation_ours(mesh)
    hierarchy_build_ms = (time.perf_counter() - start) * 1000.0
    return solver, hierarchy_build_ms


def solve_problem(solver: Any, problem: ProblemData, max_cycles: int = 150) -> dict[str, Any]:
    """Solve one assembled system and return the metrics used in the figure."""
    x0 = np.zeros(problem.b.shape[0], dtype=np.float64)

    setup_start = time.perf_counter()
    ok = solver.build_vcycle_hierarchy(problem.A)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    if not ok:
        raise RuntimeError(f"Failed to build V-cycle hierarchy for {problem.name}")

    result = solver.solve_vcycle(
        b=problem.b,
        x0=x0,
        max_cycles=max_cycles,
        pre_sweeps=2,
        post_sweeps=2,
        tol=problem.tolerance,
        timeout_ms=0.0,
        smoother="jacobi",
        jacobi_omega=0.6667,
        collect_timing=True,
        mass_diag_inv=np.array([], dtype=np.float64),
    )

    x = np.asarray(result["x"], dtype=np.float64)
    rel_error = float(np.linalg.norm(x - problem.x_true) / max(np.linalg.norm(problem.x_true), 1e-30))
    rel_residual = float(np.linalg.norm(problem.A @ x - problem.b) / max(np.linalg.norm(problem.b), 1e-30))

    return {
        "problem": problem.name,
        "converged": bool(result["converged"]),
        "timed_out": bool(result["timed_out"]),
        "num_cycles": int(result["num_cycles"]),
        "total_time_ms": float(result["total_time_ms"]),
        "hierarchy_build_ms": float(getattr(problem, "hierarchy_build_ms", 0.0)),
        "setup_time_ms": setup_ms,
        "relative_error": rel_error,
        "relative_residual": rel_residual,
        "tolerance": float(problem.tolerance),
        "coarse_solver_name": str(result.get("coarse_solver_name", "unknown")),
        "residual_history": list(result.get("residual_history", [])),
        "cycle_time_ms_history": list(result.get("cycle_time_ms_history", [])),
        "vcycle_timing": dict(result.get("vcycle_timing", {})),
        "num_vertices": int(problem.num_vertices),
        "num_tetrahedra": int(problem.num_tetrahedra),
        "num_fixed_vertices": int(problem.fixed_idx.size),
    }
