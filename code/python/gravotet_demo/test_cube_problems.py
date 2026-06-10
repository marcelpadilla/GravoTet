"""Smoke test for the supplementary Poisson and biharmonic cube examples.

Quick validation of a clean checkout: builds the GravoTet hierarchy on a
small cube and checks convergence for both PDEs.  Exits non-zero on any
failure.

Run from the repository root:

    python -m gravotet_demo.test_cube_problems --resolution 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gravotet_demo import ensure_extension, make_cube_mesh, run_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    gravotet = ensure_extension()
    mesh = make_cube_mesh(gravotet, args.resolution)
    summary = run_suite(
        gravotet, mesh,
        label=f"Cube smoke (res {args.resolution})",
        output_subdir=f"smoke_cube_{args.resolution}",
        problems=("poisson", "biharmonic"),
        make_figure=False,
        verbose=False,
    )

    failed: list[str] = []
    for name, result in summary["problems"].items():
        if not result["converged"]:
            failed.append(f"{name}: solver did not converge")
        if result["timed_out"]:
            failed.append(f"{name}: solver timed out")
        if result["relative_residual"] > max(10.0 * result["tolerance"], 1e-10):
            failed.append(
                f"{name}: relative residual {result['relative_residual']:.3e} exceeds threshold"
            )
        if result["relative_error"] > 1e-4:
            failed.append(
                f"{name}: relative error {result['relative_error']:.3e} exceeds threshold"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    if failed:
        print("Supplement smoke test FAILED")
        for item in failed:
            print(f"  - {item}")
        return 1

    print("Supplement smoke test PASSED")
    for name, result in summary["problems"].items():
        print(
            f"  {name}: cycles={result['num_cycles']}, "
            f"rel_res={result['relative_residual']:.3e}, "
            f"rel_err={result['relative_error']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
