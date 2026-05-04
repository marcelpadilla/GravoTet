# GravoTet MG Supplementary Demo

This repository contains a minimal, self-contained supplementary implementation for the paper *A Fast Multigrid Hierarchy Construction for Tetrahedral Meshes*.

It reproduces the `ours_pro` hierarchy construction and solves two volumetric test problems on a tetrahedral cube mesh:
- Poisson
- biharmonic

The demo produces one combined figure with:
- relative residual versus solver time
- stacked computation-time histograms

The top row shows Poisson and the bottom row shows biharmonic.

## Relation to the paper

The code follows the same split as the method description in the manuscript.

- [src](src): C++ hierarchy construction, prolongation, and V-cycle solver
- [src/gravotet_binding.cpp](src/gravotet_binding.cpp): minimal Python binding for the supplementary interface
- [gravotet_demo/pde.py](gravotet_demo/pde.py): assembly of the Poisson and biharmonic systems used in the demo
- [gravotet_demo/solver.py](gravotet_demo/solver.py): builds the `ours_pro` hierarchy once and reuses it for both problems
- [gravotet_demo/plotting.py](gravotet_demo/plotting.py): paper-style residual and timing plots
- [run_demo.py](run_demo.py): end-to-end entry point

The notation in [gravotet_demo/pde.py](gravotet_demo/pde.py) matches the paper: `S` is the stiffness matrix, `M` is the lumped barycentric mass matrix, and the biharmonic operator is assembled as $S M^{-1} S$.
The supplementary assembly is implemented directly in NumPy and SciPy, so no external geometry-processing package is required at runtime.

## Repository layout

- [run_demo.py](run_demo.py): single entry point for the supplementary demo
- [test_cube_problems.py](test_cube_problems.py): smoke test for the two cube problems
- [setup.py](setup.py): local `pybind11` build for the C++ extension
- [requirements.txt](requirements.txt): Python dependencies installed on first run
- [src](src): minimal C++ solver and binding sources
- [gravotet_demo](gravotet_demo): Python assembly, solve, and plotting helpers
- [deps/eigen](deps/eigen): vendored Eigen headers

## Requirements

- Python 3.8 to 3.13
- a C++17 compiler
  - macOS: Xcode Command Line Tools
  - Linux: GCC or Clang
  - Windows: Visual Studio Build Tools

No conda environment is required.
No `libigl` installation is required.

## Run the demo

From the repository root:

```text
python run_demo.py
```

On first execution, the script:
1. installs only the missing Python packages from [requirements.txt](requirements.txt)
2. bootstraps `pip` automatically if the interpreter does not already provide it
3. builds the local C++ extension with [setup.py](setup.py)
4. runs the Poisson and biharmonic cube examples
5. writes the combined figure and JSON summary into `output/`

Example output files:
- `output/combined_cube40.png`
- `output/run_demo_cube40.json`

If you omit `--json`, the script still saves a default summary file named `output/run_demo_cube<resolution>.json`.

## Run the smoke test

```text
python test_cube_problems.py --resolution 10
```

The smoke test exits with a non-zero status if either problem fails to converge or exceeds its verification thresholds.

## Notes on the implementation

- The supplement is intentionally limited to the `ours_pro` method.
- The hierarchy is built once and reused for both PDE examples.
- CHOLMOD / SuiteSparse are not used in this supplement path.
- Eigen is vendored to keep the package self-contained and reproducible.
- The tested Python range is 3.8 to 3.13 for the current self-contained NumPy/SciPy-based supplement path.

## Third-party code

The repository vendors Eigen headers in [deps/eigen](deps/eigen). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
