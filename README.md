# GravoTet — Supplementary Code

Self-contained reference implementation accompanying the paper
**_GravoTet: A Fast Multigrid Hierarchy Construction for Tetrahedral Meshes_**.

The demo reproduces the Ours hierarchy construction and solves the
Poisson and biharmonic test problems on a tetrahedral cube mesh, exactly
matching the workflow described in the paper.

## Abstract

> Geometric multigrid (GMG) methods are a fundamental tool for efficiently
> solving large sparse linear systems. A requirement for GMG is a hierarchy
> of grids; however, many practical volumetric domains are available only as
> single, irregular tetrahedral meshes, making the construction of a multigrid
> hierarchy necessary. Existing approaches often trade off speed against
> hierarchy quality: remeshing- or coarsening-based methods can be expensive
> to construct, whereas graph-based techniques are fast but often yield weaker
> multigrid performance. We introduce **GravoTet**, which bridges this gap by
> combining geometric structure with graph-based efficiency to construct fast
> and effective multigrid hierarchies. GravoTet builds a vertex hierarchy and
> then generates graph-Voronoi diagrams whose dual cells define coarse
> tetrahedra, enabling rapid construction of multigrid levels. Boundary
> elements are explicitly prioritized during both sampling and tet generation
> to preserve boundary. In our evaluation, we solve Poisson and biharmonic
> problems on irregular tetrahedral meshes and compare GravoTet against
> state-of-the-art geometric multigrid, algebraic multigrid and direct
> solvers, demonstrating superior performance, particularly on large meshes.

## Quick start

From the repository root, run the **single command** below. It works on
**macOS, Linux, and Windows**:

```bash
python run_demo.py
```

That one command will:

1. install any missing Python packages (`numpy`, `scipy`, `matplotlib`, `pybind11`),
2. build the local C++ extension via `pybind11`,
3. build the Ours hierarchy once,
4. solve the Poisson and biharmonic cube problems,
5. write the combined figure and a JSON summary into `output/`.

Example outputs:

- `output/combined_cube40.png`
- `output/run_demo_cube40.json`

Optional flags:

```bash
python run_demo.py --resolution 20    # smaller cube for faster runs
python run_demo.py --verbose          # enable C++ hierarchy logging
```

## Requirements

- **Python 3.8 – 3.13**
- a **C++17** compiler:
  - macOS — Xcode Command Line Tools
  - Linux — GCC or Clang
  - Windows — Visual Studio Build Tools

No conda environment is required. No `libigl` or SuiteSparse / CHOLMOD
installation is required — the supplement is intentionally self-contained.

## Repository layout

```
GravoTet/
├── run_demo.py            ← single entry point (run this)
├── README.md
├── THIRD_PARTY_LICENSES.md
├── src/                   ← C++ hierarchy + V-cycle solver + pybind11 binding
├── deps/eigen/            ← vendored Eigen headers
└── gravotet_demo/         ← Python helpers and build scaffolding
    ├── __init__.py
    ├── pde.py             ← stiffness, mass, Poisson and biharmonic assembly
    ├── solver.py          ← Ours hierarchy + V-cycle wrapper
    ├── plotting.py        ← paper-style residual and timing figure
    ├── setup.py           ← pybind11 build configuration
    ├── requirements.txt   ← Python dependencies
    └── test_cube_problems.py  ← optional smoke test
```

## Relation to the paper

| Paper concept | Source location |
|---|---|
| Ours hierarchy construction | [src/multigrid_solver.cpp](src/multigrid_solver.cpp) |
| V-cycle solver and Galerkin coarse operators | [src/multigrid_solver_vcycle.cpp](src/multigrid_solver_vcycle.cpp) |
| Python binding | [src/gravotet_binding.cpp](src/gravotet_binding.cpp) |
| Stiffness `S`, lumped mass `M`, biharmonic `S M⁻¹ S` | [gravotet_demo/pde.py](gravotet_demo/pde.py) |
| Build hierarchy once, reuse for both PDEs | [gravotet_demo/solver.py](gravotet_demo/solver.py) |
| Paper-style residual and timing plot | [gravotet_demo/plotting.py](gravotet_demo/plotting.py) |
| End-to-end driver | [run_demo.py](run_demo.py) |

The supplement is intentionally limited to the Ours method and to the
two PDEs reported in the paper. CHOLMOD / SuiteSparse are not used in this
supplement path; Eigen is vendored to keep the package self-contained and
reproducible.

## Optional smoke test

```bash
python -m gravotet_demo.test_cube_problems --resolution 10
```

Exits non-zero if either problem fails to converge or exceeds its
verification thresholds.

## Third-party code

This repository vendors the Eigen headers in [deps/eigen](deps/eigen). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for license terms.
