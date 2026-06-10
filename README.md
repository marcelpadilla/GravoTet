# GravoTet — Supplementary Code

Self-contained reference implementation accompanying the paper
**_GravoTet: A Fast Multigrid Hierarchy Construction for Tetrahedral Meshes_**.

> Published at **Shape Modeling International (SMI) 2026**, Istanbul, Turkey
> (July 6-9, 2026), in the _Computers & Graphics_ SMI 2026 Special Issue.

This supplement ships four end-to-end demos at increasing problem sizes. Each
builds the GravoTet hierarchy, solves Poisson and biharmonic with the V-cycle,
and writes a residual/timing figure, a hierarchy strip, per-level meshes and
renders, prolongation matrices, and a JSON metrics file.

| Driver | Mesh | Vertices | Input |
|---|---|---:|---|
| `run_demo_cube_200k.py`   | regular cube | 205,379   | generated in C++, no input file |
| `run_demo_spot_700k.py`   | Spot         | 708,877   | `data/spot.ply` (bundled parts) |
| `run_demo_sphere_1.5M.py` | unit ball    | 1,449,956 | `data/sphere_1.5M.ply` (bundled parts) |
| `run_demo_torus_2M.py`    | solid torus  | 2,010,682 | `data/torus_2M.ply` (bundled parts) |

All four drivers forward to a single shared entry point in
[code/python/gravotet_demo/runner.py](code/python/gravotet_demo/runner.py);
they differ only in the input mesh.

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

## Authors

| Author | Affiliation | Email |
|---|---|---|
| Marcel Padilla (corresponding) | ETH Zürich | marcel.padilla@inf.ethz.ch |
| Ruben Wiersma | ETH Zürich | ruben.wiersma@inf.ethz.ch |
| Tim Huisman | Delft University of Technology | T.J.Huisman-1@student.tudelft.nl |
| Jackson Campolattaro | Delft University of Technology | J.R.C.Campolatarro@tudelft.nl |
| Olga Sorkine-Hornung | ETH Zürich | sorkine@inf.ethz.ch |
| Klaus Hildebrandt | Delft University of Technology | k.a.hildebrandt@tudelft.nl |

**Affiliations.** ETH Zürich, Department of Computer Science,
Universitätstrasse 6, 8092 Zürich, Switzerland. Delft University of Technology,
Faculty of Electrical Engineering, Mathematics and Computer Science,
Mekelweg 4, 2628 CD Delft, The Netherlands.

**Keywords.** geometry processing, multigrid solver, volumetric variational
problems, Poisson problem, tetrahedral meshes.

## Quick start

The **cube demo runs out of the box** on macOS, Linux, and Windows:

```bash
python run_demo_cube_200k.py     # ~205 k verts, finishes in seconds
```

On the first run the driver installs any missing Python packages
(`numpy`, `scipy`, `matplotlib`, `pybind11`, `pyvista`, `pillow`) and builds
the local C++ extension via `pybind11`.

The spot, sphere, and torus meshes are larger than GitHub's 100 MB per-file
limit, so each one is committed as sub-100 MB parts under `data_chunks/`. They
are reassembled into `data/` automatically the first time a demo needs them
(no download, no network). You can also pre-build them explicitly:

```bash
python scripts/fetch_data.py     # assemble spot / sphere / torus into data/

python run_demo_spot_700k.py     # ~709 k verts, ~2 min
python run_demo_sphere_1.5M.py   # ~1.45 M verts, ~5 min
python run_demo_torus_2M.py      # ~2.01 M verts, ~10 min
```

Common flags (every driver accepts the same set):

```bash
python run_demo_<name>.py --problems poisson           # only Poisson
python run_demo_<name>.py --problems poisson biharmonic
python run_demo_<name>.py --max-cycles 1000            # raise V-cycle cap
python run_demo_<name>.py --verbose                    # C++ hierarchy logs
```

## What gets written to `output/`

Each driver creates one subdirectory under `output/` (wiped and rewritten on
every run):

```
output/<demo>/
├── output.png          residual-vs-time + timing bars + hierarchy strip
├── data.json           hierarchy + solve metrics
├── prolongations.npz   all prolongation matrices (CSR)
└── meshes/
    ├── level_{i}.ply   mesh per hierarchy level (loads in Blender)
    └── level_{i}.png   rendered exterior surface per level
```

Only `output/cube_200k/output.png` is committed to git (so the Results section
below renders); everything else under `output/` is gitignored.

## Requirements

- **Python 3.8 - 3.13**
- a **C++17** compiler:
  - macOS — Xcode Command Line Tools
  - Linux — GCC or Clang
  - Windows — Visual Studio Build Tools
- around **4 GB of free RAM** for the 2 M-vertex torus demo.
- around **0.5 GB of free disk** for the assembled meshes written into `data/`
  (in addition to the committed parts under `data_chunks/`).

No SuiteSparse/CHOLMOD and no conda are required. The V-cycle uses an Eigen
`SimplicialLDLT` (or a dense `LDLT` for small coarsest levels) as the
coarse-grid direct solver.

## Repository layout

```
GravoTet/
├── run_demo_cube_200k.py     <- entry point: cube  (~205 k verts, no download)
├── run_demo_spot_700k.py     <- entry point: Spot  (~709 k verts)
├── run_demo_sphere_1.5M.py   <- entry point: sphere (~1.45 M verts)
├── run_demo_torus_2M.py      <- entry point: torus  (~2.01 M verts)
├── README.md
├── THIRD_PARTY_LICENSES.md
├── scripts/
│   ├── fetch_data.py             <- assemble spot/sphere/torus meshes from data_chunks/
│   ├── split_meshes.py           <- (maintainer) split data/*.ply into sub-100 MB parts
│   ├── import_tetgen_meshes.py   <- (maintainer) build binary tet PLYs from TetGen output
│   └── reset.py                  <- wipe output/ and build artifacts
├── data_chunks/             <- committed sub-100 MB mesh parts (reassembled into data/)
├── data/                    <- assembled input meshes; built on demand, gitignored
└── code/
    ├── cpp/                  <- C++ hierarchy + V-cycle + pybind11 binding
    ├── python/
    │   └── gravotet_demo/    <- Python helpers and build scaffolding
    │       ├── runner.py         <- shared bootstrap, run_demo, run_suite
    │       ├── meshes.py         <- cube generator + PLY loader
    │       ├── pde.py            <- stiffness, mass, Poisson, biharmonic
    │       ├── solver.py         <- hierarchy + V-cycle wrapper
    │       ├── plotting.py       <- residual and timing figure
    │       ├── export.py         <- per-level PLY + PNG + prolongation export
    │       ├── setup.py          <- pybind11 build configuration
    │       ├── requirements.txt
    │       └── test_cube_problems.py  <- optional smoke test
    └── deps/
        └── eigen/            <- vendored Eigen headers (header-only)
```

## Where the paper's pieces live

| Paper concept | Source location |
|---|---|
| Hierarchy construction (sampling, exterior-first clustering, simplicial complex) | [code/cpp/multigrid_solver.cpp](code/cpp/multigrid_solver.cpp) |
| V-cycle solver with Chebyshev-accelerated Jacobi smoothing | [code/cpp/multigrid_solver_vcycle.cpp](code/cpp/multigrid_solver_vcycle.cpp) |
| Python binding | [code/cpp/gravotet_binding.cpp](code/cpp/gravotet_binding.cpp) |
| Stiffness `S`, lumped mass `M`, biharmonic `S M^{-1} S` | [code/python/gravotet_demo/pde.py](code/python/gravotet_demo/pde.py) |
| Build hierarchy once, reuse for both PDEs | [code/python/gravotet_demo/solver.py](code/python/gravotet_demo/solver.py) |
| Per-level PLY, PNG render, prolongation export | [code/python/gravotet_demo/export.py](code/python/gravotet_demo/export.py) |

## Optional smoke test

```bash
cd code/python
python -m gravotet_demo.test_cube_problems --resolution 10
```

Runs Poisson and biharmonic on a 1,000-vertex cube and exits non-zero if either
problem fails to converge or exceeds its verification thresholds. A fast sanity
check after rebuilding the C++ extension.

## Resetting to a clean state

```bash
python scripts/reset.py
```

Removes `output/`, the compiled extension, and setuptools build artifacts.
Cached meshes under `data/` are left in place.

## Results

The image below is produced by `run_demo_cube_200k.py` and committed directly
from `output/`. It shows the Poisson and biharmonic residual curves (left),
stacked Hierarchy / Setup / Solve timing bars (right), and the multigrid
hierarchy level strip (bottom). Running the spot, sphere, and torus demos
produces the same layout for those meshes.

![Cube 200k](output/cube_200k/output.png)

## Third-party code

This repository vendors the Eigen headers in
[code/deps/eigen](code/deps/eigen). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for license terms.

## Acknowledgments

The Spot tetrahedral mesh is derived from a surface model by
[Keenan Crane](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/),
released under the CC0 1.0 Universal (public domain) license. We tetrahedralized
it with TetGen using the same pipeline applied to all evaluation meshes in the
paper. The cube, sphere, and torus are derived from primitive geometry.
