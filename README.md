# GravoTet — Supplementary Code

Reference implementation for **GravoTet: A Fast Multigrid Hierarchy
Construction for Tetrahedral Meshes**, presented at Shape Modeling
International (SMI) 2026, Istanbul, and published in the _Computers & Graphics_
SMI 2026 Special Issue.

Marcel Padilla<sup>1</sup>, Ruben Wiersma<sup>1</sup>, Tim Huisman<sup>2</sup>, Jackson Campolattaro<sup>2</sup>, Olga Sorkine-Hornung<sup>1</sup>, Klaus Hildebrandt<sup>2</sup>

<sub><sup>1</sup> ETH Zürich &nbsp;&middot;&nbsp; <sup>2</sup> Delft University of Technology &nbsp;&middot;&nbsp; correspondence: marcel.padilla@inf.ethz.ch</sub>

GravoTet constructs a multigrid hierarchy on a tetrahedral mesh by coarsening
the vertex set and forming graph-Voronoi cells whose duals define coarse
tetrahedra, prioritizing boundary elements to preserve fidelity. Each demo
builds the hierarchy, solves a Poisson and a biharmonic problem with a V-cycle,
and writes a figure of the residual curves, timing bars, and hierarchy levels.

## Clone and run

```bash
python run_demo_cube_200k.py     # runs out of the box, finishes in seconds
python run_demo_spot_700k.py     # ~2 min
python run_demo_sphere_1.5M.py   # ~5 min
python run_demo_torus_2M.py      # ~10 min
```

The cube is generated in C++ and needs no input. The other three meshes ship in
the repository and are reassembled on first run, with no download. Every driver
also accepts `--problems`, `--max-cycles`, and `--verbose`.

## Results

**Cube — 205k vertices**
![Cube 200k](output/cube_200k/output.png)

**Spot — 709k vertices**
![Spot 700k](output/spot_700k/output.png)

**Sphere — 1.45M vertices**
![Sphere 1.5M](output/sphere_1.5M/output.png)

**Torus — 2.01M vertices**
![Torus 2M](output/torus_2M/output.png)

Each run also writes per-level meshes and renders, the prolongation matrices,
and a `data.json` of metrics under `output/<demo>/`.

---

## Requirements and setup

- Python 3.8-3.13
- a C++17 compiler: Xcode Command Line Tools (macOS), GCC or Clang (Linux),
  Visual Studio Build Tools (Windows)
- around 4 GB of free RAM for the 2M-vertex torus demo

On the first run the driver installs any missing Python packages (`numpy`,
`scipy`, `matplotlib`, `pybind11`, `pyvista`, `pillow`) and builds the local
C++ extension via `pybind11`. No conda and no SuiteSparse/CHOLMOD are required:
the V-cycle uses an Eigen `SimplicialLDLT` as its coarse-grid direct solver.

## Where the paper's pieces live

| Paper concept | Source |
|---|---|
| Hierarchy construction (sampling, exterior-first clustering, simplicial complex) | `code/cpp/multigrid_solver.cpp` |
| V-cycle with Chebyshev-accelerated Jacobi smoothing | `code/cpp/multigrid_solver_vcycle.cpp` |
| Stiffness, lumped mass, and biharmonic assembly | `code/python/gravotet_demo/pde.py` |

## Maintenance

```bash
python scripts/fetch_data.py    # pre-assemble the large meshes into data/
python scripts/reset.py         # wipe output/ and build artifacts
cd code/python && python -m gravotet_demo.test_cube_problems --resolution 10
```

## Third-party code and acknowledgments

This repository vendors the Eigen headers under `code/deps/eigen`; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The Spot mesh is derived
from a CC0 surface model by
[Keenan Crane](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/),
tetrahedralized with TetGen.
