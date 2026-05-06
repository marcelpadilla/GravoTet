# GravoTet Supplementary — Agent Context

This repository is the **self-contained supplementary release** for the paper
*GravoTet: A Fast Multigrid Hierarchy Construction for Tetrahedral Meshes*.

The full research codebase (solver, experiments, paper writing) lives in the
separate repository at `c:\Users\mpadill\Desktop\code\gravo_mg_tet\`. Consult
that repository's `CLAUDE.md` for project-wide context.

## Purpose

A minimal, reviewer-facing demo. It has no CHOLMOD/SuiteSparse dependency and
no conda requirement. Any reviewer with Python 3.8–3.13 and a C++17 compiler
can run `python run_demo.py` and reproduce the Poisson and biharmonic results.

## Repository layout

```
GravoTet/
├── run_demo.py                  <- single entry point
├── CLAUDE.md                    <- this file
├── README.md
├── THIRD_PARTY_LICENSES.md
├── .scripts/reset.py            <- cleans output/ and build artifacts
├── code/
│   ├── cpp/                     <- C++ hierarchy + V-cycle + pybind11 binding
│   ├── python/
│   │   └── gravotet_demo/       <- Python helpers (pde, solver, plotting, export)
│   │       ├── pde.py           <- stiffness, mass, Poisson and biharmonic assembly
│   │       ├── solver.py        <- hierarchy build + V-cycle wrapper
│   │       ├── plotting.py      <- paper-style residual and timing figure
│   │       ├── setup.py         <- pybind11 build (invoked by run_demo.py)
│   │       ├── requirements.txt
│   │       └── test_cube_problems.py <- smoke test
│   └── deps/
│       └── eigen/               <- vendored Eigen (header-only)
└── output/
```

## Running

```bash
python run_demo.py               # default resolution 40
python run_demo.py --resolution 20   # faster
python .scripts/reset.py        # wipe output/ and build artifacts
```

## Platform notes

- **Windows (MSVC):** `#ifdef _MSC_VER / #define _USE_MATH_DEFINES` guards are
  at the top of each `.cpp` file to expose `M_PI` from `<cmath>`.
- **macOS:** `-mmacosx-version-min=10.15` in `gravotet_demo/setup.py`.
- The `D9025` warning (`overriding /std:c++latest with /std:c++17`) comes from
  pybind11's build_ext injecting `/std:c++latest`; it is harmless.

## Output

`run_demo.py` writes into `output/` (gitignored):
- `output/combined_cube<N>.png` — residual curves + timing histograms
- `output/run_demo_cube<N>.json` — numeric summary

## Do's and Don'ts

* **DO** use `run_demo.py` as the single entry point.
* **DO** use `.scripts/reset.py` to clean before re-testing from scratch.
* **DON'T** add CHOLMOD or other external dependencies — the supplement must
  remain self-contained.
* **DON'T** copy changes from `gravo_mg_tet/src/` blindly — this repo's C++
  is a stripped subset of the full solver.
