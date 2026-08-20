<div align="center">

# PySNE

**Finding *all* solutions of a system of nonlinear equations**
using Spiral Optimization (SPO) with Clustering Technique

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
[![Documentation](https://img.shields.io/badge/docs-website-0b78bf.svg)](https://p2ms-optimization.github.io/pysne-web/)

[Overview](#overview) ·
[Installation](#installation) ·
[Quick start](#quick-start) ·
[Custom problems](#defining-a-custom-nonlinear-system) ·
[Algorithm](#how-pysne-works) ·
[Testing](#running-the-tests)

</div>

---

## Overview

PySNE is a Python library for searching **every root** (all solutions) of a system of nonlinear equations (SNE) inside a bounded search space — not just a single solution, as classical Newton-type methods typically return. Its main use case is a system of nonlinear equations

```math
\vec{f}:D\to\mathbb{R}^m,
\qquad
\vec{f}(\vec{x})
=
\begin{bmatrix}
f_1(\vec{x}) \\
f_2(\vec{x}) \\
\vdots \\
f_m(\vec{x})
\end{bmatrix}
=
\vec{0}_m,
\qquad
\vec{x}
=
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}
\in
D
=
\prod_{j=1}^{n}[a_j,b_j]
\subset\mathbb{R}^n.
```

PySNE transforms the root-finding problem into a maximization problem by defining
the fitness $F$ as

```math
F(\vec{x})
=
\frac{1}{1+\sum_{i=1}^{m}\left|f_i(\vec{x})\right|}.
```

For finite equation values, $0<F(\vec{x})\le1$, and
$F(\vec{x})=1$ if and only if $\vec{f}(\vec{x})=\vec{0}$.

Candidate solution regions are identified through the Function Cluster procedure and then
refined independently with Spiral Optimization (SPO). The package also contains
experimental support for multimodal optimization and integer/Diophantine
systems.

> [!IMPORTANT]
> PySNE is research software under active development. It is designed to search
> for multiple solutions, but it does **not** mathematically guarantee that every
> root or optimum in a domain will be found. Results depend on the search bounds,
> parameter settings, dimensionality, and numerical properties of the problem.

PySNE is being developed as part of undergraduate research in Actuarial Science
at Institut Teknologi Bandung (ITB), with the goal of providing a reproducible
research implementation and a reusable experimental library.

## Current status

- Current package version: **0.2.0**.
- The source repository is currently private and accessible to authorized
  contributors.
- The package is not yet published on PyPI.
- A development release is available through TestPyPI.
- The package metadata currently declares support for Python **3.8 or newer**.
- Required runtime dependencies are NumPy and SciPy.

## Supported problem types

| Problem type | Base class | Result terminology | Status |
| --- | --- | --- | --- |
| Systems of nonlinear equations | `SNEProblem` | Roots | Primary use case |
| Multimodal optimization | `MultimodalProblem` | Optimal points | Experimental |
| Integer/Diophantine systems | `DiophantineProblem` | Integer roots | Experimental |

## Features

- Bounded search for multiple roots of nonlinear equation systems.
- Three-stage solver pipeline: clustering, local SPO refinement, and final
  filtering.
- Sobol low-discrepancy initialization through `scipy.stats.qmc`.
- An $n$-dimensional spiral transformation based on a composed rotation matrix.
- Duplicate filtering using a configurable distance threshold.
- Built-in SNE, multimodal, and Diophantine benchmark collections.
- Object-oriented problem classes for custom equations and search domains.
- Visualization module for plotting SNE and multimodal optimization results in 1D, 2D, and 3D.
- Only NumPy and SciPy are required by the core package.

## Installation

### Requirements

The current `pyproject.toml` declares:

- Python `>=3.8`
- NumPy `>=1.20.0`
- SciPy `>=1.7.0`

### Installation from source

```bash
git clone https://github.com/p2ms-optimization/pysne.git
cd pysne
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import pysne; print(pysne.__version__)"
```

The expected version for this repository state is:

```text
0.2.0
```

### PyPI installation

PySNE is not currently available as a public PyPI package. The following command
is planned for a future public release, but should not be presented as an
available installation method yet:

```bash
pip install pysne
```

## Quick start

The public solver entry point is `solve_system(problem, params)`. The following
example runs the first built-in nonlinear-system benchmark with its configured
search domain and parameters:

```python
import numpy as np

from pysne import solve_system
from pysne.problems import get_problem_set

# Instantiate benchmark problem 1.
problem = get_problem_set()[1]()

# Built-in problems provide recommended solver parameters.
_, params = problem.get_info()

# Run the clustering phase, SPO per Cluster, and final selection.
result = solve_system(problem, params, verbose=True)

print(f"Roots found: {len(result['roots'])}")
print(f"Elapsed time: {result['time_elapsed']:.3f} seconds")

# Independently inspect the equation residuals of each returned root.
for index, root in enumerate(result["roots"], start=1):
    residuals = np.abs([equation(root) for equation in problem.equations])
    print(
        f"Root {index}: {np.round(root, 8)} | "
        f"max residual={residuals.max():.3e}"
    )
```

A typical run of benchmark problem 1 finds six roots. Exact execution
time and numerical coordinates can vary across environments.

### Solver result

`solve_system` returns a dictionary with the following keys:

| Key | Type | Description |
| --- | --- | --- |
| `roots` | `numpy.ndarray` | Filtered roots or optimal points returned by the problem class. |
| `optimals` | `numpy.ndarray` | Alias of `roots` for optimization-oriented usage. |
| `clusters` | `list[Cluster]` | Candidate regions found during iterative clustering. |
| `time_elapsed` | `float` | Total solver time in seconds. |

For SNE problems, returned points pass the package's internal fitness-based
acceptance rule. Users should still evaluate the original equations directly,
as shown above, when reporting numerical results.

## Defining a custom nonlinear system

Create a subclass of `SNEProblem` and implement:

- `name`
- `get_equations()`
- `get_info()`

This example searches the positive quadrant for the intersection of a unit
circle and the line $x_1=x_2$:

```python
import numpy as np

from pysne import solve_system
from pysne.problems.base import SNEProblem


class PositiveCircleLineIntersection(SNEProblem):
    @property
    def name(self):
        return "Positive circle-line intersection"

    def get_equations(self):
        return [
            lambda x: x[0] ** 2 + x[1] ** 2 - 1.0,
            lambda x: x[0] - x[1],
        ]

    def get_info(self):
        domain = [
            (0.0, 1.5),
            (0.0, 1.5),
        ]

        params = {
            # Iterative clustering
            "m_cluster": 128,
            "k_cluster": 10,
            "gamma": 0.2,
            "r_cl": 0.95,
            "theta_cl": np.pi / 4,
            "num_check_points": 1,

            # SPO refinement inside each cluster
            "spo_m": 128,
            "spo_k_max": 200,
            "r": 0.95,
            "theta": np.pi / 4,

            # Final acceptance and duplicate filtering
            "epsilon": 1e-7,
            "delta": 1e-3,
        }

        return domain, params


problem = PositiveCircleLineIntersection()
_, params = problem.get_info()
result = solve_system(problem, params, verbose=True)

print(result["roots"])
```

The exact solution in the specified positive domain is
```math
\vec{x}^* = (\frac{1}{\sqrt{2}},\,\frac{1}{\sqrt{2}})^T
\quad \approx \quad
\vec{x}^*=(0.70710678,\,0.70710678)^T.
```

## How PySNE works

PySNE executes a three-phase pipeline (`pysne/solver.py`):

### Phase 1 — Clustering Phase

The solver generates a Sobol population over the bounded domain. It evaluates
the points, identifies promising regions using `gamma`, and updates cluster
centers and radius while the population follows spiral dynamics toward the
current best point.

Implementation: `pysne/clustering/modified_clustering_process.py`.

### Phase 2 — Spiral Optimization Phase (SPO per cluster)

For each cluster, PySNE constructs a local bounded search region, generates a new
Sobol population, and applies Spiral Optimization:

```math
\begin{aligned}\vec{x}_i(k+1)&= \vec{x}^*(k) + S_n(r,\theta)   \left(\vec{x}_i(k)-\vec{x}^{\star}(k)\right) \\&= S_n(r,\theta)\vec{x}_i(k) - \left(S_n(r,\theta)-I_n\right)\vec{x}^*(k), \\S_n(r,\theta) &= rR^{(n)}(\theta).\end{aligned}
```

Here, $\vec{x}_i(k)$ is search point $i$ at iteration $k$,
$\vec{x}^*(k)$ is the current best point, $0<r<1$ is the
contraction factor, $I_n$ is the $n\times n$ identity matrix, and
$R^{(n)}(\theta)$ is the composed $n$-dimensional rotation matrix.

Implementation: `pysne/optimizers/spo/`.

### Phase 3 — Final Selection

Each problem class determines how final candidates are selected:

* `SNEProblem` accepts a candidate $\vec{x}$ when
  $1-F(\vec{x})<\varepsilon$ and it lies inside the domain.
* Candidates $\vec{x}$ and $\vec{y}$ are treated as duplicates when
  $\lVert\vec{x}-\vec{y}\rVert\le\delta$; the point with the higher
  fitness is retained.
* `MultimodalProblem` accepts in-domain candidates that pass a local-extremum
  check in every coordinate direction, then removes nearby duplicates.
* `DiophantineProblem` forms
  $\vec{q}=\mathrm{round}(\vec{x})$, checks that $\vec{q}$ lies
  in the integer domain, and accepts it when
  $1-F(\vec{q})\le\varepsilon$; repeated integer solutions are removed.


## Key parameters

Parameters are supplied as a dictionary to `solve_system`.

| Parameter | Stage | Description |
| --- | --- | --- |
| `m_cluster` | Clustering | Number of initial Sobol points. Required. Powers of two provide the best Sobol balance properties. |
| `k_cluster` | Clustering | Number of clustering iterations. Required. |
| `gamma` | Clustering | Fitness cutoff. SNE and Diophantine problems use the absolute condition ($F(\vec{x})>\gamma$). Multimodal problems can use the relative condition ($F(\vec{x})>\gamma*F(\vec{x}^{\star})$). |
| `r_cl` | Clustering | Spiral contraction factor during clustering. Default: `0.95`. |
| `theta_cl` | Clustering | Spiral rotation angle during clustering. Default: $\pi/4$. |
| `num_check_points` | Clustering | Number of interpolation points evaluated between a candidate and its nearest cluster center. Default: `1`. |
| `spo_m` | SPO | Number of Sobol points generated inside each cluster. Fallback alias: `m`. |
| `spo_k_max` | SPO | Maximum number of SPO iterations per cluster. Fallback alias: `k_max`. |
| `r` | SPO | SPO contraction factor. Default: `0.95`. |
| `theta` | SPO | SPO rotation angle. Default: $\pi/4$. |
| `epsilon` | Selection | For SNE and Diophantine problems, the fitness-gap tolerance $\varepsilon$ applied to $1-F$. Multimodal problems also use it in final peak filtering. Default: $10^{-7}$. |
| `delta` | Selection | Euclidean duplicate threshold: candidates within $\lVert\vec{x}-\vec{y}\rVert\le\delta$ are merged. |

Built-in benchmarks may also define `expected_roots` as testing metadata. It is
used by tests and examples to evaluate solver results, but it is not used by the
solver to discover solutions.

## Benchmark collections

The repository currently includes these factories:

| Factory | Module | Contents |
| --- | --- | --- |
| `get_problem_set()` | `pysne.problems.benchmarks_sne` | Seven nonlinear-equation benchmark systems with IDs `1-7`. |
| `get_multimodal_problems()` | `pysne.problems.benchmarks_multimodal` | Seven numbered benchmark configurations and four named entries. |
| `get_diophantine_problems()` | `pysne.problems.benchmarks_diophantine` | Eighteen integer/Diophantine benchmark configurations with IDs `1-18`. |

Example runners are available under `examples/`:

```bash
python examples/run_sne.py 1
python examples/run_multimodal.py 1
python examples/run_diophantine.py 1
```

The SNE and multimodal runners also accept `all` and `all-verbose` to run all
numbered benchmark configurations. Some benchmark runs may take
substantially longer depending on their parameters.

## Running the tests

Install PySNE in editable mode together with its development dependencies:

```bash
pip install -e ".[dev]"
```

Run the complete test suite:

```bash
pytest -q 
```

To run a single lightweight SNE benchmark test:

```bash
pytest -q "tests/test_sne.py::test_sne_problem[1]"
```

Some tests run complete optimization benchmarks and may take longer than
ordinary unit tests.

## Project structure

```text
pysne/
├── pysne/
│   ├── __init__.py
│   ├── solver.py
│   ├── utils.py
│   ├── version.py
│   ├── clustering/
│   │   ├── model.py
│   │   ├── clustering_process.py
│   │   └── modified_clustering_process.py
│   ├── initialization/
│   │   └── sampling.py
│   ├── optimizers/
│   │   └── spo/
│   │       ├── engine.py
│   │       └── matrix.py
│   ├── problems/
│   │   ├── base.py
│   │   ├── benchmarks_sne.py
│   │   ├── benchmarks_multimodal.py
│   │   └── benchmarks_diophantine.py
│   └── visualization/
│       ├── __init__.py
│       ├── multimodal.py
│       └── sne.py
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

## Limitations

- PySNE requires finite bounds for every decision variable.
- Finding every root or optimum is not guaranteed.
- Large domains and higher-dimensional problems may require more evaluations.
- The value of `delta` affects whether nearby solutions are merged or retained
as separate candidates.
- Poorly scaled variables, discontinuities, singularities, overflow, and invalid function evaluations may reduce solver reliability.
- Sobol population sizes that are powers of two generally provide better
balance properties.
- The API and default parameters may still change in the future releases.

## Documentation

The documentation website contains the user guide, API notes, algorithm
explanations, examples, case studies, and research references:

**https://p2ms-optimization.github.io/pysne-web/**

| Repository | Purpose |
| --- | --- |
| [`p2ms-optimization/pysne`](https://github.com/p2ms-optimization/pysne) | Python package source, solver, algorithms, benchmarks, and tests |
| [`p2ms-optimization/pysne-web`](https://github.com/p2ms-optimization/pysne-web) | MkDocs documentation website source and research pages. |

## Contributing

Contributions, bug reports, and benchmark additions are welcome.

1. Fork the repository and create a branch for each focused change.
2. Add or update tests under `tests/` when behavior changes.
3. Run the relevant unit and benchmark tests locally.
4. Update the documentation when public APIs, parameters, or behavior change.
5. Ensure the relevant tests pass before opening a pull request.
6. Open a pull request with a clear description of the change.

Bugs and feature requests can be submitted through the
[issue tracker](https://github.com/p2ms-optimization/pysne/issues).

## References

- Sidarto, K. A. & Kania, A. (2015). Finding all solutions of systems of nonlinear equations using spiral dynamics inspired optimization with clustering. JACIII, 19(5).
- Sidarto, K. A., Kania, A., & Sumarti, N. (2017). Finding multiple solutions of multimodal optimization using spiral optimization algorithm with clustering. MENDEL, 23(1).
- Sumarti, N., et al. (2023). A method for finding numerical solutions to diophantine equations using spiral optimization algorithm with clustering. Applied Soft Computing.
- Tamura, K. & Yasuda, K. (2011). Spiral Dynamics Inspired Optimization. JACIII, 15(8).

## Citation

If you use PySNE in academic work, please cite it:

```bibtex
@software{pysne2026,
  title   = {PySNE: Finding All Solutions of Systems of Nonlinear Equations using Spiral Optimization with Clustering},
  author  = {Hermawan, Aldy Nugraha and Isriyanto, Azarya Benhanan},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/p2ms-optimization/pysne}
}
```


## Authors

- **Aldy Nugraha Hermawan** — `aldynugrahahermawan1702@gmail.com`
- **Azarya Benhanan Isriyanto** — `azaryaben@gmail.com`

Undergraduate research · Actuarial Science · Institut Teknologi Bandung

## License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
