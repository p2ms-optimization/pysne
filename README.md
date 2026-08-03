<div align="center">

# PySNE

**Finding *all* solutions of a system of nonlinear equations**
via Spiral Optimization (SPO) + Iterative Clustering

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/p2ms-optimization/pysne)
[![Docs](https://img.shields.io/badge/docs-website-0b78bf.svg)](https://p2ms-optimization.github.io/pysne-web/)

[Documentation](https://p2ms-optimization.github.io/pysne-web/) ·
[Quick Start](#-quick-start) ·
[Algorithm](#-how-it-works) ·
[Benchmarks](#-benchmark-problem-sets) ·
[Contributing](#-contributing)

</div>

---

## 📌 Overview

**PySNE** is a Python library for locating **every root** of a system of nonlinear
equations (SNE) inside a bounded search space — not just a single solution, as
classical Newton-type methods typically return.

Given a system

```text
f₁(x) = 0
f₂(x) = 0
   ⋮
fₙ(x) = 0,        x ∈ Ω ⊂ ℝⁿ
```

PySNE reformulates the problem as a global optimization task and combines two
mechanisms to recover all isolated roots in a single run:

1. **Iterative Clustering** — partitions the domain into neighborhoods (clusters)
   that are likely to contain a root.
2. **Spiral Optimization (SPO)** — a deterministic, rotation-based
   metaheuristic that refines each cluster down to a precise candidate root.

A final **selection & validation** stage merges duplicates and keeps only the
unique solutions whose residual `‖F(x)‖` falls below a tolerance `ε`.

> PySNE is developed as part of an undergraduate thesis in **Actuarial Science, ITB**,
> and is intended both as a reproducible research artifact and a reusable library.

---

## ✨ Features

- **All-solutions search** for multivariate nonlinear systems over a bounded domain.
- **Three-phase pipeline**: clustering → spiral optimization → validated selection.
- **Low-discrepancy initialization** using Sobol sequences (via SciPy QMC) for uniform domain coverage.
- **n-dimensional spiral operator** built from a composed rotation matrix `Sₙ = r·Rₙ(θ)`.
- **Built-in benchmark suites** (SNE, multimodal, GECCO, Diophantine) for reproducible evaluation.
- **Object-oriented `Problem` API** — bring your own equations, domain, and hyperparameters.
- Lightweight dependencies: only **NumPy** and **SciPy**.

---

## 📦 Installation

**Requirements:** Python ≥ 3.8, NumPy ≥ 1.20, SciPy ≥ 1.7.

From PyPI (when published):

```bash
pip install pysne
```

From source (recommended during development):

```bash
git clone https://github.com/p2ms-optimization/pysne.git
cd pysne
pip install -e .
```

---

## 🚀 Quick Start

The core entry point is `solve_system(problem, params)`. The fastest way to try
PySNE is to load a built-in benchmark `Problem` object:

```python
from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system

# 1. Load a benchmark problem
problems = get_problem_set()
problem = problems[1]()                 # instantiate the chosen problem

# 2. Use the problem's recommended domain + hyperparameters
domain, params = problem.get_info()

# 3. Run the full clustering + SPO pipeline
result = solve_system(problem, params, verbose=True)

# 4. Inspect the results
print("Roots found:", len(result["roots"]))
print(result["roots"])
print(f"Elapsed: {result['time_elapsed']:.3f}s")
```

`solve_system` returns a dictionary:

| Key            | Type            | Description                                            |
| -------------- | --------------- | ------------------------------------------------------ |
| `roots`        | `np.ndarray`    | Validated, de-duplicated solution points.              |
| `clusters`     | `list[Cluster]` | Clusters discovered during the localization phase.     |
| `time_elapsed` | `float`         | Total computation time in seconds.                     |

### Defining your own system

Subclass the base `Problem` to solve a custom system — for example the
**Circle–Exponential** system:

```text
x₁² + x₂² − 1 = 0
x₁ − e^(−x₂)  = 0
```

Provide the equations, the search domain `[(min, max), ...]`, and the
hyperparameters described in [Configuration](#%EF%B8%8F-key-hyperparameters), then
pass the instance to `solve_system`. See `pysne/problems/base.py` and the existing
benchmark modules for complete, working templates.

---

## 🧠 How It Works

PySNE executes a deterministic three-phase pipeline (`pysne/solver.py`):

### Phase 1 — Iterative Clustering
A Sobol-distributed population is dynamically grouped into clusters around
high-fitness regions. Points are moved iteratively using the spiral operator
toward the current best point, growing and merging clusters that bracket
potential roots. *(`pysne/clustering/`)*

### Phase 2 — SPO per Cluster
For each cluster, a local hypercube domain is constructed from its center and
radius (clamped to the global bounds). Fresh Sobol points are generated inside
that hypercube and refined by SPO:

```text
xₖ₊₁ = Sₙ · xₖ − (Sₙ − Iₙ) · x*ₖ ,     Sₙ = r · Rₙ(θ)
```

where `Rₙ(θ)` is the composed n-dimensional rotation matrix, `r` the spiral
radius, `θ` the rotation angle, and `x*` the incumbent best. Early stopping
triggers once the residual drops below `ε`. *(`pysne/optimizers/spo/`)*

### Phase 3 — Selection & Validation
Candidate roots are filtered: near-duplicates within distance `δ` are merged, and
only points satisfying the residual tolerance `ε` are retained as final roots.

---

## ⚙️ Key Hyperparameters

Hyperparameters are passed via the `params` dictionary.

| Parameter            | Phase       | Meaning                                              |
| -------------------- | ----------- | ---------------------------------------------------- |
| `m_cluster`          | Clustering  | Number of initial Sobol points.                      |
| `k_cluster`          | Clustering  | Iterations of the clustering loop.                   |
| `gamma`              | Clustering  | Fitness threshold for accepting/creating clusters.   |
| `r_cl`, `theta_cl`   | Clustering  | Spiral radius / angle used during clustering.        |
| `spo_m`             | SPO        | Points per cluster (aliased as `m`).                 |
| `spo_k_max`         | SPO        | Max SPO iterations (aliased as `k_max`).            |
| `r`, `theta`         | SPO        | Spiral radius (≈0.95) and angle (≈π/4).              |
| `epsilon` (`ε`)      | Validation  | Residual tolerance for accepting a root.             |
| `delta` (`δ`)        | Validation  | Distance threshold for merging duplicate roots.      |

> Each built-in benchmark ships with tuned defaults via `problem.get_info()`.

---

## 🧪 Benchmark Problem Sets

PySNE bundles several reproducible suites under `pysne/problems/`:

| Factory                     | Module                       | Contents                                                        |
| --------------------------- | ---------------------------- | --------------------------------------------------------------- |
| `get_problem_set()`         | `benchmarks_sne.py`          | Classic SNE problems (e.g. Circle–Exponential).                 |
| `get_multimodal_problems()` | `benchmarks_multimodal.py`   | Multimodal functions (Himmelblau, Six-Hump Camel Back, Shubert).|
| `get_gecco_problems()`      | `benchmarks_gecco.py`        | GECCO niching-style multimodal benchmarks.                      |
| `get_diophantine_problems()`| `benchmarks_diophantine.py`  | Integer/Diophantine-flavored systems.                           |

---

## 🗂️ Project Structure

```text
pysne/                              <-- Root repository
│
├── pysne/                          <-- Main library package
│   ├── __init__.py
│   ├── solver.py                   # solve_system() — the 3-phase pipeline
│   ├── utils.py                    # objective_function, is_in_domain, validate_solutions
│   ├── version.py                  # __version__ = "0.2.0"
│   │
│   ├── initialization/
│   │   └── sampling.py             # generate_sobol_points (Sobol / SciPy QMC)
│   │
│   ├── clustering/
│   │   ├── model.py                # Cluster object
│   │   ├── clustering_process.py
│   │   └── modified_clustering_process.py   # perform_iterative_clustering
│   │
│   ├── optimizers/
│   │   └── spo/
│   │       ├── engine.py           # spiral_optimization (SPO core loop)
│   │       └── matrix.py           # get_rotation_matrix (n-D rotation)
│   │
│   └── problems/
│       ├── base.py                 # BaseProblem / MinimizedProblem / MultimodalProblem
│       ├── benchmarks_sne.py       # get_problem_set
│       ├── benchmarks_multimodal.py# get_multimodal_problems
│       └── benchmarks_diophantine.py# get_diophantine_problems
│
├── tests/                          # Unit & integration tests
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## ✅ Running the Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest -v
```

Smoke/integration tests (e.g. `tests/test_gecco.py`) run the full pipeline on a
fast benchmark to verify that clustering, SPO, and selection compose correctly.

---

## 📚 Documentation

Full documentation — installation, user guide, API reference, algorithms,
case studies, and an interactive 2D solution-landscape demo — lives at:

**👉 https://p2ms-optimization.github.io/pysne-web/**

| Repository                                                                 | Purpose                                          |
| -------------------------------------------------------------------------- | ------------------------------------------------ |
| [`p2ms-optimization/pysne`](https://github.com/p2ms-optimization/pysne)         | Python package, solver, algorithms, and tests.   |
| [`p2ms-optimization/pysne-web`](https://github.com/p2ms-optimization/pysne-web) | MkDocs documentation website and research pages.  |

---

## 🤝 Contributing

Contributions, bug reports, and benchmark additions are welcome.

1. Fork the repository and create a feature branch.
2. Add tests for new behavior under `tests/`.
3. Ensure `pytest` passes before opening a pull request.
4. Open issues at the [Bug Tracker](https://github.com/p2ms-optimization/pysne/issues).

---

## 📝 Citation

If you use PySNE in academic work, please cite it:

```bibtex
@software{pysne2026,
  title   = {PySNE: Finding All Solutions of Systems of Nonlinear Equations
             via Spiral Dynamics Optimization with Clustering},
  author  = {Hermawan, Aldy Nugraha and Isriyanto, Azarya Benhanan},
  year    = {2026},
  url     = {https://github.com/p2ms-optimization/pysne},
  note    = {Version 0.1.0}
}
```

---

## 👥 Authors

- **Aldy Nugraha Hermawan** — aldynugrahahermawan1702@gmail.com
- **Azarya Benhanan Isriyanto** — azaryaben@gmail.com

Undergraduate Thesis · Actuarial Science · Institut Teknologi Bandung

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
