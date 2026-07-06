"""
pysne.SPOC_int
--------------
Spiral-dynamics-based system-of-equation solver for **integer** domains
(Diophantine equations and integer nonlinear systems).

Public API
~~~~~~~~~~
    from pysne import SPOC_int

    solutions = SPOC_int.solve(equations, domain, param=None)

Parameters reference
~~~~~~~~~~~~~~~~~~~~
    m_cluster       : int   — initial search points for clustering phase    (default 300)
    k_cluster       : int   — clustering iterations                         (default 50)
    gamma           : float — cut-off threshold for F(x)                    (default 1e-4)
    epsilon         : float — root-acceptance threshold  1-F(x) < epsilon   (default 1e-5)
    delta           : float — distance threshold for duplicate removal       (default 0.01)
    r               : float — spiral contraction rate  ∈ (0,1)              (default 0.95)
    theta           : float — rotation angle (radians)                      (default π/4)
    m_sdoa          : int   — SDOA search points per cluster                (default 50)
    k_max           : int   — SDOA max iterations                           (default 50)
    r_sdoa          : float — SDOA contraction rate                         (default 0.988)
    theta_sdoa      : float — SDOA rotation angle                           (default π/4)
    num_check_points: int   — midpoints checked in clustering logic         (default 1)
"""

import numpy as np
import time
from typing import List, Callable, Optional

from .core.rotation import get_rotation_matrix
from .core.objective import (
    objective_function_int,
    generate_korobov_points,
)
from .core.clustering import (
    perform_iterative_clustering_int,
    create_continuous_bounds,
    is_in_domain,
)


# ──────────────────────────────────────────────────────────────────────────────
#  DEFAULT PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS = {
    'm_cluster'       : 300,
    'k_cluster'       : 50,
    'gamma'           : 1e-4,
    'epsilon'         : 1e-5,
    'delta'           : 0.01,
    'r'               : 0.95,
    'theta'           : np.pi / 4,
    'm_sdoa'          : 50,
    'k_max'           : 50,
    'r_sdoa'          : 0.988,
    'theta_sdoa'      : np.pi / 4,
    'num_check_points': 1,
}


def _merge_params(user_params: Optional[dict]) -> dict:
    p = dict(_DEFAULT_PARAMS)
    if user_params:
        p.update(user_params)
    return p


# ──────────────────────────────────────────────────────────────────────────────
#  SDOA (Phase 2) — integer-aware variant with global-best memory
# ──────────────────────────────────────────────────────────────────────────────

def _sdoa_int(obj_func, continuous_bounds, sdoa_params,
              custom_initial_points=None, epsilon=1e-5):
    """
    SDOA for integer domain.  Operates in continuous space but evaluates
    the objective at rounded points; keeps a global-best memory to prevent
    algorithm amnesia.
    """
    m     = sdoa_params['m']
    r     = sdoa_params['r']
    theta = sdoa_params['theta']
    k_max = sdoa_params['k_max']
    n     = len(continuous_bounds)

    if custom_initial_points is not None:
        points = np.array(custom_initial_points, dtype=float)
        m      = len(points)
    else:
        points = generate_korobov_points(m, n, continuous_bounds)

    R_n    = get_rotation_matrix(n, theta)
    S_n    = r * R_n
    diff_S = S_n - np.identity(n)

    global_best_val = -1.0
    global_best_x   = None

    for _ in range(k_max):
        rounded = np.round(points)
        vals    = np.array([obj_func(sp) for sp in points])
        best_i  = np.argmax(vals)

        if vals[best_i] > global_best_val:
            global_best_val = vals[best_i]
            global_best_x   = rounded[best_i].copy()

        if global_best_x is None:
            continue

        if 1.0 - global_best_val <= epsilon:
            return global_best_x

        points = points @ S_n.T - diff_S @ global_best_x

    return global_best_x if global_best_x is not None else np.round(points[0])


def _run_sdoa_on_clusters(clusters, equations, integer_domain,
                           continuous_bounds, sdoa_params, epsilon):
    candidates = []
    obj        = lambda x: objective_function_int(x, equations, integer_domain)

    for cluster in clusters:
        effective_r = max(cluster.radius, 1.0)
        sub = []
        for dim, (lo, hi) in enumerate(integer_domain):
            c_lo = max(continuous_bounds[dim][0], cluster.center[dim] - effective_r)
            c_hi = min(continuous_bounds[dim][1], cluster.center[dim] + effective_r)
            if c_lo >= c_hi:
                mid  = (continuous_bounds[dim][0] + continuous_bounds[dim][1]) / 2.0
                half = max(0.5, (continuous_bounds[dim][1] - continuous_bounds[dim][0]) / 2.0)
                c_lo = max(continuous_bounds[dim][0], mid - half)
                c_hi = min(continuous_bounds[dim][1], mid + half)
            sub.append((c_lo, c_hi))

        init_pts  = generate_korobov_points(sdoa_params['m'], len(integer_domain), sub)
        candidate = _sdoa_int(obj, continuous_bounds, sdoa_params,
                              custom_initial_points=init_pts, epsilon=epsilon)
        candidates.append(candidate)

    return np.array(candidates) if candidates else np.empty((0, len(integer_domain)))


# ──────────────────────────────────────────────────────────────────────────────
#  SELECTION (Phase 3)
# ──────────────────────────────────────────────────────────────────────────────

def _select_roots(candidates, equations, integer_domain, epsilon, delta):
    ok = []
    for cand in candidates:
        cand_int = np.round(cand).astype(int)
        if not is_in_domain(cand_int, integer_domain):
            continue
        F = objective_function_int(cand, equations, integer_domain)
        if 1.0 - F <= epsilon:
            ok.append((cand_int, F))
    if not ok:
        return []

    ok.sort(key=lambda t: t[1], reverse=True)
    final = []
    for cand, F in ok:
        if not any(np.linalg.norm(cand - e) <= delta for e, _ in final):
            final.append((cand, F))

    return [tuple(r.tolist()) for r, _ in final]


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def solve(
    equations: List[Callable],
    domain,
    param: Optional[dict] = None,
    verbose: bool = True,
    sort_solutions: bool = False,
) -> list:
    """
    Find all integer-valued roots of a system of equations over *domain*.

    Parameters
    ----------
    equations      : list of callables  — each f_i accepts a numpy array x
    domain         : list of (lo, hi)   — integer search bounds per dimension
    param          : dict, optional     — override any default parameter (see module docstring)
    verbose        : bool               — print progress (default True)
    sort_solutions : bool               — sort & deduplicate by unordered tuple (default False)

    Returns
    -------
    list of tuple  — each element is an integer root expressed as a tuple
    """
    p        = _merge_params(param)
    epsilon  = p['epsilon']
    cont_bnd = create_continuous_bounds(domain, margin=0.5)

    if verbose:
        print("=" * 70)
        print("SPOC_int  —  Spiral Optimisation with Clustering (Integer Domain)")
        print(f"Integer domain    : {domain}")
        print(f"Continuous bounds : {cont_bnd}  (margin = 0.5)")
        print(f"Dimension : {len(domain)}D   |   ε = {epsilon}")
        print("=" * 70)

    t0 = time.time()

    # Phase 1: Clustering
    if verbose:
        print("Phase 1 : Clustering ...", end=" ", flush=True)
    clusters = perform_iterative_clustering_int(equations, domain, cont_bnd, p)
    if verbose:
        print(f"{len(clusters)} clusters found")

    # Phase 2: SDOA per cluster
    if verbose:
        print("Phase 2 : SDOA optimisation ...", end=" ", flush=True)
    sdoa_params = {
        'm'    : p['m_sdoa'],
        'r'    : p['r_sdoa'],
        'theta': p['theta_sdoa'],
        'k_max': p['k_max'],
    }
    candidates = _run_sdoa_on_clusters(clusters, equations, domain, cont_bnd, sdoa_params, epsilon)
    if verbose:
        print(f"{len(candidates)} candidates")

    # Phase 3: Selection
    if verbose:
        print("Phase 3 : Final selection ...")
    solutions = _select_roots(candidates, equations, domain, epsilon, p['delta'])

    if sort_solutions:
        seen, unique = set(), []
        for sol in solutions:
            key = tuple(sorted(sol))
            if key not in seen:
                seen.add(key)
                unique.append(tuple(sol))
        solutions = sorted(unique)

    elapsed = time.time() - t0

    if verbose:
        print()
        print(f"Time elapsed  : {elapsed:.3f} s")
        print(f"Solutions found : {len(solutions)}")
        for i, sol in enumerate(solutions):
            q   = np.array(sol)
            F   = objective_function_int(q, equations, domain)
            res = 1.0 - F
            print(f"  Solution {i+1}: {sol}  |  F(x) = {F:.6f}  |  residual = {res:.2e}")
        print("=" * 70)

    return solutions
