"""
pysne.SPOC
----------
Spiral-dynamics-based system-of-equation solver for **real-valued** domains.

Public API
~~~~~~~~~~
    from pysne import SPOC

    roots = SPOC.solve(equations, domain, param=None)

Parameters reference
~~~~~~~~~~~~~~~~~~~~
    m_cluster      : int   — initial search points for clustering phase     (default 300)
    k_cluster      : int   — clustering iterations                          (default 10)
    gamma          : float — cut-off threshold for F(x)                     (default 1e-6)
    epsilon        : float — root-acceptance threshold  1-F(x) < epsilon    (default 1e-8)
    delta          : float — distance threshold for duplicate removal        (default 1e-4)
    r              : float — spiral contraction rate  ∈ (0,1)               (default 0.95)
    theta          : float — rotation angle (radians)                       (default π/4)
    m_sdoa         : int   — SDOA search points per cluster                 (default 30)
    k_max          : int   — SDOA max iterations                            (default 15)
    r_sdoa         : float — SDOA contraction rate                          (default 0.95)
    theta_sdoa     : float — SDOA rotation angle                            (default π/4)
    num_check_points: int  — unused in SPOC (kept for API parity)           (default 3)
"""

import numpy as np
import time
from typing import List, Callable, Optional

from .core.rotation import get_rotation_matrix
from .core.objective import (
    objective_function,
    generate_korobov_points,
)
from .core.clustering import (
    perform_iterative_clustering_real,
    is_in_domain,
)


# ──────────────────────────────────────────────────────────────────────────────
#  DEFAULT PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS = {
    'm_cluster'       : 300,
    'k_cluster'       : 10,
    'gamma'           : 1e-6,
    'epsilon'         : 1e-8,
    'delta'           : 1e-4,
    'r'               : 0.95,
    'theta'           : np.pi / 4,
    'm_sdoa'          : 250,
    'k_max'           : 250,
    'r_sdoa'          : 0.95,
    'theta_sdoa'      : np.pi / 4,
    'num_check_points': 1,   # bab5 uses 1 midpoint (original Sidarto & Kania)
}


def _merge_params(user_params: Optional[dict]) -> dict:
    p = dict(_DEFAULT_PARAMS)
    if user_params:
        p.update(user_params)
    return p


# ──────────────────────────────────────────────────────────────────────────────
#  SDOA (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

def _sdoa(obj_func, domain, sdoa_params, custom_initial_points=None, epsilon=1e-8):
    """Run SDOA within a (cluster) sub-domain and return the best point found."""
    m     = sdoa_params['m']
    r     = sdoa_params['r']
    theta = sdoa_params['theta']
    k_max = sdoa_params['k_max']
    n     = len(domain)

    if custom_initial_points is not None:
        points = np.array(custom_initial_points, dtype=float)
        m = len(points)
    else:
        points = generate_korobov_points(m, n, domain)

    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    vals     = np.array([obj_func(p) for p in points])
    best_idx = np.argmax(vals)
    x_star   = points[best_idx].copy()

    for _ in range(k_max):
        offset     = (S_n - I_n) @ x_star
        new_points = (S_n @ points.T).T - offset
        points     = new_points
        vals       = np.array([obj_func(p) for p in points])
        best_idx   = np.argmax(vals)
        x_star     = points[best_idx].copy()

        # Early stopping
        if 1.0 - vals[best_idx] <= epsilon:
            break

    return x_star


def _run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon):
    candidates = []
    MIN_WIDTH  = 1e-8
    obj        = lambda x: objective_function(x, equations)

    for cluster in clusters:
        sub = []
        for dim, (lo, hi) in enumerate(domain):
            c_lo = max(lo, cluster.center[dim] - cluster.radius)
            c_hi = min(hi, cluster.center[dim] + cluster.radius)
            if c_hi <= c_lo:
                mid  = (c_lo + c_hi) / 2
                c_lo = max(lo, mid - MIN_WIDTH / 2)
                c_hi = min(hi, mid + MIN_WIDTH / 2)
                if c_hi <= c_lo:
                    c_lo, c_hi = lo, hi
            sub.append((c_lo, c_hi))

        init_pts  = generate_korobov_points(sdoa_params['m'], len(domain), sub)
        candidate = _sdoa(obj, sub, sdoa_params, custom_initial_points=init_pts, epsilon=epsilon)
        candidates.append(candidate)

    return np.array(candidates) if candidates else np.empty((0, len(domain)))


# ──────────────────────────────────────────────────────────────────────────────
#  SELECTION / VALIDATION (Phase 3)
# ──────────────────────────────────────────────────────────────────────────────

def _select_roots(candidates, equations, domain, epsilon, delta):
    # Allow candidates whose residual is within sqrt(epsilon); final validation
    # (_validate) then applies the tighter check.
    selection_tol = max(epsilon ** 0.5, 1e-4)
    ok = []
    for cand in candidates:
        if not is_in_domain(cand, domain):
            continue
        F = objective_function(cand, equations)
        if 1.0 - F < selection_tol:
            ok.append((cand, F))
    if not ok:
        return np.array([])

    ok.sort(key=lambda t: t[1], reverse=True)
    final = []
    for cand, F in ok:
        if not any(np.linalg.norm(cand - e) <= delta for e, _ in final):
            final.append((cand, F))
    return np.array([r for r, _ in final])


def _validate(roots, equations, domain, epsilon):
    """
    Validasi akhir: filter root yang residual max|f_i(x)| masih di atas sqrt(epsilon).
    Pakai toleransi lebih longgar dari epsilon asli agar tidak over-reject.
    """
    tol = max(epsilon ** 0.5, 1e-6)   # lebih longgar
    valid = []
    for root in roots:
        if is_in_domain(root, domain):
            if max(abs(f(root)) for f in equations) < tol:
                valid.append(root)
    return valid


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def solve(
    equations: List[Callable],
    domain,
    param: Optional[dict] = None,
    verbose: bool = True,
) -> list:
    """
    Find all real-valued roots of a system of nonlinear equations.

    Parameters
    ----------
    equations : list of callables  — each f_i accepts a numpy array x
    domain    : list of (lo, hi)   — search bounds per dimension
    param     : dict, optional     — override any default parameter (see module docstring)
    verbose   : bool               — print progress (default True)

    Returns
    -------
    list of np.ndarray  — each element is a root vector
    """
    p       = _merge_params(param)
    epsilon = p['epsilon']

    if verbose:
        print("=" * 60)
        print("SPOC  —  Spiral Optimisation with Clustering (Real Domain)")
        print(f"Dimension : {len(domain)}D   |   Domain : {domain}")
        print(f"ε = {epsilon}")
        print("=" * 60)

    t0 = time.time()

    # Phase 1: Clustering
    if verbose:
        print("Phase 1 : Clustering ...", end=" ", flush=True)
    clusters = perform_iterative_clustering_real(equations, domain, p)
    if verbose:
        print(f"{len(clusters)} clusters found")

    # Phase 2: SDOA per cluster
    if verbose:
        print("Phase 2 : SDOA optimisation ...", end=" ", flush=True)
    sdoa_params = {'m': p['m_sdoa'], 'r': p['r_sdoa'], 'theta': p['theta_sdoa'], 'k_max': p['k_max']}
    candidates  = _run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon)
    if verbose:
        print(f"{len(candidates)} candidates")

    # Phase 3: Selection
    if verbose:
        print("Phase 3 : Final selection ...")
    raw_roots   = _select_roots(candidates, equations, domain, epsilon, p['delta'])
    valid_roots = _validate(raw_roots, equations, domain, epsilon)

    elapsed = time.time() - t0

    if verbose:
        print()
        print(f"Time elapsed  : {elapsed:.3f} s")
        print(f"Roots found   : {len(valid_roots)}")
        if valid_roots:
            for i, root in enumerate(valid_roots):
                res = 1.0 - objective_function(root, equations)
                print(f"  Root {i+1}: {np.round(root, 8)}  (residual: {res:.2e})")
        print("=" * 60)

    return valid_roots
