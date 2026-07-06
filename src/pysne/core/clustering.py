"""
pysne.core.clustering
---------------------
Cluster data structure and iterative-clustering logic for SPOC (real domain)
and SPOC_int (integer domain).  Both are kept in one module because they share
the Cluster class and the domain-validity helper; the objective function
evaluation differs and is passed in as a callable.
"""

import numpy as np
from .rotation import get_rotation_matrix
from .objective import (
    objective_function,
    objective_function_int,
    generate_korobov_points,
)


# ──────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

class Cluster:
    """A hypersphere defined by a center and a radius."""

    def __init__(self, center, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def __repr__(self):
        return f"Cluster(center={self.center.round(4)}, radius={self.radius:.4f})"


def is_in_domain(point, domain) -> bool:
    """Return True iff every coordinate of *point* lies within *domain*."""
    for i, (lo, hi) in enumerate(domain):
        if not (lo <= point[i] <= hi):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  REAL-DOMAIN CLUSTERING  (used by SPOC)
# ──────────────────────────────────────────────────────────────────────────────

def _process_point_real(y, clusters, equations, gamma, domain, num_check_points=1):
    """Add / update clusters based on point *y* (real domain)."""
    F_y = objective_function(y, equations)
    if F_y <= gamma:
        return clusters

    if not clusters:
        r0 = 0.1 * min(hi - lo for lo, hi in domain)
        clusters.append(Cluster(y, r0))
        return clusters

    # Find nearest cluster
    nearest = min(clusters, key=lambda c: np.linalg.norm(y - c.center))
    x_C  = nearest.center
    F_xC = objective_function(x_C, equations)

    # Support multiple check points (num_check_points=1 reproduces original midpoint logic)
    t_vals = [i / (num_check_points + 1) for i in range(1, num_check_points + 1)]
    x_ts   = [y + t * (x_C - y) for t in t_vals]
    F_xts  = [objective_function(xt, equations) for xt in x_ts]
    F_min  = min(F_xts)
    F_max  = max(F_xts)
    dist_h = np.linalg.norm(y - x_C) / 2.0

    if F_min < F_y and F_min < F_xC:
        clusters.append(Cluster(y, dist_h))
    elif F_max > F_y and F_max > F_xC:
        clusters.append(Cluster(y, dist_h))
        best_xt = x_ts[F_xts.index(F_max)]
        clusters = _process_point_real(best_xt, clusters, equations, gamma, domain, num_check_points)
    elif F_y > F_xC:
        nearest.center = y.copy()
        nearest.radius = dist_h
    else:
        nearest.radius = dist_h

    return clusters


def perform_iterative_clustering_real(equations, domain, params):
    """
    Phase 1 clustering for the real-domain solver (SPOC).

    Parameters
    ----------
    equations : list of callables
    domain    : list of (lo, hi) float pairs
    params    : dict with keys m_cluster, gamma, k_cluster, r, theta

    Returns
    -------
    list of Cluster
    """
    m     = params['m_cluster']
    gamma = params['gamma']
    k     = params['k_cluster']
    r     = params['r']
    theta = params['theta']
    n     = len(domain)

    num_check_pts = params.get('num_check_points', 1)

    points = generate_korobov_points(m, n, domain)
    R_n    = get_rotation_matrix(n, theta)
    S_n    = r * R_n
    I_n    = np.identity(n)

    F_vals   = np.array([objective_function(p, equations) for p in points])
    x_prime  = points[np.argmax(F_vals)].copy()
    r0       = 0.5 * min(hi - lo for lo, hi in domain)
    clusters = [Cluster(x_prime, r0)]

    for _ in range(k):
        for i in range(m):
            if objective_function(points[i], equations) > gamma:
                if not any(np.allclose(points[i], c.center, atol=1e-8) for c in clusters):
                    clusters = _process_point_real(
                        points[i], clusters, equations, gamma, domain, num_check_pts
                    )

        F_vals  = np.array([objective_function(p, equations) for p in points])
        x_p     = points[np.argmax(F_vals)].copy()
        # x_k+1 = S @ x_k - (S - I) @ x*   (applied row-wise)
        offset  = (S_n - I_n) @ x_p          # shape (n,)
        points  = (S_n @ points.T).T - offset  # shape (m, n)

    return clusters


# ──────────────────────────────────────────────────────────────────────────────
#  INTEGER-DOMAIN CLUSTERING  (used by SPOC_int)
# ──────────────────────────────────────────────────────────────────────────────

def create_continuous_bounds(integer_domain, margin: float = 0.5):
    """Extend each integer dimension by *margin* on each side."""
    return [(lo - margin, hi + margin) for lo, hi in integer_domain]


def _process_point_int(y, clusters, equations, integer_domain,
                        gamma, continuous_bounds, F_y=None, num_check_points=1):
    """Add / update clusters based on point *y* (integer domain)."""
    if F_y is None:
        F_y = objective_function_int(y, equations, integer_domain)
    if F_y <= gamma:
        return clusters

    nearest = min(clusters, key=lambda c: np.linalg.norm(y - c.center))
    x_C  = nearest.center
    F_xC = objective_function_int(x_C, equations, integer_domain)

    t_vals  = [i / (num_check_points + 1) for i in range(1, num_check_points + 1)]
    x_ts    = [y + t * (x_C - y) for t in t_vals]
    F_xts   = [objective_function_int(xt, equations, integer_domain) for xt in x_ts]
    F_min   = min(F_xts)
    F_max   = max(F_xts)
    dist_h  = np.linalg.norm(y - x_C) / 2.0

    if F_min < F_y and F_min < F_xC:
        clusters.append(Cluster(y, dist_h))
    elif F_max > F_y and F_max > F_xC:
        clusters.append(Cluster(y, dist_h))
        best_xt = x_ts[F_xts.index(F_max)]
        clusters = _process_point_int(
            best_xt, clusters, equations, integer_domain,
            gamma, continuous_bounds, F_y=F_max,
            num_check_points=num_check_points,
        )
    elif F_y > F_xC:
        nearest.center = y.copy()

    nearest.radius = dist_h
    return clusters


def perform_iterative_clustering_int(equations, integer_domain, continuous_bounds, params):
    """
    Phase 1 clustering for the integer-domain solver (SPOC_int).

    Parameters
    ----------
    equations        : list of callables
    integer_domain   : list of (lo, hi) int pairs
    continuous_bounds: list of (lo, hi) float pairs (with margin)
    params           : dict with keys m_cluster, gamma, k_cluster, r, theta,
                       num_check_points (optional, default 1)

    Returns
    -------
    list of Cluster
    """
    m              = params['m_cluster']
    gamma          = params['gamma']
    k              = params['k_cluster']
    r              = params['r']
    theta          = params['theta']
    num_check_pts  = params.get('num_check_points', 1)
    n              = len(integer_domain)

    points = generate_korobov_points(m, n, continuous_bounds)
    R_n    = get_rotation_matrix(n, theta)
    S_n    = r * R_n
    diff_S = S_n - np.identity(n)

    obj = lambda x: objective_function_int(x, equations, integer_domain)

    # Initialise first cluster from the best rounded point
    rounded_init = np.round(points)
    F_init       = np.array([obj(p) for p in rounded_init])
    x_star       = rounded_init[np.argmax(F_init)].copy()
    r0           = 0.5 * float(min(hi - lo for lo, hi in continuous_bounds))
    clusters     = [Cluster(x_star, r0)]

    for _ in range(k):
        rounded = np.round(points)
        F_cache = {}
        center_set = {
            tuple(np.round(c.center).astype(int).tolist()) for c in clusters
        }

        for i in range(m):
            if is_in_domain(rounded[i], integer_domain):
                F_cache[i] = obj(points[i])

        for i, F_val in F_cache.items():
            if F_val > gamma:
                key = tuple(rounded[i].astype(int).tolist())
                if key not in center_set:
                    clusters = _process_point_int(
                        points[i], clusters, equations, integer_domain,
                        gamma, continuous_bounds,
                        F_y=F_val, num_check_points=num_check_pts,
                    )
                    center_set.add(key)

        if F_cache:
            x_star = rounded[max(F_cache, key=F_cache.get)].copy()

        points = points @ S_n.T - diff_S @ x_star

    # Round all cluster centers to integers at end of Phase 1
    for c in clusters:
        c.center = np.round(c.center)

    return clusters
