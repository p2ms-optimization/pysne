"""
pysne.core.objective
--------------------
Objective function wrapper F(x) = 1 / (1 + Σ|f_i(x)|) and
Korobov / Sobol point generators used by both SPOC and SPOC_int.
"""

import numpy as np
from math import gcd
from scipy.stats import qmc

# pydoe is a required dependency
from pydoe.space_filling.quasi_random.korobov import korobov_sequence


# ──────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def objective_function(x, system_of_equations) -> float:
    """
    F(x) = 1 / (1 + Σ |f_i(x)|).

    Parameters
    ----------
    x                  : array-like — evaluation point (real-valued)
    system_of_equations: list of callables

    Returns
    -------
    float in (0, 1], where 1 means exact root.
    """
    try:
        total = sum(abs(f_i(x)) for f_i in system_of_equations)
        return 1.0 / (1.0 + float(total))
    except (OverflowError, ZeroDivisionError, Exception):
        return 0.0


def objective_function_int(x, system_of_equations, integer_domain) -> float:
    """
    Integer-domain variant: evaluates F at round(x).
    Returns 0.0 if rounded point is outside integer_domain.

    Parameters
    ----------
    x               : array-like — continuous search point
    system_of_equations: list of callables
    integer_domain  : list of (lo, hi) int pairs
    """
    q = np.round(x).astype(object)
    for i, (lo, hi) in enumerate(integer_domain):
        if not (lo <= q[i] <= hi):
            return 0.0
    try:
        total = sum(abs(f_i(q)) for f_i in system_of_equations)
        return 1.0 / (1.0 + float(total))
    except (OverflowError, ZeroDivisionError, Exception):
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  POINT GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def _auto_korobov_param(m: int) -> int:
    phi = (5 ** 0.5 - 1) / 2
    a_ideal = round(m * phi)
    for delta in range(0, m // 2):
        for sign in [0, 1, -1]:
            a = a_ideal + sign * delta
            if 1 < a < m and gcd(a, m) == 1:
                return a
    return 2  # fallback


def generate_korobov_points(num_points: int, dimension: int, domain) -> np.ndarray:
    """
    Generate *num_points* Korobov quasi-random points scaled to *domain*.

    Parameters
    ----------
    num_points : int
    dimension  : int
    domain     : list of (lo, hi) pairs

    Returns
    -------
    np.ndarray, shape (num_points, dimension)
    """
    lower = np.array([d[0] for d in domain], dtype=float)
    upper = np.array([d[1] for d in domain], dtype=float)
    a = _auto_korobov_param(num_points)
    unit_pts = korobov_sequence(
        num_points=num_points,
        dimension=dimension,
        generator_param=a,
    ) / float(num_points)
    return qmc.scale(unit_pts, lower, upper)


def generate_sobol_points(num_points: int, dimension: int, domain) -> np.ndarray:
    """
    Generate *num_points* Sobol quasi-random points scaled to *domain*.
    Falls back to uniform random if Sobol fails.
    """
    lower = np.array([d[0] for d in domain], dtype=float)
    upper = np.array([d[1] for d in domain], dtype=float)
    try:
        sampler = qmc.Sobol(d=dimension, scramble=False)
        unit_pts = sampler.random(n=num_points)
        return qmc.scale(unit_pts, lower, upper)
    except Exception:
        return np.random.uniform(lower, upper, (num_points, dimension))
