import numpy as np
import warnings
from typing import List, Tuple, Callable

def objective_function(
    x: np.ndarray, 
    system_of_equations: List[Callable[[np.ndarray], float]]
) -> float:
    """
    Calculates the fitness value based on the total absolute residual of the system of equations.
    
    This function transforms the root-finding problem into a maximization problem using the formula: F(x) = 1 / (1 + sum |f_i(x)|).

    Parameters
    ----------
    x : numpy.ndarray
        The input variable vector or candidate solution.
    system_of_equations : list of callable
        A list of functions representing the system of equations f_i(x) = 0.

    Returns
    -------
    float
        The fitness value in the range (0, 1]. A value approaching 1.0 indicates a highly accurate root. Returns 0.0 if mathematical 
        evaluation fails (e.g., division by zero or overflow).
    """
    try:
        # Calculate the sum of absolute values for each equation in the system
        sum_of_abs_f = sum(abs(f_i(x)) for f_i in system_of_equations)
        
        return 1.0 / (1.0 + sum_of_abs_f)
        
    except (TypeError, ValueError, ZeroDivisionError) as e:
        # Catch specific mathematical or input errors and issue a safe warning
        warnings.warn(f"Mathematical evaluation failed in objective_function: {e}", RuntimeWarning)
        return 0.0

def is_in_domain(point: np.ndarray, domain: List[Tuple[float, float]]) -> bool:
    """
    Checks whether a given coordinate point lies strictly within the defined domain boundaries.

    Parameters
    ----------
    point : numpy.ndarray
        The coordinate point to be evaluated.
    domain : list of tuple
        The boundaries of the search space for each dimension in the format [(min_1, max_1), (min_2, max_2), ...].

    Returns
    -------
    bool
        True if the point is within the domain boundaries, False otherwise.
    """
    for i, (lo, hi) in enumerate(domain):
        if not (lo <= point[i] <= hi):
            return False
    return True

def validate_solutions(
    roots: List[np.ndarray], 
    equations: List[Callable], 
    domain: List[Tuple[float, float]], 
    epsilon: float
) -> List[np.ndarray]:
    """
    Validates a list of candidate roots by ensuring they strictly fall within the domain and their maximum absolute residual is below the 
    specified tolerance.

    Parameters
    ----------
    roots : list of numpy.ndarray
        The list of candidate roots found by the solver.
    equations : list of callable
        The system of nonlinear equations to verify against.
    domain : list of tuple
        The defined search space boundaries.
    epsilon : float
        The maximum acceptable residual for a point to be considered a valid root.

    Returns
    -------
    list of numpy.ndarray
        A filtered list containing only the coordinate points that meet both the domain and accuracy criteria.
    """
    valid_roots = []
    for root in roots:
        in_domain = is_in_domain(root, domain)
        residuals = [abs(f(root)) for f in equations]
        if max(residuals) < epsilon and in_domain:
            valid_roots.append(root)
    return valid_roots

def create_continuous_bounds(
    integer_domain: List[Tuple[int, int]],
    margin: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Expands each dimension of the integer domain by a margin on both sides.

    Purpose: allows spiral search points to move freely in a continuous space
    slightly wider than the integer grid, so that points at the edges of the
    integer domain can still be evaluated correctly.

    Example:
        integer_domain = [(-50, 50), (-50, 50)]
        create_continuous_bounds(integer_domain, margin=0.5)
        → [(-50.5, 50.5), (-50.5, 50.5)]

    Parameters
    ----------
    integer_domain : list of tuple
        Integer bounds per dimension, e.g. [(-50, 50), (-50, 50)].
    margin : float
        The expansion amount on each side. Default 0.5 (half the distance between integers).

    Returns
    -------
    list of tuple
        Expanded continuous bounds.
    """
    return [(lo - margin, hi + margin) for lo, hi in integer_domain]

def filter_unique_roots(candidates: List[Tuple[np.ndarray, float]], delta: float) -> np.ndarray:
    """
    Filters candidates such that only unique roots are kept.
    Each candidate in the input list is a tuple of (coordinate_point, fitness_value).
    If two points are closer than delta, only the one with the higher fitness value is retained.
    """
    if not candidates:
        return np.array([])
    
    # Sort in descending order based on fitness value
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    final_roots = []
    for cand, f_val in sorted_candidates:
        found_close = False
        for i, (existing, existing_f) in enumerate(final_roots):
            if np.linalg.norm(cand - existing) <= delta:
                found_close = True
                if f_val > existing_f:
                    final_roots[i] = (cand, f_val)
                break
        if not found_close:
            final_roots.append((cand, f_val))
            
    return np.array([root for root, _ in final_roots])

def calculate_sobol_discrepancy(
    num_points: int = None, 
    dimension: int = None, 
    points: np.ndarray = None, 
    domain: List[Tuple[float, float]] = None
) -> float:
    """
    Calculates the discrepancy of a point distribution.

    If `points` is provided, computes the discrepancy of those points (rescaled
    back to [0, 1]^d if `domain` is given). Otherwise, generates new Sobol
    points using scipy.stats.qmc for the specified dimension and number of points.
    
    Parameters
    ----------
    num_points : int, optional
        Number of sample points to generate (if points is not provided).
    dimension : int, optional
        Dimensionality of the search space (if points is not provided).
    points : numpy.ndarray, optional
        Sample points whose discrepancy will be calculated.
    domain : list of tuple, optional
        Search bounds for rescaling sample points back to [0, 1]^d.
        
    Returns
    -------
    float
        The discrepancy value (lower discrepancy indicates a more uniform distribution).
    """
    from scipy.stats import qmc
    
    if points is not None:
        try:
            pts = np.asarray(points)
            if domain is not None:
                lower_bounds = np.array([d[0] for d in domain])
                upper_bounds = np.array([d[1] for d in domain])
                denom = upper_bounds - lower_bounds
                denom[denom == 0] = 1.0
                pts = (pts - lower_bounds) / denom
            pts = np.clip(pts, 0.0, 1.0)
            discrepancy_val = qmc.discrepancy(pts)
            return float(discrepancy_val)
        except Exception as e:
            warnings.warn(f"Failed to compute discrepancy from points: {e}", RuntimeWarning)
            return -1.0

    if num_points is None or dimension is None:
        warnings.warn("Parameters num_points and dimension must be provided if points is not supplied.", RuntimeWarning)
        return -1.0

    # Generate points using the built-in scipy Sobol sampler
    sampler = qmc.Sobol(d=dimension, scramble=True)
    
    try:
        points_gen = sampler.random(n=num_points)
        discrepancy_val = qmc.discrepancy(points_gen)
        return float(discrepancy_val)
    except Exception as e:
        warnings.warn(f"Failed to compute discrepancy: {e}", RuntimeWarning)
        return -1.0
    
def sort_unique_roots(roots, sort=False):
    """
    Removes duplicate solutions based on value, with an option to ignore order.

    Parameters
    ----------
    roots : list of tuple
        Selected solutions (each as an integer tuple).
    sort : bool
        If True, each solution is sorted before comparison, so solutions
        that differ only in order are treated as identical.
        If False, only exactly identical solutions are removed.

    Returns
    -------
    list of tuple
        Unique solutions in tuple format.
    """
    seen = set()
    unique = []
    for root in roots:
        # Create key: sort if sort=True, otherwise use as-is
        key = tuple(sorted(root)) if sort else tuple(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique
