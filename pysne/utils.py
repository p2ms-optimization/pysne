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

def create_continuous_bounds(integer_domain: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """
    Converts an integer domain to continuous bounds by adding a half-unit pad on both ends.
    For example, [(1, 5)] becomes [(0.5, 5.5)].
    """
    return [(float(lo) - 0.5, float(hi) + 0.5) for lo, hi in integer_domain]

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

