import numpy as np
from .matrix import get_rotation_matrix
from pysne.utils import is_in_domain
from pysne.initialization.sampling import generate_sobol_points

def spiral_optimization(objective_func, domain, params, minimization=False,
                                 custom_initial_points=None, equations=None, epsilon=None, return_history=False):
    """
    Performs Spiral Optimization (SPO) with early stopping.

    This function optimizes an objective function over a bounded domain using
    the spiral dynamics operator. Points are iteratively rotated and contracted
    toward the current global best, converging on optimal solutions.

    Parameters
    ----------
    objective_func : callable
        The objective function to optimize. Accepts a numpy.ndarray and returns a float.
    domain : list of tuple
        The search space boundaries in the format [(min, max), ...].
    params : dict
        SPO hyperparameters with keys 'm' (population size), 'r' (contraction rate),
        'theta' (rotation angle), and 'k_max' (maximum iterations).
    minimization : bool, optional
        If True, minimizes the objective function. Default is False (maximization).
    custom_initial_points : numpy.ndarray, optional
        Pre-generated initial points. If None, Sobol points are generated.
    equations : list of callable, optional
        System of equations for early stopping in SNE problems. If provided,
        the algorithm stops when the residual falls below epsilon.
    epsilon : float, optional
        Tolerance for early stopping. Default is 1e-7.
    return_history : bool, optional
        If True, returns a tuple of (best_point, convergence_history). Default is False.

    Returns
    -------
    numpy.ndarray or tuple
        The best solution found. If return_history is True, returns
        (best_point, list_of_best_values_per_iteration).
    """
    # Parameter extraction
    m = params.get('m', 20)
    r = params.get('r', 0.95)
    theta = params.get('theta', np.pi/4)
    k_max = params.get('k_max', 100)
    n = len(domain)

    if epsilon is None:
        epsilon = 1e-7
    
    # Points Initialization (Using Sobol from sampling.py)
    if custom_initial_points is not None:
        search_points = np.array(custom_initial_points)
        m = len(search_points)
    else:
        search_points = generate_sobol_points(m, n, domain)

    # Precompute spiral transformation matrix
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    # Initialize best solution
    try:
        best_values = np.array(objective_func(search_points))
        if best_values.shape != (m,):
            raise ValueError("Shape mismatch")
    except:
        best_values = np.array([objective_func(p) for p in search_points])
    best_idx = np.argmin(best_values) if minimization else np.argmax(best_values)
    x_star = search_points[best_idx].copy()
    best_value = best_values[best_idx]

    history = [best_value]

    # Main optimization loop with early stopping
    for k in range(k_max):
        # Update all search points (Vectorized version)
        term1 = search_points @ S_n.T  # (m, n) @ (n, n) = (m, n)
        term2 = (S_n - I_n) @ x_star   # (n,)
        search_points = term1 - term2

        # Evaluate all points
        try:
            current_values = np.array(objective_func(search_points))
            # Safety check: ensure shape is valid for SNE equations that don't support 2D input
            if current_values.shape != (m,):
                raise ValueError("Shape mismatch")
        except:
            current_values = np.array([objective_func(point) for point in search_points])

        # Update Global Best
        current_best_idx = np.argmin(current_values) if minimization else np.argmax(current_values)
        current_best_value = current_values[current_best_idx]

        # Compare with the x_star
        if (minimization and current_best_value < best_value) or \
           (not minimization and current_best_value > best_value):
            x_star = search_points[current_best_idx].copy()
            best_value = current_best_value

        history.append(best_value)

        # Early Stopping (only for SNE problems with known equations)
        if equations is not None:
            residual = 1.0 - best_value
            if residual <= epsilon:
                break

    if return_history:
        return x_star, history
    return x_star