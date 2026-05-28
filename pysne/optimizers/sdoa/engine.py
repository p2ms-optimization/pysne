# this will later become the main optimization engine with SDOA after all related functions are pushed

import numpy as np
from .matrix import get_rotation_matrix
from pysne.utils import is_in_domain
# from scipy.stats import qmc #
from pysne.initialization.sampling import generate_sobol_points

def spiral_dynamics_optimization(objective_func, domain, params, minimization=False,
                                 custom_initial_points=None, equations=None, epsilon=None, return_history=False):
    """
    Implementation of SDOA with early stopping.
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
            # Vectorized
            current_values = np.array(objective_func(search_points))
            
            # Validasi aman: jika shape kacau akibat persamaan SNE yang tidak mendukung 2D
            if current_values.shape != (m,):
                raise ValueError("Shape mismatch")
        except:
            # list comprehension
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

        # 6. Early Stopping (Hanya untuk SNE)
        if equations is not None:
            residual = 1.0 - best_value
            if residual <= epsilon:
                break

    if return_history:
        return x_star, history
    return x_star