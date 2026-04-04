# ini nantinya akan jadi mesin utama optimisasi dengan SDOA setelah semua fungsi yang berkaitan di push

import numpy as np
from .matrix import get_rotation_matrix
from utils import is_in_domain
from scipy.stats import qmc

def spiral_dynamics_optimization(objective_func, domain, params, minimization=False,
                                 custom_initial_points=None, equations=None, epsilon=None):
    """
    Implementasi SDOA dengan early stopping.
    """
    m = params.get('m', 20)
    r = params.get('r', 0.95)
    theta = params.get('theta', np.pi/4)
    k_max = params.get('k_max', 100)
    n = len(domain)

    if epsilon is None:
        epsilon = 1e-7
    
    # Initialize search points
    lower_bounds = np.array([d[0] for d in domain])
    upper_bounds = np.array([d[1] for d in domain])

    if custom_initial_points is not None:
        search_points = np.array(custom_initial_points)
        m = len(search_points)
    else:
        try:
            sampler = qmc.Sobol(d=n, scramble=False)
            unit_points = sampler.random(n=m)
            search_points = qmc.scale(unit_points, lower_bounds, upper_bounds)
        except:
            search_points = np.random.uniform(lower_bounds, upper_bounds, (m, n))

    # Precompute spiral transformation matrix
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    # Initialize best solution
    best_values = np.array([objective_func(point) for point in search_points])
    best_idx = np.argmin(best_values) if minimization else np.argmax(best_values)
    x_star = search_points[best_idx].copy()
    best_value = best_values[best_idx]

    # Calculate initial residual
    residual = 1.0 - best_value if equations is not None else float('inf')

    # Main optimization loop with early stopping
    for k in range(k_max):
        # Update all search points
        new_search_points = np.zeros_like(search_points)
        for i in range(m):
            term1 = S_n @ search_points[i]
            term2 = (S_n - I_n) @ x_star
            new_search_points[i] = term1 - term2
        search_points = new_search_points

        # Evaluate all points
        current_values = np.array([objective_func(point) for point in search_points])

        # Update best solution
        if minimization:
            current_best_idx = np.argmin(current_values)
            current_best_value = current_values[current_best_idx]
            x_star = search_points[current_best_idx].copy()
            best_value = current_best_value
        else:
            current_best_idx = np.argmax(current_values)
            current_best_value = current_values[current_best_idx]
            x_star = search_points[current_best_idx].copy()
            best_value = current_best_value

        # Calculate residual
        residual = 1.0 - best_value if equations is not None else float('inf')

        # Check stopping criteria
        if residual <= epsilon:
            break

    return x_star