import numpy as np
from typing import Callable, Dict, Tuple, Any

def get_multimodal_functions_set() -> Dict[int, Callable]:
    """Returns a dictionary of multimodal problem caller functions."""
    return {
        1: problem_1
    }

def problem_1() -> Tuple[Callable, list, dict, int]:
    """Problem 1: 2D Second Minima Function"""
    def g_func(x):
        x = np.asarray(x)
        x0 = x[0] if x.ndim == 1 else x[:, 0]
        x1 = x[1] if x.ndim == 1 else x[:, 1]
        val_x = 0.5 * (x0**4 - 16*x0**2 + 5*x0)
        val_y = 0.5 * (x1**4 - 16*x1**2 + 5*x1)
        return val_x + val_y

    domain = [(-4, 4), (-4, 4)]
    params = {
        'm_cluster': 300,
        'r_cl': 0.95,
        'theta_cl': np.pi/4,
        'k_cluster': 10,
        'epsilon': 1e-7,
        'delta': 0.1,
        'm': 200,
        'k_max': 200,
        'r': 0.95,
        'theta': np.pi/4,
        'gamma': -float('inf')  # Gamma used for thresholding, typically not 0 for multimodal
    }
    expected_peaks = 4
    
    return g_func, domain, params, expected_peaks