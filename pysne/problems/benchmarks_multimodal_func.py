import numpy as np
from typing import Callable, Dict, Tuple, Any

def get_multimodal_functions_set() -> Dict[int, Callable]:
    """Returns a dictionary of multimodal problem caller functions."""
    return {
        1: problem_1,
        2: problem_2,
        3: problem_3_2d,
        4: problem_3_3d
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

def problem_2() -> Tuple[Callable, list, dict, int]:
    """Problem 2: Six Hump Camel Back Function"""
    def g_func(x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return term1 + term2 + term3

    domain = [(-1.9, 1.9), (-1.1, 1.1)]
    params = {
        'm_cluster': 1000,
        'r_cl': 0.99,
        'theta_cl': np.pi/2,
        'k_cluster': 20,
        'epsilon': 1e-5,
        'delta': 0.1,
        'm': 200,
        'k_max': 200,
        'r': 0.95,
        'theta': np.pi/4,
        'gamma': -float('inf')
    }
    expected_peaks = 6
    
    return g_func, domain, params, expected_peaks

def problem_3_2d() -> Tuple[Callable, list, dict, int]:
    """Problem 3: 2D Rastrigin Function"""
    def g_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
        else:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)

    domain = [(-1, 1)] * 2
    params = {
        'm_cluster': 500,
        'r_cl': 0.95,
        'theta_cl': np.pi/4,
        'k_cluster': 10,
        'epsilon': 1e-6,
        'delta': 0.1,
        'm': 200,
        'k_max': 200,
        'r': 0.95,
        'theta': np.pi/4,
        'gamma': -float('inf')
    }
    expected_peaks = 4 # Adjust based on domain mapping
    
    return g_func, domain, params, expected_peaks

def problem_3_3d() -> Tuple[Callable, list, dict, int]:
    """Problem 3: 3D Rastrigin Function"""
    def g_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
        else:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)

    domain = [(-1, 1)] * 3
    params = {
        'm_cluster': 20000,
        'r_cl': 0.95,
        'theta_cl': np.pi/4,
        'k_cluster': 20,
        'epsilon': 1e-5,
        'delta': 0.1,
        'm': 200,
        'k_max': 200,
        'r': 0.95,
        'theta': np.pi/4,
        'gamma': -float('inf')
    }
    expected_peaks = 27 # Adjust based on domain mapping
    
    return g_func, domain, params, expected_peaks
