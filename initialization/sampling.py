import numpy as np
from scipy.stats import qmc

def generate_sobol_points(num_points, dimension, domain):
    """Generate Sobol points scaled to domain."""
    lower_bounds = np.array([d[0] for d in domain])
    upper_bounds = np.array([d[1] for d in domain])
    try:
        sampler = qmc.Sobol(d=dimension, scramble=False)
        unit_points = sampler.random(n=num_points)
        points = qmc.scale(unit_points, lower_bounds, upper_bounds)
        return points
    except Exception:
        points = np.random.uniform(lower_bounds, upper_bounds, (num_points, dimension))
        return points