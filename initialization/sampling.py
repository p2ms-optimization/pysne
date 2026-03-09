import numpy as np
from scipy.stats import qmc

def generate_sobol_points(num_points, dimension, domain):
    """
    Generates sample points using a Sobol sequence (low-discrepancy sequence) scaled to a specified domain.

    If the Sobol sequence generation fails, this function falls back to a pseudo-random uniform distribution.

    Args:
        num_points (int): The number of points to generate. For optimal spatial balance properties of the Sobol sequence, this value should ideally be a power of two (2^m).
        dimension (int): The dimensionality of the sample space.
        domain (list or tuple): The lower and upper bounds for each dimension. The expected format is a sequence of tuples, e.g., [(min_1, max_1), (min_2, max_2), ..., (min_d, max_d)].

    Returns:
        numpy.ndarray: An array of shape (num_points, dimension) containing the sample points scaled to the specified `domain`.

    Raises:
        IndexError: If the `domain` structure lacks the appropriate lower and upper bounds for each dimension.
    """
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
