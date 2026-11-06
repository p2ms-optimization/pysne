import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.optimize import root

from pysne.optimizers.base import BaseOptimizer

class ScipyRootOptimizer(BaseOptimizer):
    """
    Local root-finding optimizer using SciPy's root-finding algorithms
    (e.g., Powell hybrid 'hybr', Levenberg-Marquardt 'lm', Broyden, etc.).
    """

    def __init__(self, method: str = 'hybr'):
        """
        Parameters
        ----------
        method : str, optional
            The SciPy solver method ('hybr', 'lm', 'broyden1', 'broyden2', etc.).
            Default is 'hybr' (modified Powell hybrid method).
        """
        self.method = method

    def optimize(
        self,
        problem: Any,
        cluster: Any,
        cluster_domain: List[Tuple[float, float]],
        params: Dict[str, Any]
    ) -> np.ndarray:
        # Check if the problem has explicit equations
        if not hasattr(problem, 'equations') or not problem.equations:
            return cluster.center.copy()

        def residual_vector(x):
            return [eq(x) for eq in problem.equations]

        # Use cluster center as the initial guess x0
        x0 = cluster.center.copy()
        tol = params.get('epsilon', 1e-7)

        try:
            sol = root(residual_vector, x0=x0, method=self.method, tol=tol)
            if sol.success:
                return sol.x
            else:
                # Even if not completely converged according to SciPy's criteria,
                # check if sol.x yields a better fitness than x0
                cand = sol.x
                f_cand = problem.evaluate_fitness(cand)
                f_x0 = problem.evaluate_fitness(x0)
                return cand if f_cand >= f_x0 else x0
        except Exception:
            return x0
