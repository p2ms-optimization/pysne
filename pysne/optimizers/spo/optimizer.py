import numpy as np
from typing import List, Tuple, Dict, Any

from pysne.optimizers.base import BaseOptimizer
from pysne.optimizers.spo.engine import spiral_optimization
from pysne.initialization.sampling import generate_sobol_points

class SPOOptimizer(BaseOptimizer):
    """
    Spiral Optimization (SPO) adapter for local search within a cluster.
    """

    def optimize(
        self,
        problem: Any,
        cluster: Any,
        cluster_domain: List[Tuple[float, float]],
        params: Dict[str, Any]
    ) -> np.ndarray:
        # Extract SPO-specific parameters with fallback aliases
        spo_params = {
            'm': params.get('spo_m', params.get('sdoa_m', params.get('m', 20))),
            'r': params.get('spo_r', params.get('sdoa_r', params.get('r', 0.95))),
            'theta': params.get('spo_theta', params.get('sdoa_theta', params.get('theta', np.pi / 4))),
            'k_max': params.get('spo_k_max', params.get('sdoa_k_max', params.get('k_max', 100)))
        }

        epsilon = params.get('epsilon', 1e-7)
        is_sne = hasattr(problem, 'equations')

        # Generate sample points using Sobol sequence inside the cluster domain
        initial_points = generate_sobol_points(spo_params['m'], len(cluster_domain), cluster_domain)

        # Run Spiral Optimization in cluster domain
        candidate = spiral_optimization(
            objective_func=problem.evaluate_fitness,
            domain=cluster_domain,
            params=spo_params,
            minimization=False,
            custom_initial_points=initial_points,
            equations=problem.equations if is_sne else None,
            epsilon=epsilon
        )

        return candidate
