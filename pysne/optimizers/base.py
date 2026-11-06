from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

class BaseOptimizer(ABC):
    """
    Abstract base class for local optimizers within each cluster.
    """

    @abstractmethod
    def optimize(
        self,
        problem: Any,
        cluster: Any,
        cluster_domain: List[Tuple[float, float]],
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Executes local optimization within the cluster domain
        and returns the candidate solution coordinate vector.

        Parameters
        ----------
        problem : BaseProblem
            The problem instance providing equations, fitness evaluation, and domain.
        cluster : Cluster
            The cluster object with center and radius.
        cluster_domain : list of tuple
            The bounding box bounds for each dimension: [(lo_1, hi_1), ...].
        params : dict
            Hyperparameters for the algorithm.

        Returns
        -------
        numpy.ndarray
            The candidate coordinate vector found in this cluster.
        """
        pass
