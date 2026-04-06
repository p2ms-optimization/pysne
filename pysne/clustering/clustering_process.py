import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from .model import Cluster
from ..utils import objective_function, is_in_domain
from ..initialization.sampling import generate_sobol_points
from ..optimizers.sdoa.matrix import get_rotation_matrix

def process_point_for_clustering(
    y: np.ndarray, 
    clusters: List[Cluster], 
    equations: List[Callable], 
    gamma: float, 
    domain: List[Tuple[float, float]]
) -> List[Cluster]:
    """
    Evaluates a single coordinate point to determine its cluster assignment or if it should form a new cluster based on the objective function landscape.

    This function implements the clustering logic where a point is compared against a threshold gamma. If the point qualifies, it calculates the midpoint 
    between the point and the nearest existing cluster center. By comparing the objective values of the point, the nearest center, and their midpoint, the 
    algorithm determines whether to create a new cluster, update the existing cluster's center, or recursively evaluate the midpoint.

    Parameters
    ----------
    y : numpy.ndarray
        The current search point being evaluated.
    clusters : list of Cluster
        The current list of identified clusters in the search space.
    equations : list of callable
        The system of nonlinear equations to be solved.
    gamma : float
        The cut-off threshold for the objective function. Points with an objective value below this threshold are ignored.
    domain : list of tuple
        The boundaries of the search space in the format [(min, max), ...].

    Returns
    -------
    list of Cluster
        The updated list of clusters after processing the point `y`.
    """
    F_y = objective_function(y, equations)

    if F_y <= gamma:
        return clusters

    # Initialize the first cluster if the list is empty
    if not clusters:
        initial_radius = 0.1 * min(hi - lo for lo, hi in domain)
        clusters.append(Cluster(y, initial_radius))
        return clusters

    # Nearest Cluster Search
    min_dist = float('inf')
    nearest_cluster = None
    for cluster in clusters:
        dist = np.linalg.norm(y - cluster.center)
        if dist < min_dist:
            min_dist = dist
            nearest_cluster = cluster

    # Mid-point Check Logic
    x_C = nearest_cluster.center
    F_xC = objective_function(x_C, equations)
    x_t = (y + x_C) / 2.0
    F_xt = objective_function(x_t, equations)

    # Clustering Logic
    if F_xt < F_y and F_xt < F_xC:
        # Case 1: Valley between points; form a new cluster
        new_radius = np.linalg.norm(y - x_t)
        clusters.append(Cluster(y, new_radius))
    elif F_xt > F_y and F_xt > F_xC:
        # Case 2: Midpoint is a better peak; form a new cluster and recurse
        new_radius = np.linalg.norm(y - x_t)
        clusters.append(Cluster(y, new_radius))
        clusters = process_point_for_clustering(x_t, clusters, equations, gamma, domain)
    elif F_y > F_xC:
        # Case 3: Update center as y is closer to the root's peak
        nearest_cluster.center = y.copy()
        nearest_cluster.radius = np.linalg.norm(y - x_t)
    else:
        # Case 4: x_C remains the best center; just update the radius
        nearest_cluster.radius = np.linalg.norm(y - x_t)

    return clusters

def perform_iterative_clustering(
    equations: List[Callable], 
    domain: List[Tuple[float, float]], 
    params: Dict[str, Any]
) -> List[Cluster]:
    """
    Executes the main iterative clustering phase to identify all potential root regions within the bounded domain.

    This function generates an initial population of points using a low-discrepancy Sobol sequence to ensure uniform distribution. It then iteratively evaluates 
    each point to dynamically build clusters. After evaluating all points in an iteration, the points are moved iteratively using the spiral dynamics operator 
    toward the current best global point.

    Parameters
    ----------
    equations : list of callable
        The system of nonlinear equations to be solved.
    domain : list of tuple
        The boundaries of the search space in the format [(min, max), ...].
    params : dict
        A dictionary containing hyperparameters for the clustering phase. Expected keys include 'm_cluster', 'gamma', 'k_cluster', 'r', and 'theta'.

    Returns
    -------
    list of Cluster
        A list of distinct Cluster objects representing the neighborhoods of potential roots found in the search space.
    """
    # Parameter Extraction
    m_cluster = int(params.get('m_cluster', 200))
    gamma = float(params.get('gamma', 0.1))
    k_cluster = int(params.get('k_cluster', 10))
    r = float(params.get('r', 0.95))
    theta = float(params.get('theta', np.pi/4))
    n = len(domain)

    # 1. Initialize Points Using Sobol Sequence
    points = generate_sobol_points(m_cluster, n, domain)

    # 2. Precompute Spiral Transformation Matrix
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    # 3. Initialize First Cluster based on the current Best Point
    clusters: List[Cluster] = []
    F_values = np.array([objective_function(p, equations) for p in points])
    best_idx = np.argmax(F_values)
    
    x_prime = points[best_idx].copy()
    initial_radius = 0.5 * min(hi - lo for lo, hi in domain)
    clusters.append(Cluster(x_prime, initial_radius))

    # 4. Main clustering loop
    for k in range(k_cluster):
        # Process points for clustering
        for i in range(m_cluster):
            # Dismiss points outside of the domain
            if not is_in_domain(points[i], domain):
                continue
            
            F_val = objective_function(points[i], equations)
            if F_val > gamma:
                is_center = any(np.allclose(points[i], cluster.center, atol=1e-8) for cluster in clusters)
                if not is_center:
                    clusters = process_point_for_clustering(points[i], clusters, equations, gamma, domain)

        # Update points using spiral dynamics
        F_values = np.array([objective_function(p, equations) for p in points])
        best_idx = np.argmax(F_values)
        x_p = points[best_idx].copy()

        # Update position of all points
        new_points = np.zeros_like(points)
        for i in range(m_cluster):
            new_points[i] = S_n @ points[i] - (S_n - I_n) @ x_p
        points = new_points
        # points = (points @ S_n.T) - (x_p @ (S_n - I_n).T) # Vectorized Alternative
  
    return clusters
