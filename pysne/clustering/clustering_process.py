import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from .model import Cluster
from ..utils import objective_function, is_in_domain
from ..initialization.sampling import generate_sobol_points
from ..optimizers.spo.matrix import get_rotation_matrix

def process_point_for_clustering(
    y: np.ndarray, 
    clusters: List[Cluster], 
    problem,
    gamma: float,
    history: List[Dict[str, Any]] = None
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
    problem : BaseProblem
        The problem instance providing domain bounds and fitness evaluation.
    gamma : float
        The cut-off threshold for the objective function. Points with an objective value below this threshold are ignored.
    history : list of dict, optional
        If provided, records clustering decisions for debugging and visualization.

    Returns
    -------
    list of Cluster
        The updated list of clusters after processing the point `y`.
    """
    F_y = problem.evaluate_fitness(y)

    if F_y <= gamma:
        return clusters

    # Initialize the first cluster if the list is empty
    if not clusters:
        initial_radius = 0.5 * min(hi - lo for lo, hi in problem.domain)
        clusters.append(Cluster(y, initial_radius))
        if history is not None:
            history.append({
                'case': 'Init', 'y': y.copy(), 'center': y.copy(), 'radius': initial_radius, 'F_y': F_y
            })
        return clusters

    # Nearest Cluster Search (Vectorized)
    centers = np.array([c.center for c in clusters])
    dists = np.linalg.norm(centers - y, axis=1)
    closest_idx = np.argmin(dists)
    nearest_cluster = clusters[closest_idx]
    min_dist = dists[closest_idx]

    # Mid-point Check Logic
    x_C = nearest_cluster.center
    F_xC = problem.evaluate_fitness(x_C)
    x_t = (y + x_C) / 2.0
    F_xt = problem.evaluate_fitness(x_t)

    # Clustering Logic
    dist_y_xt = np.linalg.norm(y - x_t)
    case_triggered = None
    if F_xt < F_y and F_xt < F_xC:
        # Case 1: Valley between points; form a new cluster
        case_triggered = 'Case 1 (Valley)'
        clusters.append(Cluster(y.copy(), dist_y_xt))
    elif F_xt > F_y and F_xt > F_xC:
        # Case 2: Midpoint is a better peak; form a new cluster and recurse
        case_triggered = 'Case 2 (Mid better)'
        clusters.append(Cluster(y.copy(), dist_y_xt))
        clusters = process_point_for_clustering(x_t, clusters, problem, gamma, history)
    elif F_y > F_xC:
        # Case 3: Update center as y is closer to the root's peak
        case_triggered = 'Case 3 (Update Center)'
        nearest_cluster.center = y.copy()
    else:
        case_triggered = 'None (Only radius updated)'
        
    nearest_cluster.radius = dist_y_xt

    if history is not None and case_triggered is not None:
        history.append({
            'case': case_triggered,
            'y': y.copy(),
            'x_C': x_C.copy(),
            'x_t': x_t.copy(),
            'dist': dist_y_xt,
            'F_y': F_y,
            'F_xC': F_xC,
            'F_xt': F_xt
        })

    return clusters

def perform_iterative_clustering(
    problem, 
    params: Dict[str, Any],
    history: List[Dict[str, Any]] = None
) -> List[Cluster]:
    """
    Executes the main iterative clustering phase to identify all potential root regions within the bounded domain.

    This function generates an initial population of points using a low-discrepancy Sobol sequence to ensure uniform distribution. It then iteratively evaluates 
    each point to dynamically build clusters. After evaluating all points in an iteration, the points are moved iteratively using the spiral dynamics operator 
    toward the current best global point.

    Parameters
    ----------
    problem : BaseProblem
        The problem instance providing domain bounds and fitness evaluation.
    params : dict
        A dictionary containing hyperparameters for the clustering phase. Expected keys include 'm_cluster', 'gamma', 'k_cluster', 'r_cl', and 'theta_cl'.
    history : list of dict, optional
        If provided, records clustering state for debugging and visualization.

    Returns
    -------
    list of Cluster
        A list of distinct Cluster objects representing the neighborhoods of potential roots found in the search space.
    """
    # Parameter Extraction
    m_cluster = params['m_cluster']
    gamma = params.get('gamma', -float('inf'))
    k_cluster = params['k_cluster']
    r = params.get('r_cl', 0.95)
    theta = params.get('theta_cl', np.pi/4)

    n = problem.n_var
    domain = problem.domain

    # 1. Initialize Points Using Sobol Sequence
    points = generate_sobol_points(m_cluster, n, domain)

    # 2. Precompute Spiral Transformation Matrix
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    # 3. Initialize First Cluster based on the current Best Point
    clusters: List[Cluster] = []
    F_values = np.array([problem.evaluate_fitness(p) for p in points])
    best_idx = np.argmax(F_values)
    
    x_prime = points[best_idx].copy()
    initial_radius = 0.5 * min(hi - lo for lo, hi in domain)
    clusters.append(Cluster(x_prime, initial_radius))

    if history is not None:
        history.append({
            'case': 'InitialState',
            'points': points.copy(),
            'clusters': [Cluster(x_prime.copy(), initial_radius)]
        })

    # 4. Main clustering loop
    for k in range(k_cluster):
        F_values = np.array([problem.evaluate_fitness(p) for p in points])
        F_best = np.max(F_values)
        
        # Process points for clustering
        for i in range(m_cluster):
            # Dismiss points outside of the domain
            if not is_in_domain(points[i], domain):
                continue
            
            F_val = problem.evaluate_fitness(points[i])
            is_sne = getattr(problem, 'problem_type', None) == 'SNE'
            
            if is_sne:
                cutoff = gamma
            else:
                if gamma != -float('inf') and gamma is not None:
                    cutoff = gamma * F_best if F_best > 0 else gamma
                else:
                    cutoff = -float('inf')
                    
            if F_val > cutoff:
                centers = np.array([c.center for c in clusters])
                is_center = np.any(np.all(np.abs(centers - points[i]) < 1e-8, axis=1)) if len(centers) > 0 else False
                if not is_center:
                    clusters = process_point_for_clustering(points[i], clusters, problem, cutoff, history)

        # Update points using spiral dynamics
        F_values = np.array([problem.evaluate_fitness(p) for p in points])
        best_idx = np.argmax(F_values)
        x_p = points[best_idx].copy()

        # Update position of all points
        new_points = np.zeros_like(points)
        for i in range(m_cluster):
            new_points[i] = S_n @ points[i] - (S_n - I_n) @ x_p
        points = new_points
  
    return clusters