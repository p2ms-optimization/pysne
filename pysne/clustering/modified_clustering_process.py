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
    params: Dict[str, Any],
    history: List[Dict[str, Any]] = None
) -> List[Cluster]:
    """
    Evaluates a single coordinate point to determine its cluster assignment or if it should form a new cluster based on the objective function landscape.

    This function implements the clustering logic where a point is compared against a threshold gamma. If the point qualifies, it calculates multiple
    interpolation points between the point and the nearest existing cluster center. By comparing the objective values of the point, the nearest center,
    and the interpolation points, the algorithm determines whether to create a new cluster, update the existing cluster's center, or recursively evaluate
    the best interpolation point.

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
    params : dict
        Algorithm hyperparameters, including 'num_check_points' for multi-point interpolation checks.
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

    # Dynamic Multi-point Check Logic
    x_C = nearest_cluster.center
    F_xC = problem.evaluate_fitness(x_C)
    
    # Generate t values dynamically based on num_check_points parameter
    num_check_points = params.get('num_check_points', 1)
    t_vals = [i / (num_check_points + 1) for i in range(1, num_check_points + 1)]
    x_ts = [y + t * (x_C - y) for t in t_vals]
    F_xts = [problem.evaluate_fitness(xt) for xt in x_ts]
    
    F_xt_min = min(F_xts)
    F_xt_max = max(F_xts)

    # Clustering Logic
    dist_half = np.linalg.norm(y - x_C) / 2.0
    case_triggered = None
    
    if F_xt_min < F_y and F_xt_min < F_xC:
        # Case 1: Valley between points; form a new cluster
        case_triggered = 'Case 1 (Valley)'
        clusters.append(Cluster(y.copy(), dist_half))
    elif F_xt_max > F_y and F_xt_max > F_xC:
        # Case 2: A better peak found; form a new cluster and recurse
        case_triggered = 'Case 2 (Mid better)'
        clusters.append(Cluster(y.copy(), dist_half))
        best_xt_idx = int(np.argmax(F_xts))
        x_t_best = x_ts[best_xt_idx]
        clusters = process_point_for_clustering(x_t_best, clusters, problem, gamma, params, history)
    elif F_y > F_xC:
        # Case 3: Update center as y is closer to the root's peak
        case_triggered = 'Case 3 (Update Center)'
        nearest_cluster.center = y.copy()
    else:
        case_triggered = 'None (Only radius updated)'

    nearest_cluster.radius = dist_half

    if history is not None and case_triggered is not None:
        mid_idx = num_check_points // 2
        history.append({
            'case': case_triggered,
            'y': y.copy(),
            'x_C': x_C.copy(),
            'x_t': x_ts[mid_idx].copy(),
            'dist': dist_half,
            'F_y': F_y,
            'F_xC': F_xC,
            'F_xt': F_xts[mid_idx],
            'F_xt_min': F_xt_min,
            'F_xt_max': F_xt_max
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
        A dictionary containing hyperparameters for the clustering phase. Expected keys include 'm_cluster', 'gamma', 'k_cluster', 'r_cl', 'theta_cl', and 'num_check_points'.
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
    num_check_points = params.get('num_check_points', 1)

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
            uses_absolute_cutoff = getattr(problem, 'problem_type', None) in ('SNE', 'Diophantine')

            if uses_absolute_cutoff:
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
                    clusters = process_point_for_clustering(points[i], clusters, problem, cutoff, params, history)


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