import numpy as np
import time

from pysne.clustering.modified_clustering_process import perform_iterative_clustering
from pysne.initialization.sampling import generate_sobol_points
from pysne.utils import objective_function, is_in_domain
from pysne.optimizers.spo.engine import spiral_optimization


def run_spo_on_clusters(clusters, problem, params):
    """
    Executes the Spiral Optimization (SPO) on each cluster
    to find more precise root points.

    This function constructs local domain boundaries (hypercubes) for each cluster
    based on its center and radius, generates new initial points using
    a Sobol sequence, and runs SPO within those local domains.

    Parameters
    ----------
    clusters : list
        List of Cluster objects generated from the iterative clustering phase.
    problem : BaseProblem
        The problem instance providing equations, domain, and fitness evaluation.
    params : dict
        Dictionary containing all hyperparameters for the algorithm
        (epsilon, delta, gamma, m_cluster, k_cluster, spo_m, spo_k_max, r, theta).

    Returns
    -------
    numpy.ndarray
        Array containing the candidate root points optimized by SPO.
    """
    candidates = []
    domain = problem.domain

    # Extract SPO-specific parameters with fallback to old SDOA names
    spo_params = {
        'm': params.get('spo_m', params.get('sdoa_m', params.get('m', 20))),
        'r': params.get('spo_r', params.get('sdoa_r', params.get('r', 0.95))), 
        'theta': params.get('spo_theta', params.get('sdoa_theta', params.get('theta', np.pi/4))),
        'k_max': params.get('spo_k_max', params.get('sdoa_k_max', params.get('k_max', 100)))
    }

    epsilon = params.get('epsilon', 1e-7)

    # Check if this is an SNE problem (to trigger early stopping in the engine)
    is_sne = hasattr(problem, 'equations')

    is_diophantine = getattr(problem, 'problem_type', None) == 'Diophantine'
    init_method = params.get('init_method', 'sobol')

    for i, cluster in enumerate(clusters):
        effective_radius = max(cluster.radius, 1.0) if is_diophantine else cluster.radius

        cluster_domain = []
        for dim in range(problem.n_var):
            cluster_lo = max(domain[dim][0], cluster.center[dim] - effective_radius)
            cluster_hi = min(domain[dim][1], cluster.center[dim] + effective_radius)

            if is_diophantine and cluster_lo >= cluster_hi:
                mid = (domain[dim][0] + domain[dim][1]) / 2.0
                half = max(0.5, (domain[dim][1] - domain[dim][0]) / 2.0)
                cluster_lo = max(domain[dim][0], mid - half)
                cluster_hi = min(domain[dim][1], mid + half)

            cluster_domain.append((cluster_lo, cluster_hi))

        if any(hi - lo < 1e-12 for lo, hi in cluster_domain):
            candidates.append(cluster.center.copy())
            continue

        initial_points = generate_sobol_points(spo_params['m'], len(domain), cluster_domain)

        # Run SPO in cluster domain
        candidate = spiral_optimization(
            objective_func=problem.evaluate_fitness,
            domain=cluster_domain,
            params=spo_params,
            minimization=False, 
            custom_initial_points=initial_points,
            equations=problem.equations if is_sne else None,
            epsilon=epsilon
        )

        candidates.append(candidate)

    return np.array(candidates)

def solve_system(problem, params, verbose=False):
    """
    Solves a system of nonlinear equations using the integration of
    Spiral Optimization (SPO) and the Clustering method.
    
    This function executes an entire pipeline consisting of three phases:
    1. Clustering Phase: Localizes potential root areas.
    2. Optimization Phase: Runs SPO on each cluster.
    3. Selection Phase: Filters and validates the unique final roots.

    Parameters
    ----------
    problem : BaseProblem
        The problem instance providing equations, domain, fitness evaluation,
        and solution selection logic.
    params : dict
        Dictionary containing all hyperparameters for the algorithm
        (epsilon, delta, gamma, m_cluster, k_cluster, spo_m, spo_k_max, r, theta).
    verbose : bool, optional
        If True, prints execution time and the number of clusters found (default: False).

    Returns
    -------
    dict
        Dictionary containing the execution results with the keys:
        - 'roots': numpy.ndarray of the validated roots.
        - 'optimals': numpy.ndarray alias for roots (general-purpose name).
        - 'clusters': list of Cluster objects found in Phase 1.
        - 'time_elapsed': float representing the computation time in seconds.
    """
    start_time = time.time()

    # PHASE 1: Iterative Clustering
    clusters = perform_iterative_clustering(problem, params)
    
    # PHASE 2: SPO on each cluster
    candidates = run_spo_on_clusters(clusters, problem, params)
    
    # PHASE 3: Final Selection and Validation
    final_roots = problem.select_final_optimal(candidates)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"Search completed in {elapsed_time:.3f} seconds.")
        solution_label = "roots" if getattr(problem, 'problem_type', '') in ('SNE', 'Diophantine') else "optimal solutions"
        print(f"Found {len(clusters)} clusters and {len(final_roots)} valid {solution_label}.")

    return {
        'roots': np.array(final_roots),
        'optimals': np.array(final_roots),
        'clusters': clusters,
        'time_elapsed': elapsed_time
    }


# Public alias — more general name, not restricted to "system" semantics.
solve = solve_system
