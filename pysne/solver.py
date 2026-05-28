import numpy as np
import time

# Internal imports from other modules
from pysne.clustering.modified_clustering_process import perform_iterative_clustering
from pysne.initialization.sampling import generate_sobol_points
from pysne.utils import objective_function, is_in_domain, validate_solutions
from pysne.optimizers.sdoa.engine import spiral_dynamics_optimization

# def run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon):
def run_sdoa_on_clusters(clusters, problem, params):
    """
    Executes the Spiral Dynamics Optimization Algorithm (SDOA) on each cluster
    to find more precise root points.

    This function constructs local domain boundaries (hypercubes) for each cluster
    based on its center and radius [cite: 536], generates new initial points using
    a Sobol sequence, and runs SDOA within those local domains.

    Parameters
    ----------
    clusters : list
        List of Cluster objects generated from the iterative clustering phase.
    equations : list of callable
        A list of functions representing the system of nonlinear equations.
    domain : list of tuple
        The global search space boundaries in the format [(min, max), ...].
    sdoa_params : dict
        Hyperparameters for the SDOA algorithm (m, k_max, r, theta).
    epsilon : float
        Tolerance value (residual) for the early stopping criteria.

    Returns
    -------
    numpy.ndarray
        Array containing the candidate root points optimized by SDOA.
    """
    candidates = []
    domain = problem.domain

    # Ambil parameter khusus SDOA
    sdoa_params = {
        'm': params.get('sdoa_m', params.get('m', 20)),
        'r': params.get('sdoa_r', params.get('r', 0.95)), 
        'theta': params.get('sdoa_theta', params.get('theta', np.pi/4)),
        'k_max': params.get('sdoa_k_max', params.get('k_max', 100))
    }

    epsilon = params.get('epsilon', 1e-7)

    for i, cluster in enumerate(clusters):
        # Determine cluster domain (ensuring it does not exceed global boundaries)
        cluster_domain = []
        # for dim in range(len(domain)):
        for dim in range(problem.n_var):
            cluster_lo = max(domain[dim][0], cluster.center[dim] - cluster.radius)
            cluster_hi = min(domain[dim][1], cluster.center[dim] + cluster.radius)
            cluster_domain.append((cluster_lo, cluster_hi))

        # Skip degenerate domains
        if any(hi - lo < 1e-12 for lo, hi in cluster_domain):
            candidates.append(cluster.center.copy())
            continue

        # Generate initial points in cluster domain
        initial_points = generate_sobol_points(sdoa_params['m'], len(domain), cluster_domain)

        # Run SDOA in cluster domain
        candidate = spiral_dynamics_optimization(
            # cluster_objective, 
            objective_func=problem.evaluate_fitness,
            # cluster_domain, 
            domain=cluster_domain,
            # sdoa_params,
            params=sdoa_params,
            minimization=False, 
            custom_initial_points=initial_points,
            equations=problem.equations,
            epsilon=epsilon
        )

        candidates.append(candidate)


    return np.array(candidates)


# def solve_system(equations, domain, params, verbose=False):
def solve_system(problem, params, verbose=False):
    """
    Solves a system of nonlinear equations using the integration of
    Spiral Dynamics Inspired Optimization (SDOA) and the Clustering method.
    
    This function executes an entire pipeline consisting of three phases:
    1. Clustering Phase: Localizes potential root areas.
    2. Optimization Phase: Runs SDOA on each cluster.
    3. Selection Phase: Filters and validates the unique final roots.

    Parameters
    ----------
    equations : list of callable
        A list of functions representing the nonlinear equation system f_i(x).
    domain : list of tuple
        The search space boundaries in the format [(min, max), (min, max), ...].
    params : dict
        Dictionary containing all hyperparameters for the algorithm
        (epsilon, delta, gamma, m_cluster, k_cluster, sdoa_m, sdoa_k_max, r, theta).
    verbose : bool, optional
        If True, prints execution time and the number of clusters found (default: False).

    Returns
    -------
    dict
        Dictionary containing the execution results with the keys:
        - 'roots': numpy.ndarray of the validated roots.
        - 'clusters': list of Cluster objects found in Phase 1.
        - 'time_elapsed': float representing the computation time in seconds.
    """
    start_time = time.time()

    # PHASE 1: Iterative Clustering
    # clusters = perform_iterative_clustering(equations, domain, params)
    clusters = perform_iterative_clustering(problem, params)
    
    # PHASE 2: SDOA on each cluster
    # candidates = run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon)
    candidates = run_sdoa_on_clusters(clusters, problem, params)
    
    # PHASE 3: Final Selection and Validation
    # final_roots = select_final_roots(candidates, equations, domain, epsilon, params['delta'])
    final_roots = problem.select_final_roots(candidates)

    # Dipindahkan ke class problem
    # Residual validation to ensure accuracy (optional, depending on implementation)
    # valid_roots = validate_solutions(final_roots, equations, domain, epsilon)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"Search completed in {elapsed_time:.3f} seconds.")
        print(f"Found {len(clusters)} clusters and {len(final_roots)} valid roots.")

    return {
        'roots': np.array(final_roots),
        'clusters': clusters,
        'time_elapsed': elapsed_time
    }