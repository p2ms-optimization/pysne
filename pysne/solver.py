import numpy as np
import time

# Internal imports from other modules
from pysne.clustering.clustering_process import perform_iterative_clustering
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

    # Cek apakah ini problem SNE (untuk trigger early stopping di engine)
    is_sne = hasattr(problem, 'equations')

    # DEFINE FUNCTION OUTSIDE LOOP FOR MEMORY EFFICIENCY
    # def cluster_objective(x):
        # return objective_function(x, equations)

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

        # Definisikan fungsi objektif lokal
        # def cluster_objective(x):
            # return problem.evaluate_fitness(x)

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
            # equations=equations, 
            equations=problem.equations if is_sne else None,
            epsilon=epsilon
        )

        candidates.append(candidate)

    return np.array(candidates)

# def select_final_roots dipindah ke base problem 
# def select_final_roots(candidates, equations, domain, epsilon, delta):
#     """
#     Performs the final selection phase to determine valid roots from
#     the candidate points optimized by SDOA.
    
#     This function implements Steps 10 and 11 from the clustering method
#     by Sidarto & Kania (2015). The selection process involves:
#     1. Discarding candidates that fall outside the domain boundaries.
#     2. Discarding candidates with a residual 1 - F(x) >= epsilon.
#     3. Merging candidates that are close to each other (distance <= delta),
#        retaining only the candidate with the highest F(x) value.

#     Parameters
#     ----------
#     candidates : numpy.ndarray or list
#         List of candidate root points resulting from the optimization phase.
#     equations : list of callable
#         A list of functions representing the system of nonlinear equations.
#     domain : list of tuple
#         The global search space boundaries in the format [(min, max), ...].
#     epsilon : float
#         Tolerance value for root accuracy. Candidates are accepted if 1 - F(x) < epsilon.
#     delta : float
#         Minimum distance boundary between distinct roots (equivalence radius).

#     Returns
#     -------
#     numpy.ndarray
#         Array containing the final validated and unique root points.
#     """
#     # Filter 1: Validate domain and epsilon threshold value
#     accurate_candidates = []
#     for cand in candidates:
#         if not is_in_domain(cand, domain):
#             continue
            
#         F_val = objective_function(cand, equations)
#         if 1.0 - F_val < epsilon:
#             accurate_candidates.append((cand, F_val))

#     if not accurate_candidates:
#         return np.array([])

#     # Filter 2: Eliminate adjacent candidates (based on delta)
#     # Sort in descending order based on F_val so that the root 
#     # with the highest accuracy is always evaluated first.
#     accurate_candidates.sort(key=lambda x: x[1], reverse=True)
    
#     final_roots = []

#     # the section below contains slight modifications from the main code
#     for cand, F_val in accurate_candidates:
#         found_close = False
#         for existing, _ in final_roots:
#             distance = np.linalg.norm(cand - existing)
#             if distance <= delta:
#                 found_close = True
#                 break  # Immediately discard the candidate because the existing one is guaranteed to be better (due to sorting)
                
#         if not found_close:
#             final_roots.append((cand, F_val))

#     return np.array([root for root, _ in final_roots])
    

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

    # Extract SDOA-specific parameters
    # sdoa_params = {
    #     'm': params['sdoa_m'],
    #     'r': params['r'],  # Uses the 'r' parameter from the main dictionary
    #     'theta': params['theta'],
    #     'k_max': params['sdoa_k_max']
    # }
    # epsilon = params['epsilon']

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