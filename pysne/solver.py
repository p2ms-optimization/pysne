import numpy as np
import time

from pysne.clustering.modified_clustering_process import perform_iterative_clustering
from pysne.optimizers import get_optimizer

def run_optimizer_on_clusters(clusters, problem, params, optimizer="spo"):
    """
    Executes the specified local optimizer on each cluster
    to find more precise root points.

    This function constructs local domain boundaries (hypercubes) for each cluster
    based on its center and radius, then runs the chosen optimizer within those local domains.

    Parameters
    ----------
    clusters : list
        List of Cluster objects generated from the iterative clustering phase.
    problem : BaseProblem
        The problem instance providing equations, domain, and fitness evaluation.
    params : dict
        Dictionary containing all hyperparameters for the algorithm.
    optimizer : str, BaseOptimizer, or callable, optional
        The local optimizer to use (default: "spo").

    Returns
    -------
    numpy.ndarray
        Array containing the candidate root points found across clusters.
    """
    opt_instance = get_optimizer(optimizer)
    candidates = []
    domain = problem.domain

    is_diophantine = getattr(problem, 'problem_type', None) == 'Diophantine'

    for i, cluster in enumerate(clusters):
        effective_radius = max(cluster.radius, 0.5) if is_diophantine else cluster.radius

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

        if hasattr(opt_instance, 'optimize'):
            candidate = opt_instance.optimize(problem, cluster, cluster_domain, params)
        else:
            candidate = opt_instance(problem, cluster, cluster_domain, params)

        candidates.append(candidate)

    return np.array(candidates)


def run_spo_on_clusters(clusters, problem, params):
    """
    Backward-compatible alias for running Spiral Optimization on each cluster.
    """
    return run_optimizer_on_clusters(clusters, problem, params, optimizer="spo")


def solve_system(problem, params, optimizer="spo", verbose=False):
    """
    Solves a system of nonlinear equations using the integration of
    a local optimizer (e.g. Spiral Optimization or Newton-type) and the Clustering method.
    
    This function executes an entire pipeline consisting of three phases:
    1. Clustering Phase: Localizes potential root areas into clusters.
    2. Optimization Phase: Runs local optimizer on each cluster.
    3. Selection Phase: Filters and validates the unique final roots.

    Parameters
    ----------
    problem : BaseProblem
        The problem instance providing equations, domain, fitness evaluation,
        and solution selection logic.
    params : dict
        Dictionary containing all hyperparameters for the algorithm
        (epsilon, delta, gamma, m_cluster, k_cluster, spo_m, spo_k_max, r, theta).
    optimizer : str, BaseOptimizer, or callable, optional
        The local optimizer to apply to each cluster (default: "spo").
        Supported strings: "spo", "newton", "scipy_root", "hybr", "lm".
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
    # Safe handling if verbose was passed as 3rd positional argument
    if isinstance(optimizer, bool):
        verbose = optimizer
        optimizer = "spo"

    start_time = time.time()

    # PHASE 1: Iterative Clustering
    clusters = perform_iterative_clustering(problem, params)
    
    # PHASE 2: Local Optimization on each cluster
    candidates = run_optimizer_on_clusters(clusters, problem, params, optimizer=optimizer)
    
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
