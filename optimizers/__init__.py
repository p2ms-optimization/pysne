import numpy as np
from clustering.dynamic import perform_iterative_clustering
from .sdoa.engine import spiral_dynamics_optimization
from initialization.sampling import generate_sobol_points
from utils import objective_function, is_in_domain

def run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon):
    """Jalankan SDOA pada setiap cluster dengan early stopping."""
    candidates = []

    for i, cluster in enumerate(clusters):
        # Determine cluster domain
        cluster_domain = []
        for dim in range(len(domain)):
            cluster_lo = max(domain[dim][0], cluster.center[dim] - cluster.radius)
            cluster_hi = min(domain[dim][1], cluster.center[dim] + cluster.radius)
            cluster_domain.append((cluster_lo, cluster_hi))

        # Generate initial points in cluster domain
        initial_points = generate_sobol_points(sdoa_params['m'], len(domain), cluster_domain)

        # Run SDOA in cluster domain
        def cluster_objective(x):
            return objective_function(x, equations)

        candidate = spiral_dynamics_optimization(
            cluster_objective, cluster_domain, sdoa_params,
            minimization=False, custom_initial_points=initial_points,
            equations=equations, epsilon=epsilon
        )

        candidates.append(candidate)

    return np.array(candidates)

def select_final_roots(candidates, equations, domain, epsilon, delta):
    """Seleksi akhir roots."""
    # Filter based on epsilon and domain
    accurate_candidates = []
    for cand in candidates:
        if not is_in_domain(cand, domain):
            continue
        F_val = objective_function(cand, equations)
        if 1.0 - F_val < epsilon:
            accurate_candidates.append((cand, F_val))

    if not accurate_candidates:
        return np.array([])

    # Filter based on delta
    final_roots = []
    accurate_candidates.sort(key=lambda x: x[1], reverse=True)

    for cand, F_val in accurate_candidates:
        found_close = False
        for i, (existing, existing_F) in enumerate(final_roots):
            distance = np.linalg.norm(cand - existing)
            if distance <= delta:
                found_close = True
                if F_val > existing_F:
                    final_roots[i] = (cand, F_val)
                break
        if not found_close:
            final_roots.append((cand, F_val))

    return np.array([root for root, _ in final_roots])