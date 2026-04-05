import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Any

from pysne.problems.benchmarks import get_problem_set
from pysne.utils import objective_function, validate_solutions
from pysne.clustering.clustering_process import perform_iterative_clustering
from pysne.solver import solve_system, run_sdoa_on_clusters, select_final_roots



def test_integration_run():
    print("="*60)
    print("PYSNE FULL INTEGRATION TEST")
    print("="*60 + "\n")

    try:
        # 1. Import Problem
        problems = get_problem_set()
        problem_id = 2
        equations, domain, params, expected_roots = problems[problem_id]()
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.01)
        
        print(f"[STEP 1] Loading Problem {problem_id}")
        print(f"Target: Finding {expected_roots} solution roots.")
        start_time = time.time()

        # 2. Clustering Phase
        print("\n[STEP 2] Running Iterative Clustering...")
        clusters = perform_iterative_clustering(equations, domain, params)
        print(f"Found {len(clusters)} potential regions (clusters).")

        # 3. SDOA Phase (Local Optimization)
        print("\n[STEP 3] Running SDOA on each cluster...")
        sdoa_params = {
            'm': params.get('sdoa_m', 50),
            'r': params.get('sdoa_r', 0.95),
            'theta': params.get('sdoa_theta', np.pi/4),
            'k_max': params.get('sdoa_k_max', 200)
        }
        candidates = run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon)
        print(f"Generated {len(candidates)} candidate points from SDOA.")

        # 4. Selection & Validation Phase
        print("\n[STEP 4] Performing final selection and duplicate elimination...")
        raw_roots = select_final_roots(candidates, equations, domain, epsilon, delta)
        final_roots = validate_solutions(raw_roots, equations, domain, epsilon)
        
        elapsed = time.time() - start_time

        # 5. Result Summary
        print("\n" + "="*20 + " RESULT SUMMARY " + "="*20)
        print(f"Execution Time   : {elapsed:.2f} seconds")
        print(f"Expected Roots   : {expected_roots}")
        print(f"Roots Found      : {len(final_roots)}")
        
        if len(final_roots) > 0:
            print("\nList of Found Roots:")
            for i, root in enumerate(final_roots):
                fitness = objective_function(root, equations)
                print(f"  Root {i+1}: {root.round(6)} | Residual: {1.0-fitness:.2e}")

        # Final Evaluation
        if len(final_roots) == expected_roots:
            print("\n[STATUS]: SUCCESS! All roots found with high precision.")
        elif len(final_roots) > expected_roots:
            print("\n[STATUS]: WARNING! Found more points (delta might be too small).")
        else:
            print("\n[STATUS]: FAILED! Some roots missed. Needs parameter tuning.")

    except Exception as e:
        print(f"\n[ERROR]: System failure occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration_run()