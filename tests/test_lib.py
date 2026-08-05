import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Any

from pysne.problems.benchmarks_sne import get_problem_set
from pysne.utils import objective_function, validate_solutions
from pysne.clustering.clustering_process import perform_iterative_clustering
from pysne.solver import solve_system, run_spo_on_clusters



def test_integration_run():
    print("="*60)
    print("PYSNE FULL INTEGRATION TEST")
    print("="*60 + "\n")

    try:
        # 1. Import Problem
        problems = get_problem_set()
        problem_id = 4
        problem = problems[problem_id]()
        domain, params = problem.get_info()
        expected_roots = params.get('expected_roots', 0)
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.01)
        
        print(f"[STEP 1] Loading Problem {problem_id}")
        print(f"Target: Finding {expected_roots} solution roots.")
        start_time = time.time()

        # 2. Clustering Phase
        print("\n[STEP 2] Running Iterative Clustering...")
        clusters = perform_iterative_clustering(problem, params)
        print(f"Found {len(clusters)} potential regions (clusters).")

        # 3. SPO Phase (Local Optimization)
        print("\n[STEP 3] Running SPO on each cluster...")
        sdoa_params = {
            'm': params.get('spo_m', 50),
            'r': params.get('spo_r', 0.95),
            'theta': params.get('spo_theta', np.pi/4),
            'k_max': params.get('spo_k_max', 200)
        }
        candidates = run_spo_on_clusters(clusters, problem, params)
        print(f"Generated {len(candidates)} candidate points from SPO.")

        # 4. Selection & Validation Phase
        print("\n[STEP 4] Performing final selection and duplicate elimination...")
        raw_roots = problem.select_final_roots(candidates)
        final_roots = validate_solutions(raw_roots, problem.equations, domain, epsilon)
        
        elapsed = time.time() - start_time

        # 5. Result Summary
        print("\n" + "="*20 + " RESULT SUMMARY " + "="*20)
        print(f"Execution Time   : {elapsed:.2f} seconds")
        print(f"Expected Roots   : {expected_roots}")
        print(f"Roots Found      : {len(final_roots)}")
        
        if len(final_roots) > 0:
            print("\nList of Found Roots:")
            for i, root in enumerate(final_roots):
                fitness = problem.evaluate_fitness(root)
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