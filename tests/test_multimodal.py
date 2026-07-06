import numpy as np
import time
import sys
from typing import Dict, Any
import pytest

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.solver import solve_system
from pysne.problems.base import BaseProblem, MinimizedProblem



def test_multimodal_pipeline_execution():
    """
    Tes sederhana (smoke test) untuk memastikan pipeline multimodal 
    berjalan dari awal hingga akhir menggunakan OOP benchmarks.
    """
    # 1. Ambil daftar problem
    problems = get_multimodal_problems()
    
    # 2. Inisialisasi Problem 2 (Six Hump Camel Back)
    # Problem yang digunakan
    prob2 = problems[3]()
    
    # 3. Ambil domain dan parameter asli
    _, original_params = prob2.get_info()
    
    # 4. Modifikasi (override) sebagian parameter agar tes berjalan dalam hitungan detik
    test_params = original_params.copy()
    # test_params.update({
        # 'm_cluster': 50,   # Diperkecil dari 1000
        # 'k_cluster': 5,    # Diperkecil dari 20
        # 'm': 50,           # Diperkecil dari 200
        # 'k_max': 50,        # Diperkecil dari 200
        # 'delta': 0.1
    # })

    # 5. Eksekusi fungsi utama
    result = solve_system(prob2, test_params, verbose=True)

    # 6. Assertions
    assert result is not None, "Output solver tidak boleh None"
    assert 'roots' in result, "Hasil harus mengandung key 'roots'"

    roots = result['roots']
    assert isinstance(roots, np.ndarray), "Output roots harus berupa numpy array"
    assert len(roots) > 0, f"Harus menemukan setidaknya satu optima untuk {prob2.name}"

    # Print hasil untuk verifikasi
    print(f"\n[TEST RESULT] Found {len(roots)} optima for {prob2.name}")

# if __name__ == "__main__":
    # test_multimodal_pipeline_execution()

def test_multimodal_problem(
    problem_id: int,
    verbose: bool = True
    ) -> Dict[str, Any]:
    result = {
        'problem_id': problem_id,
        'success': False,
        'points_found': 0,
        'time': 0.0,
        'error': None,
    }
    try:
        problems = get_multimodal_problems()
        if problem_id not in problems:
            result['error'] = f"Problem {problem_id} not found"
            return result
            
        # Inisialisasi object problem
        prob = problems[problem_id]()

        if verbose:
            print("="*60)
            print(f"SOLVING {prob.name.upper()}")
            print("="*60)
            
        start_time = time.time()

        # ===============================================
        # 1. FIND MAXIMA (Running the standard solver)
        # ===============================================

        final_max = []
        if prob.optima_type in ['max', 'both']:
            if verbose:
                print(f"\n[STEP 1] Loading Problem {problem_id} (Multimodal) - Finding Maxima")
            
            print(f"info parameter: {prob.get_info()}")
            # Using solve_system
            res_max = solve_system(prob, prob.get_info()[1], verbose=verbose)
            final_max = res_max['roots']
        else:
            if verbose:
                print(f"\n[STEP 1] Skipping Maxima search for Problem {problem_id} (optima_type='{prob.optima_type}')")

        # ===============================================
        # 2. FIND MINIMA
        # ===============================================
        final_min = []
        if prob.optima_type in ['min', 'both']:
            if verbose:
                print("\n[STEP 2] Running Universal Solver (Finding Minima)...")

            prob_min = MinimizedProblem(prob)
            res_min = solve_system(prob_min, prob.get_info()[1], verbose=verbose)
            final_min = res_min['roots']
        else:
            if verbose:
                print(f"\n[STEP 2] Skipping Minima search for Problem {problem_id} (optima_type='{prob.optima_type}')")

        # ===============================================
        # 3. CONSOLIDATE RESULTS
        # ===============================================

        elapsed_time = time.time() - start_time
        result['time'] = elapsed_time
        result['points_found'] = len(final_max) + len(final_min)
        result['success'] = True

        if verbose:
            print("\n" + "="*20 + " RESULT SUMMARY " + "="*20)
            print(f"Execution Time   : {elapsed_time:.2f} seconds")
            print(f"Total Optima     : {result['points_found']} (Max: {len(final_max)}, Min: {len(final_min)})")
            
            print("\nList of MAXIMA Points:")
            for i, root in enumerate(final_max):
                # print(f"  Max {i+1}: {root.round(6)} | g(x): {g_func(root):.6f}")
                print(f" Max {i+1}: {root.round(6)} | g(x): {prob.evaluate_fitness(root)}")

            print("\nList of MINIMA Points:")
            for i, root in enumerate(final_min):
                # print(f"  Min {i+1}: {root.round(6)} | g(x): {g_func(root):.6f}")
                print(f" Min {i+1}: {root.round(9)} | g(x): {prob.evaluate_fitness(root)}")
                
            print("\n[STATUS]: SUCCESS! Multimodal analysis complete.")

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()

    return result

def test_all_multimodal_problems(problem_ids: list = None, show_details: bool = False):
    if problem_ids is None:
        problem_ids = [1, 2, 3, 4, 5, 6, 7]
        
    print("="*60)
    print("PYSNE MULTIMODAL COMPREHENSIVE TEST")
    print("="*60 + "\n")
    
    results = {}
    total_start = time.time()
    
    for problem_id in problem_ids:
        results[problem_id] = test_multimodal_problem(problem_id, verbose=show_details)
        
    total_time = time.time() - total_start

    print("\n" + "="*60)
    print("SUMMARY MULTIMODAL RESULTS")
    print("="*60)
    print(f"{'Problem':<10} {'Status':<15} {'Total Optima':<15} {'Time(s)':<10}")
    print("-" * 60)
    
    for problem_id in problem_ids:
        res = results[problem_id]
        status = "PASS" if res['success'] else "FAIL"
        print(f"{problem_id:<10} {status:<15} {res['points_found']:<15} {res['time']:<10.3f}")
        
    print("-" * 60)
    print(f"Total Time: {total_time:.2f}s\n")
    return results

if __name__ == "__main__":
    print("\nPySNE Multimodal Test Suite")
    print("Testing objective function optimizers\n")
    
    # test_multimodal_problem(2, verbose=True)
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "all":
            test_all_multimodal_problems(show_details=False)
        elif command == "all-verbose":
            test_all_multimodal_problems(show_details=True)
        elif command.isdigit():
            test_multimodal_problem(int(command), verbose=True)
    else:
        test_all_multimodal_problems(show_details=False)
