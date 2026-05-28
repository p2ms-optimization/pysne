# test_benchmarks.py
import numpy as np
import time
import sys
from typing import Dict, Any

from pysne.problems.cec_benchmarks import CEC2014Benchmark, NameBasedBenchmark
from pysne.solver import solve_system
from pysne.problems.base import BaseProblem, MinimizedProblem

def test_composition_problem(func_name: str, ndim: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Menjalankan algoritma SDOA-Clustering untuk problem composition yang kompleks dari Opfunu.
    Mencari minima dan/atau maxima (tergantung landskap fungsi).
    """
    result = {
        'func_name': func_name,
        'success': False,
        'points_found': 0,
        'time': 0.0,
        'error': None,
    }
    try:
        # Parameter penyesuaian untuk fungsi yang kompleks
        custom_params = {
            'm_cluster': 500,
            'k_cluster': 10,
            'sdoa_m': 100,
            'sdoa_k_max': 50,
            'delta': 0.1,
            'epsilon': 1e-4,
            'gamma': 0.05
        }
        
        prob = CEC2014Benchmark.get_function(func_name, ndim=ndim, custom_params=custom_params)
        # Print Informasi Problem (Metadata dari Opfunu)
        print("\n" + "-"*60)
        print(f"INFORMASI PROBLEM: {prob.name}")
        print("-"*60)
        # Menampilkan LaTeX formula persamaannya
        print(f"Persamaan (LaTeX)   : {prob.func.latex_formula}")
        print(f"Global Optimum      : {prob.func.latex_formula_global_optimum}")
        print(f"Bounds Formula      : {prob.func.latex_formula_bounds}")

        # Menampilkan karakteristik (jika ada)
        if hasattr(prob.func, 'characteristics'):
            print(f"Karakteristik       : {prob.func.characteristics}")

        # Menampilkan status/sifat problem
        print(f"Apakah Unimodal?    : {prob.func.unimodal}")
        print(f"Apakah Differentiable?: {prob.func.differentiable}")

        # Menampilkan referensi paper / docstring
        print(f"\nReferensi/Docstring :\n{prob.func.__doc__.strip()}")
        print("-"*60 + "\n")

        
        if verbose:
            print("="*60)
            print(f"SOLVING {prob.name.upper()} (Composition Function)")
            print("="*60)
            
        start_time = time.time()

        # ===============================================
        # 1. FIND MAXIMA (Running the standard solver)
        # ===============================================
        if verbose:
            print(f"\n[STEP 1] Running Standard Solver (Finding Maxima)...")
            
        # Untuk opfunu CEC, ini akan mencari local maxima dari landscape (nilai yang besar)
        res_max = solve_system(prob, prob.get_info()[1], verbose=verbose)
        final_max = res_max['roots']

        # ===============================================
        # 2. FIND MINIMA
        # ===============================================
        if verbose:
            print("\n[STEP 2] Running Minimized Solver (Finding Minima)...")
        
        prob_min = MinimizedProblem(prob)
        res_min = solve_system(prob_min, prob.get_info()[1], verbose=verbose)
        final_min = res_min['roots']

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
                # evaluate_fitness asli (tidak dinegasikan)
                print(f" Max {i+1}: {root.round(4)} | Fitness: {prob.evaluate_fitness(root):.6f}")

            print("\nList of MINIMA Points:")
            for i, root in enumerate(final_min):
                # evaluate_fitness asli (untuk melihat nilai objektif sesungguhnya)
                print(f" Min {i+1}: {root.round(4)} | Fitness: {prob.evaluate_fitness(root):.6f}")
                
            print("\n[STATUS]: SUCCESS! Composition analysis complete.")

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if verbose:
            import traceback
            traceback.print_exc()

    return result

if __name__ == "__main__":
    print("\nPySNE Opfunu Composition Function Test")
    print("Testing SDOA-Clustering on a complex composition landscape\n")
    
    # Menjalankan F23 (Composition Function 1 dari CEC 2014) dengan dimensi 10
    # Dimensi harus ada dalam list supported Opfunu (biasanya 10, 20, 30, 50, 100)
    test_composition_problem('F2', ndim=10, verbose=True)
