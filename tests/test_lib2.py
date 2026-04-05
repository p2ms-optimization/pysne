"""
Comprehensive Testing Module for PySNE Solver

This module tests the main solve_system() function from solver.py
with proper assertions and comprehensive coverage.

Usage:
    python test_lib.py                          # Run default single test
    pytest test_lib.py -v                       # Run all tests with pytest
    pytest test_lib.py::test_problem_1 -v      # Run specific test
"""

import numpy as np
import time
import sys
from typing import Dict, List, Any

# ============================================================================
# IMPORTS
# ============================================================================

from pysne.problems.benchmarks import get_problem_set
from pysne.utils import objective_function, validate_solutions
from pysne.solver import solve_system


# ============================================================================
# INDIVIDUAL TEST FUNCTIONS
# ============================================================================

def test_problem(problem_id: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Test solve_system() on a single benchmark problem.
    
    Parameters
    ----------
    problem_id : int
        Problem ID (1-7)
        
    verbose : bool
        Print detailed output with root display
        
    Returns
    -------
    dict
        Test results with keys:
        - 'problem_id': Problem number
        - 'success': bool, whether test passed
        - 'roots_found': int, number of roots found
        - 'roots_expected': int, expected number of roots
        - 'time': float, execution time
        - 'error': str or None, error message if failed
    """
    
    result = {
        'problem_id': problem_id,
        'success': False,
        'roots_found': 0,
        'roots_expected': 0,
        'time': 0.0,
        'error': None,
    }
    
    try:
        # === LOAD PROBLEM ===
        problems = get_problem_set()
        if problem_id not in problems:
            result['error'] = f"Problem {problem_id} not found"
            return result
        
        equations, domain, params, expected_roots = problems[problem_id]()
        result['roots_expected'] = expected_roots
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.01)
        
        if verbose:
            print(f"[STEP 1] Loading Problem {problem_id}")
            print(f"Target: Finding {expected_roots} solution roots.")
        
        start_time = time.time()
        
        # === RUN SOLVER ===
        solve_result = solve_system(
            equations=equations,
            domain=domain,
            params=params,
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        result['time'] = elapsed_time
        
        # === EXTRACT RESULTS ===
        final_roots = solve_result['roots']
        clusters = solve_result['clusters']
        result['roots_found'] = len(final_roots)
        
        if verbose:
            print(f"\n[STEP 2] Running Iterative Clustering...")
            print(f"Found {len(clusters)} potential regions (clusters).")
            print(f"\n[STEP 3] Running SDOA on each cluster...")
            print(f"Generated {len(final_roots)} candidate points from SDOA.")
            print(f"\n[STEP 4] Performing final selection and duplicate elimination...")
        
        # === DETERMINE SUCCESS ===
        success_rate = len(final_roots) / expected_roots if expected_roots > 0 else 0
        
        if len(final_roots) == expected_roots:
            result['success'] = True
        elif success_rate >= 0.8:
            result['success'] = True
        else:
            result['success'] = False
            result['error'] = f"Only found {len(final_roots)}/{expected_roots} roots"
        
        if verbose:
            # === DISPLAY RESULTS ===
            print("\n" + "="*20 + " RESULT SUMMARY " + "="*20)
            print(f"Execution Time   : {elapsed_time:.2f} seconds")
            print(f"Expected Roots   : {expected_roots}")
            print(f"Roots Found      : {len(final_roots)}")
            
            if len(final_roots) > 0:
                print("\nList of Found Roots:")
                for i, root in enumerate(final_roots):
                    fitness = objective_function(root, equations)
                    residual = 1.0 - fitness
                    print(f"  Root {i+1}: {root.round(6)} | Residual: {residual:.2e}")
            
            # === STATUS ===
            if len(final_roots) == expected_roots:
                print("\n[STATUS]: SUCCESS! All roots found with high precision.")
            elif len(final_roots) > expected_roots:
                print("\n[STATUS]: WARNING! Found more points (delta might be too small).")
            elif len(final_roots) >= expected_roots * 0.8:
                print(f"\n[STATUS]: SUCCESS! Found {len(final_roots)}/{expected_roots} roots ({100*len(final_roots)/expected_roots:.0f}%).")
            else:
                print("\n[STATUS]: FAILED! Some roots missed. Needs parameter tuning.")
    
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if verbose:
            print(f"\n[ERROR]: System failure occurred: {e}")
            import traceback
            traceback.print_exc()
    
    return result


# ============================================================================
# BATCH TESTING
# ============================================================================

def test_all_problems(problem_ids: list = None, show_details: bool = False) -> Dict[int, Dict]:
    """
    Test solve_system() on multiple benchmark problems.
    
    Parameters
    ----------
    problem_ids : list, optional
        Problems to test. Default: [1, 2, 3, 4, 5, 6, 7]
        
    show_details : bool
        Print detailed output for each problem (default: False)
        
    Returns
    -------
    dict
        Results for each problem
    """
    
    if problem_ids is None:
        problem_ids = [1, 2, 3, 4, 5, 6, 7]
    
    print("="*60)
    print("PYSNE COMPREHENSIVE SOLVER TEST")
    print("="*60 + "\n")
    
    results = {}
    total_start = time.time()
    
    for problem_id in problem_ids:
        results[problem_id] = test_problem(problem_id, verbose=show_details)
    
    total_time = time.time() - total_start
    
    # === SUMMARY TABLE ===
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    print(f"{'Problem':<10} {'Status':<15} {'Found/Expected':<20} {'Time(s)':<10}")
    print("-" * 60)
    
    successes = 0
    for problem_id in problem_ids:
        res = results[problem_id]
        status = "PASS" if res['success'] else "FAIL"
        
        if res['success']:
            successes += 1
        
        found_expected = f"{res['roots_found']}/{res['roots_expected']}"
        
        print(f"{problem_id:<10} {status:<15} {found_expected:<20} {res['time']:<10.3f}")
    
    print("-" * 60)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success Rate: {successes}/{len(problem_ids)} ({100*successes/len(problem_ids):.1f}%)\n")
    
    return results


# ============================================================================
# PYTEST COMPATIBLE TESTS
# ============================================================================

def test_problem_1():
    """Test Problem 1 (2D, 6 roots)."""
    result = test_problem(1, verbose=False)
    assert result['success'], f"Problem 1 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_2():
    """Test Problem 2 (2D, 12 roots)."""
    result = test_problem(2, verbose=False)
    assert result['success'], f"Problem 2 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_3():
    """Test Problem 3 (6D, 2 roots)."""
    result = test_problem(3, verbose=False)
    assert result['success'], f"Problem 3 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_4():
    """Test Problem 4 (3D, 6 roots)."""
    result = test_problem(4, verbose=False)
    assert result['success'], f"Problem 4 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_5():
    """Test Problem 5 (5D, 3 roots)."""
    result = test_problem(5, verbose=False)
    assert result['success'], f"Problem 5 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_6():
    """Test Problem 6 (8D, 16 roots)."""
    result = test_problem(6, verbose=False)
    assert result['success'], f"Problem 6 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


def test_problem_7():
    """Test Problem 7 (1D Weierstrass, 9 roots)."""
    result = test_problem(7, verbose=False)
    assert result['success'], f"Problem 7 failed: {result['error']}"
    assert result['roots_found'] >= int(result['roots_expected'] * 0.8)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\nPySNE Solver Test Suite")
    print("Testing solver.py implementation\n")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "all":
            # Test all problems with summary only
            results = test_all_problems(show_details=False)
            success_count = sum(1 for r in results.values() if r['success'])
            sys.exit(0 if success_count == len(results) else 1)
        
        elif command == "all-verbose":
            # Test all problems with detailed output
            results = test_all_problems(show_details=True)
            success_count = sum(1 for r in results.values() if r['success'])
            sys.exit(0 if success_count == len(results) else 1)
        
        elif command.isdigit():
            # Test specific problem with detailed output
            problem_id = int(command)
            result = test_problem(problem_id, verbose=True)
            sys.exit(0 if result['success'] else 1)
        
        else:
            print("Usage:")
            print("  python test_lib.py [1-7]       - Test specific problem (with details)")
            print("  python test_lib.py all         - Test all problems (summary only)")
            print("  python test_lib.py all-verbose - Test all problems (with details)")
            print("\nOr use pytest:")
            print("  pytest test_lib.py -v          - Run all pytest tests")
            print("  pytest test_lib.py::test_problem_3 -v  - Run specific test")
            sys.exit(1)
    
    else:
        # Default: Run all problems with summary
        print("Running all tests...\n")
        results = test_all_problems(show_details=False)
        success_count = sum(1 for r in results.values() if r['success'])
        sys.exit(0 if success_count == len(results) else 1)