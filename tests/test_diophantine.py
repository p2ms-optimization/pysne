import numpy as np
from pysne.problems.benchmarks_diophantine import get_diophantine_problems
from pysne.solver import solve_system

def test_diophantine_problem1():
    problems = get_diophantine_problems()
    prob = problems[1]()
    domain, params = prob.get_info()
    
    # Run solver
    result = solve_system(prob, params, verbose=True)
    roots = result['roots']
    
    assert roots is not None
    assert len(roots) == 2
    
    # Sort roots by first coordinate
    roots_sorted = sorted(roots, key=lambda r: r[0])
    
    # Expected solutions: (-3, -4) and (4, 3)
    np.testing.assert_array_almost_equal(roots_sorted[0], [-3.0, -4.0])
    np.testing.assert_array_almost_equal(roots_sorted[1], [4.0, 3.0])
    print("[TEST PASSED] Diophantine Problem 1 solved correctly!")

def test_diophantine_problem2():
    problems = get_diophantine_problems()
    prob = problems[2]() # Intersecting Parabolas
    domain, params = prob.get_info()
    
    # Run solver
    result = solve_system(prob, params, verbose=True)
    roots = result['roots']
    
    assert roots is not None
    assert len(roots) == 2
    
    roots_sorted = sorted(roots, key=lambda r: r[0])
    # Expected solutions: (0, 0) and (1, 1)
    np.testing.assert_array_almost_equal(roots_sorted[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(roots_sorted[1], [1.0, 1.0])
    print("[TEST PASSED] Diophantine Problem 2 solved correctly!")

if __name__ == "__main__":
    test_diophantine_problem1()
    test_diophantine_problem2()
