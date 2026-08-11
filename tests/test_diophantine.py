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
    assert len(roots) == params.get('expected_roots', 7)
    
    print("[TEST PASSED] Diophantine Problem 1 solved correctly!")

def test_diophantine_problem2():
    problems = get_diophantine_problems()
    prob = problems[2]() # DiophantineProblem3a
    domain, params = prob.get_info()
    
    # Run solver
    result = solve_system(prob, params, verbose=True)
    roots = result['roots']
    
    assert roots is not None
    assert len(roots) == params.get('expected_roots', 1)
    
    print("[TEST PASSED] Diophantine Problem 2 solved correctly!")

if __name__ == "__main__":
    test_diophantine_problem1()
    test_diophantine_problem2()
