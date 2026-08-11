import pytest
from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system

@pytest.mark.parametrize("problem_id", [1, 2, 3, 4, 5, 6, 7])
def test_sne_problem(problem_id):
    """
    Automated tests for SNE benchmark problems (1-7).
    Ensures that the SPO solver finds at least 80% of the expected roots.
    """
    problems = get_problem_set()
    if problem_id not in problems:
        pytest.skip(f"Problem {problem_id} not implemented")
        
    problem = problems[problem_id]()
    domain, params = problem.get_info()
    expected_roots = params.get('expected_roots', 0)
    
    # Run solver silently
    solve_result = solve_system(
        problem=problem,
        params=params,
        verbose=False
    )
    
    final_roots = solve_result['roots']
    roots_found = len(final_roots)
    
    assert roots_found > 0, f"No roots found for Problem {problem_id}"
    
    # We expect at least 80% success rate on the benchmark
    if expected_roots > 0:
        assert roots_found >= int(expected_roots * 0.8), (
            f"Problem {problem_id} failed: Found {roots_found}/{expected_roots} roots."
        )
