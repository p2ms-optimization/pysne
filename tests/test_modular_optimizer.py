import pytest
import numpy as np
from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system
from pysne.optimizers import ScipyRootOptimizer, SPOOptimizer

def test_modular_optimizers_sne():
    """
    Tests that both SPO and Newton (ScipyRootOptimizer) can find the expected roots
    in benchmark problem 1 using the modular optimizer interface.
    """
    problem = get_problem_set()[1]()
    domain, params = problem.get_info()

    # 1. Default (SPO)
    res_spo = solve_system(problem, params, optimizer="spo", verbose=False)
    assert len(res_spo['roots']) >= 5

    # 2. String alias "newton"
    res_newton = solve_system(problem, params, optimizer="newton", verbose=False)
    assert len(res_newton['roots']) >= 5

    # 3. Instance of ScipyRootOptimizer with Levenberg-Marquardt
    res_lm = solve_system(problem, params, optimizer=ScipyRootOptimizer(method="lm"), verbose=False)
    assert len(res_lm['roots']) >= 5

    # 4. Custom callable optimizer
    def dummy_opt(prob, cl, cl_dom, par):
        return cl.center.copy()
    res_custom = solve_system(problem, params, optimizer=dummy_opt, verbose=False)
    assert 'roots' in res_custom
