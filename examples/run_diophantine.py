import sys
import numpy as np
from pysne.problems.benchmarks_diophantine import get_diophantine_problems
from pysne.solver import solve_system

def main(problem_id: int):
    problems = get_diophantine_problems()
    if problem_id not in problems:
        print(f"ID {problem_id} does not exist. Available options: {sorted(problems.keys())}")
        return

    p = problems[problem_id]()
    domain, params = p.get_info()
    print(f"=== {p.name} ===")
    result = solve_system(p, params, verbose=True)
    print("Roots found:")
    print(np.round(result['roots'], 6))

if __name__ == "__main__":
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(problem_id)