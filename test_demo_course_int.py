import numpy as np
from pysne.problems.base import DiophantineProblem
from pysne.solver import solve_system

class MyDiophantineProblem(DiophantineProblem):
    @property
    def name(self):
        return "Contoh: 15x + 11y = 12"

    def get_equations(self):
        return [lambda x: 15*x[0] + 11*x[1] - 12]

    def get_info(self):
        domain = [(-50, 50), (-50, 50)]
        params = {
            'm_cluster': 375, 'k_cluster': 10,
            'r_cl': 0.95, 'theta_cl': np.pi/4,
            'gamma': 0.01,
            'sdoa_m': 30, 'sdoa_k_max': 10,
            'r': 0.95, 'theta': np.pi/4,
            'epsilon': 1e-7, 'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 7,
        }
        return domain, params

if __name__ == "__main__":
    prob = MyDiophantineProblem()
    domain, params = prob.get_info()
    result = solve_system(prob, params, verbose=True)
    print(np.round(result['roots'], 6))