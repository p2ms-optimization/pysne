import numpy as np
from typing import Callable, Dict, List, Tuple
from pysne.problems.base import DiophantineProblem

class DiophantineProblem1(DiophantineProblem):
    """
    Problem 1: 15x + 11y = 12  (Linear, 2D)

    A linear Diophantine equation with two variables.
    Expected roots: 7 integer solutions in domain [-50, 50]^2.

    Reference: Sumarti et al. (2023)
    """

    @property
    def name(self) -> str:
        return "Diophantine Problem 1: 15x + 11y = 12"

    def get_integer_domain(self) -> List[tuple]:
        """
        Original integer domain.
        In the base class, the solver domain is automatically converted
        to continuous bounds (integer ± 0.5) via get_info().
        """
        return [(-50, 50), (-50, 50)]

    def get_equations(self) -> List[Callable]:
        """Equations f_i(x) = 0 whose integer solutions are sought."""
        return [
            lambda var: 15 * var[0] + 11 * var[1] - 12
        ]

    def get_params(self) -> dict:
        """
        Algorithm hyperparameters for this problem.

        Parameter keys:
        - m_cluster, k_cluster, gamma, r_cl, theta_cl : clustering phase
        - spo_m, spo_k_max, r, theta               : SPO phase
        - epsilon, delta                              : solution selection
        - num_check_points                            : number of interpolation check points
        - expected_roots                              : metadata (expected number of solutions)
        """
        return {
            # Clustering Parameters
            'm_cluster': 375,
            'k_cluster': 10,
            'gamma': 0.01,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            # SPO Parameters
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 30,
            'spo_k_max': 10,
            # Selection Parameters
            'epsilon': 1e-7,
            'delta': 0.1,
            # Multi-point check parameter
            'num_check_points': 3,
            # Metadata
            'expected_roots': 7
        }

class Problem2(DiophantineProblem):
    """
    Exponential-Polynomial Diophantine System (2D):
    f1(x) = 3^x1 - x2^2 - 8 = 0
    f2(x) = x1^2 + x2^2 - 13 = 0
    """
    @property
    def name(self):
        return "Diophantine Problem 2: 2D Exponential-Polynomial System"

    def get_equations(self):
        return [
            lambda x: 3**int(x[0]) - x[1]**2 - 8,
            lambda x: x[0]**2 + x[1]**2 - 13
        ]

    def get_integer_domain(self):
        return [(-5, 5), (-5, 5)]

    def get_params(self):
        return {
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-5,
            'delta': 0.5,
            'spo_m': 30,
            'spo_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'expected_roots': 2
        }


class DiophantineProblem3a(DiophantineProblem):
    """
    Problem 3a: x1^3 + x2^3 = 1008 (Cubic, 2D)
    Expected roots: 1 solution.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 3a: x1^3 + x2^3 = 1008"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 10), (1, 10)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**3 + var[1]**3 - 1008]

    def get_params(self) -> dict:
        return {
            'm_cluster': 200,
            'k_cluster': 20,
            'gamma': 0.1,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 50,
            'spo_k_max': 50,
            'epsilon': 1e-5,
            'delta': 0.01,
            'num_check_points': 3,
            'expected_roots': 1
        }

class DiophantineProblem3b(DiophantineProblem):
    """
    Problem 3b: x1^9 + x2^9 = 1000019683 (9th Degree, 2D)
    Expected roots: 1 solution.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 3b: x1^9 + x2^9 = 1000019683"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 10), (1, 10)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**9 + var[1]**9 - 1000019683]

    def get_params(self) -> dict:
        return {
            'm_cluster': 100,
            'k_cluster': 20,
            'gamma': 0.1,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 50,
            'spo_k_max': 50,
            'epsilon': 1e-5,
            'delta': 0.01,
            'num_check_points': 3,
            'expected_roots': 1
        }
    
class DiophantineProblem4_4(DiophantineProblem):
    """
    Problem 4.4: x1^2 + x2^2 + x3^2 + x4^2 = 4 * x1*x2*x3*x4 (4D)
    Expected roots: 5 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.4: n=4, k=4"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 200), (1, 20), (1, 5), (1, 5)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 - 4 * var[0] * var[1] * var[2] * var[3]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1500,
            'k_cluster': 20,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 80,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 5
        }

class DiophantineProblem4_5(DiophantineProblem):
    """
    Problem 4.5: x1^2 + ... + x5^2 = 4 * x1*...*x5 (5D)
    Expected roots: 5 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.5: n=5, k=4"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 100), (1, 50), (1, 5), (1, 5), (1, 5)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 - 4 * var[0] * var[1] * var[2] * var[3] * var[4]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 3250,
            'k_cluster': 30,
            'gamma': 0.001,
            'r_cl': 0.975,
            'theta_cl': np.pi / 4,
            'r': 0.975,
            'theta': np.pi / 6,
            'spo_m': 150,
            'spo_k_max': 30,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 5
        }
    
class DiophantineProblem4_6(DiophantineProblem):
    """
    Problem 4.6: x1^2 + ... + x6^2 = 3 * x1*...*x6 (6D)
    Expected roots: 5 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.6: n=6, k=3"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 50), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 + var[5]**2 - 3 * var[0] * var[1] * var[2] * var[3] * var[4] * var[5]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1000,
            'k_cluster': 10,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 50,
            'spo_k_max': 10,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 5
        }

class DiophantineProblem4_7(DiophantineProblem):
    """
    Problem 4.7: x1^2 + ... + x7^2 = 2 * x1*...*x7 (7D)
    Expected roots: 5 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.7: n=7, k=2"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 60), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3), (1, 3)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 + var[5]**2 + var[6]**2 - 2 * var[0] * var[1] * var[2] * var[3] * var[4] * var[5] * var[6]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1000,
            'k_cluster': 15,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 250,
            'spo_k_max': 15,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 5
        }
    
class DiophantineProblem4_8(DiophantineProblem):
    """
    Problem 4.8: x1^2 + ... + x8^2 = 1 * x1*...*x8 (8D)
    Expected roots: 3 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.8: n=8, k=1"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 50), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 + var[5]**2 + var[6]**2 + var[7]**2 - var[0] * var[1] * var[2] * var[3] * var[4] * var[5] * var[6] * var[7]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 2000,
            'k_cluster': 20,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 100,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 3
        }

class DiophantineProblem4_9(DiophantineProblem):
    """
    Problem 4.9: x1^2 + ... + x9^2 = 6 * x1*...*x9 (9D)
    Expected roots: 4 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.9: n=9, k=6"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 30), (1, 5), (1, 3), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 + var[5]**2 + var[6]**2 + var[7]**2 + var[8]**2 - 6 * var[0] * var[1] * var[2] * var[3] * var[4] * var[5] * var[6] * var[7] * var[8]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1500,
            'k_cluster': 15,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 80,
            'spo_k_max': 15,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 4
        }
    
class DiophantineProblem4_10(DiophantineProblem):
    """
    Problem 4.10: x1^2 + ... + x10^2 = 1 * x1*...*x10 (10D)
    Expected roots: 3 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 4.10: n=10, k=1"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 20), (1, 10), (1, 5), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + var[1]**2 + var[2]**2 + var[3]**2 + var[4]**2 + var[5]**2 + var[6]**2 + var[7]**2 + var[8]**2 + var[9]**2 - var[0] * var[1] * var[2] * var[3] * var[4] * var[5] * var[6] * var[7] * var[8] * var[9]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1500,
            'k_cluster': 20,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 100,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 3
        }

class DiophantineProblem5a(DiophantineProblem):
    """
    Problem 5a: Ramanujan-Nagell: x^2 + 7 = y^n (3D: x, y, n)
    Expected roots: 7 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 5a: Ramanujan-Nagell"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 500), (1, 50), (3, 50)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + 7 - var[1]**var[2]]

    def get_params(self) -> dict:
        return {
            'm_cluster': 4000,
            'k_cluster': 40,
            'gamma': 1e-5,
            'r_cl': 0.975,
            'theta_cl': np.pi / 16,
            'r': 0.975,
            'theta': np.pi / 16,
            'spo_m': 150,
            'spo_k_max': 30,
            'epsilon': 0.001,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 7
        }

class DiophantineProblem6a(DiophantineProblem):
    """
    Problem 6a: x^2 + 2^a * 11^b = y^n (For n=3, 4D: x, y, a, b)
    Expected roots: 8 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 6a: Exponential (n=3)"

    def get_integer_domain(self) -> List[tuple]:
        return [(0, 20)] * 4

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + (2**var[2]) * (11**var[3]) - var[1]**3]

    def get_params(self) -> dict:
        return {
            'm_cluster': 500,
            'k_cluster': 20,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 100,
            'spo_k_max': 20,
            'epsilon': 1e-5,
            'delta': 0.01,
            'num_check_points': 3,
            'expected_roots': 8
        }


class DiophantineProblem6b(DiophantineProblem):
    """
    Problem 6b: x^2 + 2^a * 11^b = y^n (For n=4, 4D: x, y, a, b)
    Expected roots: 5 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 6b: Exponential (n=4)"

    def get_integer_domain(self) -> List[tuple]:
        return [(0, 20)] * 4

    def get_equations(self) -> List[Callable]:
        return [lambda var: var[0]**2 + (2**var[2]) * (11**var[3]) - var[1]**4]

    def get_params(self) -> dict:
        return {
            'm_cluster': 15000,
            'k_cluster': 20,
            'gamma': 0.01,
            'r_cl': 0.95,
            'theta_cl': np.pi / 3,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 100,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.001,
            'num_check_points': 3,
            'expected_roots': 5
        }


class DiophantineProblem7(DiophantineProblem):
    """
    Problem 7: 2^k + 3*x^2 = y^3 (3D: x, y, k)
    Expected roots: 9 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 7: 2^k + 3*x^2 = y^3"

    def get_integer_domain(self) -> List[tuple]:
        return [(0, 50), (0, 50), (0, 50)]

    def get_equations(self) -> List[Callable]:
        return [lambda var: 2**var[2] + 3*(var[0]**2) - var[1]**3]

    def get_params(self) -> dict:
        return {
            'm_cluster': 30000,
            'k_cluster': 20,
            'gamma': 0.1,
            'r_cl': 0.90,
            'theta_cl': np.pi / 60,
            'r': 0.90,
            'theta': np.pi / 60,
            'spo_m': 50,
            'spo_k_max': 50,
            'epsilon': 1e-5,
            'delta': 0.01,
            'num_check_points': 3,
            'expected_roots': 9
        }


class DiophantineProblem8(DiophantineProblem):
    """
    Problem 8: 5^x1 + 5^x2 = 3^x3 + 7^x4 (4D)
    Expected roots: 9 solutions.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 8: Exponential System"

    def get_integer_domain(self) -> List[tuple]:
        return [(-1, 10)] * 4

    def get_equations(self) -> List[Callable]:
        return [lambda var: 5**var[0] + 5**var[1] - (3**var[2] + 7**var[3])]

    def get_params(self) -> dict:
        return {
            'm_cluster': 450,
            'k_cluster': 35,
            'gamma': 0.1,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 20,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 9
        }


class DiophantineProblem9a(DiophantineProblem):
    """
    Problem 9a: Pell Equations (p=2, 3D: x, y, z)
    Expected roots: 1 solution.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 9a: Pell Equations (p=2)"

    def get_integer_domain(self) -> List[tuple]:
        return [(400, 500), (1, 100), (1, 100)]

    def get_equations(self) -> List[Callable]:
        return [
            lambda v: v[0]**2 - 24*(v[1]**2) - 1,
            lambda v: v[1]**2 - 2*(v[2]**2) - 1
        ]

    def get_params(self) -> dict:
        return {
            'm_cluster': 400,
            'k_cluster': 20,
            'gamma': 0.0001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 100,
            'spo_k_max': 20,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 1
        }


class DiophantineProblem9b(DiophantineProblem):
    """
    Problem 9b: Pell Equations (p=11, 3D: x, y, z)
    Expected roots: 1 solution.
    """
    @property
    def name(self) -> str:
        return "Diophantine Problem 9b: Pell Equations (p=11)"

    def get_integer_domain(self) -> List[tuple]:
        return [(1, 75)] * 3

    def get_equations(self) -> List[Callable]:
        return [
            lambda v: v[0]**2 - 24*(v[1]**2) - 1,
            lambda v: v[1]**2 - 11*(v[2]**2) - 1
        ]

    def get_params(self) -> dict:
        return {
            'm_cluster': 1500,
            'k_cluster': 20,
            'gamma': 0.001,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'r': 0.95,
            'theta': np.pi / 4,
            'spo_m': 20,
            'spo_k_max': 10,
            'epsilon': 1e-7,
            'delta': 0.1,
            'num_check_points': 3,
            'expected_roots': 1
        }

class problem_4(DiophantineProblem):
    @property
    def name(self):
        return "Problem 4 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: x[0]*x[1] - (x[0]-2*x[2])*(x[1]-2*x[2]) - 165,
            lambda x: (x[0]*x[1]**3)/12 - ((x[0]-2*x[2])*(x[1]-2*x[2])**3)/12 - 9369,
            lambda x: ((2 * (x[1]-x[2])**2 * (x[0]-x[2])**2 * x[2]) / (x[1] + x[0] - 2*x[2] + 1e-10)) - 6835
        ]
        return equations 

    def get_info(self):
        domain = [(-40, 40), (-40, 40), (-40, 40)]
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-3,
            'delta': 0.001,
            'spo_m': 100,
            'spo_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 1e-7,
            'num_check_points': 5
        }
        params['expected_roots'] = 6
        return domain, params 

class problemIMW_Fix(DiophantineProblem):
    @property
    def name(self):
        return "Problem project IMW FIX"
    
    def get_equations(self):
        equations = [lambda d: (
            60.0*d[0] + 70.0*d[1] + 30.0*d[2] + 80.0*d[3] + 90.0*d[4] +
            40.0*d[5] + 55.0*d[6] + 65.0*d[7] + 55.0*d[8] + 200.0*d[9] +
            180.0*d[10] + 30.0*d[11] + 80.0*d[12] + 130.0*d[13] + 110.0*d[14] +
            50.0*d[15] + 40.0*d[16] + 250.0*d[17] + 40.0*d[18] + 20.0*d[19] +
            10.0*d[20] + 10.0*d[21] + 30.0*d[22] + 40.0*d[23] + 0.0*d[24] - 38690
        )]
        return equations

    def get_info(self):
        # Hardcoded bounds for each task (d_min, d_max)
        domain = [
            (7, 10), (7, 10), (5, 7), (18, 22), (25, 30),
            (6, 8), (12, 17), (25, 30), (14, 19), (25, 30),
            (20, 30), (4, 5), (15, 20), (25, 30), (25, 30),
            (15, 20), (3, 5), (18, 23), (8, 12), (1, 1),
            (1, 1), (1, 1), (6, 9), (10, 14), (1, 1)
        ]
        params = {
            'm_cluster': 1024,
            'k_cluster': 500,
            'gamma': 0.0001,
            'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
            'spo_m': 2048, 'spo_k_max': 100, 'spo_r': 0.95, 'spo_theta': np.pi/4,
            'num_check_points': 3
        }
        return domain, params

class ProblemIMW(DiophantineProblem):
    @property
    def name(self):
        return "Problem project IMW"
    
    def get_equations(self):
        equations = [
            lambda d: (60.0*d[0] + 70.0*d[1] + 30.0*d[2] + 80.0*d[3] + 90.0*d[4] + 
                      40.0*d[5] + 55.0*d[6] + 65.0*d[7] + 55.0*d[8] + 200.0*d[9] + 
                      180.0*d[10] + 30.0*d[11] + 80.0*d[12] + 130.0*d[13] + 110.0*d[14] + 
                      50.0*d[15] + 40.0*d[16] + 250.0*d[17] + 40.0*d[18] + 20.0*d[19] + 
                      10.0*d[20] + 10.0*d[21] + 30.0*d[22] + 40.0*d[23] + 0.0*d[24]
        )]
        return equations 

    def get_info(self):
        # Hardcoded bounds for each task (d_min, d_max)
        domain = [
            (7, 10), (7, 10), (5, 7), (18, 22), (25, 30), 
            (6, 8), (12, 17), (25, 30), (14, 19), (25, 30), 
            (20, 30), (4, 5), (15, 20), (25, 30), (25, 30), 
            (15, 20), (3, 5), (18, 23), (8, 12), (1, 1), 
            (1, 1), (1, 1), (6, 9), (10, 14), (1, 1)
        ]
        params = {
            'm_cluster': 128,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-7,
            'delta': 0.01,
            'spo_m': 128,
            'spo_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0.2,
            'num_check_points': 1,
        }
        return domain, params

def get_diophantine_problems() -> Dict[int, Callable[[], DiophantineProblem]]:
    """Returns a mapping of IDs to Diophantine problem instances."""
    return {
        1: lambda: DiophantineProblem1(),
        2: lambda: DiophantineProblem3a(),
        3: lambda: DiophantineProblem3b(),
        4: lambda: DiophantineProblem4_4(),
        5: lambda: DiophantineProblem4_5(),
        6: lambda: DiophantineProblem4_6(),
        7: lambda: DiophantineProblem4_7(),
        8: lambda: DiophantineProblem4_8(),
        9: lambda: DiophantineProblem4_9(),
        10: lambda: DiophantineProblem4_10(),
        11: lambda: DiophantineProblem5a(),
        12: lambda: DiophantineProblem6a(),
        13: lambda: DiophantineProblem6b(),
        14: lambda: DiophantineProblem7(),
        15: lambda: DiophantineProblem8(),
        16: lambda: DiophantineProblem9a(),
        17: lambda: DiophantineProblem9b(),
        18: lambda: problem_4(),
        19: lambda: ProblemIMW(),
        20: lambda: problemIMW_Fix()
    }
