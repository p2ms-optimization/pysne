import numpy as np
from typing import Callable, Dict, List, Tuple, Any
from pysne.problems.base import SNEProblem
from pysne.utils import objective_function




def get_problem_set() -> Dict[int, Callable]:
    """Returns a dictionary of problem caller functions."""
    return{
        1: lambda: problem_1(),
        2: lambda: problem_2(),
        3: lambda: problem_3(),
        4: lambda: problem_4(),
        5: lambda: problem_5(),
        6: lambda: problem_6(),
        7: lambda: problem_7()
    }

class problem_1(SNEProblem):
    @property
    def name(self):
        return "Problem 1 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: np.exp(x[0] - x[1]) - np.sin(x[0] + x[1]),
            lambda x: x[0]**2 * x[1]**2 - np.cos(x[0] + x[1])
        ]
        return equations 

    def get_info(self):
        domain = [(-10, 10), (-10, 10)]
        params = {
            'm_cluster': 250,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 10,
            'epsilon': 1e-7,
            'delta': 0.01,
            'sdoa_m': 250,
            'sdoa_k_max': 270,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.2
        }
        params['expected_roots'] = 6
        return domain, params


class problem_2(SNEProblem):
    @property
    def name(self):
        return "Problem 2 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: 0.5 * np.sin(x[0] * x[1]) - 0.25 * x[1]/np.pi - 0.5 * x[0],
            lambda x: (1 - 0.25/np.pi) * (np.exp(2*x[0]) - np.e) + np.e * x[1]/np.pi - 2*np.e*x[0]
        ]
        return equations 

    def get_info(self):
        domain = [(-1, 3), (-17, 4)]
        params = {
            'm_cluster': 2000,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 10,
            'epsilon': 1e-7,
            'delta': 0.1,
            'sdoa_m': 300,
            'sdoa_k_max': 300,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.3
        }
        params['expected_roots'] = 12
        return domain, params 

class problem_3(SNEProblem):
    @property
    def name(self):
        return "Problem 3 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: x[0] + (x[1]**2 * x[3] * x[5])/4 + 0.75,
            lambda x: x[1] + 0.405 * np.exp(1 + x[0]*x[1]) - 1.405,
            lambda x: x[2] - (x[3] * x[5])/2 + 1.5,
            lambda x: x[3] - 0.605 * np.exp(1 - x[2]**2) - 0.395,
            lambda x: x[4] - (x[1] * x[5])/2 + 1.5,
            lambda x: x[5] - x[0] * x[4]
        ]
        return equations 

    def get_info(self):
        domain = [(-5, 5)] * 6
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 15,
            'epsilon': 1e-7,
            'delta': 0.5,
            'sdoa_m': 420,
            'sdoa_k_max': 420,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.1
        }
        params['expected_roots'] = 2
        return domain, params 

class problem_4(SNEProblem):
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
            'm_cluster': 2000,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 10,
            'epsilon': 1e-3,
            'delta': 0.001,
            'sdoa_m': 500,
            'sdoa_k_max': 500,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 1e-7
        }
        params['expected_roots'] = 6
        return domain, params 

class problem_5(SNEProblem):
    @property
    def name(self):
        return "Problem 5 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: 2*x[0] + x[1] + x[2] + x[3] + x[4] - 6,
            lambda x: x[0] + 2*x[1] + x[2] + x[3] + x[4] - 6,
            lambda x: x[0] + x[1] + 2*x[2] + x[3] + x[4] - 6,
            lambda x: x[0] + x[1] + x[2] + 2*x[3] + x[4] - 6,
            lambda x: x[0]*x[1]*x[2]*x[3]*x[4] - 1
        ]
        return equations 

    def get_info(self):
        domain = [(-10, 10)] * 5
        params = {
            'm_cluster': 9800,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 10,
            'epsilon': 5e-4,
            'delta': 0.01,
            'sdoa_m': 200,
            'sdoa_k_max': 200,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.1
        }
        params['expected_roots'] = 3
        return domain, params

class problem_6(SNEProblem):
    @property
    def name(self):
        return "Problem 6 Benchmark System Non-linear Equation"

    def get_equations(self):
        equations = [
            lambda x: 4.731e-3*x[0]*x[2] - 0.3578*x[1]*x[2] - 0.1238*x[0] + x[6] - 1.637e-3*x[1] - 0.9338*x[3] - 0.3571,
            lambda x: 0.2238*x[0]*x[2] + 0.7623*x[1]*x[2] + 0.2638*x[0] - x[6] - 0.07745*x[1] - 0.6734*x[3] - 0.6022,
            lambda x: x[5]*x[7] + 0.3578*x[0] + 4.731e-3*x[1],
            lambda x: -0.7623*x[0] + 0.2238*x[1] + 0.3461,
            lambda x: x[0]**2 + x[1]**2 - 1,
            lambda x: x[2]**2 + x[3]**2 - 1,
            lambda x: x[4]**2 + x[5]**2 - 1,
            lambda x: x[6]**2 + x[7]**2 - 1
        ]
        return equations 

    def get_info(self):
        domain = [(-1, 1)] * 8
        params = {
            'm_cluster': 1500,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 5,
            'epsilon': 1e-6,
            'delta': 0.01,
            'sdoa_m': 300,
            'sdoa_k_max': 300,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.2
        }
        params['expected_roots'] = 16
        return domain, params 

class problem_7(SNEProblem):
    @property
    def name(self):
        return "Problem 7 Benchmark System Non-linear Equation Weierstrass"

    def get_equations(self):
        s = 1.1
        lam = 1.5
        N = 20

        def weierstrass(x):
            result = 0.0
            for k in range(1, N+1):
                result += lam**((s-2)*k) * np.sin(lam**k * x[0])
            return result

        equations = [lambda x: weierstrass(x)]

        return equations 

    def get_info(self):
        domain = [(0, 5.05)]
        params = {
            'm_cluster': 2000,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 50,
            'epsilon': 1e-7,
            'delta': 0.0001,
            'sdoa_m': 150,
            'sdoa_k_max': 150,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.9
        }
        params['expected_roots'] = 9
        return domain, params