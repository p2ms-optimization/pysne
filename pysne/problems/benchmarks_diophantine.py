import numpy as np
from typing import Callable, Dict, List, Tuple
from pysne.problems.base import DiophantineProblem

class Problem1(DiophantineProblem):
    """
    Sistem Persamaan Diophantine:
    
    """
    @property
    def name(self):
        return "Diophantine Problem 1: "

    def get_equations(self):
        return [
            lambda x: ,
            lambda x: 
        ]

    def get_integer_domain(self):
        return [(-10, 10), (-10, 10)]

    def get_params(self):
        return {
            'm_cluster': 100,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-5,
            'delta': 0.5,
            'sdoa_m': 30,
            'sdoa_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'expected_roots': 2
        }


class Problem2(DiophantineProblem):
    """
    Sistem Persamaan Diophantine:
    
    """
    @property
    def name(self):
        return "Diophantine Problem 2: "

    def get_equations(self):
        return [
            lambda x: ,
            lambda x: 
        ]

    def get_integer_domain(self):
        return [(-5, 5), (-5, 5)]

    def get_params(self):
        return {
            'm_cluster': 100,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-5,
            'delta': 0.5,
            'sdoa_m': 30,
            'sdoa_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'expected_roots': 2
        }


def get_diophantine_problems() -> Dict[int, Callable[[], DiophantineProblem]]:
    return {
        1: lambda: Problem1(),
        2: lambda: Problem2()
    }
