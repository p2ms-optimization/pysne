import numpy as np
from typing import Callable, Dict, List, Tuple, Any

def get_problem_set() -> Dict[int, Callable]:
    """"Mengembalikan daftar fungsi pemanggil problem."""
    return{
        1: problem 1,
        2: problem 2,
        3: problem 3,
        4: problem 4,
        5: problem 5,
        6: problem 6,
        7: problem 7
    }

def problem_1():
    equations = [
        lambda x: np.exp(x[0] - x[1]) - np.sin(x[0] + x[1]),
        lambda x: x[0]**2 * x[1]**2 - np.cos(x[0] + x[1])
    ]
    domain = [(-10, 10), (-10, 10)]
    params = {
        'm_cluster': 260, 'gamma': 0.2, 'epsilon': 1e-7, 'delta': 0.01,
        'k_cluster': 10, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 250, 'sdoa_k_max': 250, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 6

def problem_2():
    equations = [
        lambda x: 0.5 * np.sin(x[0] * x[1]) - 0.25 * x[1]/np.pi - 0.5 * x[0],
        lambda x: (1 - 0.25/np.pi) * (np.exp(2*x[0]) - np.e) + np.e * x[1]/np.pi - 2*np.e*x[0]
    ]
    domain = [(-1, 3), (-17, 4)]
    params = {
        'm_cluster': 2000, 'gamma': 0.3, 'epsilon': 1e-7, 'delta': 0.1,
        'k_cluster': 10, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 300, 'sdoa_k_max': 300, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 12

def problem_3():
    equations = [
        lambda x: x[0] + (x[1]**2 * x[3] * x[5])/4 + 0.75,
        lambda x: x[1] + 0.405 * np.exp(1 + x[0]*x[1]) - 1.405,
        lambda x: x[2] - (x[3] * x[5])/2 + 1.5,
        lambda x: x[3] - 0.605 * np.exp(1 - x[2]**2) - 0.395,
        lambda x: x[4] - (x[1] * x[5])/2 + 1.5,
        lambda x: x[5] - x[0] * x[4]
    ]
    domain = [(-5, 5)] * 6
    params = {
        'm_cluster': 1000, 'gamma': 0.1, 'epsilon': 1e-7, 'delta': 0.5,
        'k_cluster': 15, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 420, 'sdoa_k_max': 420, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 2

def problem_4():
    equations = [
        lambda x: x[0]*x[1] - (x[0]-2*x[2])*(x[1]-2*x[2]) - 165,
        lambda x: (x[0]*x[1]**3)/12 - ((x[0]-2*x[2])*(x[1]-2*x[2])**3)/12 - 9369,
        lambda x: ((2 * (x[1]-x[2])**2 * (x[0]-x[2])**2 * x[2]) / (x[1] + x[0] - 2*x[2] + 1e-10)) - 6835
    ]
    domain = [(-40, 40), (-40, 40), (-40, 40)]
    params = {
        'm_cluster': 2000, 'gamma': 1e-3, 'epsilon': 1e-7, 'delta': 0.5,
        'k_cluster': 10, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 500, 'sdoa_k_max': 500, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 6

def problem_5():
    equations = [
        lambda x: 2*x[0] + x[1] + x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + 2*x[1] + x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + x[1] + 2*x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + x[1] + x[2] + 2*x[3] + x[4] - 6,
        lambda x: x[0]*x[1]*x[2]*x[3]*x[4] - 1
    ]
    domain = [(-10, 10)] * 5
    params = {
        'm_cluster': 9800, 'gamma': 0.1, 'epsilon': 5e-4, 'delta': 0.01,
        'k_cluster': 10, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 200, 'sdoa_k_max': 200, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 3

def problem_6():
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
    domain = [(-1, 1)] * 8
    params = {
        'm_cluster': 1500, 'gamma': 0.2, 'epsilon': 1e-6, 'delta': 0.01,
        'k_cluster': 5, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 300, 'sdoa_k_max': 300, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }
    return equations, domain, params, 16

def problem_7():
    """Weierstrass Function."""
    s = 1.1
    lam = 1.5
    N = 20

    def weierstrass(x):
        result = 0.0
        for k in range(1, N+1):
            result += lam**((s-2)*k) * np.sin(lam**k * x[0])
        return result

    equations = [lambda x: weierstrass(x)]
    domain = [(0, 5.05)]

    params = {
        'm_cluster': 2000, 'gamma': 0.9, 'epsilon': 1e-7, 'delta': 0.0001,
        'k_cluster': 50, 'r': 0.95, 'theta': np.pi/4,
        'sdoa_m': 150, 'sdoa_k_max': 150, 'sdoa_r': 0.95, 'sdoa_theta': np.pi/4
    }

    return equations, domain, params, 9