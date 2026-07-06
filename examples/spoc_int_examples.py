"""
examples/spoc_int_examples.py
==============================
Semua contoh problem dari bab6 (domain integer / Diophantine) yang digunakan di Tugas Akhir.
Setiap fungsi problem mengembalikan (equations, domain, param, expected_solutions).

Cara pakai
----------
    python examples/spoc_int_examples.py

Atau dari script lain:
    from examples.spoc_int_examples import run_all, run_one, PROBLEMS
    solutions = run_one("problem_1")
"""

import numpy as np
from pysne import SPOC_int


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINISI PROBLEM  (Sumarti et al., 2023 + Tugas Akhir)
# ══════════════════════════════════════════════════════════════════════════════

def problem_1():
    """15x + 11y = 12  (Linear, 2 var)"""
    equations = [lambda v: 15*v[0] + 11*v[1] - 12]
    domain    = [(-50, 50), (-50, 50)]
    param = {
        'm_cluster': 350, 'k_cluster': 10, 'gamma': 0.01,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 30, 'k_max': 10, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 7


def problem_2a():
    """Σ xi² = 720  (9 variables)"""
    equations = [lambda v: sum(v[i]**2 for i in range(9)) - 720]
    domain    = [(1, 26)] * 9
    param = {
        'm_cluster': 23000, 'k_cluster': 45, 'gamma': 0.1,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.971, 'theta': 9*np.pi/80,
        'm_sdoa': 70, 'k_max': 60, 'r_sdoa': 0.973, 'theta_sdoa': 29*np.pi/80,
        'num_check_points': 1,
    }
    return equations, domain, param, 6


def problem_2b():
    """Σ xi² = 956  (10 variables)"""
    equations = [lambda v: sum(v[i]**2 for i in range(10)) - 956]
    domain    = [(1, 26)] * 10
    param = {
        'm_cluster': 53500, 'k_cluster': 40, 'gamma': 0.1,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.966, 'theta': 5*np.pi/16,
        'm_sdoa': 190, 'k_max': 75, 'r_sdoa': 0.978, 'theta_sdoa': 5*np.pi/16,
        'num_check_points': 3,
    }
    return equations, domain, param, 6


def problem_3a():
    """x1³ + x2³ = 1008"""
    equations = [lambda v: v[0]**3 + v[1]**3 - 1008]
    domain    = [(1, 10), (1, 10)]
    param = {
        'm_cluster': 200, 'k_cluster': 20, 'gamma': 0.1,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 50, 'k_max': 50, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 1


def problem_3b():
    """x1⁹ + x2⁹ = 1000019683"""
    equations = [lambda v: v[0]**9 + v[1]**9 - 1000019683]
    domain    = [(1, 10), (1, 10)]
    param = {
        'm_cluster': 100, 'k_cluster': 20, 'gamma': 0.1,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 50, 'k_max': 50, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 1


def problem_4_1():
    """Markoff-Hurwitz (n=3, k=1): x1²+x2²+x3² = x1·x2·x3"""
    equations = [lambda v: v[0]**2 + v[1]**2 + v[2]**2 - v[0]*v[1]*v[2]]
    domain    = [(1, 400), (1, 200), (1, 20)]
    param = {
        'm_cluster': 200, 'k_cluster': 75, 'gamma': 1e-7,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.969, 'theta': np.pi/16,
        'm_sdoa': 90, 'k_max': 70, 'r_sdoa': 0.982, 'theta_sdoa': 37*np.pi/80,
        'num_check_points': 3,
    }
    return equations, domain, param, 7


def problem_4_2():
    """Markoff-Hurwitz (n=3, k=3): x1²+x2²+x3² = 3·x1·x2·x3"""
    equations = [lambda v: v[0]**2 + v[1]**2 + v[2]**2 - 3*v[0]*v[1]*v[2]]
    domain    = [(1, 500), (1, 100), (1, 50)]
    param = {
        'm_cluster': 8100, 'k_cluster': 36, 'gamma': 1e-4,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.972, 'theta': np.pi/16,
        'm_sdoa': 395, 'k_max': 26, 'r_sdoa': 0.971, 'theta_sdoa': np.pi/16,
        'num_check_points': 1,
    }
    return equations, domain, param, 11


def problem_4_3():
    """Markoff-Hurwitz (n=4, k=1): x1²+x2²+x3²+x4² = x1·x2·x3·x4"""
    equations = [lambda v: v[0]**2+v[1]**2+v[2]**2+v[3]**2 - v[0]*v[1]*v[2]*v[3]]
    domain    = [(1, 300), (1, 50), (1, 10), (1, 5)]
    param = {
        'm_cluster': 2500, 'k_cluster': 75, 'gamma': 0.0001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.9284, 'theta': np.pi/16,
        'm_sdoa': 100, 'k_max': 35, 'r_sdoa': 0.9639, 'theta_sdoa': 3*np.pi/16,
        'num_check_points': 3,
    }
    return equations, domain, param, 5


def problem_4_4():
    """Markoff-Hurwitz (n=4, k=4): x1²+x2²+x3²+x4² = 4·x1·x2·x3·x4"""
    equations = [lambda v: v[0]**2+v[1]**2+v[2]**2+v[3]**2 - 4*v[0]*v[1]*v[2]*v[3]]
    domain    = [(1, 200), (1, 20), (1, 5), (1, 5)]
    param = {
        'm_cluster': 3000, 'k_cluster': 75, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.973, 'theta': np.pi/16,
        'm_sdoa': 50, 'k_max': 35, 'r_sdoa': 0.968, 'theta_sdoa': 33*np.pi/80,
        'num_check_points': 2,
    }
    return equations, domain, param, 5


def problem_4_5():
    """Markoff-Hurwitz (n=5, k=4): Σxi² = 4·Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(5)) - 4*v[0]*v[1]*v[2]*v[3]*v[4]]
    domain    = [(1, 100), (1, 50), (1, 5), (1, 5), (1, 5)]
    param = {
        'm_cluster': 3500, 'k_cluster': 30, 'gamma': 0.0001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.975, 'theta': np.pi/4,
        'm_sdoa': 150, 'k_max': 30, 'r_sdoa': 0.975, 'theta_sdoa': np.pi/6,
        'num_check_points': 1,
    }
    return equations, domain, param, 5


def problem_4_6():
    """Markoff-Hurwitz (n=6, k=3): Σxi² = 3·Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(6)) - 3*v[0]*v[1]*v[2]*v[3]*v[4]*v[5]]
    domain    = [(1, 50), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3)]
    param = {
        'm_cluster': 1000, 'k_cluster': 10, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 50, 'k_max': 10, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 5


def problem_4_7():
    """Markoff-Hurwitz (n=7, k=2): Σxi² = 2·Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(7)) - 2*v[0]*v[1]*v[2]*v[3]*v[4]*v[5]*v[6]]
    domain    = [(1, 60), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3), (1, 3)]
    param = {
        'm_cluster': 1000, 'k_cluster': 15, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 250, 'k_max': 15, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 5


def problem_4_8():
    """Markoff-Hurwitz (n=8, k=1): Σxi² = Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(8)) - v[0]*v[1]*v[2]*v[3]*v[4]*v[5]*v[6]*v[7]]
    domain    = [(1, 50), (1, 10), (1, 5), (1, 3), (1, 3), (1, 3), (1, 3), (1, 3)]
    param = {
        'm_cluster': 2000, 'k_cluster': 20, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 100, 'k_max': 20, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 3


def problem_4_9():
    """Markoff-Hurwitz (n=9, k=6): Σxi² = 6·Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(9)) - 6*v[0]*v[1]*v[2]*v[3]*v[4]*v[5]*v[6]*v[7]*v[8]]
    domain    = [(1, 30), (1, 5), (1, 3), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
    param = {
        'm_cluster': 3500, 'k_cluster': 20, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.975, 'theta': 37*np.pi/80,
        'm_sdoa': 10, 'k_max': 45, 'r_sdoa': 0.967, 'theta_sdoa': 13*np.pi/80,
        'num_check_points': 2,
    }
    return equations, domain, param, 4


def problem_4_10():
    """Markoff-Hurwitz (n=10, k=1): Σxi² = Πxi"""
    equations = [lambda v: sum(v[i]**2 for i in range(10)) - v[0]*v[1]*v[2]*v[3]*v[4]*v[5]*v[6]*v[7]*v[8]*v[9]]
    domain    = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
    param = {
        'm_cluster': 2750, 'k_cluster': 55, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.989, 'theta': 21*np.pi/80,
        'm_sdoa': 100, 'k_max': 40, 'r_sdoa': 0.962, 'theta_sdoa': 29*np.pi/80,
        'num_check_points': 1,
    }
    return equations, domain, param, 3


def problem_5a():
    """Ramanujan-Nagell: x² + 7 = y^n  (3 vars: x, y, n)"""
    equations = [lambda v: v[0]**2 + 7 - v[1]**v[2]]
    domain    = [(1, 500), (1, 50), (3, 50)]
    param = {
        'm_cluster': 4000, 'k_cluster': 40, 'gamma': 1e-5,
        'epsilon': 0.001, 'delta': 0.1, 'r': 0.975, 'theta': np.pi/16,
        'm_sdoa': 150, 'k_max': 30, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/16,
        'num_check_points': 3,
    }
    return equations, domain, param, 7


def problem_5b():
    """x² + 11^b = y³  (3 vars: x, y, b)"""
    equations = [lambda v: v[0]**2 + 11**v[2] - v[1]**3]
    domain    = [(1, 15000), (1, 600), (1, 100)]
    param = {
        'm_cluster': 20000, 'k_cluster': 35, 'gamma': 1e-77,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.969, 'theta': np.pi/8,
        'm_sdoa': 280, 'k_max': 70, 'r_sdoa': 0.956, 'theta_sdoa': np.pi/4,
        'num_check_points': 5,
    }
    return equations, domain, param, 4


def problem_6a():
    """x² + 2^a · 11^b = y³  (4 vars: x, y, a, b)"""
    equations = [lambda v: v[0]**2 + (2**v[2])*(11**v[3]) - v[1]**3]
    domain    = [(0, 20)] * 4
    param = {
        'm_cluster': 500, 'k_cluster': 20, 'gamma': 0.001,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 100, 'k_max': 20, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 8


def problem_6b():
    """x² + 2^a · 11^b = y⁴  (4 vars: x, y, a, b)"""
    equations = [lambda v: v[0]**2 + (2**v[2])*(11**v[3]) - v[1]**4]
    domain    = [(0, 20)] * 4
    param = {
        'm_cluster': 15000, 'k_cluster': 20, 'gamma': 0.01,
        'epsilon': 1e-7, 'delta': 0.001, 'r': 0.95, 'theta': np.pi/3,
        'm_sdoa': 100, 'k_max': 20, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 5


def problem_7():
    """2^k + 3x² = y³  (3 vars: x, y, k)"""
    equations = [lambda v: 2**v[2] + 3*(v[0]**2) - v[1]**3]
    domain    = [(0, 50)] * 3
    param = {
        'm_cluster': 10000, 'k_cluster': 45, 'gamma': 0.1,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.981, 'theta': 33*np.pi/80,
        'm_sdoa': 350, 'k_max': 20, 'r_sdoa': 0.964, 'theta_sdoa': 13*np.pi/80,
        'num_check_points': 2,
    }
    return equations, domain, param, 9


def problem_8():
    """5^x1 + 5^x2 = 3^x3 + 7^x4  (4 vars)"""
    equations = [lambda v: 5**v[0] + 5**v[1] - (3**v[2] + 7**v[3])]
    domain    = [(-1, 10)] * 4
    param = {
        'm_cluster': 900, 'k_cluster': 30, 'gamma': 0.1,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 50, 'k_max': 25, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 2,
    }
    return equations, domain, param, 9


def problem_9a():
    """Pell: x²-24y²=1 & y²-2z²=1  (3 vars)"""
    equations = [
        lambda v: v[0]**2 - 24*(v[1]**2) - 1,
        lambda v: v[1]**2 - 2*(v[2]**2) - 1,
    ]
    domain = [(400, 500), (1, 100), (1, 100)]
    param = {
        'm_cluster': 400, 'k_cluster': 20, 'gamma': 0.0001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 100, 'k_max': 20, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 1


def problem_9b():
    """Pell: x²-24y²=1 & y²-11z²=1  (3 vars)"""
    equations = [
        lambda v: v[0]**2 - 24*(v[1]**2) - 1,
        lambda v: v[1]**2 - 11*(v[2]**2) - 1,
    ]
    domain = [(1, 75)] * 3
    param = {
        'm_cluster': 1500, 'k_cluster': 20, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 20, 'k_max': 10, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 1


def problem_10():
    """Linear System (7 variables)"""
    equations = [
        lambda v: v[0] - 1,
        lambda v: 3*v[0] + v[1] - 6,
        lambda v: 4*v[0] + 3*v[1] + v[2] + v[4] - 15,
        lambda v: 3*v[0] + 4*v[1] + 3*v[2] + v[3] + v[4] + v[5] - 20,
        lambda v: 3*v[1] + 4*v[2] + 3*v[3] + v[4] + v[5] + v[6] - 15,
        lambda v: 3*v[2] + 4*v[3] + v[5] + v[6] - 6,
        lambda v: 3*v[3] + v[6] - 1,
    ]
    domain = [(-10, 10)] * 7
    param = {
        'm_cluster': 150, 'k_cluster': 25, 'gamma': 0.01,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.971, 'theta': 21*np.pi/80,
        'm_sdoa': 440, 'k_max': 20, 'r_sdoa': 0.955, 'theta_sdoa': 9*np.pi/80,
        'num_check_points': 1,
    }
    return equations, domain, param, 1


def problem_11():
    """Nonlinear System (6 variables)"""
    equations = [
        lambda v: 5*v[0] + 10*v[1] - 5*v[2] + v[4]**3 + 8*v[5] - 1772,
        lambda v: 3*v[0] + 18*v[2] - 5*v[4] + 17*v[5] - 153,
        lambda v: 6*v[0] + v[2] - 99*v[1] + (15*v[5])**2 - 1772,
        lambda v: -v[0] + 5*v[1] + 8*v[2] - 6*v[3] + 15*v[4] + 10*v[5] - 277,
        lambda v: (v[0]+v[1])**2 - 7*v[2] + 5*v[3] + 12*v[4] - 8*v[5] - 150,
        lambda v: v[1] + 5*v[2] - 3*v[4] - v[5] - 4,
    ]
    domain = [(1, 20)] * 6
    param = {
        'm_cluster': 200, 'k_cluster': 20, 'gamma': 0.001,
        'epsilon': 1e-7, 'delta': 0.1, 'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 50, 'k_max': 40, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
        'num_check_points': 3,
    }
    return equations, domain, param, 1


def problem_12():
    """Complex Nonlinear System (10 variables)"""
    equations = [
        lambda v: v[0]**2 - 2*(v[1]+v[3])**3 + v[4] - 3*v[5] - v[6] + 4*v[8] + 15*v[9] + 24,
        lambda v: 2*v[0] + (v[1]+3*v[3])**3 + (5*v[6])**2 - 6*v[7] + v[8] - 9*v[9] - 31,
        lambda v: 3*v[0] - (2*v[1])**2 + 10*v[2] - 9*v[3] + 3*v[4] + v[5] - 2*v[6] - 8*v[7] + 12*v[8] - 5*v[9] + 25,
        lambda v: 5*v[0] + 2*v[1] - 8*v[3] - 3*v[4] + 4*v[5] + v[6] - v[8] - 23,
        lambda v: v[0] - v[2] + 2*v[4] - v[6] - v[8] + 3,
        lambda v: v[1] + (2*v[3])**2 - 6*v[5] - v[7] + 2*v[9] - 8,
        lambda v: 3*v[0] + 2*v[1] - 5*v[2] - v[3]**4 - 2*v[4] + v[5] + 4*v[6] - 10*v[7] + 8*v[8] + 9,
        lambda v: v[0] - 3*v[1] + 4*v[3] + v[5] - 6*v[6] + v[7] - 2*v[8] + 16,
        lambda v: (2*v[0]+v[1])**2 + 3*v[2] - 10*v[4] - (v[5]+3*v[6])**3 - v[7] - 6*v[8] - 27,
    ]
    domain = [(0, 10)] * 10
    param = {
        'm_cluster': 50000, 'k_cluster': 15, 'gamma': 0.01,
        'epsilon': 1e-5, 'delta': 0.01, 'r': 0.957, 'theta': 29*np.pi/80,
        'm_sdoa': 475, 'k_max': 40, 'r_sdoa': 0.983, 'theta_sdoa': 5*np.pi/16,
        'num_check_points': 2,
    }
    return equations, domain, param, 1


# Registry
PROBLEMS = {
    "problem_1"   : (problem_1,    "15x+11y=12 (linear, 2 var, ~7 sol)"),
    "problem_2a"  : (problem_2a,   "Σxi²=720 (9 var, ~6 sol)"),
    "problem_2b"  : (problem_2b,   "Σxi²=956 (10 var, ~6 sol)"),
    "problem_3a"  : (problem_3a,   "x1³+x2³=1008 (1 sol)"),
    "problem_3b"  : (problem_3b,   "x1⁹+x2⁹=1000019683 (1 sol)"),
    "problem_4_1" : (problem_4_1,  "Markoff-Hurwitz n=3 k=1 (7 sol)"),
    "problem_4_2" : (problem_4_2,  "Markoff-Hurwitz n=3 k=3 (11 sol)"),
    "problem_4_3" : (problem_4_3,  "Markoff-Hurwitz n=4 k=1 (5 sol)"),
    "problem_4_4" : (problem_4_4,  "Markoff-Hurwitz n=4 k=4 (5 sol)"),
    "problem_4_5" : (problem_4_5,  "Markoff-Hurwitz n=5 k=4 (5 sol)"),
    "problem_4_6" : (problem_4_6,  "Markoff-Hurwitz n=6 k=3 (5 sol)"),
    "problem_4_7" : (problem_4_7,  "Markoff-Hurwitz n=7 k=2 (5 sol)"),
    "problem_4_8" : (problem_4_8,  "Markoff-Hurwitz n=8 k=1 (3 sol)"),
    "problem_4_9" : (problem_4_9,  "Markoff-Hurwitz n=9 k=6 (4 sol)"),
    "problem_4_10": (problem_4_10, "Markoff-Hurwitz n=10 k=1 (3 sol)"),
    "problem_5a"  : (problem_5a,   "Ramanujan-Nagell (7 sol)"),
    "problem_5b"  : (problem_5b,   "x²+11^b=y³ (4 sol)"),
    "problem_6a"  : (problem_6a,   "x²+2^a·11^b=y³ (8 sol)"),
    "problem_6b"  : (problem_6b,   "x²+2^a·11^b=y⁴ (5 sol)"),
    "problem_7"   : (problem_7,    "2^k+3x²=y³ (9 sol)"),
    "problem_8"   : (problem_8,    "5^x1+5^x2=3^x3+7^x4 (9 sol)"),
    "problem_9a"  : (problem_9a,   "Pell (p=2, 1 sol)"),
    "problem_9b"  : (problem_9b,   "Pell (p=11, 1 sol)"),
    "problem_10"  : (problem_10,   "Linear 7-var system (1 sol)"),
    "problem_11"  : (problem_11,   "Nonlinear 6-var system (1 sol)"),
    "problem_12"  : (problem_12,   "Complex nonlinear 10-var system (1 sol)"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_one(name: str, verbose: bool = True):
    """
    Jalankan satu problem berdasarkan nama.

    Parameters
    ----------
    name    : str  — kunci dari PROBLEMS, mis. "problem_1"
    verbose : bool — tampilkan output (default True)

    Returns
    -------
    list of tuple
    """
    if name not in PROBLEMS:
        raise ValueError(f"Problem '{name}' tidak ditemukan. Pilihan: {list(PROBLEMS)}")
    func, desc = PROBLEMS[name]
    equations, domain, param, expected = func()
    if verbose:
        print(f"\n{'='*70}")
        print(f"Example: {name}  —  {desc}")
        print(f"Expected solutions: {expected}")
        print(f"{'='*70}")
    return SPOC_int.solve(equations, domain, param=param, verbose=verbose)


def run_all(verbose: bool = True):
    """
    Jalankan semua problem secara berurutan.

    Returns
    -------
    dict {name: solutions}
    """
    results = {}
    print("\n" + "="*70)
    print("SPOC_int — RUNNING ALL EXAMPLES (Integer Domain)")
    print("="*70)
    for name, (func, desc) in PROBLEMS.items():
        equations, domain, param, expected = func()
        print(f"\n▶ {name}: {desc}  (expected: {expected})")
        sols = SPOC_int.solve(equations, domain, param=param, verbose=verbose)
        results[name] = sols
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, (func, desc) in PROBLEMS.items():
        _, _, _, expected = func()
        found = len(results.get(name, []))
        status = "✓" if found >= max(1, int(expected * 0.8)) else "✗"
        print(f"  {status}  {name:<14}  found {found}/{expected}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — demo interaktif
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SPOC_int — Integer-Domain Solver  |  Demo")
    print("=" * 70)
    print("\nPilih mode:")
    print("  1. Jalankan satu contoh problem")
    print("  2. Jalankan semua contoh problem")
    print("  3. Masukkan sistem persamaan sendiri")

    mode = input("\nMasukkan pilihan (1/2/3): ").strip()

    if mode == "1":
        print("\nDaftar problem:")
        for k, (_, desc) in PROBLEMS.items():
            print(f"  {k}: {desc}")
        name = input("\nNama problem (mis. problem_1): ").strip()
        run_one(name)

    elif mode == "2":
        run_all()

    elif mode == "3":
        print("\n--- Input Sistem Persamaan Sendiri ---")
        print("Contoh: untuk f(x) = 15*x[0] + 11*x[1] - 12, ketik: 15*x[0] + 11*x[1] - 12")
        n_eq = int(input("Jumlah persamaan: "))
        equations = []
        for i in range(n_eq):
            expr = input(f"  f{i+1}(x) = ").strip()
            equations.append(eval(f"lambda x: {expr}"))

        n_dim = int(input("Jumlah dimensi (variabel): "))
        domain = []
        for i in range(n_dim):
            lo = int(input(f"  Domain x[{i}] — batas bawah (integer): "))
            hi = int(input(f"  Domain x[{i}] — batas atas  (integer): "))
            domain.append((lo, hi))

        print("\nParameter (tekan Enter untuk pakai default):")
        param = {}
        keys = [
            ('m_cluster',        int,   300),
            ('k_cluster',        int,    50),
            ('gamma',          float,   1e-4),
            ('epsilon',        float,   1e-5),
            ('delta',          float,   0.01),
            ('r',              float,   0.95),
            ('theta',          float,   np.pi/4),
            ('m_sdoa',           int,    50),
            ('k_max',            int,    50),
            ('r_sdoa',         float,   0.988),
            ('theta_sdoa',     float,   np.pi/4),
            ('num_check_points', int,     1),
        ]
        for key, cast, default in keys:
            raw = input(f"  {key} [{default}]: ").strip()
            param[key] = cast(raw) if raw else default

        solutions = SPOC_int.solve(equations, domain, param=param)
        print(f"\nSelesai. Ditemukan {len(solutions)} solusi.")

    else:
        print("Pilihan tidak valid.")
