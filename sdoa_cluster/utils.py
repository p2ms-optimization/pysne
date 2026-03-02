import numpy as np
from typing import Callable, List

def objective_function(x: np.ndarray, system_of_equations: List[Callable]) -> float:
    """
    Transformasi sistem persamaan ke fungsi objektif:
    F(x) = 1 / (1 + sum|f_i(x)|)
    Akar solusi akan memiliki nilai F(x) mendekati atau sama dengan 1.
    """
    try:
        # Menghitung sigma |f_i(x)|
        sum_of_abs_f = sum(abs(f_i(x)) for f_i in system_of_equations)
        # f(x) = 1 / (1 + sum|f_i(x)|)
        return 1.0 / (1.0 + sum_of_abs_f)
    except Exception:
        # Menghindari pembagian nol atau error komputasi
        return 0.0