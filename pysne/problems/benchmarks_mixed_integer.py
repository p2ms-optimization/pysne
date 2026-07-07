import numpy as np
from pysne.problems.base import MixedIntegerProblem


class SpeedReducerProblem(MixedIntegerProblem):
    """
    Speed Reducer Design Optimization Problem (Golinski's Speed Reducer).

    7 variabel desain (x1..x7), x3 (jumlah gigi) harus integer, sisanya kontinu.
    11 fungsi kendala g_i(x) <= 0. Diselesaikan dengan metode fungsi penalti:

        F(x) = f(x) + M * sum(max(0, g_i(x)))   ,  M = 1e15

    Referensi: benchmark klasik constrained engineering design optimization,
    solusi optimum global (literature): x* = [3.5, 0.7, 17, 7.3, 7.8, 3.350214, 5.286683],
    f(x*) = 2996.348165.
    """

    @property
    def name(self):
        return "Speed Reducer Design Optimization Problem"

    def get_raw_domain(self):
        return [
            (2.6, 3.6),   # x1
            (0.7, 0.8),   # x2
            (17, 28),     # x3 (integer)
            (7.3, 8.3),   # x4
            (7.8, 8.3),   # x5
            (2.9, 3.9),   # x6
            (5.0, 5.5),   # x7
        ]

    def get_integer_dims(self):
        return [2]  # x3 (indeks ke-2, 0-based)

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        x1, x2, x3, x4, x5, x6, x7 = (x[..., i] for i in range(7))
        return (
            0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )

    def constraints(self, x):
        x = np.asarray(x, dtype=float)
        x1, x2, x3, x4, x5, x6, x7 = (x[..., i] for i in range(7))

        g1 = 27.0 / (x1 * x2**2 * x3) - 1.0
        g2 = 397.5 / (x1 * x2**2 * x3**2) - 1.0
        g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0
        g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0
        g5 = (1.0 / (110.0 * x6**3)) * np.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6) - 1.0
        g6 = (1.0 / (85.0 * x7**3)) * np.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6) - 1.0
        g7 = (x2 * x3) / 40.0 - 1.0
        g8 = (5.0 * x2) / x1 - 1.0
        g9 = x1 / (12.0 * x2) - 1.0
        g10 = (1.5 * x6 + 1.9) / x4 - 1.0
        g11 = (1.1 * x7 + 1.9) / x5 - 1.0

        return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]

    def get_params(self):
        return {
            # Parameter Clustering
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi / 4,
            'k_cluster': 100,
            'gamma': 0.5,
            # Parameter SDOA (sesuai setup di slide: m=30000, k_max=1000, r=0.99, theta=pi/32)
            'sdoa_m': 50000,
            'sdoa_k_max': 3000,
            'r': 0.999,
            'theta': np.pi / 32,
            # Parameter Seleksi
            'epsilon': 1e-1,
            'delta': 0.0001,
            # Koefisien penalti
            'M': 1e15,
        }


def get_mixed_integer_problems():
    """Dictionary pemanggil problem mixed-integer."""
    return {
        1: lambda: SpeedReducerProblem(),
    }