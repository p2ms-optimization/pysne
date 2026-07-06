import numpy as np
from pysne.problems.base import MultimodalProblem
from pysne.solver import solve_system
 
class FungsiSaya(MultimodalProblem):
    @property
    def name(self):
        return "Contoh Fungsi Kustom: Paraboloid Terbalik"
 
    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        return -(x1**2 + x2**2)
 
    def get_info(self):
        domain = [(-5, 5), (-5, 5)]
        params = {
            'm_cluster': 500, 'k_cluster': 10, 'gamma': -float('inf'),
            'sdoa_m': 100, 'sdoa_k_max': 100,
            'r': 0.95, 'theta': np.pi/4,
            'epsilon': 1e-5, 'delta': 0.1,
            'num_check_points': 2
        }
        return domain, params
 
# Instansiasi dan eksekusi -- API identik dengan fungsi benchmark bawaan
prob = FungsiSaya()
hasil = solve_system(prob, prob.get_info()[1], verbose=True)
print(hasil['optimals'])