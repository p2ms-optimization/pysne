import numpy as np
from pysne.problems.base import SNEProblem
from pysne.solver import solve_system
 
class SistemSaya(SNEProblem):
    @property
    def name(self):
        return "Contoh Sistem Persamaan Kustom"
 
    def get_equations(self):
        return [
            lambda x: x[0]**2 + x[1]**2 - 4,
            lambda x: x[0] - x[1]
        ]
 
    def get_info(self):
        domain = [(-5, 5), (-5, 5)]
        params = {
            'm_cluster': 512, 'k_cluster': 10, 'gamma': 0.5,
            'sdoa_m': 128, 'sdoa_k_max': 400,
            'r': 0.95, 'theta': np.pi/4,
            'epsilon': 1e-7, 'delta': 0.01
        }
        return domain, params
 
# Instansiasi dan eksekusi -- API pemanggilan identik dengan Skenario 1
prob = SistemSaya()
hasil = solve_system(prob, prob.get_info()[1], verbose=True)
print(hasil['roots'])