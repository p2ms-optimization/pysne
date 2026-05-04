import numpy as np
from pysne.problems.base import BaseProblem



class F1_FiveUnevenPeakTrap(BaseProblem):
    @property
    def name(self):
        return "F1: Five-Uneven-Peak Trap (1D)"

    def g_func(self, x):
        x = np.asarray(x)
        x_vals = x if x.ndim == 1 else x[:, 0]
        
        y = np.zeros_like(x_vals)
        y = np.where((x_vals >= 0) & (x_vals < 2.5), 80*(2.5 - x_vals), y)
        y = np.where((x_vals >= 2.5) & (x_vals < 5.0), 64*(x_vals - 2.5), y)
        y = np.where((x_vals >= 5.0) & (x_vals < 7.5), 64*(7.5 - x_vals), y)
        y = np.where((x_vals >= 7.5) & (x_vals < 12.5), 28*(x_vals - 7.5), y)
        y = np.where((x_vals >= 12.5) & (x_vals < 17.5), 28*(17.5 - x_vals), y)
        y = np.where((x_vals >= 17.5) & (x_vals < 22.5), 32*(x_vals - 17.5), y)
        y = np.where((x_vals >= 22.5) & (x_vals < 27.5), 32*(27.5 - x_vals), y)
        y = np.where((x_vals >= 27.5) & (x_vals <= 30.0), 80*(x_vals - 27.5), y)
        
        return y

    def get_info(self):
        domain = [(0.0, 30.0)]
        params = {
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-7,
            'delta': 0.01,
            'sdoa_m': 100,
            'sdoa_k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 100
        }
        return domain, params

class F2_EqualMaxima(BaseProblem):
    @property
    def name(self):
        return "F2: Equal Maxima (1D)"
    
    def g_func(self, x):
        x = np.asarray(x)
        x_vals = x if x.ndim == 1 else x[:, 0]
        return np.sin(5 * np.pi * x_vals)**6

    def get_info(self):
        domain = [(0.0, 1.0)]
        params = {
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-5,
            'delta': 0.01,
            'sdoa_m': 50,
            'sdoa_k_max': 50,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0.5
        }
        return domain, params


class F3_UnevenDecreasingMaxima(BaseProblem):
    @property
    def name(self):
        return "F3: Uneven Decreasing Maxima (1D)"

    def g_func(self, x):
        x = np.asarray(x)
        x_vals = x if x.ndim == 1 else x[:, 0]
        term1 = np.exp(-2 * np.log(2) * ((x_vals - 0.08) / 0.854)**2)
        x_safe = np.clip(x_vals, 0, None)
        term2 = np.sin(5 * np.pi * (x_safe**0.75 - 0.05))**6
        return term1 * term2

    def get_info(self):
        domain = [(0.0, 1.0)]
        
        # Contoh pengambilan parameter berdasarkan dimensi n
        params = {
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-7,
            'delta': 0.01,
            'sdoa_m': 100,
            'sdoa_k_max': 100,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0.8
        }
        return domain, params

class F4_Himmelblau(BaseProblem):
    def __init__(self, n=2):
        self.n = n
        # Panggil init parent untuk setup n_var dan domain
        super().__init__()

    @property
    def name(self):
        return "F4: Himmelblau (2D)"

    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        return 200 - (x1**2 + x2 - 11)**2 - (x1 + x2**2 - 7)**2

    def get_info(self):
        domain = [(-6.0, 6.0), (-6.0, 6.0)]
        
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 15,
            'epsilon': 1e-5,
            'delta': 0.01,
            'm': 200,
            'k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 150
        }
        return domain, params


class F5_SixHumpCamelBack(BaseProblem):
    def __init__(self, n=2):
        self.n = n
        super().__init__()

    @property
    def name(self):
        return "F5: Six-Hump Camel Back (2D)"

    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return (term1 + term2 + term3)

    def get_info(self):
        domain = [(-1.9, 1.9), (-1.1, 1.1)]
        
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 15,
            'epsilon': 1e-6,
            'delta': 0.5,
            'sdoa_m': 200,
            'sdoa_k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0
        }
        return domain, params


class Problem4(BaseProblem):
    def __init__(self, n=2):
        self.n = n
        # Panggil init parent untuk setup n_var dan domain
        super().__init__()

    @property
    def name(self):
        return f"Problem 4: 2D Vincent Function"

    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        
        # Protect log from negative values when Nelder-Mead probes out of bounds
        # x1 = np.clip(x1, 1e-9, None)
        # x2 = np.clip(x2, 1e-9, None)
        
        term1 = np.sin(10 * np.log(x1))
        term2 = np.sin(10 * np.log(x2))
        return 0.5 * (term1 + term2)

    def get_info(self):
        domain = [(0.25, 10), (0.25, 10)]
        
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-5,
            'delta': 0.01,
            'm': 150,
            'k_max': 150,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0.2
        }
        return domain, params


class F6_Shubert(BaseProblem):
    def __init__(self, n=2):
        self.n = n
        super().__init__()

    @property
    def name(self):
        return "Problem 5: 2D Shubert Function"

    def g_func(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            prod = 1.0
            for i in range(self.n):
                sum_part = 0.0
                for j in range(1, 6):
                    sum_part += j * np.cos((j + 1) * x[i] + j)
                prod *= sum_part
            return -prod
        else:
            prod = np.ones(x.shape[0])
            for i in range(self.n):
                sum_part = np.zeros(x.shape[0])
                for j in range(1, 6):
                    sum_part += j * np.cos((j + 1) * x[:, i] + j)
                prod *= sum_part
            return -prod
        
    def get_info(self):
        domain = [(-10, 10)] * self.n
        
        params = {
            'm_cluster': 3000 if self.n == 2 else 10000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 20,
            'epsilon': 1e-6,
            'delta': 0.5,
            'sdoa_m': 200,
            'sdoa_k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 100.0
        }
        return domain, params

class F7_Vincent(BaseProblem):
    def __init__(self, n=2):
        self.n = n
        # Panggil init parent untuk setup n_var dan domain
        super().__init__()

    @property
    def name(self):
        return f"F7: Vincent ({n}D)"

    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        
        # Protect log from negative values when Nelder-Mead probes out of bounds
        # x1 = np.clip(x1, 1e-9, None)
        # x2 = np.clip(x2, 1e-9, None)
        
        term1 = np.sin(10 * np.log(x1))
        term2 = np.sin(10 * np.log(x2))
        return (1/self.n) * (term1 + term2)

    def get_info(self):
        domain = [(0.25, 10)] * self.n
        
        params = {
            'm_cluster': 2000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 15,
            'epsilon': 1e-5,
            'delta': 0.2,
            'm': 200,
            'k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': 0.8
        }
        return domain, params

class F8_ModifiedRastrigin(BaseProblem):
    
    @property
    def name(self):
        return "F8: Modified Rastrigin - All Global Optima (2D)"

    def g_func(x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        k1, k2 = 3, 4
        term1 = 10 + 9 * np.cos(2 * np.pi * k1 * x1)
        term2 = 10 + 9 * np.cos(2 * np.pi * k2 * x2)
        return - (term1 + term2)
        
    def get_info(self):
        domain =  [(0, 1), (0, 1)]
        
        params = {
            'm_cluster': 2000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 15,
            'epsilon': 1e-5,
            'delta': 0.01,
            'm': 200,
            'k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -5.0
        }
        return domain, params

def get_gecco_problems():
    """Dictionary pemanggil problem."""
    return {
        1: lambda: F1_FiveUnevenPeakTrap(),
        2: lambda: F2_EqualMaxima(),
        3: lambda: F3_UnevenDecreasingMaxima(),
        4: lambda: F4_Himmelblau(),
        5: lambda: F5_SixHumpCamelBack(),
        6: lambda: F6_Shubert(),
        7: lambda: F7_Vincent(),
        8: lambda: F8_ModifiedRastrigin()
    }