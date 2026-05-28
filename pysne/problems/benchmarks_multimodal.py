import numpy as np
from pysne.problems.base import MultimodalProblem

class Problem1(MultimodalProblem):
    @property
    def name(self):
        return "Problem 1: 2D Second Minima Function"

    def g_func(self, x):
        x = np.asarray(x)
        x0 = x[0] if x.ndim == 1 else x[:, 0]
        x1 = x[1] if x.ndim == 1 else x[:, 1]
        val_x = 0.5 * (x0**4 - 16*x0**2 + 5*x0)
        val_y = 0.5 * (x1**4 - 16*x1**2 + 5*x1)
        return (val_x + val_y) 

    def get_info(self):
        domain = [(-4, 4), (-4, 4)]
        params = {
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-7,
            'delta': 0.1,
            'sdoa_m': 200,
            'sdoa_k_max': 230,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf')
        }
        return domain, params

class Problem2(MultimodalProblem):
    @property
    def name(self):
        return "Problem 2: Six Hump Camel Back Function"
    
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
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-5,
            'delta': 0.1,
            'sdoa_m': 50,
            'sdoa_k_max': 250,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf'),
            'num_check_points': 2

        }
        return domain, params


class Problem3(MultimodalProblem):
    def __init__(self, n=2):
        self.n = n
        # Panggil init parent untuk setup n_var dan domain
        super().__init__()

    @property
    def name(self):
        return f"Problem 3: {self.n}D Rastrigin Function"

    def g_func(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
        else:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)

    def get_info(self):
        domain = [(-1, 1)] * self.n
        
        # Contoh pengambilan parameter berdasarkan dimensi n
        params = {
            'm_cluster': 500 * (1 if self.n == 2 else 1 if self.n == 3 else self.n),
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10 if self.n != 3 else 10,
            'epsilon': 1e-6 if self.n == 2 else 1e-5,
            'delta': 0.1,
            'sdoa_m': 200,
            'sdoa_k_max': 300,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf'),
            'num_check_points': 1 * (1 if self.n == 2 else 2 if self.n == 3 else self.n) 
        }
        return domain, params

class Problem4(MultimodalProblem):
    def __init__(self, n=2):
        self.n = n
        # Panggil init parent untuk setup n_var dan domain
        super().__init__()

    @property
    def name(self):
        return f"Problem 4: 2D Vincent Function"

    @property
    def optima_type(self):
        return "max"

    def g_func(self, x):
        x = np.asarray(x)
        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        
        # Protect log from negative values
        x1 = np.clip(x1, 1e-9, None)
        x2 = np.clip(x2, 1e-9, None)
        
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
            'gamma': 0.2,
            'num_check_points': 3
        }
        return domain, params

class Problem5(MultimodalProblem):
    def __init__(self, n=2):
        self.n = n
        super().__init__()

    @property
    def name(self):
        return f"Problem 5: {self.n}D Shubert Function"

    @property
    def optima_type(self):
        return "max"

    def g_func(self, x):
        x = np.asarray(x)
        prod = 1.0
        for j in range(self.n):
            xj = x[j] if x.ndim == 1 else x[:, j]
            prod *= sum(i * np.cos((i + 1) * xj + i) for i in range(1, 6))
        return -prod

    def get_info(self):
        domain = [(-10, 10)] * self.n
        
        if self.n == 2:
            params = {
                'm_cluster': 2000,
                'r_cl': 0.95,
                'theta_cl': np.pi/4,
                'k_cluster': 15,
                'epsilon': 1e-06,
                'delta': 0.1,
                'sdoa_m': 100,
                'sdoa_k_max': 500,
                'r': 0.95,
                'theta': np.pi/4,
                'gamma': 0.5,
                'num_check_points': 2
            }
        else:
            params = {
                'm_cluster': 16384,
                'r_cl': 0.99,
                'theta_cl': np.pi/4,
                'k_cluster': 100,
                'epsilon': 1e-4,
                'delta': 0.2,
                'sdoa_m': 512,
                'sdoa_k_max': 300,
                'r': 0.95,
                'theta': np.pi/4,
                'gamma': 0.1,
                'num_check_points': 2
            }
        return domain, params

class ProblemSchwefel(MultimodalProblem):
    def __init__(self, n=3):
        self.n = n
        super().__init__()

    @property
    def name(self):
        return f"Schwefel - {self.n}D"

    def g_func(self, x):
        x = np.asarray(x)
        # Using Schwefel 2.22 which has minimum at 0 with value 0
        if x.ndim == 1:
            return np.sum(np.abs(x)) + np.prod(np.abs(x))
        else:
            return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

    def get_info(self):
        domain = [(-4.0, 6.0)] * self.n
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-6,
            'delta': 0.1,
            'sdoa_m': 20,
            'sdoa_k_max': 100,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf'),
            'num_check_points': 2
        }
        return domain, params

class ProblemGriewank(MultimodalProblem):
    def __init__(self, n=2):
        self.n = n
        super().__init__()

    @property
    def name(self):
        return f"Griewank - {self.n}D"

    def g_func(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            sum_sq = np.sum(x**2) / 4000.0
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, self.n + 1))))
            return sum_sq - prod_cos + 1.0
        else:
            sum_sq = np.sum(x**2, axis=1) / 4000.0
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, self.n + 1))), axis=1)
            return sum_sq - prod_cos + 1.0

    def get_info(self):
        domain = [(-600, 600)] * self.n
        params = {
            'm_cluster': 1000,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 10,
            'epsilon': 1e-6,
            'delta': 0.1,
            'sdoa_m': 50,
            'sdoa_k_max': 200,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf'),
            'num_check_points': 2
        }
        return domain, params

class Problemiwm(MultimodalProblem):
    @property
    def name(self):
        return "Problem project IWM"
    
    def g_func(self, x):
        x = np.asarray(x)
        cost_function = lambda d: (
    60.0*d[0] + 70.0*d[1] + 30.0*d[2] + 80.0*d[3] + 90.0*d[4] + 
    40.0*d[5] + 55.0*d[6] + 65.0*d[7] + 55.0*d[8] + 200.0*d[9] + 
    180.0*d[10] + 30.0*d[11] + 80.0*d[12] + 130.0*d[13] + 110.0*d[14] + 
    50.0*d[15] + 40.0*d[16] + 250.0*d[17] + 40.0*d[18] + 20.0*d[19] + 
    10.0*d[20] + 10.0*d[21] + 30.0*d[22] + 40.0*d[23] + 0.0*d[24]
)

        x1 = x[0] if x.ndim == 1 else x[:, 0]
        x2 = x[1] if x.ndim == 1 else x[:, 1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return (term1 + term2 + term3)

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
            'm_cluster': 300,
            'r_cl': 0.95,
            'theta_cl': np.pi/4,
            'k_cluster': 5,
            'epsilon': 1e-5,
            'delta': 0.1,
            'sdoa_m': 50,
            'sdoa_k_max': 250,
            'r': 0.95,
            'theta': np.pi/4,
            'gamma': -float('inf'),
            'num_check_points': 2

        }
        return domain, params

def get_multimodal_problems():
    """Dictionary pemanggil problem."""
    return {
        1: lambda: Problem1(), # Two N Minima
        2: lambda: Problem2(), # Camel Back
        3: lambda: Problem3(n=2), # Rastrigin 2D
        4: lambda: Problem3(n=3), # Rastrigin 3D
        5: lambda: Problem4(),
        6: lambda: Problem5(),
        7: lambda: Problem5(n=3),
        'schwefel': lambda: ProblemSchwefel(n=3),
        'griewank': lambda: ProblemGriewank(n=2),
        'two_n_minima': lambda: Problem1(),
        'rastrigin': lambda: Problem3(n=2),
        'iwm': lambda: Problemiwm()
    }