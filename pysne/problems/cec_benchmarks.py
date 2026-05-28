"""
Integration module untuk menggunakan benchmark functions dari opfunu
dengan SDOA algorithm di pysne.
"""

import numpy as np
import opfunu
from pysne.problems.base import MultimodalProblem

class OpfunuBenchmarkWrapper(MultimodalProblem):
    """Base class untuk wrapper opfunu benchmark functions, compatible with pysne solver"""

    def __init__(self, func_instance, ndim, name=None, custom_params=None):
        self.func = func_instance
        self.ndim = ndim
        self._name = name or func_instance.__class__.__name__
        self.bounds = (func_instance.lb, func_instance.ub)
        self.custom_params = custom_params or {}
        
        # Prepare bounds
        lb = np.array(func_instance.lb) if not np.isscalar(func_instance.lb) else np.full(ndim, func_instance.lb)
        ub = np.array(func_instance.ub) if not np.isscalar(func_instance.ub) else np.full(ndim, func_instance.ub)
        
        # Domain should be (N, 2) shape
        self.domain = np.column_stack((lb, ub))
        self.n_var = self.ndim
        self.equations = None

    @property
    def name(self):
        return self._name

    def g_func(self, x):
        """Evaluasi fungsi pada titik x"""
        return self.func.evaluate(x)
        
    def evaluate(self, x):
        """Alias untuk g_func untuk compatibility luar"""
        return self.g_func(x)

    def get_info(self):
        """Return (domain, params) untuk pysne solver"""
        params = {
            # 'm_cluster': 1500,
            # 'k_cluster': 20,
            # 'm': 100,
            # 'k_max': 100,
            # 'delta': 0.01,
            # 'epsilon': 1e-7
            'm_cluster': 1500,
            'r_cl': 0.95, # r
            'theta_cl': np.pi/4, # theta
            'k_cluster': 20,
            'epsilon': 1e-7,
            'delta': 0.01,
            'sdoa_m': 250,
            'sdoa_k_max': 250,
            'r': 0.95, # sdoa_r
            'theta': np.pi/4, # sdoa_theta
            'gamma': 0.01
        }
        # Timpa default parameter dengan parameter spesifik jika ada
        params.update(self.custom_params)
        return self.domain, params

class CEC2013Benchmark:
    """Collection dari CEC 2013 benchmark functions"""
    FUNCTIONS = {f'F{i}': f'F{i}2013' for i in range(1, 29)}

    @staticmethod
    def get_function(func_name, ndim, custom_params=None):
        if func_name not in CEC2013Benchmark.FUNCTIONS:
            raise ValueError(f"Function {func_name} tidak ditemukan. "
                           f"Available: {list(CEC2013Benchmark.FUNCTIONS.keys())}")
        class_name = CEC2013Benchmark.FUNCTIONS[func_name]
        func_class = getattr(opfunu.cec_based, class_name)
        func_instance = func_class(ndim=ndim)
        return OpfunuBenchmarkWrapper(func_instance, ndim, name=f"CEC2013_{func_name}", custom_params=custom_params)

class CEC2014Benchmark:
    """Collection dari CEC 2014 benchmark functions"""
    FUNCTIONS = {f'F{i}': f'F{i}2014' for i in range(1, 31)}

    @staticmethod
    def get_function(func_name, ndim, custom_params=None):
        if func_name not in CEC2014Benchmark.FUNCTIONS:
            raise ValueError(f"Function {func_name} tidak ditemukan. "
                           f"Available: {list(CEC2014Benchmark.FUNCTIONS.keys())}")
        class_name = CEC2014Benchmark.FUNCTIONS[func_name]
        func_class = getattr(opfunu.cec_based, class_name)
        func_instance = func_class(ndim=ndim)
        return OpfunuBenchmarkWrapper(func_instance, ndim, name=f"CEC2014_{func_name}", custom_params=custom_params)

class NameBasedBenchmark:
    """Collection dari name-based benchmark functions"""
    AVAILABLE_FUNCTIONS = [
        'Ackley02', 'Ackley03', 'Beale', 'BiggsExp02', 'BiggsExp03',
        'Bohachevsky1', 'Bohachevsky2', 'Booth', 'Branin01', 'Brown',
        'Bukin06', 'Camel3', 'Camel6', 'Chichinadze', 'Colville',
        'Rastrigin', 'Rosenbrock', 'Sphere', 'StyblinskiTang'
    ]

    @staticmethod
    def get_function(func_name, ndim=None, custom_params=None):
        if not hasattr(opfunu.name_based, func_name):
            raise ValueError(f"Function {func_name} tidak ditemukan")
        func_class = getattr(opfunu.name_based, func_name)
        if ndim is not None:
            func_instance = func_class(ndim=ndim)
        else:
            func_instance = func_class()
        actual_ndim = func_instance.ndim
        return OpfunuBenchmarkWrapper(func_instance, actual_ndim, name=func_name, custom_params=custom_params)

def create_benchmark_suite(benchmark_type='cec2014', ndim=10, num_functions=5):
    functions = {}
    if benchmark_type.lower() == 'cec2014':
        available = list(CEC2014Benchmark.FUNCTIONS.keys())[:num_functions]
        for func_name in available:
            functions[func_name] = CEC2014Benchmark.get_function(func_name, ndim)
    elif benchmark_type.lower() == 'cec2013':
        available = list(CEC2013Benchmark.FUNCTIONS.keys())[:num_functions]
        for func_name in available:
            functions[func_name] = CEC2013Benchmark.get_function(func_name, ndim)
    elif benchmark_type.lower() == 'name_based':
        available = NameBasedBenchmark.AVAILABLE_FUNCTIONS[:num_functions]
        for func_name in available:
            try:
                functions[func_name] = NameBasedBenchmark.get_function(func_name, ndim)
            except Exception as e:
                print(f"Warning: Tidak bisa load {func_name}: {e}")
    return functions
