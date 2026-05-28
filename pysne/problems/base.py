from abc import ABC, abstractmethod
import numpy as np
from pysne.utils import is_in_domain, objective_function, filter_unique_roots, create_continuous_bounds

class BaseProblem(ABC):
    """
    Abstract Base Class untuk semua problem optimisasi di pysne.
    """

    @property
    @abstractmethod
    def name(self):
        """Nama problem (wajib diisi oleh class turunan)"""
        pass

    @property
    def optima_type(self):
        """Tipe optimisasi: 'max', 'min', atau 'both'"""
        return "both"

    @abstractmethod
    def g_func(self, x):
        """
        Fungsi objektif utama (Objective Function).
        Menerima input x dan mengembalikan nilai fitness.
        """
        pass

    def __init__(self):
        domain_info = self.get_info()
        if isinstance(domain_info, tuple) and len(domain_info) == 2:
            self.domain = domain_info[0]
        else:
            self.domain = domain_info
        self.n_var = len(self.domain)
        self.equations = None
        
    @abstractmethod
    def get_info(self):
        """
        Mengembalikan tuple (domain, params) atau just domain.
        """
        pass

    @abstractmethod
    def evaluate_fitness(self, x):
        """Setiap tipe problem mendefinisikan sendiri cara menghitung fitness."""
        pass

    @abstractmethod
    def select_final_roots(self, candidates):
        """Setiap tipe problem mendefinisikan sendiri cara memfilter solusi."""
        pass


class SNEProblem(BaseProblem):
    """Base class khusus SNE agar punya filter 1.0 - f < eps"""
    problem_type = 'SNE'

    def __init__(self):
        super().__init__()
        self.equations = self.get_equations()

    def get_equations(self):
        return []

    def g_func(self, x):
        return objective_function(x, self.equations)

    def evaluate_fitness(self, x):
        return self.g_func(x)

    def select_final_roots(self, candidates):
        domain, params = self.get_info()
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.01)
        
        accurate_candidates = []
        for cand in candidates:
            if not is_in_domain(cand, domain):
                continue
            
            f_val = self.evaluate_fitness(cand)
            if 1.0 - f_val < epsilon:
                accurate_candidates.append((cand, f_val))
                
        return filter_unique_roots(accurate_candidates, delta)


class MultimodalProblem(BaseProblem):
    """Base class khusus Multimodal"""
    problem_type = 'Multimodal'

    def evaluate_fitness(self, x):
        return self.g_func(x)

    def select_final_roots(self, candidates):
        domain, params = self.get_info()
        delta = params.get('delta', 0.1)
        epsilon = params.get('epsilon', 1e-7)
        gamma = params.get('gamma', None)
        
        # Calculate F_star for multimodal global filter
        F_star = 0
        if candidates is not None and len(candidates) > 0:
            evals = [self.evaluate_fitness(c) for c in candidates if is_in_domain(c, domain)]
            if evals:
                F_star = max(evals)

        accurate_candidates = []
        for cand in candidates:
            if not is_in_domain(cand, domain):
                continue
            
            f_val = self.evaluate_fitness(cand)
            
            if gamma is not None and gamma != -float('inf') and F_star > 0:
                if f_val <= (1.0 - gamma) * F_star:
                    continue
                    
            # Filter Tetangga
            is_peak = True
            pert_step = epsilon
            
            for i in range(len(cand)):
                step = np.zeros_like(cand)
                step[i] = pert_step
                
                nb_minus = cand - step
                if is_in_domain(nb_minus, domain) and self.evaluate_fitness(nb_minus) > f_val:
                    is_peak = False
                    break
                    
                nb_plus = cand + step
                if is_in_domain(nb_plus, domain) and self.evaluate_fitness(nb_plus) > f_val:
                    is_peak = False
                    break
                    
            if is_peak:
                accurate_candidates.append((cand, f_val))

        return filter_unique_roots(accurate_candidates, delta)


class DiophantineProblem(BaseProblem):
    """Base class khusus Diophantine (Integer)"""
    problem_type = 'Diophantine'

    def __init__(self):
        self.integer_domain = self.get_integer_domain()
        super().__init__()
        self.equations = self.get_equations()

    def get_integer_domain(self):
        return []

    def get_equations(self):
        return []

    def get_params(self):
        return {}

    def get_info(self):
        domain = create_continuous_bounds(self.integer_domain)
        params = self.get_params()
        return domain, params

    def g_func(self, x):
        # By default for Diophantine we evaluate using rounded integer values
        q = np.round(x).astype(object)
        return objective_function(q, self.equations)

    def evaluate_fitness(self, x):
        q = np.round(x).astype(object)
        if not is_in_domain(q, self.integer_domain):
            return 0.0
        return objective_function(q, self.equations)

    def select_final_roots(self, candidates):
        domain, params = self.get_info()
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.5)
        
        accurate_candidates = []
        seen = set()
        for cand in candidates:
            q_cand = np.round(cand)
            q_cand_int = q_cand.astype(int)
            q_tuple = tuple(q_cand_int)
            
            if q_tuple in seen:
                continue
            seen.add(q_tuple)
            
            if not is_in_domain(q_cand_int, self.integer_domain):
                continue
                
            f_val = self.evaluate_fitness(q_cand_int)
            if 1.0 - f_val <= epsilon:
                accurate_candidates.append((q_cand_int.astype(float), f_val))
                
        return filter_unique_roots(accurate_candidates, delta)


class MinimizedProblem(MultimodalProblem):
    """
    Wrapper class to invert the fitness of an existing problem for minimization search.
    """
    def __init__(self, original_prob):
        self.original = original_prob
        self.domain = original_prob.domain
        self.n_var = original_prob.n_var
        self.equations = original_prob.equations
        self.problem_type = getattr(original_prob, 'problem_type', 'Multimodal')

    @property
    def name(self):
        return f"{self.original.name} (Minimized)"

    def get_info(self):
        return self.original.get_info()

    def g_func(self, x):
        return -self.original.g_func(x)

    def evaluate_fitness(self, x):
        return -self.original.evaluate_fitness(x)

    def select_final_roots(self, candidates):
        original_class = self.original.__class__
        if hasattr(self.original, 'select_final_roots') and original_class.select_final_roots != MultimodalProblem.select_final_roots:
            domain, params = self.get_info()
            delta = params.get('delta', 0.5)
            accurate_candidates = []
            for cand in candidates:
                if is_in_domain(cand, domain):
                    accurate_candidates.append((cand, self.evaluate_fitness(cand)))
            return filter_unique_roots(accurate_candidates, delta)
        else:
            return super().select_final_roots(candidates)
