from abc import ABC, abstractmethod
import numpy as np
from pysne.utils import is_in_domain

class BaseProblem(ABC):
    """
    Abstract Base Class untuk semua problem optimisasi di pysne.
    Menggunakan penamaan variabel dan struktur asli repositori.
    """

    @property
    @abstractmethod
    def name(self):
        """Nama problem (wajib diisi oleh class turunan)"""
        pass

    @abstractmethod
    def g_func(self, x):
        """
        Fungsi objektif utama (Objective Function).
        Menerima input x dan mengembalikan nilai fitness.
        """
        pass

    def __init__(self):
        self.domain, _ = self.get_info()
        self.n_var = len(self.domain)
        self.equations = None
        
    @abstractmethod
    def get_info(self):
        """
        Mengembalikan tuple (domain, params).
        """
        pass

    def evaluate_fitness(self, x):
        """
        Fungsi pembungkus agar engine bisa memanggil fitness secara universal.
        Sangat penting untuk integrasi dengan clustering_process.py.
        """
        return self.g_func(x)

    def select_final_roots(self, candidates):
        """
        Logika seleksi akhir (dari solver_multimodal lama).
        Secara default ini akan melakukan filter delta.
        Class SNE nanti bisa melakukan override fungsi ini untuk menambahkan filter epsilon.
        """
        # Ambil domain dan parameter langsung dari dalam class ini
        domain, params = self.get_info()
        delta = params.get('delta', 0.1)
        epsilon = params.get('epsilon', 1e-7)
        print(f"DEBUG: Filtering with delta = {delta}")
        print(f"DEBUG: Filtering with epsilon = {epsilon}")

        accurate_candidates = []
        for cand in candidates:
            # Pengecekan is_in_domain menggunakan fungsi dari pysne.utils
            if not is_in_domain(cand, domain):
                continue
            
            f_val = self.evaluate_fitness(cand)
            # Panggil fungsi objektif asli
            # F_val = self.g_func(cand)
            if getattr(self, 'problem_type', None) == 'SNE':
                if 1.0 - f_val < epsilon:
                    accurate_candidates.append((cand, f_val))
            else:
                # 2. Filter Tetangga (Khusus Multimodal)
                # Cek apakah cand benar-benar lebih tinggi dari tetangganya
                # Menggunakan parameter epsilon bawaan dari paper
                is_peak = True
                
                for i in range(len(cand)):
                    step = np.zeros_like(cand)
                    step[i] = epsilon
                    if self.evaluate_fitness(cand - step) > f_val or \
                       self.evaluate_fitness(cand + step) > f_val:
                        is_peak = False
                        break
                if is_peak:
                    accurate_candidates.append((cand, f_val))

        if not accurate_candidates:
            return np.array([])

        final_roots = []
        # Urutkan berdasarkan F_val tertinggi
        accurate_candidates.sort(key=lambda x: x[1], reverse=True)

        for cand, f_val in accurate_candidates:
            found_close = False
            for i, (existing, existing_f) in enumerate(final_roots):
                if np.linalg.norm(cand - existing) <= delta:
                    found_close = True
                    # Ganti jika F_val lebih besar
                    if f_val > existing_f:
                        final_roots[i] = (cand, f_val)
                    break
            if not found_close:
                final_roots.append((cand, f_val))

        return np.array([root for root, _ in final_roots])
        # unique_roots = []
        # for cand in accurate_candidates:
        #     if not any(np.linalg.norm(cand - r) < delta for r in unique_roots):
        #         unique_roots.append(cand)
        
        # return np.array(unique_roots)