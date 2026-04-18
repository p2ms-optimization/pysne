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
        Fungsi matematis inti (Objective Function).
        Menerima input x dan mengembalikan nilai fitness.
        """
        pass

    @abstractmethod
    def get_info(self):
        """
        Mengembalikan tuple (domain, params).
        """
        pass

    def select_final_roots(self, candidates):
        """
        Logika seleksi akhir (dari solver_multimodal lama).
        Secara default ini akan melakukan filter delta.
        Class SNE nanti bisa melakukan override fungsi ini untuk menambahkan filter epsilon.
        """
        # Ambil domain dan parameter langsung dari dalam class ini
        domain, params = self.get_info()
        delta = params['delta']

        accurate_candidates = []
        for cand in candidates:
            # Pengecekan is_in_domain menggunakan fungsi dari pysne.utils
            if not is_in_domain(cand, domain):
                continue

            # Panggil fungsi objektif asli
            F_val = self.g_func(cand)
            accurate_candidates.append((cand, F_val))

        if not accurate_candidates:
            return np.array([])

        final_roots = []
        # Urutkan berdasarkan F_val tertinggi
        accurate_candidates.sort(key=lambda x: x[1], reverse=True)

        for cand, F_val in accurate_candidates:
            found_close = False
            for i, (existing, existing_F) in enumerate(final_roots):
                if np.linalg.norm(cand - existing) <= delta:
                    found_close = True
                    # Ganti jika F_val lebih besar
                    if F_val > existing_F:
                        final_roots[i] = (cand, F_val)
                    break
            if not found_close:
                final_roots.append((cand, F_val))

        return np.array([root for root, _ in final_roots])