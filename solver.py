import numpy as np

# Import internal dari modul lain (asumsi struktur folder pysne sudah dibuat)
from pysne.initialization.sampling import generate_sobol_points
from pysne.utils import objective_function, is_in_domain
from pysne.optimizers.sdoa_engine import spiral_dynamics_optimization

def run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon):
    """
    Menjalankan proses optimasi Spiral Dynamics (SDOA) pada setiap cluster 
    untuk menemukan titik akar yang lebih presisi.

    Fungsi ini membangun batasan domain lokal (hypercube) untuk setiap cluster
    berdasarkan pusat dan radiusnya[cite: 536], membangkitkan titik awal baru menggunakan 
    Sobol sequence, dan menjalankan SDOA pada domain lokal tersebut.

    Parameters
    ----------
    clusters : list
        Daftar objek Cluster yang dihasilkan dari fase iteratif clustering.
    equations : list of callable
        Daftar fungsi sistem persamaan non-linear.
    domain : list of tuple
        Batasan domain global ruang pencarian dalam format [(min, max), ...].
    sdoa_params : dict
        Hyperparameter untuk algoritma SDOA (m, k_max, r, theta).
    epsilon : float
        Nilai toleransi (residual) untuk kriteria early stopping.

    Returns
    -------
    numpy.ndarray
        Array berisi titik-titik kandidat akar yang telah dioptimasi oleh SDOA.
    """
    candidates = []

    # DEFINISI FUNGSI DI LUAR LOOP UNTUK EFISIENSI MEMORI
    def cluster_objective(x):
        return objective_function(x, equations)

    for i, cluster in enumerate(clusters):
        # Determine cluster domain (memastikan tidak keluar dari batas global)
        cluster_domain = []
        for dim in range(len(domain)):
            cluster_lo = max(domain[dim][0], cluster.center[dim] - cluster.radius)
            cluster_hi = min(domain[dim][1], cluster.center[dim] + cluster.radius)
            cluster_domain.append((cluster_lo, cluster_hi))

        # Generate initial points in cluster domain
        initial_points = generate_sobol_points(sdoa_params['m'], len(domain), cluster_domain)

        # Run SDOA in cluster domain
        candidate = spiral_dynamics_optimization(
            cluster_objective, 
            cluster_domain, 
            sdoa_params,
            minimization=False, 
            custom_initial_points=initial_points,
            equations=equations, 
            epsilon=epsilon
        )

        candidates.append(candidate)

    return np.array(candidates)


def select_final_roots(candidates, equations, domain, epsilon, delta):
    """
    Melakukan seleksi tahap akhir untuk menentukan akar-akar valid dari
    titik-titik kandidat hasil optimasi SDOA.
    
    Fungsi ini mengimplementasikan Step 10 dan 11 dari metode clustering
    Sidarto & Kania (2015). Proses seleksi melibatkan:
    1. Membuang kandidat yang keluar dari batas domain.
    2. Membuang kandidat dengan residual 1 - F(x) >= epsilon.
    3. Menggabungkan kandidat yang berdekatan (jarak <= delta), dengan 
       hanya mempertahankan kandidat yang memiliki nilai F(x) tertinggi.

    Parameters
    ----------
    candidates : numpy.ndarray or list
        Daftar titik kandidat akar hasil dari fase optimasi.
    equations : list of callable
        Daftar fungsi sistem persamaan non-linear.
    domain : list of tuple
        Batasan domain global dalam format [(min, max), ...].
    epsilon : float
        Nilai toleransi akurasi akar. Kandidat diterima jika 1 - F(x) < epsilon.
    delta : float
        Batas jarak minimum antar akar yang berbeda (radius ekuivalensi).

    Returns
    -------
    numpy.ndarray
        Array berisi titik-titik akar final yang telah tervalidasi dan unik.
    """
    # Filter 1: Validasi domain dan nilai threshold epsilon
    accurate_candidates = []
    for cand in candidates:
        if not is_in_domain(cand, domain):
            continue
            
        F_val = objective_function(cand, equations)
        if 1.0 - F_val < epsilon:
            accurate_candidates.append((cand, F_val))

    if not accurate_candidates:
        return np.array([])

    # Filter 2: Eliminasi kandidat berdekatan (berdasarkan delta)
    # Urutkan secara descending berdasarkan nilai F_val agar akar 
    # dengan akurasi tertinggi selalu dievaluasi lebih dulu.
    accurate_candidates.sort(key=lambda x: x[1], reverse=True)
    
    final_roots = []

    # bagian ke bawah ada sedikit perubahan dengan main code
    for cand, F_val in accurate_candidates:
        found_close = False
        for existing, _ in final_roots:
            distance = np.linalg.norm(cand - existing)
            if distance <= delta:
                found_close = True
                break  # Langsung buang cand karena existing pasti lebih baik (hasil sorting)
                
        if not found_close:
            final_roots.append((cand, F_val))

    return np.array([root for root, _ in final_roots])
