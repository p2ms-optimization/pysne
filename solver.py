import numpy as np

# Import internal dari modul lain (asumsi struktur folder pysne sudah dibuat)
from pysne.initialization.sampling import generate_sobol_points
from pysne.utils import objective_function
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
