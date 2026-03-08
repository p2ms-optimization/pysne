import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from .model import Cluster
from utils import objective_function
from initialization.sampling import generate_sobol_points
from optimizers.sdoa.matrix import get_rotation_matrix

def process_point_for_clustering(
    y: np.ndarray, 
    clusters: List[Cluster], 
    equations: List[Callable], 
    gamma: float, 
    domain: List[Tuple[float, float]]
) -> List[Cluster]:
    """
    Fungsi internal untuk mengevaluasi satu titik koordinat y (Kasus Clustering).
    """
    F_y = objective_function(y, equations)

    if F_y <= gamma:
        return clusters

    # Inisialisasi cluster pertama jika masih kosong
    if not clusters:
        initial_radius = 0.1 * min(hi - lo for lo, hi in domain)
        clusters.append(Cluster(y, initial_radius))
        return clusters

    # Cari cluster terdekat (Nearest Cluster Search)
    min_dist = float('inf')
    nearest_cluster = None
    for cluster in clusters:
        dist = np.linalg.norm(y - cluster.center)
        if dist < min_dist:
            min_dist = dist
            nearest_cluster = cluster

    # Logika Mid-point Check
    x_C = nearest_cluster.center
    F_xC = objective_function(x_C, equations)
    x_t = (y + x_C) / 2.0
    F_xt = objective_function(x_t, equations)

    # Clustering Logic
    if F_xt < F_y and F_xt < F_xC:
        # Case 1: Lembah di antara titik, bentuk cluster baru
        new_radius = np.linalg.norm(y - x_t)
        clusters.append(Cluster(y, new_radius))
    elif F_xt > F_y and F_xt > F_xC:
        # Case 2: x_t adalah puncak yang lebih baik
        new_radius = np.linalg.norm(y - x_t)
        clusters.append(Cluster(y, new_radius))
        # Rekursi untuk mengevaluasi titik tengah x_t
        clusters = process_point_for_clustering(x_t, clusters, equations, gamma, domain)
    elif F_y > F_xC:
        # Case 3: Update pusat karena y lebih mendekati puncak akar
        nearest_cluster.center = y.copy()
        nearest_cluster.radius = np.linalg.norm(y - x_t)
    else:
        nearest_cluster.radius = np.linalg.norm(y - x_t)

    return clusters

def perform_iterative_clustering(
    equations: List[Callable], 
    domain: List[Tuple[float, float]], 
    params: Dict[str, Any]
) -> List[Cluster]:
    """
    Fungsi utama fase clustering untuk mengidentifikasi seluruh wilayah potensial akar.
    """
    # Ekstraksi Parameter
    m_cluster = int(params.get('m_cluster', 200))
    gamma = float(params.get('gamma', 0.1))
    k_cluster = int(params.get('k_cluster', 10))
    r = float(params.get('r', 0.95))
    theta = float(params.get('theta', np.pi/4))
    n = len(domain)

    # 1. Inisialisasi Titik Menggunakan Sobol Sequence
    points = generate_sobol_points(m_cluster, n, domain)

    # 2. Precompute Transformasi Spiral
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    # 3. Inisialisasi Cluster Pertama berdasarkan Best Point saat ini
    clusters: List[Cluster] = []
    F_values = np.array([objective_function(p, equations) for p in points])
    best_idx = np.argmax(F_values)
    
    x_prime = points[best_idx].copy()
    initial_radius = 0.5 * min(hi - lo for lo, hi in domain)
    clusters.append(Cluster(x_prime, initial_radius))

    # 4. Main clustering loop
    for k in range(k_cluster):
        # Process points for clustering
        for i in range(m_cluster):
            F_val = objective_function(points[i], equations)
            if F_val > gamma:
                is_center = any(np.allclose(points[i], cluster.center, atol=1e-8) for cluster in clusters)
                if not is_center:
                    clusters = process_point_for_clustering(points[i], clusters, equations, gamma, domain)

        # Update points using spiral dynamics
        F_values = np.array([objective_function(p, equations) for p in points])
        best_idx = np.argmax(F_values)
        x_p = points[best_idx].copy()

        new_points = np.zeros_like(points)
        for i in range(m_cluster):
            new_points[i] = S_n @ points[i] - (S_n - I_n) @ x_p
        points = new_points
        #points = (points @ S_n.T) - (x_p @ (S_n - I_n).T) #alternatif
  
    return clusters
