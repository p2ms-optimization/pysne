import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Any

from pysne.problems.benchmarks import get_problem_set
from pysne.utils import objective_function, validate_solutions
from pysne.clustering.clustering_process import perform_iterative_clustering
from pysne.solver import solve_system, run_sdoa_on_clusters, select_final_roots



def test_integration_run():
    print("="*60)
    print("PYSNE FULL INTEGRATION TEST")
    print("="*60 + "\n")

    try:
        # 1. Import Problem
        problems = get_problem_set()
        problem_id = 1
        equations, domain, params, expected_roots = problems[problem_id]()
        epsilon = params.get('epsilon', 1e-7)
        delta = params.get('delta', 0.01)
        
        print(f"[STEP 1] Memuat Problem {problem_id}")
        print(f"Target: Mencari {expected_roots} akar solusi.")
        start_time = time.time()

        # 2. Fase Clustering
        print("\n[STEP 2] Menjalankan Iterative Clustering...")
        clusters = perform_iterative_clustering(equations, domain, params)
        print(f"Ditemukan {len(clusters)} wilayah potensial (clusters).")

        # 3. Fase SDOA (Local Optimization)
        print("\n[STEP 3] Menjalankan SDOA pada setiap cluster...")
        sdoa_params = {
            'm': params.get('sdoa_m', 50),
            'r': params.get('sdoa_r', 0.95),
            'theta': params.get('sdoa_theta', np.pi/4),
            'k_max': params.get('sdoa_k_max', 200)
        }
        candidates = run_sdoa_on_clusters(clusters, equations, domain, sdoa_params, epsilon)
        print(f"Dihasilkan {len(candidates)} kandidat titik dari SDOA.")

        # 4. Fase Seleksi & Validasi
        print("\n[STEP 4] Melakukan seleksi akhir dan eliminasi duplikat...")
        raw_roots = select_final_roots(candidates, equations, domain, epsilon, delta)
        final_roots = validate_solutions(raw_roots, equations, domain, epsilon)
        
        elapsed = time.time() - start_time

        # 5. Summary Hasil
        print("\n" + "="*20 + " RINGKASAN HASIL " + "="*20)
        print(f"Waktu Eksekusi   : {elapsed:.2f} detik")
        print(f"Akar Diharapkan  : {expected_roots}")
        print(f"Akar Ditemukan   : {len(final_roots)}")
        
        if len(final_roots) > 0:
            print("\nDaftar Akar yang Ditemukan:")
            for i, root in enumerate(final_roots):
                fitness = objective_function(root, equations)
                print(f"  Akar {i+1}: {root.round(6)} | Residu: {1.0-fitness:.2e}")

        # Evaluasi Akhir
        if len(final_roots) == expected_roots:
            print("\n[STATUS]: SUKSES! Seluruh akar ditemukan dengan presisi tinggi.")
        elif len(final_roots) > expected_roots:
            print("\n[STATUS]: WARNING! Ditemukan lebih banyak titik (mungkin delta terlalu kecil).")
        else:
            print("\n[STATUS]: GAGAL! Beberapa akar terlewat. Perlu tuning parameter.")

    except Exception as e:
        print(f"\n[ERROR]: Terjadi kegagalan sistem: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration_run()