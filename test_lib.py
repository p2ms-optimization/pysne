import numpy as np
from sdoa_cluster.problems import get_problem_set
from sdoa_cluster.utils import objective_function
from sdoa_cluster.clustering import perform_iterative_clustering

def run_integration_test():
    print("=== PYSNE INTEGRATION TEST (CLUSTERING PHASE) ===\n")

    try:
        # 1. Memilih Problem dari Benchmarks (Sesuai parameter TA Anda)
        # Kita ambil Problem 1 sebagai contoh uji coba
        problems = get_problem_set()
        problem_id = 1
        equations, domain, params, expected_roots = problems[problem_id]()
        
        print(f"[STEP 1] Memuat {params.get('m_cluster')} titik untuk Problem {problem_id}")
        print(f"Target: Mencari {expected_roots} akar solusi.\n")

        # 2. Menjalankan Fase Clustering (Manager memanggil Worker)
        # Fungsi ini akan mengambil params.get('gamma'), params.get('r'), dll secara otomatis
        print("[STEP 2] Menjalankan Iterative Dynamic Clustering...")
        clusters = perform_iterative_clustering(equations, domain, params)
        
        # 3. Menampilkan Hasil Identifikasi Cluster
        print(f"\n[STEP 3] Hasil Identifikasi:")
        print(f"Ditemukan {len(clusters)} wilayah potensial (clusters).")
        
        for i, cluster in enumerate(clusters):
            # Menggunakan __repr__ dari model.py milik teman Anda
            print(f"  - Cluster {i+1}: {cluster}")
            
            # Cek nilai fitness di pusat cluster
            fitness = objective_function(cluster.center, equations)
            print(f"    Nilai F(x): {fitness:.6f} (Makin dekat ke 1.0 makin potensial)")

        # 4. Evaluasi Sederhana
        if len(clusters) >= expected_roots:
            print("\n[STATUS]: SUKSES! Seluruh wilayah akar berhasil diidentifikasi. ✅")
        else:
            print("\n[STATUS]: PERINGATAN! Jumlah cluster kurang dari jumlah akar. Perlu tuning parameter 'gamma'. ⚠️")

    except Exception as e:
        print(f"\n[ERROR]: Terjadi kegagalan integrasi: {e}")
        print("Saran: Pastikan folder 'sdoa_cluster' memiliki file '__init__.py' di setiap sub-foldernya.")

if __name__ == "__main__":
    run_integration_test()git