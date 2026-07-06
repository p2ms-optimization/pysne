import sys
import numpy as np
from pysne.problems.benchmarks_mixed_integer import get_mixed_integer_problems
from pysne.solver import solve_system

# ============================================================
# OVERRIDE PARAMETER DI SINI
# Kosongkan/hapus key kalau mau pakai default dari get_params()
# di dalam class problem (pysne/problems/benchmarks_mixed_integer.py).
# ============================================================
params = {
    # --- Parameter Clustering ---
    'm_cluster': 30000,        # jumlah titik Sobol awal untuk clustering
    'r_cl': 0.99,             # radius shrink factor tiap iterasi cluster
    'theta_cl': np.pi / 32,    # sudut spiral untuk clustering
    'k_cluster': 1000,          # jumlah iterasi clustering
    'gamma': 0.0001,             # threshold filter global (Multimodal/MixedInteger)

    # --- Parameter SDOA (local refinement per cluster) ---
    'sdoa_m': 30000,          # jumlah titik spiral per cluster
    'sdoa_k_max': 1000,       # jumlah iterasi SDOA per cluster
    'r': 0.99,                # radius shrink factor SDOA
    'theta': np.pi / 32,      # sudut rotasi spiral SDOA

    # --- Parameter Seleksi & Penalti ---
    'epsilon': 1e-7,
    'delta': 0.05,
    'M': 1e15,
}


def main(problem_id: int):
    problems = get_mixed_integer_problems()
    if problem_id not in problems:
        print(f"ID {problem_id} tidak ada. Pilihan: {sorted(problems.keys())}")
        return

    p = problems[problem_id]()
    domain, params = p.get_info()

    # Gabungkan default params dari class problem dengan override di atas
    params = {**params, **params}

    print(f"=== {p.name} ===")
    print("Parameter dipakai:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    print()

    result = solve_system(p, params, verbose=True)

    optimals = result['optimals']
    if len(optimals) == 0:
        print("Tidak ada solusi valid ditemukan.")
        return

    # optimals sudah terurut menurun berdasarkan fitness (= -F(x)),
    # jadi kandidat pertama = solusi terbaik. Setiap elemen di sini adalah
    # titik x itu sendiri (bukan tuple (x, fitness)).
    x_best = optimals[0]
    x_best = p.round_mixed(x_best)
    f_val = p.objective(x_best)

    print("x* =", np.round(x_best, 6))
    print("f(x*) =", f_val)


if __name__ == "__main__":
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(problem_id)