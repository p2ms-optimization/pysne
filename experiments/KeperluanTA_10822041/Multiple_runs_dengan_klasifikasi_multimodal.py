"""
SDOA MURNI untuk Problem Multimodal dengan Klasifikasi Optimum
Implementasi menggunakan SDOA engine dari pysne dan problem dari benchmarks_multimodal
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict

# Pastikan module pysne dapat diimport
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.utils import is_in_domain
from pysne.optimizers.sdoa.engine import spiral_dynamics_optimization

warnings.filterwarnings('ignore')

# ================================================================
#  OPTIMUM REFERENSI UNTUK PROBLEM MULTIMODAL
# ================================================================

REFERENCE_OPTIMA = {
    2: {  # Problem 2: Six-Hump Camelback
        1: np.array([-1.703607, 0.796084]),
        2: np.array([-1.607105, -0.568651]),
        3: np.array([-0.089842, 0.712656]),
        4: np.array([0.089842, -0.712656]),
        5: np.array([1.607105, 0.568651]),
        6: np.array([1.703607, -0.796084])
    },
    3: {  # Problem 3: 2D Rastrigin
        1: np.array([-0.994959, -0.994959]),
        2: np.array([-0.994959, 0.000000]),
        3: np.array([-0.994959, 0.994959]),
        4: np.array([0.000000, -0.994959]),
        5: np.array([0.000000, 0.000000]),
        6: np.array([0.000000, 0.994959]),
        7: np.array([0.994959, -0.994959]),
        8: np.array([0.994959, 0.000000]),
        9: np.array([0.994959, 0.994959])
    },
    5: {  # Problem 4: 2D Vincent (ID 5 di benchmarks)
        1: np.array([0.333018, 0.333019]),
        2: np.array([0.333019, 0.624228]),
        3: np.array([0.333018, 1.170089]),
        4: np.array([0.333018, 2.193279]),
        5: np.array([0.333018, 4.111209]),
        6: np.array([0.333019, 7.706274]),
        7: np.array([0.624229, 0.624228]),
        8: np.array([0.624228, 1.170089]),
        9: np.array([0.624228, 2.193280]),
        10: np.array([0.624228, 4.111207]),
        11: np.array([0.624228, 7.706278]),
        12: np.array([1.170088, 0.333018]),
        13: np.array([1.170089, 0.624229]),
        14: np.array([1.170089, 1.170089]),
        15: np.array([1.170089, 2.193280]),
        16: np.array([1.170089, 4.111207]),
        17: np.array([1.170089, 7.706277]),
        18: np.array([2.193280, 0.333018]),
        19: np.array([2.193280, 0.624228]),
        20: np.array([2.193280, 1.170089]),
        21: np.array([2.193280, 2.193280]),
        22: np.array([2.193280, 4.111207]),
        23: np.array([2.193280, 7.706278]),
        24: np.array([4.111207, 0.333018]),
        25: np.array([4.111207, 0.624228]),
        26: np.array([4.111207, 1.170089]),
        27: np.array([4.111207, 2.193280]),
        28: np.array([4.111207, 4.111207]),
        29: np.array([4.111208, 7.706278]),
        30: np.array([7.706276, 0.333019]),
        31: np.array([7.706277, 0.624228]),
        32: np.array([7.706275, 1.170089]),
        33: np.array([7.706280, 2.193280]),
        34: np.array([7.706279, 4.111207]),
        35: np.array([7.706277, 7.706277]),
        36: np.array([0.624228, 0.333019])
    }
}

def classify_optimal_found(found_optimum, reference_optima, tolerance=0.1):
    """ Mengklasifikasikan optimum yang ditemukan ke optimum referensi terdekat """
    min_distance = float('inf')
    closest_opt_id = 0

    for opt_id, ref_opt in reference_optima.items():
        distance = np.linalg.norm(found_optimum - ref_opt)
        if distance < min_distance:
            min_distance = distance
            closest_opt_id = opt_id

    if min_distance <= tolerance:
        return closest_opt_id, min_distance
    else:
        return 0, min_distance

# ================================================================
#  MULTIPLE RUNS DENGAN KLASIFIKASI OPTIMUM
# ================================================================

def run_multiple_trials_with_classification_multimodal(problem_id, num_runs=100, tolerance=0.1):
    """ Menjalankan multiple runs SDOA untuk problem multimodal dan mengklasifikasikan optimum """
    problems = get_multimodal_problems()
    prob = problems[problem_id]()
    domain, params = prob.get_info()
    ndim = len(domain)
    
    # Tentukan minimization/maximization
    minimization = True
    if problem_id == 5:  # Vincent is maximization
        minimization = False

    print("\n" + "="*80)
    print(f"MULTIPLE RUNS SDOA MURNI UNTUK {prob.name.upper()}")
    print(f"Jumlah runs: {num_runs}, Toleransi klasifikasi: {tolerance}")
    print(f"Mode Optimisasi: {'Minimization' if minimization else 'Maximization'}")
    print("="*80)

    reference_optima = REFERENCE_OPTIMA.get(problem_id, {})
    num_optima = len(reference_optima)

    # Inisialisasi counters
    classification_counts = {0: 0}
    for i in range(1, num_optima + 1):
        classification_counts[i] = 0

    all_solutions_data = []
    fitness_values = []
    distances_to_ref = []

    print("\nMenjalankan multiple runs...")
    print("="*60)

    sdoa_m = params.get('sdoa_m', params.get('m', 50))
    k_max = params.get('sdoa_k_max', params.get('k_max', 250))
    
    sdoa_pack = {
        'm': sdoa_m,
        'k_max': k_max,
        'r': params.get('r', 0.95),
        'theta': params.get('theta', np.pi/4)
    }

    lower_bounds = np.array([d[0] for d in domain])
    upper_bounds = np.array([d[1] for d in domain])

    for run in range(num_runs):
        # Set global seed untuk reproduksibilitas
        np.random.seed(42 + run)
        
        # Generate initial points acak uniform
        rng = np.random.RandomState(42 + run)
        init_points = rng.uniform(lower_bounds, upper_bounds, (sdoa_m, ndim))
        
        # Panggil SDOA engine
        best_solution = spiral_dynamics_optimization(
            objective_func=prob.evaluate_fitness,
            domain=domain,
            params=sdoa_pack,
            minimization=minimization,
            custom_initial_points=init_points
        )

        fitness = prob.evaluate_fitness(best_solution)

        # Klasifikasikan optimum yang ditemukan
        opt_id, distance = classify_optimal_found(best_solution, reference_optima, tolerance)
        classification_counts[opt_id] += 1

        all_solutions_data.append({
            'run': run + 1,
            'x1': best_solution[0],
            'x2': best_solution[1] if ndim > 1 else 0.0,
            'opt_id': opt_id,
            'distance_to_ref': distance,
            'fitness': fitness,
            'in_domain': is_in_domain(best_solution, domain)
        })

        fitness_values.append(fitness)
        distances_to_ref.append(distance)

        if (run + 1) % 10 == 0 or run == num_runs - 1:
            status = "[IN]" if is_in_domain(best_solution, domain) else "[OUT]"
            coord_str = f"({best_solution[0]:.6f}, {best_solution[1]:.6f})" if ndim > 1 else f"({best_solution[0]:.6f})"
            print(f"Run {run+1:4d}/{num_runs}: {status} {coord_str} | fitness = {fitness:.6f}")
            if opt_id > 0:
                print(f"       Diklasifikasikan sebagai Optimum {opt_id} (jarak: {distance:.6f})")
            else:
                print(f"       Tidak terklasifikasi (jarak terdekat: {distance:.6f})")

    # ================================================================
    #  TAMPILKAN HASIL KLASIFIKASI
    # ================================================================

    print(f"\n{'='*80}")
    print(f"HASIL KLASIFIKASI OPTIMUM - {prob.name}")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("TABEL FREKUENSI OPTIMUM YANG DITEMUKAN")
    print(f"{'='*80}")
    print(f"{'Optimum':^8} | {'Koordinat Referensi':^30} | {'Frekuensi':^12} | {'Persentase':^12}")
    print(f"{'-'*8}+{'-'*30}+{'-'*12}+{'-'*12}")

    for opt_id in range(1, num_optima + 1):
        ref_coord = reference_optima[opt_id]
        freq = classification_counts[opt_id]
        percentage = (freq / num_runs) * 100
        coord_str = f"({ref_coord[0]:>10.6f}, {ref_coord[1]:>10.6f})" if ndim > 1 else f"({ref_coord[0]:>10.6f})"
        print(f"{f'{opt_id}':^8} | {coord_str:^30} | {freq:^12} | {percentage:>10.2f}%")

    print(f"{'-'*8}+{'-'*30}+{'-'*12}+{'-'*12}")

    unclassified_freq = classification_counts[0]
    unclassified_percentage = (unclassified_freq / num_runs) * 100
    print(f"{'Lain':^8} | {'Tidak terklasifikasi':^30} | "
          f"{unclassified_freq:^12} | {unclassified_percentage:>10.2f}%")

    print(f"{'Total':^8} | {'':^30} | "
          f"{num_runs:^12} | {100.0:>10.2f}%")

    return classification_counts, all_solutions_data, fitness_values

# ================================================================
#  VISUALISASI HASIL
# ================================================================

def plot_classification_results_multimodal(problem_id, classification_counts, all_roots, fitness_values):
    """ Visualisasi spasial posisi optimum yang ditemukan vs referensi """
    problems = get_multimodal_problems()
    prob = problems[problem_id]()
    domain, _ = prob.get_info()
    ndim = len(domain)
    reference_optima = REFERENCE_OPTIMA.get(problem_id, {})
    num_optima = len(reference_optima)

    fig = plt.figure(figsize=(15, 6))

    # 1. 2D Scatter plot
    ax1 = fig.add_subplot(1, 2, 1)
    color_map = plt.cm.tab20(np.linspace(0, 1, num_optima + 1))
    color_map[0] = [0.7, 0.7, 0.7, 1.0]

    # Plot referensi
    for opt_id, coord in reference_optima.items():
        coord_2d = coord if ndim > 1 else np.array([coord[0], 0.0])
        ax1.scatter(coord_2d[0], coord_2d[1], color=color_map[opt_id],
                   s=200, marker='*', edgecolor='black', linewidth=2,
                   label=f'Opt {opt_id}' if num_optima <= 10 else "")

    # Plot yang ditemukan
    for root in all_roots:
        color = color_map[root['opt_id']] if root['opt_id'] > 0 else color_map[0]
        alpha = 0.7 if root['opt_id'] > 0 else 0.3
        y_val = root['x2'] if ndim > 1 else 0.0
        ax1.scatter(root['x1'], y_val, color=color, alpha=alpha, s=50)

    if ndim > 1:
        domain_x = [domain[0][0], domain[0][1], domain[0][1], domain[0][0], domain[0][0]]
        domain_y = [domain[1][0], domain[1][0], domain[1][1], domain[1][1], domain[1][0]]
        ax1.plot(domain_x, domain_y, 'k--', alpha=0.5, label='Batas domain')
    else:
        ax1.axvline(x=domain[0][0], color='r', linestyle='--')
        ax1.axvline(x=domain[0][1], color='r', linestyle='--', label='Batas domain')

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2' if ndim > 1 else 'y')
    ax1.set_title('Posisi Optimum yang Ditemukan vs Referensi')
    if num_optima <= 10:
        ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 3D Surface plot jika 2 dimensi
    if ndim == 2:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        x_vals = np.linspace(domain[0][0], domain[0][1], 50)
        y_vals = np.linspace(domain[1][0], domain[1][1], 50)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = prob.evaluate_fitness([X[i, j], Y[i, j]])

        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

        # Plot yang ditemukan
        for root in all_roots:
            color = color_map[root['opt_id']] if root['opt_id'] > 0 else color_map[0]
            ax2.scatter(root['x1'], root['x2'], root['fitness'],
                      color=color, s=50, alpha=0.8, edgecolor='black')

        # Plot referensi
        for opt_id, coord in reference_optima.items():
            ref_fitness = prob.evaluate_fitness(coord)
            ax2.scatter(coord[0], coord[1], ref_fitness,
                      color=color_map[opt_id], s=150, marker='o',
                      edgecolor='black', linewidth=2)

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('g(x)')
        ax2.set_title(f'Landscape Fungsi Objektif dan Distribusi Solusi')
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='g(x)')

    plt.tight_layout()
    plt.show()

# ================================================================
#  FUNGSI UTAMA
# ================================================================

if __name__ == "__main__":
    print("SDOA MURNI UNTUK PROBLEM MULTIMODAL - DENGAN KLASIFIKASI OPTIMUM")
    print("=" * 80)

    print("PILIH PROBLEM MULTIMODAL:")
    print("1. Problem 2: Six-Hump Camelback (2D, 6 Minima)")
    print("2. Problem 3: Rastrigin (2D, 9 Minima)")
    print("3. Problem 4: Vincent (2D, 36 Maxima)")
    
    try:
        prob_choice = int(input("Pilih problem (1-3, default=1): ") or "1")
    except:
        prob_choice = 3

    problem_mapping = {1: 2, 2: 3, 3: 5}
    problem_id = problem_mapping.get(prob_choice, 2)
    problems = get_multimodal_problems()
    prob = problems[problem_id]()
    domain, params = prob.get_info()

    print(f"\nProblem yang dipilih: {prob.name}")
    print(f"Domain: {domain}")
    print("=" * 80)

    try:
        num_runs = int(input("Jumlah runs (default=100): ") or "100")
        tolerance = float(input("Toleransi untuk klasifikasi (default=0.1): ") or "0.1")
    except:
        num_runs = 100
        tolerance = 0.1

    classification_counts, all_roots, fitness_values = run_multiple_trials_with_classification_multimodal(
        problem_id=problem_id,
        num_runs=num_runs,
        tolerance=tolerance
    )

    print("\nGenerating visualizations...")
    plot_classification_results_multimodal(problem_id, classification_counts, all_roots, fitness_values)

    total_runs = num_runs
    total_classified = sum(classification_counts.values()) - classification_counts[0]
    success_rate = (total_classified / total_runs) * 100

    print(f"\nTotal runs: {total_runs}")
    print(f"Optimum yang berhasil diklasifikasikan: {total_classified} ({success_rate:.2f}%)")
    print(f"Optimum yang tidak terklasifikasi: {classification_counts[0]} ({100-success_rate:.2f}%)")

    unique_opt_found = sum(1 for i in range(1, len(REFERENCE_OPTIMA[problem_id]) + 1) if classification_counts[i] > 0)
    print(f"Optimum unik yang ditemukan: {unique_opt_found} dari {len(REFERENCE_OPTIMA[problem_id])}")

    print(f"\n{'='*60}")
    print("ANALISIS BIAS SDOA MURNI:")
    print(f"{'='*60}")
    if unique_opt_found < len(REFERENCE_OPTIMA[problem_id]):
        print("1. SDOA murni mengalami bias berat akibat population collapse.")
        print("2. Titik-titik pencarian cenderung mengerucut hanya pada 1 atau 2 optimum dominan saja.")
        print("3. Diperlukan algoritma Clustering untuk menemukan semua optimum.")
    else:
        print("1. Hebat! SDOA murni berhasil menemukan semua optimum setidaknya sekali.")
        print("2. Hal ini biasanya hanya terjadi jika inisialisasi sangat beruntung atau jumlah run sangat besar.")

    print("\n" + "="*80)
    print("PROGRAM SELESAI")
    print("="*80)
