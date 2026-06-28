"""
SDOA MURNI untuk Problem 2 dengan Klasifikasi Akar
Menggunakan modul bawaan pysne dan akar referensi dari Tabel 2 jurnal
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict

# Pastikan module pysne dapat diimport
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysne.problems.benchmarks_sne import problem_2
from pysne.utils import is_in_domain
from pysne.optimizers.sdoa.engine import spiral_dynamics_optimization

warnings.filterwarnings('ignore')

# ================================================================
#  AKAR REFERENSI PROBLEM 2 (Dari Tabel 2 Jurnal)
# ================================================================

REFERENCE_ROOTS_PROBLEM_2 = {
    1: np.array([-0.260599, 0.622531]),
    2: np.array([1.530510, -10.202200]),
    3: np.array([1.663420, -16.282800]),
    4: np.array([1.654580, -15.819200]),
    5: np.array([1.433950, -6.820760]),
    6: np.array([1.578220, -12.176700]),
    7: np.array([1.337430, -4.140440]),
    8: np.array([0.500000, 3.141590]),
    9: np.array([0.299449, 2.836930]),
    10: np.array([1.604570, -13.362900]),
    11: np.array([1.294360, -3.137220]),
    12: np.array([1.481320, -8.383610])
}

def classify_root_found(found_root, reference_roots, tolerance=0.5):
    """ Mengklasifikasikan akar yang ditemukan ke akar referensi terdekat """
    min_distance = float('inf')
    closest_root_id = 0

    for root_id, ref_root in reference_roots.items():
        distance = np.linalg.norm(found_root - ref_root)
        if distance < min_distance:
            min_distance = distance
            closest_root_id = root_id

    if min_distance <= tolerance:
        return closest_root_id, min_distance
    else:
        return 0, min_distance

# ================================================================
#  FUNGSI MULTIPLE RUNS DENGAN KLASIFIKASI AKAR
# ================================================================

def run_multiple_trials_with_classification_problem_2(num_runs=100, tolerance=0.5):
    """ Menjalankan multiple runs SDOA untuk Problem 2 dan mengklasifikasikan akar """
    print("\n" + "="*80)
    print(f"MULTIPLE RUNS SDOA MURNI UNTUK PROBLEM 2")
    print(f"Jumlah runs: {num_runs}, Toleransi klasifikasi: {tolerance}")
    print("="*80)

    prob = problem_2()
    domain, params = prob.get_info()
    ndim = len(domain)
    equations = prob.get_equations()

    sdoa_pack = {
        'm': params.get('sdoa_m', 300),
        'k_max': params.get('sdoa_k_max', 300),
        'r': params.get('r', 0.95),
        'theta': params.get('theta', np.pi/4)
    }

    # Inisialisasi counters
    classification_counts = {0: 0}
    for i in range(1, 13):
        classification_counts[i] = 0

    all_roots = []
    error_statistics = {
        'f_norms': [],
        'F_values': [],
        'distances_to_ref': []
    }

    print("\nMenjalankan multiple runs...")
    print("="*60)

    lower_bounds = np.array([d[0] for d in domain])
    upper_bounds = np.array([d[1] for d in domain])

    for run in range(num_runs):
        # Set seed agar run reproducible
        np.random.seed(42 + run)
        
        # Generate initial points secara acak
        rng = np.random.RandomState(42 + run)
        init_points = rng.uniform(lower_bounds, upper_bounds, (sdoa_pack['m'], ndim))

        # Jalankan SDOA dari engine
        best_solution = spiral_dynamics_optimization(
            objective_func=prob.evaluate_fitness,
            domain=domain,
            params=sdoa_pack,
            minimization=False,
            custom_initial_points=init_points
        )

        # Klasifikasikan akar
        root_id, distance = classify_root_found(best_solution, REFERENCE_ROOTS_PROBLEM_2, tolerance)
        classification_counts[root_id] += 1

        f_values = [eq(best_solution) for eq in equations]
        f_norm = np.sqrt(np.sum(np.array(f_values)**2))
        F_value = prob.evaluate_fitness(best_solution)

        all_roots.append({
            'run': run + 1,
            'x1': best_solution[0],
            'x2': best_solution[1],
            'root_id': root_id,
            'distance_to_ref': distance,
            'f_norm': f_norm,
            'F_value': F_value,
            'in_domain': is_in_domain(best_solution, domain)
        })

        error_statistics['f_norms'].append(f_norm)
        error_statistics['F_values'].append(F_value)
        error_statistics['distances_to_ref'].append(distance)

        if (run + 1) % 10 == 0 or run == num_runs - 1:
            status = "[IN]" if is_in_domain(best_solution, domain) else "[OUT]"
            print(f"Run {run+1:4d}/{num_runs}: {status} ({best_solution[0]:.6f}, {best_solution[1]:.6f})")
            if root_id > 0:
                print(f"       Diklasifikasikan sebagai Akar {root_id} (jarak: {distance:.6f})")
            else:
                print(f"       Tidak terklasifikasi (jarak terdekat: {distance:.6f})")

    # ================================================================
    #  TAMPILKAN HASIL KLASIFIKASI
    # ================================================================

    print(f"\n{'='*80}")
    print("HASIL KLASIFIKASI AKAR - PROBLEM 2")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("TABEL FREKUENSI AKAR YANG DITEMUKAN")
    print(f"{'='*80}")
    print(f"{'Akar':^6} | {'Koordinat Referensi':^30} | {'Frekuensi':^12} | {'Persentase':^12}")
    print(f"{'-'*6}+{'-'*30}+{'-'*12}+{'-'*12}")

    for root_id in range(1, 13):
        ref_coord = REFERENCE_ROOTS_PROBLEM_2[root_id]
        freq = classification_counts[root_id]
        percentage = (freq / num_runs) * 100
        print(f"{f'{root_id}':^6} | ({ref_coord[0]:>10.6f}, {ref_coord[1]:>10.6f}) | "
              f"{freq:^12} | {percentage:>10.2f}%")

    print(f"{'-'*6}+{'-'*30}+{'-'*12}+{'-'*12}")

    unclassified_freq = classification_counts[0]
    unclassified_percentage = (unclassified_freq / num_runs) * 100
    print(f"{'Lain':^6} | {'Tidak terklasifikasi':^30} | "
          f"{unclassified_freq:^12} | {unclassified_percentage:>10.2f}%")

    print(f"{'Total':^6} | {'':^30} | "
          f"{num_runs:^12} | {100.0:>10.2f}%")

    # ================================================================
    #  STATISTIK ERROR DAN KUALITAS
    # ================================================================

    print(f"\n{'='*60}")
    print("STATISTIK KUALITAS SOLUSI")
    print(f"{'='*60}")

    total_classified = sum(classification_counts.values()) - classification_counts[0]
    if total_classified > 0:
        classified_roots = [r for r in all_roots if r['root_id'] > 0]
        classified_f_norms = [r['f_norm'] for r in classified_roots]
        classified_F_values = [r['F_value'] for r in classified_roots]
        classified_distances = [r['distance_to_ref'] for r in classified_roots]

        print(f"\nAkar yang terklasifikasi ({total_classified} dari {num_runs}):")
        print(f"  Rata-rata ||f(x)||: {np.mean(classified_f_norms):.6e}")
        print(f"  Minimum ||f(x)||: {np.min(classified_f_norms):.6e}")
        print(f"  Maksimum ||f(x)||: {np.max(classified_f_norms):.6e}")
        print(f"  Rata-rata F(x): {np.mean(classified_F_values):.6f}")
        print(f"  Rata-rata jarak ke referensi: {np.mean(classified_distances):.6f}")

    if classification_counts[0] > 0:
        unclassified_roots = [r for r in all_roots if r['root_id'] == 0]
        unclassified_f_norms = [r['f_norm'] for r in unclassified_roots]
        unclassified_F_values = [r['F_value'] for r in unclassified_roots]
        unclassified_distances = [r['distance_to_ref'] for r in unclassified_roots]

        print(f"\nAkar tidak terklasifikasi ({classification_counts[0]} dari {num_runs}):")
        print(f"  Rata-rata ||f(x)||: {np.mean(unclassified_f_norms):.6e}")
        print(f"  Rata-rata F(x): {np.mean(unclassified_F_values):.6f}")
        print(f"  Rata-rata jarak ke referensi terdekat: {np.mean(unclassified_distances):.6f}")

    # Kelompokkan akar berdasarkan area
    area_counts = {'x_negatif': 0, 'x_kecil_positif': 0, 'x_besar_y_negatif': 0, 'lainnya': 0}
    area_mapping = {
        1: 'x_negatif', 8: 'x_kecil_positif', 9: 'x_kecil_positif',
        2: 'x_besar_y_negatif', 3: 'x_besar_y_negatif', 4: 'x_besar_y_negatif',
        5: 'x_besar_y_negatif', 6: 'x_besar_y_negatif', 7: 'x_besar_y_negatif',
        10: 'x_besar_y_negatif', 11: 'x_besar_y_negatif', 12: 'x_besar_y_negatif'
    }

    for root in all_roots:
        if root['root_id'] > 0:
            area = area_mapping.get(root['root_id'], 'lainnya')
            area_counts[area] += 1
        else:
            area_counts['lainnya'] += 1

    print(f"\n{'='*60}")
    print("DISTRIBUSI AKAR YANG DITEMUKAN BERDASARKAN AREA")
    print(f"{'='*60}")
    print(f"  x negatif (akar 1): {area_counts['x_negatif']} ({area_counts['x_negatif']/num_runs*100:.1f}%)")
    print(f"  x kecil positif (akar 8, 9): {area_counts['x_kecil_positif']} ({area_counts['x_kecil_positif']/num_runs*100:.1f}%)")
    print(f"  x besar, y negatif (akar 2-7, 10-12): {area_counts['x_besar_y_negatif']} ({area_counts['x_besar_y_negatif']/num_runs*100:.1f}%)")
    print(f"  Tidak terklasifikasi/lainnya: {area_counts['lainnya']} ({area_counts['lainnya']/num_runs*100:.1f}%)")

    return classification_counts, all_roots, error_statistics

def plot_classification_results_problem_2(classification_counts, all_roots):
    """ Visualisasi spasial posisi akar yang ditemukan vs referensi """
    prob = problem_2()
    domain, _ = prob.get_info()

    fig = plt.figure(figsize=(15, 6))

    # 1. 2D Scatter plot
    ax1 = fig.add_subplot(1, 2, 1)
    color_map = plt.cm.tab20(np.linspace(0, 1, 13))
    color_map[0] = [0.7, 0.7, 0.7, 1.0]

    # Plot referensi
    for root_id, coord in REFERENCE_ROOTS_PROBLEM_2.items():
        ax1.scatter(coord[0], coord[1], color=color_map[root_id],
                   s=200, marker='*', edgecolor='black', linewidth=2,
                   label=f'Akar {root_id}' if root_id <= 6 else "")

    # Plot yang ditemukan
    for root in all_roots:
        color = color_map[root['root_id']] if root['root_id'] > 0 else color_map[0]
        alpha = 0.7 if root['root_id'] > 0 else 0.3
        ax1.scatter(root['x1'], root['x2'], color=color, alpha=alpha, s=50)

    domain_x = [domain[0][0], domain[0][1], domain[0][1], domain[0][0], domain[0][0]]
    domain_y = [domain[1][0], domain[1][0], domain[1][1], domain[1][1], domain[1][0]]
    ax1.plot(domain_x, domain_y, 'k--', alpha=0.5, label='Batas domain')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Posisi Akar yang Ditemukan vs Referensi')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.5, 3.5)
    ax1.set_ylim(-18, 5)

    # 2. 3D Surface plot
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
        color = color_map[root['root_id']] if root['root_id'] > 0 else color_map[0]
        ax2.scatter(root['x1'], root['x2'], root['F_value'],
                  color=color, s=50, alpha=0.8, edgecolor='black')

    # Plot referensi
    for root_id, coord in REFERENCE_ROOTS_PROBLEM_2.items():
        ax2.scatter(coord[0], coord[1], 1.0,  # F(x) = 1 di akar sejati
                  color=color_map[root_id], s=100, marker='*',
                  edgecolor='black', linewidth=2)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('F(x)')
    ax2.set_title('Landscape Fungsi Objektif dan Distribusi Solusi')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='F(x)')

    plt.tight_layout()
    plt.show()

# ================================================================
#  FUNGSI UTAMA
# ================================================================

if __name__ == "__main__":
    print("SDOA MURNI UNTUK PROBLEM 2 - DENGAN KLASIFIKASI AKAR")
    print("Menggunakan referensi akar dari Tabel 2 jurnal")
    print("=" * 80)

    try:
        num_runs = int(input("Jumlah runs (default=100): ") or "100")
        tolerance = float(input("Toleransi untuk klasifikasi (default=0.5): ") or "0.5")
    except:
        num_runs = 100
        tolerance = 0.5

    classification_counts, all_roots, error_statistics = run_multiple_trials_with_classification_problem_2(
        num_runs=num_runs,
        tolerance=tolerance
    )

    print("\nGenerating visualizations...")
    plot_classification_results_problem_2(classification_counts, all_roots)

    print(f"\n{'='*80}")
    print("RINGKASAN AKHIR - PROBLEM 2")
    print(f"{'='*80}")

    total_runs = num_runs
    total_classified = sum(classification_counts.values()) - classification_counts[0]
    success_rate = (total_classified / total_runs) * 100

    print(f"\nTotal runs: {total_runs}")
    print(f"Akar yang berhasil diklasifikasikan: {total_classified} ({success_rate:.2f}%)")
    print(f"Akar yang tidak terklasifikasi: {classification_counts[0]} ({100-success_rate:.2f}%)")

    print(f"\nAkar dengan frekuensi tertinggi:")
    max_freq = max([classification_counts[i] for i in range(1, 13)])
    top_roots = [root_id for root_id in range(1, 13) if classification_counts[root_id] == max_freq]

    if top_roots:
        roots_str = ', '.join([f'Akar {id}' for id in top_roots])
        print(f"  {roots_str}: {max_freq} kali ({max_freq/total_runs*100:.2f}%)")

    not_found = [root_id for root_id in range(1, 13) if classification_counts[root_id] == 0]
    if not_found:
        print(f"\nAkar yang BELUM PERNAH ditemukan: {', '.join([f'Akar {id}' for id in not_found])}")
        print(f"  Jumlah: {len(not_found)} dari 12 akar ({len(not_found)/12*100:.1f}%)")
    else:
        print(f"\nSEMUA 12 akar referensi ditemukan setidaknya sekali!")

    print(f"\n{'='*60}")
    print("ANALISIS BIAS SDOA MURNI:")
    print(f"{'='*60}")

    frequencies = [classification_counts[i] for i in range(1, 13)]
    cv = np.std(frequencies) / np.mean(frequencies) if np.mean(frequencies) > 0 else 0

    print(f"  Koefisien variasi frekuensi: {cv:.3f}")
    print(f"  (Semakin tinggi, semakin bias distribusi)")

    unique_roots_found = sum(1 for i in range(1, 13) if classification_counts[i] > 0)
    print(f"  Akar unik yang ditemukan: {unique_roots_found} dari 12")

    print(f"\n{'='*60}")
    print("REKOMENDASI UNTUK PENINGKATAN:")
    print(f"{'='*60}")

    if unique_roots_found < 6:
        print("1. SDOA MURNI SANGAT TERBATAS untuk Problem 2")
        print("2. Diperlukan CLUSTERING untuk menemukan lebih banyak akar")
        print("3. Pertimbangkan multi-start strategy dengan inisialisasi berbeda")
        print("4. Gunakan mekanisme niching untuk menjaga keragaman")
    elif unique_roots_found < 12:
        print("1. SDOA murni dapat menemukan beberapa akar, tapi tidak semua")
        print("2. Implementasi CLUSTERING akan meningkatkan coverage secara signifikan")
        print("3. Gunakan algoritma hybrid: SDOA + clustering")
        print("4. Pertimbangkan adaptive parameter tuning")
    else:
        print("1. Performa SDOA murni sudah baik untuk Problem 2")
        print("2. Untuk meningkatkan efisiensi, tetap pertimbangkan clustering")
        print("3. Fokus pada peningkatan akurasi dan kecepatan konvergensi")

    print("\n" + "="*80)
    print("PROGRAM SELESAI")
    print("="*80)