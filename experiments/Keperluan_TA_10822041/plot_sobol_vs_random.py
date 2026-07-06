import os
import numpy as np
import matplotlib.pyplot as plt
from pysne.initialization.sampling import generate_sobol_points
from pysne.utils import calculate_sobol_discrepancy

def main():
    # 1. Parameter Setup
    num_points = 100
    dimension = 2
    domain = [(-10.0, 10.0), (-10.0, 10.0)]
    
    # 2. Generate Points
    # Menggunakan generator Sobol yang sudah ada di pysne/initialization
    sobol_points = generate_sobol_points(num_points, dimension, domain)
    print(f"sobol: {sobol_points}")
    
    # Menggunakan generator pseudo-random (Biasa)
    # Gunakan seed agar perbandingannya adil dan reproducible
    rng = np.random.default_rng(12345)
    random_points = rng.uniform(-10.0, 10.0, (num_points, dimension))
    print(f"random: {random_points}")
    
    # Hitung discrepancy menggunakan calculate_sobol_discrepancy dari pysne.utils
    sobol_disc = calculate_sobol_discrepancy(points=sobol_points, domain=domain, method='CD')
    random_disc = calculate_sobol_discrepancy(points=random_points, domain=domain, method='CD')
    
    print(f"Discrepancy (CD) - Random: {random_disc:.8f}")
    print(f"Discrepancy (CD) - Sobol: {sobol_disc:.8f}")
    
    # 3. Plotting Setup dengan Estetika Premium
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # --- PLOT 1: Pseudo-Random ---
    ax_rand = axes[0]
    ax_rand.scatter(
        random_points[:, 0], 
        random_points[:, 1], 
        marker='o', 
        color='#ff4d4d',  # Coral red premium
        s=60, 
        linewidths=1.5,
        label='Random Points'
    )
    ax_rand.set_title(
        f"Pseudo-Random\nHigh Discrepancy (CD = {random_disc:.6f})", 
        fontsize=14, 
        fontweight='bold', 
        pad=15
    )
    ax_rand.set_xlabel(r"$x_1$", fontsize=12)
    ax_rand.set_ylabel(r"$x_2$", fontsize=12)
    ax_rand.set_xlim(-10.0, 10.0)
    ax_rand.set_ylim(-10.0, 10.0)
    ax_rand.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax_rand.set_aspect('equal')
    
    # --- PLOT 2: Sobol Sequence ---
    ax_sob = axes[1]
    ax_sob.scatter(
        sobol_points[:, 0], 
        sobol_points[:, 1], 
        marker='o', 
        color='#3b5998',  # Royal/Indigo Blue premium
        s=60, 
        linewidths=1.5,
        label='Sobol Points'
    )
    ax_sob.set_title(
        f"Sobol Sequence\nLow Discrepancy (CD = {sobol_disc:.6f})", 
        fontsize=14, 
        fontweight='bold', 
        pad=15
    )
    ax_sob.set_xlabel(r"$x_1$", fontsize=12)
    ax_sob.set_ylabel(r"$x_2$", fontsize=12)
    ax_sob.set_xlim(-10.0, 10.0)
    ax_sob.set_ylim(-10.0, 10.0)
    ax_sob.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax_sob.set_aspect('equal')
    
    plt.tight_layout()
    
    # Buat direktori output jika belum ada
    output_dir = 'tests/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'sobol_vs_random_100.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot perbandingan berhasil dibuat dan disimpan di: {plot_path}")

if __name__ == '__main__':
    main()
