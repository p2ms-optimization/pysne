"""
Multimodal Landscape & Solver Optima Visualization.

This script runs the solver on multimodal optimization benchmarks to discover
maxima and minima, and plots 2D surface/contour heatmaps and 3D landscapes.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.solver import solve_system
from pysne.problems.base import MinimizedProblem


def parse_args():
    """Parse command line arguments for multimodal visualization."""
    problems = get_multimodal_problems()
    
    desc_lines = ["Visualize multimodal problem landscapes overlaid with solver results."]
    desc_lines.append("\nAvailable problem IDs/names:")
    for k, pfunc in problems.items():
        try:
            prob = pfunc()
            desc_lines.append(f"  {k}: {prob.name}")
        except Exception:
            desc_lines.append(f"  {k}: [Error instantiating]")
            
    parser = argparse.ArgumentParser(
        description="\n".join(desc_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--problem', 
        type=str, 
        default='2', 
        help='Problem key/ID to run and visualize (default: 2 - Six Hump Camel Back)'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='.', 
        help='Directory to save output plots (default: current directory)'
    )
    parser.add_argument(
        '--no_show', 
        action='store_true', 
        help='Do not show plt.show() windows, only save file'
    )
    return parser.parse_args()


def solve_problem(prob_id, problem_func):
    """Execute the solver for maxima and minima on a multimodal problem."""
    prob = problem_func()
    domain, params = prob.get_info()
    
    print("=" * 60)
    print(f"Solving {prob.name.upper()}")
    print(f"Domain: {domain}")
    print(f"Params: {params}")
    print("=" * 60)
    
    # Find Maxima
    maxima = []
    if prob.optima_type in ['max', 'both']:
        print("\n>>> Searching for Maxima...")
        try:
            res_max = solve_system(prob, params, verbose=True)
            maxima = res_max['roots']
            print(f"Found {len(maxima)} maxima.")
        except Exception as e:
            print(f"Error solving for Maxima: {e}")
            
    # Find Minima
    minima = []
    if prob.optima_type in ['min', 'both']:
        print("\n>>> Searching for Minima...")
        try:
            prob_min = MinimizedProblem(prob)
            res_min = solve_system(prob_min, params, verbose=True)
            minima = res_min['roots']
            print(f"Found {len(minima)} minima.")
        except Exception as e:
            print(f"Error solving for Minima: {e}")
            
    return prob, np.array(maxima), np.array(minima)


def plot_2d_results(prob, maxima, minima, save_path, no_show=False):
    """Plot 2D 3D-surface and contour heatmap with overlaid optima."""
    domain, _ = prob.get_info()
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    
    n_grid = 200
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    
    try:
        Z = prob.g_func(pts)
        if isinstance(Z, (list, tuple)):
            Z = Z[0]
        Z = Z.reshape(X.shape)
    except Exception as e:
        print(f"Error evaluating g_func for plotting: {e}")
        return

    fig = plt.figure(figsize=(16, 7))
    
    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
    
    if len(maxima) > 0:
        z_maxima = []
        for pt in maxima:
            z_val = prob.g_func(pt)
            if isinstance(z_val, (list, tuple, np.ndarray)) and len(np.shape(z_val)) > 0:
                z_val = z_val[0]
            z_maxima.append(z_val)
        ax1.scatter(maxima[:, 0], maxima[:, 1], z_maxima, color='red', marker='o', s=100, 
                    edgecolor='black', depthshade=False, label='Maxima', zorder=10)
        
    if len(minima) > 0:
        z_minima = []
        for pt in minima:
            z_val = prob.g_func(pt)
            if isinstance(z_val, (list, tuple, np.ndarray)) and len(np.shape(z_val)) > 0:
                z_val = z_val[0]
            z_minima.append(z_val)
        ax1.scatter(minima[:, 0], minima[:, 1], z_minima, color='blue', marker='o', s=100, 
                    edgecolor='white', depthshade=False, label='Minima', zorder=10)
        
    ax1.set_title(f"{prob.name}\n3D Landscape Surface", fontsize=12, fontweight='bold')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$g(x_1, x_2)$')
    if len(maxima) > 0 or len(minima) > 0:
        ax1.legend(loc='upper right')
        
    # 2. 2D Contour Heatmap
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax2, label='$g(x_1, x_2)$')
    
    if len(maxima) > 0:
        ax2.scatter(maxima[:, 0], maxima[:, 1], color='red', marker='o', s=100, 
                    edgecolor='black', label='Maxima', zorder=5)
        for i, pt in enumerate(maxima):
            ax2.text(pt[0] + (x_max - x_min)*0.015, pt[1] + (y_max - y_min)*0.015, 
                     f"Max {i+1}", color='darkred', fontsize=9, fontweight='bold', zorder=6)
            
    if len(minima) > 0:
        ax2.scatter(minima[:, 0], minima[:, 1], color='blue', marker='o', s=100, 
                    edgecolor='white', label='Minima', zorder=5)
        for i, pt in enumerate(minima):
            ax2.text(pt[0] + (x_max - x_min)*0.015, pt[1] + (y_max - y_min)*0.015, 
                     f"Min {i+1}", color='darkblue', fontsize=9, fontweight='bold', zorder=6)
            
    ax2.set_title(f"{prob.name}\n2D Contour Heatmap with Solver Optima", fontsize=12, fontweight='bold')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(True, linestyle='--', alpha=0.5)
    if len(maxima) > 0 or len(minima) > 0:
        ax2.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved 2D visualization plot to: {save_path}")
    
    if not no_show:
        plt.show()
    plt.close()


def plot_3d_results(prob, maxima, minima, save_path_prefix, no_show=False):
    """Plot 3D landscape scatter and 2D cross-section slices."""
    domain, _ = prob.get_info()
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    z_min, z_max = domain[2]
    
    # 1. Plot 3D Scatter (Thresholded Valley/Peak visualization)
    n_points = 30
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    z = np.linspace(z_min, z_max, n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    try:
        G = prob.g_func(pts)
        if isinstance(G, (list, tuple)):
            G = G[0]
    except Exception as e:
        print(f"Error evaluating g_func for 3D plot: {e}")
        return
        
    optima_type = getattr(prob, 'optima_type', 'min')
    
    if optima_type == 'max':
        threshold = np.percentile(G, 85)
        mask = G >= threshold
        title_suffix = " (Showing top 15% values - Peaks)"
    else:
        threshold = np.percentile(G, 15)
        mask = G <= threshold
        title_suffix = " (Showing bottom 15% values - Valleys)"
        
    fig_scatter = plt.figure(figsize=(10, 8))
    ax_sc = fig_scatter.add_subplot(111, projection='3d')
    sc = ax_sc.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], 
                       c=G[mask], cmap='viridis', alpha=0.6, s=12)
    
    if len(maxima) > 0:
        ax_sc.scatter(maxima[:, 0], maxima[:, 1], maxima[:, 2], 
                      color='red', marker='*', s=250, edgecolor='black', label='Found Maxima', depthshade=False, zorder=10)
                      
    if len(minima) > 0:
        ax_sc.scatter(minima[:, 0], minima[:, 1], minima[:, 2], 
                      color='blue', marker='o', s=150, edgecolor='white', label='Found Minima', depthshade=False, zorder=10)
                      
    ax_sc.set_title(f"{prob.name}\n3D Landscape & Found Optima{title_suffix}", fontsize=12, fontweight='bold')
    ax_sc.set_xlabel('$x_1$')
    ax_sc.set_ylabel('$x_2$')
    ax_sc.set_zlabel('$x_3$')
    cbar = fig_scatter.colorbar(sc, ax=ax_sc, shrink=0.5, aspect=15)
    cbar.set_label('$f(x_1, x_2, x_3)$')
    if len(maxima) > 0 or len(minima) > 0:
        ax_sc.legend(loc='upper right')
        
    plt.tight_layout()
    scatter_save_path = f"{save_path_prefix}_3d_scatter.png"
    plt.savefig(scatter_save_path, dpi=300)
    print(f"Saved 3D scatter visualization to: {scatter_save_path}")
    
    if not no_show:
        plt.show()
    plt.close()

    # 2. Plot 2D Slices (Cross-Sections)
    fig_slices, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    z_slices = [z_min, (z_min + z_max) / 2.0, z_max]
    n_slice_pts = 100
    xs = np.linspace(x_min, x_max, n_slice_pts)
    ys = np.linspace(y_min, y_max, n_slice_pts)
    XS, YS = np.meshgrid(xs, ys)
    
    z_range = z_max - z_min
    slice_tolerance = 0.15 * z_range
    
    for idx, z_val in enumerate(z_slices):
        ZS = np.full_like(XS, z_val)
        slice_pts = np.column_stack([XS.ravel(), YS.ravel(), ZS.ravel()])
        try:
            G_slice = prob.g_func(slice_pts)
            if isinstance(G_slice, (list, tuple)):
                G_slice = G_slice[0]
            G_slice = G_slice.reshape(XS.shape)
        except Exception as e:
            print(f"Failed to evaluate slice at x_3={z_val}: {e}")
            continue
            
        im = axes[idx].contourf(XS, YS, G_slice, levels=50, cmap='viridis')
        axes[idx].set_title(f"Slice at $x_3 = {z_val:.2f}$", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('$x_1$')
        axes[idx].set_ylabel('$x_2$')
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
        fig_slices.colorbar(im, ax=axes[idx], shrink=0.8)
        
        if len(maxima) > 0:
            mask_max = np.abs(maxima[:, 2] - z_val) <= slice_tolerance
            pts_near = maxima[mask_max]
            if len(pts_near) > 0:
                axes[idx].scatter(pts_near[:, 0], pts_near[:, 1], color='red', marker='*', s=150, 
                                  edgecolor='black', label='Near Maxima' if idx==0 else "", zorder=5)
                for pt in pts_near:
                    axes[idx].text(pt[0] + (x_max - x_min)*0.02, pt[1] + (y_max - y_min)*0.02, 
                                   f"Max ($x_3={pt[2]:.2f}$)", color='darkred', fontsize=8, fontweight='bold', zorder=6)
                    
        if len(minima) > 0:
            mask_min = np.abs(minima[:, 2] - z_val) <= slice_tolerance
            pts_near = minima[mask_min]
            if len(pts_near) > 0:
                axes[idx].scatter(pts_near[:, 0], pts_near[:, 1], color='blue', marker='o', s=100, 
                                  edgecolor='white', label='Near Minima' if idx==0 else "", zorder=5)
                for pt in pts_near:
                    axes[idx].text(pt[0] + (x_max - x_min)*0.02, pt[1] + (y_max - y_min)*0.02, 
                                   f"Min ($x_3={pt[2]:.2f}$)", color='darkblue', fontsize=8, fontweight='bold', zorder=6)
        
        if (len(maxima) > 0 or len(minima) > 0) and idx == 0:
            axes[idx].legend(loc='upper right', fontsize=10)

    fig_slices.suptitle(f"{prob.name} - 2D Cross-Sections (Slices)\n(Optima within ±{slice_tolerance:.2f} of slice coordinate are projected)", 
                         fontsize=13, fontweight='bold')
    plt.tight_layout()
    slices_save_path = f"{save_path_prefix}_slices.png"
    plt.savefig(slices_save_path, dpi=300)
    print(f"Saved 2D slices visualization to: {slices_save_path}")
    
    if not no_show:
        plt.show()
    plt.close()


def main():
    """Main execution function for multimodal visualization."""
    args = parse_args()
    problems = get_multimodal_problems()
    
    try:
        prob_key = int(args.problem)
    except ValueError:
        prob_key = args.problem
        
    if prob_key not in problems:
        print(f"Error: Problem key/ID '{args.problem}' not found in benchmarks_multimodal.")
        sys.exit(1)
        
    problem_func = problems[prob_key]
    
    prob, maxima, minima = solve_problem(prob_key, problem_func)
    
    print("\n" + "="*30 + " SOLVER RESULT SUMMARY " + "="*30)
    print(f"Problem Name   : {prob.name}")
    print(f"Total Optima   : {len(maxima) + len(minima)} (Maxima: {len(maxima)}, Minima: {len(minima)})")
    
    if len(maxima) > 0:
        print("\nDiscovered Maxima points:")
        for idx, pt in enumerate(maxima):
            val = prob.evaluate_fitness(pt)
            print(f"  Maxima {idx+1}: {pt.round(6)} | Value: {val:.6f}")
            
    if len(minima) > 0:
        print("\nDiscovered Minima points:")
        for idx, pt in enumerate(minima):
            val = prob.evaluate_fitness(pt)
            print(f"  Minima {idx+1}: {pt.round(6)} | Value: {val:.6f}")
            
    print("=" * 83)
    
    os.makedirs(args.save_dir, exist_ok=True)
    clean_name = prob.name.split(":")[0].strip().replace(" ", "_").lower()
    
    if prob.n_var == 2:
        save_path = os.path.join(args.save_dir, f"{clean_name}_results.png")
        plot_2d_results(prob, maxima, minima, save_path, args.no_show)
    elif prob.n_var == 3:
        save_prefix = os.path.join(args.save_dir, f"{clean_name}_results")
        plot_3d_results(prob, maxima, minima, save_prefix, args.no_show)
    else:
        print(f"\nNote: Visualization supports 2D and 3D landscapes only. This problem has {prob.n_var} variables.")


if __name__ == '__main__':
    main()
