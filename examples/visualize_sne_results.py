"""
SNE Problem Landscape & Solver Result Visualization.

This script runs the SNE solver on benchmark problems and visualizes
the system equation contours overlaid with discovered roots in 1D, 2D, and 3D.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system


def parse_args():
    """Parse command line arguments for SNE visualization."""
    problems = get_problem_set()
    
    desc_lines = ["Visualize SNE problem landscapes overlaid with solver results."]
    desc_lines.append("\nAvailable SNE problem IDs/names:")
    for k, pfunc in problems.items():
        try:
            prob = pfunc()
            desc_lines.append(f"  {k}: {prob.name} (Dim: {prob.n_var})")
        except Exception:
            desc_lines.append(f"  {k}: [Error instantiating]")
            
    parser = argparse.ArgumentParser(
        description="\n".join(desc_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--problem', 
        type=str, 
        default='1', 
        help='Problem key/ID to run and visualize (default: 1)'
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
    """Execute the SNE solver for a given benchmark problem."""
    prob = problem_func()
    domain, params = prob.get_info()
    
    print("=" * 60)
    print(f"Solving {prob.name.upper()}")
    print(f"Domain: {domain}")
    print(f"Params: {params}")
    print("=" * 60)
    
    roots = []
    try:
        res = solve_system(prob, params, verbose=True)
        roots = res['roots']
        print(f"Found {len(roots)} roots.")
    except Exception as e:
        print(f"Error solving SNE: {e}")
        
    return prob, np.array(roots)


def plot_1d_results(prob, roots, save_path, no_show=False):
    """Plot 1D equation curve and discovered roots."""
    domain, _ = prob.get_info()
    x_min, x_max = domain[0]
    
    n_pts = 1000
    x = np.linspace(x_min, x_max, n_pts)
    
    eq = prob.equations[0]
    try:
        y = np.array([eq(np.array([val])) for val in x])
    except Exception as e:
        print(f"Error evaluating equation for 1D plot: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color='#ff4d4d', linewidth=2, label='Equation $f_1(x)$')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    if len(roots) > 0:
        y_roots = np.array([eq(np.array([val])) for val in roots[:, 0]])
        ax.scatter(roots[:, 0], y_roots, color='green', marker='o', s=150, 
                   edgecolor='black', label='Roots', zorder=10)
            
    ax.set_title(f"{prob.name} - System Equation and Roots", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$f(x_1)$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved 1D equation plot to: {save_path}")
    
    if not no_show:
        plt.show()
    plt.close()


def plot_2d_results(prob, roots, save_path, no_show=False):
    """Plot 2D zero-level equation contours and discovered roots."""
    domain, _ = prob.get_info()
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    
    n_grid = 200
    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = ['#ff4d4d', '#3b42f5', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, eq in enumerate(prob.equations):
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = eq(np.array([X[i, j], Y[i, j]]))
                except Exception:
                    Z[i, j] = np.nan
        
        color = colors[idx % len(colors)]
        ax.contour(X, Y, Z, levels=[0], colors=color, alpha=0.8, linewidths=2)
        
    if len(roots) > 0:
        ax.scatter(roots[:, 0], roots[:, 1], marker='o', facecolor='blue', edgecolor='black', 
                   s=150, linewidths=1, label='Roots', zorder=10)
        
    ax.set_title(f"{prob.name} - System Equations and Roots", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    
    if len(roots) > 0:
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved 2D equation contour plot to: {save_path}")
    
    if not no_show:
        plt.show()
    plt.close()


def plot_3d_results(prob, roots, save_path_prefix, no_show=False):
    """Plot 3D fitness landscape scatter and discovered roots."""
    domain, _ = prob.get_info()
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    z_min, z_max = domain[2]
    
    n_points = 20
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    z = np.linspace(z_min, z_max, n_points)
    X, Y, Z = np.meshgrid(x, y, z)
    
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    G = np.array([prob.g_func(pt) for pt in pts])
        
    threshold = np.percentile(G, 95)
    mask = G >= threshold
    
    fig_scatter = plt.figure(figsize=(10, 8))
    ax_sc = fig_scatter.add_subplot(111, projection='3d')
    sc = ax_sc.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], 
                       c=G[mask], cmap='viridis', alpha=0.6, s=12)
    
    if len(roots) > 0:
        ax_sc.scatter(roots[:, 0], roots[:, 1], roots[:, 2], 
                      color='red', marker='*', s=250, edgecolor='black', label='Found Roots', depthshade=False, zorder=10)
                      
    ax_sc.set_title(f"{prob.name}\n3D Landscape & Found Roots (Top 5% Fitness)", fontsize=12, fontweight='bold')
    ax_sc.set_xlabel('$x_1$')
    ax_sc.set_ylabel('$x_2$')
    ax_sc.set_zlabel('$x_3$')
    cbar = fig_scatter.colorbar(sc, ax=ax_sc, shrink=0.5, aspect=15)
    cbar.set_label('Fitness $F(x_1, x_2, x_3)$')
    if len(roots) > 0:
        ax_sc.legend(loc='upper right')
        
    plt.tight_layout()
    scatter_save_path = f"{save_path_prefix}_3d_scatter.png"
    plt.savefig(scatter_save_path, dpi=300)
    print(f"Saved 3D scatter visualization to: {scatter_save_path}")
    
    if not no_show:
        plt.show()
    plt.close()


def main():
    """Main execution function for SNE visualization."""
    args = parse_args()
    problems = get_problem_set()
    
    try:
        prob_key = int(args.problem)
    except ValueError:
        prob_key = args.problem
        
    if prob_key not in problems:
        print(f"Error: Problem key/ID '{args.problem}' not found in benchmarks_sne.")
        sys.exit(1)
        
    problem_func = problems[prob_key]
    
    prob, roots = solve_problem(prob_key, problem_func)
    
    print("\n" + "="*30 + " SOLVER RESULT SUMMARY " + "="*30)
    print(f"Problem Name   : {prob.name}")
    print(f"Total Roots    : {len(roots)}")
    
    if len(roots) > 0:
        print("\nDiscovered Roots:")
        for idx, pt in enumerate(roots):
            val = prob.evaluate_fitness(pt)
            print(f"  Root {idx+1}: {pt.round(6)} | Fitness: {val:.6f}")
            
    print("=" * 83)
    
    os.makedirs(args.save_dir, exist_ok=True)
    clean_name = prob.name.split(":")[0].strip().replace(" ", "_").lower()
    
    if prob.n_var == 1:
        save_path = os.path.join(args.save_dir, f"{clean_name}_results.png")
        plot_1d_results(prob, roots, save_path, args.no_show)
    elif prob.n_var == 2:
        save_path = os.path.join(args.save_dir, f"{clean_name}_results.png")
        plot_2d_results(prob, roots, save_path, args.no_show)
    elif prob.n_var == 3:
        save_prefix = os.path.join(args.save_dir, f"{clean_name}_results")
        plot_3d_results(prob, roots, save_prefix, args.no_show)
    else:
        print(f"\nNote: Visualization supports 1D, 2D, and 3D landscapes. This problem has {prob.n_var} variables.")


if __name__ == '__main__':
    main()
