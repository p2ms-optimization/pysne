"""
Visualization functions for SNE (System of Non-linear Equations) landscapes and solved roots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_1d_sne_results(prob, roots, save_path=None, no_show=False):
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
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 1D equation plot to: {save_path}")
    
    if not no_show:
        plt.show()
    plt.close()
    return fig


def plot_2d_sne_results(prob, roots, save_path=None, no_show=False):
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
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved 2D equation contour plot to: {save_path}")
    
    if not no_show:
        plt.show()
    plt.close()
    return fig


def plot_3d_sne_results(prob, roots, save_path_prefix=None, no_show=False):
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
    if save_path_prefix:
        scatter_save_path = f"{save_path_prefix}_3d_scatter.png"
        plt.savefig(scatter_save_path, dpi=300)
        print(f"Saved 3D scatter visualization to: {scatter_save_path}")
    
    if not no_show:
        plt.show()
    plt.close()
    return fig_scatter
