import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pysne.problems.benchmarks_multimodal import get_multimodal_problems

def plot_function(prob_id, problem_func, save_dir):
    problem = problem_func()
    
    domain, _ = problem.get_info()
    
    if len(domain) == 2:
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        
        # Increase resolution
        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
        
        # Flatten for evaluation
        pts = np.column_stack([X.ravel(), Y.ravel()])
        
        try:
            Z = problem.g_func(pts)
            if isinstance(Z, (list, tuple)):
                Z = Z[0]  # Just in case some return a tuple
            Z = Z.reshape(X.shape)
        except Exception as e:
            print(f"Failed to evaluate {problem.name}: {e}")
            return

        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Surface plot
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title(problem.name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.show()
        
    elif len(domain) == 3:
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        z_min, z_max = domain[2]
        
        # Use a grid for 3D evaluation
        n_points = 30
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        z = np.linspace(z_min, z_max, n_points)
        X, Y, Z = np.meshgrid(x, y, z)
        
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        try:
            G = problem.g_func(pts)
            if isinstance(G, (list, tuple)):
                G = G[0]
        except Exception as e:
            print(f"Failed to evaluate {problem.name}: {e}")
            return
            
        # Determine optima type (default is min)
        optima_type = getattr(problem, 'optima_type', 'min')
        
        # --- PLOT 1: Thresholded 3D Scatter Plot ---
        # Only plot points with high/low fitness to reveal the structure inside the cube.
        if optima_type == 'max':
            # Plot only top 15% (peaks)
            threshold = np.percentile(G, 85)
            mask = G >= threshold
            title_suffix = " (Showing top 15% values - Peaks)"
        else:
            # Plot only bottom 15% (valleys)
            threshold = np.percentile(G, 15)
            mask = G <= threshold
            title_suffix = " (Showing bottom 15% values - Valleys)"
            
        fig_scatter = plt.figure(figsize=(9, 7))
        ax_sc = fig_scatter.add_subplot(111, projection='3d')
        sc = ax_sc.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2], 
                           c=G[mask], cmap='viridis', alpha=0.8, s=15)
        
        ax_sc.set_title(f"{problem.name}\n{title_suffix}")
        ax_sc.set_xlabel('X')
        ax_sc.set_ylabel('Y')
        ax_sc.set_zlabel('Z')
        cbar = fig_scatter.colorbar(sc, ax=ax_sc, shrink=0.5, aspect=15)
        cbar.set_label('f(X, Y, Z)')
        
        plt.tight_layout()
        
        # --- PLOT 2: 2D Slices (Cross-Sections) at different Z values ---
        fig_slices, axes = plt.subplots(1, 3, figsize=(15, 5))
        z_slices = [z_min, (z_min + z_max) / 2.0, z_max]
        
        # Grid for slice evaluation
        n_slice_pts = 100
        xs = np.linspace(x_min, x_max, n_slice_pts)
        ys = np.linspace(y_min, y_max, n_slice_pts)
        XS, YS = np.meshgrid(xs, ys)
        
        for idx, z_val in enumerate(z_slices):
            ZS = np.full_like(XS, z_val)
            slice_pts = np.column_stack([XS.ravel(), YS.ravel(), ZS.ravel()])
            try:
                G_slice = problem.g_func(slice_pts)
                if isinstance(G_slice, (list, tuple)):
                    G_slice = G_slice[0]
                G_slice = G_slice.reshape(XS.shape)
            except Exception as e:
                print(f"Failed to evaluate slice at Z={z_val}: {e}")
                continue
                
            im = axes[idx].contourf(XS, YS, G_slice, levels=50, cmap='viridis')
            axes[idx].set_title(f"Slice at Z = {z_val:.2f}")
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            fig_slices.colorbar(im, ax=axes[idx], shrink=0.8)
            
        fig_slices.suptitle(f"{problem.name} - 2D Cross-Sections (Slices)", fontsize=14)
        plt.tight_layout()
        
        plt.show()
    else:
        print(f"Skipping {problem.name} as it has {len(domain)} dimensions. Visualization supports 2D and 3D only.")
        return

if __name__ == '__main__':
    problems = get_multimodal_problems()
    for pid, pfunc in problems.items():
        plot_function(pid, pfunc, None)
