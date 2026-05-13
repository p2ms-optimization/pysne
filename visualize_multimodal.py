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
            
        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Scatter plot for 3D coordinates, with color indicating function value G
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=G, cmap='viridis', alpha=0.6, s=15)
        
        ax.set_title(problem.name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Separate colorbar for g(x,y,z)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('f(X, Y, Z)')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping {problem.name} as it has {len(domain)} dimensions. Visualization supports 2D and 3D only.")
        return

if __name__ == '__main__':
    problems = get_multimodal_problems()
    for pid, pfunc in problems.items():
        plot_function(pid, pfunc, None)
