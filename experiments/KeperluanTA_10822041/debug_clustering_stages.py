import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import argparse
import os

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.clustering.modified_clustering_process import perform_iterative_clustering
from pysne.problems.base import BaseProblem, MinimizedProblem

def run_debug_heatmap(problem_id=1, target=None, resolution=300, mode='max'):
    problems = get_multimodal_problems()
    if problem_id not in problems:
        raise ValueError(f"Problem ID {problem_id} not found in get_multimodal_problems()")
        
    prob_original = problems[problem_id]()
    domain, params = prob_original.get_info()
    
    print(f"Initializing {prob_original.name}...")

    if mode == 'min':
        prob = MinimizedProblem(prob_original)
        print("\n--- Running Clustering for MINIMA ---")
    else:
        prob = prob_original
        print("\n--- Running Clustering for MAXIMA ---")
    
    history_logs = []
    
    clusters = perform_iterative_clustering(prob, params, history=history_logs)
    
    print(f"Total clustering events logged: {len(history_logs)}")
    
    # Separate the initial state log from other iterative events
    initial_state_entry = None
    iterative_logs = []
    for log in history_logs:
        if log.get('case') == 'InitialState':
            initial_state_entry = log
        else:
            iterative_logs.append(log)

    # Plotting
    print("\n--- Generating Heatmaps ---")
    # Dynamically extract bounds based on the problem's domain
    x1_min, x1_max = domain[0]
    x2_min, x2_max = domain[1]
    
    x1 = np.linspace(x1_min, x1_max, resolution)
    x2 = np.linspace(x2_min, x2_max, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    print("Evaluating fitness over the grid...")
    grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)
    try:
        Z_flat = prob.evaluate_fitness(grid_points)
        if Z_flat.shape == (grid_points.shape[0],):
            Z = Z_flat.reshape(X1.shape)
            print("Successfully evaluated fitness using vectorized approach.")
        else:
            raise ValueError("Vectorized evaluation returned incorrect shape.")
    except Exception as e:
        print(f"Vectorized evaluation failed ({e}). Falling back to loop...")
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i,j] = prob.evaluate_fitness([X1[i,j], X2[i,j]])
            
    # -------------------------------------------------------------
    # STAGE 1: Initial State Visualization (Awal Fase Sebelum Clustering)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Fitness' if mode == 'max' else 'Fitness (-g(x))')
    plt.title(f"Clustering Process Initial State\n({prob_original.name} - {mode.upper()})")
    plt.xlabel('x1')
    plt.ylabel('x2')

    if initial_state_entry is not None:
        init_points = initial_state_entry['points']
        init_clusters = initial_state_entry['clusters']
        
        # Plot initial points
        plt.scatter(init_points[:, 0], init_points[:, 1], color='cyan', s=15, alpha=0.7, edgecolors='black', linewidths=0.5, label='Initial Sobol Points')
        
        # Plot first cluster
        for i, c in enumerate(init_clusters):
            circle = plt.Circle((c.center[0], c.center[1]), c.radius, color='red', fill=False, linewidth=2.0, linestyle='--')
            plt.gca().add_patch(circle)
            plt.scatter(c.center[0], c.center[1], color='red', marker='x', s=120, label='First Cluster Center')
            plt.text(c.center[0]+0.05, c.center[1]+0.05, f"C0 (Initial)", color='red', fontsize=11, fontweight='bold')
    else:
        print("Warning: InitialState log entry was not found in history_logs.")

    # Highlight missing target if provided
    if target is not None:
        plt.scatter(target[0], target[1], color='lime', marker='*', s=200, edgecolors='black', label='Target')

    plt.legend(loc='upper right')
    plt.tight_layout()
    initial_output_filename = f'clustering_heatmap_initial_prob_{problem_id}.png'
    plt.savefig(initial_output_filename, dpi=300)
    print(f"Initial state heatmap saved to '{initial_output_filename}'")
    plt.close()

    # -------------------------------------------------------------
    # STAGE 2: Final Clustering Result Visualization (Beres Fase Clustering)
    # -------------------------------------------------------------
    if target is not None:
        missing_target = np.array(target)
        print(f"\n--- Investigating events near target {missing_target} ---")
        for log in iterative_logs:
            if np.linalg.norm(log['y'] - missing_target) < 0.2:
                print(f"\nNear target event: y={log['y']}, Case={log['case']}")
                if 'x_C' in log:
                    print(f"  Nearest center x_C={log['x_C']}")
                    print(f"  F_y={log['F_y']:.4f}, F_xC={log['F_xC']:.4f}, F_xt={log['F_xt']:.4f}")
                    if 'F_xt_min' in log:
                        print(f"  Condition F_xt_min < F_y: {log['F_xt_min'] < log['F_y']}")
                        print(f"  Condition F_xt_min < F_xC: {log['F_xt_min'] < log['F_xC']}")
                    else:
                        print(f"  Condition F_xt < F_y: {log['F_xt'] < log['F_y']}")
                        print(f"  Condition F_xt < F_xC: {log['F_xt'] < log['F_xC']}")
                    if log['case'] == 'None (Only radius updated)':
                        print(f"  >> Integrity Issue Identified: Midpoint x_t did not fall into the actual fitness valley between y and x_C.")
                        print(f"  >> Consequently, point y was assumed to be on the same peak as x_C, despite being a distinct local minimum.")

    plt.figure(figsize=(12, 8))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Fitness' if mode == 'max' else 'Fitness (-g(x))')
    plt.title(f"Clustering Process Final Result\n({prob_original.name} - {mode.upper()})")
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Color map for cases
    colors = {
        'Init': 'white',
        'Case 1 (Valley)': 'magenta',
        'Case 2 (Mid better)': 'cyan',
        'Case 3 (Update Center)': 'yellow',
        'None (Only radius updated)': 'black'
    }
    
    # Plot history points
    for log in iterative_logs:
        case = log['case']
        plt.scatter(log['y'][0], log['y'][1], color=colors.get(case, 'red'), s=10, alpha=0.5)

    # Add legend for cases
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=case)
               for case, color in colors.items()]
    plt.legend(handles=handles, loc='upper right')

    # Plot final clusters
    print("\n--- Final Clusters ---")
    min_dist_to_target = float('inf')
    closest_cluster_idx = -1
    
    for i, c in enumerate(clusters):
        if target is not None:
            dist = np.linalg.norm(c.center - target)
            if dist < min_dist_to_target:
                min_dist_to_target = dist
                closest_cluster_idx = i
                
        circle = plt.Circle((c.center[0], c.center[1]), c.radius, color='red', fill=False, linewidth=1.5, linestyle='--')
        plt.gca().add_patch(circle)
        plt.scatter(c.center[0], c.center[1], color='red', marker='x', s=100)
        plt.text(c.center[0]+0.05, c.center[1]+0.05, f"C{i}", color='red', fontsize=10, fontweight='bold')

    if target is not None:
        print(f"Target {target}:")
        if closest_cluster_idx != -1:
            closest_c = clusters[closest_cluster_idx]
            print(f"  Closest cluster: C{closest_cluster_idx} at {closest_c.center}")
            print(f"  Distance to target: {min_dist_to_target:.4f} (Cluster Radius: {closest_c.radius:.4f})")
            if min_dist_to_target <= closest_c.radius:
                print("  -> The target IS within the radius of this cluster.")
            else:
                print("  -> The target is NOT within the radius of any cluster.")

    # Highlight missing target if provided
    if target is not None:
        plt.scatter(target[0], target[1], color='lime', marker='*', s=200, edgecolors='black', label='Target')
        handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markeredgecolor='black', markersize=15, label='Target'))
        plt.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    final_output_filename = f'clustering_heatmap_final_prob_{problem_id}.png'
    plt.savefig(final_output_filename, dpi=300)
    print(f"Final clustering heatmap saved to '{final_output_filename}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Clustering Stages (Initial & Final)")
    parser.add_argument('--problem', type=int, default=2, help='Problem ID (e.g., 5 for Problem 4 - Vincent)')
    parser.add_argument('--target', type=float, nargs=2, default=None, help='Target point to investigate (e.g., 0.624228 0.333018)')
    parser.add_argument('--resolution', type=int, default=1000, help='Resolution for the heatmap grid')
    parser.add_argument('--mode', type=str, choices=['max', 'min'], default='max', help='Whether to cluster for max or min fitness')
    
    args = parser.parse_args()
    
    run_debug_heatmap(problem_id=args.problem, target=args.target, resolution=args.resolution, mode=args.mode)
