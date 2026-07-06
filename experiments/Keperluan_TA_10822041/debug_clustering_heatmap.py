import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import argparse

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.clustering.modified_clustering_process import perform_iterative_clustering
from pysne.problems.base import BaseProblem, MinimizedProblem

target_point = [1.607105, 0.568651]  # Target point to investigate (e.g., missing local minimum in Problem 4 - Vincent)
    
def run_debug_heatmap(problem_id=2, target=None, resolution=300, mode='max'):
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
    
    if target is not None:
        missing_target = np.array(target)
        print(f"\n--- Investigating events near target {missing_target} ---")
        for log in history_logs:
            if 'y' not in log:
                continue
            if np.linalg.norm(log['y'] - missing_target) < 0.2:
                print(f"\nNear target event: y={log['y']}, Case={log['case']}")
                if 'x_C' in log:
                    print(f"  Nearest center x_C={log['x_C']}")
                    print(f"  F_y={log['F_y']:.4f}, F_xC={log['F_xC']:.4f}, F_xt={log['F_xt']:.4f}")
                    print(f"  Condition F_xt < F_y (Valley test part 1): {log['F_xt'] < log['F_y']}")
                    print(f"  Condition F_xt < F_xC (Valley test part 2): {log['F_xt'] < log['F_xC']}")
                    if log['case'] == 'None (Only radius updated)':
                        print(f"  >> Integrity Issue Identified: Midpoint x_t did not fall into the actual fitness valley between y and x_C.")
                        print(f"  >> Consequently, point y was assumed to be on the same peak as x_C, despite being a distinct local minimum.")
    
    # Plotting
    print("\n--- Generating Heatmap ---")
    # Dynamically extract bounds based on the problem's domain
    x1_min, x1_max = domain[0]
    x2_min, x2_max = domain[1]
    
    x1 = np.linspace(x1_min, x1_max, resolution)
    x2 = np.linspace(x2_min, x2_max, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    print("Evaluating fitness over the grid...")
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            # Fitness matches mode
            Z[i,j] = prob.evaluate_fitness([X1[i,j], X2[i,j]])
            
    plt.figure(figsize=(12, 8))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Fitness (-g(x))')
    plt.title(f"Clustering Process Debugging Heatmap\n({prob_original.name} - {mode.upper()})")
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
    for log in history_logs:
        case = log['case']
        if 'y' not in log:
            continue
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
        plt.scatter(target[0], target[1], color='lime', marker='*', s=200, edgecolors='black', label='Missing Target')
        handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markeredgecolor='black', markersize=15, label='Missing Target'))
        plt.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    output_filename = f'clustering_heatmap_prob_{problem_id}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved to '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Clustering Heatmap")
    parser.add_argument('--problem', type=int, default=2, help='Problem ID (e.g., 2 for Problem 2 - Six Hump Camel Back)')
    parser.add_argument('--target', type=float, nargs=2, default=[1.607105, 0.568651], help='Target point to investigate (e.g., 0.624228 0.333018)')
    parser.add_argument('--resolution', type=int, default=300, help='Resolution for the heatmap grid')
    parser.add_argument('--mode', type=str, choices=['max', 'min'], default='min', help='Whether to cluster for max or min fitness')
    
    args = parser.parse_args()
    
    run_debug_heatmap(problem_id=args.problem, target=args.target, resolution=args.resolution, mode=args.mode)
