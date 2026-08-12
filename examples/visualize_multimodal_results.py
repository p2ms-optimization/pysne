"""
Multimodal Landscape & Solver Optima Visualization CLI Tool.

This script runs the solver on multimodal optimization benchmarks to discover
maxima and minima, and plots 2D surface/contour heatmaps and 3D landscapes.
"""

import os
import sys
import argparse
import numpy as np

from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.solver import solve_system
from pysne.problems.base import MinimizedProblem
from pysne.visualization import (
    plot_2d_multimodal_results,
    plot_3d_multimodal_results,
)


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
        plot_2d_multimodal_results(prob, maxima, minima, save_path, args.no_show)
    elif prob.n_var == 3:
        save_prefix = os.path.join(args.save_dir, f"{clean_name}_results")
        plot_3d_multimodal_results(prob, maxima, minima, save_prefix, args.no_show)
    else:
        print(f"\nNote: Visualization supports 2D and 3D landscapes only. This problem has {prob.n_var} variables.")


if __name__ == '__main__':
    main()
