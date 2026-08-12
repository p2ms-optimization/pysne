"""
SNE Problem Landscape & Solver Result Visualization CLI Tool.

This script runs the SNE solver on benchmark problems and visualizes
the system equation contours overlaid with discovered roots in 1D, 2D, and 3D.
"""

import os
import sys
import argparse
import numpy as np

from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system
from pysne.visualization import (
    plot_1d_sne_results,
    plot_2d_sne_results,
    plot_3d_sne_results,
)


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
        plot_1d_sne_results(prob, roots, save_path, args.no_show)
    elif prob.n_var == 2:
        save_path = os.path.join(args.save_dir, f"{clean_name}_results.png")
        plot_2d_sne_results(prob, roots, save_path, args.no_show)
    elif prob.n_var == 3:
        save_prefix = os.path.join(args.save_dir, f"{clean_name}_results")
        plot_3d_sne_results(prob, roots, save_prefix, args.no_show)
    else:
        print(f"\nNote: Visualization supports 1D, 2D, and 3D landscapes. This problem has {prob.n_var} variables.")


if __name__ == '__main__':
    main()
