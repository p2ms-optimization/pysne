import numpy as np
import pandas as pd
import time
import itertools
from typing import Dict, List, Any

# Import dari pysne repository
from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.solver import solve_system
from pysne.problems.base import BaseProblem, MinimizedProblem

class PysneTuner:
    """
    Hyperparameter Tuner khusus untuk PySNE.
    Mengadaptasi konsep dari mealpy/tuner.py untuk mengevaluasi kombinasi parameter 
    pada algoritma SPO-Clustering.
    """
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self.results = []
        
    def _generate_grid(self):
        """Membuat kombinasi grid parameter layaknya scikit-learn/mealpy ParameterGrid"""
        keys, values = zip(*self.param_grid.items())
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    def execute(self, problem: BaseProblem, n_trials: int = 3, verbose: bool = True):
        """
        Menjalankan tuning parameter untuk suatu problem multimodal.
        Objektif utama dalam multimodal adalah jumlah peak (optima) yang ditemukan,
        kemudian baru diukur dari waktu eksekusinya.
        """
        grid_list = list(self._generate_grid())
        total_configs = len(grid_list)
        
        if verbose:
            print("="*60)
            print(f"PYSNE TUNER EXECUTING ON: {problem.name}")
            print(f"Total Configurations: {total_configs}")
            print(f"Trials per Config   : {n_trials}")
            print("="*60)

        for idx, params in enumerate(grid_list):
            if verbose:
                print(f"\n[Config {idx+1}/{total_configs}] Testing parameters: {params}")
                
            config_metrics = {
                'points_found': [],
                'time': []
            }
            
            # Eksekusi sebanyak n_trials untuk mendapatkan hasil yang ajeg
            for trial in range(n_trials):
                start_time = time.time()
                try:
                    # Mencari Maxima
                    res_max = solve_system(problem, params, verbose=False)
                    max_roots = res_max['roots']
                    
                    # Cek apakah problem hanya butuh maksimum (Problem 4 dan Problem 5 / Index 5,6,7)
                    skip_min = "Problem 4" in problem.name or "Problem 5" in problem.name
                    
                    if skip_min:
                        min_roots = []
                    else:
                        res_min = solve_system(MinimizedProblem(problem), params, verbose=False)
                        min_roots = res_min['roots']
                    
                    # Rekap
                    total_roots = len(max_roots) + len(min_roots)
                    elapsed = time.time() - start_time
                    
                    config_metrics['points_found'].append(total_roots)
                    config_metrics['time'].append(elapsed)
                    
                except Exception as e:
                    print(f"Error on trial {trial+1}: {e}")
                    config_metrics['points_found'].append(0)
                    config_metrics['time'].append(time.time() - start_time)

            # Hitung rata-rata evaluasi untuk parameter configuration ini
            avg_points = np.mean(config_metrics['points_found'])
            std_points = np.std(config_metrics['points_found'])
            avg_time = np.mean(config_metrics['time'])
            
            # Simpan hasil dictionary
            result_row = {**params} # copy parameters
            result_row.update({
                'avg_optima_found': avg_points,
                'std_optima_found': std_points,
                'avg_time_sec': avg_time
            })
            self.results.append(result_row)
            
            if verbose:
                print(f" -> Result: Avg Optima = {avg_points:.2f} (±{std_points:.2f}), Avg Time = {avg_time:.3f}s")

        # Compile menjadi pandas DataFrame agar mudah dianalisis
        self.df_results = pd.DataFrame(self.results)
        
        # Penentuan parameter terbaik:
        # 1. Mengurutkan berdasarkan jumlah optima yang ditemukan secara descending (sebanyak mungkin)
        # 2. Tie-breaker menggunakan waktu eksekusi secara ascending (secepat mungkin)
        self.df_results = self.df_results.sort_values(
            by=['avg_optima_found', 'avg_time_sec'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        self.best_params = self.df_results.iloc[0].to_dict()
        
        if verbose:
            print("\n" + "="*60)
            print("TUNING COMPLETED. BEST PARAMETERS FOUND:")
            print("="*60)
            for k, v in self.best_params.items():
                if k not in ['avg_optima_found', 'std_optima_found', 'avg_time_sec']:
                    print(f" - {k:<15}: {v}")
            print(f"\nExpected Optima : {self.best_params['avg_optima_found']:.2f} (±{self.best_params['std_optima_found']:.2f})")
            print(f"Expected Time   : {self.best_params['avg_time_sec']:.3f} seconds")

    def export_results(self, filename="tuning_results.csv"):
        if hasattr(self, 'df_results'):
            self.df_results.to_csv(filename, index=False)
            print(f"Tuning results exported to {filename}")

if __name__ == "__main__":
    print("Testing PySNE Tuner on Multimodal Problem")
    
    # Ambil problem multimodal 
    problems = get_multimodal_problems()
    problem_to_tune = problems["iwm"]() #1. 2d second minima 2. six hump camel back 3. 2d rastrigin 4. 3d 5. 2d vincent 6. 2d shubert 7. 2d shubert 
    
    # 1. Tentukan Parameter Grid untuk dievaluasi
    # Fokus pada delta, num_check_points, dan m_cluster untuk menemukan 81 peak
    param_grid = {
        'm_cluster': [300], # Menggunakan pangkat 2 (2^14) untuk Sobol points
        'k_cluster': [5],
        'spo_m': [50],
        'spo_k_max': [250],
        'delta': [0.1],
        'epsilon': [1e-5],
        'gamma': [-float('inf')],
        'num_check_points': [1, 2]
    }
    
    # 2. Inisialisasi Tuner
    tuner = PysneTuner(param_grid=param_grid)
    
    # 3. Eksekusi Tuning (Grid Search)
    # n_trials=1 agar tidak memakan waktu terlalu lama dalam pencarian awal
    tuner.execute(problem_to_tune, n_trials=1, verbose=True)
    
    # 4. Ekspor ke CSV
    tuner.export_results("pysne_tuner_results.csv")
