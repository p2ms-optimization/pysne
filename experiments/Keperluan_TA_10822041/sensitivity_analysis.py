"""
sensitivity_analysis.py
=======================
Modul analisis sensitivitas parameter untuk PySNE.

Menguji pengaruh variasi parameter kunci terhadap Discovery Rate (DR)
pada problem multimodal dan SNE.

Cara menjalankan eksperimen:
    python sensitivity_analysis.py --experiment gamma
    python sensitivity_analysis.py --experiment k_cluster
    python sensitivity_analysis.py --all --export csv
"""

import numpy as np
import time
import sys
import copy
import argparse
import csv
import os

try:
    from pysne.problems.benchmarks_multimodal import get_multimodal_problems
    from pysne.problems.benchmarks_sne import get_problem_set as get_sne_problems
    from pysne.problems.base import BaseProblem, MinimizedProblem
    from pysne.solver import solve_system
    PYSNE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Tidak dapat mengimpor PySNE: {e}")
    print("Pastikan script dijalankan dari root direktori repositori PySNE.")
    PYSNE_AVAILABLE = False


# Konfigurasi benchmark multimodal
MULTIMODAL_BENCHMARK_CONFIG = [
    (1, "2D Second Minima",     1,  4),
    (2, "Six-Hump Camel Back",  2,  6),
    (3, "Rastrigin 2D",         4,  9),
    (5, "Vincent 2D",          36, 25),
    (6, "Shubert 2D",          18, 9),
]

# Konfigurasi benchmark SNE
SNE_BENCHMARK_CONFIG = [
    (1, "Problem 1 (2D, 6 akar)",   6),
    (2, "Problem 2 (2D, 12 akar)", 12),
    (3, "Problem 3 (6D, 2 akar)",   2),
]

# Definisi eksperimen sensitivitas
EXPERIMENTS = {
    "gamma": {
        "description": "Pengaruh parameter cut-off gamma terhadap Discovery Rate (mode global-only)",
        "param_key": "gamma",
        "values": [-float('inf'), 0.1, 0.2, 0.5, 0.8],
        "value_labels": ["-inf", "0.1", "0.2", "0.5", "0.8"],
        "target": "multimodal",
        "note": "gamma=-inf mencari semua optima. gamma>0 memfilter hanya optima global.",
    },
    "k_cluster": {
        "description": "Pengaruh jumlah iterasi clustering (k_cluster) terhadap Discovery Rate dan waktu",
        "param_key": "k_cluster",
        "values": [5, 10, 15, 20, 30],
        "value_labels": ["5", "10", "15", "20", "30"],
        "target": "multimodal",
        "note": "Menguji trade-off kualitas diversifikasi vs waktu komputasi.",
    },
    "delta": {
        "description": "Pengaruh parameter resolusi jarak (delta) terhadap jumlah kandidat unik",
        "param_key": "delta",
        "values": [0.001, 0.01, 0.05, 0.1, 0.5],
        "value_labels": ["0.001", "0.01", "0.05", "0.1", "0.5"],
        "target": "multimodal",
        "note": "delta mengontrol resolusi penggabungan kandidat akar terdekat.",
    },
    "sdoa_m": {
        "description": "Pengaruh ukuran populasi SDOA (sdoa_m) per cluster terhadap Discovery Rate",
        "param_key": "sdoa_m",
        "values": [50, 100, 150, 200, 300],
        "value_labels": ["50", "100", "150", "200", "300"],
        "target": "multimodal",
        "note": "Menguji pengaruh jumlah populasi pencarian lokal SDOA.",
    },
    "m_cluster": {
        "description": "Pengaruh jumlah titik awal clustering (m_cluster) terhadap Discovery Rate",
        "param_key": "m_cluster",
        "values": [100, 200, 500, 1000, 2000],
        "value_labels": ["100", "200", "500", "1000", "2000"],
        "target": "multimodal",
        "note": "Menguji pengaruh cakupan titik awal sampling domain.",
    },
    "gamma_sne": {
        "description": "Pengaruh parameter cut-off gamma pada problem SNE",
        "param_key": "gamma",
        "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        "value_labels": ["0.05", "0.1", "0.2", "0.3", "0.5"],
        "target": "sne",
        "note": "Threshold keakuratan absolut F(x) > gamma pada SNE.",
    },
    "num_check_points": {
        "description": "Pengaruh parameter num_check_points(multi-point check)",
        "param_key": "num_check_points",
        "values": [1, 2, 3, 4, 5],
        "value_labels": ["1", "2", "3", "4", "5"],
        "target": "multimodal",
        "note": "Menguji penambahan titik pada fase clustering.",
    },
}


def _run_single(problem, params):
    """Jalankan solver sekali, kembalikan (n_roots_found, elapsed_time)."""
    start = time.time()

    orig_get_info = problem.get_info
    def custom_get_info():
        domain, base_params = orig_get_info()
        updated_params = base_params.copy()
        updated_params.update(params)
        return domain, updated_params
    
    problem.get_info = custom_get_info

    try:
        if getattr(problem, 'problem_type', None) == 'SNE':
            result = solve_system(problem, params, verbose=False)
            n_found = len(result['roots'])
        else:
            # Untuk multimodal, cari maxima dan minima
            res_max = solve_system(problem, params, verbose=False)
            n_max = len(res_max['roots'])

            prob_min = MinimizedProblem(problem)
            res_min = solve_system(prob_min, params, verbose=False)
            n_min = len(res_min['roots'])

            n_found = n_max + n_min
    finally:
        problem.get_info = orig_get_info

    elapsed = time.time() - start
    return n_found, elapsed


class SensitivityResult:
    """Class penampung hasil analisis sensitivitas."""
    def __init__(self, param_value, param_label, func_name, n_optima_expected, n_found, elapsed, n_runs=1):
        self.param_value = param_value
        self.param_label = param_label
        self.func_name = func_name
        self.n_optima_expected = n_optima_expected
        self.n_found = n_found
        self.elapsed = elapsed
        self.n_runs = n_runs

    @property
    def discovery_rate(self):
        if self.n_optima_expected == 0:
            return 0.0
        return self.n_found / self.n_optima_expected

    @property
    def is_success(self):
        return self.n_found >= self.n_optima_expected

    def to_dict(self):
        return {
            "param_label": self.param_label,
            "func_name": self.func_name,
            "n_expected": self.n_optima_expected,
            "n_found": self.n_found,
            "discovery_rate": round(self.discovery_rate, 4),
            "success": self.is_success,
            "elapsed_s": round(self.elapsed, 2),
        }


def run_sensitivity_experiment(experiment_name, n_runs=1, benchmark_ids=None, verbose=True):
    """Jalankan satu analisis sensitivitas parameter."""
    if not PYSNE_AVAILABLE:
        raise RuntimeError("PySNE tidak tersedia. Jalankan dari root repositori.")

    cfg = EXPERIMENTS[experiment_name]
    param_key = cfg["param_key"]
    target = cfg["target"]
    results = []

    if target == "multimodal":
        all_problems = get_multimodal_problems()
        benchmark_cfg = MULTIMODAL_BENCHMARK_CONFIG
    else:
        all_problems = get_sne_problems()
        benchmark_cfg = [(pid, name, n_roots) for pid, name, n_roots in SNE_BENCHMARK_CONFIG]

    if benchmark_ids is not None:
        benchmark_cfg = [b for b in benchmark_cfg if b[0] in benchmark_ids]

    if verbose:
        _print_experiment_header(experiment_name, cfg)

    for param_val, param_label in zip(cfg["values"], cfg["value_labels"]):
        if verbose:
            print(f"\n  -- {param_key} = {param_label} --")

        for bench_entry in benchmark_cfg:
            if target == "multimodal":
                prob_id, func_name, n_max, n_min = bench_entry
                n_expected = n_max + n_min
            else:
                prob_id, func_name, n_expected = bench_entry

            try:
                prob = all_problems[prob_id]()
                _, base_params = prob.get_info()

                test_params = copy.deepcopy(base_params)
                test_params[param_key] = param_val

                n_found_list = []
                elapsed_list = []

                for _ in range(n_runs):
                    n_found, elapsed = _run_single(prob, test_params)
                    n_found_list.append(n_found)
                    elapsed_list.append(elapsed)

                final_n_found = int(np.median(n_found_list))
                final_elapsed = float(np.mean(elapsed_list))

                res = SensitivityResult(
                    param_value=param_val,
                    param_label=param_label,
                    func_name=func_name,
                    n_optima_expected=n_expected,
                    n_found=final_n_found,
                    elapsed=final_elapsed,
                    n_runs=n_runs,
                )
                results.append(res)

                if verbose:
                    dr_str = f"{res.discovery_rate:.0%}"
                    status = "[+]" if res.is_success else "[-]"
                    print(f"    {status} {func_name:<25} "
                          f"Found: {final_n_found:>3}/{n_expected:<3} "
                          f"DR: {dr_str:>5}   "
                          f"({final_elapsed:.1f}s)")

            except Exception as e:
                if verbose:
                    print(f"    [ERROR] {func_name}: {e}")
                    import traceback
                    traceback.print_exc()

    if verbose:
        _print_summary_table(experiment_name, cfg, results)

    return results


def _print_experiment_header(name, cfg):
    print("\n" + "=" * 70)
    print(f"EKSPERIMEN SENSITIVITAS: {name.upper()}")
    print(f"  {cfg['description']}")
    print(f"  Catatan: {cfg['note']}")
    print("=" * 70)


def _print_summary_table(experiment_name, cfg, results):
    """Cetak tabel ringkasan Discovery Rate."""
    if not results:
        return

    param_labels = cfg["value_labels"]
    func_names = list(dict.fromkeys(r.func_name for r in results))

    col_w = 8
    name_w = 26

    print(f"\n{'='*70}")
    print(f"RINGKASAN DISCOVERY RATE - {experiment_name.upper()}")
    print(f"{'='*70}")

    header = f"{'Fungsi Benchmark':<{name_w}}"
    for label in param_labels:
        header += f"  {label:>{col_w}}"
    print(header)
    print("-" * (name_w + (col_w + 2) * len(param_labels)))

    for func_name in func_names:
        row = f"{func_name:<{name_w}}"
        for label in param_labels:
            match = next((r for r in results if r.func_name == func_name and r.param_label == label), None)
            if match:
                dr_str = f"{match.discovery_rate:.0%}"
                row += f"  {dr_str:>{col_w}}"
            else:
                row += f"  {'N/A':>{col_w}}"
        print(row)

    print("-" * (name_w + (col_w + 2) * len(param_labels)))
    avg_row = f"{'Rata-rata DR':<{name_w}}"
    for label in param_labels:
        vals = [min(r.discovery_rate, 1.0) for r in results if r.param_label == label]
        avg = np.mean(vals) if vals else 0.0
        avg_row += f"  {avg:.2%}".rjust(col_w + 2)
    print(avg_row)

    time_row = f"{'Rata-rata Waktu (s)':<{name_w}}"
    for label in param_labels:
        vals = [r.elapsed for r in results if r.param_label == label]
        avg = np.mean(vals) if vals else 0.0
        time_row += f"  {avg:.1f}s".rjust(col_w + 2)
    print(time_row)
    print(f"{'='*70}\n")


def export_to_csv(results, filepath):
    """Export hasil ke CSV."""
    if not results:
        print("[WARNING] Tidak ada hasil untuk di-export.")
        return

    fieldnames = ["param_label", "func_name", "n_expected", "n_found", "discovery_rate", "success", "elapsed_s"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"[OK] Hasil disimpan ke: {filepath}")


def generate_latex_table(results, experiment_name, cfg, caption=None, label=None):
    """Generate LaTeX table untuk dokumen TA."""
    if not results:
        return "% Tidak ada hasil untuk ditampilkan."

    param_labels = cfg["value_labels"]
    param_key = cfg["param_key"]
    func_names = list(dict.fromkeys(r.func_name for r in results))
    n_cols = len(param_labels)

    if caption is None:
        caption = f"Analisis Sensitivitas Parameter ${param_key}$: Discovery Rate pada Setiap Fungsi Benchmark"
    if label is None:
        label = f"tab:sensitivity_{experiment_name}"

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \vspace{2mm}",
        r"    \small"
    ]

    col_spec = "l" + "c" * n_cols
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    header_top = f"        \\multirow{{2}}{{*}}{{\\textbf{{Fungsi Benchmark}}}} & \\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{Parameter ${param_key}$}}}} \\\\"
    lines.append(header_top)

    header_vals = "        " + " & ".join(f"${v}$" for v in param_labels) + r" \\"
    lines.append(header_vals)
    lines.append(r"        \midrule")

    for func_name in func_names:
        dr_per_val = {}
        for label_v in param_labels:
            match = next((r for r in results if r.func_name == func_name and r.param_label == label_v), None)
            dr_per_val[label_v] = match.discovery_rate if match else None

        valid_drs = [v for v in dr_per_val.values() if v is not None]
        best_dr = max(valid_drs) if valid_drs else None

        cells = []
        for label_v in param_labels:
            dr = dr_per_val[label_v]
            if dr is None:
                cells.append("---")
            else:
                dr_str = f"{dr:.0%}".replace("%", "\\%")
                if dr == best_dr and len(set(valid_drs)) > 1:
                    cells.append(f"\\textbf{{{dr_str}}}")
                else:
                    cells.append(dr_str)

        row = f"        {func_name} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    lines.append(r"        \midrule")

    avg_cells = []
    for label_v in param_labels:
        vals = [min(r.discovery_rate, 1.0) for r in results if r.param_label == label_v]
        avg = np.mean(vals) if vals else 0.0
        avg_cells.append(f"{avg:.2%}")

    best_avg = max(float(c.strip('%')) for c in avg_cells) if avg_cells else 0
    avg_row_cells = []
    for c in avg_cells:
        val = float(c.strip('%'))
        c_esc = c.replace("%", "\\%")
        if val == best_avg and len(set(avg_cells)) > 1:
            avg_row_cells.append(f"\\textbf{{{c_esc}}}")
        else:
            avg_row_cells.append(c_esc)

    lines.append(f"        \\textbf{{Rata-rata DR}} & " + " & ".join(avg_row_cells) + r" \\")

    time_cells = []
    for label_v in param_labels:
        vals = [r.elapsed for r in results if r.param_label == label_v]
        avg = np.mean(vals) if vals else 0.0
        time_cells.append(f"{avg:.1f}s")

    lines.append(f"        \\textbf{{Waktu Rata-rata}} & " + " & ".join(time_cells) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def run_all_experiments(n_runs=1, export_format=None, output_dir=".", verbose=True):
    """Jalankan semua eksperimen sensitivitas."""
    all_results = {}

    for exp_name in EXPERIMENTS:
        try:
            results = run_sensitivity_experiment(exp_name, n_runs=n_runs, verbose=verbose)
            all_results[exp_name] = results

            if export_format and results:
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, f"sensitivity_{exp_name}")
                if export_format == "csv":
                    export_to_csv(results, filepath + ".csv")

        except Exception as e:
            print(f"\n[ERROR] Eksperimen '{exp_name}' gagal: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analisis Sensitivitas Parameter PySNE")
    parser.add_argument("--experiment", "-e", choices=list(EXPERIMENTS.keys()), help="Nama eksperimen.")
    parser.add_argument("--all", "-a", action="store_true", help="Jalankan semua eksperimen.")
    parser.add_argument("--list", "-l", action="store_true", help="Daftar eksperimen.")
    parser.add_argument("--export", choices=["csv"], default=None, help="Format export hasil.")
    parser.add_argument("--output-dir", default="sensitivity_results", help="Direktori output.")
    parser.add_argument("--latex", action="store_true", help="Generate tabel LaTeX.")
    parser.add_argument("--n-runs", type=int, default=1, help="Jumlah run per konfigurasi.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Kurangi output.")

    args = parser.parse_args()

    if args.list:
        print("\nEksperimen sensitivitas yang tersedia:\n")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:<15}  [{cfg['target'].upper()}]  {cfg['description']}")
        print()
        return

    verbose = not args.quiet

    if args.all:
        all_results = run_all_experiments(
            n_runs=args.n_runs,
            export_format=args.export,
            output_dir=args.output_dir,
            verbose=verbose,
        )
        if args.latex:
            for exp_name, results in all_results.items():
                if results:
                    latex = generate_latex_table(results, exp_name, EXPERIMENTS[exp_name])
                    latex_path = os.path.join(args.output_dir, f"table_sensitivity_{exp_name}.tex")
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(latex_path, "w", encoding="utf-8") as f:
                        f.write(latex)
                    print(f"[OK] LaTeX disimpan ke: {latex_path}")

    elif args.experiment:
        results = run_sensitivity_experiment(
            args.experiment,
            n_runs=args.n_runs,
            verbose=verbose,
        )

        if args.export and results:
            os.makedirs(args.output_dir, exist_ok=True)
            filepath = os.path.join(args.output_dir, f"sensitivity_{args.experiment}")
            if args.export == "csv":
                export_to_csv(results, filepath + ".csv")

        if args.latex and results:
            latex = generate_latex_table(results, args.experiment, EXPERIMENTS[args.experiment])
            print("\n" + "=" * 70)
            print("KODE LATEX:")
            print("=" * 70)
            print(latex)

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                latex_path = os.path.join(args.output_dir, f"table_sensitivity_{args.experiment}.tex")
                with open(latex_path, "w", encoding="utf-8") as f:
                    f.write(latex)
                print(f"\n[OK] LaTeX disimpan ke: {latex_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
