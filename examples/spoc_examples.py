"""
examples/spoc_examples.py
=========================
Semua contoh problem dari bab5 (domain real) yang digunakan di Tugas Akhir.
Setiap fungsi problem mengembalikan (equations, domain, param, expected_roots).

Cara pakai
----------
    python examples/spoc_examples.py

Atau dari script lain:
    from examples.spoc_examples import run_all, PROBLEMS
    roots = run_one("problem_1")
"""

import numpy as np
from pysne import SPOC


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINISI PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

def problem_1():
    """2D system — 6 roots (Sidarto & Kania benchmark 1)"""
    equations = [
        lambda x: np.exp(x[0] - x[1]) - np.sin(x[0] + x[1]),
        lambda x: x[0]**2 * x[1]**2 - np.cos(x[0] + x[1]),
    ]
    domain = [(-10, 10), (-10, 10)]
    param = {
        'm_cluster': 260, 'k_cluster': 10,
        'gamma': 0.2, 'epsilon': 1e-7, 'delta': 0.01,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 250, 'k_max': 250, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 6


def problem_2():
    """2D system — 12 roots (Sidarto & Kania benchmark 2)"""
    equations = [
        lambda x: 0.5 * np.sin(x[0]*x[1]) - 0.25*x[1]/np.pi - 0.5*x[0],
        lambda x: (1 - 0.25/np.pi)*(np.exp(2*x[0]) - np.e) + np.e*x[1]/np.pi - 2*np.e*x[0],
    ]
    domain = [(-1, 3), (-17, 4)]
    param = {
        'm_cluster': 2500, 'k_cluster': 10,
        'gamma': 0.3, 'epsilon': 1e-7, 'delta': 0.1,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 300, 'k_max': 300, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 12


def problem_3():
    """6D system — 2 roots (Sidarto & Kania benchmark 3)"""
    equations = [
        lambda x: x[0] + (x[1]**2 * x[3] * x[5])/4 + 0.75,
        lambda x: x[1] + 0.405*np.exp(1 + x[0]*x[1]) - 1.405,
        lambda x: x[2] - (x[3]*x[5])/2 + 1.5,
        lambda x: x[3] - 0.605*np.exp(1 - x[2]**2) - 0.395,
        lambda x: x[4] - (x[1]*x[5])/2 + 1.5,
        lambda x: x[5] - x[0]*x[4],
    ]
    domain = [(-5, 5)] * 6
    param = {
        'm_cluster': 100, 'k_cluster': 15,
        'gamma': 0.1, 'epsilon': 1e-7, 'delta': 0.5,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 420, 'k_max': 420, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 2


def problem_4():
    """3D engineering system — 6 roots (Sidarto & Kania benchmark 4)"""
    equations = [
        lambda x: x[0]*x[1] - (x[0]-2*x[2])*(x[1]-2*x[2]) - 165,
        lambda x: (x[0]*x[1]**3)/12 - ((x[0]-2*x[2])*(x[1]-2*x[2])**3)/12 - 9369,
        lambda x: ((2*(x[1]-x[2])**2*(x[0]-x[2])**2*x[2]) /
                   (x[1]+x[0]-2*x[2]+1e-10)) - 6835,
    ]
    domain = [(-40, 40), (-40, 40), (-40, 40)]
    param = {
        'm_cluster': 2000, 'k_cluster': 10,
        'gamma': 1e-3, 'epsilon': 1e-7, 'delta': 0.5,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 500, 'k_max': 500, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 6


def problem_5():
    """5D system — 3 roots (Sidarto & Kania benchmark 5)"""
    equations = [
        lambda x: 2*x[0] + x[1] + x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + 2*x[1] + x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + x[1] + 2*x[2] + x[3] + x[4] - 6,
        lambda x: x[0] + x[1] + x[2] + 2*x[3] + x[4] - 6,
        lambda x: x[0]*x[1]*x[2]*x[3]*x[4] - 1,
    ]
    domain = [(-10, 10)] * 5
    param = {
        'm_cluster': 5000, 'k_cluster': 10,
        'gamma': 0.1, 'epsilon': 5e-4, 'delta': 0.01,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 200, 'k_max': 200, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 3


def problem_6():
    """8D system — 16 roots (Sidarto & Kania benchmark 6)"""
    equations = [
        lambda x: 4.731e-3*x[0]*x[2] - 0.3578*x[1]*x[2] - 0.1238*x[0] + x[6] - 1.637e-3*x[1] - 0.9338*x[3] - 0.3571,
        lambda x: 0.2238*x[0]*x[2] + 0.7623*x[1]*x[2] + 0.2638*x[0] - x[6] - 0.07745*x[1] - 0.6734*x[3] - 0.6022,
        lambda x: x[5]*x[7] + 0.3578*x[0] + 4.731e-3*x[1],
        lambda x: -0.7623*x[0] + 0.2238*x[1] + 0.3461,
        lambda x: x[0]**2 + x[1]**2 - 1,
        lambda x: x[2]**2 + x[3]**2 - 1,
        lambda x: x[4]**2 + x[5]**2 - 1,
        lambda x: x[6]**2 + x[7]**2 - 1,
    ]
    domain = [(-1, 1)] * 8
    param = {
        'm_cluster': 1500, 'k_cluster': 5,
        'gamma': 0.2, 'epsilon': 1e-6, 'delta': 0.01,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 300, 'k_max': 300, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 16


def problem_7():
    """1D Weierstrass function — 9 roots"""
    s, lam, N = 1.1, 1.5, 20

    def weierstrass(x):
        return sum(lam**((s-2)*k) * np.sin(lam**k * x[0]) for k in range(1, N+1))

    equations = [lambda x: weierstrass(x)]
    domain    = [(0, 5.05)]
    param = {
        'm_cluster': 2000, 'k_cluster': 50,
        'gamma': 0.9, 'epsilon': 1e-7, 'delta': 0.0001,
        'r': 0.95, 'theta': np.pi/4,
        'm_sdoa': 150, 'k_max': 150, 'r_sdoa': 0.95, 'theta_sdoa': np.pi/4,
    }
    return equations, domain, param, 9


# Registry — dipakai oleh run_one() dan run_all()
PROBLEMS = {
    "problem_1": (problem_7, "1D Weierstrass function, 9 roots"),
    "problem_2": (problem_1, "2D system (exp/trig), 6 roots"),
    "problem_3": (problem_2, "2D system (sin/exp), 12 roots"),
    "problem_4": (problem_3, "6D system, 2 roots"),
    # "problem_4": (problem_4, "3D engineering system, 6 roots"),
    "problem_5": (problem_5, "5D polynomial system, 3 roots"),
    "problem_6": (problem_6, "8D trigonometric system, 16 roots"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_one(name: str, verbose: bool = True):
    """
    Jalankan satu problem berdasarkan nama.

    Parameters
    ----------
    name    : str  — kunci dari PROBLEMS, mis. "problem_1"
    verbose : bool — tampilkan output (default True)

    Returns
    -------
    list of np.ndarray
    """
    if name not in PROBLEMS:
        raise ValueError(f"Problem '{name}' tidak ditemukan. Pilihan: {list(PROBLEMS)}")
    func, desc = PROBLEMS[name]
    equations, domain, param, expected = func()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Example: {name}  —  {desc}")
        print(f"Expected roots: {expected}")
        print(f"{'='*60}")
    return SPOC.solve(equations, domain, param=param, verbose=verbose)


def run_all(verbose: bool = True):
    """
    Jalankan semua problem secara berurutan.

    Returns
    -------
    dict {name: roots}
    """
    results = {}
    print("\n" + "="*70)
    print("SPOC — RUNNING ALL EXAMPLES (Real Domain)")
    print("="*70)
    for name, (func, desc) in PROBLEMS.items():
        equations, domain, param, expected = func()
        print(f"\n▶ {name}: {desc}  (expected: {expected})")
        roots = SPOC.solve(equations, domain, param=param, verbose=verbose)
        results[name] = roots
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, (func, desc) in PROBLEMS.items():
        _, _, _, expected = func()
        found = len(results.get(name, []))
        status = "✓" if found >= int(expected * 0.8) else "✗"
        print(f"  {status}  {name:<12}  found {found}/{expected}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — demo interaktif
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SPOC — Real-Domain Solver  |  Demo")
    print("=" * 60)
    print("\nPilih mode:")
    print("  1. Jalankan satu contoh problem")
    print("  2. Jalankan semua contoh problem")
    print("  3. Masukkan sistem persamaan sendiri")

    mode = input("\nMasukkan pilihan (1/2/3): ").strip()

    if mode == "1":
        print("\nDaftar problem:")
        for k, (_, desc) in PROBLEMS.items():
            print(f"  {k}: {desc}")
        name = input("\nNama problem (mis. problem_1): ").strip()
        run_one(name)

    elif mode == "2":
        run_all()

    elif mode == "3":
        print("\n--- Input Sistem Persamaan Sendiri ---")
        print("Contoh: untuk f(x) = x[0]**2 + x[1]**2 - 1, ketik: x[0]**2 + x[1]**2 - 1")
        n_eq = int(input("Jumlah persamaan: "))
        equations = []
        for i in range(n_eq):
            expr = input(f"  f{i+1}(x) = ").strip()
            equations.append(eval(f"lambda x: {expr}"))

        n_dim = int(input("Jumlah dimensi (variabel): "))
        domain = []
        for i in range(n_dim):
            lo = float(input(f"  Domain x[{i}] — batas bawah: "))
            hi = float(input(f"  Domain x[{i}] — batas atas : "))
            domain.append((lo, hi))

        print("\nParameter (tekan Enter untuk pakai default):")
        param = {}
        keys = [
            ('m_cluster', int, 300),
            ('k_cluster', int, 10),
            ('gamma',     float, 1e-6),
            ('epsilon',   float, 1e-8),
            ('delta',     float, 1e-4),
            ('r',         float, 0.95),
            ('theta',     float, np.pi/4),
            ('m_sdoa',    int, 30),
            ('k_max',     int, 15),
            ('r_sdoa',    float, 0.95),
            ('theta_sdoa',float, np.pi/4),
            ('num_check_points', int, 1),
        ]
        for key, cast, default in keys:
            raw = input(f"  {key} [{default}]: ").strip()
            param[key] = cast(raw) if raw else default

        roots = SPOC.solve(equations, domain, param=param)
        print(f"\nSelesai. Ditemukan {len(roots)} root(s).")

    else:
        print("Pilihan tidak valid.")
