import numpy as np
import warnings
from typing import List, Tuple, Callable

def objective_function(
    x: np.ndarray, 
    system_of_equations: List[Callable[[np.ndarray], float]]
) -> float:
    """
    Calculates the fitness value based on the total absolute residual of the system of equations.
    
    This function transforms the root-finding problem into a maximization problem using the formula: F(x) = 1 / (1 + sum |f_i(x)|).

    Parameters
    ----------
    x : numpy.ndarray
        The input variable vector or candidate solution.
    system_of_equations : list of callable
        A list of functions representing the system of equations f_i(x) = 0.

    Returns
    -------
    float
        The fitness value in the range (0, 1]. A value approaching 1.0 indicates a highly accurate root. Returns 0.0 if mathematical 
        evaluation fails (e.g., division by zero or overflow).
    """
    try:
        # Calculate the sum of absolute values for each equation in the system
        sum_of_abs_f = sum(abs(f_i(x)) for f_i in system_of_equations)
        
        return 1.0 / (1.0 + sum_of_abs_f)
        
    except (TypeError, ValueError, ZeroDivisionError) as e:
        # Catch specific mathematical or input errors and issue a safe warning
        warnings.warn(f"Mathematical evaluation failed in objective_function: {e}", RuntimeWarning)
        return 0.0

def is_in_domain(point: np.ndarray, domain: List[Tuple[float, float]]) -> bool:
    """
    Checks whether a given coordinate point lies strictly within the defined domain boundaries.

    Parameters
    ----------
    point : numpy.ndarray
        The coordinate point to be evaluated.
    domain : list of tuple
        The boundaries of the search space for each dimension in the format [(min_1, max_1), (min_2, max_2), ...].

    Returns
    -------
    bool
        True if the point is within the domain boundaries, False otherwise.
    """
    for i, (lo, hi) in enumerate(domain):
        if not (lo <= point[i] <= hi):
            return False
    return True

def validate_solutions(
    roots: List[np.ndarray], 
    equations: List[Callable], 
    domain: List[Tuple[float, float]], 
    epsilon: float
) -> List[np.ndarray]:
    """
    Validates a list of candidate roots by ensuring they strictly fall within the domain and their maximum absolute residual is below the 
    specified tolerance.

    Parameters
    ----------
    roots : list of numpy.ndarray
        The list of candidate roots found by the solver.
    equations : list of callable
        The system of nonlinear equations to verify against.
    domain : list of tuple
        The defined search space boundaries.
    epsilon : float
        The maximum acceptable residual for a point to be considered a valid root.

    Returns
    -------
    list of numpy.ndarray
        A filtered list containing only the coordinate points that meet both the domain and accuracy criteria.
    """
    valid_roots = []
    for root in roots:
        in_domain = is_in_domain(root, domain)
        residuals = [abs(f(root)) for f in equations]
        if max(residuals) < epsilon and in_domain:
            valid_roots.append(root)
    return valid_roots

def create_continuous_bounds(
    integer_domain: List[Tuple[int, int]],
    margin: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Memperluas setiap dimensi integer_domain sebesar margin ke kiri dan kanan.

    Tujuan: titik spiral bebas bergerak di ruang kontinu yang sedikit
    lebih lebar dari grid integer, sehingga titik-titik di tepi domain
    integer tetap bisa dievaluasi dengan benar.

    Contoh:
        integer_domain = [(-50, 50), (-50, 50)]
        create_continuous_bounds(integer_domain, margin=0.5)
        → [(-50.5, 50.5), (-50.5, 50.5)]

    Parameters
    ----------
    integer_domain : list of tuple
        Batas bilangan bulat per dimensi, misal [(-50, 50), (-50, 50)].
    margin : float
        Besarnya perluasan ke kiri dan kanan. Default 0.5 (setengah jarak antar integer).

    Returns
    -------
    list of tuple
        Continuous bounds yang sudah diperluas.
    """
    return [(lo - margin, hi + margin) for lo, hi in integer_domain]

def penalty_function(
    f_val: float,
    g_vals: List[float],
    M: float = 1e15
) -> float:
    """
    Static/exterior penalty function untuk masalah constrained optimization:

        F(x) = f(x) + M * sum(max(0, g_i(x)))

    Parameters
    ----------
    f_val : float
        Nilai fungsi objektif asli f(x) (yang ingin diminimumkan).
    g_vals : list of float
        Nilai-nilai fungsi kendala g_i(x). Kendala dianggap melanggar batas
        jika g_i(x) > 0.
    M : float
        Koefisien penalti (default 1e15, sesuai konvensi umum di literatur).

    Returns
    -------
    float
        Nilai F(x) setelah penalti ditambahkan.
    """
    violation = sum(np.maximum(0.0, g) for g in g_vals)
    return f_val + M * violation


def create_mixed_bounds(
    raw_domain: List[Tuple[float, float]],
    integer_dims,
    margin: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Seperti create_continuous_bounds, tapi margin hanya diterapkan pada
    dimensi yang ditandai sebagai integer (integer_dims). Dimensi kontinu
    dibiarkan apa adanya.

    Parameters
    ----------
    raw_domain : list of tuple
        Batas asli per dimensi, misal [(2.6, 3.6), (0.7, 0.8), (17, 28), ...].
    integer_dims : iterable of int
        Indeks (0-based) dimensi yang bernilai integer.
    margin : float
        Besarnya perluasan ke kiri/kanan untuk dimensi integer saja.

    Returns
    -------
    list of tuple
        Domain kontinu yang siap dipakai solver, dengan dimensi integer
        diperlebar agar titik-titik di tepi grid tetap terjangkau.
    """
    integer_dims = set(integer_dims)
    bounds = []
    for i, (lo, hi) in enumerate(raw_domain):
        if i in integer_dims:
            bounds.append((lo - margin, hi + margin))
        else:
            bounds.append((lo, hi))
    return bounds

def filter_unique_roots(candidates: List[Tuple[np.ndarray, float]], delta: float) -> np.ndarray:
    """
    Filters candidates such that only unique roots are kept.
    Each candidate in the input list is a tuple of (coordinate_point, fitness_value).
    If two points are closer than delta, only the one with the higher fitness value is retained.
    """
    if not candidates:
        return np.array([])
    
    # Sort in descending order based on fitness value
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    final_roots = []
    for cand, f_val in sorted_candidates:
        found_close = False
        for i, (existing, existing_f) in enumerate(final_roots):
            if np.linalg.norm(cand - existing) <= delta:
                found_close = True
                if f_val > existing_f:
                    final_roots[i] = (cand, f_val)
                break
        if not found_close:
            final_roots.append((cand, f_val))
            
    return np.array([root for root, _ in final_roots])

def calculate_sobol_discrepancy(
    num_points: int = None, 
    dimension: int = None, 
    points: np.ndarray = None, 
    domain: List[Tuple[float, float]] = None
) -> float:
    """
    Menghitung nilai discrepancy dari distribusi titik.
    Jika `points` diberikan, menghitung discrepancy dari titik tersebut (diskalakan kembali ke [0, 1]^d jika `domain` diberikan).
    Jika tidak, men-generate titik Sobol baru menggunakan scipy.stats.qmc pada dimensi dan jumlah titik tertentu.
    
    Parameters
    ----------
    num_points : int, optional
        Jumlah titik sampel yang digenerasikan (jika points tidak disuapkan).
    dimension : int, optional
        Dimensi dari ruang pencarian (jika points tidak disuapkan).
    points : numpy.ndarray, optional
        Titik-titik sampel yang akan dihitung nilai discrepancy-nya.
    domain : list of tuple, optional
        Batas pencarian untuk penskalaan kembali titik sampel ke [0, 1]^d.
        
    Returns
    -------
    float
        Nilai discrepancy (discrepancy yang lebih rendah menunjukkan distribusi yang lebih merata).
    """
    from scipy.stats import qmc
    
    if points is not None:
        try:
            pts = np.asarray(points)
            if domain is not None:
                lower_bounds = np.array([d[0] for d in domain])
                upper_bounds = np.array([d[1] for d in domain])
                denom = upper_bounds - lower_bounds
                denom[denom == 0] = 1.0
                pts = (pts - lower_bounds) / denom
            pts = np.clip(pts, 0.0, 1.0)
            discrepancy_val = qmc.discrepancy(pts)
            return float(discrepancy_val)
        except Exception as e:
            warnings.warn(f"Gagal menghitung discrepancy dari points: {e}", RuntimeWarning)
            return -1.0

    if num_points is None or dimension is None:
        warnings.warn("Parameter num_points dan dimension harus diberikan jika points tidak disuapkan.", RuntimeWarning)
        return -1.0

    # Menggunakan Sobol sampler bawaan scipy
    sampler = qmc.Sobol(d=dimension, scramble=True)
    
    try:
        points_gen = sampler.random(n=num_points)
        discrepancy_val = qmc.discrepancy(points_gen)
        return float(discrepancy_val)
    except Exception as e:
        warnings.warn(f"Gagal menghitung discrepancy: {e}", RuntimeWarning)
        return -1.0
    
def sort_unique_roots(roots, sort=False):
    """
    Menghapus solusi duplikat berdasarkan nilai, dengan opsi mengabaikan urutan.

    Parameters
    ----------
    roots : list of tuple
        Solusi-solusi yang sudah terseleksi (masing-masing sebagai tuple integer).
    sort : bool
        Jika True, setiap solusi diurutkan (sorted) sebelum dibandingkan,
        sehingga solusi yang hanya berbeda urutan dianggap sama.
        Jika False, hanya solusi identik persis yang akan dihapus.

    Returns
    -------
    list of tuple
        Solusi unik dalam format tuple.
    """
    seen = set()
    unique = []
    for root in roots:
        # Buat kunci: urutkan jika sort=True, jika tidak gunakan aslinya
        key = tuple(sorted(root)) if sort else tuple(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique