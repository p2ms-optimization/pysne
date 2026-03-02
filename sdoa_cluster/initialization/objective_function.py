import numpy as np
from typing import List, Callable, Any

def objective_function(x: np.ndarray, system_of_equations: List[Callable[[np.ndarray], float]]) -> float:
    """
    Menghitung nilai fitness berdasarkan total residu absolut dari sistem persamaan.
    
    Fungsi ini mengubah masalah pencarian akar menjadi masalah maksimasi dengan formula:
    F(x) = 1 / (1 + sum |f_i(x)|)

    Arguments:
        x (np.ndarray): Vektor variabel input atau kandidat solusi.
        system_of_equations (List[Callable]): Daftar fungsi yang merepresentasikan 
            sistem persamaan f_i(x) = 0.

    Returns:
        float: Nilai fitness dalam rentang (0, 1]. Nilai mendekati 1 menunjukkan 
            solusi yang lebih akurat.
    """
    try:
        # Menghitung jumlah nilai absolut dari setiap persamaan dalam sistem
        # Menggunakan np.abs untuk mendukung operasi vektor jika f_i mengembalikan array
        sum_of_abs_f = sum(abs(f_i(x)) for f_i in system_of_equations)
        
        return 1.0 / (1.0 + sum_of_abs_f)
        
    except (TypeError, ValueError) as e:
        # Menangkap error spesifik input daripada Exception umum
        print(f"Error dalam perhitungan objective_function: {e}")
        return 0.0
