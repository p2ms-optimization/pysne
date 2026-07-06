# pysne — Spiral-based Nonlinear Equation Solver

Pustaka modular hasil Tugas Akhir yang mengimplementasikan **SOAC** (*Spiral Optimization with Adaptive Clustering*) untuk mencari **semua** akar sistem persamaan nonlinear, baik di domain real maupun domain bilangan bulat.

---

## Struktur

```
pysne/
  src/
    pysne/
      __init__.py
      core/
        rotation.py      ← matriks rotasi Givens n-dimensi
        objective.py     ← fungsi F(x) dan generator titik Korobov/Sobol
        clustering.py    ← logika clustering (real & integer)
      SPOC.py            ← solver domain real
      SPOC_int.py        ← solver domain integer (Diophantine)
  tests/
  examples/
    spoc_examples.py     ← 7 contoh problem bab5 (real)
    spoc_int_examples.py ← 26 contoh problem bab6 (integer)
  pyproject.toml
  README.md
```

---

## Instalasi

```bash
pip install -e .
```

---

## Penggunaan: Domain Real (SPOC)

```python
from pysne import SPOC
import numpy as np

def f1(x): return x[0]**2 + x[1]**2 - 1
def f2(x): return x[0] - x[1]

domain = [(-2, 2), (-2, 2)]

# Semua parameter sudah ada default-nya — cukup ini untuk mulai:
roots = SPOC.solve([f1, f2], domain)

# Atau dengan parameter custom (hanya yang ingin diubah):
param = {
    'gamma': 1e-6,
    'epsilon': 1e-8,
    'delta': 1e-4,
    'm_sdoa': 30,
    'k_max': 15,
    'r_sdoa': 0.95,
    'theta_sdoa': np.pi/4,
    'num_check_points': 3,
}
roots = SPOC.solve([f1, f2], domain, param)
```

### Parameter SPOC (semua opsional)

| Parameter        | Default   | Keterangan                                      |
|------------------|-----------|------------------------------------------------|
| `m_cluster`      | 300       | Jumlah titik awal fase clustering              |
| `k_cluster`      | 10        | Iterasi clustering                              |
| `gamma`          | 1e-6      | Threshold pemotongan F(x) di clustering        |
| `epsilon`        | 1e-8      | Threshold penerimaan akar: 1-F(x) < ε          |
| `delta`          | 1e-4      | Jarak minimum antar akar (duplikat removal)    |
| `r`              | 0.95      | Laju kontraksi spiral (clustering)             |
| `theta`          | π/4       | Sudut rotasi (clustering)                      |
| `m_sdoa`         | 30        | Titik pencarian SDOA per cluster               |
| `k_max`          | 15        | Iterasi maksimum SDOA                          |
| `r_sdoa`         | 0.95      | Laju kontraksi spiral (SDOA)                   |
| `theta_sdoa`     | π/4       | Sudut rotasi (SDOA)                            |
| `num_check_points`| 1        | Jumlah titik tengah di clustering              |

---

## Penggunaan: Domain Integer (SPOC_int)

```python
from pysne import SPOC_int
import numpy as np

def f1(x): return 15*x[0] + 11*x[1] - 12

domain = [(-50, 50), (-50, 50)]

# Cukup ini:
solutions = SPOC_int.solve([f1], domain)

# Dengan parameter:
param = {
    'gamma': 1e-4,
    'epsilon': 1e-5,
    'delta': 0.01,
    'm_sdoa': 300,
    'k_max': 50,
    'r_sdoa': 0.988,
    'theta_sdoa': np.pi/4,
    'num_check_points': 1,
}
solutions = SPOC_int.solve([f1], domain, param)
```

### Parameter SPOC_int (semua opsional)

| Parameter         | Default   | Keterangan                                      |
|-------------------|-----------|------------------------------------------------|
| `m_cluster`       | 300       | Jumlah titik awal fase clustering              |
| `k_cluster`       | 50        | Iterasi clustering                              |
| `gamma`           | 1e-4      | Threshold pemotongan F(x) di clustering        |
| `epsilon`         | 1e-5      | Threshold penerimaan solusi: 1-F(x) ≤ ε        |
| `delta`           | 0.01      | Jarak minimum antar solusi (duplikat removal)  |
| `r`               | 0.95      | Laju kontraksi spiral (clustering)             |
| `theta`           | π/4       | Sudut rotasi (clustering)                      |
| `m_sdoa`          | 50        | Titik pencarian SDOA per cluster               |
| `k_max`           | 50        | Iterasi maksimum SDOA                          |
| `r_sdoa`          | 0.988     | Laju kontraksi spiral (SDOA)                   |
| `theta_sdoa`      | π/4       | Sudut rotasi (SDOA)                            |
| `num_check_points`| 1         | Jumlah titik tengah di clustering              |

---

## Menjalankan Examples

```bash
# Demo interaktif real domain (pilih problem atau input sendiri)
python examples/spoc_examples.py

# Demo interaktif integer domain
python examples/spoc_int_examples.py
```

Atau dari Python:

```python
from examples.spoc_examples import run_one, run_all, PROBLEMS
from examples.spoc_int_examples import run_one as run_one_int

# Jalankan satu problem
run_one("problem_1")

# Jalankan semua
run_all()

# Lihat daftar problem
print(list(PROBLEMS.keys()))
```
