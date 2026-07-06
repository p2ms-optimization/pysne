# ROADMAP & STATUS PEKERJAAN — pysne
*Dokumen ini dibuat agar sesi Claude berikutnya (atau siapapun) bisa langsung melanjutkan.*

---

## KONTEKS PROYEK

**Tugas Akhir:** Implementasi metode SOAC (Spiral Optimization with Adaptive Clustering) untuk mencari semua akar sistem persamaan nonlinear.

**Dua solver utama:**
- `SPOC`     → domain real (bab5: Sidarto & Kania, 2015)
- `SPOC_int` → domain integer / Diophantine (bab6: Sumarti et al., 2023)

**File asli mahasiswa:**
- `ta1_bab5-2.py` → solver real domain (monolitik)
- `ta1_bab6_2.py` → solver integer domain (monolitik)

**Tujuan:** Mengubah kedua file menjadi pustaka Python modular `pysne` dengan struktur yang rapi, parameter default, dan contoh problem siap pakai untuk demo sidang.

---

## KEPUTUSAN DESAIN YANG SUDAH DIBUAT

| Komponen | Keputusan | Alasan |
|---|---|---|
| `get_rotation_matrix` | Pakai versi bab6 (loop `i<j`, Givens standard) | Lebih bersih secara matematis |
| `objective_function` | Dipisah: `objective_function` (real) dan `objective_function_int` (integer) | Evaluasi beda: real langsung, integer lewat `np.round()` dulu |
| `num_check_points` | Ada di keduanya; SPOC default=1, SPOC_int default=1 | bab5 aslinya 1 midpoint; bab6 sudah support multiple |
| Global best memory di SPOC_int | ✅ Dipertahankan | Pemisahan domain pencarian (kontinu) vs evaluasi (diskret) butuh ini |
| API | `SPOC.solve(equations, domain, param=None)` | Sesuai gambar di slide presentasi mahasiswa |

---

## STATUS FILE — SELESAI ✅

```
pysne/
  src/pysne/
    __init__.py              ✅ expose SPOC dan SPOC_int
    core/
      __init__.py            ✅
      rotation.py            ✅ get_rotation_matrix (versi bab6)
      objective.py           ✅ objective_function, objective_function_int,
                                generate_korobov_points, generate_sobol_points
      clustering.py          ✅ Cluster class, is_in_domain,
                                perform_iterative_clustering_real,
                                perform_iterative_clustering_int,
                                create_continuous_bounds
    SPOC.py                  ✅ solver real domain, API publik: solve()
    SPOC_int.py              ✅ solver integer domain, API publik: solve()
  examples/
    __init__.py              ✅
    spoc_examples.py         ✅ 7 problem bab5 + demo interaktif (mode 1/2/3)
    spoc_int_examples.py     ✅ 26 problem bab6 + demo interaktif (mode 1/2/3)
  tests/
    __init__.py              ✅ (kosong, belum ada unit test)
  pyproject.toml             ✅
  README.md                  ✅ dokumentasi lengkap

```

---

## APA YANG BELUM / BISA DILANJUTKAN

### 1. Unit Tests (opsional, tidak wajib untuk demo)
File: `tests/test_spoc.py` dan `tests/test_spoc_int.py`
Isi yang direkomendasikan:
- Smoke test: problem_1 dari masing-masing solver
- Assert bahwa jumlah root ≥ expected * 0.8

### 2. Verifikasi instalasi
Setelah extract zip:
```bash
cd pysne
pip install -e .
python -c "from pysne import SPOC, SPOC_int; print('OK')"
```

### 3. Verifikasi cepat satu problem (sanity check sebelum demo)
```python
# Test SPOC
from pysne import SPOC
import numpy as np
f1 = lambda x: x[0]**2 + x[1]**2 - 1
f2 = lambda x: x[0] - x[1]
roots = SPOC.solve([f1, f2], [(-2,2),(-2,2)])
# Harusnya 2 roots: [√2/2, √2/2] dan [-√2/2, -√2/2]

# Test SPOC_int
from pysne import SPOC_int
f = lambda x: 15*x[0] + 11*x[1] - 12
sols = SPOC_int.solve([f], [(-50,50),(-50,50)])
# Harusnya ~7 solusi
```

### 4. Visualisasi (opsional, untuk presentasi lebih menarik)
Kode visualisasi (`plot_final_solution_2d`, `plot_weierstrass_solution`) dari bab5 bisa dipindahkan ke modul `pysne/core/visualization.py` dan dipanggil dari `SPOC.solve(..., plot=True)`.

### 5. Problem 4 (bab5) tidak diikutkan di run_all
Di kode asli bab5, `problem_4` di-comment-out di `run_all_problems()`. Ini sudah dipertahankan di `spoc_examples.py` — problem_4 ada di registry PROBLEMS dan bisa dijalankan lewat `run_one("problem_4")`, tapi tidak masuk `run_all()`. Jika ingin dimasukkan, tambahkan ke loop `run_all()`.

---

## CARA DEMO (SKENARIO SIDANG)

### Skenario A — Contoh yang sudah ada
```bash
cd pysne
python examples/spoc_examples.py
# Pilih: 1 → problem_1
```

```bash
python examples/spoc_int_examples.py
# Pilih: 1 → problem_1
```

### Skenario B — Input sendiri
```bash
python examples/spoc_examples.py
# Pilih: 3 → masukkan persamaan manual
```

### Skenario C — Dari script Python langsung
```python
from pysne import SPOC
import numpy as np

f1 = lambda x: x[0]**2 + x[1]**2 - 1
f2 = lambda x: x[0] - x[1]
roots = SPOC.solve([f1, f2], [(-2, 2), (-2, 2)])
```

---

## DEPENDENSI

```
numpy
scipy
pydoe    ← untuk Korobov sequence generator
```

Install: `pip install numpy scipy pydoe`

---

## CATATAN TEKNIS PENTING

- **`vectorised` spiral update di clustering real** menggunakan broadcasting NumPy:
  `points = (S_n @ points.T - (S_n - I_n)[:, np.newaxis] * x_p[:, np.newaxis]).T`
  Ini setara dengan loop aslinya di bab5 tapi lebih cepat.

- **SPOC_int** menggunakan `np.round(x).astype(object)` di `objective_function_int` supaya
  operasi seperti `11**v[2]` tidak overflow saat v[2] besar (Python int arbitrary precision).

- **Margin continuous bounds** di SPOC_int = 0.5 (hardcoded di `create_continuous_bounds`),
  sesuai implementasi asli bab6.

- **Cluster radius minimum** di `run_sdoa_on_clusters` SPOC_int di-clamp ke ≥ 1.0
  agar selalu mencakup setidaknya integer tetangga terdekat.
