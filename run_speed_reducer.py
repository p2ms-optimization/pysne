import numpy as np
from pysne.optimizers.sdoa.engine import spiral_dynamics_optimization

# ------------------ 1. DEFINE PROBLEM (Fungsi Inti) ------------------
def f(x):
    """Fungsi objektif asli (hanya untuk x yang sudah valid)"""
    x1, x2, x3, x4, x5, x6, x7 = x
    return (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2))

def constraints(x):
    """Menghitung 11 kendala g_i(x)"""
    x1, x2, x3, x4, x5, x6, x7 = x
    g1 = 27 / (x1 * x2**2 * x3) - 1
    g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
    g3 = (1.93 * x4**3) / (x2 * x3 * x6**4) - 1
    g4 = 27 / (x1 * x2**2 * x3) - 1  # duplikat g1
    g5 = (1.0 / (110 * x6**3)) * np.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6) - 1
    g6 = (1.0 / (85 * x7**3)) * np.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6) - 1
    g7 = (x2 * x3) / 40 - 1
    g8 = (5 * x2) / x1 - 1
    g9 = x1 / (12 * x2) - 1
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    g11 = (1.1 * x7 + 1.9) / x5 - 1
    return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

# ------------------ 2. FUNGSI PENALTI (REJECTION-BASED) ------------------
M_PENALTY = 1e15

# Definisikan batas bawah dan atas (HANYA untuk pengecekan, BUKAN untuk clamping)
LOW = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
HIGH = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])

def penalty_eval(points):
    """
    Evaluasi fungsi penalti dengan DEATH PENALTY untuk titik di luar domain.
    - Titik di luar batas -> dikasih np.inf (diabaikan)
    - Titik di dalam batas -> dihitung normal (dengan pembulatan x3)
    """
    # Ubah ke 2D array agar seragam
    points = np.atleast_2d(points)
    m = points.shape[0]
    results = np.zeros(m)

    for i in range(m):
        x_raw = points[i]
        
        # ========== STEP 1: CEK APAKAH TITIK DI DALAM DOMAIN? ==========
        # Jika ADA SATU SAJA variabel yang keluar batas, langsung reject!
        if np.any((x_raw < LOW) | (x_raw > HIGH)):
            results[i] = np.inf  # Diabaikan untuk iterasi ini
            continue
        # ================================================================
        
        # ========== STEP 2: TITIK SAH (di dalam domain) ==========
        x = x_raw.copy()
        
        # --- MIXED-INTEGER HANDLING (hanya x3 yang integer) ---
        # (Saya tetap pakai clip di sini karena x3 harus bulat, 
        #  dan karena x3 sudah pasti di [17,28] dari step 1, clip ini hanya memastikan rounding)
        x[2] = np.clip(np.round(x[2]), 17, 28)
        
        # Hitung objektif
        obj = f(x)
        
        # Hitung kendala & penalti
        g = constraints(x)
        violation_sum = np.sum(np.maximum(0, g))  # total pelanggaran
        penalty = M_PENALTY * violation_sum
        
        results[i] = obj + penalty
    
    # Jika inputnya 1D, kembalikan skalar
    return results[0] if m == 1 else results

# ------------------ 3. SETUP SDOA (SESUAI SIGNATURE ASLI) ------------------
# Buat domain dari batas bawah & atas
lb = np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0])
ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
domain = [(lb[i], ub[i]) for i in range(7)]

# Parameter SDOA (sesuai screenshotmu)
params = {
    'm': 10000,           # populasi
    'r': 0.99,            # radius shrink
    'theta': np.pi / 32,  # sudut spiral
    'k_max': 1000         # iterasi maksimal
}

print("Menjalankan optimasi Speed Reducer (Mixed-Integer) dengan SDOA...")
print(f"Populasi: {params['m']}, Iterasi Maks: {params['k_max']}")

# ------------------ 4. JALANKAN OPTIMASI ------------------
# return_history=True biar kita bisa lihat nilai fitness terbaik tiap iterasi
best_x, history = spiral_dynamics_optimization(
    objective_func=penalty_eval,
    domain=domain,
    params=params,
    minimization=True,          # Kita cari nilai MINIMUM
    return_history=True         # Ambil history biar tau fitness akhir
)

# ------------------ 5. TAMPILKAN HASIL ------------------
# Hitung nilai fitness akhir dari solusi terbaik
final_x = best_x.copy()
final_x[2] = np.clip(np.round(final_x[2]), 17, 28)  # pastikan integer

final_f_x = f(final_x)
g_final = constraints(final_x)
violation_sum_final = np.sum(np.maximum(0, g_final))
final_penalty = final_f_x + M_PENALTY * violation_sum_final

print("\n" + "="*60)
print("HASIL OPTIMASI SPEED REDUCER")
print("="*60)
print(f"x1  = {final_x[0]:.6f}")
print(f"x2  = {final_x[1]:.6f}")
print(f"x3  = {int(final_x[2])}  <--- INTEGER")
print(f"x4  = {final_x[3]:.6f}")
print(f"x5  = {final_x[4]:.6f}")
print(f"x6  = {final_x[5]:.6f}")
print(f"x7  = {final_x[6]:.6f}")
print("-"*60)
print(f"Nilai f(x) aktual   = {final_f_x:.8f}")
print(f"Nilai Penalti F(x)  = {final_penalty:.8f} (seharusnya = f(x) jika feasible)")
print(f"Nilai terbaik dari history SDOA = {history[-1]:.8f}")

# Cek kelayakan
violations = g_final[g_final > 0]
if len(violations) == 0:
    print("\n✅ STATUS: FEASIBLE (Semua g_i <= 0)")
else:
    print(f"\n❌ STATUS: INFEASIBLE (Masih ada {len(violations)} pelanggaran)")
    print(f"Nilai g_i yang melanggar: {violations}")

print("\nSeluruh g_i:")
for i, gi in enumerate(g_final, 1):
    print(f"  g{i:2d} = {gi:.6e}")

print(f"\n⏱️  Waktu eksekusi tergantung spek PC (di screenshot ~64 detik)")