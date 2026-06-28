from pysne.problems.benchmarks_multimodal import get_multimodal_problems
from pysne.solver import solve_system
 
# 1. Pilih problem bawaan (contoh: Six-Hump Camel Back)
problems = get_multimodal_problems()
prob = problems[2]()
 
# 2. Ambil domain & parameter default yang sudah dikalibrasi
domain, params = prob.get_info()
 
# 3. Jalankan solver -> mencari titik MAKSIMUM
hasil_max = solve_system(prob, params, verbose=True)
print(hasil_max['optimals'])        # array koordinat solusi
print(hasil_max['time_elapsed']) # waktu eksekusi (detik)

from pysne.problems.base import MinimizedProblem
 
# Membungkus problem yang sama untuk pencarian titik minimum
prob_min = MinimizedProblem(prob)
hasil_min = solve_system(prob_min, params, verbose=True)
 
print(f"Maksimum ditemukan: {len(hasil_max['optimals'])} titik")  
print(f"Minimum ditemukan : {len(hasil_min['optimals'])} titik")  
