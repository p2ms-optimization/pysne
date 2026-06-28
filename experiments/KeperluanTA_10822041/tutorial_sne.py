from pysne.problems.benchmarks_sne import get_problem_set
from pysne.solver import solve_system
 
# 1. Memanggil koleksi sistem persamaan bawaan
problems = get_problem_set()
prob = problems[1]()  # Problem 1: sistem Chen dkk. (1999)
 
# 2. Mengambil domain pencarian dan parameter default
domain, params = prob.get_info()
print(domain)
# [(-10, 10), (-10, 10)]
 
# 3. Menjalankan solver untuk mencari seluruh akar
hasil = solve_system(prob, params, verbose=True)

# print(hasil)