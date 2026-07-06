## REAL
# from pysne import SPOC
# import numpy as np

# f1 = lambda x: x[0]**2 + x[1]**2 - 1
# f2 = lambda x: x[0] - x[1]
# equations = [f1, f2]

# domain = [(-2, 2), (-2, 2)]

# param = {
#     'm_cluster': 300,     
#     'k_cluster': 10,     
#     # 'gamma': 1e-3,       
#     # 'epsilon': 1e-8,     
#     # 'delta': 1e-4,       
#     # 'r': 0.95,           
#     # 'theta': np.pi/4,    
#     # 'm_sdoa': 30,        
#     # 'k_max': 15,         
#     # 'r_sdoa': 0.95,
#     # 'theta_sdoa': np.pi/4,
#     # 'num_check_points': 1,
# }

# roots = SPOC.solve(equations, domain, param=param)


## EXAMPLE
# from examples.spoc_examples import run_one, run_all, PROBLEMS

# run_one("problem_2")  

# run_all()                 

# print(list(PROBLEMS.keys())) 



## INTEGER
from pysne import SPOC_int
import numpy as np

f1 = lambda x: 15*x[0] + 11*x[1] - 12
f2 = lambda x: x[0] + x[1] - 0          
f3 = lambda x: x[0]**2 + x[1]**2 - 18  
equations = [f1, f2, f3]

domain=[(-50,50), (-50,50)]

param = {
    'm_cluster': 300,     
    'k_cluster': 10,     
    'gamma': 1e-3,       
    'epsilon': 1e-8,     
    'delta': 1e-4,       
    'r': 0.95,           
    'theta': np.pi/4,    
    'm_sdoa': 30,        
    'k_max': 15,         
    'r_sdoa': 0.95,
    'theta_sdoa': np.pi/4,
    'num_check_points': 1,
}

solutions = SPOC_int.solve(equations, domain, param=None)




## EXAMPLE
# from examples.spoc_int_examples import run_one, run_all, PROBLEMS

# run_one("problem_8")
# run_all()
# print(list(PROBLEMS.keys()))
