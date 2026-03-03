# Project PySNE

A Python library for finding all solutions of nonlinear equation systems using Spiral Dynamics Optimization Algorithm (SDOA) with Clustering Technique. 
Research for Undergraduate Thesis at Actuarial Science ITB.

## Repository Structure
```
pysne/                          <-- Root Folder
│
├── sdoa_cluster/               <-- Folder Library Utama
│   ├── __init__.py             # Gatekeeper utama (import SDOA, Clustering, dll)
│   │
│   ├── initialization/         
│   │   ├── __init__.py         # Ekspos: generate_sobol_points
│   │   └── sampling.py         # Kode Sobol Sequence  & SciPy QMC
│   │
│   ├── problems/               
│   │   ├── __init__.py         # Ekspos: get_problem_set
│   │   └── benchmarks.py       # Daftar Problem 1-7 (Equations, Domain, Params)
│   │
│   ├── clustering/             
│   │   ├── __init__.py         # Ekspos: Cluster, DynamicClustering
│   │   ├── dynamic.py          # Logika Kasus Clustering
│   │   └── model.py            # Representasi objek Cluster
│   │
│   ├── optimizers/             
│   │   └── sdoa/
│   │       ├── __init__.py
│   │       ├── engine.py       # Mesin utama iterasi SDOA
│   │       └── matrix.py       # Pembangkit Matriks Rotasi n-Dimensi
│   │
│   └── utils.py                # Objective Function & Domain Checker
│
├── tests/                      # Unit Testing
│   ├── test_initialization.py
│   └── test_problems.py
│
├── test_lib.py                 # Script penguji integrasi
├── CHANGELOG.md                # Log perubahan proyek
└── README.md                   # Dokumentasi utama Proyek
```

