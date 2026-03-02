# Project PySNE

A Python library for finding all solutions of nonlinear equation systems using Spiral Dynamics Optimization Algorithm (SDOA) with Clustering Technique. 
Research for Undergraduate Thesis at Actuarial Science ITB.



## Repository Structure

```
pysne/                          <-- Root Folder (Repositori GitHub)
│
├── sdoa_cluster/               <-- Folder Library Utama
│   ├── __init__.py             # Gatekeeper utama (import SDOA, Clustering, dll)
│   │
│   ├── initialization/         
│   │   ├── __init__.py         # Ekspos: generate_sobol_points
│   │   └── sampling.py         # Kode Sobol Sequence & SciPy QMC
│   │
│   ├── problems/               
│   │   ├── __init__.py         # Ekspos: get_problem_set
│   │   └── benchmarks.py       # Daftar Problem 1-7 (Equations, Domain, Params)
│   │
│   ├── clustering/             
│   │   ├── __init__.py         # Ekspos: DynamicClustering
│   │   └── dynamic.py          # Logika 5 Kasus Clustering (Pusat Cluster)
│   │
│   ├── core/                   
│   │   ├── __init__.py         # Ekspos: SDOA
│   │   └── spiral.py           # Logika SDOA n-Dimensi & Matriks Rotasi
│   │
│   └── utils.py                # Fungsi objektif (1/1+sigma|f(x)|) & Helper umum
│
├── tests/                      # Folder Pengujian (Unit Testing)
│   ├── test_initialization.py
│   └── test_problems.py
│
├── test_lib.py                 # Script penguji sementara (Main Entry)
├── requirements.txt            # numpy, scipy, matplotlib
├── .gitignore                  # Mengabaikan file sampah (__pycache__)
└── README.md                   # Dokumentasi Proyek
└── README.md                   # Dokumentasi Proyek
```

