"""
pysne — Spiral-based Nonlinear Equation Solver
===============================================
Implementasi pustaka modular dari Tugas Akhir:
  - SPOC      : domain real  (Sidarto & Kania, 2015)
  - SPOC_int  : domain integer / Diophantine

Cara pakai cepat
----------------
    from pysne import SPOC, SPOC_int
    import numpy as np

    # Real domain
    f1 = lambda x: x[0]**2 + x[1]**2 - 1
    f2 = lambda x: x[0] - x[1]
    roots = SPOC.solve([f1, f2], [(-2, 2), (-2, 2)])

    # Integer domain
    f  = lambda x: 15*x[0] + 11*x[1] - 12
    solutions = SPOC_int.solve([f], [(-50, 50), (-50, 50)])
"""

from . import SPOC       # noqa: F401
from . import SPOC_int   # noqa: F401

__version__ = "1.0.0"
__all__ = ["SPOC", "SPOC_int"]
