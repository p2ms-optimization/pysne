"""
pysne.core.rotation
-------------------
Rotation matrix builder for Spiral Dynamics Optimization Algorithm (SDOA).
"""

import numpy as np


def get_rotation_matrix(n: int, theta: float) -> np.ndarray:
    """
    Build an n-dimensional rotation matrix by composing all Givens rotations
    in the (i, j) plane for i < j.

    Parameters
    ----------
    n     : int   — dimension of the space
    theta : float — rotation angle in radians

    Returns
    -------
    R : np.ndarray, shape (n, n)
    """
    if n == 1:
        return np.array([[-1.0]])

    c, s = np.cos(theta), np.sin(theta)
    R_total = np.identity(n)

    for i in range(n - 1):
        for j in range(i + 1, n):
            R_ij = np.identity(n)
            R_ij[i, i] =  c;  R_ij[i, j] = -s
            R_ij[j, i] =  s;  R_ij[j, j] =  c
            R_total = R_total @ R_ij

    return R_total
