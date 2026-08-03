import numpy as np

def get_rotation_matrix(n: int, theta: float) -> np.ndarray:
    """
    Constructs an n-dimensional rotation matrix based on a given rotation angle.

    This function generates the total rotation matrix by multiplying partial rotation matrices for every unique combination of 2D planes in the 
    n-dimensional Euclidean space. The resulting matrix satisfies the orthogonal properties of a valid rotation matrix.

    Parameters
    ----------
    n : int
        The dimensionality of the search space (number of variables).
    theta : float
        The rotation angle in radians (typically pi/4 or pi/2 in standard SPO).

    Returns
    -------
    numpy.ndarray
        A total rotation matrix of shape (n, n).
    """
    if n == 1:
        return np.identity(1)

    # Initialize the n x n identity matrix
    R_total = np.identity(n)
    
    # Pre-calculate cos and sin values for efficiency
    c, s = np.cos(theta), np.sin(theta)

    # Loop to build the total rotation matrix through partial matrix multiplication
    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            p = n - i - 2
            q = n - j - 1
            
            # Construct the partial rotation matrix R_pq
            R_pq = np.identity(n)
            R_pq[p, p] = c
            R_pq[p, q] = -s
            R_pq[q, p] = s
            R_pq[q, q] = c
            
            # Accumulate the rotation using matrix multiplication
            R_total = R_pq @ R_total
            
    return R_total
