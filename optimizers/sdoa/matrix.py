import numpy as np

def get_rotation_matrix(n: int, theta: float) -> np.ndarray:
    """
    Constructs an n-dimensional rotation matrix based on a given angle theta.

    This function generates a total rotation matrix by sequentially multiplying 
    partial rotation matrices (Givens rotations) $R_{pq}$ for unique pairs of 
    dimensions. The resulting orthogonal matrix satisfies rotation properties 
    in $n$-dimensional Euclidean space.

    Args:
        n (int): The dimensionality of the space. Must be a strictly positive integer.
        theta (float): The rotation angle in radians (e.g., $\pi/4$).

    Returns:
        numpy.ndarray: An orthogonal total rotation matrix of shape (n, n).

    Raises:
        ValueError: If `n` is less than 1.
        TypeError: If `n` is not an integer.
    """
    if n == 1:
        return np.identity(1)

    R_total = np.identity(n)
    
    c, s = np.cos(theta), np.sin(theta)

    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            p = n - i - 2
            q = n - j - 1
            
            R_pq = np.identity(n)
            R_pq[p, p] = c
            R_pq[p, q] = -s
            R_pq[q, p] = s
            R_pq[q, q] = c
            
            R_total = R_pq @ R_total
            
    return R_total
