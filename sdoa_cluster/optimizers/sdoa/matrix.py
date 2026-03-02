import numpy as np

def get_rotation_matrix(n: int, theta: float) -> np.ndarray:
    """
    Membangun matriks rotasi n-dimensi berdasarkan sudut theta.
    
    Fungsi ini menghasilkan matriks rotasi total dengan mengalikan matriks 
    rotasi parsial $R_{pq}$ untuk setiap kombinasi dimensi yang unik. 
    Matriks yang dihasilkan memenuhi properti rotasi di ruang Euclidean $n$-dimensi.

    Arguments:
        n (int): Dimensi dari ruang pencarian (jumlah variabel).
        theta (float): Sudut rotasi dalam radian (biasanya $\pi/4$ dalam SDOA standar).

    Returns:
        np.ndarray: Matriks rotasi total berukuran (n, n).
    """
    if n == 1:
        return np.identity(1)

    # Inisialisasi matriks identitas berukuran n x n
    R_total = np.identity(n)
    
    # Pre-calculate nilai cos dan sin untuk efisiensi
    c, s = np.cos(theta), np.sin(theta)

    # Loop untuk membangun matriks rotasi melalui perkalian matriks parsial
    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            p = n - i - 2
            q = n - j - 1
            
            # Membangun matriks rotasi parsial R_pq
            R_pq = np.identity(n)
            R_pq[p, p] = c
            R_pq[p, q] = -s
            R_pq[q, p] = s
            R_pq[q, q] = c
            
            # Akumulasi rotasi menggunakan perkalian matriks (@)
            R_total = R_pq @ R_total
            
    return R_total
