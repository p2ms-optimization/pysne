import numpy as np

class Cluster:
    """
    Representasi sebuah wilayah dalam ruang pencarian.
    
    Digunakan untuk mengelompokkan kandidat solusi agar algoritma optimasi 
    bisa fokus mencari akar di wilayah yang berbeda-beda.
    """
    def __init__(self, center: np.ndarray, radius: float):
        """
        Arguments:
            center (np.ndarray): Koordinat titik pusat cluster.
            radius (float): Jari-jari wilayah cluster.
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def __repr__(self) -> str:
        """Representasi string untuk debugging."""
        return f"Cluster(center={self.center.round(4)}, radius={self.radius:.4f})"
