import numpy as np

class Cluster:
    """
    Representation of a localized search region in the domain space.

    This class is used to group candidate solutions, allowing the optimization algorithm to focus its search for distinct roots across multiple separate 
    regions simultaneously. Mathematically, a cluster with a center x and radius p is the set of all points y satisfying ||x - y|| < p.

    Attributes
    ----------
    center : numpy.ndarray
        The coordinate point representing the center of the cluster.
    radius : float
        The radius defining the boundary of the cluster's region.
    """
    def __init__(self, center: np.ndarray, radius: float):
        """
        Initializes a new Cluster instance.

        Parameters
        ----------
        center : numpy.ndarray or array-like
            The starting central coordinate of the cluster.
        radius : float
            The initial radius of the cluster.
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Cluster object for debugging.

        Returns
        -------
        str
            A formatted string showing the rounded center coordinates and radius.
        """
        return f"Cluster(center={self.center.round(4)}, radius={self.radius:.4f})"
