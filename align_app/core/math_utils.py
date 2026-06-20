import numpy as np

def _weighted_pca(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute intensity-weighted Principal Component Analysis (PCA) on a coordinate dataset.

    Mathematics Context:
    1. Weighted Centroid: Computes the intensity-weighted center of mass of the point cloud:
       C = sum(points * weights) / sum(weights)
    2. SVD Covariance Solution: Centered coordinate vectors are scaled by the square root of their 
       weights (`sqrt_w = sqrt(weights)`), forming a weighted data matrix. Singular Value Decomposition 
       (SVD) is performed on this matrix:
       U, S, Vt = svd(weighted_centered)
       The first row of `Vt` represents the eigenvector associated with the largest singular value, 
       which defines the principal direction of the beam line.
    3. Vector Ambiguity Resolution: SVD eigenvectors have a 180-degree sign ambiguity. We enforce a 
       consistent orientation (pointing towards positive x, or positive y if horizontal component is 0) 
       to prevent coordinate flips in downstream calculations.
    4. Poisson Noise Robustness: Very high, because the calculation integrates coordinate distributions 
       over the entire beam profile, averaging out random fluctuations.

    Args:
        points: (N, 2) float64 array of coordinates.
        weights: (N,) float64 array of intensity weights.

    Returns:
        tuple[np.ndarray, np.ndarray]: (weighted_centroid, principal_direction_unit_vector).
    """
    if len(points) < 2:
        return np.mean(points, axis=0), np.array([1.0, 0.0])
    
    # Weighted centroid
    total_weight = np.sum(weights)
    if total_weight < 1e-9:
        total_weight = 1.0
    centroid = np.sum(points * weights[:, np.newaxis], axis=0) / total_weight
    
    # Weighted covariance via SVD on sqrt(w)-scaled centered points
    centered = points - centroid
    sqrt_w = np.sqrt(weights)
    weighted_centered = centered * sqrt_w[:, np.newaxis]
    
    _, _, Vt = np.linalg.svd(weighted_centered, full_matrices=False)
    direction = Vt[0, :]
    
    # Ensure consistent direction (positive x or positive y if x~0)
    if direction[0] < -1e-9 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
        direction = -direction
    
    return centroid, direction
