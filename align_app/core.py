import re
import os
import numpy as np
import cv2
import tifffile
import matplotlib

def natural_sort(file_list: list[str]) -> list[str]:
    """
    Sorts a list of file paths/names in-place and returns the same list
    using alphanumeric (natural) sorting.
    
    Case-insensitive. Raises TypeError if non-string values are present.
    """
    if not isinstance(file_list, list):
        raise TypeError("Input must be a list of strings")
    
    for item in file_list:
        if not isinstance(item, str):
            raise TypeError("All elements in file_list must be strings")
            
    def key_func(s: str) -> list:
        return [int(text[:4000]) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
    file_list.sort(key=key_func)
    return file_list

def find_peak_line(image_data: np.ndarray, percentile_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies coordinate indices where intensity >= threshold, applies
    intensity-weighted PCA (SVD) with iterative perpendicular outlier rejection
    and cross-section profile centering on their (x, y) coordinates.
    Returns the line origin (centroid) and the unit direction vector.
    
    Algorithm:
    1. Select pixels above the percentile threshold
    2. Weighted PCA using pixel intensities as weights (brighter = more influence)
    3. Iteratively reject points far from the fitted line (using MAD-based threshold)
    4. Re-fit on inliers until convergence
    5. Cross-section profile centering: project inliers onto perpendicular axis,
       find the intensity profile peak to get the true center of the line
    
    Raises ValueError if image dimensions are not 2D or percentile is out of [0, 100].
    """
    if not isinstance(image_data, np.ndarray) or image_data.ndim != 2:
        raise ValueError("image_data must be a 2D numpy array")
        
    if image_data.shape[0] == 0 or image_data.shape[1] == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
        
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
    if not (0.0 <= percentile_threshold <= 100.0):
        raise ValueError("percentile_threshold must be between 0.0 and 100.0")
        
    # Flat image check
    if np.max(image_data) <= np.min(image_data) + 1e-9:
        h, w = image_data.shape
        return np.array([w / 2.0, h / 2.0]), np.array([1.0, 0.0])
        
    threshold_val = np.percentile(image_data, percentile_threshold)
    rows, cols = np.where(image_data >= threshold_val)
    points = np.column_stack((cols, rows)).astype(np.float64)
    
    if len(points) < 2:
        h, w = image_data.shape
        return np.array([w / 2.0, h / 2.0]), np.array([1.0, 0.0])

    # Get intensity weights for these points (normalized to [0, 1])
    weights = image_data[rows, cols].astype(np.float64)
    weight_min = np.min(weights)
    weight_range = np.max(weights) - weight_min
    if weight_range > 1e-9:
        weights = (weights - weight_min) / weight_range
    else:
        weights = np.ones(len(points))
    # Square the weights to emphasize bright center pixels even more
    weights = weights ** 2
    # Ensure no zero weights
    weights = np.clip(weights, 1e-6, None)

    centroid, direction = _weighted_pca(points, weights)

    # Iterative outlier rejection (up to 5 iterations)
    for _ in range(5):
        if len(points) < 2:
            break
        centered = points - centroid
        perp_dist = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
        
        median_dist = np.median(perp_dist)
        mad = np.median(np.abs(perp_dist - median_dist))
        if mad < 1e-6:
            mad = 1e-6
        threshold = median_dist + 3.0 * mad
        inlier_mask = perp_dist <= threshold
        
        n_inliers = np.sum(inlier_mask)
        if n_inliers < 2 or n_inliers == len(points):
            break
            
        points = points[inlier_mask]
        weights = weights[inlier_mask]
        
        centroid, direction = _weighted_pca(points, weights)

    # ── Cross-section profile centering ──
    # After PCA gives direction + initial centroid, refine the centroid
    # by finding the true perpendicular center of the line using
    # intensity profile peak finding.
    if len(points) >= 10:
        centroid = _cross_section_center(points, weights, centroid, direction)
    
    return centroid, direction


def _cross_section_center(points: np.ndarray, weights: np.ndarray,
                          centroid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Refine the centroid by finding the peak of the perpendicular intensity profile.
    Uses vectorized np.bincount for speed.
    """
    perp = np.array([-direction[1], direction[0]])
    centered = points - centroid
    perp_dists = centered[:, 0] * perp[0] + centered[:, 1] * perp[1]
    
    dist_range = np.max(perp_dists) - np.min(perp_dists)
    if dist_range < 1e-6:
        return centroid
    
    n_bins = min(50, max(10, len(points) // 20))
    bin_edges = np.linspace(np.min(perp_dists), np.max(perp_dists), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Vectorized weighted histogram using np.bincount
    bin_indices = np.clip(np.digitize(perp_dists, bin_edges) - 1, 0, n_bins - 1)
    profile = np.bincount(bin_indices, weights=weights, minlength=n_bins)[:n_bins]
    
    if np.max(profile) < 1e-9:
        return centroid
    
    # Smooth with small kernel
    kernel_size = max(3, n_bins // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(profile, kernel, mode='same')
    
    # Peak + sub-bin refinement
    peak_idx = np.argmax(smoothed)
    window = max(1, kernel_size // 2)
    lo = max(0, peak_idx - window)
    hi = min(n_bins, peak_idx + window + 1)
    local_weights = smoothed[lo:hi]
    local_centers = bin_centers[lo:hi]
    total_w = np.sum(local_weights)
    peak_offset = np.sum(local_centers * local_weights) / total_w if total_w > 1e-9 else bin_centers[peak_idx]
    
    return centroid + peak_offset * perp


def find_peak_line_fast(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Lightweight PCA for threshold search evaluation.
    Skips outlier rejection and cross-section centering for speed.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        weights: (N,) array of intensity weights (already normalized/squared)
    
    Returns:
        (centroid, direction, median_perp_spread)
    """
    if len(points) < 2:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), float('inf')
    
    centroid, direction = _weighted_pca(points, weights)
    centered = points - centroid
    perp_dist = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
    return centroid, direction, float(np.median(perp_dist))


def compute_line_based_offset(ref_raw: np.ndarray, target_raw: np.ndarray,
                               ref_direction: np.ndarray, ref_origin: np.ndarray,
                               ref_threshold: float, target_threshold: float
                               ) -> tuple[float, float]:
    """
    Compute the translational offset needed to align a target frame's spectroscopic
    line with the reference frame's line, using line-specific analysis rather than
    whole-image phase correlation.
    
    Algorithm:
    1. Use the reference line's direction (slope) — assumed constant across frames
    2. Find the perpendicular center of the target frame's line
    3. Compute the perpendicular offset between reference and target centers
    4. Convert to (dx, dy) in image coordinates
    
    Falls back to phase correlation if line-based analysis fails.
    """
    perp = np.array([-ref_direction[1], ref_direction[0]])
    
    # Find the target frame's line using PCA (gets its own center)
    try:
        target_origin, target_dir = find_peak_line(target_raw, target_threshold)
    except Exception:
        return phase_correlation_offset(ref_raw, target_raw)
    
    # Compute the perpendicular offset from reference origin to target origin
    delta = target_origin - ref_origin
    perp_offset = delta[0] * perp[0] + delta[1] * perp[1]
    
    # Also compute the along-line offset using phase correlation for the
    # parallel component (line center shifts along the line don't change
    # the line's appearance, but the image content shifts)
    try:
        pc_dx, pc_dy = phase_correlation_offset(ref_raw, target_raw)
    except Exception:
        pc_dx, pc_dy = 0.0, 0.0
    
    # The parallel component from phase correlation
    parallel_offset = pc_dx * ref_direction[0] + pc_dy * ref_direction[1]
    
    # Reconstruct (dx, dy) = perpendicular_component + parallel_component
    dx = perp_offset * perp[0] + parallel_offset * ref_direction[0]
    dy = perp_offset * perp[1] + parallel_offset * ref_direction[1]
    
    return float(dx), float(dy)


def _weighted_pca(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted PCA on a set of 2D points.
    Returns (weighted_centroid, principal_direction_unit_vector).
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

def phase_correlation_offset(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[float, float]:
    """
    Applies OpenCV's phaseCorrelate on float32/64 single-channel arrays
    to return the translation offset (dx, dy).
    
    Raises ValueError if image dimensions mismatch.
    """
    if ref_img.shape != target_img.shape:
        raise ValueError("Reference and target images must have the same shape")
        
    if ref_img.ndim != 2 or target_img.ndim != 2:
        raise ValueError("Reference and target images must be 2D arrays")
        
    if not np.isfinite(ref_img).all() or not np.isfinite(target_img).all():
        return (0.0, 0.0)
        
    if ref_img.shape[0] < 2 or ref_img.shape[1] < 2:
        return (0.0, 0.0)
        
    if np.std(ref_img) < 1e-5 or np.std(target_img) < 1e-5:
        return (0.0, 0.0)
        
    h, w = ref_img.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_64F)
    
    shift, _ = cv2.phaseCorrelate(ref_img.astype(np.float64), target_img.astype(np.float64), window)
    dx, dy = shift
    
    if np.isnan(dx) or np.isnan(dy):
        return (0.0, 0.0)
        
    return float(dx), float(dy)

def warp_image(image_data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Warps the input image by translating it by (dx, dy) using affine warping.
    Fills empty regions with 0.
    
    Supports 2D and 3D images. Returns a copy if translation is zero.
    """
    if image_data.ndim not in (2, 3):
        raise ValueError("image_data must be 2D or 3D")
        
    if image_data.shape[0] == 0 or image_data.shape[1] == 0:
        return image_data.copy()
        
    if dx == 0.0 and dy == 0.0:
        return image_data.copy()
        
    h, w = image_data.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    warped = cv2.warpAffine(
        image_data,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def preprocess_image(image_path: str, cmap_name: str, percentile_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a float32 TIFF, normalizes values based on percentile threshold,
    applies the colormap, and returns the RGB uint8 image and raw intensity values.
    
    Raises FileNotFoundError if path doesn't exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    if not (0.0 <= percentile_threshold <= 100.0):
        raise ValueError("percentile_threshold must be between 0.0 and 100.0")
        
    raw = tifffile.imread(image_path).astype(np.float32)
    if raw.ndim != 2:
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError("TIFF image must be 2D")
            
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = np.min(raw)
    if percentile_threshold < 100.0:
        vmax = np.percentile(raw, percentile_threshold)
    else:
        vmax = np.max(raw)
        
    if vmax <= vmin:
        vmax = np.max(raw)
        
    norm_range = vmax - vmin
    if norm_range > 1e-8:
        scaled = np.clip((raw - vmin) / norm_range, 0.0, 1.0)
    else:
        scaled = np.zeros_like(raw)
        
    is_grayscale = False
    if isinstance(cmap_name, str):
        try:
            cmap_name_lower = cmap_name.lower()
            if cmap_name_lower in ["grayscale", "gray"]:
                is_grayscale = True
        except Exception:
            is_grayscale = True
    else:
        is_grayscale = True
        
    if is_grayscale:
        scaled_8bit = (scaled * 255.0).astype(np.uint8)
        rgb_image = np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)
    else:
        try:
            cmap_name_lower = cmap_name.lower()
            if hasattr(matplotlib, 'colormaps'):
                cmap = matplotlib.colormaps[cmap_name_lower]
            else:
                import matplotlib.cm as cm
                cmap = cm.get_cmap(cmap_name_lower)
            rgba_float = cmap(scaled)
            rgb_image = (rgba_float[:, :, :3] * 255.0).astype(np.uint8)
        except Exception:
            # Fall back to standard grayscale 3-channel
            scaled_8bit = (scaled * 255.0).astype(np.uint8)
            rgb_image = np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)
            
    return rgb_image, raw
