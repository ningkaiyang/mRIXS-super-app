import numpy as np
import cv2
from align_app.core.math_utils import _weighted_pca

def _cross_section_center(points: np.ndarray, weights: np.ndarray,
                          centroid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Refine the initial PCA centroid by locating the peak of the perpendicular intensity profile.
    
    Mathematics & Alignment Context:
      1. Projecting points: Projects the coordinate vectors of above-threshold pixels onto the unit 
         vector perpendicular to the line direction. This maps the 2D point cloud into a 1D coordinate 
         space representing distance from the PCA line.
      2. Dynamic Binning: Discretizes the perpendicular distances into a 1D histogram using a vectorized 
         binning method (`np.bincount`), where bin count is dynamically scaled based on point density 
         (clamped between 10 and 50). Bin weights correspond to pixel intensities.
      3. Poisson Noise Mitigation: Convolves the 1D intensity profile with a uniform smoothing kernel 
         to smooth out high-frequency fluctuations caused by shot noise (Poisson statistics) or hot pixels.
      4. Sub-bin Centroid Refinement: Locates the peak of the smoothed profile and performs localized center-of-mass 
         interpolation within a window around the peak. This yields a sub-pixel shift along the perpendicular 
         axis, correcting the PCA centroid to align precisely with the intensity peak, which represents the 
         true physical center of the RIXS beam profile.

    Args:
        points: np.ndarray of shape (N, 2) containing (x, y) coordinate indices.
        weights: np.ndarray of shape (N,) containing corresponding pixel intensities.
        centroid: np.ndarray of shape (2,) representing the initial PCA centroid (x, y).
        direction: np.ndarray of shape (2,) representing the unit direction vector of the line.

    Returns:
        np.ndarray: The refined (x, y) sub-pixel centroid coordinates perpendicular to the line.
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

def find_peak_line(image_data: np.ndarray, percentile_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the dominant peak line in a 2D image via intensity-weighted PCA (SVD) and outlier rejection.

    Physics Context:
    In Resonant Inelastic X-ray Scattering (RIXS) and other synchrotron beamlines, the energy-dispersive 
    spectroscopic line drifts over time due to thermal fluctuations of monochromator crystals, mirrors, or 
    beam instabilities. This drift degrades energy resolution. Since individual frames are governed by 
    Poisson (shot) noise and low photon counts, extracting a stable, sub-pixel accurate peak line is crucial 
    before frame summation.

    Mathematical Algorithm:
    1. Percentile Thresholding: Selects pixels above the specified percentile. By targeting the top 
       fraction (e.g., 99.9%), we isolate the high-intensity line core from low-frequency detector noise 
       and cosmic rays, ensuring high signal-to-noise ratio (SNR) for the subsequent fit.
    2. Intensity Weighting: Normalizes pixel intensities to [0, 1] and squares them. Squaring increases 
       the influence of the brightest pixels at the exact beam center, making the fit highly robust 
       against Poisson fluctuations in the line wings.
    3. Singular Value Decomposition (SVD): Solves the weighted covariance matrix of the 2D coordinates. 
       The right singular vector corresponding to the largest singular value defines the line direction.
    4. Iterative Outlier Rejection: Computes perpendicular distances of points from the fitted line. 
       Points exceeding `Median + 3.0 * MAD` (Median Absolute Deviation) are rejected as outliers (e.g. 
       cosmic ray strikes or hot pixels). SVD is re-run on inliers for up to 5 iterations or convergence.
    5. Cross-Section Profile Centering: Refines the centroid perpendicular to the line using a 
       smoothed 1D intensity profile histogram, providing sub-pixel centroid correction.

    Args:
        image_data: 2D numpy array of raw float32 intensities.
        percentile_threshold: Value in [0, 100] specifying the intensity cutoff.

    Returns:
        tuple[np.ndarray, np.ndarray]: (centroid, direction_unit_vector) representing the line.

    Raises:
        ValueError: If image_data is not a 2D array, or if percentile_threshold is outside [0.0, 100.0].
    """
    if not isinstance(image_data, np.ndarray) or image_data.ndim != 2:
        raise ValueError("image_data must be 2D numpy array")
        
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
    if percentile_threshold == 0.0:
        weights = np.ones(len(points))
    else:
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

    if percentile_threshold == 0.0:
        return centroid, direction

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

def find_peak_line_fast(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Lightweight, high-performance PCA designed for threshold sweep operations.

    Mathematics & Performance:
    Bypasses the iterative outlier rejection and cross-section profile centering steps to minimize SVD 
    computation latency. This is optimized for automated parameter optimization (Auto/Auto All sweeps), 
    where hundreds of threshold evaluations are executed. It measures the line's perpendicular spread 
    as the median absolute perpendicular distance from the fitted line.

    Args:
        points: (N, 2) float64 array of coordinates.
        weights: (N,) float64 array of normalized weights.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: (centroid, direction_unit_vector, median_perpendicular_spread).
    """
    if len(points) < 2:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), float('inf')
    
    centroid, direction = _weighted_pca(points, weights)
    centered = points - centroid
    perp_dist = np.abs(centered[:, 0] * direction[1] - centered[:, 1] * direction[0])
    return centroid, direction, float(np.median(perp_dist))

def phase_correlation_offset(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[float, float]:
    """
    Calculate the translation vector (dx, dy) between two frames using frequency-domain phase correlation.

    Mathematics & Physics Context:
    1. Cross-Power Spectrum: Computes the 2D Fast Fourier Transform (FFT) of the reference and target 
       images. The normalized cross-power spectrum is computed in the frequency domain as:
       R = (F * G*) / |F * G*|
       where F and G* are the FFT of the reference and the complex conjugate of the target, respectively.
    2. Inverse FFT: Taking the inverse FFT of R yields a Dirac delta peak (or a sharp Kronecker delta-like peak) 
       at the location of the spatial translation.
    3. Sub-pixel Estimation: Uses SVD and centroid fitting around the peak of the 2D correlation matrix to 
       estimate the shift vector with sub-pixel resolution.
    4. Boundary Mitigation (Hanning Window): Applies a 2D Hanning window to both images before FFT. 
       This attenuates intensities near the borders to zero, eliminating high-frequency spectral leakage 
       caused by non-periodic image boundaries.
    5. Noise Robustness: The normalization step makes phase correlation highly robust to low-frequency 
       intensity variations, uniform gain variations, and Poisson noise, since the correlation phase 
       contains the structural offset information.

    Args:
        ref_img: 2D float32 numpy array representing the reference frame.
        target_img: 2D float32 numpy array representing the target frame.

    Returns:
        tuple[float, float]: (dx, dy) representing the sub-pixel translation vector. Returns (0.0, 0.0) 
                             if computation fails or if standard deviation is below 1e-5.

    Raises:
        ValueError: If reference and target shapes do not match, or if they are not 2D.
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

def compute_line_based_offset(ref_raw: np.ndarray, target_raw: np.ndarray,
                               ref_direction: np.ndarray, ref_origin: np.ndarray,
                               ref_threshold: float, target_threshold: float
                               ) -> tuple[float, float]:
    """
    Calculate the translation vector (dx, dy) between a target frame and reference frame using line-based projection.

    Physics & Mathematics Context:
    - RIXS Drift Characteristics: Spectroscopic lines are highly anisotropic. Shift perpendicular to the line 
      (cross-dispersion drift) is physical beam drift that directly degrades resolution. Shift along the line 
      (dispersion direction) represents spatial features and is less constrained by the line profile.
    - Decomposition Strategy:
      1. Perpendicular Drift: Determines the target frame's line centroid using PCA. The distance vector from 
         reference centroid to target centroid is projected onto the perpendicular direction vector:
         perp_offset = delta_centroid · perp_vector
         This captures the critical sub-pixel cross-dispersion drift.
      2. Parallel Drift: Computes the whole-frame shift via phase correlation, then projects this translation 
         onto the line's parallel direction vector:
         parallel_offset = phase_correlation_offset · direction_vector
      3. Vector Synthesis: Reconstructs the complete sub-pixel displacement vector as:
         displacement_vector = (perp_offset * perp_vector) + (parallel_offset * direction_vector)
      This hybrid method combines high-accuracy PCA for the resolution-critical perpendicular axis with 
      robust whole-image phase correlation for the parallel axis.

    Args:
        ref_raw: 2D float32 array of the reference image.
        target_raw: 2D float32 array of the target image.
        ref_direction: (2,) float64 unit direction vector of the reference line.
        ref_origin: (2,) float64 centroid of the reference line.
        ref_threshold: Percentile threshold used for reference line detection.
        target_threshold: Percentile threshold used for target line detection.

    Returns:
        tuple[float, float]: Sub-pixel translation vector (dx, dy).
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
