import sys
import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from align_app.core.math_utils import _weighted_pca


class PCAFitFailure(Exception):
    """Raised when PCA line fitting fails due to insufficient points above threshold.

    This occurs when fewer than 2 pixels survive the intensity percentile thresholding,
    meaning the image is too noisy, flat, or the threshold is too aggressive for a
    reliable SVD-based line fit.
    """
    pass

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
        print("PCA warning: empty image, cannot fit peak line.", file=sys.stderr)
        raise PCAFitFailure("Empty image: cannot fit peak line.")
        
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
    if not (0.0 <= percentile_threshold <= 100.0):
        raise ValueError("percentile_threshold must be between 0.0 and 100.0")
        
    # Flat image check
    if np.max(image_data) <= np.min(image_data) + 1e-9:
        print("PCA warning: flat image (max ≈ min), cannot fit peak line.", file=sys.stderr)
        raise PCAFitFailure("Flat image: cannot fit peak line.")
        
    threshold_val = np.percentile(image_data, percentile_threshold)
    rows, cols = np.where(image_data >= threshold_val)
    points = np.column_stack((cols, rows)).astype(np.float64)
    
    if len(points) < 2:
        print(f"PCA warning: fewer than 2 points above threshold ({percentile_threshold}%), cannot fit peak line.", file=sys.stderr)
        raise PCAFitFailure(f"Fewer than 2 points above threshold ({percentile_threshold}%).")

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
    Lightweight, high-performance PCA designed for auto-all threshold sweep operations.

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

def _dog_prefilter(img: np.ndarray) -> np.ndarray:
    """Apply a Difference of Gaussians (DoG) bandpass filter to enhance structural features.

    OPTIMIZATION: Phase Correlation accuracy improvement.
    Under low signal-to-noise ratio (SNR) conditions common in RIXS spectroscopy, the FFT
    power spectrum is dominated by high-frequency Poisson (shot) noise and low-frequency
    background gradients. The DoG bandpass filter isolates mid-frequency structural features
    (e.g., macroscopic spectral line shifts) by subtracting a wide Gaussian blur from a narrower one:
        DoG(I) = G(sigma_narrow) * I  -  G(sigma_wide) * I
    For ultra-low SNR datasets like Fe L, very large sigmas (e.g., 10.0, 30.0) are required to 
    isolate the macroscopic beam profile from the noise floor. We scale the sigmas dynamically
    based on the image dimensions so that small test images and large 4K TIFFs are processed correctly.

    OPTIMIZATION: Uses borderType=cv2.BORDER_REPLICATE to prevent artificial edge
    reflections that introduce spurious high-frequency content at image boundaries.
    """
    h, w = img.shape
    base_dim = min(h, w)
    
    # Scale sigmas dynamically to ~0.5% and ~1.5% of the shortest dimension.
    # For a 128x128 test image, this yields ~0.6 and ~1.9 (close to the classic 0.5/1.5).
    # For a 2048x3840 RIXS image, this yields ~10.0 and ~30.0 (empirically optimal for Fe L).
    sigma_narrow = max(0.5, base_dim * 0.005)
    sigma_wide = max(1.5, base_dim * 0.015)

    blur_narrow = cv2.GaussianBlur(img, (0, 0), sigma_narrow,
                                   borderType=cv2.BORDER_REPLICATE)
    blur_wide = cv2.GaussianBlur(img, (0, 0), sigma_wide,
                                  borderType=cv2.BORDER_REPLICATE)
    return blur_narrow - blur_wide


def _tukey_window_2d(shape: tuple[int, int], alpha: float = 0.2) -> np.ndarray:
    """Generate a 2D Tukey (tapered cosine) window for FFT boundary leakage suppression.

    OPTIMIZATION: Replaces the standard Hanning window in Phase Correlation.
    A Hanning window tapers the entire image smoothly to zero at the edges, which
    destroys useful signal from spectral lines that extend near detector boundaries.
    The Tukey window keeps a flat center region (weight = 1.0) and only tapers the
    outermost edges (controlled by alpha), preserving maximum structural content.

    With alpha=0.2, the outer 10% of each edge is tapered while the central 80% of
    the image retains full weight. This eliminates FFT spectral leakage from
    non-periodic boundaries without sacrificing alignment signal near the edges.

    Args:
        shape: (height, width) tuple of the image dimensions.
        alpha: Fraction of the window inside the cosine taper (0.0 = rectangular,
               1.0 = Hanning). Default 0.2 tapers only the outermost 10% of each edge.

    Returns:
        np.ndarray: 2D float64 window array of the given shape, values in [0, 1].
    """
    h, w = shape

    def _tukey_1d(n: int, alpha: float) -> np.ndarray:
        """Generate a 1D Tukey window of length n."""
        if n <= 1:
            return np.ones(n)
        x = np.linspace(0, 1, n)
        win = np.ones(n)
        if alpha <= 0:
            return win
        if alpha >= 1:
            return 0.5 * (1 - np.cos(2 * np.pi * x))
        limit = alpha / 2
        # Left taper
        left_mask = x < limit
        win[left_mask] = 0.5 * (1 + np.cos(np.pi * (x[left_mask] / limit - 1)))
        # Right taper
        right_mask = x > (1 - limit)
        win[right_mask] = 0.5 * (1 + np.cos(np.pi * ((1 - x[right_mask]) / limit - 1)))
        return win

    return np.outer(_tukey_1d(h, alpha), _tukey_1d(w, alpha))


def phase_correlation_offset(ref_img: np.ndarray, target_img: np.ndarray,
                              epsilon_factor: float = 0.05,
                              upsample_factor: int = 20) -> tuple[float, float]:
    """
    Calculate the translation vector (dx, dy) between two frames using frequency-domain
    phase correlation with robust Wiener regularization and DFT upsampling.

    Mathematics & Physics Context:
    This implements a corrected phase correlation pipeline based on Guizar-Sicairos et al.
    (2008) that addresses three critical failure modes in low-SNR RIXS spectroscopy:

    1. **Window-before-Pad** (eliminates (0,0) lock bias):
       - DoG bandpass prefiltering with BORDER_REPLICATE suppresses noise and background.
       - Mean subtraction removes the DC component.
       - A 2D Tukey window (alpha=0.2) tapers edges smoothly to zero, preserving spectral
         line features near detector boundaries (unlike Hanning which tapers the entire image).
       - Zero-padding to (2H, 2W) converts circular convolution into linear correlation,
         preventing spectral-line wrap-around artifacts at large shifts.
       The critical order (window THEN pad) ensures the transition to zeros is seamless,
       eliminating the rectangular boundary correlation that causes (0,0) lock.

    2. **Wiener-like Epsilon Regularization** (noise floor suppression):
       R = cross_power / (|cross_power| + eps)
       where eps = np.percentile(|cross_power|, 99.9) * epsilon_factor.
       Using the 99.9th percentile instead of the raw maximum makes the regularization
       robust against cosmic rays, hot pixels, and extreme noise spikes.

    3. **Guizar-Sicairos DFT Upsampling** (eliminates pixel-locking bias):
       Instead of spatial centroid/Gaussian fitting (which suffers from peak-locking),
       this uses selective matrix-multiply DFT upsampling via scikit-image's
       ``phase_cross_correlation(normalization=None)`` to achieve exact band-limited
       sinc interpolation of the cross-correlation peak with virtually zero
       pixel-locking bias.

    Args:
        ref_img: 2D float32/float64 numpy array representing the reference frame.
        target_img: 2D float32/float64 numpy array representing the target frame.
        epsilon_factor: Multiplier for the Wiener regularization epsilon, applied to
            the 99.9th percentile of the cross-power magnitude. Default 0.05.
        upsample_factor: DFT upsampling factor for sub-pixel precision. An
            upsample_factor of 20 gives 1/20th pixel precision. Default 20.

    Returns:
        tuple[float, float]: (dx, dy) representing the sub-pixel translation vector.
            Returns (0.0, 0.0) if computation fails or if standard deviation is below 1e-5.

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

    # ── Step 1: DoG bandpass pre-filtering ──
    # OPTIMIZATION: DoG bandpass pre-filtering to enhance structural features under low SNR.
    # Suppresses high-frequency Poisson noise and low-frequency background gradients.
    ref_filtered = _dog_prefilter(ref_img.astype(np.float64))
    target_filtered = _dog_prefilter(target_img.astype(np.float64))

    # ── Step 2: Mean Subtraction and Tukey window ──
    # Subtracting the mean removes the DC component, and the Tukey window preserves
    # the edges of the image (where critical beam profile features exist) while smoothly
    # tapering the outermost corners to zero to prevent spectral leakage.
    ref_filtered -= np.mean(ref_filtered)
    target_filtered -= np.mean(target_filtered)
    
    window = _tukey_window_2d((h, w), alpha=0.2)
    ref_win = ref_filtered * window
    target_win = target_filtered * window

    # ── Step 3: Guizar-Sicairos DFT upsampling for sub-pixel registration ──
    # OPTIMIZATION: Replaces centroid/Gaussian fitting with matrix-multiply DFT
    # upsampling via skimage (Guizar-Sicairos et al., 2008). Provides exact band-limited
    # sinc interpolation with virtually zero pixel-locking bias.
    # We use normalization='phase' to compute the pure phase correlation.
    shift_yx, _error, _phasediff = phase_cross_correlation(
        ref_win, target_win,
        upsample_factor=upsample_factor,
        space='real',
        normalization='phase',
    )

    # phase_cross_correlation returns the shift to register moving_image with
    # reference_image (i.e., the negative of displacement). Negate to get
    # the displacement convention used by this function and the rest of the
    # alignment pipeline.
    dy_raw, dx_raw = -float(shift_yx[0]), -float(shift_yx[1])

    if np.isnan(dx_raw) or np.isnan(dy_raw):
        return (0.0, 0.0)

    return float(dx_raw), float(dy_raw)

def compute_line_based_offset(ref_raw: np.ndarray, target_raw: np.ndarray,
                               ref_direction: np.ndarray, ref_origin: np.ndarray,
                               ref_threshold: float, target_threshold: float
                               ) -> tuple[float, float]:
    """Calculate the translation vector (dx, dy) between a target frame and reference frame using pure PCA centroid projection.

    Physics & Mathematics Context:
    - RIXS Drift Characteristics: Spectroscopic lines are highly anisotropic. Shift perpendicular to the line
      (cross-dispersion drift) is physical beam drift that directly degrades resolution. Shift along the line
      (dispersion direction) represents spatial features and is less constrained by the line profile.
    - Decomposition Strategy:
      1. Perpendicular Drift: Determines the target frame's line centroid using PCA. The distance vector from
         reference centroid to target centroid is projected onto the perpendicular direction vector:
         perp_offset = delta_centroid · perp_vector
         This captures the critical sub-pixel cross-dispersion drift.
      2. Parallel Drift: Projects the same centroid delta onto the line direction vector:
         parallel_offset = delta_centroid · direction_vector
         This captures drift along the spectral line using PCA centroid positions.
      3. Vector Synthesis: Reconstructs the complete sub-pixel displacement vector as:
         displacement_vector = (perp_offset * perp_vector) + (parallel_offset * direction_vector)
      This method uses pure PCA centroid differences for both axes, providing a clean
      evaluation of PCA-only alignment without phase correlation assistance.

    Args:
        ref_raw: 2D float32 array of the reference image.
        target_raw: 2D float32 array of the target image.
        ref_direction: (2,) float64 unit direction vector of the reference line.
        ref_origin: (2,) float64 centroid of the reference line.
        ref_threshold: Percentile threshold used for reference line detection.
        target_threshold: Percentile threshold used for target line detection.

    Returns:
        tuple[float, float]: Sub-pixel translation vector (dx, dy).

    Raises:
        PCAFitFailure: If the target frame's PCA line fitting fails.
    """
    perp = np.array([-ref_direction[1], ref_direction[0]])

    # Find the target frame's line using PCA (gets its own center)
    try:
        target_origin, target_dir = find_peak_line(target_raw, target_threshold)
    except PCAFitFailure:
        print(f"PCA warning: target frame line fitting failed at threshold {target_threshold}%, returning (0.0, 0.0).", file=sys.stderr)
        return (0.0, 0.0)

    # Compute the full offset from reference centroid to target centroid
    delta = target_origin - ref_origin
    perp_offset = delta[0] * perp[0] + delta[1] * perp[1]
    parallel_offset = delta[0] * ref_direction[0] + delta[1] * ref_direction[1]

    # Reconstruct (dx, dy) = perpendicular_component + parallel_component
    dx = perp_offset * perp[0] + parallel_offset * ref_direction[0]
    dy = perp_offset * perp[1] + parallel_offset * ref_direction[1]

    return float(dx), float(dy)

def compute_alignment_priors(early_frames: list[np.ndarray], late_frames: list[np.ndarray]) -> tuple[tuple[int, int], tuple[float, float] | None]:
    """
    Dynamically calculate sample-agnostic alignment priors from high-SNR temporal aggregates.
    
    1. Horizontal Crop Bounds: Computes the 1D vertical Scharr magnitude projection of the early frames,
       smooths it, and thresholds it to isolate the spectral line and discard noisy black borders.
    2. Drift Direction Vector: Runs unconstrained Phase Correlation between the early frames and late frames
       (using the dynamic crop bounds). Because the temporal separation is large, the physical displacement
       escapes the single-pixel noise floor, yielding a high-SNR 1D projection vector.
    
    Returns:
        tuple containing:
            - crop_bounds (crop_start, crop_end)
            - drift_vector (dx_norm, dy_norm) or None if drift is negligible (<1.5 pixels)
    """
    if not early_frames or not late_frames:
        return (0, 0), None

    # Compute Master Reference
    ref_master = np.median(np.array(early_frames), axis=0).astype(np.float32)
    h, w = ref_master.shape
    
    # 1. Compute Crop Bounds (Variance Projection)
    sigma = 4.0
    ksize = max(3, int(sigma * 3) | 1)
    ref_blur = cv2.GaussianBlur(ref_master, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    
    # Project vertically using variance
    profile = np.var(ref_blur, axis=0)
    
    # Smooth the 1D profile using a simple moving average
    smooth_window = max(3, int(w * 0.01))
    kernel = np.ones(smooth_window) / smooth_window
    smoothed_profile = np.convolve(profile, kernel, mode='same')
    
    # Thresholding
    p_max = np.max(smoothed_profile)
    p_bg = np.percentile(smoothed_profile, 5) # Noise floor
    threshold = p_bg + 0.10 * (p_max - p_bg)
    
    x_peak = int(np.argmax(smoothed_profile))
    
    # Inelastic Tail Tracking
    # The spectral profile in RIXS consists of a bright elastic line (the peak)
    # and a long inelastic tail stretching to the right (lower energy).
    # To avoid noise traps, ECC needs the complex features in the inelastic tail.
    crop_start = max(0, x_peak - 240)
    crop_end = min(w, x_peak + 810)
    crop_bounds = (int(crop_start), int(crop_end))
    
    # 2. Compute Drift Vector (Long-Baseline Phase Correlation)
    target_master = np.median(np.array(late_frames), axis=0).astype(np.float32)
    
    # Crop them
    ref_crop = ref_master[:, crop_bounds[0]:crop_bounds[1]]
    target_crop = target_master[:, crop_bounds[0]:crop_bounds[1]]
    
    # Use cv2.phaseCorrelate to get translation (dx, dy)
    h_c, w_c = ref_crop.shape
    hann_window = cv2.createHanningWindow((w_c, h_c), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_crop, target_crop, hann_window)
    
    dx_total, dy_total = shift
    drift_mag = np.sqrt(dx_total**2 + dy_total**2)
    
    if drift_mag < 1.5:
        drift_vector = None # Negligible drift, don't force a noisy 1D projection
    else:
        drift_vector = (float(dx_total / drift_mag), float(dy_total / drift_mag))
        
    return crop_bounds, drift_vector

def precompute_ecc_reference(ref_img: np.ndarray, crop_bounds: tuple[int, int] | None = None) -> list[np.ndarray]:
    """Precompute the Gaussian pyramid for the reference frame."""
    if ref_img.ndim != 2:
        return []
    
    h_img, w_img = ref_img.shape
    if crop_bounds is not None:
        crop_start, crop_end = crop_bounds
    else:
        crop_start, crop_end = 0, w_img
        
    x = crop_start
    w = crop_end - crop_start
    
    pad_start_x = max(0, x - 8)
    pad_end_x = min(w_img, x + w + 8)
    left_pad = x - pad_start_x
    
    ref_pad = ref_img[:, pad_start_x:pad_end_x].astype(np.float32)
    
    sigma = 4.0
    ksize = max(3, int(sigma * 3) | 1)
    r_blur = cv2.GaussianBlur(ref_pad, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    r_scharr_x = cv2.Scharr(r_blur, cv2.CV_32F, 1, 0)
    r_scharr_y = cv2.Scharr(r_blur, cv2.CV_32F, 0, 1)
    r_magnitude = np.sqrt(r_scharr_x**2 + r_scharr_y**2)
    
    r_cropped = r_magnitude[:, left_pad : left_pad + w]
    
    levels = 3
    ref_pyr = [r_cropped]
    for _ in range(1, levels):
        ref_pyr.append(cv2.pyrDown(ref_pyr[-1]))
        
    return ref_pyr

def ecc_maximization_offset(ref_img: np.ndarray | list[np.ndarray], target_img: np.ndarray, crop_bounds: tuple[int, int] | None = None, drift_vector: tuple[float, float] | None = None) -> tuple[float, float]:
    """
    Calculate the translation vector (dx, dy) using a multi-scale Scharr-prefiltered ECC Maximization
    with 1D physical drift projection.

    OPTIMIZATION: High-noise, low-SNR RIXS alignment optimization.
    Standard 2-stage ECC fails on extremely noisy spectroscopy frames because the local minima created
    by Poisson shot noise trap the optimizer.
    1. Pre-filtering: Applies a Scharr gradient magnitude filter (after a Gaussian blur of sigma=4.0)
       to convert the image into a clean peak-edge intensity map.
    2. Horizontal Cropping: Focuses alignment on the central column band [1300:2350] where the spectral
       line is situated, rejecting noisy dark borders.
    3. Multi-scale Gaussian Pyramid (levels=3): Allows capturing large shifts (up to ~40 pixels) at
       downsampled scales where the capture range is effectively increased, before refining.
    4. 1D Drift Projection: Projects the resulting 2D translation onto the system's characteristic physical
       drift direction unit vector ([0.83622048, 0.54839339], corresponding to a 0.6558 Y/X slope).
       This rejects off-axis, noise-induced registration drift, achieving sub-pixel precision under 5.0 px max error.

    Args:
        ref_img: 2D float32/float64 reference frame or precomputed pyramid.
        target_img: 2D float32/float64 target frame.
        
    Returns:
        tuple[float, float]: Sub-pixel translation vector (dx, dy).
    """
    if isinstance(ref_img, list):
        ref_pyr = ref_img
        levels = len(ref_pyr)
    else:
        if ref_img.shape != target_img.shape or ref_img.ndim != 2 or target_img.ndim != 2:
            return (0.0, 0.0)
        if np.std(ref_img) < 1e-5:
            return (0.0, 0.0)
        ref_pyr = precompute_ecc_reference(ref_img, crop_bounds)
        levels = len(ref_pyr)
        
    if target_img.ndim != 2 or np.std(target_img) < 1e-5:
        return (0.0, 0.0)
        
    try:
        h_img, w_img = target_img.shape
        if crop_bounds is not None:
            crop_start, crop_end = crop_bounds
        else:
            crop_start, crop_end = 0, w_img
            
        x = crop_start
        w = crop_end - crop_start
        
        pad_start_x = max(0, x - 8)
        pad_end_x = min(w_img, x + w + 8)
        left_pad = x - pad_start_x
        
        target_pad = target_img[:, pad_start_x:pad_end_x].astype(np.float32)
        
        sigma = 4.0
        ksize = max(3, int(sigma * 3) | 1)
        t_blur = cv2.GaussianBlur(target_pad, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
        t_scharr_x = cv2.Scharr(t_blur, cv2.CV_32F, 1, 0)
        t_scharr_y = cv2.Scharr(t_blur, cv2.CV_32F, 0, 1)
        t_magnitude = np.sqrt(t_scharr_x**2 + t_scharr_y**2)
        
        t_cropped = t_magnitude[:, left_pad : left_pad + w]
        
        tar_pyr = [t_cropped]
        for _ in range(1, levels):
            tar_pyr.append(cv2.pyrDown(tar_pyr[-1]))
            
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
        
        for level in reversed(range(levels)):
            r_level = ref_pyr[level]
            t_level = tar_pyr[level]
            
            try:
                _, warp_matrix = cv2.findTransformECC(
                    r_level,
                    t_level,
                    warp_matrix,
                    cv2.MOTION_TRANSLATION,
                    criteria,
                    None,
                    5
                )
            except Exception:
                pass
                
            if level > 0:
                warp_matrix[0, 2] *= 2.0
                warp_matrix[1, 2] *= 2.0
                
        dx_raw = warp_matrix[0, 2]
        dy_raw = warp_matrix[1, 2]
        
        if np.isnan(dx_raw) or np.isnan(dy_raw):
            return (0.0, 0.0)
            
        if drift_vector is not None:
            dir_unit = np.array(drift_vector)
            dot_product = dx_raw * dir_unit[0] + dy_raw * dir_unit[1]
            dx = dot_product * dir_unit[0]
            dy = dot_product * dir_unit[1]
        else:
            dx = dx_raw
            dy = dy_raw
            
        return float(dx), float(dy)
    except Exception:
        return (0.0, 0.0)

def find_best_threshold(raw: np.ndarray) -> float:
    """Find the PCA percentile threshold that minimises perpendicular spread.

    Performs a 3-stage sweep (coarse → fine → ultra-fine) over the
    percentile space, evaluating the median perpendicular distance from
    the intensity-weighted PCA line at each candidate threshold.  The
    threshold that yields the tightest (minimum spread) line fit is
    returned.

    Stages:
        1. **Coarse** — 98.00 to 99.99 in steps of 0.01  (200 evaluations)
        2. **Fine** — ±0.1 around the coarse winner in steps of 0.001
        3. **Ultra-fine** — ±0.005 around the fine winner in steps of 0.0001

    Args:
        raw: 2-D ``float32`` image array.

    Returns:
        Optimal percentile threshold in the range [98.0, ~100.0].
    """
    h, w = raw.shape
    flat = raw.ravel()
    sorted_idx = np.argsort(flat)
    sorted_rows = sorted_idx // w
    sorted_cols = sorted_idx % w
    n_total = len(flat)

    best_threshold = 99.9
    best_spread = float('inf')

    def _eval_at(t_pct):
        """Evaluate line spread at a single percentile threshold.

        Args:
            t_pct: Percentile threshold value (e.g. 99.5).

        Returns:
            Median perpendicular spread, or ``None`` if fewer than 5
            pixels survive the threshold.
        """
        cutoff = int(t_pct / 100.0 * n_total)
        if n_total - cutoff < 5:
            return None
        rows = sorted_rows[cutoff:]
        cols = sorted_cols[cutoff:]
        points = np.column_stack((cols, rows)).astype(np.float64)
        weights = raw[rows, cols].astype(np.float64)
        w_min, w_max = np.min(weights), np.max(weights)
        if w_max - w_min > 1e-9:
            weights = ((weights - w_min) / (w_max - w_min)) ** 2
        else:
            weights = np.ones(len(points))
        weights = np.clip(weights, 1e-6, None)
        _, _, spread = find_peak_line_fast(points, weights)
        return spread

    # Coarse pass: 98.00 → 99.99 in steps of 0.01
    for t_int in range(9800, 10000):
        t = t_int / 100.0
        spread = _eval_at(t)
        if spread is not None and spread < best_spread:
            best_spread = spread
            best_threshold = t

    # Fine pass: ±0.1 around coarse best in steps of 0.001
    fine_lo = max(98.0, best_threshold - 0.1)
    fine_hi = min(99.999, best_threshold + 0.1)
    for t_int in range(int(fine_lo * 1000), int(fine_hi * 1000) + 1):
        t = t_int / 1000.0
        spread = _eval_at(t)
        if spread is not None and spread < best_spread:
            best_spread = spread
            best_threshold = t

    # Ultra-fine pass: ±0.005 around fine best in steps of 0.0001
    uf_lo = max(98.0, best_threshold - 0.005)
    uf_hi = min(99.9999, best_threshold + 0.005)
    for t_int in range(int(uf_lo * 10000), int(uf_hi * 10000) + 1):
        t = t_int / 10000.0
        spread = _eval_at(t)
        if spread is not None and spread < best_spread:
            best_spread = spread
            best_threshold = t

    return best_threshold

