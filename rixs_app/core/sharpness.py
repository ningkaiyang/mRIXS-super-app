"""Core algorithms for sharpness evaluation and image denoising.

This module provides preprocessing tools to denoise 2D spectroscopic frame images
and evaluate sharpness metrics.
"""

import numpy as np
import scipy.ndimage
import cv2

def denoise_image(
    img: np.ndarray,
    clip: bool = True,
    despike: bool = True,
    anscombe: bool = True,
    bilateral: bool = True,
    inverse_anscombe: bool = True,
    mad_threshold: float = 5.0,
    d: int = 5,
    sigma_color: float = 1.5,
    sigma_space: float = 3.0
) -> np.ndarray:
    """Denoise a 2D spectroscopic frame image using a standard or custom pipeline.

    The pipeline sequence consists of:
    1. Clipping: Clamp negative values to zero.
    2. MAD despiking: Identify and replace extreme positive or negative spike pixels
       using a Median Absolute Deviation (MAD) filter.
    3. Anscombe VST: Variance-stabilizing transformation for Poisson noise.
    4. Bilateral Filter: Smooths image while preserving sharp edges.
    5. Inverse Anscombe VST: Converts stabilized values back to raw scale.

    Args:
        img: 2D numpy array representing the raw image.
        clip: If True, clamps negative values to 0.0.
        despike: If True, performs Median Absolute Deviation (MAD) despiking.
        anscombe: If True, applies the Anscombe VST.
        bilateral: If True, applies a bilateral filter.
        inverse_anscombe: If True, applies the inverse Anscombe VST.
        mad_threshold: Threshold multiplier for MAD despiking (default: 5.0).
        d: Diameter of each pixel neighborhood for bilateral filtering (default: 5).
        sigma_color: Filter sigma in the color space (default: 1.5).
        sigma_space: Filter sigma in the coordinate space (default: 3.0).

    Returns:
        2D float32 numpy array containing the denoised image.

    Raises:
        ValueError: If input is not a 2D array or is empty (shape (0,0) or size == 0).
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Input array cannot be empty")

    # Cast first to ensure any overflow becomes inf and is sanitized
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # a. Clipping
    if clip:
        img = np.clip(img, 0.0, None)

    # b. MAD despiking
    if despike:
        median_img = scipy.ndimage.median_filter(img, size=3)
        dev = img - median_img
        mad = np.median(np.abs(dev))
        if mad < 1e-6:
            mad = np.std(dev, dtype=np.float64)
            if np.isnan(mad) or np.isinf(mad) or mad <= 0.0:
                mad = 1e-6
        if mad > 1e-6:
            threshold = mad_threshold * 1.4826 * mad
            img = np.where(np.abs(dev) > threshold, median_img, img)

    # c. Anscombe VST
    if anscombe:
        img = 2.0 * np.sqrt(np.maximum(img + 0.375, 0.0))

    # d. Bilateral Filter
    if bilateral:
        img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    # e. Inverse Anscombe VST
    if inverse_anscombe:
        img = (img / 2.0)**2 - 0.375

    # Final clipping pass when clip=True to prevent negative inverse Anscombe output
    if clip:
        img = np.clip(img, 0.0, None)

    return img.astype(np.float32)

def evaluate_sharpness(img: np.ndarray, metric: str) -> float:
    """Evaluate sharpness of a 2D spectroscopic frame image using a specified metric.

    Args:
        img: 2D numpy array representing the image.
        metric: Sharpness metric to use. One of:
            - 'dog_laplacian': Difference of Gaussians + Laplacian.
            - 'directional_tenengrad': Gradient magnitude based on Sobel operators.
            - 'fft_bandpass': Energy in a specific frequency band.

    Returns:
        float: Calculated sharpness score based on the selected metric.

    Raises:
        ValueError: If input is not a 2D array, is empty, or the metric is unsupported.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Input array cannot be empty")

    valid_metrics = {"dog_laplacian", "directional_tenengrad", "fft_bandpass"}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    # Cast to np.float64 and sanitize NaNs/Infs to 0.0
    img_d = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # Clip or clamp the input image values to a safe range to prevent float64 overflow
    img_d = np.clip(img_d, -1e10, 1e10)

    if metric == "dog_laplacian":
        g1 = scipy.ndimage.gaussian_filter(img_d, sigma=1.5)
        g2 = scipy.ndimage.gaussian_filter(img_d, sigma=4.5)
        dog = g1 - g2
        lap = cv2.Laplacian(dog, cv2.CV_64F)
        return float(np.var(lap))

    elif metric == "directional_tenengrad":
        Gx = cv2.Sobel(img_d, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img_d, cv2.CV_64F, 0, 1, ksize=3)
        P = Gx * np.cos(135 * np.pi / 180.0) + Gy * np.sin(135 * np.pi / 180.0)
        abs_P = np.abs(P)
        thresh = np.percentile(abs_P, 75.0)
        mask = abs_P > thresh
        if not np.any(mask):
            return 0.0
        score = np.sum(P[mask] ** 2)
        return float(score)

    elif metric == "fft_bandpass":
        H, W = img_d.shape
        W_2D = np.outer(np.hanning(H), np.hanning(W))
        img_win = img_d * W_2D
        F = np.fft.fftshift(np.fft.fft2(img_win))
        power_spec = np.abs(F) ** 2
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        R_max = np.max(r)
        
        mask = (r >= 0.05 * R_max) & (r <= 0.8 * R_max)
        E_band = np.sum(power_spec[mask])
        E_total = np.sum(power_spec)
        if E_total > 0:
            return float(E_band / E_total)
        return 0.0
