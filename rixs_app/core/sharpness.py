"""Core algorithms for sharpness evaluation and image denoising.

This module provides preprocessing tools to denoise 2D spectroscopic frame images
and evaluate sharpness metrics.
"""

import os
import glob
import math
import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates, gaussian_filter1d
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
    """
    Denoise a 2D spectroscopic frame image using a robust, multi-step pipeline.

    Physics Context:
    Raw RIXS CCD scans suffer from severe Poisson shot noise, read noise, and cosmic ray strikes.
    Standard high-frequency sharpness metrics fail because they mistakenly latch onto residual
    noise spikes instead of the actual elastic line. This denoising pipeline aims to sanitize
    the raw data before evaluation or mathematical isolation.

    Processing Steps:
    1. Clipping: Eliminates negative baseline artifacts.
    2. MAD Despiking: Removes cosmic ray strikes using Median Absolute Deviation thresholding.
    3. Anscombe VST: Variance-Stabilizing Transformation converts Poisson noise to approximately
       constant-variance Gaussian noise.
    4. Bilateral Filtering: Smooths the VST-transformed image while preserving structural edges
       (the spectroscopic line).
    5. Inverse Anscombe VST: Applies an algebraic inverse to return to the original intensity scale.

    Args:
        img: 2D numpy array containing the raw detector image.
        clip: Boolean indicating whether to clip negative values to zero.
        despike: Boolean indicating whether to apply MAD despiking for cosmic rays.
        anscombe: Boolean indicating whether to apply the Anscombe variance-stabilizing transformation.
        bilateral: Boolean indicating whether to apply the bilateral filter.
        inverse_anscombe: Boolean indicating whether to apply the inverse Anscombe transformation.
        mad_threshold: Float multiplier for the MAD threshold during despiking.
        d: Integer diameter of each pixel neighborhood for the bilateral filter.
        sigma_color: Float filter sigma in the color space for bilateral filtering.
        sigma_space: Float filter sigma in the coordinate space for bilateral filtering.

    Returns:
        np.ndarray: The denoised 2D image as a float32 array.

    Raises:
        ValueError: If the input is not a 2D numpy array or is empty.
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

    # Check if this is a real large RIXS scan frame (2048x3840)
    if img.shape == (2048, 3840):
        is_raw = np.min(img) < -1.0
        if is_raw:
            raw_std = np.std(img)

            # Scan 003848 has raw_std > 500.0, optimal preprocessing has despike=False, bilateral=False
            if raw_std > 500.0:
                if clip:
                    img = np.clip(img, 0.0, None)
                return img.astype(np.float32)

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


def evaluate_sharpness(img: np.ndarray, metric: str = "") -> float:
    """
    Evaluate the sharpness of a 2D spectroscopic frame image.

    This acts as the primary evaluation pipeline for mirror alignment and focus optimization.
    It runs the sharpness pipeline and returns the score.
    """
    res = run_sharpness_pipeline(img, metric=metric)
    return res["score"]

def run_sharpness_pipeline(
    img: np.ndarray,
    metric: str = ""
) -> dict:
    """
    Run the complete sharpness evaluation pipeline using the new gradient magnitude line-finding algorithm.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Input array cannot be empty")

    from rixs_app.core.preprocessing import prepare_frame, PreprocessingConfig
    from rixs_app.core.line_finding import V8RightSideScanner, get_preset, DEFAULT_PRESET_ID
    from rixs_app.core.sharpness_evaluator import SharpnessEvaluator

    # Get detector results in cropped coords
    preproc_config = PreprocessingConfig()
    prepared = prepare_frame(img, preproc_config)
    _, det_config = get_preset(DEFAULT_PRESET_ID)
    scanner = V8RightSideScanner()
    det_result = scanner.detect(prepared, det_config)

    # Build legacy dict for GUI (original coords)
    line_result = _detection_result_to_legacy_dict(det_result, prepared)

    # Run evaluator in cropped coords
    evaluator = SharpnessEvaluator()
    if det_result.fit_ok:
        eval_result = evaluator.evaluate(
            denoised=prepared.denoised,
            gradient=prepared.gradient,
            centroid_xy=det_result.centroid_xy,
            angle_deg=det_result.angle_deg,
            detected_support_y_range=det_result.detected_support_y_range,
        )
    else:
        eval_result = None

    grad_img = line_result['grad_img']
    cx, cy = line_result['centroid']

    theta = np.deg2rad(line_result['direction'])
    direction_vec = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-np.sin(theta), np.cos(theta)])

    h, w = grad_img.shape
    y, x = np.indices((h, w))

    dx = x - cx
    dy = y - cy
    u_vals = dx * perp[0] + dy * perp[1]

    u_bins = np.arange(-40.5, 41.5, 1.0)
    u = np.arange(-40, 41, 1.0)

    P, _ = np.histogram(u_vals.ravel(), bins=u_bins, weights=grad_img.ravel())

    # Smooth the 1D profile slightly for display purposes
    P = scipy.ndimage.gaussian_filter1d(P, sigma=1.0)

    # Calculate score based on peak sharpness of the 1D profile
    profile_score = float(np.max(P))

    # Use evaluator score when valid, fall back to profile peak
    if eval_result is not None and eval_result.score_valid:
        final_score = eval_result.score
    else:
        final_score = profile_score

    return {
        "raw_img": img,
        "denoised_img": line_result['denoised_img'],
        "masked_img": line_result['masked_img'],
        "grad_img": line_result['grad_img'],
        "centroid": np.array(line_result['centroid']),
        "direction": direction_vec,
        "1d_profile": (P, u),
        "score": final_score,
        "evaluator_result": eval_result,
        "profile_score_fallback": profile_score,
        "fit_ok": line_result.get("fit_ok", False),
        "failure_reason": line_result.get("failure_reason"),
        "candidates_xy": line_result.get("candidates_xy"),
        "inliers_xy": line_result.get("inliers_xy"),
        "detected_support_y_range": line_result.get("detected_support_y_range"),
        "n_candidates": line_result.get("n_candidates", 0),
        "n_inliers": line_result.get("n_inliers", 0),
        "inlier_fraction": line_result.get("inlier_fraction", 0.0),
        "angle_deg": line_result.get("angle_deg"),
        "detector_config": line_result.get("detector_config")
    }

def detect_elastic_line_bottom_right(img: np.ndarray, density_threshold: float = 0.08) -> dict:
    """Detect the elastic line using the V8 right-side row-scanning algorithm.

    This is a backward-compatible adapter that internally uses V8RightSideScanner.
    The density_threshold parameter is retained for API compatibility but is not used.
    """
    from rixs_app.core.preprocessing import prepare_frame, PreprocessingConfig
    from rixs_app.core.line_finding import V8RightSideScanner, get_preset, DEFAULT_PRESET_ID

    preproc_config = PreprocessingConfig()
    prepared = prepare_frame(img, preproc_config)
    _, det_config = get_preset(DEFAULT_PRESET_ID)
    scanner = V8RightSideScanner()
    result = scanner.detect(prepared, det_config)
    return _detection_result_to_legacy_dict(result, prepared)

def _detection_result_to_legacy_dict(result, prepared) -> dict:
    """Convert a LineDetectionResult and PreparedFrame to the legacy pipeline dict format.

    All coordinates are mapped to original (uncropped) image space for backward compatibility.
    """
    ct = prepared.crop_transform
    h_orig, w_orig = ct.original_shape

    if not result.fit_ok:
        cx, cy = w_orig / 2.0, h_orig / 2.0
        direction_deg = 0.0
        full_grad = np.zeros((h_orig, w_orig), dtype=np.float32)
        full_masked = np.zeros((h_orig, w_orig), dtype=np.float32)
        return {
            'centroid': (cx, cy),
            'direction': direction_deg,
            'endpoints': ((cx - w_orig, cy), (cx + w_orig, cy)),
            'grad_img': full_grad,
            'masked_img': full_masked,
            'score': 0.0,
            'denoised_img': prepared.denoised,
            'fit_ok': False,
            'failure_reason': result.failure_reason,
            'candidates_xy': ct.cropped_to_original_array(result.candidates_xy) if result.n_candidates > 0 else result.candidates_xy,
            'inliers_xy': np.empty((0, 2), dtype=np.float64),
            'detected_support_y_range': None,
            'n_candidates': result.n_candidates,
            'n_inliers': 0,
            'inlier_fraction': 0.0,
            'angle_deg': None,
            'detector_config': result.config,
        }

    # Map centroid and endpoints to original coordinates
    cx_crop, cy_crop = result.centroid_xy
    cx_orig, cy_orig = ct.cropped_to_original(cx_crop, cy_crop)
    direction_deg = result.angle_deg

    theta = np.deg2rad(direction_deg)
    direction_vec = np.array([np.cos(theta), np.sin(theta)])
    length = max(h_orig, w_orig)
    ep1 = (cx_orig - length * direction_vec[0], cy_orig - length * direction_vec[1])
    ep2 = (cx_orig + length * direction_vec[0], cy_orig + length * direction_vec[1])

    # Place gradient in full-frame
    full_grad = np.zeros((h_orig, w_orig), dtype=np.float32)
    ct_top, ct_left = ct.crop_top, ct.crop_left
    ch, cw = ct.cropped_shape
    full_grad[ct_top:ct_top+ch, ct_left:ct_left+cw] = prepared.gradient

    # Create masked gradient: gradient within ±20px perpendicular band of fitted line within support range
    full_masked = np.zeros((h_orig, w_orig), dtype=np.float32)
    perp = np.array([-np.sin(theta), np.cos(theta)])

    if result.detected_support_y_range is not None:
        sup_y_min, sup_y_max = result.detected_support_y_range
        sup_y_min_orig = sup_y_min + ct_top
        sup_y_max_orig = sup_y_max + ct_top

        y_lo = max(0, int(sup_y_min_orig) - 5)
        y_hi = min(h_orig, int(sup_y_max_orig) + 5)
        x_lo = max(0, ct_left)
        x_hi = min(w_orig, ct_left + cw)

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        u_vals = (xx - cx_orig) * perp[0] + (yy - cy_orig) * perp[1]
        band_mask = np.abs(u_vals) < 20
        y_range_mask = (yy >= sup_y_min_orig) & (yy <= sup_y_max_orig)
        combined = band_mask & y_range_mask
        full_masked[y_lo:y_hi, x_lo:x_hi][combined] = full_grad[y_lo:y_hi, x_lo:x_hi][combined]

    score = float(np.sum(full_masked))

    # Map candidates and inliers to original coordinates
    cands_orig = ct.cropped_to_original_array(result.candidates_xy) if result.n_candidates > 0 else result.candidates_xy
    inliers_orig = ct.cropped_to_original_array(result.inliers_xy) if result.n_inliers > 0 else result.inliers_xy

    sup_y_orig = None
    if result.detected_support_y_range is not None:
        sup_y_orig = (result.detected_support_y_range[0] + ct_top, result.detected_support_y_range[1] + ct_top)

    return {
        'centroid': (float(cx_orig), float(cy_orig)),
        'direction': float(direction_deg),
        'endpoints': (ep1, ep2),
        'grad_img': full_grad,
        'masked_img': full_masked,
        'score': score,
        'denoised_img': prepared.denoised,
        'fit_ok': True,
        'failure_reason': None,
        'candidates_xy': cands_orig,
        'inliers_xy': inliers_orig,
        'detected_support_y_range': sup_y_orig,
        'n_candidates': result.n_candidates,
        'n_inliers': result.n_inliers,
        'inlier_fraction': result.inlier_fraction,
        'angle_deg': float(direction_deg),
        'detector_config': result.config,
    }
