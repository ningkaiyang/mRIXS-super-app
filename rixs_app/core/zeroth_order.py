"""Core algorithms for zeroth-order line detection, denoising, and FWHM evaluation.

This module provides preprocessing tools to denoise 2D spectroscopic frame images
and evaluate zeroth-order line metrics.
"""

import numpy as np
import scipy.ndimage
import cv2

from rixs_app.core.preprocessing import prepare_frame, PreprocessingConfig, denoise_image
from rixs_app.core.line_finding import V8RightSideScanner, get_preset, DEFAULT_PRESET_ID
from rixs_app.core.zeroth_order_evaluator import ZerothOrderEvaluator


def evaluate_zeroth_order(img: np.ndarray, metric: str = "") -> float:
    """
    Evaluate a zeroth-order spectroscopic frame image.

    This acts as the primary evaluation pipeline for mirror alignment and focus optimization.
    It runs the zeroth-order pipeline and returns the score.
    """
    res = run_zeroth_order_pipeline(img, metric=metric)
    return res["score"]

def run_zeroth_order_pipeline(
    img: np.ndarray,
    metric: str = "",
    energy_dispersion: float = 0.0
) -> dict:
    """
    Run the complete zeroth-order evaluation pipeline using the gradient magnitude line-finding algorithm.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if img.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Input array cannot be empty")


    # Get detector results in cropped coords
    preproc_config = PreprocessingConfig()
    prepared = prepare_frame(img, preproc_config)
    _, det_config = get_preset(DEFAULT_PRESET_ID)
    scanner = V8RightSideScanner()
    det_result = scanner.detect(prepared, det_config)

    # Build legacy dict for GUI (original coords)
    line_result = _detection_result_to_legacy_dict(det_result, prepared)

    # Run evaluator in cropped coords
    evaluator = ZerothOrderEvaluator()
    if det_result.fit_ok:
        eval_result = evaluator.evaluate(
            denoised=prepared.denoised,
            gradient=prepared.gradient,
            centroid_xy=det_result.centroid_xy,
            angle_deg=det_result.angle_deg,
            detected_support_y_range=det_result.detected_support_y_range,
            energy_dispersion=energy_dispersion,
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

    # Calculate score based on the 1D profile peak
    profile_score = float(np.max(P))

    # Use evaluator score when valid, fall back to profile peak
    if eval_result is not None and eval_result.score_valid:
        final_score = eval_result.score
    else:
        final_score = profile_score

    # Compute fwhm_mev for the result dict (client-side convenience)
    fwhm_mev = None
    if eval_result is not None and eval_result.fwhm_px is not None and energy_dispersion > 0:
        fwhm_mev = eval_result.fwhm_px * energy_dispersion

    return {
        "raw_img": img,
        "denoised_img": line_result['denoised_img'],
        "dsm_img": line_result.get('dsm_img'),
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
        "detector_config": line_result.get("detector_config"),
        "fwhm_mev": fwhm_mev,
        "r_squared": eval_result.r_squared if eval_result else None,
    }



def _detection_result_to_legacy_dict(result, prepared) -> dict:
    """Convert a LineDetectionResult and PreparedFrame to the legacy pipeline dict format.

    All coordinates are mapped to original (uncropped) image space for backward compatibility.
    """
    ct = prepared.crop_transform
    h_orig, w_orig = ct.original_shape
    ct_top, ct_left = ct.crop_top, ct.crop_left
    ch, cw = ct.cropped_shape

    full_denoised = np.zeros((h_orig, w_orig), dtype=np.float32)
    full_dsm = np.zeros((h_orig, w_orig), dtype=np.float32)
    full_denoised[ct_top:ct_top+ch, ct_left:ct_left+cw] = prepared.denoised
    full_dsm[ct_top:ct_top+ch, ct_left:ct_left+cw] = prepared.row_smoothed

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
            'denoised_img': full_denoised,
            'dsm_img': full_dsm,
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
        'denoised_img': full_denoised,
        'dsm_img': full_dsm,
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
