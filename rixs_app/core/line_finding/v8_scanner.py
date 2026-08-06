"""V8 Line Finder Core Algorithms.

Implements the V8RightSideScanner using robust detection and RANSAC fitting.
"""

import math
import numpy as np
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rixs_app.core.preprocessing import PreparedFrame

from rixs_app.core.line_finding.base import BaseLineDetector, DetectorConfig, LineDetectionResult


def get_pct(sorted_a: np.ndarray, q: float) -> float:
    """Compute percentile on a pre-sorted array.

    Args:
        sorted_a: A 1D numpy array that is already sorted.
        q: The percentile to compute (0 to 100).

    Returns:
        The computed percentile value.
    """
    N = sorted_a.size
    if N == 0:
        return 0.0
    idx = (N - 1) * q / 100.0
    low = int(idx)
    high = min(N - 1, low + 1)
    weight = idx - low
    return float(sorted_a[low] * (1.0 - weight) + sorted_a[high] * weight)


def robust_scale_opt(a: np.ndarray, eps: float) -> float:
    """Compute a robust scale estimate for array `a` using MAD and interquantile ranges.

    Args:
        a: A 1D numpy array of values.
        eps: Minimum scale value to return.

    Returns:
        A robust scale estimate.
    """
    sorted_a = np.sort(a)
    N = sorted_a.size
    if N == 0:
        return eps
    idx = (N - 1) * 0.5
    low = int(idx)
    high = min(N - 1, low + 1)
    weight = idx - low
    center = sorted_a[low] * (1.0 - weight) + sorted_a[high] * weight

    abs_dev = np.abs(a - center)
    sorted_dev = np.sort(abs_dev)
    center_dev = get_pct(sorted_dev, 50.0)
    scale_mad = 1.4826 * center_dev

    p1 = get_pct(sorted_a, 1.0)
    p5 = get_pct(sorted_a, 5.0)
    p16 = get_pct(sorted_a, 16.0)
    p84 = get_pct(sorted_a, 84.0)
    p95 = get_pct(sorted_a, 95.0)
    p99 = get_pct(sorted_a, 99.0)

    scale_68 = (p84 - p16) / 2.0
    scale_90 = (p95 - p5) / 3.289707
    scale_98 = (p99 - p1) / 4.652696

    return float(max(scale_mad, scale_68, scale_90, scale_98, eps))


def robust_detect_row(
    s: np.ndarray,
    ref_frac: float,
    k_rise: float,
    k_level: float,
    sustain: int,
    win: int = 6,
    peak_win: int = 14
) -> Tuple[int, int, float, float, float, float]:
    """Detect peak and foot positions in a row using robust statistics.

    Args:
        s: 1D numpy array representing the row signal.
        ref_frac: Fraction of the row to use as reference region.
        k_rise: Cutoff multiplier for rise z-score.
        k_level: Cutoff multiplier for level z-score.
        sustain: Number of pixels required to sustain the edge.
        win: Window size for moving averages.
        peak_win: Window size to search for the peak before the foot.

    Returns:
        A tuple containing (peak, foot, scale_intensity, scale_rise, rise_cut, level_cut).
    """
    n = s.size
    ref_len = int(n * ref_frac)
    if n < 4 * win + peak_win:
        return -1, -1, 0.0, 0.0, 0.0, 0.0

    ref_intensity = s[n - ref_len:]
    c = np.concatenate(([0.0], np.cumsum(s)))
    left_mean = np.zeros(n)
    right_mean = np.zeros(n)
    left_mean[win: n - win] = (c[win: n - win] - c[0: n - 2 * win]) / win
    right_mean[win: n - win] = (c[2 * win: n] - c[win: n - win]) / win
    rise = left_mean - right_mean

    ref_rise = rise[n - ref_len: n - win]

    sorted_s = np.sort(s)
    p995 = get_pct(sorted_s, 99.5)
    p005 = get_pct(sorted_s, 0.5)
    eps = max(1e-6, 1e-6 * max(p995 - p005, 1.0))

    scale_intensity = robust_scale_opt(ref_intensity, eps)
    scale_rise = robust_scale_opt(ref_rise, eps)

    med_ref_intensity = get_pct(np.sort(ref_intensity), 50.0)
    med_ref_rise = get_pct(np.sort(ref_rise), 50.0)

    z_rise = (rise - med_ref_rise) / scale_rise
    z_level = (s - med_ref_intensity) / scale_intensity

    ref_z_rise = z_rise[n - ref_len: n - win]
    ref_z_level = z_level[n - ref_len:]

    rise_cut = max(k_rise, get_pct(np.sort(ref_z_rise), 99.9))
    level_cut = max(k_level, get_pct(np.sort(ref_z_level), 99.0))

    level_cut_raw = med_ref_intensity + level_cut * scale_intensity
    above = (s > level_cut_raw).astype(np.float64)

    foot = -1
    for x in range(n - ref_len - 1, win, -1):
        if z_rise[x] >= rise_cut:
            if (left_mean[x] - med_ref_intensity) / scale_intensity >= level_cut:
                a = max(0, x - sustain)
                frac = np.mean(above[a:x])
                if frac >= 0.75:
                    foot = x
                    break

    if foot < 0:
        return -1, -1, scale_intensity, scale_rise, rise_cut, level_cut

    lo = max(0, foot - peak_win)
    peak = lo + int(np.argmax(s[lo: foot + 1]))
    return peak, foot, scale_intensity, scale_rise, rise_cut, level_cut


def _norm(a: float) -> float:
    """Normalize an angle to the [-90, 90) range.

    Args:
        a: The input angle in degrees.

    Returns:
        The normalized angle in degrees.
    """
    a = (a + 180) % 180
    return a - 180 if a > 90 else a


def ransac_v7(
    xs: np.ndarray,
    ys: np.ndarray,
    thresh: float = 4.0,
    iters: int = 3000,
    seed: int = 0,
    svd_refine_iters: int = 6
) -> Optional[Dict[str, Any]]:
    """Fit a line using RANSAC with SVD refinement.

    Args:
        xs: 1D numpy array of x coordinates.
        ys: 1D numpy array of y coordinates.
        thresh: Distance threshold for inliers.
        iters: Number of RANSAC iterations.
        seed: Random seed for determinism.
        svd_refine_iters: Number of SVD refinement iterations.

    Returns:
        A dictionary containing the fitted line parameters or None if unsuccessful.
    """
    n = len(xs)
    if n < 12:
        return None
    P = np.column_stack([xs, ys])
    rng = np.random.default_rng(seed)
    best = None
    bc = -1
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        dv = P[j] - P[i]
        nr = np.hypot(*dv)
        if nr < 1e-6:
            continue
        nv = np.array([-dv[1], dv[0]]) / nr
        inl = np.abs((P - P[i]) @ nv) < thresh
        c = int(inl.sum())
        if c > bc:
            bc = c
            best = inl
    if best is None or best.sum() < 8:
        return None
    inl = best
    for _ in range(svd_refine_iters):
        Q = P[inl]
        ctr = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - ctr)
        dvv = vt[0]
        nv = np.array([-dvv[1], dvv[0]])
        ninl = np.abs((P - ctr) @ nv) < thresh
        if int(ninl.sum()) == int(inl.sum()):
            inl = ninl
            break
        inl = ninl
    if inl.sum() < 8:
        return None
    Q = P[inl]
    ctr = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - ctr)
    dvv = vt[0]
    ang = _norm(math.degrees(math.atan2(dvv[1], dvv[0])))
    if dvv[1] > 0:
        dvv = -dvv
    return dict(centroid=ctr, direction=dvv, angle=ang, inliers=inl, n_inliers=int(inl.sum()))


class V8RightSideScanner(BaseLineDetector):
    """V8 algorithm for right-side emission line detection."""

    def detect(self, prepared: 'PreparedFrame', config: 'DetectorConfig') -> 'LineDetectionResult':
        """Detect the emission line in the prepared frame.

        Args:
            prepared: The prepared frame containing the row-smoothed image data.
            config: The detection configuration.

        Returns:
            The detection result.
        """
        Dsm = prepared.row_smoothed
        H, W = Dsm.shape
        x0 = config.scan_margin_px
        x1 = W - config.scan_margin_px

        candidates = []
        for y in range(config.scan_margin_px, H - config.scan_margin_px, config.y_step):
            s = Dsm[y, x0:x1]
            pk, _, _, _, _, _ = robust_detect_row(
                s,
                config.ref_frac,
                config.k_rise,
                config.k_level,
                config.sustain,
                win=config.win,
                peak_win=config.peak_win
            )
            if pk >= 0:
                candidates.append((pk + x0, y))

        if len(candidates) < 12:
            empty_cands = np.array(candidates, dtype=np.float64) if candidates else np.empty((0, 2), dtype=np.float64)
            return LineDetectionResult(
                fit_ok=False,
                n_candidates=len(candidates),
                candidates_xy=empty_cands,
                config=config,
                failure_reason=f'Insufficient candidates ({len(candidates)} < 12)',
            )

        candidates_xy = np.array(candidates, dtype=np.float64)
        xs = candidates_xy[:, 0]
        ys = candidates_xy[:, 1]

        result = ransac_v7(
            xs,
            ys,
            thresh=config.ransac_thresh,
            iters=config.ransac_iters,
            seed=config.ransac_seed,
            svd_refine_iters=config.svd_refine_iters
        )

        if result is None:
            return LineDetectionResult(
                fit_ok=False,
                n_candidates=len(candidates_xy),
                candidates_xy=candidates_xy,
                config=config,
                failure_reason='RANSAC failed to find sufficient inliers',
            )

        centroid_xy = tuple(result['centroid'].tolist())
        direction_vec = tuple(result['direction'].tolist())
        angle_deg = float(result['angle'])
        inlier_mask = result['inliers']
        inliers_xy = candidates_xy[inlier_mask]

        if inliers_xy.shape[0] == 0:
            support_y_range = (0.0, 0.0)
            endpoints = ((0.0, 0.0), (0.0, 0.0))
            res_med = res_p95 = res_max = 0.0
        else:
            inliers_y = inliers_xy[:, 1]
            y_min = float(np.percentile(inliers_y, 5))
            y_max = float(np.percentile(inliers_y, 95))
            support_y_range = (y_min, y_max)

            # Intersection of fitted line with support y-range
            # Line is p = centroid + t * direction
            # y = centroid[1] + t * direction[1] => t = (y - centroid[1]) / direction[1]
            if abs(direction_vec[1]) > 1e-9:
                t1 = (y_min - centroid_xy[1]) / direction_vec[1]
                t2 = (y_max - centroid_xy[1]) / direction_vec[1]
                ep1 = (centroid_xy[0] + t1 * direction_vec[0], y_min)
                ep2 = (centroid_xy[0] + t2 * direction_vec[0], y_max)
                endpoints = (ep1, ep2)
            else:
                endpoints = ((centroid_xy[0], y_min), (centroid_xy[0], y_max))

            # Residual statistics
            # Normal vector is (-direction[1], direction[0])
            norm_vec = np.array([-direction_vec[1], direction_vec[0]])
            dists = np.abs((inliers_xy - result['centroid']) @ norm_vec)
            res_med = float(np.median(dists))
            res_p95 = float(np.percentile(dists, 95))
            res_max = float(np.max(dists))

        n_inliers = int(inlier_mask.sum())
        return LineDetectionResult(
            fit_ok=True,
            n_candidates=len(candidates_xy),
            candidates_xy=candidates_xy,
            config=config,
            centroid_xy=centroid_xy,
            direction_vec=direction_vec,
            angle_deg=angle_deg,
            inlier_mask=inlier_mask,
            inliers_xy=inliers_xy,
            n_inliers=n_inliers,
            inlier_fraction=n_inliers / len(candidates_xy) if len(candidates_xy) > 0 else 0.0,
            detected_support_y_range=support_y_range,
            segment_endpoints=endpoints,
            residual_median=res_med,
            residual_p95=res_p95,
            residual_max=res_max,
        )
