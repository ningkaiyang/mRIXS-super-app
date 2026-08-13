"""Zeroth-order line FWHM evaluator using perpendicular Gaussian profile fitting.

This module computes a resolution (FWHM) score from the 1D perpendicular intensity
profile of the detected elastic line. The score is experimental and should
not be treated as validated physics.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class ZerothOrderConfig:
    """Configuration for the zeroth-order FWHM evaluator."""
    profile_half_width_px: float = 40.0
    profile_bin_size_px: float = 1.0
    smoothing_sigma: float = 1.5
    min_occupied_bins: int = 20
    min_prominence: float = 0.0
    max_sigma_px: float = 30.0


@dataclass
class ZerothOrderResult:
    """Result of zeroth-order FWHM evaluation."""
    score_valid: bool
    score: float | None = None
    score_label: str = "Zeroth-order: inverse fitted FWHM (px⁻¹)"
    failure_reason: str | None = None
    # Profile data
    profile_u: np.ndarray | None = None
    intensity_profile: np.ndarray | None = None
    gradient_profile: np.ndarray | None = None
    # Gaussian fit parameters
    gaussian_amplitude: float | None = None
    gaussian_center: float | None = None
    gaussian_sigma: float | None = None
    gaussian_background: float | None = None
    fwhm_px: float | None = None
    fwhm_mev: float | None = None
    prominence: float | None = None
    fit_covariance_finite: bool = False
    n_occupied_bins: int = 0
    config: ZerothOrderConfig | None = None


def _gaussian(u, B, A, u0, sigma):
    """Gaussian model: B + A * exp(-(u - u0)^2 / (2 * sigma^2))."""
    return B + A * np.exp(-((u - u0) ** 2) / (2.0 * sigma ** 2))


class ZerothOrderEvaluator:
    """Evaluates zeroth-order line width by fitting a Gaussian to the perpendicular intensity profile.

    The evaluator extracts a 1D intensity profile perpendicular to the fitted
    elastic line, fits a Gaussian, and returns 1/FWHM as the score.
    Higher values = sharper line.

    This is EXPERIMENTAL and should not be used as a validated physics measurement.
    """

    def evaluate(self, denoised, gradient, centroid_xy, angle_deg,
                 detected_support_y_range=None,
                 config: ZerothOrderConfig | None = None,
                 energy_dispersion: float = 0.0) -> ZerothOrderResult:
        """Compute zeroth-order FWHM from the 1D perpendicular intensity profile.

        Args:
            denoised: 2D denoised image array (D) in the coordinate space matching centroid_xy.
            gradient: 2D gradient magnitude array (G) in same coordinate space.
            centroid_xy: (x, y) center of the fitted line.
            angle_deg: Angle of the fitted line in degrees.
            detected_support_y_range: Optional (y_min, y_max) bounding the line support.
            config: Zeroth-order FWHM evaluation configuration.
            energy_dispersion: Energy scale in meV/px. Used to compute fwhm_mev.

        Returns:
            ZerothOrderResult with score and profile data.
        """
        if config is None:
            config = ZerothOrderConfig()

        h, w = denoised.shape
        cx, cy = centroid_xy
        theta = np.deg2rad(angle_deg)
        perp = np.array([-np.sin(theta), np.cos(theta)])

        # Determine y range
        if detected_support_y_range is not None:
            y_min, y_max = detected_support_y_range
            y_min = max(0, int(y_min))
            y_max = min(h, int(y_max))
        else:
            y_min, y_max = 0, h

        if y_max - y_min < 10:
            return ZerothOrderResult(
                score_valid=False, failure_reason="Support range too small",
                config=config
            )

        # Build bin edges
        half_w = config.profile_half_width_px
        bin_size = config.profile_bin_size_px
        u_bins = np.arange(-half_w - bin_size/2, half_w + bin_size/2 + 1e-9, bin_size)
        u_centers = (u_bins[:-1] + u_bins[1:]) / 2.0
        n_bins = len(u_centers)

        # Accumulate intensity and gradient profiles using only the support region
        intensity_accum = np.zeros(n_bins, dtype=np.float64)
        gradient_accum = np.zeros(n_bins, dtype=np.float64)
        count = np.zeros(n_bins, dtype=np.int64)

        # Efficient: iterate over rows in support range
        for y in range(y_min, y_max):
            row_d = denoised[y, :]
            row_g = gradient[y, :]
            xs = np.arange(w, dtype=np.float64)
            u_vals = (xs - cx) * perp[0] + (y - cy) * perp[1]

            # Bin assignment
            bin_idx = np.searchsorted(u_bins, u_vals) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)

            for b_idx in np.unique(bin_idx[valid]):
                mask = (bin_idx == b_idx) & valid
                intensity_accum[b_idx] += np.sum(row_d[mask])
                gradient_accum[b_idx] += np.sum(row_g[mask])
                count[b_idx] += np.sum(mask)

        # Normalize by count (occupancy-weighted mean)
        occupied = count > 0
        n_occupied = int(np.sum(occupied))

        if n_occupied < config.min_occupied_bins:
            return ZerothOrderResult(
                score_valid=False,
                failure_reason=f"Only {n_occupied} occupied bins (need {config.min_occupied_bins})",
                profile_u=u_centers,
                intensity_profile=np.zeros(n_bins),
                gradient_profile=np.zeros(n_bins),
                n_occupied_bins=n_occupied,
                config=config
            )

        intensity_profile = np.zeros(n_bins)
        gradient_profile_arr = np.zeros(n_bins)
        intensity_profile[occupied] = intensity_accum[occupied] / count[occupied]
        gradient_profile_arr[occupied] = gradient_accum[occupied] / count[occupied]

        # Smooth
        intensity_smooth = gaussian_filter1d(intensity_profile, sigma=config.smoothing_sigma)

        # Baseline correction using outer shoulders
        shoulder_mask = (np.abs(u_centers) > half_w * 0.7) & occupied
        if np.sum(shoulder_mask) >= 4:
            baseline = np.mean(intensity_smooth[shoulder_mask])
        else:
            baseline = np.min(intensity_smooth[occupied])

        peak_val = np.max(intensity_smooth)
        prominence = peak_val - baseline

        if prominence <= config.min_prominence:
            return ZerothOrderResult(
                score_valid=False,
                failure_reason=f"Insufficient prominence: {prominence:.2f}",
                profile_u=u_centers,
                intensity_profile=intensity_smooth,
                gradient_profile=gradient_profile_arr,
                prominence=prominence,
                n_occupied_bins=n_occupied,
                config=config
            )

        # Fit Gaussian: B + A * exp(-(u - u0)^2 / (2*sigma^2))
        peak_idx = np.argmax(intensity_smooth)
        p0 = [baseline, prominence, u_centers[peak_idx], 3.0]
        bounds = ([0, 0, -half_w, 0.5], [peak_val * 2, prominence * 3, half_w, config.max_sigma_px])

        try:
            popt, pcov = curve_fit(
                _gaussian, u_centers[occupied], intensity_smooth[occupied],
                p0=p0, bounds=bounds, maxfev=2000
            )
            B_fit, A_fit, u0_fit, sigma_fit = popt
            cov_finite = bool(np.all(np.isfinite(pcov)))
        except (RuntimeError, ValueError) as e:
            return ZerothOrderResult(
                score_valid=False,
                failure_reason=f"Gaussian fit failed: {e}",
                profile_u=u_centers,
                intensity_profile=intensity_smooth,
                gradient_profile=gradient_profile_arr,
                prominence=prominence,
                n_occupied_bins=n_occupied,
                config=config
            )

        # Validity checks
        fwhm = 2.35482 * sigma_fit  # 2 * sqrt(2 * ln(2)) * sigma

        if sigma_fit >= config.max_sigma_px - 0.1:
            failure = f"Sigma railed at max: {sigma_fit:.2f} >= {config.max_sigma_px}"
        elif abs(u0_fit) > half_w * 0.8:
            failure = f"Center outside window: u0={u0_fit:.2f}"
        elif not cov_finite:
            failure = "Fit covariance not finite"
        elif A_fit < 0.1:
            failure = f"Negligible amplitude: {A_fit:.4f}"
        else:
            failure = None

        score = 1.0 / fwhm if failure is None and fwhm > 0 else None

        # Compute fwhm_mev if energy_dispersion is provided
        fwhm_mev = float(fwhm * energy_dispersion) if (failure is None and energy_dispersion > 0) else None

        return ZerothOrderResult(
            score_valid=(failure is None),
            score=score,
            failure_reason=failure,
            profile_u=u_centers,
            intensity_profile=intensity_smooth,
            gradient_profile=gradient_profile_arr,
            gaussian_amplitude=float(A_fit),
            gaussian_center=float(u0_fit),
            gaussian_sigma=float(sigma_fit),
            gaussian_background=float(B_fit),
            fwhm_px=float(fwhm),
            fwhm_mev=fwhm_mev,
            prominence=float(prominence),
            fit_covariance_finite=cov_finite,
            n_occupied_bins=n_occupied,
            config=config
        )
