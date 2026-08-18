import numpy as np
import pytest
from rixs_app.core.zeroth_order_evaluator import ZerothOrderEvaluator, ZerothOrderConfig, _gaussian

def test_zeroth_order_config_defaults():
    config = ZerothOrderConfig()
    assert config.profile_half_width_px == 40.0
    assert config.min_occupied_bins == 20

def test_synthetic_gaussian_line():
    evaluator = ZerothOrderEvaluator()

    # Create an image with a clear vertical Gaussian line at x=50
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.float64)
    grad = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            val = _gaussian(x, B=10, A=100, u0=50, sigma=2.0)
            img[y, x] = val
            grad[y, x] = val # just some non-zero values

    # Centroid at 50, 50, angle 90 degrees (vertical line)
    # The normal to vertical line (angle=90) is perp = (-1, 0)
    # So u_vals = (x-cx)*(-1) + (y-cy)*(0) = 50 - x
    res = evaluator.evaluate(img, grad, centroid_xy=(50, 50), angle_deg=90)

    assert res.score_valid
    assert res.score is not None
    assert res.r_squared is not None and res.r_squared > 0.95
    # Check that FWHM is approx 2.355 * sqrt(2.0^2 + 1.5^2) = 5.887
    assert abs(res.fwhm_px - 5.887) < 0.2
    assert abs(res.score - 1.0 / res.fwhm_px) < 1e-5
    assert abs(res.gaussian_center) < 1.0

def test_flat_image():
    evaluator = ZerothOrderEvaluator()
    img = np.zeros((100, 100), dtype=np.float64)
    grad = np.zeros((100, 100), dtype=np.float64)

    res = evaluator.evaluate(img, grad, centroid_xy=(50, 50), angle_deg=90)
    assert not res.score_valid
    assert res.failure_reason is not None

def test_none_support_range():
    evaluator = ZerothOrderEvaluator()
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.float64)
    grad = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            img[y, x] = _gaussian(x, B=10, A=100, u0=50, sigma=2.0)

    # detected_support_y_range = None
    res = evaluator.evaluate(img, grad, centroid_xy=(50, 50), angle_deg=90, detected_support_y_range=None)
    assert res.score_valid
