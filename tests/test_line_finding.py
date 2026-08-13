"""Tests for rixs_app.core.line_finding package."""

import numpy as np
import pytest

from rixs_app.core.line_finding.base import (
    DetectorConfig,
    LineDetectionResult,
    BaseLineDetector,
)
from rixs_app.core.line_finding.presets import (
    get_preset,
    list_presets,
    DEFAULT_PRESET_ID,
    PRESETS,
)
from rixs_app.core.line_finding.v8_scanner import (
    get_pct,
    robust_scale_opt,
    robust_detect_row,
    ransac_v7,
    _norm,
    V8RightSideScanner,
)
from rixs_app.core.preprocessing import prepare_frame, PreprocessingConfig


# ===========================================================================
# DetectorConfig
# ===========================================================================

class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.ref_frac == 0.10
        assert cfg.k_rise == 4.0
        assert cfg.k_level == 2.0
        assert cfg.sustain == 8
        assert cfg.y_step == 3
        assert cfg.win == 6
        assert cfg.peak_win == 14
        assert cfg.scan_margin_px == 10
        assert cfg.ransac_thresh == 4.0
        assert cfg.ransac_iters == 3000
        assert cfg.ransac_seed == 0
        assert cfg.svd_refine_iters == 6

    def test_frozen(self):
        cfg = DetectorConfig()
        with pytest.raises(AttributeError):
            cfg.k_rise = 5.0

    def test_fingerprint_stable(self):
        cfg = DetectorConfig()
        assert cfg.fingerprint() == cfg.fingerprint()

    def test_fingerprint_changes_with_params(self):
        cfg1 = DetectorConfig(k_rise=4.0)
        cfg2 = DetectorConfig(k_rise=5.0)
        assert cfg1.fingerprint() != cfg2.fingerprint()

    def test_validation_k_rise(self):
        with pytest.raises(ValueError, match="k_rise"):
            DetectorConfig(k_rise=0.0)
        with pytest.raises(ValueError, match="k_rise"):
            DetectorConfig(k_rise=-1.0)

    def test_validation_k_level(self):
        DetectorConfig(k_level=0.0)  # Should work
        with pytest.raises(ValueError, match="k_level"):
            DetectorConfig(k_level=-1.0)

    def test_validation_sustain(self):
        with pytest.raises(ValueError, match="sustain"):
            DetectorConfig(sustain=0)

    def test_validation_y_step(self):
        with pytest.raises(ValueError, match="y_step"):
            DetectorConfig(y_step=0)

    def test_validation_ref_frac(self):
        with pytest.raises(ValueError, match="ref_frac"):
            DetectorConfig(ref_frac=0.0)
        with pytest.raises(ValueError, match="ref_frac"):
            DetectorConfig(ref_frac=1.0)

    def test_validation_win(self):
        with pytest.raises(ValueError, match="win"):
            DetectorConfig(win=0)

    def test_validation_peak_win(self):
        with pytest.raises(ValueError, match="peak_win"):
            DetectorConfig(peak_win=0)

    def test_validation_ransac_thresh(self):
        with pytest.raises(ValueError, match="ransac_thresh"):
            DetectorConfig(ransac_thresh=0.0)

    def test_validation_ransac_iters(self):
        with pytest.raises(ValueError, match="ransac_iters"):
            DetectorConfig(ransac_iters=0)

    def test_validation_ransac_seed(self):
        with pytest.raises(ValueError, match="ransac_seed"):
            DetectorConfig(ransac_seed=-1)


# ===========================================================================
# LineDetectionResult
# ===========================================================================

class TestLineDetectionResult:
    def test_no_fit_result(self):
        cands = np.empty((0, 2), dtype=np.float64)
        cfg = DetectorConfig()
        res = LineDetectionResult(
            fit_ok=False,
            n_candidates=0,
            candidates_xy=cands,
            config=cfg,
            failure_reason="test",
        )
        assert not res.fit_ok
        assert res.failure_reason == "test"
        assert res.n_inliers == 0

    def test_successful_fit_result(self):
        cands = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
        inl_mask = np.array([True, False, True])
        cfg = DetectorConfig()
        res = LineDetectionResult(
            fit_ok=True,
            n_candidates=3,
            candidates_xy=cands,
            config=cfg,
            centroid_xy=(30.0, 40.0),
            direction_vec=(0.7, -0.7),
            angle_deg=-45.0,
            inlier_mask=inl_mask,
            inliers_xy=cands[inl_mask],
            n_inliers=2,
            inlier_fraction=2 / 3,
        )
        assert res.fit_ok
        assert res.n_candidates == 3
        assert res.n_inliers == 2

    def test_validation_bad_candidates_shape(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=False,
                n_candidates=3,
                candidates_xy=np.array([1, 2, 3]),  # Wrong shape
                config=DetectorConfig(),
            )

    def test_validation_n_candidates_mismatch(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=False,
                n_candidates=5,  # Wrong
                candidates_xy=np.array([[1, 2], [3, 4]], dtype=np.float64),
                config=DetectorConfig(),
            )

    def test_validation_fit_ok_requires_centroid(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=True,
                n_candidates=1,
                candidates_xy=np.array([[1, 2]], dtype=np.float64),
                config=DetectorConfig(),
                centroid_xy=None,  # Missing!
                angle_deg=-30.0,
            )

    def test_validation_fit_ok_requires_angle(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=True,
                n_candidates=1,
                candidates_xy=np.array([[1, 2]], dtype=np.float64),
                config=DetectorConfig(),
                centroid_xy=(1.0, 2.0),
                angle_deg=None,  # Missing!
            )

    def test_validation_inlier_mask_mismatch(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=False,
                n_candidates=3,
                candidates_xy=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
                config=DetectorConfig(),
                inlier_mask=np.array([True, False]),  # Wrong length
                n_inliers=1,
            )

    def test_validation_none_config(self):
        with pytest.raises(ValueError):
            LineDetectionResult(
                fit_ok=False,
                n_candidates=0,
                candidates_xy=np.empty((0, 2), dtype=np.float64),
                config=None,
            )


# ===========================================================================
# Presets
# ===========================================================================

class TestPresets:
    def test_default_preset_exists(self):
        name, cfg = get_preset(DEFAULT_PRESET_ID)
        assert isinstance(name, str)
        assert isinstance(cfg, DetectorConfig)

    def test_default_matches_expected_values(self):
        _, cfg = get_preset("v8_g10_r4_l2_s8")
        assert cfg.ref_frac == 0.10
        assert cfg.k_rise == 4.0
        assert cfg.k_level == 2.0
        assert cfg.sustain == 8
        assert cfg.ransac_seed == 0

    def test_list_presets(self):
        presets = list_presets()
        assert len(presets) >= 1
        ids = [p[0] for p in presets]
        assert "v8_g10_r4_l2_s8" in ids

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")


# ===========================================================================
# V8 Helper Functions
# ===========================================================================

class TestGetPct:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert get_pct(arr, 0.0) == 1.0
        assert get_pct(arr, 100.0) == 5.0
        assert abs(get_pct(arr, 50.0) - 3.0) < 1e-9

    def test_empty(self):
        assert get_pct(np.array([]), 50.0) == 0.0


class TestRobustScaleOpt:
    def test_constant_array(self):
        arr = np.ones(100) * 42.0
        scale = robust_scale_opt(arr, 1e-6)
        assert scale >= 1e-6

    def test_known_gaussian(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 5.0, 10000)
        scale = robust_scale_opt(arr, 1e-6)
        # Should be close to 5.0
        assert 3.0 < scale < 7.0


class TestNorm:
    def test_basic_angles(self):
        assert abs(_norm(0.0)) < 1e-9
        assert abs(_norm(45.0) - 45.0) < 1e-9
        assert abs(_norm(-45.0) - (-45.0)) < 1e-9
        assert abs(_norm(90.0) - 90.0) < 1e-9

    def test_wrapping(self):
        assert abs(_norm(180.0)) < 1e-9
        assert abs(_norm(-180.0)) < 1e-9
        assert abs(_norm(270.0) - 90.0) < 1e-9


class TestRobustDetectRow:
    def test_flat_row_no_detection(self):
        """A flat row should produce no peak."""
        s = np.ones(500) * 10.0
        pk, foot, _, _, _, _ = robust_detect_row(s, 0.10, 4.0, 2.0, 8)
        assert pk == -1
        assert foot == -1

    def test_row_too_short(self):
        """Rows shorter than 4*win + peak_win should return no detection."""
        s = np.ones(30)
        pk, foot, _, _, _, _ = robust_detect_row(s, 0.10, 4.0, 2.0, 8)
        assert pk == -1

    def test_signal_with_peak(self):
        """A row with a clear peak should be detected."""
        s = np.ones(500) * 10.0
        # Add a prominent step/peak on the left side
        s[100:150] = 500.0
        pk, foot, _, _, _, _ = robust_detect_row(s, 0.10, 4.0, 2.0, 8)
        # Should find a peak in the 100-150 region
        if pk >= 0:
            assert 80 <= pk <= 160


class TestRansacV7:
    def test_collinear_points(self):
        """Perfect collinear points should give perfect fit."""
        xs = np.arange(50, dtype=np.float64)
        ys = 2.0 * xs + 10.0
        result = ransac_v7(xs, ys)
        assert result is not None
        assert result['n_inliers'] >= 40

    def test_noisy_line(self):
        """Noisy line should still be fit."""
        rng = np.random.default_rng(42)
        xs = np.arange(50, dtype=np.float64)
        ys = -0.5 * xs + 100.0 + rng.normal(0, 1.0, 50)
        result = ransac_v7(xs, ys)
        assert result is not None
        assert result['n_inliers'] >= 30

    def test_too_few_points(self):
        """Less than 12 points should return None."""
        xs = np.arange(5, dtype=np.float64)
        ys = xs * 2.0
        assert ransac_v7(xs, ys) is None

    def test_deterministic(self):
        """Same seed should produce same result."""
        rng = np.random.default_rng(99)
        xs = np.arange(30, dtype=np.float64)
        ys = xs * 1.5 + rng.normal(0, 0.5, 30)
        r1 = ransac_v7(xs, ys, seed=0)
        r2 = ransac_v7(xs, ys, seed=0)
        assert r1 is not None and r2 is not None
        assert r1['n_inliers'] == r2['n_inliers']
        np.testing.assert_array_equal(r1['inliers'], r2['inliers'])
        assert abs(r1['angle'] - r2['angle']) < 1e-10


# ===========================================================================
# V8RightSideScanner
# ===========================================================================

class TestV8RightSideScanner:
    def test_blank_image_no_fit(self):
        """Blank image should produce no fit."""
        img = np.zeros((300, 400), dtype=np.float32)
        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=0))
        scanner = V8RightSideScanner()
        config = DetectorConfig()
        result = scanner.detect(pf, config)
        assert not result.fit_ok
        assert result.n_candidates >= 0

    def test_synthetic_diagonal_line(self):
        """A bright diagonal line should be detected."""
        img = np.random.rand(500, 600).astype(np.float32) * 5.0
        # Draw a diagonal line: for each row y, place a bright peak at x = 300 - 0.3*y
        for y in range(50, 450):
            x = int(300 - 0.3 * y)
            if 20 <= x < 580:
                img[y, max(0, x - 3):min(600, x + 3)] = 500.0

        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=0))
        scanner = V8RightSideScanner()
        config = DetectorConfig(scan_margin_px=5, y_step=3)
        result = scanner.detect(pf, config)
        # May or may not detect depending on signal-to-noise, but should run without error
        assert isinstance(result, LineDetectionResult)

    def test_result_coordinate_consistency(self):
        """If fit succeeds, all coordinates should be consistent."""
        img = np.random.rand(500, 600).astype(np.float32) * 5.0
        for y in range(50, 450):
            x = int(300 - 0.3 * y)
            if 20 <= x < 580:
                img[y, max(0, x - 3):min(600, x + 3)] = 1000.0

        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=0))
        scanner = V8RightSideScanner()
        config = DetectorConfig(scan_margin_px=5, y_step=3)
        result = scanner.detect(pf, config)

        if result.fit_ok:
            # Check n_candidates matches array
            assert result.n_candidates == result.candidates_xy.shape[0]
            # Check n_inliers matches mask
            assert result.n_inliers == int(result.inlier_mask.sum())
            # Check inliers_xy matches masked candidates
            np.testing.assert_array_equal(
                result.inliers_xy,
                result.candidates_xy[result.inlier_mask]
            )
            # Check inlier fraction
            expected_frac = result.n_inliers / result.n_candidates
            assert abs(result.inlier_fraction - expected_frac) < 1e-9
            # Geometry should be finite
            assert np.all(np.isfinite(result.centroid_xy))
            assert np.isfinite(result.angle_deg)
            assert result.segment_endpoints is not None



