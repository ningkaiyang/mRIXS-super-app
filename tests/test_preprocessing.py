"""Tests for rixs_app.core.preprocessing module."""

import numpy as np
import pytest

from rixs_app.core.preprocessing import (
    PreprocessingConfig,
    CropTransform,
    PreparedFrame,
    prepare_frame,
    compute_gradients,
)


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------

class TestPreprocessingConfig:
    def test_defaults(self):
        cfg = PreprocessingConfig()
        assert cfg.edge_crop_px == 100
        assert cfg.smooth_sigma == 2.5
        assert cfg.mad_threshold == 5.0
        assert cfg.bilateral_d == 5
        assert cfg.bilateral_sigma_color == 1.5
        assert cfg.bilateral_sigma_space == 3.0

    def test_frozen(self):
        cfg = PreprocessingConfig()
        with pytest.raises(AttributeError):
            cfg.edge_crop_px = 50

    def test_custom_values(self):
        cfg = PreprocessingConfig(edge_crop_px=50, smooth_sigma=3.0)
        assert cfg.edge_crop_px == 50
        assert cfg.smooth_sigma == 3.0


# ---------------------------------------------------------------------------
# CropTransform
# ---------------------------------------------------------------------------

class TestCropTransform:
    @pytest.fixture
    def ct(self):
        return CropTransform(
            crop_top=100, crop_left=100,
            crop_bottom=100, crop_right=100,
            original_shape=(2048, 3840),
            cropped_shape=(1848, 3640),
        )

    def test_cropped_to_original(self, ct):
        ox, oy = ct.cropped_to_original(0.0, 0.0)
        assert ox == 100.0
        assert oy == 100.0

    def test_cropped_to_original_nonzero(self, ct):
        ox, oy = ct.cropped_to_original(50.5, 25.3)
        assert abs(ox - 150.5) < 1e-9
        assert abs(oy - 125.3) < 1e-9

    def test_original_to_cropped(self, ct):
        cx, cy = ct.original_to_cropped(150.0, 125.0)
        assert abs(cx - 50.0) < 1e-9
        assert abs(cy - 25.0) < 1e-9

    def test_roundtrip(self, ct):
        x, y = 42.7, 99.1
        ox, oy = ct.cropped_to_original(x, y)
        rx, ry = ct.original_to_cropped(ox, oy)
        assert abs(rx - x) < 1e-9
        assert abs(ry - y) < 1e-9

    def test_cropped_to_original_array(self, ct):
        pts = np.array([[0.0, 0.0], [50.0, 25.0], [100.0, 100.0]])
        result = ct.cropped_to_original_array(pts)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], [100.0, 100.0])
        np.testing.assert_allclose(result[1], [150.0, 125.0])
        np.testing.assert_allclose(result[2], [200.0, 200.0])

    def test_cropped_to_original_array_empty(self, ct):
        pts = np.empty((0, 2))
        result = ct.cropped_to_original_array(pts)
        assert result.shape == (0, 2)

    def test_cropped_to_original_array_bad_shape(self, ct):
        with pytest.raises(ValueError):
            ct.cropped_to_original_array(np.array([1, 2, 3]))

    def test_frozen(self, ct):
        with pytest.raises(AttributeError):
            ct.crop_top = 50

    def test_no_crop(self):
        ct = CropTransform(0, 0, 0, 0, (100, 100), (100, 100))
        ox, oy = ct.cropped_to_original(50.0, 50.0)
        assert ox == 50.0 and oy == 50.0


# ---------------------------------------------------------------------------
# compute_gradients
# ---------------------------------------------------------------------------

class TestComputeGradients:
    def test_output_shape(self):
        img = np.random.rand(100, 200).astype(np.float32)
        G = compute_gradients(img)
        assert G.shape == (100, 200)
        assert G.dtype == np.float32

    def test_flat_image_near_zero(self):
        img = np.ones((50, 50), dtype=np.float32) * 100.0
        G = compute_gradients(img)
        # Flat image should have near-zero gradients (except edges from filter)
        center = G[10:-10, 10:-10]
        assert np.max(center) < 1.0

    def test_nonnegative(self):
        img = np.random.rand(80, 120).astype(np.float32) * 1000
        G = compute_gradients(img)
        assert np.all(G >= 0)


# ---------------------------------------------------------------------------
# prepare_frame
# ---------------------------------------------------------------------------

class TestPrepareFrame:
    def test_basic_output_structure(self):
        img = np.random.rand(300, 400).astype(np.float32) * 100
        pf = prepare_frame(img)
        assert isinstance(pf, PreparedFrame)
        np.testing.assert_array_equal(pf.raw, img)
        assert isinstance(pf.crop_transform, CropTransform)
        assert isinstance(pf.config, PreprocessingConfig)

    def test_crop_applied_for_large_image(self):
        img = np.random.rand(500, 600).astype(np.float32) * 100
        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=50))
        assert pf.cropped_raw.shape == (400, 500)
        assert pf.denoised.shape == (400, 500)
        assert pf.row_smoothed.shape == (400, 500)
        assert pf.gradient.shape == (400, 500)
        assert pf.crop_transform.crop_top == 50
        assert pf.crop_transform.crop_left == 50

    def test_no_crop_for_small_image(self):
        img = np.random.rand(200, 200).astype(np.float32) * 100
        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=100))
        # Image is not large enough: 200 < 2*100+50 = 250
        assert pf.cropped_raw.shape == (200, 200)
        assert pf.crop_transform.crop_top == 0

    def test_default_config(self):
        img = np.random.rand(300, 400).astype(np.float32) * 100
        pf = prepare_frame(img)
        assert pf.config.edge_crop_px == 100

    def test_denoised_is_float32(self):
        img = np.random.rand(300, 400).astype(np.float32) * 100
        pf = prepare_frame(img)
        assert pf.denoised.dtype == np.float32

    def test_row_smoothed_differs_from_denoised(self):
        """Row smoothing should change the image (unless perfectly flat)."""
        img = np.random.rand(300, 400).astype(np.float32) * 1000
        pf = prepare_frame(img)
        # They should be different unless the image is perfectly flat
        assert not np.array_equal(pf.denoised, pf.row_smoothed)

    def test_crop_order_matches_v8(self):
        """Verify crop happens BEFORE denoise: cropped shape should be used for denoised."""
        img = np.random.rand(500, 600).astype(np.float32) * 100
        pf = prepare_frame(img, PreprocessingConfig(edge_crop_px=50))
        # If crop happened first, denoised shape matches cropped, not original
        assert pf.denoised.shape == pf.cropped_raw.shape
        assert pf.denoised.shape != img.shape

    def test_original_preserved(self):
        """Raw image should be stored without modification."""
        img = np.random.rand(300, 400).astype(np.float32) * 100
        pf = prepare_frame(img)
        # raw should be the float32 version of input
        assert pf.raw.shape == (300, 400)
