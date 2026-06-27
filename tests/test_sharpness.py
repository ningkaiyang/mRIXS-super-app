import os
import re
import sys
import subprocess
import json
import tempfile
import time
import numpy as np
import pytest
import tifffile
from rixs_app.core.sharpness import denoise_image, evaluate_sharpness, run_sharpness_pipeline

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------------------------
# 1. Extremely Small Images
# -----------------------------------------------------------------------------

def test_small_images_norm_sum_sq_grad():
    """Test evaluate_sharpness with norm_sum_sq_grad on very small images."""
    for size in [1, 2, 3, 5]:
        img = np.ones((size, size), dtype=np.float64)
        score = evaluate_sharpness(img, "norm_sum_sq_grad")
        assert isinstance(score, float)
        assert score == 0.0  # Constant images should have 0.0 sharpness score

    img_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    score_2x2 = evaluate_sharpness(img_2x2, "norm_sum_sq_grad")
    assert isinstance(score_2x2, float)
    assert score_2x2 >= 0.0


def test_small_images_peak_height():
    """Test evaluate_sharpness with peak_height on very small images."""
    for size in [1, 2, 3, 5]:
        img = np.ones((size, size), dtype=np.float64)
        score = evaluate_sharpness(img, "peak_height")
        assert isinstance(score, float)
        assert score == 0.0  # Constant small flat images sum to ~0.0 or less

    img_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    score_2x2 = evaluate_sharpness(img_2x2, "peak_height")
    assert isinstance(score_2x2, float)
    assert score_2x2 >= 0.0


def test_small_images_denoise():
    """Test denoise_image on extremely small images."""
    for size in [1, 2, 3]:
        img = np.ones((size, size), dtype=np.float32)
        
        # Test full pipeline
        denoised = denoise_image(img)
        assert denoised.shape == (size, size)
        assert denoised.dtype == np.float32

        # Test with and without specific pipeline stages
        denoised_no_bilateral = denoise_image(img, bilateral=False)
        assert denoised_no_bilateral.shape == (size, size)

        denoised_no_anscombe = denoise_image(img, anscombe=False, inverse_anscombe=False)
        assert denoised_no_anscombe.shape == (size, size)


# -----------------------------------------------------------------------------
# 2. Flat Constant Images
# -----------------------------------------------------------------------------

def test_flat_constant_images():
    """Test how flat constant images affect the 1D profile metrics."""
    sizes = [10, 50, 100]
    constants = [1.0, 50.0, 1000.0]

    for size in sizes:
        for c in constants:
            img = np.ones((size, size), dtype=np.float64) * c
            
            score_grad = evaluate_sharpness(img, "norm_sum_sq_grad")
            assert score_grad == 0.0
            
            score_peak = evaluate_sharpness(img, "peak_height")
            assert score_peak == 0.0


# -----------------------------------------------------------------------------
# 3. NaNs, Infs, and Negative Values
# -----------------------------------------------------------------------------

def test_denoise_image_nan_inf_negatives():
    """Test denoise_image behavior with NaN, Inf, and negative values."""
    img = np.array([
        [np.nan, 10.0, np.inf],
        [-5.0, -100.0, -np.inf],
        [20.0, np.nan, 30.0]
    ], dtype=np.float32)

    # With clip=True (default): negatives clipped to 0.0, NaNs/Infs mapped to 0.0
    denoised_clipped_no_bilat = denoise_image(img, clip=True, despike=False, bilateral=False)
    assert not np.isnan(denoised_clipped_no_bilat).any()
    assert not np.isinf(denoised_clipped_no_bilat).any()
    assert (denoised_clipped_no_bilat >= 0.0).all()
    assert np.isclose(denoised_clipped_no_bilat[0, 0], 0.0, atol=1e-7)  # nan -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[0, 2], 0.0, atol=1e-7)  # inf -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[1, 0], 0.0, atol=1e-7)  # -5.0 -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[1, 2], 0.0, atol=1e-7)  # -inf -> 0.0

    # With bilateral=True (default): bilateral smoothing spreads values from neighbors
    denoised_clipped_bilat = denoise_image(img, clip=True, despike=False, bilateral=True)
    assert not np.isnan(denoised_clipped_bilat).any()
    assert not np.isinf(denoised_clipped_bilat).any()
    assert (denoised_clipped_bilat >= 0.0).all()
    assert denoised_clipped_bilat[0, 0] > 0.0

    # With clip=False: negative values can be preserved
    denoised_unclipped = denoise_image(img, clip=False, despike=False, bilateral=False)
    assert not np.isnan(denoised_unclipped).any()
    assert not np.isinf(denoised_unclipped).any()
    assert (denoised_unclipped >= -0.375).all()
    assert np.isclose(denoised_unclipped[1, 0], -0.375)


def test_evaluate_sharpness_nan_inf_negatives():
    """Test evaluate_sharpness behavior with NaN, Inf, and negative values."""
    img = np.array([
        [np.nan, 10.0, np.inf],
        [-5.0, -100.0, -np.inf],
        [20.0, np.nan, 30.0]
    ], dtype=np.float64)

    for metric in ["norm_sum_sq_grad", "peak_height"]:
        score = evaluate_sharpness(img, metric)
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)


# -----------------------------------------------------------------------------
# 4. CLI Frame Index Parsing Robustness
# -----------------------------------------------------------------------------

def test_cli_regex_pattern_unit():
    """Directly test the frame index parsing logic on typical and unusual filenames."""
    from sharpness_cli import extract_frame_index

    # Standard prefixed cases work as expected
    assert extract_frame_index("frame_001.tif") == 1
    assert extract_frame_index("frame_-001.tif") == -1
    assert extract_frame_index("CMOS Detector 123.tif") == 123

    # Non-prefixed numbers should work correctly
    assert extract_frame_index("123.tif") == 123
    assert extract_frame_index("-123.tif") == -123
    assert extract_frame_index("123_456.tif") == 456
    assert extract_frame_index("image_123.tif") == 123
    assert extract_frame_index("20260625_frame_12.tif") == 12


def test_cli_frame_index_parsing_integration(tmp_path):
    """Integration test checking CLI behavior with various unusual file names."""
    scan_dir = tmp_path / "unusual_scan"
    scan_dir.mkdir(exist_ok=True)

    test_files = {
        "frame_015.tif": 15,
        "CMOS Detector 042.tif": 42,
        "123.tif": 123,
        "image_256.tif": 256,
        "frame_-007.tif": -7,
        "20260625_frame_099.tif": 99
    }

    # Write small dummy TIFF files
    for name in test_files.keys():
        data = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(scan_dir / name, data)

    cmd = [
        sys.executable, "sharpness_cli.py",
        "--dir", str(scan_dir),
        "--metrics", "norm_sum_sq_grad",
        "--print-scores"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    stdout = res.stdout
    print("CLI STDOUT:\n", stdout)

    for name, expected_idx in test_files.items():
        expected_line = f"Frame {expected_idx} (norm_sum_sq_grad)"
        assert expected_line in stdout


# -----------------------------------------------------------------------------
# 5. Performance Benchmarking & Pipeline tests
# -----------------------------------------------------------------------------

def test_run_sharpness_pipeline_structure():
    """Verify that run_sharpness_pipeline runs successfully and returns the correct dictionary format."""
    img = np.random.rand(100, 100).astype(np.float32)
    res = run_sharpness_pipeline(img, metric="norm_sum_sq_grad")
    
    assert isinstance(res, dict)
    assert "raw_img" in res
    assert "denoised_img" in res
    assert "masked_img" in res
    assert "centroid" in res
    assert "direction" in res
    assert "1d_profile" in res
    assert "score" in res
    
    assert res["raw_img"].shape == (100, 100)
    assert res["denoised_img"].shape == (100, 100)
    assert res["masked_img"].shape == (100, 100)
    assert isinstance(res["centroid"], np.ndarray)
    assert isinstance(res["direction"], np.ndarray)
    assert isinstance(res["1d_profile"], tuple)
    assert len(res["1d_profile"]) == 2
    assert isinstance(res["score"], float)


def test_performance_benchmarking(tmp_path):
    """Performance benchmarking on simulated TIFF files."""
    scan_dir = tmp_path / "large_scan"
    scan_dir.mkdir()
    
    num_frames = 50
    height, width = 256, 256
    best_frame = 25
    
    for idx in range(num_frames):
        data = np.random.normal(loc=100.0, scale=10.0, size=(height, width)).astype(np.float32)
        dist = abs(idx - best_frame)
        sigma = 1.0 + dist * 0.5
        y, x = np.ogrid[:height, :width]
        profile = 500.0 * np.exp(-((x - width//2)**2) / (2.0 * sigma**2))
        data += profile
        
        tifffile.imwrite(scan_dir / f"frame_{idx:03d}.tif", data)
        
    ranks = {str(idx): float(1.0 + abs(idx - best_frame)) for idx in range(num_frames)}
    gt_data = {
        "experiment_id": "large_scan",
        "best_frame_index": best_frame,
        "fractional_ranks": ranks
    }
    with open(scan_dir / "ground_truth.json", "w") as f:
        json.dump(gt_data, f)
        
    for metric in ["norm_sum_sq_grad", "peak_height"]:
        start_time = time.time()
        for idx in range(num_frames):
            img = tifffile.imread(scan_dir / f"frame_{idx:03d}.tif")
            denoised = denoise_image(img)
            score = evaluate_sharpness(denoised, metric)
            assert not np.isnan(score)
            assert not np.isinf(score)
        duration = time.time() - start_time
        avg_time_ms = (duration / num_frames) * 1000.0
        print(f"Direct evaluation average time for {metric}: {avg_time_ms:.2f} ms/frame")
        assert avg_time_ms < 150.0
        
    cmd = [
        sys.executable, "sharpness_cli.py",
        "--dir", str(scan_dir),
        "--denoise",
        "--correlation"
    ]
    start_time = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cli_duration = time.time() - start_time
    print(f"CLI tool execution on 50 frames: {cli_duration:.2f} seconds")
    assert "|" in res.stdout
    assert "norm_sum_sq_grad" in res.stdout
    assert "peak_height" in res.stdout
    assert cli_duration < 10.0


def test_extreme_intensities_safe_1e12():
    """Verify correctness and stability under extreme intensities."""
    shapes = [(100, 100), (50, 150)]
    values = [1e12, -1e12, 1e-12, -1e-12]
    
    for shape in shapes:
        for val in values:
            img_flat = np.full(shape, val, dtype=np.float64)
            
            denoised = denoise_image(img_flat, clip=True)
            assert not np.isnan(denoised).any()
            assert not np.isinf(denoised).any()
            
            denoised_no_clip = denoise_image(img_flat, clip=False)
            assert not np.isnan(denoised_no_clip).any()
            assert not np.isinf(denoised_no_clip).any()
            
            for metric in ["norm_sum_sq_grad", "peak_height"]:
                score = evaluate_sharpness(img_flat, metric)
                assert isinstance(score, (float, int, np.floating))
                assert not np.isnan(score)
                assert not np.isinf(score)


def test_extreme_intensities_overflow_bug():
    """Stress test denoise_image under values exceeding float32 max limit."""
    img_flat = np.full((100, 100), 1e150, dtype=np.float64)
    try:
        denoised = denoise_image(img_flat, clip=True)
        assert not np.isnan(denoised).any(), "Image was corrupted with NaNs during float32 cast overflow"
        assert not np.isinf(denoised).any(), "Image contains unhandled Infinities"
    except AssertionError as e:
        print(f"Discovered vulnerability: {e}")
        raise


def test_1d_metrics_overflow_safety():
    """Stress test sharpness evaluation under values that cause float64 overflow."""
    img = np.zeros((100, 100), dtype=np.float64)
    img[50:, :] = 1e200
    
    for metric in ["norm_sum_sq_grad", "peak_height"]:
        score = evaluate_sharpness(img, metric)
        assert np.isfinite(score), f"{metric} metric returned non-finite score due to float64 overflow"


def test_non_square_aspect_ratio_compatibility():
    """Non-square aspect ratio compatibility."""
    aspect_ratios = [
        (1000, 10),
        (10, 1000),
        (100, 20),
        (20, 100),
        (50, 2),
        (2, 50)
    ]
    
    for H, W in aspect_ratios:
        img = np.zeros((H, W), dtype=np.float32)
        for i in range(min(H, W)):
            h_start = i * (H // min(H, W))
            h_end = min((i + 1) * (H // min(H, W)), H)
            w_start = i * (W // min(H, W))
            w_end = min((i + 1) * (W // min(H, W)), W)
            img[h_start:h_end, w_start:w_end] = 100.0
            
        try:
            denoised = denoise_image(img)
            assert denoised.shape == (H, W)
        except Exception as e:
            pytest.fail(f"denoise_image failed on non-square shape ({H}, {W}): {e}")
            
        for metric in ["norm_sum_sq_grad", "peak_height"]:
            try:
                score = evaluate_sharpness(img, metric)
                assert isinstance(score, (float, int, np.floating))
                assert not np.isnan(score)
                assert not np.isinf(score)
            except Exception as e:
                pytest.fail(f"evaluate_sharpness({metric}) failed on non-square shape ({H}, {W}): {e}")
