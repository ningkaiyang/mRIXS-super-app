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
from rixs_app.core.sharpness import denoise_image, evaluate_sharpness

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -----------------------------------------------------------------------------
# 1. Extremely Small Images
# -----------------------------------------------------------------------------

def test_small_images_dog_laplacian():
    """Test evaluate_sharpness with dog_laplacian on very small images."""
    for size in [1, 2, 3, 5]:
        img = np.ones((size, size), dtype=np.float64)
        score = evaluate_sharpness(img, "dog_laplacian")
        assert isinstance(score, float)
        assert score == 0.0  # Constant images should have 0 variance of Laplacian

    # Non-constant small images
    img_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    score_2x2 = evaluate_sharpness(img_2x2, "dog_laplacian")
    assert isinstance(score_2x2, float)
    assert score_2x2 >= 0.0

    img_3x3 = np.arange(9, dtype=np.float64).reshape((3, 3))
    score_3x3 = evaluate_sharpness(img_3x3, "dog_laplacian")
    assert isinstance(score_3x3, float)
    assert score_3x3 >= 0.0


def test_small_images_directional_tenengrad():
    """Test evaluate_sharpness with directional_tenengrad on small images.
    
    Verifies that Sobel's mirroring behavior causes 2x2 and smaller images
    to always return 0.0.
    """
    # 1x1 and 2x2 images should always return 0.0 because of Sobel border mirroring
    for size in [1, 2]:
        img = np.arange(size * size, dtype=np.float64).reshape((size, size))
        score = evaluate_sharpness(img, "directional_tenengrad")
        assert score == 0.0

    # 3x3 non-constant image can have non-zero Tenengrad score
    img_3x3 = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 5.0, 0.0],
        [1.0, 0.0, 1.0]
    ], dtype=np.float64)
    score_3x3 = evaluate_sharpness(img_3x3, "directional_tenengrad")
    assert isinstance(score_3x3, float)


def test_small_images_fft_bandpass():
    """Test evaluate_sharpness with fft_bandpass on small images.
    
    Checks how Hanning window length behaves for 1x1, 2x2, and larger sizes.
    """
    # 1x1 flat image returns 1.0 because hanning(1) is [1.] and mask covers the single point
    img_1x1 = np.ones((1, 1), dtype=np.float64)
    score_1x1 = evaluate_sharpness(img_1x1, "fft_bandpass")
    assert score_1x1 == 1.0

    # 2x2 image returns 0.0 because hanning(2) is [0., 0.], meaning total energy is 0
    img_2x2 = np.ones((2, 2), dtype=np.float64)
    score_2x2 = evaluate_sharpness(img_2x2, "fft_bandpass")
    assert score_2x2 == 0.0

    # 3x3 flat image returns 4/9 (~0.4444) because of Hanning window zeroes and mask radius
    img_3x3 = np.ones((3, 3), dtype=np.float64)
    score_3x3 = evaluate_sharpness(img_3x3, "fft_bandpass")
    assert np.isclose(score_3x3, 4.0 / 9.0)


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
    """Test how flat constant images affect the metrics.
    
    Particularly:
    - directional_tenengrad: Should return 0.0 because percentile thresholding
      excludes all pixels when all gradients are equal to the threshold.
    - fft_bandpass: Can return non-zero energy ratios because the Hanning window
      introduces artificial gradients at the borders.
    """
    sizes = [10, 50, 100]
    constants = [1.0, 50.0, 1000.0]

    for size in sizes:
        for c in constants:
            img = np.ones((size, size), dtype=np.float64) * c
            
            # Tenengrad should return 0.0
            tenengrad_score = evaluate_sharpness(img, "directional_tenengrad")
            assert tenengrad_score == 0.0
            
            # FFT bandpass should return a small non-zero score due to Hanning window edge tapering
            fft_score = evaluate_sharpness(img, "fft_bandpass")
            assert fft_score > 0.0
            # The FFT score is independent of the scale of the constant value
            fft_score_base = evaluate_sharpness(np.ones((size, size), dtype=np.float64), "fft_bandpass")
            assert np.isclose(fft_score, fft_score_base)


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
    # To check exact mapping without bilateral smoothing, we disable bilateral filter
    denoised_clipped_no_bilat = denoise_image(img, clip=True, despike=False, bilateral=False)
    assert not np.isnan(denoised_clipped_no_bilat).any()
    assert not np.isinf(denoised_clipped_no_bilat).any()
    assert (denoised_clipped_no_bilat >= 0.0).all()
    # Mapped values:
    # np.nan, np.inf, -np.inf, and negatives should all end up very close to 0.0 after clipping
    assert np.isclose(denoised_clipped_no_bilat[0, 0], 0.0, atol=1e-7)  # nan -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[0, 2], 0.0, atol=1e-7)  # inf -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[1, 0], 0.0, atol=1e-7)  # -5.0 -> 0.0
    assert np.isclose(denoised_clipped_no_bilat[1, 2], 0.0, atol=1e-7)  # -inf -> 0.0

    # With bilateral=True (default): bilateral smoothing spreads values from neighbors
    denoised_clipped_bilat = denoise_image(img, clip=True, despike=False, bilateral=True)
    assert not np.isnan(denoised_clipped_bilat).any()
    assert not np.isinf(denoised_clipped_bilat).any()
    assert (denoised_clipped_bilat >= 0.0).all()
    # Because of bilateral smoothing, the sanitized 0.0 at (0, 0) receives energy from neighbors
    assert denoised_clipped_bilat[0, 0] > 0.0

    # With clip=False: negative values can be preserved (specifically after inverse Anscombe VST)
    denoised_unclipped = denoise_image(img, clip=False, despike=False, bilateral=False)
    assert not np.isnan(denoised_unclipped).any()
    assert not np.isinf(denoised_unclipped).any()
    # The negative values and NaNs (converted to 0.0) go through Anscombe: 2 * sqrt(x + 0.375)
    # 0.0 + 0.375 = 0.375 -> sqrt(0.375) -> Anscombe VST value ~ 1.2247
    # For a negative input like -5.0, np.maximum(-5 + 0.375, 0) = 0.0 -> Anscombe VST value = 0.0
    # Then inverse Anscombe: (val / 2.0)**2 - 0.375
    # For val = 0.0, this becomes -0.375.
    # So we expect negative outputs to reach -0.375
    assert (denoised_unclipped >= -0.375).all()
    # Let's verify that -5.0 input maps to -0.375 when clip=False
    assert np.isclose(denoised_unclipped[1, 0], -0.375)


def test_evaluate_sharpness_nan_inf_negatives():
    """Test evaluate_sharpness behavior with NaN, Inf, and negative values."""
    img = np.array([
        [np.nan, 10.0, np.inf],
        [-5.0, -100.0, -np.inf],
        [20.0, np.nan, 30.0]
    ], dtype=np.float64)

    for metric in ["dog_laplacian", "directional_tenengrad", "fft_bandpass"]:
        # Should not crash and should return a valid float
        score = evaluate_sharpness(img, metric)
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)


# -----------------------------------------------------------------------------
# 4. CLI Frame Index Parsing Robustness
# -----------------------------------------------------------------------------

def test_cli_regex_pattern_unit():
    """Directly test the frame index parsing logic in sharpness_cli.py on typical and unusual filenames.
    
    Verifies that the new robust multi-stage match correctly parses typical and unusual filenames
    without backtracking issues or incorrect digit splitting.
    """
    from sharpness_cli import extract_frame_index

    # Standard prefixed cases work as expected
    assert extract_frame_index("frame_001.tif") == 1
    assert extract_frame_index("frame_-001.tif") == -1
    assert extract_frame_index("CMOS Detector 123.tif") == 123

    # Non-prefixed numbers should now work correctly and not have the backtracking flaw
    assert extract_frame_index("123.tif") == 123
    assert extract_frame_index("-123.tif") == -123
    assert extract_frame_index("123_456.tif") == 456
    assert extract_frame_index("image_123.tif") == 123
    assert extract_frame_index("20260625_frame_12.tif") == 12


def test_cli_frame_index_parsing_integration(tmp_path):
    """Integration test checking CLI behavior with various unusual file names.
    
    Creates a temporary scan directory containing TIFF images named with unusual
    patterns, runs sharpness_cli.py via subprocess, and verifies how it parses the index.
    """
    scan_dir = tmp_path / "unusual_scan"
    scan_dir.mkdir(exist_ok=True)

    # Create dummy images with various names
    # Name -> expected parsed index according to robust behavior
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

    # Write a dummy ground_truth.json to avoid errors
    # Note: we use the indices that the CLI ACTUALLY parses so it finds them,
    # or just run with --print-scores to output parsed frame indices.
    # Let's inspect stdout of the CLI with --print-scores and --metrics dog_laplacian.
    cmd = [
        sys.executable, "sharpness_cli.py",
        "--dir", str(scan_dir),
        "--metrics", "dog_laplacian",
        "--print-scores"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    # Check for expected output patterns
    stdout = res.stdout
    print("CLI STDOUT:\n", stdout)

    # Verify that the parsed index matches the robust behavior in actual CLI execution
    for name, expected_idx in test_files.items():
        expected_line = f"Frame {expected_idx} (dog_laplacian)"
        assert expected_line in stdout


# -----------------------------------------------------------------------------
# 5. Performance Benchmarking & Stress Tests
# -----------------------------------------------------------------------------

def test_performance_benchmarking(tmp_path):
    """
    1. Performance benchmarking on large directories of simulated TIFF files.
    Generates 50 simulated spectroscopic frame TIFF files with a peak-decay structure,
    writes a corresponding ground_truth.json, runs the sharpness metrics, and checks speed.
    """
    scan_dir = tmp_path / "large_scan"
    scan_dir.mkdir()
    
    num_frames = 50
    height, width = 256, 256
    best_frame = 25
    
    # Generate 50 simulated frames
    for idx in range(num_frames):
        # Base background noise
        data = np.random.normal(loc=100.0, scale=10.0, size=(height, width)).astype(np.float32)
        # Create a peak profile whose sharpness varies with distance to best_frame
        dist = abs(idx - best_frame)
        sigma = 1.0 + dist * 0.5
        y, x = np.ogrid[:height, :width]
        profile = 500.0 * np.exp(-((x - width//2)**2) / (2.0 * sigma**2))
        data += profile
        
        tifffile.imwrite(scan_dir / f"frame_{idx:03d}.tif", data)
        
    # Write ground_truth.json
    ranks = {str(idx): float(1.0 + abs(idx - best_frame)) for idx in range(num_frames)}
    gt_data = {
        "experiment_id": "large_scan",
        "best_frame_index": best_frame,
        "fractional_ranks": ranks
    }
    with open(scan_dir / "ground_truth.json", "w") as f:
        json.dump(gt_data, f)
        
    # Benchmark execution of evaluate_sharpness directly
    for metric in ["dog_laplacian", "directional_tenengrad", "fft_bandpass"]:
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
        
    # Benchmark CLI tool
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
    assert "dog_laplacian" in res.stdout
    assert "directional_tenengrad" in res.stdout
    assert "fft_bandpass" in res.stdout
    assert cli_duration < 10.0


def test_extreme_intensities_safe_1e12():
    """
    Verify correctness and stability of denoise_image and evaluate_sharpness
    under the specified extreme intensities of 1e12 and -1e12.
    These values should fit within float32 limits and run without NaNs/Infs.
    """
    shapes = [(100, 100), (50, 150)]
    values = [1e12, -1e12, 1e-12, -1e-12]
    
    for shape in shapes:
        for val in values:
            img_flat = np.full(shape, val, dtype=np.float64)
            
            # Check denoise_image survives with clip=True
            denoised = denoise_image(img_flat, clip=True)
            assert not np.isnan(denoised).any()
            assert not np.isinf(denoised).any()
            
            # Check denoise_image survives with clip=False
            denoised_no_clip = denoise_image(img_flat, clip=False)
            assert not np.isnan(denoised_no_clip).any()
            assert not np.isinf(denoised_no_clip).any()
            
            # Check evaluate_sharpness survives
            for metric in ["dog_laplacian", "directional_tenengrad", "fft_bandpass"]:
                score = evaluate_sharpness(img_flat, metric)
                assert isinstance(score, (float, int, np.floating))
                assert not np.isnan(score)
                assert not np.isinf(score)


def test_extreme_intensities_overflow_bug():
    """
    Stress test denoise_image under values exceeding float32 max limit (e.g., 1e150, 1e300).
    This highlights a bug where the cast to float32 happens AFTER np.nan_to_num,
    resulting in unhandled Infinities and subsequent NaN propagation.
    We assert that the returned array is not corrupted (which will fail under the current implementation).
    """
    img_flat = np.full((100, 100), 1e150, dtype=np.float64)
    
    # Under current code, casting to float32 overflows to positive infinity.
    # The subsequent MAD filter and bilateral filter corrupt the image to NaNs.
    # We assert that the output contains no NaNs or Infs (this will fail, demonstrating the vulnerability).
    try:
        denoised = denoise_image(img_flat, clip=True)
        assert not np.isnan(denoised).any(), "Image was corrupted with NaNs during float32 cast overflow"
        assert not np.isinf(denoised).any(), "Image contains unhandled Infinities"
    except AssertionError as e:
        print(f"Discovered vulnerability: {e}")
        raise


def test_fft_bandpass_nan_overflow():
    """
    Stress test evaluate_sharpness(fft_bandpass) under values that cause float64 overflow (e.g., 1e200).
    This highlights a bug where the FFT power spectrum overflows to Inf, leading to inf/inf division,
    which results in a NaN sharpness score instead of a finite float.
    We assert that the score is a finite number (this will fail, demonstrating the vulnerability).
    """
    img_flat = np.full((100, 100), 1e200, dtype=np.float64)
    
    # We evaluate sharpness using fft_bandpass.
    # It should return a finite number or 0.0, but currently returns NaN.
    score = evaluate_sharpness(img_flat, "fft_bandpass")
    assert not np.isnan(score), "FFT Bandpass metric returned NaN due to float64 overflow in power spectrum"


def test_directional_tenengrad_overflow():
    """
    Stress test evaluate_sharpness(directional_tenengrad) under values that cause float64 overflow (e.g., 1e200).
    We assert that the score remains finite instead of overflowing to infinity.
    """
    img = np.zeros((100, 100), dtype=np.float64)
    img[50:, :] = 1e200
    
    score = evaluate_sharpness(img, "directional_tenengrad")
    assert np.isfinite(score), "Directional Tenengrad metric returned non-finite score due to float64 overflow in gradient squaring"


def test_dog_laplacian_overflow():
    """
    Stress test evaluate_sharpness(dog_laplacian) under values that cause float64 overflow (e.g., 1e200).
    We assert that the score remains finite instead of overflowing to infinity.
    """
    img = np.zeros((100, 100), dtype=np.float64)
    img[50:, :] = 1e200
    
    score = evaluate_sharpness(img, "dog_laplacian")
    assert np.isfinite(score), "DoG Laplacian metric returned non-finite score due to float64 overflow in variance calculation"


def test_non_square_aspect_ratio_compatibility():
    """
    3. Non-square aspect ratio compatibility (e.g. 1000x10, 10x1000, 100x2, 2x100).
    """
    aspect_ratios = [
        (1000, 10),
        (10, 1000),
        (100, 20),
        (20, 100),
        (50, 2),
        (2, 50)
    ]
    
    for H, W in aspect_ratios:
        # Create a non-square test image with some structure
        img = np.zeros((H, W), dtype=np.float32)
        for i in range(min(H, W)):
            h_start = i * (H // min(H, W))
            h_end = min((i + 1) * (H // min(H, W)), H)
            w_start = i * (W // min(H, W))
            w_end = min((i + 1) * (W // min(H, W)), W)
            img[h_start:h_end, w_start:w_end] = 100.0
            
        # Verify denoise_image handles it
        try:
            denoised = denoise_image(img)
            assert denoised.shape == (H, W)
        except Exception as e:
            pytest.fail(f"denoise_image failed on non-square shape ({H}, {W}): {e}")
            
        # Verify evaluate_sharpness handles it for all metrics
        for metric in ["dog_laplacian", "directional_tenengrad", "fft_bandpass"]:
            try:
                score = evaluate_sharpness(img, metric)
                assert isinstance(score, (float, int, np.floating))
                assert not np.isnan(score)
                assert not np.isinf(score)
            except Exception as e:
                pytest.fail(f"evaluate_sharpness({metric}) failed on non-square shape ({H}, {W}): {e}")
