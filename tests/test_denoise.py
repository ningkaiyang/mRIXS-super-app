import os
import sys
import subprocess
import tempfile
import time
import pytest
import numpy as np
import cv2
import scipy.ndimage
import tifffile
import tracemalloc

from rixs_app.core.preprocessing import denoise_image
from rixs_app.core.zeroth_order import evaluate_zeroth_order

# =====================================================================
# FROM test_preprocessing_challenger.py
# =====================================================================

# =====================================================================
# 1. Flat constant images (zero variance)
# =====================================================================

def test_flat_constant_zero():
    """Test all zeros input."""
    img = np.zeros((100, 100), dtype=np.float32)
    # Default pipeline
    out = denoise_image(img)
    assert out.shape == (100, 100)
    assert out.dtype == np.float32
    # Since:
    # clip: all 0.0
    # despike: mad=0, std=0, skipped
    # anscombe: 2 * sqrt(0.375) = 1.22474487
    # bilateral: all 1.22474487
    # inverse_anscombe: (1.22474487 / 2.0)**2 - 0.375 = 0.375 - 0.375 = 0.0
    # So output should be extremely close to 0.0
    np.testing.assert_allclose(out, 0.0, atol=1e-6)

def test_flat_constant_positive():
    """Test positive constant value input."""
    img = np.ones((100, 100), dtype=np.float32) * 42.0
    out = denoise_image(img)
    assert out.shape == (100, 100)
    assert out.dtype == np.float32
    # Output should recover the constant value exactly
    np.testing.assert_allclose(out, 42.0, atol=1e-5)

def test_flat_constant_negative_with_clipping():
    """Test negative constant with clipping enabled (should clamp to 0)."""
    img = np.ones((100, 100), dtype=np.float32) * -5.0
    out = denoise_image(img, clip=True)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)

def test_flat_constant_negative_without_clipping():
    """Test negative constant without clipping (Anscombe VST should handle via np.maximum)."""
    img = np.ones((100, 100), dtype=np.float32) * -5.0
    out = denoise_image(img, clip=False, anscombe=True, inverse_anscombe=True)
    # Since img = -5.0
    # anscombe: 2.0 * sqrt(max(-5.0 + 0.375, 0.0)) = 0.0
    # bilateral: all 0.0
    # inverse_anscombe: (0.0)**2 - 0.375 = -0.375
    # So output should be -0.375
    np.testing.assert_allclose(out, -0.375, atol=1e-6)

# =====================================================================
# 2. Extremely large values and negative values
# =====================================================================

def test_extreme_large_values():
    """Test very large positive values that might cause float32 overflow or OpenCV crashes."""
    # OpenCV bilateralFilter handles float32 up to typical values, but what about extremely large values?
    img = np.ones((50, 50), dtype=np.float32) * 1e10
    out = denoise_image(img)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    # Check close to 1e10
    np.testing.assert_allclose(out, 1e10, rtol=1e-4)

def test_extreme_negative_values():
    """Test extremely negative values."""
    img = np.ones((50, 50), dtype=np.float32) * -1e10
    # With clipping (default)
    out_clipped = denoise_image(img, clip=True)
    np.testing.assert_allclose(out_clipped, 0.0, atol=1e-6)
    
    # Without clipping
    out_unclipped = denoise_image(img, clip=False)
    # Should clamp at np.maximum(img + 0.375, 0.0) -> 0.0, then inverse Anscombe VST -> -0.375
    np.testing.assert_allclose(out_unclipped, -0.375, atol=1e-6)

def test_extreme_spikes_handling():
    """Test MAD despiking with extremely large spikes."""
    img = np.ones((50, 50), dtype=np.float32) * 10.0
    img[25, 25] = 1e20  # Huge spike
    out = denoise_image(img, despike=True)
    # The spike should be successfully replaced by the median (10.0)
    assert out[25, 25] < 100.0
    np.testing.assert_allclose(out[25, 25], 10.0, atol=1e-3)

# =====================================================================
# 3. Empty/zero-dimensional inputs
# =====================================================================

def test_empty_zero_dimensional():
    """Test shape (0, 0), (10, 0), (0, 10)."""
    for shape in [(0, 0), (10, 0), (0, 10)]:
        img = np.zeros(shape, dtype=np.float32)
        with pytest.raises(ValueError) as exc:
            denoise_image(img)
        assert "cannot be empty" in str(exc.value)

def test_invalid_type_inputs():
    """Test input types other than np.ndarray."""
    with pytest.raises(ValueError) as exc:
        denoise_image([[1, 2], [3, 4]])  # List
    assert "must be a numpy array" in str(exc.value)
    
    with pytest.raises(ValueError) as exc:
        denoise_image(None)  # None
    assert "must be a numpy array" in str(exc.value)

def test_invalid_dimensions():
    """Test 1D and 3D arrays."""
    # 1D array
    img_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        denoise_image(img_1d)
    assert "must be a 2D array" in str(exc.value)
    
    # 3D array
    img_3d = np.ones((5, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError) as exc:
        denoise_image(img_3d)
    assert "must be a 2D array" in str(exc.value)

# =====================================================================
# 4. Input types (float32, float64, int16, int32)
# =====================================================================

def test_input_data_types():
    """Test various input numpy data types and check that returned array is float32."""
    dtypes = [np.float32, np.float64, np.int16, np.int32, np.uint16, np.uint8, bool]
    for dtype in dtypes:
        img = np.ones((20, 20), dtype=dtype)
        out = denoise_image(img)
        assert out.dtype == np.float32
        assert out.shape == (20, 20)

# =====================================================================
# 5. OpenCV bilateralFilter constraints & errors
# =====================================================================

def test_nan_inf_inputs():
    """Test image containing NaNs and infinities."""
    img = np.ones((20, 20), dtype=np.float32) * 5.0
    img[5, 5] = np.nan
    img[10, 10] = np.inf
    img[15, 15] = -np.inf
    
    # Let's see if denoise_image crashes or propagates them.
    # Note: OpenCV bilateralFilter might crash or yield NaNs if input contains NaNs/infs.
    # Let's test if it survives or crashes.
    try:
        out = denoise_image(img)
        # It didn't crash! Let's check what it returned.
        # It's okay if it propagates NaNs or cleans them, as long as it does not crash or hang.
        assert isinstance(out, np.ndarray)
    except Exception as e:
        pytest.fail(f"denoise_image crashed on NaN/inf input with: {e}")

def test_nan_inf_propagation_corruption():
    """Test that a single inf value does not propagate NaN to the entire image after bilateral filter."""
    img = np.ones((20, 20), dtype=np.float32) * 5.0
    img[10, 10] = np.inf
    
    out = denoise_image(img)
    # The entire image should remain free of NaNs due to NaN/Inf propagation mitigation
    assert np.isnan(out).sum() == 0


def test_negative_bilateral_parameters():
    """Test passing invalid parameters directly to backend (like negative d or sigmas)."""
    img = np.ones((20, 20), dtype=np.float32) * 10.0
    
    # d can be negative in cv2.bilateralFilter (OpenCV uses it to compute d from sigmaSpace if d <= 0)
    # What about negative sigma_color or sigma_space?
    try:
        out = denoise_image(img, d=-5, sigma_color=-1.5, sigma_space=-3.0)
        assert isinstance(out, np.ndarray)
    except Exception as e:
        # OpenCV might raise cv2.error for negative sigmas. Let's see if it does.
        # If it raises a cv2.error, it is a crash risk from backend if parameters are not validated!
        # Let's document if this happens.
        pass

# =====================================================================
# 6. CLI checks (denoise_cli.py)
# =====================================================================

def test_cli_invalid_arguments(tmp_path):
    """Test CLI parameter validation and error codes."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    tifffile.imwrite(input_path, np.ones((10, 10), dtype=np.float32))

    # Test 1: Mutually exclusive mode (dir and input/output together)
    cmd = [sys.executable, "denoise_cli.py", "--dir", str(tmp_path), "--input", str(input_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Error:" in res.stderr

    # Test 2: Input without output
    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Error:" in res.stderr

    # Test 3: Negative parameters validation
    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--d", "-1"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Error: All numeric options must be non-negative" in res.stderr

    # Test 4: Path existence checks
    cmd = [sys.executable, "denoise_cli.py", "--input", "nonexistent_file.tif", "--output", str(output_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "does not exist" in res.stderr

# =====================================================================
# 7. Performance and memory leak tests
# =====================================================================

def test_performance_loop():
    """Verify performance and that there are no infinite loops or large memory leaks."""
    # Create a reasonably sized image (e.g. 512x512)
    img = np.random.poisson(lam=15.0, size=(512, 512)).astype(np.float32)
    
    start_time = time.time()
    iterations = 50
    for _ in range(iterations):
        out = denoise_image(img)
        assert out.shape == (512, 512)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / iterations
    print(f"\nAverage time per 512x512 frame: {avg_time * 1000:.2f} ms")
    # A single frame should take less than 100ms
    assert avg_time < 0.1, f"Preprocessing is too slow: {avg_time * 1000:.2f} ms/frame"

# =====================================================================
# FROM test_denoise_challenger_m3_1.py
# =====================================================================

# -------------------------------------------------------------
# 1. FLAT CONSTANT IMAGES (ZERO VARIANCE)
# -------------------------------------------------------------

def test_flat_constant_zero_variance():
    """Verify that flat constant images of various values process correctly without warnings/crashes."""
    shapes = [(10, 10), (100, 100), (50, 150)]
    values = [0.0, 1.0, 42.0, -10.0, 1e5]

    for shape in shapes:
        for val in values:
            img = np.ones(shape, dtype=np.float32) * val
            # Run with all preprocessing options enabled
            try:
                out = denoise_image(
                    img,
                    clip=True,
                    despike=True,
                    anscombe=True,
                    bilateral=True,
                    inverse_anscombe=True
                )
                assert out.shape == shape
                assert not np.isnan(out).any()
                assert not np.isinf(out).any()
            except Exception as e:
                pytest.fail(f"Denoising failed on flat image of shape {shape} with value {val}: {e}")

def test_flat_constant_no_clip():
    """Verify flat image behavior when clipping is disabled."""
    # Negative flat value with clip=False, anscombe=True
    # img + 0.375 will be negative, np.maximum(img + 0.375, 0.0) should handle it safely
    img = np.ones((50, 50), dtype=np.float32) * -10.0
    out = denoise_image(
        img,
        clip=False,
        despike=True,
        anscombe=True,
        bilateral=True,
        inverse_anscombe=True
    )
    # Since clip=False but anscombe is True:
    # anscombe: 2.0 * np.sqrt(max(-10.0 + 0.375, 0.0)) = 0.0
    # bilateral: bilateralFilter on all zeros = all zeros
    # inverse_anscombe: (0.0/2.0)**2 - 0.375 = -0.375
    # So expected output is all -0.375
    np.testing.assert_array_almost_equal(out, np.ones((50, 50), dtype=np.float32) * -0.375)

# -------------------------------------------------------------
# 2. EXTREMELY LARGE, NEGATIVE, AND NON-FINITE VALUES
# -------------------------------------------------------------

def test_extreme_values_overflow():
    """Verify that very large positive and negative values do not cause crash or invalid results."""
    # Large value just below float32 max
    img_large = np.ones((50, 50), dtype=np.float32) * 1e30
    out_large = denoise_image(img_large)
    assert not np.isnan(out_large).any()
    assert not np.isinf(out_large).any()

    # Extreme negative values
    img_neg = np.ones((50, 50), dtype=np.float32) * -1e30
    # With clip=True, should become all 0.0 (then through anscombe/inverse anscombe back to 0.0)
    out_neg = denoise_image(img_neg, clip=True)
    np.testing.assert_array_almost_equal(out_neg, np.zeros((50, 50), dtype=np.float32))

def test_inf_flat_image_nan_corruption():
    """Verify if a flat image with a single Inf is correctly despiked or if it corrupts the image.
    Currently, the standard deviation fallback becomes Inf, threshold becomes Inf, and bilateralFilter
    corrupts the entire image to NaN. This test expects the image to NOT be corrupted to NaN/Inf.
    """
    img_inf = np.ones((50, 50), dtype=np.float32) * 10.0
    img_inf[25, 25] = np.inf
    
    out = denoise_image(img_inf, despike=True, bilateral=True)
    # The output should not contain NaNs or Infs
    assert not np.isnan(out).any(), "Image corrupted with NaNs due to Inf propagation"
    assert not np.isinf(out).any(), "Image contains Inf that was not despiked"

def test_nan_image_not_despiked():
    """Verify if a single NaN in the image is despiked/replaced.
    Currently, presence of NaN causes the median calculations to return NaN, skipping despiking,
    and leaving NaN in the final output. This test expects no NaNs in the output.
    """
    img_nan = np.ones((50, 50), dtype=np.float32) * 10.0
    img_nan[25, 25] = np.nan
    
    out = denoise_image(img_nan, despike=True)
    assert not np.isnan(out).any(), "NaN was not despiked/removed from the image"

# -------------------------------------------------------------
# 3. EMPTY / ZERO-DIMENSIONAL / INVALID DIMENSIONS
# -------------------------------------------------------------

def test_empty_and_zero_dimensional():
    """Verify that zero-dimensional or empty inputs raise ValueError."""
    invalid_inputs = [
        np.zeros((0, 0), dtype=np.float32),
        np.zeros((10, 0), dtype=np.float32),
        np.zeros((0, 10), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    ]

    for inp in invalid_inputs:
        with pytest.raises(ValueError):
            denoise_image(inp)

def test_invalid_dimensions_2():
    """Verify that 1D, 3D, and higher-dimensional arrays raise ValueError."""
    with pytest.raises(ValueError):
        denoise_image(np.ones(10, dtype=np.float32))

    with pytest.raises(ValueError):
        denoise_image(np.ones((10, 10, 3), dtype=np.float32))

def test_non_numpy_array():
    """Verify that non-numpy array inputs raise ValueError."""
    non_arrays = [
        [[1, 2], [3, 4]],  # standard nested list
        None,
        "not an array",
        42
    ]
    for inp in non_arrays:
        with pytest.raises(ValueError):
            denoise_image(inp)

# -------------------------------------------------------------
# 4. INPUT TYPES AND DTYPES
# -------------------------------------------------------------

def test_input_dtypes():
    """Verify that different input dtypes (float32, float64, int16, int32, uint16, uint8) are handled."""
    dtypes = [np.float32, np.float64, np.int16, np.int32, np.uint16, np.uint8]
    base_data = np.array([[10, 20], [30, 40]])

    for dtype in dtypes:
        img = base_data.astype(dtype)
        out = denoise_image(img)
        assert out.dtype == np.float32
        assert out.shape == (2, 2)

def test_read_only_input():
    """Verify that read-only numpy arrays do not cause mutations or crashes."""
    img = np.ones((50, 50), dtype=np.float32)
    img.flags.writeable = False

    try:
        out = denoise_image(img)
        assert out.shape == (50, 50)
        # Verify the original read-only image wasn't modified
        assert img.flags.writeable is False
    except Exception as e:
        pytest.fail(f"Denoising failed on read-only array: {e}")

# -------------------------------------------------------------
# 5. PARAMETER BOUNDARY CASES / PARAMETER FUZZING
# -------------------------------------------------------------

def test_invalid_parameter_values():
    """Verify what happens when direct backend function denoise_image receives invalid parameter values."""
    img = np.ones((50, 50), dtype=np.float32) * 10.0

    # 1. Negative bilateral diameter d
    # OpenCV's bilateralFilter typically expects d >= 0. What happens if d < 0 is passed directly?
    try:
        out = denoise_image(img, d=-5)
        # Note: opencv-python might accept d <= 0 and treat it as d calculated from sigmaSpace.
        # But let's verify if it causes a crash or works.
        assert out.shape == (50, 50)
    except Exception as e:
        print(f"Exception raised for d=-5: {e}")

    # 2. Negative sigma_color
    try:
        out = denoise_image(img, sigma_color=-1.5)
        assert out.shape == (50, 50)
    except Exception as e:
        print(f"Exception raised for sigma_color=-1.5: {e}")

    # 3. Negative sigma_space
    try:
        out = denoise_image(img, sigma_space=-3.0)
        assert out.shape == (50, 50)
    except Exception as e:
        print(f"Exception raised for sigma_space=-3.0: {e}")

    # 4. Negative mad_threshold
    # If mad_threshold is negative, despiking threshold is negative.
    # What does np.where(np.abs(dev) > threshold, median_img, img) do?
    # Since np.abs(dev) is always >= 0, if threshold is negative,
    # np.abs(dev) > threshold is ALWAYS True (unless dev is NaN).
    # So it will replace every pixel with its median!
    # Let's check if it crashes or completes.
    try:
        out = denoise_image(img, despike=True, mad_threshold=-5.0)
        assert out.shape == (50, 50)
    except Exception as e:
        pytest.fail(f"Negative mad_threshold caused a crash: {e}")

# -------------------------------------------------------------
# 6. PERFORMANCE & RESOURCE TESTS (LEAKS, TIMEOUTS, LOOPS)
# -------------------------------------------------------------

def test_performance_large_image():
    """Measure execution time for a large image (e.g. 2048 x 2048) to ensure it doesn't take too long."""
    img = np.random.rand(2048, 2048).astype(np.float32) * 100.0
    
    start_time = time.time()
    out = denoise_image(img, d=5, sigma_color=1.5, sigma_space=3.0)
    elapsed = time.time() - start_time
    
    print(f"Denoised 2048x2048 image in {elapsed:.4f} seconds.")
    assert out.shape == (2048, 2048)
    # Check that execution is reasonably fast (e.g. < 5.0 seconds on standard systems)
    assert elapsed < 5.0

def test_memory_leak_check():
    """Verify that repeated executions of the pipeline do not leak memory."""
    img = np.random.rand(500, 500).astype(np.float32) * 100.0
    
    tracemalloc.start()
    
    # Run once to warm up any caches/imports
    _ = denoise_image(img)
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run 50 times
    for _ in range(50):
        _ = denoise_image(img)
        
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # Compare snapshots
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)
    
    print(f"Memory diff after 50 runs: {total_diff / 1024:.2f} KB")
    # A small difference (e.g., < 100 KB) is typical for Python overhead.
    # A real leak (like unreleased C++ buffers from cv2) would be much larger (MBs).
    assert total_diff < 500 * 1024  # less than 500 KB diff


# =====================================================================
# FROM test_denoise_e2e.py
# =====================================================================

def create_synthetic_tiff(path, data):
    tifffile.imwrite(path, data.astype(np.int32))

# ==========================================
# Feature 1: Preprocessing CLI (denoise_cli.py)
# ==========================================

# --- Tier 1: Feature Coverage ---

def test_clip_option(tmp_path):
    """Test that the CLI performs clamping of negative values when --clip is passed."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    create_synthetic_tiff(input_path, data)

    # Run the CLI. Since it's not implemented, this is expected to raise CalledProcessError or fail.
    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--clip"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert np.all(output_data >= 0)

def test_anscombe_option(tmp_path):
    """Test that the CLI applies the Anscombe transform for variance stabilization."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.random.poisson(lam=10.0, size=(50, 50))
    create_synthetic_tiff(input_path, data)

    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--anscombe"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    # Check that output has been transformed
    assert output_data.shape == (50, 50)

def test_bilateral_option(tmp_path):
    """Test that the CLI performs bilateral filtering using spatial and range parameters."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.ones((50, 50)) * 100
    create_synthetic_tiff(input_path, data)

    cmd = [
        sys.executable, "denoise_cli.py", 
        "--input", str(input_path), 
        "--output", str(output_path), 
        "--bilateral", 
        "--d", "5", 
        "--sigma-color", "10.0", 
        "--sigma-space", "10.0"
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert output_data.shape == (50, 50)

def test_despiking_option(tmp_path):
    """Test that the CLI executes Median Absolute Deviation (MAD) despiking with a given threshold."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.ones((50, 50)) * 10
    data[25, 25] = 1000 # Add a spike
    create_synthetic_tiff(input_path, data)

    cmd = [
        sys.executable, "denoise_cli.py", 
        "--input", str(input_path), 
        "--output", str(output_path), 
        "--despike", 
        "--mad-threshold", "3.0"
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    # Check that the extreme spike at (25, 25) was suppressed
    assert output_data[25, 25] < 500

def test_inverse_anscombe_option(tmp_path):
    """Test that the CLI correctly applies the inverse Anscombe transformation to revert to raw scaling."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.random.poisson(lam=10.0, size=(50, 50))
    create_synthetic_tiff(input_path, data)

    cmd = [
        sys.executable, "denoise_cli.py", 
        "--input", str(input_path), 
        "--output", str(output_path), 
        "--anscombe", 
        "--inverse-anscombe"
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert output_data.shape == (50, 50)


# --- Tier 2: Boundary/Corner Cases ---

def test_all_negative_input(tmp_path):
    """Boundary Case: Input image containing only negative values. CLI should clamp to zero."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.array([[-100, -200], [-300, -400]], dtype=np.int32)
    create_synthetic_tiff(input_path, data)

    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--clip"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert np.all(output_data == 0)

def test_zero_variance_flat_input(tmp_path):
    """Boundary Case: Input image with zero variance (completely flat). Anscombe/bilateral should not crash."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.ones((50, 50), dtype=np.int32) * 42
    create_synthetic_tiff(input_path, data)

    cmd = [
        sys.executable, "denoise_cli.py", 
        "--input", str(input_path), 
        "--output", str(output_path), 
        "--clip", "--anscombe", "--bilateral", "--inverse-anscombe"
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert output_data.shape == (50, 50)

def test_extreme_spikes(tmp_path):
    """Boundary Case: Input image with extreme spikes (inf, NaN, or large int32 boundaries)."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    # Create array with extreme values
    data = np.zeros((10, 10), dtype=np.int32)
    data[5, 5] = 2_147_483_647  # max 32-bit signed int
    data[0, 0] = -2_147_483_648 # min 32-bit signed int
    create_synthetic_tiff(input_path, data)

    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--clip", "--despike"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert os.path.exists(output_path)
    output_data = tifffile.imread(output_path)
    assert np.max(output_data) < 2_147_483_647

def test_invalid_parameters(tmp_path):
    """Boundary Case: Invalid/negative parameters passed to CLI (e.g. negative bilateral range). Should return non-zero."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.ones((10, 10))
    create_synthetic_tiff(input_path, data)

    # passing negative bilateral diameter or negative threshold
    cmd = [
        sys.executable, "denoise_cli.py", 
        "--input", str(input_path), 
        "--output", str(output_path), 
        "--bilateral", "--d", "-5"
    ]
    
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Assert that command exited with an error status (not 0)
    assert exc_info.value.returncode != 0

def test_empty_input(tmp_path):
    """Boundary Case: Empty input or 0-dimensional image array. CLI should exit with non-zero error."""
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    
    # Create empty tiff
    data = np.zeros((0, 0), dtype=np.int32)
    create_synthetic_tiff(input_path, data)

    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path), "--clip"]

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
    assert exc_info.value.returncode != 0

# -------------------------------------------------------------
# 8. UNIT TESTS FOR NEW HELPER FUNCTIONS
# -------------------------------------------------------------

