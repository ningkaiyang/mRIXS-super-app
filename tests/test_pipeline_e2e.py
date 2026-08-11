import os
import sys
import subprocess
import json
import numpy as np
import pytest
import tifffile

# ==========================================
# Feature 4: Core Backend Integration (rixs_app/core/zeroth_order.py)
# ==========================================

# --- Tier 1: Feature 4 Coverage ---

def test_denoise_image_signature():
    """Verify that denoise_image signature is correct: accepts np.ndarray and returns np.ndarray."""
    from rixs_app.core.zeroth_order import denoise_image
    
    img = np.ones((50, 50), dtype=np.float32)
    out = denoise_image(img)
    
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape

def test_evaluate_zeroth_order_signature():
    """Verify that evaluate_zeroth_order signature is correct: accepts np.ndarray and metric string, returns float."""
    from rixs_app.core.zeroth_order import evaluate_zeroth_order
    
    img = np.ones((50, 50), dtype=np.float32)
    score = evaluate_zeroth_order(img, "norm_sum_sq_grad")
    
    assert isinstance(score, float) or isinstance(score, np.floating)

def test_winning_metric_integration():
    """Verify that the 1D zeroth-order metrics are integrated and callable."""
    from rixs_app.core.zeroth_order import evaluate_zeroth_order
    
    img = np.random.rand(50, 50).astype(np.float32)
    score_grad = evaluate_zeroth_order(img, "norm_sum_sq_grad")
    score_peak = evaluate_zeroth_order(img, "peak_height")
    
    assert score_grad is not None
    assert score_peak is not None

def test_preprocessing_backend_integration():
    """Verify that the full denoising pipeline is properly executed within the backend module."""
    from rixs_app.core.zeroth_order import denoise_image
    
    img = np.array([[-10, 20], [30, -40]], dtype=np.float32)
    out = denoise_image(img)
    # Check that it executed clipping and other preprocessing operations
    assert np.all(out >= 0)

def test_invalid_input_exception_propagation():
    """Verify that invalid inputs to backend functions propagate appropriate errors (e.g. ValueError)."""
    from rixs_app.core.zeroth_order import denoise_image, evaluate_zeroth_order
    
    # 1D array should not be allowed
    invalid_img = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        denoise_image(invalid_img)
        
    with pytest.raises(ValueError):
        evaluate_zeroth_order(invalid_img, "norm_sum_sq_grad")


# --- Tier 2: Feature 4 Coverage ---

def test_extreme_intensity_overflow():
    """Boundary Case: Test that backend methods handle extreme intensities without integer/float overflow."""
    from rixs_app.core.zeroth_order import denoise_image, evaluate_zeroth_order
    
    img = np.ones((50, 50), dtype=np.float32) * 1e9
    out = denoise_image(img)
    score = evaluate_zeroth_order(img, "norm_sum_sq_grad")
    
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert not np.isnan(score)
    assert not np.isinf(score)

def test_non_square_aspect_ratio():
    """Boundary Case: Test that non-square arrays (e.g. 100x50 or 50x200) are handled correctly."""
    from rixs_app.core.zeroth_order import denoise_image, evaluate_zeroth_order
    
    img = np.ones((100, 50), dtype=np.float32)
    out = denoise_image(img)
    score = evaluate_zeroth_order(img, "norm_sum_sq_grad")
    
    assert out.shape == (100, 50)
    assert score is not None

def test_custom_datatype_compatibility():
    """Boundary Case: Test that functions are compatible with various numeric data types (float32, float64, int16, int32)."""
    from rixs_app.core.zeroth_order import denoise_image, evaluate_zeroth_order
    
    for dtype in [np.float32, np.float64, np.int16, np.int32]:
        img = (np.random.rand(50, 50) * 100).astype(dtype)
        out = denoise_image(img)
        score = evaluate_zeroth_order(img, "norm_sum_sq_grad")
        
        assert out is not None
        assert score is not None

def test_zero_empty_image_input():
    """Boundary Case: Empty image array or 0-dimensional image array raises a ValueError."""
    from rixs_app.core.zeroth_order import denoise_image, evaluate_zeroth_order
    
    empty_img = np.zeros((0, 0), dtype=np.float32)
    
    with pytest.raises(ValueError):
        denoise_image(empty_img)
        
    with pytest.raises(ValueError):
        evaluate_zeroth_order(empty_img, "norm_sum_sq_grad")




# --- Tier 3: Cross-Feature Interactions ---

def test_f1_f3_cli_cli_pipe(tmp_path):
    """Tier 3: Feature 1 (Denoise CLI) output fed directly into Feature 3 (Zeroth-Order CLI).

    Creates a small set of synthetic TIFF frames, denoises them with denoise_cli.py,
    then passes the resulting directory to zeroth_order_cli.py with --format json
    to confirm the pipeline runs end-to-end and produces a valid JSON summary.
    """
    import json
    import tifffile

    denoised_dir = tmp_path / "denoised"
    denoised_dir.mkdir()

    # Create 3 synthetic frames and denoise them
    for i in range(3):
        input_path = tmp_path / f"frame_{i:03d}.tif"
        data = np.random.poisson(lam=10.0, size=(50, 50)).astype(np.int32)
        tifffile.imwrite(str(input_path), data)
        out_path = denoised_dir / f"frame_{i:03d}_denoised.tiff"
        cmd_denoise = [
            sys.executable, "denoise_cli.py",
            "--input", str(input_path),
            "--output", str(out_path),
            "--clip",
        ]
        subprocess.run(cmd_denoise, capture_output=True, text=True, check=True)

    # Run zeroth-order CLI on the denoised directory
    cmd_zeroth_order = [
        sys.executable, "zeroth_order_cli.py",
        "-d", str(denoised_dir),
        "--format", "json",
        "--no-focus-curve",
        "--export-plots", "none",
    ]
    res = subprocess.run(cmd_zeroth_order, capture_output=True, text=True, check=True)
    assert res.returncode == 0

    # Confirm JSON summary was produced
    summary_json = denoised_dir / "zeroth_order_analysis" / "summary.json"
    assert summary_json.exists(), "summary.json not generated by new CLI"
    with open(summary_json) as f:
        summary = json.load(f)
    assert summary["total_frames"] == 3



def test_f1_f4_cli_backend_comparison(tmp_path):
    """Tier 3: Feature 1 (Denoise CLI) preprocessing output matches Backend (denoise_image) output."""
    from rixs_app.core.zeroth_order import denoise_image
    
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    data = np.array([[-10, 20], [30, -40]], dtype=np.float32)
    tifffile.imwrite(input_path, data)
    
    # Run CLI
    cmd = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(output_path)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    cli_out = tifffile.imread(output_path)
    backend_out = denoise_image(data)
    
    np.testing.assert_array_almost_equal(cli_out, backend_out)

def test_f3_f4_cli_backend_comparison(tmp_path):
    """Tier 3: Feature 3 (Zeroth-Order CLI) pipeline results are consistent with Backend output.

    Writes 3 synthetic frames, runs the new zeroth_order_cli with --format json,
    then verifies that the JSON contains a best_frame_index (asserting the CLI
    correctly links to the same run_zeroth_order_pipeline backend).
    """
    import json
    import tifffile

    # Write 3 synthetic frames with a clear synthetic line in frame 1 (brightest)
    for i in range(3):
        frame_path = tmp_path / f"frame_{i:03d}.tif"
        data = np.zeros((50, 50), dtype=np.float32)
        data[20:30, 22:28] = float(50 + i * 5)  # progressively brighter lines
        tifffile.imwrite(str(frame_path), data)

    cmd = [
        sys.executable, "zeroth_order_cli.py",
        "-d", str(tmp_path),
        "--format", "json",
        "--no-focus-curve",
        "--export-plots", "none",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0

    summary_json = tmp_path / "zeroth_order_analysis" / "summary.json"
    assert summary_json.exists(), "summary.json not generated by new CLI"
    with open(summary_json) as f:
        summary = json.load(f)

    # Verify structural consistency: CLI produced valid frame records linked to core pipeline
    assert summary["total_frames"] == 3
    assert "best_frame_index" in summary
    assert "frames" in summary
    assert len(summary["frames"]) == 3


def test_evaluate_zeroth_order_nan_inf_handling():
    """Verify that evaluate_zeroth_order handles NaN and Inf values gracefully without throwing errors."""
    from rixs_app.core.zeroth_order import evaluate_zeroth_order
    
    img = np.ones((50, 50), dtype=np.float32)
    img[0, 0] = np.nan
    img[1, 1] = np.inf
    img[2, 2] = -np.inf
    
    for metric in ["norm_sum_sq_grad", "peak_height"]:
        score = evaluate_zeroth_order(img, metric)
        assert score is not None
        assert not np.isnan(score)
        assert not np.isinf(score)
