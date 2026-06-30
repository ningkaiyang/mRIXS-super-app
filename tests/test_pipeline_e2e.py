import os
import sys
import subprocess
import json
import numpy as np
import pytest
import tifffile

# ==========================================
# Feature 4: Core Backend Integration (rixs_app/core/sharpness.py)
# ==========================================

# --- Tier 1: Feature 4 Coverage ---

def test_denoise_image_signature():
    """Verify that denoise_image signature is correct: accepts np.ndarray and returns np.ndarray."""
    from rixs_app.core.sharpness import denoise_image
    
    img = np.ones((50, 50), dtype=np.float32)
    out = denoise_image(img)
    
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape

def test_evaluate_sharpness_signature():
    """Verify that evaluate_sharpness signature is correct: accepts np.ndarray and metric string, returns float."""
    from rixs_app.core.sharpness import evaluate_sharpness
    
    img = np.ones((50, 50), dtype=np.float32)
    score = evaluate_sharpness(img, "norm_sum_sq_grad")
    
    assert isinstance(score, float) or isinstance(score, np.floating)

def test_winning_metric_integration():
    """Verify that the 1D sharpness metrics are integrated and callable."""
    from rixs_app.core.sharpness import evaluate_sharpness
    
    img = np.random.rand(50, 50).astype(np.float32)
    score_grad = evaluate_sharpness(img, "norm_sum_sq_grad")
    score_peak = evaluate_sharpness(img, "peak_height")
    
    assert score_grad is not None
    assert score_peak is not None

def test_preprocessing_backend_integration():
    """Verify that the full denoising pipeline is properly executed within the backend module."""
    from rixs_app.core.sharpness import denoise_image
    
    img = np.array([[-10, 20], [30, -40]], dtype=np.float32)
    out = denoise_image(img)
    # Check that it executed clipping and other preprocessing operations
    assert np.all(out >= 0)

def test_invalid_input_exception_propagation():
    """Verify that invalid inputs to backend functions propagate appropriate errors (e.g. ValueError)."""
    from rixs_app.core.sharpness import denoise_image, evaluate_sharpness
    
    # 1D array should not be allowed
    invalid_img = np.array([1, 2, 3])
    
    with pytest.raises(ValueError):
        denoise_image(invalid_img)
        
    with pytest.raises(ValueError):
        evaluate_sharpness(invalid_img, "norm_sum_sq_grad")


# --- Tier 2: Feature 4 Coverage ---

def test_extreme_intensity_overflow():
    """Boundary Case: Test that backend methods handle extreme intensities without integer/float overflow."""
    from rixs_app.core.sharpness import denoise_image, evaluate_sharpness
    
    img = np.ones((50, 50), dtype=np.float32) * 1e9
    out = denoise_image(img)
    score = evaluate_sharpness(img, "norm_sum_sq_grad")
    
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert not np.isnan(score)
    assert not np.isinf(score)

def test_non_square_aspect_ratio():
    """Boundary Case: Test that non-square arrays (e.g. 100x50 or 50x200) are handled correctly."""
    from rixs_app.core.sharpness import denoise_image, evaluate_sharpness
    
    img = np.ones((100, 50), dtype=np.float32)
    out = denoise_image(img)
    score = evaluate_sharpness(img, "norm_sum_sq_grad")
    
    assert out.shape == (100, 50)
    assert score is not None

def test_custom_datatype_compatibility():
    """Boundary Case: Test that functions are compatible with various numeric data types (float32, float64, int16, int32)."""
    from rixs_app.core.sharpness import denoise_image, evaluate_sharpness
    
    for dtype in [np.float32, np.float64, np.int16, np.int32]:
        img = (np.random.rand(50, 50) * 100).astype(dtype)
        out = denoise_image(img)
        score = evaluate_sharpness(img, "norm_sum_sq_grad")
        
        assert out is not None
        assert score is not None

def test_zero_empty_image_input():
    """Boundary Case: Empty image array or 0-dimensional image array raises a ValueError."""
    from rixs_app.core.sharpness import denoise_image, evaluate_sharpness
    
    empty_img = np.zeros((0, 0), dtype=np.float32)
    
    with pytest.raises(ValueError):
        denoise_image(empty_img)
        
    with pytest.raises(ValueError):
        evaluate_sharpness(empty_img, "norm_sum_sq_grad")




# --- Tier 3: Cross-Feature Interactions ---

def test_f1_f3_cli_cli_pipe(tmp_path):
    """Tier 3: Feature 1 (Denoise CLI) output fed directly into Feature 3 (Sharpness CLI)."""
    input_path = tmp_path / "frame_000.tif"
    data = np.random.poisson(lam=10.0, size=(50, 50))
    tifffile.imwrite(input_path, data.astype(np.int32))
    
    # Step 1: Preprocess using CLI
    cmd_denoise = [sys.executable, "denoise_cli.py", "--input", str(input_path), "--output", str(tmp_path / "denoised" / "frame_000_denoised.tiff"), "--clip"]
    subprocess.run(cmd_denoise, capture_output=True, text=True, check=True)
    
    # Step 2: Write a dummy ground truth JSON to allow sharpness correlation calculation
    gt_data = {
        "experiment_id": "test_pipe",
        "best_frame_index": 0,
        "fractional_ranks": {"0": 1.0}
    }
    with open(tmp_path / "denoised" / "ground_truth.json", "w") as f:
        json.dump(gt_data, f)
        
    # Step 3: Run Sharpness CLI on the denoised directory
    cmd_sharpness = [sys.executable, "sharpness_cli.py", "--dir", str(tmp_path / "denoised")]
    res = subprocess.run(cmd_sharpness, capture_output=True, text=True, check=True)
    
    assert res.returncode == 0
    assert "Directory" in res.stdout



def test_f1_f4_cli_backend_comparison(tmp_path):
    """Tier 3: Feature 1 (Denoise CLI) preprocessing output matches Backend (denoise_image) output."""
    from rixs_app.core.sharpness import denoise_image
    
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
    """Tier 3: Feature 3 (Sharpness CLI) computed metrics match Backend (evaluate_sharpness) computed metrics."""
    from rixs_app.core.sharpness import evaluate_sharpness
    
    # Write a frame
    input_path = tmp_path / "frame_000.tif"
    data = np.random.rand(50, 50).astype(np.float32)
    tifffile.imwrite(input_path, data)
    
    # Run sharpness CLI and capture output
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(tmp_path), "--metrics", "norm_sum_sq_grad", "--print-scores"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    backend_score = evaluate_sharpness(data, "norm_sum_sq_grad")
    
    # Parse CLI score from output and verify it is close to backend score
    assert str(round(backend_score, 2)) in res.stdout


def test_evaluate_sharpness_nan_inf_handling():
    """Verify that evaluate_sharpness handles NaN and Inf values gracefully without throwing errors."""
    from rixs_app.core.sharpness import evaluate_sharpness
    
    img = np.ones((50, 50), dtype=np.float32)
    img[0, 0] = np.nan
    img[1, 1] = np.inf
    img[2, 2] = -np.inf
    
    for metric in ["norm_sum_sq_grad", "peak_height"]:
        score = evaluate_sharpness(img, metric)
        assert score is not None
        assert not np.isnan(score)
        assert not np.isinf(score)
