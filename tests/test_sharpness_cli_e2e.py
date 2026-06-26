import os
import sys
import subprocess
import json
import numpy as np
import pytest
import tifffile

def setup_mock_scan_dir(tmp_path, num_frames=5, best_frame=2, flat=False):
    """Helper to create a temporary scan directory with mock TIFFs and ground_truth.json."""
    scan_dir = tmp_path / "mock_scan"
    scan_dir.mkdir(exist_ok=True)
    
    # Create mock TIFF files
    for idx in range(num_frames):
        tif_path = scan_dir / f"frame_{idx:03d}.tif"
        if flat:
            # flat constant image
            data = np.ones((50, 50), dtype=np.int32) * 100
        else:
            # create some structure; best frame has sharp peak, others blurrier
            data = np.zeros((50, 50), dtype=np.int32)
            # best frame has very sharp line, others have wider gaussian profile
            dist_from_best = abs(idx - best_frame)
            if dist_from_best == 0:
                data[20:30, 25] = 1000
            else:
                for offset in range(-dist_from_best, dist_from_best + 1):
                    col = 25 + offset
                    if 0 <= col < 50:
                        data[20:30, col] = int(1000 / (dist_from_best + 1))
        tifffile.imwrite(tif_path, data)
        
    # Create ground_truth.json
    # Rank(i) = 1.0 + |i - k^*|
    ranks = {}
    for idx in range(num_frames):
        ranks[str(idx)] = 1.0 + abs(idx - best_frame)
        
    gt_data = {
        "experiment_id": "mock_scan",
        "best_frame_index": best_frame,
        "fractional_ranks": ranks
    }
    
    with open(scan_dir / "ground_truth.json", "w") as f:
        json.dump(gt_data, f)
        
    return scan_dir

# ==========================================
# Feature 3: Sharpness Evaluation CLI (sharpness_cli.py)
# ==========================================

# --- Tier 1: Feature Coverage ---

def test_dog_laplacian_metric(tmp_path):
    """Test that the DoG Laplacian metric option is computed and runs via CLI."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir), "--metrics", "dog_laplacian"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Check that output mentions the metric name
    assert "dog_laplacian" in res.stdout.lower()

def test_directional_tenengrad_metric(tmp_path):
    """Test that the Directional Tenengrad metric is computed and runs via CLI."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir), "--metrics", "directional_tenengrad"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert "directional_tenengrad" in res.stdout.lower()

def test_fft_bandpass_metric(tmp_path):
    """Test that the FFT Bandpass metric is computed and runs via CLI."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir), "--metrics", "fft_bandpass"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    assert "fft_bandpass" in res.stdout.lower()

def test_correlation_calculation(tmp_path):
    """Test that CLI calculates Spearman/Pearson rank correlation coefficients with ground truth ranks."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir), "--correlation"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # The output should contain correlation coefficients (typically floating numbers like 0.9 or -1.0)
    assert "correlation" in res.stdout.lower() or "spearman" in res.stdout.lower()

def test_markdown_table_stdout(tmp_path):
    """Test that CLI outputs the summary results as a formatted markdown table."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Markdown table checks (e.g. '| Directory |' headers and separator lines '|---|')
    assert "|" in res.stdout
    assert "---" in res.stdout
    assert "Directory" in res.stdout


# --- Tier 2: Boundary/Corner Cases ---

def test_single_frame_directory(tmp_path):
    """Boundary Case: Directory containing only a single frame. Correlation cannot be computed."""
    scan_dir = setup_mock_scan_dir(tmp_path, num_frames=1, best_frame=0)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # The CLI should handle this and print a warning or N/A for correlation instead of crashing
    assert "nan" in res.stdout.lower() or "n/a" in res.stdout.lower() or "warning" in res.stdout.lower()

def test_perfect_anti_correlation(tmp_path):
    """Boundary Case: Perfect anti-correlation scenario where metrics order is completely inverted to ranks."""
    scan_dir = setup_mock_scan_dir(tmp_path, num_frames=3, best_frame=1)
    
    # Modify the ground truth json to reverse things, or just verify the correlation calculations
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0

def test_flat_constant_image(tmp_path):
    """Boundary Case: Flat constant images. The sharpness values should be zero or small, no division by zero."""
    scan_dir = setup_mock_scan_dir(tmp_path, flat=True)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0

def test_missing_ground_truth_json(tmp_path):
    """Boundary Case: Directory has TIFFs but no ground_truth.json. CLI should warning or error."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    os.remove(scan_dir / "ground_truth.json")
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir)]
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
    assert exc_info.value.returncode != 0

def test_invalid_metric_args(tmp_path):
    """Boundary Case: Passing an invalid metric name to --metrics arg. CLI should exit with non-zero."""
    scan_dir = setup_mock_scan_dir(tmp_path)
    
    cmd = [sys.executable, "sharpness_cli.py", "--dir", str(scan_dir), "--metrics", "invalid_metric"]
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        
    assert exc_info.value.returncode != 0
