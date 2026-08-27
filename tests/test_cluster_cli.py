"""Integration tests for cluster_cli.py headless tool."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import tifffile


@pytest.fixture
def cli_test_data():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dark_dir = tmp_path / "dark"
        signal_dir = tmp_path / "signal"
        output_dir = tmp_path / "output"
        dark_dir.mkdir()
        signal_dir.mkdir()
        output_dir.mkdir()

        # Create 10 synthetic dark frames (100x100)
        rng = np.random.default_rng(123)
        for i in range(10):
            dark = rng.normal(loc=500.0, scale=5.0, size=(100, 100)).astype(np.float32)
            tifffile.imwrite(dark_dir / f"dark_{i:03d}.tif", dark)

        # Create 5 synthetic signal frames (100x100) with known photon hits
        for i in range(5):
            sig = rng.normal(loc=500.0, scale=5.0, size=(100, 100)).astype(np.float32)
            # Add photon cluster at (30 + i, 40 + i)
            sig[30 + i, 40 + i] += 200.0
            tifffile.imwrite(signal_dir / f"signal_{i:03d}.tif", sig)

        yield {
            "root": tmp_path,
            "dark_dir": dark_dir,
            "signal_dir": signal_dir,
            "output_dir": output_dir,
        }


def test_cli_dark_mask_subcommand(cli_test_data):
    """Test 'dark-mask' subcommand."""
    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "dark-mask",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
        "--label", "TestDark",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert "Stage 1: Dark Mask Generation" in res.stdout
    assert "Final mask saved" in res.stdout

    med_file = cli_test_data["output_dir"] / "MED_TestDark.tif"
    mask_file = cli_test_data["output_dir"] / "Final_Mask_TestDark.tif"
    assert med_file.exists()
    assert mask_file.exists()


def test_cli_full_pipeline_subcommand(cli_test_data):
    """Test 'full' end-to-end subcommand."""
    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
        "--intden-low", "100.0",
        "--intden-high", "350.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert "Stage 1 complete" in res.stdout
    assert "Stage 2 complete" in res.stdout
    assert "Stage 3 complete" in res.stdout

    # Check generated files
    assert (cli_test_data["output_dir"] / "MED_Dark.tif").exists()
    assert (cli_test_data["output_dir"] / "Final_Mask_Dark.tif").exists()
    assert (cli_test_data["output_dir"] / "Results_clusters.xls").exists()
    assert (cli_test_data["output_dir"] / "Photon_Event_Map.tif").exists()
    assert (cli_test_data["output_dir"] / "IntDen_histogram.png").exists()


def test_cli_reconstruct_subcommand(cli_test_data):
    """Test 'reconstruct' subcommand from pre-existing Results_clusters.xls."""
    # First generate cluster file via full run
    subprocess.run([
        sys.executable, "cluster_cli.py", "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
    ], check=True)

    clusters_xls = cli_test_data["output_dir"] / "Results_clusters.xls"
    recon_out = cli_test_data["output_dir"] / "recon_test"
    recon_out.mkdir()

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "reconstruct",
        "--clusters-xls", str(clusters_xls),
        "--output-dir", str(recon_out),
        "--intden-low", "150.0",
        "--intden-high", "300.0",
        "--subpixel", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (recon_out / "Photon_Event_Map.tif").exists()
    assert (recon_out / "IntDen_histogram.png").exists()
