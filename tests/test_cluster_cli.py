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

        # Create 3 synthetic dark frames (32x32)
        rng = np.random.default_rng(123)
        for i in range(3):
            dark = rng.normal(loc=500.0, scale=5.0, size=(32, 32)).astype(np.float32)
            tifffile.imwrite(dark_dir / f"dark_{i:03d}.tif", dark)

        # Create 2 synthetic signal frames (32x32) with known photon hits
        for i in range(2):
            sig = rng.normal(loc=500.0, scale=5.0, size=(32, 32)).astype(np.float32)
            sig[10 + i, 15 + i] += 200.0
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
    assert (cli_test_data["output_dir"] / "Final_Mask.tif").exists()
    assert not (cli_test_data["output_dir"] / "Final_Mask_Dark.tif").exists()
    assert (cli_test_data["output_dir"] / "Results_clusters.tsv").exists()
    assert (cli_test_data["output_dir"] / "Photon_Event_Map.tif").exists()
    assert not (cli_test_data["output_dir"] / "Photon_Event_Map_total.tif").exists()
    assert (cli_test_data["output_dir"] / "IntDen_histogram.png").exists()


def test_cli_full_auto_default_output_dir(cli_test_data):
    """Verify 'full' subcommand automatically saves to <signal_dir>/clusters/ when --output-dir is omitted."""
    expected_out = cli_test_data["signal_dir"] / "clusters"

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert expected_out.exists()
    assert (expected_out / "MED_Dark.tif").exists()
    assert (expected_out / "Final_Mask.tif").exists()
    assert not (expected_out / "Final_Mask_Dark.tif").exists()
    assert (expected_out / "Results_clusters.tsv").exists()
    assert (expected_out / "Photon_Event_Map.tif").exists()
    assert not (expected_out / "Photon_Event_Map_total.tif").exists()
    assert (expected_out / "IntDen_histogram.png").exists()


def test_cli_cluster_auto_default_output_dir(cli_test_data):
    """Verify 'cluster' subcommand automatically saves to <signal_dir>/clusters/ when --output-dir is omitted."""
    med_path = cli_test_data["root"] / "MED_Dark.tif"
    mask_path = cli_test_data["root"] / "Final_Mask.tif"
    tifffile.imwrite(med_path, np.full((32, 32), 500.0, dtype=np.float32))
    tifffile.imwrite(mask_path, np.ones((32, 32), dtype=np.float32))

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "cluster",
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--dark-tif", str(med_path),
        "--mask-tif", str(mask_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    expected_out = cli_test_data["signal_dir"] / "clusters"
    assert (expected_out / "Results_clusters.tsv").exists()


def test_cli_reconstruct_subcommand(cli_test_data):
    """Test 'reconstruct' subcommand from pre-existing Results_clusters.tsv."""
    # First generate cluster file via full run
    subprocess.run([
        sys.executable, "cluster_cli.py", "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
    ], check=True)

    clusters_tsv = cli_test_data["output_dir"] / "Results_clusters.tsv"
    recon_out = cli_test_data["output_dir"] / "recon_test"
    recon_out.mkdir()

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "reconstruct",
        "--clusters-tsv", str(clusters_tsv),
        "--output-dir", str(recon_out),
        "--intden-low", "150.0",
        "--intden-high", "300.0",
        "--subpixel", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (recon_out / "Photon_Event_Map.tif").exists()
    assert not (recon_out / "Photon_Event_Map_total.tif").exists()
    assert (recon_out / "IntDen_histogram.png").exists()


def test_cli_reconstruct_auto_default_output_dir(cli_test_data):
    """Verify 'reconstruct' subcommand defaults output directory when --output-dir is omitted."""
    subprocess.run([
        sys.executable, "cluster_cli.py", "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
    ], check=True)

    clusters_tsv = cli_test_data["output_dir"] / "Results_clusters.tsv"
    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "reconstruct",
        "--clusters-tsv", str(clusters_tsv),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert (cli_test_data["output_dir"] / "Photon_Event_Map.tif").exists()
    assert not (cli_test_data["output_dir"] / "Photon_Event_Map_total.tif").exists()
    assert (cli_test_data["output_dir"] / "IntDen_histogram.png").exists()


def test_cli_missing_input_dirs_return_error(cli_test_data):
    """Verify non-existent input folders return exit code 1 with stderr logging."""
    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "full",
        "--dark-dir", str(cli_test_data["root"] / "nonexistent_dark"),
        "--signal-dir", str(cli_test_data["signal_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode != 0
    assert "Error: Dark directory not found" in res.stderr


def test_cli_empty_directory_error_return(cli_test_data):
    """Verify empty directory with no TIFFs returns exit code 1."""
    empty_dir = cli_test_data["root"] / "empty_folder"
    empty_dir.mkdir()

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "dark-mask",
        "--dark-dir", str(empty_dir),
        "--output-dir", str(cli_test_data["output_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode != 0
    assert "Error: No TIFF files found" in res.stderr


def test_cli_unbuffered_stdout_streaming(cli_test_data):
    """Verify real-time stdout streaming without line buffering delays."""
    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line.strip())
    proc.wait()
    assert proc.returncode == 0
    assert any("Stage 1" in l for l in lines)
    assert any("Stage 2" in l for l in lines)
    assert any("Stage 3" in l for l in lines)


def test_cli_dead_flag_aliases_rejected(cli_test_data):
    """Verify legacy flag aliases --tail-ratio, --med-dark, --final-mask are rejected."""
    # 1. --tail-ratio in dark-mask
    cmd1 = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "dark-mask",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--tail-ratio", "0.95",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res1.returncode != 0
    assert "unrecognized arguments: --tail-ratio" in res1.stderr

    # 2. --med-dark in cluster
    cmd2 = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "cluster",
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--med-dark", "some_path.tif",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res2.returncode != 0
    assert "unrecognized arguments: --med-dark" in res2.stderr

    # 3. --final-mask in cluster
    cmd3 = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "cluster",
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--final-mask", "some_path.tif",
    ]
    res3 = subprocess.run(cmd3, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res3.returncode != 0
    assert "unrecognized arguments: --final-mask" in res3.stderr


def test_cli_reconstruct_legacy_xls_backwards_compatibility(cli_test_data):
    """Verify 'reconstruct' subcommand can read legacy .xls file and accept --clusters-xls flag."""
    subprocess.run([
        sys.executable, "cluster_cli.py", "full",
        "--dark-dir", str(cli_test_data["dark_dir"]),
        "--signal-dir", str(cli_test_data["signal_dir"]),
        "--output-dir", str(cli_test_data["output_dir"]),
    ], check=True)

    tsv_path = cli_test_data["output_dir"] / "Results_clusters.tsv"
    legacy_xls_path = cli_test_data["output_dir"] / "Legacy_Results_clusters.xls"
    legacy_xls_path.write_text(tsv_path.read_text())

    recon_out = cli_test_data["output_dir"] / "recon_legacy"
    recon_out.mkdir()

    cmd = [
        sys.executable,
        "-u",
        "cluster_cli.py",
        "reconstruct",
        "--clusters-xls", str(legacy_xls_path),
        "--output-dir", str(recon_out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert (recon_out / "Photon_Event_Map.tif").exists()
    assert not (recon_out / "Photon_Event_Map_total.tif").exists()

