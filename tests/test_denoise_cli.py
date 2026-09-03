"""Integration and CLI tests for denoise_cli.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture
def denoise_test_data(tmp_path):
    input_dir = tmp_path / "raw_frames"
    input_dir.mkdir()
    single_input = tmp_path / "single_frame.tif"
    single_output = tmp_path / "single_frame_denoised.tif"

    rng = np.random.default_rng(42)
    # 2 frames in input_dir
    for i in range(2):
        img = rng.normal(loc=100.0, scale=10.0, size=(32, 32)).astype(np.float32)
        tifffile.imwrite(input_dir / f"frame_{i:03d}.tif", img)

    # 1 single frame
    single_img = rng.normal(loc=100.0, scale=10.0, size=(32, 32)).astype(np.float32)
    tifffile.imwrite(single_input, single_img)

    return {
        "root": tmp_path,
        "input_dir": input_dir,
        "single_input": single_input,
        "single_output": single_output,
    }


def test_denoise_single_file_mode(denoise_test_data):
    """Verify single file mode denoises and produces output TIFF without .denoise_version."""
    cmd = [
        sys.executable,
        "-u",
        "denoise_cli.py",
        "--input", str(denoise_test_data["single_input"]),
        "--output", str(denoise_test_data["single_output"]),
        "--diameter", "5",
        "--clip",
        "--despike",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    assert denoise_test_data["single_output"].exists()
    out_img = tifffile.imread(denoise_test_data["single_output"])
    assert out_img.shape == (32, 32)
    assert not (denoise_test_data["root"] / ".denoise_version").exists()


def test_denoise_directory_mode_and_no_marker_file(denoise_test_data):
    """Verify directory mode creates denoised/ with _denoised.tiff and NO .denoise_version."""
    cmd = [
        sys.executable,
        "-u",
        "denoise_cli.py",
        "-d", str(denoise_test_data["input_dir"]),
        "--diameter", "5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0

    out_dir = denoise_test_data["input_dir"] / "denoised"
    assert out_dir.is_dir()
    assert (out_dir / "frame_000_denoised.tiff").exists()
    assert (out_dir / "frame_001_denoised.tiff").exists()

    # Crucial check: .denoise_version marker file must NOT be written
    assert not (out_dir / ".denoise_version").exists()


def test_denoise_input_dir_alias(denoise_test_data):
    """Verify --input-dir and --dir aliases work identically."""
    cmd = [
        sys.executable,
        "-u",
        "denoise_cli.py",
        "--input-dir", str(denoise_test_data["input_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    assert res.returncode == 0
    out_dir = denoise_test_data["input_dir"] / "denoised"
    assert out_dir.is_dir()


def test_dead_arguments_eliminated(denoise_test_data):
    """Verify that removed dead arguments are rejected as unrecognized."""
    dead_flags = [
        ["--feature-low", "0.0"],
        ["--feature-high", "100.0"],
        ["--edge-margin", "50"],
        ["--high-dilate", "8"],
        ["--bg-sigma", "2.0"],
        ["--smooth-sigma", "1.2"],
    ]
    for flag_args in dead_flags:
        cmd = [
            sys.executable,
            "denoise_cli.py",
            "--input", str(denoise_test_data["single_input"]),
            "--output", str(denoise_test_data["single_output"]),
        ] + flag_args
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "unrecognized arguments" in res.stderr.lower()


def test_negative_parameters_validation(denoise_test_data):
    """Verify negative numeric parameters are rejected with non-zero exit code."""
    # Negative diameter
    cmd = [
        sys.executable,
        "denoise_cli.py",
        "--input", str(denoise_test_data["single_input"]),
        "--output", str(denoise_test_data["single_output"]),
        "--diameter", "-1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "All numeric options must be non-negative" in res.stderr

    # Negative mad-threshold
    cmd = [
        sys.executable,
        "denoise_cli.py",
        "--input", str(denoise_test_data["single_input"]),
        "--output", str(denoise_test_data["single_output"]),
        "--mad-threshold", "-2.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "All numeric options must be non-negative" in res.stderr


def test_mutually_exclusive_and_missing_modes(denoise_test_data):
    """Verify error when neither mode or both modes are specified."""
    # Neither
    res = subprocess.run([sys.executable, "denoise_cli.py"], capture_output=True, text=True)
    assert res.returncode != 0
    assert "Either --dir or --input must be specified" in res.stderr

    # Both
    res = subprocess.run([
        sys.executable, "denoise_cli.py",
        "--dir", str(denoise_test_data["input_dir"]),
        "--input", str(denoise_test_data["single_input"]),
    ], capture_output=True, text=True)
    assert res.returncode != 0
    assert "must NOT be specified" in res.stderr
