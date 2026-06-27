"""Tests for the headless TIFF alignment CLI (``align_cli.py``).

Each test creates small synthetic images (64 × 64) saved as TIF files in
temporary directories provided by pytest's ``tmp_path`` fixture.  The
tests import functions directly from :mod:`align_cli` — no subprocess
calls are used.
"""

import json
import os
import sys

import numpy as np
import pytest
import tifffile

# Ensure the project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from align_cli import (
    CLIZarrSequenceManager,
    discover_directories,
    find_best_threshold,
    process_directory,
    save_comparison_png,
    _parse_args,
)
from rixs_app.core.dataset import _frame_key


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_frame(
    h: int = 64,
    w: int = 64,
    line_row: int = 32,
    line_intensity: float = 1000.0,
    noise_level: float = 10.0,
    shift_y: int = 0,
) -> np.ndarray:
    """Create a synthetic 2-D float32 image with a bright horizontal line.

    A horizontal line of width 3 pixels is placed at *line_row + shift_y*,
    with optional additive Gaussian noise.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        line_row: Centre row of the bright line (before shift).
        line_intensity: Peak intensity value of the line.
        noise_level: Standard deviation of additive Gaussian noise.
        shift_y: Vertical shift applied to the line position (for drift
            simulation).

    Returns:
        2-D ``float32`` numpy array of shape ``(h, w)``.
    """
    rng = np.random.RandomState(42 + shift_y)
    img = rng.normal(loc=50.0, scale=noise_level, size=(h, w)).astype(np.float32)
    img = np.clip(img, 0, None)
    row = line_row + shift_y
    for r in range(max(0, row - 1), min(h, row + 2)):
        img[r, :] = line_intensity
    return img


def _populate_dir(directory, n_frames: int = 3, shift_step: int = 0):
    """Write *n_frames* synthetic TIF files into *directory*.

    Args:
        directory: Path object or string pointing to an existing
            directory.
        n_frames: Number of frames to create.
        shift_step: Vertical shift increment per frame (to simulate
            drift).

    Returns:
        List of absolute paths to the created files.
    """
    paths = []
    for i in range(n_frames):
        img = _make_synthetic_frame(shift_y=i * shift_step)
        p = os.path.join(str(directory), f"frame_{i:03d}.tif")
        tifffile.imwrite(p, img)
        paths.append(p)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverDirectories:
    """Tests for :func:`discover_directories`."""

    def test_discover_directories_single(self, tmp_path):
        """A single directory with ≥ 2 TIFs should be discovered."""
        _populate_dir(tmp_path, n_frames=3)
        dirs = discover_directories(str(tmp_path), recursive=False)
        assert len(dirs) == 1
        assert dirs[0] == str(tmp_path.resolve())

    def test_discover_directories_recursive(self, tmp_path):
        """Recursive scan should find multiple subdirectories with TIFs."""
        sub_a = tmp_path / "sub_a"
        sub_b = tmp_path / "sub_b"
        sub_a.mkdir()
        sub_b.mkdir()
        _populate_dir(sub_a, n_frames=3)
        _populate_dir(sub_b, n_frames=2)
        dirs = discover_directories(str(tmp_path), recursive=True)
        # Both sub_a and sub_b should be discovered
        assert len(dirs) == 2
        basenames = {os.path.basename(d) for d in dirs}
        assert basenames == {"sub_a", "sub_b"}

    def test_skip_empty_directory(self, tmp_path):
        """A directory with fewer than 2 TIF files should NOT be discovered."""
        # Only write 1 frame
        img = _make_synthetic_frame()
        tifffile.imwrite(str(tmp_path / "only_one.tif"), img)
        dirs = discover_directories(str(tmp_path), recursive=False)
        assert len(dirs) == 0

    def test_ignore_sum_and_tif_cache_directories(self, tmp_path):
        """Verify that sum and tif-cache subdirectories are ignored during recursive discovery."""
        sub_a = tmp_path / "sub_a"
        sum_dir = sub_a / "sum"
        cache_dir = sub_a / "tif-cache"
        sub_a.mkdir()
        sum_dir.mkdir()
        cache_dir.mkdir()

        # Populate directories with >= 2 TIF files
        _populate_dir(sub_a, n_frames=3)
        _populate_dir(sum_dir, n_frames=2)
        _populate_dir(cache_dir, n_frames=2)

        dirs = discover_directories(str(tmp_path), recursive=True)

        # Only sub_a should be discovered; sum and tif-cache must be ignored
        assert len(dirs) == 1
        assert dirs[0] == str(sub_a.resolve())


class TestProcessDirectory:
    """Tests for :func:`process_directory` output structure."""

    def test_output_structure_ecc(self, tmp_path):
        """Verify sum/ directory contains expected files after ECC alignment.

        Checks that ``aligned_sum_ECC.tif``, ``aligned_offsets_ECC.json``,
        and ``base_sum.tif`` are all created.
        """
        _populate_dir(tmp_path, n_frames=3, shift_step=1)
        process_directory(
            dir_path=str(tmp_path),
            engines=['ECC'],
            threshold='99.9',
            save_png=False,
            overwrite=True,
            ephemeral_cache=True,
            save_json=True,
        )
        sum_dir = tmp_path / "sum"
        assert sum_dir.is_dir()
        assert (sum_dir / "base_sum.tif").exists()
        assert (sum_dir / "aligned_sum_ECC.tif").exists()
        assert (sum_dir / "aligned_offsets_ECC.json").exists()

    def test_json_offsets_format(self, tmp_path):
        """Verify JSON offset log has the correct structure and all required keys.

        Expected keys: ``engine``, ``threshold``, ``ref_mode``,
        ``timestamp``, ``offsets``.  Each offset entry is a 2-element
        list ``[dx, dy]``.
        """
        _populate_dir(tmp_path, n_frames=3, shift_step=1)
        process_directory(
            dir_path=str(tmp_path),
            engines=['ECC'],
            threshold='99.9',
            save_png=False,
            overwrite=True,
            ephemeral_cache=True,
            save_json=True,
        )
        json_path = tmp_path / "sum" / "aligned_offsets_ECC.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        # Required top-level keys
        assert "engine" in data
        assert "threshold" in data
        assert "ref_mode" in data
        assert "timestamp" in data
        assert "offsets" in data

        assert data["engine"] == "ECC"
        assert data["ref_mode"] == "frame0"

        # Offsets structure: keys are frame indices as strings, values
        # are [dx, dy] lists.  Frame 0 is the reference and must be [0, 0].
        offsets = data["offsets"]
        assert "0" in offsets
        assert offsets["0"] == [0.0, 0.0]
        for key, val in offsets.items():
            assert isinstance(val, list)
            assert len(val) == 2

    def test_comparison_png_created(self, tmp_path):
        """Verify a comparison PNG is created when ``--png`` is used."""
        _populate_dir(tmp_path, n_frames=3, shift_step=1)
        process_directory(
            dir_path=str(tmp_path),
            engines=['ECC'],
            threshold='99.9',
            save_png=True,
            overwrite=True,
            ephemeral_cache=True,
        )
        png_path = tmp_path / "sum" / "comparison_ECC.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0

    def test_default_no_json(self, tmp_path):
        """Verify that by default, save_json is False and no JSON offset log is created."""
        _populate_dir(tmp_path, n_frames=3, shift_step=1)
        process_directory(
            dir_path=str(tmp_path),
            engines=['ECC'],
            threshold='99.9',
            save_png=False,
            overwrite=True,
            ephemeral_cache=True,
        )
        sum_dir = tmp_path / "sum"
        assert sum_dir.is_dir()
        assert (sum_dir / "base_sum.tif").exists()
        assert (sum_dir / "aligned_sum_ECC.tif").exists()
        assert not (sum_dir / "aligned_offsets_ECC.json").exists()

    def test_parse_args_json(self):
        """Verify that the --json flag is parsed correctly."""
        # When --json is NOT passed, it should default to False
        args_no_json = _parse_args(['-d', '/some/dir'])
        assert args_no_json.json is False

        # When --json is passed, it should be True
        args_json = _parse_args(['-d', '/some/dir', '--json'])
        assert args_json.json is True


class TestPCA:
    """Tests for PCA engine and auto-threshold."""

    def test_auto_threshold_pca(self, tmp_path):
        """Verify auto-threshold mode works for the PCA engine.

        When ``threshold='auto'``, the pipeline should run
        ``find_best_threshold`` per frame and produce valid outputs.
        """
        _populate_dir(tmp_path, n_frames=3, shift_step=0)
        process_directory(
            dir_path=str(tmp_path),
            engines=['PCA'],
            threshold='auto',
            save_png=False,
            overwrite=True,
            ephemeral_cache=True,
            save_json=True,
        )
        sum_dir = tmp_path / "sum"
        assert (sum_dir / "aligned_sum_PCA.tif").exists()
        json_path = sum_dir / "aligned_offsets_PCA.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["engine"] == "PCA"
        assert data["threshold"] == "auto"


class TestFindBestThreshold:
    """Tests for the standalone :func:`find_best_threshold` function."""

    def test_find_best_threshold(self):
        """Test that find_best_threshold returns a sensible value.

        A synthetic image with a sharp bright line should produce a
        threshold in the 98–100 range.  The returned value must be a
        float.
        """
        img = _make_synthetic_frame(
            h=64, w=64, line_row=32, line_intensity=5000.0, noise_level=5.0
        )
        t = find_best_threshold(img)
        assert isinstance(t, float)
        assert 98.0 <= t <= 100.0


class TestCLIZarrManager:
    """Tests for :class:`CLIZarrSequenceManager`."""

    def test_cli_zarr_manager_synchronous(self, tmp_path):
        """Verify CLIZarrSequenceManager loads frames synchronously.

        After construction completes, all frames should be present in
        the Zarr group and ``median_frame`` should be a non-None 2-D
        float array.
        """
        paths = _populate_dir(tmp_path, n_frames=4, shift_step=0)
        mgr = CLIZarrSequenceManager(paths)

        # All frames should be cached
        for fpath in paths:
            key = _frame_key(fpath)
            assert key in mgr.zarr_group, (
                f"Frame {os.path.basename(fpath)} not cached"
            )

        # Median should be computed
        assert mgr.median_frame is not None
        assert mgr.median_frame.ndim == 2
        assert mgr.median_frame.shape == (64, 64)
        assert mgr.median_frame.dtype == np.float64 or mgr.median_frame.dtype == np.float32
