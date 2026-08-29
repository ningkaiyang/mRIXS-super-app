"""Unit tests for rixs_app/core/dark_mask_store.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from rixs_app.core.dark_mask_store import (
    DEFAULT_MASK_DIR,
    DarkMaskRecord,
    CalibrationRecord,
    clear_dark_mask,
    get_mask_summary,
    get_dark_mask_dir,
    get_meta_file_path,
    has_dark_mask,
    load_dark_mask,
    save_dark_mask,
    has_calibration,
    load_calibration,
    save_calibration,
    get_calibration_summary,
    clear_calibration,
)


@pytest.fixture
def temp_mask_dir():
    """Provides an isolated clean temporary directory for dark mask store tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_dark_mask_record_schema_and_dict_conversion():
    """Verify DarkMaskRecord fields, type integrity, and dictionary round-trip."""
    record = DarkMaskRecord(
        date="2026-08-29",
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=150,
        surviving_pixels=4190000,
        total_pixels=4194304,
        suppression_pct=0.1026,
        source_dir="/path/to/dark/folder",
        med_dark_file="MED_Dark_2026-08-29.tif",
        final_mask_file="Final_Mask_2026-08-29.tif",
    )
    d = record.to_dict()
    assert d["date"] == "2026-08-29"
    assert d["stddev_thresh"] == 40.0
    assert d["surviving_pixels"] == 4190000
    assert d["dark_frame_count"] == 150

    reconstructed = DarkMaskRecord.from_dict(d)
    assert reconstructed == record
    assert CalibrationRecord == DarkMaskRecord


def test_has_dark_mask_empty_and_nonexistent_dir(temp_mask_dir: Path):
    """Verify has_dark_mask returns False for non-existent and empty directories."""
    assert has_dark_mask(temp_mask_dir / "does_not_exist") is False
    assert has_dark_mask(temp_mask_dir) is False
    assert has_dark_mask(store_dir=temp_mask_dir) is False
    assert has_calibration(temp_mask_dir) is False


def test_has_dark_mask_partial_files(temp_mask_dir: Path):
    """Verify has_dark_mask returns False if any required file is missing."""
    meta_path = temp_mask_dir / "mask_meta.json"
    med_path = temp_mask_dir / "MED_Dark_2026-08-29.tif"
    mask_path = temp_mask_dir / "Final_Mask_2026-08-29.tif"

    # Case A: Only JSON exists
    meta_path.write_text(
        json.dumps({
            "date": "2026-08-29",
            "stddev_thresh": 40.0,
            "absdev_thresh": 60.0,
            "tail_ratio": 0.9333,
            "dark_frame_count": 150,
            "surviving_pixels": 1000,
            "total_pixels": 1000,
            "suppression_pct": 0.0,
            "source_dir": "/tmp",
            "med_dark_file": "MED_Dark_2026-08-29.tif",
            "final_mask_file": "Final_Mask_2026-08-29.tif",
        })
    )
    assert has_dark_mask(temp_mask_dir) is False

    # Case B: JSON + MED exists, Final_Mask missing
    tifffile.imwrite(med_path, np.zeros((10, 10), dtype=np.float32))
    assert has_dark_mask(temp_mask_dir) is False

    # Case C: All exist -> True
    tifffile.imwrite(mask_path, np.ones((10, 10), dtype=np.float32))
    assert has_dark_mask(temp_mask_dir) is True
    assert has_calibration(temp_mask_dir) is True


def test_save_and_load_dark_mask_roundtrip(temp_mask_dir: Path):
    """Verify complete save/load round-trip preserves arrays and metadata identically."""
    h, w = 128, 128
    rng = np.random.default_rng(42)
    med_dark = rng.normal(loc=512.5, scale=8.2, size=(h, w)).astype(np.float32)
    final_mask = (rng.uniform(0, 1, size=(h, w)) > 0.05).astype(np.float32)

    record = save_dark_mask(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=150,
        surviving_pixels=int(np.sum(final_mask)),
        total_pixels=h * w,
        suppression_pct=float((1.0 - np.sum(final_mask) / (h * w)) * 100.0),
        source_dir="/data/dark_scans/scan_001",
        date="2026-08-29",
        mask_dir=temp_mask_dir,
    )

    assert has_dark_mask(temp_mask_dir) is True
    assert record.date == "2026-08-29"

    loaded_med, loaded_mask, loaded_record = load_dark_mask(temp_mask_dir)

    assert loaded_med.dtype == np.float32
    assert loaded_mask.dtype == np.float32
    assert np.array_equal(loaded_med, med_dark)
    assert np.array_equal(loaded_mask, final_mask)
    assert loaded_record.date == "2026-08-29"
    assert loaded_record.stddev_thresh == 40.0
    assert loaded_record.surviving_pixels == int(np.sum(final_mask))
    assert loaded_record.source_dir == "/data/dark_scans/scan_001"

    # Backward compatibility functions
    b_med, b_mask, b_record = load_calibration(temp_mask_dir)
    assert np.array_equal(b_med, loaded_med)
    assert np.array_equal(b_mask, loaded_mask)


def test_get_mask_summary(temp_mask_dir: Path):
    """Verify get_mask_summary format and None on uninitialized store."""
    assert get_mask_summary(temp_mask_dir) is None
    assert get_calibration_summary(temp_mask_dir) is None

    h, w = 100, 100
    med_dark = np.zeros((h, w), dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    final_mask[0, 0] = 0.0

    save_dark_mask(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=50,
        surviving_pixels=9999,
        total_pixels=10000,
        suppression_pct=0.01,
        source_dir="/test",
        date="2026-08-29",
        mask_dir=temp_mask_dir,
    )

    summary = get_mask_summary(temp_mask_dir)
    assert summary is not None
    assert "Last generated: 2026-08-29" in summary
    assert "99.99% pixels active" in summary


def test_clear_dark_mask(temp_mask_dir: Path):
    """Verify clear_dark_mask wipes files."""
    med_dark = np.zeros((10, 10), dtype=np.float32)
    final_mask = np.ones((10, 10), dtype=np.float32)
    save_dark_mask(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=10,
        surviving_pixels=100,
        total_pixels=100,
        suppression_pct=0.0,
        source_dir="/test",
        mask_dir=temp_mask_dir,
    )
    assert has_dark_mask(temp_mask_dir) is True
    clear_dark_mask(temp_mask_dir)
    assert has_dark_mask(temp_mask_dir) is False
