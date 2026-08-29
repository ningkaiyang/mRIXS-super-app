"""Unit tests for rixs_app/core/calibration_store.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from rixs_app.core.calibration_store import (
    DEFAULT_CALIBRATION_DIR,
    CalibrationRecord,
    clear_calibration,
    get_calibration_summary,
    get_dark_cal_dir,
    get_meta_file_path,
    has_calibration,
    load_calibration,
    save_calibration,
)


@pytest.fixture
def temp_cal_dir():
    """Provides an isolated clean temporary directory for calibration store tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_calibration_record_schema_and_dict_conversion():
    """Verify CalibrationRecord fields, type integrity, and dictionary round-trip."""
    record = CalibrationRecord(
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

    reconstructed = CalibrationRecord.from_dict(d)
    assert reconstructed == record


def test_has_calibration_empty_and_nonexistent_dir(temp_cal_dir: Path):
    """Verify has_calibration returns False for non-existent and empty directories."""
    assert has_calibration(temp_cal_dir / "does_not_exist") is False
    assert has_calibration(temp_cal_dir) is False
    assert has_calibration(store_dir=temp_cal_dir) is False


def test_has_calibration_partial_files(temp_cal_dir: Path):
    """Verify has_calibration returns False if any required file is missing."""
    meta_path = temp_cal_dir / "calibration_meta.json"
    med_path = temp_cal_dir / "MED_Dark_2026-08-29.tif"
    mask_path = temp_cal_dir / "Final_Mask_2026-08-29.tif"

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
    assert has_calibration(temp_cal_dir) is False

    # Case B: JSON + MED exists, Final_Mask missing
    tifffile.imwrite(med_path, np.zeros((10, 10), dtype=np.float32))
    assert has_calibration(temp_cal_dir) is False

    # Case C: All exist -> True
    tifffile.imwrite(mask_path, np.ones((10, 10), dtype=np.float32))
    assert has_calibration(temp_cal_dir) is True


def test_save_and_load_calibration_roundtrip(temp_cal_dir: Path):
    """Verify complete save/load round-trip preserves arrays and metadata identically."""
    h, w = 128, 128
    rng = np.random.default_rng(42)
    med_dark = rng.normal(loc=512.5, scale=8.2, size=(h, w)).astype(np.float32)
    final_mask = (rng.uniform(0, 1, size=(h, w)) > 0.05).astype(np.float32)

    record = save_calibration(
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
        cal_dir=temp_cal_dir,
    )

    assert has_calibration(temp_cal_dir) is True
    assert record.date == "2026-08-29"

    loaded_med, loaded_mask, loaded_record = load_calibration(temp_cal_dir)

    assert loaded_med.dtype == np.float32
    assert loaded_mask.dtype == np.float32
    assert np.array_equal(loaded_med, med_dark)
    assert np.array_equal(loaded_mask, final_mask)
    assert loaded_record.date == "2026-08-29"
    assert loaded_record.stddev_thresh == 40.0
    assert loaded_record.surviving_pixels == int(np.sum(final_mask))
    assert loaded_record.source_dir == "/data/dark_scans/scan_001"


def test_load_calibration_missing_raises_filenotfound(temp_cal_dir: Path):
    """Verify load_calibration raises FileNotFoundError when no calibration exists."""
    with pytest.raises(FileNotFoundError, match="[Nn]o.*dark calibration"):
        load_calibration(temp_cal_dir)


def test_load_calibration_corrupt_json_raises(temp_cal_dir: Path):
    """Verify load_calibration handles malformed JSON manifest gracefully."""
    meta_path = temp_cal_dir / "calibration_meta.json"
    meta_path.write_text("CORRUPTED_NOT_A_JSON{}}{")
    with pytest.raises(ValueError, match="Failed to parse calibration manifest"):
        load_calibration(temp_cal_dir)


def test_save_calibration_creates_deep_directories(temp_cal_dir: Path):
    """Verify save_calibration creates nested directory hierarchy automatically."""
    deep_cal_dir = temp_cal_dir / "nested" / "appdata" / "dark_cal"
    assert not deep_cal_dir.exists()

    med_dark = np.zeros((32, 32), dtype=np.float32)
    final_mask = np.ones((32, 32), dtype=np.float32)

    save_calibration(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=50,
        surviving_pixels=1024,
        total_pixels=1024,
        suppression_pct=0.0,
        source_dir="/source/path",
        cal_dir=deep_cal_dir,
    )

    assert deep_cal_dir.exists()
    assert has_calibration(deep_cal_dir) is True


def test_save_calibration_overwrite_behavior(temp_cal_dir: Path):
    """Verify saving again updates arrays and manifest cleanly without residual artifacts."""
    arr1 = np.full((32, 32), 100.0, dtype=np.float32)
    mask1 = np.ones((32, 32), dtype=np.float32)

    save_calibration(
        med_dark=arr1,
        final_mask=mask1,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=50,
        surviving_pixels=1024,
        total_pixels=1024,
        suppression_pct=0.0,
        source_dir="/first/path",
        date="2026-08-29",
        cal_dir=temp_cal_dir,
    )

    arr2 = np.full((32, 32), 200.0, dtype=np.float32)
    mask2 = np.zeros((32, 32), dtype=np.float32)
    mask2[0, 0] = 1.0

    save_calibration(
        med_dark=arr2,
        final_mask=mask2,
        stddev_thresh=35.0,
        absdev_thresh=50.0,
        tail_ratio=0.95,
        dark_frame_count=100,
        surviving_pixels=1,
        total_pixels=1024,
        suppression_pct=99.9,
        source_dir="/second/path",
        date="2026-08-29",
        cal_dir=temp_cal_dir,
    )

    loaded_med, loaded_mask, loaded_rec = load_calibration(temp_cal_dir)
    assert np.array_equal(loaded_med, arr2)
    assert np.array_equal(loaded_mask, mask2)
    assert loaded_rec.stddev_thresh == 35.0
    assert loaded_rec.surviving_pixels == 1
    assert loaded_rec.source_dir == "/second/path"


def test_get_calibration_summary_present(temp_cal_dir: Path):
    """Verify format: 'Last calibrated: YYYY-MM-DD (XX.XX% pixels active)'."""
    save_calibration(
        med_dark=np.zeros((100, 100), dtype=np.float32),
        final_mask=np.ones((100, 100), dtype=np.float32),
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=150,
        surviving_pixels=9990,
        total_pixels=10000,
        suppression_pct=0.10,
        source_dir="/path",
        date="2026-08-29",
        cal_dir=temp_cal_dir,
    )

    summary = get_calibration_summary(temp_cal_dir)
    assert summary is not None
    assert "Last calibrated: 2026-08-29" in summary
    assert "99.90% pixels active" in summary


def test_get_calibration_summary_missing(temp_cal_dir: Path):
    """Verify get_calibration_summary returns None if no calibration exists."""
    assert get_calibration_summary(temp_cal_dir) is None


def test_get_calibration_summary_zero_total_pixels_guard(temp_cal_dir: Path):
    """Verify get_calibration_summary does not raise ZeroDivisionError if total_pixels is 0."""
    meta_path = temp_cal_dir / "calibration_meta.json"
    meta_path.write_text(
        json.dumps({
            "date": "2026-08-29",
            "stddev_thresh": 40.0,
            "absdev_thresh": 60.0,
            "tail_ratio": 0.9333,
            "dark_frame_count": 0,
            "surviving_pixels": 0,
            "total_pixels": 0,
            "suppression_pct": 0.0,
            "source_dir": "/tmp",
            "med_dark_file": "MED_Dark_2026-08-29.tif",
            "final_mask_file": "Final_Mask_2026-08-29.tif",
        })
    )
    tifffile.imwrite(temp_cal_dir / "MED_Dark_2026-08-29.tif", np.zeros((1, 1), dtype=np.float32))
    tifffile.imwrite(temp_cal_dir / "Final_Mask_2026-08-29.tif", np.zeros((1, 1), dtype=np.float32))

    summary = get_calibration_summary(temp_cal_dir)
    assert summary is not None
    assert "Last calibrated: 2026-08-29" in summary


def test_default_calibration_dir_location():
    """Verify DEFAULT_CALIBRATION_DIR points to rixs_app/appdata/dark_calibration."""
    assert "rixs_app" in str(DEFAULT_CALIBRATION_DIR)
    assert str(DEFAULT_CALIBRATION_DIR).endswith("dark_calibration") or "appdata" in str(
        DEFAULT_CALIBRATION_DIR
    )


def test_float32_subinteger_precision_preservation(temp_cal_dir: Path):
    """Verify exact float32 decimal preservation across disk TIFF serialization."""
    med_dark = np.array([[512.1234, 513.5678], [514.9012, 515.3456]], dtype=np.float32)
    final_mask = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    save_calibration(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=10,
        surviving_pixels=3,
        total_pixels=4,
        suppression_pct=25.0,
        source_dir="/source",
        date="2026-08-29",
        cal_dir=temp_cal_dir,
    )

    loaded_med, loaded_mask, _ = load_calibration(temp_cal_dir)
    assert np.allclose(loaded_med, med_dark, atol=1e-6)
    assert np.array_equal(loaded_mask, final_mask)


def test_source_dir_path_object_support(temp_cal_dir: Path):
    """Verify source_dir can be passed as either Path or str."""
    save_calibration(
        med_dark=np.zeros((10, 10), dtype=np.float32),
        final_mask=np.ones((10, 10), dtype=np.float32),
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=10,
        surviving_pixels=100,
        total_pixels=100,
        suppression_pct=0.0,
        source_dir=Path("/path/as/pathlib/object"),
        cal_dir=temp_cal_dir,
    )
    _, _, record = load_calibration(temp_cal_dir)
    assert isinstance(record.source_dir, str)
    assert record.source_dir == str(Path("/path/as/pathlib/object"))


def test_save_validation_errors(temp_cal_dir: Path):
    """Verify save_calibration validates 2D array shapes and matching dimensions."""
    with pytest.raises(ValueError, match="must be 2D arrays"):
        save_calibration(
            med_dark=np.zeros((10,), dtype=np.float32),
            final_mask=np.zeros((10, 10), dtype=np.float32),
            stddev_thresh=40.0,
            absdev_thresh=60.0,
            tail_ratio=0.93,
            dark_frame_count=5,
            surviving_pixels=100,
            total_pixels=100,
            suppression_pct=0.0,
            source_dir="/tmp",
            cal_dir=temp_cal_dir,
        )

    with pytest.raises(ValueError, match="Shape mismatch"):
        save_calibration(
            med_dark=np.zeros((10, 10), dtype=np.float32),
            final_mask=np.zeros((20, 20), dtype=np.float32),
            stddev_thresh=40.0,
            absdev_thresh=60.0,
            tail_ratio=0.93,
            dark_frame_count=5,
            surviving_pixels=100,
            total_pixels=100,
            suppression_pct=0.0,
            source_dir="/tmp",
            cal_dir=temp_cal_dir,
        )


def test_clear_calibration(temp_cal_dir: Path):
    """Verify clear_calibration removes files and resets has_calibration."""
    save_calibration(
        med_dark=np.zeros((10, 10), dtype=np.float32),
        final_mask=np.ones((10, 10), dtype=np.float32),
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=5,
        surviving_pixels=100,
        total_pixels=100,
        suppression_pct=0.0,
        source_dir="/tmp",
        cal_dir=temp_cal_dir,
    )
    assert has_calibration(temp_cal_dir) is True
    clear_calibration(temp_cal_dir)
    assert has_calibration(temp_cal_dir) is False


def test_record_immutability_and_type_coercion():
    """Verify CalibrationRecord immutability (frozen=True) and dictionary parsing contracts."""
    record = CalibrationRecord(
        date="2026-08-28T12:00:00",
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=150,
        surviving_pixels=4190000,
        total_pixels=4194304,
        suppression_pct=0.1026,
        source_dir="/path/to/dark",
        med_dark_file="MED_Dark_2026-08-28.tif",
        final_mask_file="Final_Mask_2026-08-28.tif",
    )

    with pytest.raises(Exception):
        record.stddev_thresh = 50.0  # type: ignore

    coerced_dict = {
        "date": "2026-08-29",
        "stddev_thresh": "45.5",
        "absdev_thresh": "65.5",
        "tail_ratio": "0.95",
        "dark_frame_count": "100",
        "surviving_pixels": "990",
        "total_pixels": "1000",
        "suppression_pct": "1.0",
        "source_dir": "/custom/dir",
        "med_dark_file": "MED_Dark.tif",
        "final_mask_file": "Final_Mask.tif",
    }
    coerced_record = CalibrationRecord.from_dict(coerced_dict)
    assert coerced_record.stddev_thresh == 45.5
    assert coerced_record.surviving_pixels == 990

    with pytest.raises(KeyError):
        CalibrationRecord.from_dict({"date": "2026-08-29", "final_mask_file": "mask.tif"})


def test_has_calibration_full_negative_branch_matrix(tmp_path: Path):
    """Verify has_calibration gracefully returns False on all negative branches without raising."""
    cal_dir = tmp_path / "negative_matrix_cal"
    assert has_calibration(cal_dir=cal_dir) is False

    cal_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cal_dir / "calibration_meta.json"

    # Empty metadata file (0-byte)
    meta_file.write_text("")
    assert has_calibration(cal_dir=cal_dir) is False

    # Malformed JSON
    meta_file.write_text("{ unclosed json...")
    assert has_calibration(cal_dir=cal_dir) is False

    # Missing TIFFs
    meta_file.write_text(json.dumps({
        "date": "2026-08-28",
        "med_dark_file": "MED_Dark_missing.tif",
        "final_mask_file": "Final_Mask_missing.tif",
    }))
    assert has_calibration(cal_dir=cal_dir) is False

    # Only one TIFF exists
    tifffile.imwrite(str(cal_dir / "MED_Dark_missing.tif"), np.zeros((10, 10), dtype=np.float32))
    assert has_calibration(cal_dir=cal_dir) is False

    # Both exist
    tifffile.imwrite(str(cal_dir / "Final_Mask_missing.tif"), np.ones((10, 10), dtype=np.float32))
    assert has_calibration(cal_dir=cal_dir) is True


def test_load_calibration_array_shape_squeezing(tmp_path: Path):
    """Verify load_calibration shape squeezing (3D singletons) and dimensional validation."""
    cal_dir = tmp_path / "shape_validation_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    meta_file = cal_dir / "calibration_meta.json"
    med_file = cal_dir / "MED_Dark.tif"
    mask_file = cal_dir / "Final_Mask.tif"

    meta_file.write_text(json.dumps({
        "date": "2026-08-28",
        "stddev_thresh": 40.0,
        "absdev_thresh": 60.0,
        "tail_ratio": 0.9333,
        "dark_frame_count": 10,
        "surviving_pixels": 100,
        "total_pixels": 100,
        "suppression_pct": 0.0,
        "source_dir": "/tmp",
        "med_dark_file": "MED_Dark.tif",
        "final_mask_file": "Final_Mask.tif",
    }))

    # Write (1, 10, 10) 3D singletons
    tifffile.imwrite(str(med_file), np.zeros((1, 10, 10), dtype=np.float32))
    tifffile.imwrite(str(mask_file), np.ones((1, 10, 10), dtype=np.float32))

    med, mask, record = load_calibration(cal_dir=cal_dir)
    assert med.shape == (10, 10)
    assert mask.shape == (10, 10)

