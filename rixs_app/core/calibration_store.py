"""Persistent storage layer for detector dark frame calibrations.

Manages disk-cached dark calibration products:
- Temporal median dark baseline image (MED_Dark_*.tif)
- Binary detector bad-pixel mask (Final_Mask_*.tif)
- Calibration metadata manifest (calibration_meta.json)

Zero UI dependencies. Thread-safe and cross-platform.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# Base package directory: rixs_app/
_PACKAGE_DIR = Path(__file__).resolve().parent.parent

# Application persistent data directory: rixs_app/appdata/
APPDATA_DIR = _PACKAGE_DIR / "appdata"

# Subdirectory dedicated to detector dark frame calibrations
DEFAULT_CALIBRATION_DIR = APPDATA_DIR / "dark_calibration"
DARK_CAL_DIR = DEFAULT_CALIBRATION_DIR

# Canonical metadata manifest filename
META_FILENAME = "calibration_meta.json"
META_FILE = DARK_CAL_DIR / META_FILENAME


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable metadata record of a completed dark frame calibration.

    Attributes:
        date: ISO format timestamp string or YYYY-MM-DD date when calibration was saved.
        stddev_thresh: Maximum allowable per-pixel dark standard deviation (ADU).
        absdev_thresh: Maximum allowable single-frame dark excursion / residual (ADU).
        tail_ratio: Fraction of stable frames required for tail-count masking (e.g. 0.9333).
        dark_frame_count: Number of dark frames analyzed during calibration.
        surviving_pixels: Total number of active (unmasked, value=1.0) detector pixels.
        total_pixels: Total detector pixel count (Height * Width).
        suppression_pct: Percentage of total pixels suppressed/masked out (0.0 to 100.0).
        source_dir: Filesystem path to the folder of raw dark TIFF frames used.
        med_dark_file: Basename or relative path of the 2D float32 median dark TIFF file.
        final_mask_file: Basename or relative path of the 2D float32 binary mask TIFF file.
    """

    date: str
    stddev_thresh: float
    absdev_thresh: float
    tail_ratio: float
    dark_frame_count: int
    surviving_pixels: int
    total_pixels: int
    suppression_pct: float
    source_dir: str
    med_dark_file: str
    final_mask_file: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize record fields to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        """Construct a CalibrationRecord from a dictionary with type enforcement.

        Raises:
            KeyError: If required keys (med_dark_file, final_mask_file) are missing.
            ValueError / TypeError: If field values cannot be cast to expected types.
        """
        return cls(
            date=str(data.get("date", "")),
            stddev_thresh=float(data.get("stddev_thresh", 40.0)),
            absdev_thresh=float(data.get("absdev_thresh", 60.0)),
            tail_ratio=float(data.get("tail_ratio", 0.9333)),
            dark_frame_count=int(data.get("dark_frame_count", 0)),
            surviving_pixels=int(data.get("surviving_pixels", 0)),
            total_pixels=int(data.get("total_pixels", 0)),
            suppression_pct=float(data.get("suppression_pct", 0.0)),
            source_dir=str(data.get("source_dir", "")),
            med_dark_file=str(data["med_dark_file"]),
            final_mask_file=str(data["final_mask_file"]),
        )


def get_dark_cal_dir(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> Path:
    """Resolve the effective dark calibration storage directory.

    Args:
        cal_dir: Optional custom storage directory.
        store_dir: Alias for cal_dir.

    Returns:
        Path object pointing to the resolved calibration directory.
    """
    target = cal_dir if cal_dir is not None else store_dir
    if target is not None:
        return Path(target).resolve()
    return Path(DARK_CAL_DIR).resolve()


def get_meta_file_path(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> Path:
    """Resolve the path to the calibration metadata JSON file."""
    return get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir) / META_FILENAME


def has_calibration(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> bool:
    """Check if a valid, readable dark calibration exists on disk.

    Verification criteria:
    1. Directory exists and contains calibration_meta.json.
    2. calibration_meta.json can be loaded and parsed into a valid CalibrationRecord.
    3. Both referenced TIFF files (med_dark_file and final_mask_file) exist on disk.

    Args:
        cal_dir: Optional custom storage directory (defaults to DEFAULT_CALIBRATION_DIR).
        store_dir: Alias for cal_dir.

    Returns:
        True if all files exist and manifest is valid; False otherwise. Never raises.
    """
    try:
        target_dir = get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir)
        meta_path = target_dir / META_FILENAME
        if not meta_path.is_file():
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        record = CalibrationRecord.from_dict(data)

        med_path = Path(record.med_dark_file)
        if not med_path.is_absolute():
            med_path = target_dir / med_path

        mask_path = Path(record.final_mask_file)
        if not mask_path.is_absolute():
            mask_path = target_dir / mask_path

        return med_path.is_file() and mask_path.is_file()
    except Exception as exc:
        logger.debug("has_calibration check failed: %s", exc)
        return False


def load_calibration(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, CalibrationRecord]:
    """Load cached dark calibration arrays and metadata from disk.

    Args:
        cal_dir: Optional custom storage directory (defaults to DEFAULT_CALIBRATION_DIR).
        store_dir: Alias for cal_dir.

    Returns:
        Tuple of:
            - med_dark: 2D float32 temporal median dark image array (H, W).
            - final_mask: 2D float32 binary pixel mask array (H, W) where 1.0=valid, 0.0=masked.
            - record: CalibrationRecord containing metadata and threshold parameters.

    Raises:
        FileNotFoundError: If calibration_meta.json or either TIFF file is missing.
        ValueError: If files are corrupted, not 2D, or array dimensions mismatch.
    """
    target_dir = get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir)
    meta_path = target_dir / META_FILENAME

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No dark calibration metadata found at {meta_path}. Please run dark frame calibration first."
        )

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        record = CalibrationRecord.from_dict(data)
    except Exception as exc:
        raise ValueError(f"Failed to parse calibration manifest at {meta_path}: {exc}") from exc

    med_path = Path(record.med_dark_file)
    if not med_path.is_absolute():
        med_path = target_dir / med_path

    mask_path = Path(record.final_mask_file)
    if not mask_path.is_absolute():
        mask_path = target_dir / mask_path

    if not med_path.is_file():
        raise FileNotFoundError(f"Median dark TIFF not found: {med_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Final mask TIFF not found: {mask_path}")

    med_dark = tifffile.imread(med_path).astype(np.float32)
    final_mask = tifffile.imread(mask_path).astype(np.float32)

    if med_dark.ndim != 2:
        med_dark = np.squeeze(med_dark)
    if final_mask.ndim != 2:
        final_mask = np.squeeze(final_mask)

    if med_dark.ndim != 2 or final_mask.ndim != 2:
        raise ValueError(
            f"Calibration images must be 2D. Got med_dark {med_dark.shape}, final_mask {final_mask.shape}"
        )
    if med_dark.shape != final_mask.shape:
        raise ValueError(
            f"Calibration shape mismatch: med_dark {med_dark.shape} vs final_mask {final_mask.shape}"
        )

    med_dark = np.nan_to_num(med_dark, nan=0.0, posinf=0.0, neginf=0.0)
    final_mask = np.nan_to_num(final_mask, nan=0.0, posinf=0.0, neginf=0.0)

    return med_dark, final_mask, record


def save_calibration(
    med_dark: np.ndarray,
    final_mask: np.ndarray,
    *,
    stddev_thresh: float,
    absdev_thresh: float,
    tail_ratio: float,
    dark_frame_count: int,
    surviving_pixels: int,
    total_pixels: int,
    suppression_pct: float,
    source_dir: Path | str,
    date: str | None = None,
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> CalibrationRecord:
    """Persist dark calibration TIFF images and metadata manifest to disk.

    Creates target directory if necessary. Writes TIFF images with YYYY-MM-DD
    timestamp suffix and saves calibration_meta.json.

    Args:
        med_dark: 2D numpy array containing median dark baseline image.
        final_mask: 2D numpy array containing binary pixel mask {0.0, 1.0}.
        stddev_thresh: Applied standard deviation threshold (ADU).
        absdev_thresh: Applied excursion threshold (ADU).
        tail_ratio: Applied stable frame fraction cutoff.
        dark_frame_count: Number of dark frames used.
        surviving_pixels: Number of active unmasked pixels.
        total_pixels: Total pixel count (H * W).
        suppression_pct: Percentage of pixels suppressed.
        source_dir: Path or string of raw dark frame source folder.
        date: Optional custom date/timestamp string (defaults to current datetime ISO).
        cal_dir: Optional custom storage directory (defaults to DEFAULT_CALIBRATION_DIR).
        store_dir: Alias for cal_dir.

    Returns:
        The newly created and persisted CalibrationRecord.

    Raises:
        ValueError: If input arrays are invalid, not 2D, or shape mismatched.
        OSError: If files cannot be written to disk.
    """
    med_arr = np.asarray(med_dark, dtype=np.float32)
    mask_arr = np.asarray(final_mask, dtype=np.float32)

    if med_arr.ndim != 2 or mask_arr.ndim != 2:
        raise ValueError(
            f"med_dark and final_mask must be 2D arrays. Got {med_arr.shape} and {mask_arr.shape}"
        )
    if med_arr.shape != mask_arr.shape:
        raise ValueError(f"Shape mismatch: med_dark {med_arr.shape} != final_mask {mask_arr.shape}")

    target_dir = get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    if date is not None:
        iso_date = str(date)
        date_suffix = str(date)[:10]
    else:
        iso_date = now.isoformat(timespec="seconds")
        date_suffix = now.strftime("%Y-%m-%d")

    med_filename = f"MED_Dark_{date_suffix}.tif"
    mask_filename = f"Final_Mask_{date_suffix}.tif"
    record = CalibrationRecord(
        date=iso_date,
        stddev_thresh=float(stddev_thresh),
        absdev_thresh=float(absdev_thresh),
        tail_ratio=float(tail_ratio),
        dark_frame_count=int(dark_frame_count),
        surviving_pixels=int(surviving_pixels),
        total_pixels=int(total_pixels),
        suppression_pct=float(suppression_pct),
        source_dir=str(source_dir),
        med_dark_file=med_filename,
        final_mask_file=mask_filename,
    )

    med_path = target_dir / med_filename
    mask_path = target_dir / mask_filename
    meta_path = target_dir / META_FILENAME

    import uuid
    uid = uuid.uuid4().hex[:8]
    temp_med = target_dir / f"{med_filename}.{os.getpid()}_{uid}.tmp"
    temp_mask = target_dir / f"{mask_filename}.{os.getpid()}_{uid}.tmp"
    temp_meta = target_dir / f"{META_FILENAME}.{os.getpid()}_{uid}.tmp"

    # Save TIFF arrays atomically
    tifffile.imwrite(temp_med, med_arr)
    tifffile.imwrite(temp_mask, mask_arr)

    # Write manifest atomically
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    temp_med.replace(med_path)
    temp_mask.replace(mask_path)
    temp_meta.replace(meta_path)

    logger.info("Successfully persisted dark calibration to %s", target_dir)
    return record


def get_calibration_summary(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> str | None:
    """Generate human-readable summary string of current calibration status.

    Format:
        "Last calibrated: YYYY-MM-DD (XX.XX% pixels active)"
        e.g., "Last calibrated: 2026-08-28 (99.86% pixels active)"

    Args:
        cal_dir: Optional custom storage directory (defaults to DEFAULT_CALIBRATION_DIR).
        store_dir: Alias for cal_dir.

    Returns:
        Summary string if a valid calibration exists; None otherwise. Never raises.
    """
    if not has_calibration(cal_dir=cal_dir, store_dir=store_dir):
        return None

    try:
        target_dir = get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir)
        meta_path = target_dir / META_FILENAME
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        record = CalibrationRecord.from_dict(data)

        date_str = str(record.date)[:10]
        if record.total_pixels > 0:
            active_pct = (record.surviving_pixels / record.total_pixels) * 100.0
        else:
            active_pct = max(0.0, 100.0 - record.suppression_pct)

        return f"Last calibrated: {date_str} ({active_pct:.2f}% pixels active)"
    except Exception as exc:
        logger.debug("Failed to get calibration summary: %s", exc)
        return None


def clear_calibration(
    cal_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> None:
    """Remove calibration files in storage directory for testing or resetting cache."""
    target_dir = get_dark_cal_dir(cal_dir=cal_dir, store_dir=store_dir)
    if not target_dir.exists():
        return
    for item in target_dir.glob("*"):
        if item.is_file():
            try:
                item.unlink()
            except OSError:
                pass
