"""Backward compatibility wrapper for rixs_app.core.dark_mask_store.

Deprecated: Prefer importing directly from ``rixs_app.core.dark_mask_store``.
"""

from __future__ import annotations

from rixs_app.core.dark_mask_store import (
    APPDATA_DIR,
    DEFAULT_MASK_DIR as DEFAULT_CALIBRATION_DIR,
    DARK_MASK_DIR as DARK_CAL_DIR,
    META_FILENAME,
    META_FILE,
    DarkMaskRecord,
    CalibrationRecord,
    get_dark_mask_dir as get_dark_cal_dir,
    get_meta_file_path,
    has_dark_mask as has_calibration,
    load_dark_mask as load_calibration,
    save_dark_mask as save_calibration,
    get_mask_summary as get_calibration_summary,
    clear_dark_mask as clear_calibration,
)

__all__ = [
    "APPDATA_DIR",
    "DEFAULT_CALIBRATION_DIR",
    "DARK_CAL_DIR",
    "META_FILENAME",
    "META_FILE",
    "DarkMaskRecord",
    "CalibrationRecord",
    "get_dark_cal_dir",
    "get_meta_file_path",
    "has_calibration",
    "load_calibration",
    "save_calibration",
    "get_calibration_summary",
    "clear_calibration",
]
