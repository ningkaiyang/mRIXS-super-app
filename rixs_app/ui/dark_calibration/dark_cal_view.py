"""Backward compatibility wrapper for rixs_app.ui.dark_masking.dark_mask_view.

Deprecated: Prefer importing directly from ``rixs_app.ui.dark_masking``.
"""

from __future__ import annotations

from rixs_app.ui.dark_masking.dark_mask_view import (
    DarkMaskingView,
    DarkCalibrationView,
    DropZoneFrame,
)

__all__ = ["DarkMaskingView", "DarkCalibrationView", "DropZoneFrame"]
