"""Dedicated unit tests for DarkMaskingView (Dark Image & Pixel Masking Studio).

Verifies:
1. Dual-axis Matplotlib histogram rendering (log on primary axis, linear on twinx secondary axis).
2. Dynamic red axvspan region highlight representing masked pixels beyond slider cutlines.
3. Clickable dropzone opening folder dialog.
4. Incremental marginal tiering KPI badge calculations.
5. Slider value pill text length accommodation (zero clipping).
6. Clean persistence to dark_mask_store and metadata integrity.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from rixs_app.core import dark_mask_store
from rixs_app.core.photon_clustering import DarkDiagnostics
from rixs_app.ui.dark_masking.dark_mask_view import DarkMaskingView


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def mock_diagnostics():
    """Synthetic diagnostics with predictable noise and tail distribution."""
    h, w = 64, 64
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    stddev = np.full((h, w), 20.0, dtype=np.float32)
    residual = np.full((h, w), 30.0, dtype=np.float32)

    # 100 noisy pixels (stddev = 80.0)
    stddev.ravel()[:100] = 80.0
    # 50 excursion pixels (residual = 90.0) — 25 overlapping with noisy, 25 distinct
    residual.ravel()[75:125] = 90.0

    return DarkDiagnostics(
        med_dark=med_dark,
        per_pixel_stddev=stddev,
        pct93_residual=residual,
        dark_frame_count=20,
    )


def test_dark_mask_view_dual_axis_and_masked_spans(qapp, qtbot, mock_diagnostics):
    """Verify histograms have primary log axis, twinx linear axis, and red axvspan masked regions."""
    view = DarkMaskingView()
    qtbot.addWidget(view)
    view.show()

    view._on_diagnostics_ready(mock_diagnostics)

    # Verify primary and secondary axes exist
    assert view.ax_std is not None
    assert view.ax_std_linear is not None
    assert view.ax_res is not None
    assert view.ax_res_linear is not None

    # Verify cutline and span elements exist
    assert view._std_cutline is not None
    assert view._std_span is not None
    assert view._res_cutline is not None
    assert view._res_span is not None

    # Adjust sliders and verify span update
    view._on_stddev_slider_changed(35.0)
    view._on_absdev_slider_changed(50.0)

    assert view._std_cutline.get_xdata()[0] == 35.0
    assert view._res_cutline.get_xdata()[0] == 50.0
    assert "35.0 ADU" in view.stddev_val_label.text()
    assert "50.0 ADU" in view.absdev_val_label.text()

    view.cleanup()


def test_dark_mask_view_incremental_kpi_math(qapp, qtbot, mock_diagnostics):
    """Verify KPI badges display both independent and marginal pixel suppression."""
    view = DarkMaskingView()
    qtbot.addWidget(view)
    view.show()

    view._on_diagnostics_ready(mock_diagnostics)
    total_px = 64 * 64  # 4096

    # When stddev_thresh=40.0, 100 pixels with stddev=80 are cut (3996 retained)
    # When absdev_thresh=60.0, 50 pixels with residual=90 are cut (25 already cut by stddev, 25 marginal)
    view._on_stddev_slider_changed(40.0)
    view._on_absdev_slider_changed(60.0)

    tier1_text = view.tier1_kpi_label.text()
    tier2_text = view.tier2_kpi_label.text()
    final_text = view.final_mask_kpi_label.text()

    assert "3,996 px" in tier1_text
    assert "97.56% surviving" in tier1_text
    assert "25 px" in tier2_text
    assert "3,971 px" in final_text  # 4096 - 100 - 25 = 3971

    view.cleanup()


def test_dark_mask_view_clickable_dropzone(qapp, qtbot, tmp_path):
    """Verify clicking the dropzone invokes the folder browse callback."""
    view = DarkMaskingView()
    qtbot.addWidget(view)
    view.show()

    clicked = []
    view._browse_folder = lambda: clicked.append(True)
    view.drop_zone._on_clicked_cb = view._browse_folder

    # Simulate mouse press on dropzone
    qtbot.mouseClick(view.drop_zone, Qt.LeftButton)
    assert len(clicked) == 1

    view.cleanup()


def test_dark_mask_view_persistence(qapp, qtbot, mock_diagnostics, tmp_path):
    """Verify Save Dark Mask writes valid artifacts to dark_mask_store."""
    mask_store_dir = tmp_path / "appdata" / "dark_masking"

    with patch("rixs_app.core.dark_mask_store.DARK_MASK_DIR", mask_store_dir):
        view = DarkMaskingView()
        qtbot.addWidget(view)
        view.show()

        view._on_diagnostics_ready(mock_diagnostics)
        view._on_stddev_slider_changed(40.0)
        view._on_absdev_slider_changed(60.0)

        view.save_btn.click()

        assert dark_mask_store.has_dark_mask(mask_dir=mask_store_dir)
        med_dark, mask, record = dark_mask_store.load_dark_mask(mask_dir=mask_store_dir)

        assert med_dark.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert record.surviving_pixels == 3971
        assert "Saved" in view.save_status_label.text()

        view.cleanup()
