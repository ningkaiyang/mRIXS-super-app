"""PySide6 GUI tests for the zeroth-order calibration slideshow.

Replaces the old CustomTkinter/Tkinter-based ``TestZerothOrderGUI`` suite.
Uses pytest-qt (``qtbot`` fixture). Manager/pure-Python tests run without a
display; widget tests use ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import os
import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from rixs_app.main import RixsApp
from rixs_app.ui.zeroth_order_slideshow.manager import ZerothOrderManager
from rixs_app.core.dataset import ZarrSequenceManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_tif_files(tmp_path):
    """Three synthetic TIFF files (100×100, broad central blob)."""
    files = []
    for i in range(3):
        path = tmp_path / f"frame_{i + 1}.tif"
        data = np.zeros((100, 100), dtype=np.float32)
        data[40:60, 40:60] = 5.0
        tifffile.imwrite(str(path), data)
        files.append(str(path))
    return files


@pytest.fixture
def app_window(qapp, temp_tif_files, qtbot):
    """Instantiate RixsApp headlessly and register with qtbot."""
    window = RixsApp(show_window=False)
    qtbot.addWidget(window)
    yield window
    window.close()


# ---------------------------------------------------------------------------
# Smoke: instantiation
# ---------------------------------------------------------------------------

def test_zo_slideshow_instantiation(app_window):
    """Zeroth-order view and all its child panels must be instantiated."""
    v = app_window.zeroth_order_view
    assert v is not None
    assert v.canvas_panel is not None
    assert v.navbar is not None
    assert v.control_panel is not None
    assert v.tools_panel is not None
    assert v.bottom_bar is not None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def test_zo_navigation_next_and_prev(app_window, temp_tif_files, qtbot):
    """next_frame / prev_frame should advance and retract current_idx."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    assert v.manager.current_idx == 0
    v.next_frame()
    assert v.manager.current_idx == 1
    v.prev_frame()
    assert v.manager.current_idx == 0


def test_zo_navigation_prev_boundary(app_window, temp_tif_files):
    """prev_frame at frame 0 must not go below 0."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    v.prev_frame()
    assert v.manager.current_idx == 0


def test_zo_navigation_next_boundary(app_window, temp_tif_files):
    """next_frame at last frame must not advance beyond it."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    for _ in range(10):
        v.next_frame()
    assert v.manager.current_idx == len(temp_tif_files) - 1


# ---------------------------------------------------------------------------
# Autoplay
# ---------------------------------------------------------------------------

def test_zo_autoplay_toggle(app_window, temp_tif_files, qtbot):
    """Clicking the autoplay button twice should toggle autoplay on then off."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    assert not v.manager.autoplay_active
    qtbot.mouseClick(v.navbar.autoplay_button, Qt.LeftButton)
    assert v.manager.autoplay_active
    qtbot.mouseClick(v.navbar.autoplay_button, Qt.LeftButton)
    assert not v.manager.autoplay_active


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def test_zo_pipeline_stage_selection(app_window, temp_tif_files):
    """change_pipeline_stage must update manager.pipeline_stage."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    v.change_pipeline_stage("Denoised (D)")
    assert v.manager.pipeline_stage == "Denoised (D)"

    v.change_pipeline_stage("Fitted-Line Strip")
    assert v.manager.pipeline_stage == "Fitted-Line Strip"

    v.change_pipeline_stage("Raw")
    assert v.manager.pipeline_stage == "Raw"


# ---------------------------------------------------------------------------
# Slicing / colormap
# ---------------------------------------------------------------------------

def test_zo_slicing_floor_ceiling_submission(app_window, temp_tif_files):
    """Submitting floor/ceiling text entries must update manager values."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    v.handle_floor_entry_submit("0.1")
    v.handle_ceiling_entry_submit("0.9")
    assert abs(v.manager.slicing_floor - 0.1) < 0.01
    assert abs(v.manager.slicing_ceiling - 0.9) < 0.01


def test_zo_colormap_changes(app_window, temp_tif_files):
    """change_colormap must persist the new colormap on the manager."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    v.change_colormap("plasma")
    assert v.manager.colormap == "plasma"

    v.change_colormap("grayscale")
    assert v.manager.colormap == "grayscale"


# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------

def test_zo_zoom_in_and_click(app_window, temp_tif_files):
    """Zoom In enables zoom mode; clicking canvas increases zoom_factor."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    assert v.zoom_factor == 1.0
    v.zoom_in()
    assert v.zoom_mode

    v.handle_canvas_click(50, 50)
    assert v.zoom_factor > 1.0
    assert not v.zoom_mode
    assert v.zoom_center == (50, 50)


def test_zo_zoom_out_resets_to_one(app_window, temp_tif_files):
    """zoom_out after a single zoom-in must return zoom_factor to 1.0."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    v.zoom_in()
    v.handle_canvas_click(50, 50)  # zoom_factor ~1.5
    v.zoom_out()
    # After zooming out from 1.5 → 1.0
    assert v.zoom_factor == pytest.approx(1.0, abs=0.1)
    assert v.zoom_center is None


def test_zo_reset_view(app_window, temp_tif_files):
    """reset_view must set zoom_factor=1.0 and zoom_center=None."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    v.zoom_in()
    v.handle_canvas_click(50, 50)
    v.reset_view()
    assert v.zoom_factor == 1.0
    assert v.zoom_center is None


# ---------------------------------------------------------------------------
# Manager cache tests (pure Python, no GUI display required)
# ---------------------------------------------------------------------------

def test_zo_manager_cache_preserves_metadata_keys(app_window, temp_tif_files):
    """get_frame_pipeline_data must return fit_ok and overlay keys on cache hits."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    manager = app_window.zeroth_order_view.manager

    data_miss = manager.get_frame_pipeline_data(0)
    assert data_miss is not None
    assert "fit_ok" in data_miss

    data_hit = manager.get_frame_pipeline_data(0)
    assert data_hit is not None
    assert "fit_ok" in data_hit
    assert data_miss["fit_ok"] == data_hit["fit_ok"]
    assert "candidates_xy" in data_hit
    assert "inliers_xy" in data_hit
    assert "evaluator_result" in data_hit


def test_zo_manager_missing_first_file_graceful(app_window, tmp_path):
    """If the first file is missing, the view must initialise without crashing."""
    missing_path = str(tmp_path / "missing_first.tif")
    good_path = str(tmp_path / "frame_2.tif")
    data = np.zeros((100, 100), dtype=np.float32)
    data[40:60, 40:60] = 5.0
    tifffile.imwrite(good_path, data)

    bad_files = [missing_path, good_path]
    app_window.show_zeroth_order_calibration(bad_files)
    v = app_window.zeroth_order_view
    assert v is not None
    assert v.manager.current_idx == 0


def test_zo_manager_missing_last_file_no_crash(app_window, temp_tif_files, tmp_path):
    """Navigating to a missing frame must not raise; load_and_render gracefully no-ops."""
    missing = str(tmp_path / "missing_last.tif")
    bad_files = temp_tif_files + [missing]
    app_window.show_zeroth_order_calibration(bad_files)
    v = app_window.zeroth_order_view
    v.manager.current_idx = 3
    v.load_and_render()  # must not raise


# ---------------------------------------------------------------------------
# Zarr cache fallback (pure Python, no GUI display required)
# ---------------------------------------------------------------------------

def test_zo_zarr_cache_fallback(temp_tif_files):
    """ZarrSequenceManager must fall back to tempdir when main cache path fails."""
    import tempfile
    import hashlib
    with patch("os.makedirs", side_effect=PermissionError("Permission Denied")):
        manager = ZarrSequenceManager(temp_tif_files)
        assert manager.zarr_group is not None

    tif_dir = os.path.dirname(os.path.abspath(temp_tif_files[0]))
    dir_hash = hashlib.md5(tif_dir.encode("utf-8")).hexdigest()
    expected_fallback = os.path.join(tempfile.gettempdir(), f"rixs_cache_{dir_hash}")
    assert os.path.exists(expected_fallback)


# ---------------------------------------------------------------------------
# Precompute worker
# ---------------------------------------------------------------------------

def test_zo_precompute_worker_execution(app_window, temp_tif_files, qtbot):
    """trigger_precompute must run and re-enable nav buttons on completion."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    v.trigger_precompute()

    # Drain the result queue by processing Qt events until the nav buttons re-enable.
    # The QTimer (50 ms) in the view drains the queue on the GUI thread.
    def _buttons_enabled():
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        return v.navbar.prev_button.isEnabled()

    qtbot.waitUntil(_buttons_enabled, timeout=10000)
    assert v.navbar.prev_button.isEnabled()
    assert v.navbar.next_button.isEnabled()


def test_zo_precompute_missing_file_handled(app_window, temp_tif_files, tmp_path, qtbot):
    """Precompute worker with a missing file must not crash the view (fire-and-forget)."""
    bad_files = temp_tif_files + [str(tmp_path / "missing_frame.tif")]
    app_window.show_zeroth_order_calibration(bad_files)
    v = app_window.zeroth_order_view

    # The trigger call itself must not raise an exception.
    # We do not wait for background completion here to avoid event-loop issues
    # in the offscreen test environment.
    v.trigger_precompute()

    # Buttons are disabled during precompute — that is the expected state.
    # Stop autoplay to avoid timer interference after the test ends.
    v.stop_autoplay()



# ---------------------------------------------------------------------------
# Export focus curve (pure Python, no display required)
# ---------------------------------------------------------------------------

def test_zo_export_focus_curve_without_txt_metadata(app_window, temp_tif_files, tmp_path):
    """_export_focus_curve must generate focus_curve.png using Frame Index when txt_metadata is None."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    manager = app_window.zeroth_order_view.manager

    for idx in range(len(temp_tif_files)):
        manager.get_frame_pipeline_data(idx)

    for idx in range(len(temp_tif_files)):
        if idx not in manager.pipeline_results:
            manager.pipeline_results[idx] = {}
        er = MagicMock()
        er.score_valid = True
        er.fwhm_px = float(5 + idx)
        manager.pipeline_results[idx]["evaluator_result"] = er

    export_dir = str(tmp_path / "export_out")
    os.makedirs(export_dir, exist_ok=True)
    manager._export_focus_curve(
        export_dir=export_dir,
        txt_metadata=None,
        energy_dispersion=0.0,
        mono_energy_ev=0.0,
    )
    assert os.path.exists(os.path.join(export_dir, "focus_curve.png"))
