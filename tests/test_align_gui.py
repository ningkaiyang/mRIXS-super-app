"""PySide6 GUI tests for the alignment slideshow.

Uses pytest-qt (``qtbot`` fixture). These tests replace the previous
CustomTkinter/Tkinter-based test suite while preserving the same logical
coverage of feature areas F1-F5 and manager unit tests.

Core-logic tests (phase correlation math, warp_image, find_peak_line,
preprocess_image) are pure-Python and do NOT need the GUI — they run
unchanged in this file alongside the GUI tests.
"""

from __future__ import annotations

import os
import queue
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSlider

from rixs_app.main import RixsApp
from rixs_app.core import (
    natural_sort,
    find_peak_line,
    phase_correlation_offset,
    warp_image,
    preprocess_image,
    PCAFitFailure,
)
from rixs_app.ui.alignment_slideshow.alignment_manager import SlideshowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication to avoid repeated teardown."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def temp_tif_files(tmp_path):
    """Three tiny synthetic TIFF files inside a pytest-managed temp dir.

    Using ``tmp_path`` instead of ``tempfile.TemporaryDirectory`` avoids
    OSError on Windows/macOS when Zarr has written sub-directories that
    prevent shutil.rmtree from deleting the directory.
    """
    files = []
    for i in range(3):
        path = tmp_path / f"frame_{i + 1}.tif"
        data = np.zeros((100, 100), dtype=np.float32)
        data[:, 50 + i * 2] = 10.0
        data[10, 40] = 9.0
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
# F1 — Sorting / File Management
# ---------------------------------------------------------------------------

def test_f1_01_select_files_adds_to_list(app_window, temp_tif_files, qtbot):
    """Selecting files via mock file dialog populates the file list."""
    sv = app_window.sorting_view
    with patch(
        "rixs_app.ui.sorting_view.QFileDialog.getOpenFileNames",
        return_value=([temp_tif_files[1], temp_tif_files[0]], "")
    ):
        qtbot.mouseClick(sv.select_button, Qt.LeftButton)
    assert sv.file_list == [temp_tif_files[0], temp_tif_files[1]]


def test_f1_02_natural_sort_empty_list():
    assert natural_sort([]) == []


def test_f1_03_natural_sort_preserves_count():
    lst = ["frame_2.tif", "frame_10.tif", "frame_1.tif"]
    assert len(natural_sort(lst)) == 3


def test_f1_04_natural_sort_ordering():
    lst = ["frame_10.tif", "frame_2.tif", "frame_1.tif"]
    assert natural_sort(lst) == ["frame_1.tif", "frame_2.tif", "frame_10.tif"]


def test_f1_05_drag_drop_reorder_item(app_window):
    sv = app_window.sorting_view
    sv.file_list = ["a.tif", "b.tif", "c.tif"]
    sv.update_listbox()
    # Move item 1 ('b.tif') to top
    item = sv.list_widget.takeItem(1)
    sv.list_widget.insertItem(0, item)
    sv._on_list_reordered()
    assert sv.file_list == ["b.tif", "a.tif", "c.tif"]


def test_f1_06_auto_sort_on_import(app_window):
    sv = app_window.sorting_view
    # Natural sort order check
    from rixs_app.core import natural_sort
    unsorted = ["frame_10.tif", "frame_2.tif", "frame_1.tif"]
    assert natural_sort(unsorted) == ["frame_1.tif", "frame_2.tif", "frame_10.tif"]


def test_f1_09_remove_item(app_window, qtbot):
    sv = app_window.sorting_view
    sv.file_list = ["a.tif", "b.tif", "c.tif"]
    sv.selected_index = 1
    qtbot.mouseClick(sv.remove_button, Qt.LeftButton)
    assert sv.file_list == ["a.tif", "c.tif"]


def test_f1_10_remove_adjusts_selection(app_window, qtbot):
    sv = app_window.sorting_view
    sv.file_list = ["a.tif", "b.tif"]
    sv.selected_index = 1
    qtbot.mouseClick(sv.remove_button, Qt.LeftButton)
    assert sv.selected_index == 0


def test_f1_11_start_slideshow_disabled_if_empty(app_window, qtbot):
    sv = app_window.sorting_view
    sv.file_list = []
    qtbot.mouseClick(sv.start_button, Qt.LeftButton)
    # Slideshow should NOT be the current page
    assert app_window._stack.currentWidget() is not app_window.slideshow_view


# ---------------------------------------------------------------------------
# F2 — Slideshow navigation
# ---------------------------------------------------------------------------

def test_f2_01_transition_to_slideshow_displays_first_frame(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    assert app_window._stack.currentWidget() is app_window.slideshow_view
    assert app_window.slideshow_view.current_idx == 0


def test_f2_02_canvas_initializes(app_window):
    assert app_window.slideshow_view.canvas_panel is not None


def test_f2_03_navigation_next_frame(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    qtbot.mouseClick(app_window.slideshow_view.navbar.next_button, Qt.LeftButton)
    assert app_window.slideshow_view.current_idx == 1


def test_f2_04_navigation_prev_frame(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    qtbot.mouseClick(app_window.slideshow_view.navbar.next_button, Qt.LeftButton)
    qtbot.mouseClick(app_window.slideshow_view.navbar.prev_button, Qt.LeftButton)
    assert app_window.slideshow_view.current_idx == 0


def test_f2_05_navigation_next_boundary(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    for _ in range(5):
        qtbot.mouseClick(app_window.slideshow_view.navbar.next_button, Qt.LeftButton)
    assert app_window.slideshow_view.current_idx == 2


def test_f2_06_navigation_prev_boundary(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    qtbot.mouseClick(app_window.slideshow_view.navbar.prev_button, Qt.LeftButton)
    assert app_window.slideshow_view.current_idx == 0


def test_f2_09_jump_to_frame(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    app_window.slideshow_view.jump_to_frame(2)
    assert app_window.slideshow_view.current_idx == 2


def test_f2_10_back_to_sorting_restores_view(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    qtbot.mouseClick(app_window.slideshow_view.navbar.back_button, Qt.LeftButton)
    assert app_window._stack.currentWidget() is app_window.sorting_view


def test_f2_11_keyboard_navigation_next_prev(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    assert app_window.slideshow_view.current_idx == 0
    qtbot.keyClick(app_window, Qt.Key_Right)
    assert app_window.slideshow_view.current_idx == 1
    qtbot.keyClick(app_window, Qt.Key_Left)
    assert app_window.slideshow_view.current_idx == 0


# ---------------------------------------------------------------------------
# F3 — PCA threshold controls
# ---------------------------------------------------------------------------

def test_f3_03_pca_threshold_slider_change_updates_label(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    # change_pca_threshold sets manager.pca_threshold and syncs the PCA panel UI
    app_window.slideshow_view.change_pca_threshold(95.0)
    # The manager's pca_threshold must be updated
    assert app_window.slideshow_view.pca_threshold == 95.0


def test_f3_05_pca_flat_image_fallback_centroid():
    flat_img = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(PCAFitFailure):
        find_peak_line(flat_img, 99.0)


def test_f3_08_pca_threshold_out_of_bounds_raises():
    img = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        find_peak_line(img, -1.0)
    with pytest.raises(ValueError):
        find_peak_line(img, 101.0)


def test_f3_09_pca_invalid_image_shape_raises():
    with pytest.raises(ValueError):
        find_peak_line(np.array([1, 2, 3]), 99.0)


# ---------------------------------------------------------------------------
# F4 — Phase correlation & warp toggle
# ---------------------------------------------------------------------------

def test_f4_01_warp_switch_initial_state(app_window, temp_tif_files):
    app_window.show_slideshow(temp_tif_files)
    assert app_window.slideshow_view.warp_enabled


def test_f4_04_phase_correlation_offset_zero():
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 1.0
    dx, dy = phase_correlation_offset(img, img)
    assert abs(dx) < 0.1
    assert abs(dy) < 0.1


def test_f4_05_phase_correlation_offset_shifted():
    y, x = np.mgrid[0:128, 0:128]
    ref = np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 10 ** 2)).astype(np.float32)
    M = np.float32([[1, 0, 3.0], [0, 1, 4.0]])
    target = cv2.warpAffine(ref, M, (128, 128))
    dx, dy = phase_correlation_offset(ref, target)
    assert abs(dx - 3.0) < 0.5
    assert abs(dy - 4.0) < 0.5


def test_f4_06_phase_correlation_dimension_mismatch():
    img1 = np.zeros((100, 100), dtype=np.float32)
    img2 = np.zeros((100, 90), dtype=np.float32)
    with pytest.raises(ValueError):
        phase_correlation_offset(img1, img2)


def test_f4_07_warp_image_zero_translation():
    img = np.random.rand(10, 10).astype(np.float32)
    warped = warp_image(img, 0.0, 0.0)
    np.testing.assert_array_equal(img, warped)


def test_f4_08_warp_image_translation_coords():
    img = np.zeros((10, 10), dtype=np.float32)
    img[4, 4] = 1.0
    warped = warp_image(img, 1.0, 2.0)
    assert warped[6, 5] == 1.0
    assert warped[4, 4] == 0.0


def test_f4_09_warp_image_rgb():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[4, 4, 0] = 255
    warped = warp_image(img, 1.0, 1.0)
    assert warped[5, 5, 0] == 255


def test_f4_10_warp_image_invalid_shape():
    with pytest.raises(ValueError):
        warp_image(np.array([1, 2, 3]), 1.0, 1.0)


def test_f4_11_warp_zero_images():
    img1 = np.zeros((100, 100), dtype=np.float32)
    img2 = np.zeros((100, 100), dtype=np.float32)
    dx, dy = phase_correlation_offset(img1, img2)
    assert dx == 0.0
    assert dy == 0.0


def test_f4_12_phase_correlation_no_inplace_mutation():
    array1 = np.random.rand(100, 100).astype(np.float64)
    array2 = np.random.rand(100, 100).astype(np.float64)
    clone1 = array1.copy()
    clone2 = array2.copy()
    _ = phase_correlation_offset(array1, array2)
    np.testing.assert_array_equal(array1, clone1)
    np.testing.assert_array_equal(array2, clone2)


# ---------------------------------------------------------------------------
# F5 — Colormap / readme static checks
# ---------------------------------------------------------------------------

def test_f5_01_readme_exists():
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    assert os.path.exists(readme_path)


def test_f5_06_colormap_menu_change(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    app_window.slideshow_view.change_colormap("inferno")
    assert app_window.slideshow_view.colormap == "inferno"


def test_f5_07_colormap_menu_triggers_redraw(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    app_window.slideshow_view.change_colormap("inferno")
    rgb = app_window.slideshow_view.current_rgb
    assert rgb is not None


def test_f5_09_preprocess_image_grayscale(temp_tif_files):
    rgb, raw = preprocess_image(temp_tif_files[0], "grayscale", 100.0)
    assert rgb.shape == (100, 100, 3)
    np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])


def test_f5_10_preprocess_image_invalid_percentile(temp_tif_files):
    with pytest.raises(ValueError):
        preprocess_image(temp_tif_files[0], "grayscale", -10.0)


# ---------------------------------------------------------------------------
# Tier 3 — Cross-feature interactions
# ---------------------------------------------------------------------------

def test_t3_01_sort_retains_selection_integrity(app_window, temp_tif_files):
    sv = app_window.sorting_view
    from rixs_app.core import natural_sort
    sv.file_list = natural_sort([temp_tif_files[1], temp_tif_files[0]])
    assert sv.file_list == [temp_tif_files[0], temp_tif_files[1]]


def test_t3_02_navigating_preserves_colormap_across_frames(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    app_window.slideshow_view.change_colormap("inferno")
    qtbot.mouseClick(app_window.slideshow_view.navbar.next_button, Qt.LeftButton)
    assert app_window.slideshow_view.colormap == "inferno"


def test_t3_03_navigating_preserves_pca_threshold(app_window, temp_tif_files, qtbot):
    app_window.show_slideshow(temp_tif_files)
    app_window.slideshow_view.change_pca_threshold(85.5)
    qtbot.mouseClick(app_window.slideshow_view.navbar.next_button, Qt.LeftButton)
    assert app_window.slideshow_view.pca_threshold == 85.5


def test_t3_05_back_to_sorting_preserves_list_order(app_window, temp_tif_files, qtbot):
    sv = app_window.sorting_view
    sv.file_list = [temp_tif_files[1], temp_tif_files[0]]
    app_window.show_slideshow(sv.file_list)
    qtbot.mouseClick(app_window.slideshow_view.navbar.back_button, Qt.LeftButton)
    assert sv.file_list == [temp_tif_files[1], temp_tif_files[0]]


# ---------------------------------------------------------------------------
# Tier 4 — Real TIFF integration (skipped if sample not present)
# ---------------------------------------------------------------------------

REAL_TIF_1 = "tests/samples/Sample1VL_200F_frames_1-200.tif"
REAL_TIF_2 = "tests/samples/Sample1VL_200F_frames_201-400.tif"


@pytest.mark.skipif(not os.path.exists(REAL_TIF_1), reason="Sample TIFF not found")
def test_t4_01_real_tif_load_and_preprocess():
    rgb, raw = preprocess_image(REAL_TIF_1, "grayscale", 99.0)
    assert raw.ndim == 2
    assert rgb.shape[:2] == raw.shape
    assert rgb.shape[2] == 3


@pytest.mark.skipif(not os.path.exists(REAL_TIF_1), reason="Sample TIFF not found")
def test_t4_02_real_tif_find_peak_line():
    _, raw = preprocess_image(REAL_TIF_1, "grayscale", 99.0)
    origin, direction = find_peak_line(raw, 99.0)
    assert origin.shape == (2,)
    assert direction.shape == (2,)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-5


@pytest.mark.skipif(
    not os.path.exists(REAL_TIF_1) or not os.path.exists(REAL_TIF_2),
    reason="Sample TIFFs not found"
)
def test_t4_03_real_tif_phase_correlation():
    _, raw1 = preprocess_image(REAL_TIF_1, "grayscale", 100.0)
    _, raw2 = preprocess_image(REAL_TIF_2, "grayscale", 100.0)
    dx, dy = phase_correlation_offset(raw1, raw2)
    assert isinstance(dx, float)
    assert isinstance(dy, float)


# ---------------------------------------------------------------------------
# Manager unit tests (pure Python, no GUI)
# ---------------------------------------------------------------------------

def test_mgr_init_defines_manual_variables():
    mgr = SlideshowManager(queue.Queue())
    assert not mgr.manual_mode
    assert mgr.manual_clicks == []


def test_mgr_start_resets_manual_variables():
    mgr = SlideshowManager(queue.Queue())
    mgr.manual_mode = True
    mgr.manual_clicks = [(1, 2), (3, 4)]
    mgr.start([])
    assert not mgr.manual_mode
    assert mgr.manual_clicks == []


def test_mgr_get_offset_with_none_ref_origin():
    mgr = SlideshowManager(queue.Queue())
    mgr.per_frame_manual[0] = np.array([10.0, 20.0])
    mgr.ref_origin = None
    mgr.file_list = ["dummy.tif"]
    dx, dy = mgr.get_offset(0)
    assert (dx, dy) == (0.0, 0.0)


def test_mgr_manual_pca_line_does_not_affect_ecc_engine():
    mgr = SlideshowManager(queue.Queue())
    mgr.file_list = ["dummy1.tif", "dummy2.tif"]
    mgr.ref_raw = np.ones((10, 10), dtype=np.float32)
    mgr.per_frame_manual[1] = np.array([10.0, 20.0])
    mgr.ref_origin = np.array([5.0, 5.0])

    with patch(
        "rixs_app.ui.alignment_slideshow.alignment_manager.ecc_maximization_offset",
        return_value=(3.0, 4.0)
    ) as mock_ecc, patch(
        "rixs_app.ui.alignment_slideshow.alignment_manager.SlideshowManager.get_raw",
        return_value=np.ones((10, 10), dtype=np.float32)
    ):
        assert mgr.active_engine == "ECC"
        dx, dy = mgr.get_offset(1)
        assert (dx, dy) == (3.0, 4.0)
        mock_ecc.assert_called_once()

        mgr.active_engine = "PCA"
        mgr._invalidate_offset_cache(1)
        dx, dy = mgr.get_offset(1)
        assert (dx, dy) == (5.0, 15.0)


def test_mgr_zoom_init_and_reset():
    mgr = SlideshowManager(queue.Queue())
    assert not mgr.zoom_mode
    assert mgr.zoom_level == 0
    assert mgr.pan_offset_x == 0
    assert mgr.pan_offset_y == 0

    mgr.zoom_mode = True
    mgr.zoom_level = 2
    mgr.pan_offset_x = 10
    mgr.pan_offset_y = 20
    mgr.reset_view()

    assert not mgr.zoom_mode
    assert mgr.zoom_level == 0
    assert mgr.pan_offset_x == 0
    assert mgr.pan_offset_y == 0


def test_mgr_zoom_in_on_point_and_zoom_out():
    mgr = SlideshowManager(queue.Queue())
    mgr.zoom_in_on_point(cw=1000, ch=500, ix=300.0, iy=200.0, iw=1000, ih=500)
    assert mgr.zoom_level == 1
    assert mgr.pan_offset_x == 400
    assert mgr.pan_offset_y == 100

    mgr.zoom_out(cw=1000, ch=500, iw=1000, ih=500)
    assert mgr.zoom_level == 0
    assert mgr.pan_offset_x == 0
    assert mgr.pan_offset_y == 0


@patch("rixs_app.ui.alignment_slideshow.alignment_manager.SlideshowManager.get_raw")
def test_mgr_default_clamping_ceiling_percentile(mock_get_raw):
    mgr = SlideshowManager(queue.Queue())
    raw = np.arange(100, dtype=np.float32).reshape((10, 10))
    raw[0, 0] = 1000.0
    mock_get_raw.return_value = raw
    mock_path = os.path.join(os.path.dirname(__file__), "samples", "mock.tif")
    mgr.start([mock_path])

    assert mgr.intensity_min == 1.0
    assert mgr.intensity_max == 1000.0
    assert mgr.clamping_floor == 1.0
    active = raw[raw > mgr.intensity_min]
    expected_p60 = float(np.percentile(active, 60.0))
    assert mgr.clamping_ceiling == expected_p60
    assert mgr.clamping_ceiling < 1000.0


def test_handle_clamping_release(qtbot):
    from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView
    view = SlideshowView()
    qtbot.addWidget(view)

    # Call handle_clamping_change and verify timer started
    view.handle_clamping_change(2.0, 50.0)
    assert view.manager.clamping_floor == 2.0
    assert view.manager.clamping_ceiling == 50.0
    assert view._clamping_timer.isActive()

    # Call handle_clamping_release and verify timer stopped
    with patch.object(view, "_apply_clamping_change") as mock_apply:
        view.handle_clamping_release(2.0, 50.0)
        assert not view._clamping_timer.isActive()
        mock_apply.assert_called_once()

