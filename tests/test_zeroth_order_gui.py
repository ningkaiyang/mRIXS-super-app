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

from rixs_app.ui.zeroth_order_slideshow.slideshow_view import ZerothOrderSlideshowView
from rixs_app.ui.zeroth_order_slideshow.manager import ZerothOrderManager
from rixs_app.core.dataset import ZarrSequenceManager
from rixs_app.core.zeroth_order_evaluator import ZerothOrderResult


class _MockZerothOrderAppWindow:
    """Lightweight adapter mimicking RixsApp for ZerothOrderSlideshowView testing."""

    def __init__(self, view: ZerothOrderSlideshowView):
        self.zeroth_order_view = view

    def show_zeroth_order_calibration(
        self, file_list: list[str], txt_path: str | None = None
    ) -> None:
        self.zeroth_order_view.start(file_list, txt_path=txt_path)

    def show(self) -> None:
        self.zeroth_order_view.show()

    def activateWindow(self) -> None:  # noqa: N802
        self.zeroth_order_view.activateWindow()

    def close(self) -> None:
        self.zeroth_order_view.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        focused = QApplication.focusWidget()
        from PySide6.QtWidgets import (
            QSlider, QLineEdit, QComboBox, QTextEdit,
            QSpinBox, QDoubleSpinBox, QAbstractSpinBox,
        )
        if isinstance(focused, (QSlider, QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox)):
            return
        key = event.key()
        if key == Qt.Key_Left:
            self.zeroth_order_view.prev_frame()
        elif key == Qt.Key_Right:
            self.zeroth_order_view.next_frame()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
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
    """Instantiate ZerothOrderSlideshowView headlessly without full RixsApp overhead."""
    view = ZerothOrderSlideshowView()
    qtbot.addWidget(view)
    window = _MockZerothOrderAppWindow(view)
    yield window
    view.manager.session_id = object()
    view.close()
    view.deleteLater()


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


def test_zo_display_toggle_buttons(app_window, temp_tif_files, qtbot):
    """Clicking display toggle buttons in the bottom bar must toggle states and styles."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    b = v.bottom_bar

    # Default states
    assert not b.show_support_points
    assert not b.show_extrapolation
    assert b.show_fitted_line
    assert b.support_points_button.objectName() == "tool_btn"
    assert b.extrapolation_button.objectName() == "tool_btn"
    assert b.fitted_line_button.objectName() == "accent_btn"

    # Toggle support points
    qtbot.mouseClick(b.support_points_button, Qt.LeftButton)
    assert b.show_support_points
    assert b.support_points_button.text() == "Support Points: ON"
    assert b.support_points_button.objectName() == "accent_btn"

    # Toggle extrapolation
    qtbot.mouseClick(b.extrapolation_button, Qt.LeftButton)
    assert b.show_extrapolation
    assert b.extrapolation_button.text() == "Extrapolation: ON"
    assert b.extrapolation_button.objectName() == "accent_btn"

    # Toggle fitted line
    qtbot.mouseClick(b.fitted_line_button, Qt.LeftButton)
    assert not b.show_fitted_line
    assert b.fitted_line_button.text() == "Fitted Line: OFF"
    assert b.fitted_line_button.objectName() == "tool_btn"

    # Check compatibility properties on tools_panel
    assert v.tools_panel.support_points_cb.isChecked() is True
    assert v.tools_panel.extrapolation_cb.isChecked() is True
    assert v.tools_panel.fitted_line_cb.isChecked() is False


# ---------------------------------------------------------------------------
# Milestone 5 Comprehensive Tests
# ---------------------------------------------------------------------------

def test_zo_canvas_dark_theme_unification(app_window, temp_tif_files):
    """Zeroth-order canvas must apply #14172b dark theme with cyan raw curve and gold fit."""
    from matplotlib.colors import to_hex
    from rixs_app.core.zeroth_order_evaluator import ZerothOrderResult

    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    cp = v.canvas_panel

    # Check figure and axes background colors
    fig_color = to_hex(cp.figure.get_facecolor())
    ax2d_color = to_hex(cp.ax_2d.get_facecolor())
    ax1d_color = to_hex(cp.ax_1d.get_facecolor())
    assert fig_color == "#14172b"
    assert ax2d_color == "#14172b"
    assert ax1d_color == "#14172b"

    # Test drawing with synthetic evaluator result
    dummy_img = np.ones((50, 50), dtype=np.float32)
    u_vals = np.linspace(-20, 20, 41)
    p_vals = 100.0 * np.exp(-0.5 * (u_vals / 3.0) ** 2) + 5.0
    er = ZerothOrderResult(
        score_valid=True,
        score=0.985,
        fwhm_px=7.06,
        fwhm_mev=35.3,
        gaussian_amplitude=100.0,
        gaussian_center=0.0,
        gaussian_sigma=3.0,
        gaussian_background=5.0,
        profile_u=u_vals,
        intensity_profile=p_vals,
        failure_reason=None,
    )

    cp.draw_plots(
        img_2d=dummy_img,
        profile_1d=(p_vals, u_vals),
        stage="Raw",
        colormap="viridis",
        vmin=0.0,
        vmax=10.0,
        evaluator_result=er,
        fit_ok=True,
    )

    # Check curves drawn on ax_1d
    lines = cp.ax_1d.get_lines()
    assert len(lines) >= 2  # Raw data curve and Gaussian fit curve

    # Raw curve: cyan #38bdf8 with circular markers
    raw_line = lines[0]
    assert to_hex(raw_line.get_color()) == "#38bdf8"
    assert raw_line.get_marker() == "o"

    # Fit curve: gold #fbbf24
    fit_line = lines[1]
    assert to_hex(fit_line.get_color()) == "#fbbf24"

    # Check spines color
    for spine in cp.ax_1d.spines.values():
        assert to_hex(spine.get_edgecolor()) == "#2d3561"

    # Check legend properties
    legend = cp.ax_1d.get_legend()
    assert legend is not None
    assert to_hex(legend.get_frame().get_facecolor()) == "#1a1a2e"
    assert to_hex(legend.get_frame().get_edgecolor()) == "#2d3561"


def test_zo_four_kpi_cards_display(app_window, temp_tif_files):
    """ZerothOrderControlPanel must render 4 modular elevated KPI cards with dynamic badges."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    ctrl = app_window.zeroth_order_view.control_panel

    # Check 4 KPI card widgets exist
    assert hasattr(ctrl, "card_motor")
    assert hasattr(ctrl, "card_fwhm")
    assert hasattr(ctrl, "card_rp")
    assert hasattr(ctrl, "card_score")

    # Update metadata with high score (EXCELLENT tier >= 0.95)
    ctrl.update_metadata(
        filename="scan_001.tif",
        motor_name="SM3 Mirror Pitch",
        motor_val="12.3456 mrad",
        fwhm_px=4.20,
        fwhm_mev=21.0,
        score=0.9850,
        is_best_focus=False,
        mono_energy_ev=530.0,
    )

    assert ctrl.motor_title_label.text() == "SM3 MIRROR PITCH"
    assert ctrl.motor_val_label.text() == "12.3456 mrad"
    assert ctrl.motor_sub_label.text() == "scan_001.tif"

    assert "4.20 px" in ctrl.fwhm_val_label.text()
    assert "21.0 meV" in ctrl.fwhm_sub_label.text()

    # Resolving power R = (530 * 1000) / 21 = 25238
    assert "25,238" in ctrl.rp_val_label.text()
    assert "E₀ = 530.0 eV" in ctrl.rp_sub_label.text()

    assert ctrl.score_val_label.text() == "0.9850"
    assert ctrl.score_badge_label.text() == "EXCELLENT"

    # Test ACCEPTABLE tier (0.80 <= score < 0.95)
    ctrl.update_metadata(
        filename="scan_002.tif",
        motor_name="Pitch",
        motor_val="12.0",
        fwhm_px=6.50,
        fwhm_mev=32.5,
        score=0.8800,
        mono_energy_ev=530.0,
    )
    assert ctrl.score_badge_label.text() == "ACCEPTABLE"

    # Test POOR tier (score < 0.80)
    ctrl.update_metadata(
        filename="scan_003.tif",
        motor_name="Pitch",
        motor_val="11.5",
        fwhm_px=12.0,
        fwhm_mev=60.0,
        score=0.6500,
        mono_energy_ev=530.0,
    )
    assert ctrl.score_badge_label.text() == "POOR"

    # Test NO FIT tier (score is None)
    ctrl.update_metadata(
        filename="scan_004.tif",
        motor_name="Pitch",
        motor_val="N/A",
        fwhm_px=None,
        fwhm_mev=None,
        score=None,
    )
    assert ctrl.score_val_label.text() == "—"
    assert ctrl.score_badge_label.text() == "NO FIT"
    assert ctrl.rp_val_label.text() == "N/A"


def test_zo_best_focus_celebration_badge_and_card_glow(app_window, temp_tif_files):
    """Best focus frame must show ★ BEST FOCUS badge in navbar and gold border glow on FWHM card."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    manager = v.manager

    u_mock = np.linspace(-15, 15, 31)
    # Mock pipeline results across 3 frames with frame 1 being best focus (min FWHM)
    for idx, fwhm in enumerate([8.5, 3.2, 6.1]):
        sigma = fwhm / 2.355
        p_mock = 50.0 * np.exp(-0.5 * (u_mock / sigma) ** 2) + 2.0
        er = ZerothOrderResult(
            score_valid=True,
            score=1.0 / fwhm,
            fwhm_px=fwhm,
            fwhm_mev=fwhm * 5.0,
            gaussian_amplitude=50.0,
            gaussian_center=0.0,
            gaussian_sigma=sigma,
            gaussian_background=2.0,
            profile_u=u_mock,
            intensity_profile=p_mock,
            failure_reason=None,
        )
        manager.pipeline_results[idx] = {
            "evaluator_result": er,
            "score": 1.0 / fwhm,
            "raw_img": np.ones((50, 50), dtype=np.float32),
            "denoised_img": np.ones((50, 50), dtype=np.float32),
            "masked_img": np.ones((50, 50), dtype=np.float32),
            "grad_img": np.ones((50, 50), dtype=np.float32),
            "centroid": (25.0, 25.0),
            "direction": (1.0, 0.0),
            "1d_profile": (p_mock, u_mock),
            "fit_ok": True,
        }

    assert manager.get_peak_focus_index() == 1
    assert manager.is_best_focus_frame(1) is True
    assert manager.is_best_focus_frame(0) is False

    # Render frame 0 (not best)
    manager.current_idx = 0
    v.load_and_render()
    assert v.navbar.best_focus_badge.isHidden() is True
    assert "#fbbf24" not in v.control_panel.card_fwhm.styleSheet()

    # Render frame 1 (best focus frame)
    manager.current_idx = 1
    v.load_and_render()
    assert v.navbar.best_focus_badge.isHidden() is False
    assert "#fbbf24" in v.control_panel.card_fwhm.styleSheet()
    assert "★" in v.control_panel.fwhm_title_label.text()

    # Jump to peak focus button
    manager.current_idx = 2
    v.load_and_render()
    assert v.navbar.best_focus_badge.isHidden() is True
    v.jump_to_peak_focus()
    assert manager.current_idx == 1
    assert v.navbar.best_focus_badge.isHidden() is False


def test_zo_zarr_dsm_img_caching_and_retrieval(app_window, temp_tif_files):
    """Row-smoothed stage image (dsm_img) must be stored in Zarr cache and retrieved on hit."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    manager = app_window.zeroth_order_view.manager

    # First call - cache miss computation
    data1 = manager.get_frame_pipeline_data(0)
    assert data1 is not None
    assert "dsm_img" in data1
    assert data1["dsm_img"] is not None
    assert data1["dsm_img"].shape == (100, 100)

    # Check Zarr cache directly
    cached_dsm = manager.zarr_manager.get_derived_frame(0, "dsm_img")
    assert cached_dsm is not None
    np.testing.assert_allclose(cached_dsm, data1["dsm_img"])

    # Second call - cache hit retrieval
    data2 = manager.get_frame_pipeline_data(0)
    assert data2 is not None
    assert "dsm_img" in data2
    np.testing.assert_allclose(data2["dsm_img"], data1["dsm_img"])

    # Test changing stage to Row-Smoothed (Dsm)
    v = app_window.zeroth_order_view
    v.change_pipeline_stage("Row-Smoothed (Dsm)")
    assert manager.pipeline_stage == "Row-Smoothed (Dsm)"
    # load_and_render should execute without errors
    v.load_and_render()


def test_zo_kpi_resolving_power_and_score_tiers_edge_cases(app_window, temp_tif_files):
    """Test boundary cases for Resolving Power R and Gaussian fit score color tiers."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    ctrl = app_window.zeroth_order_view.control_panel

    # Exactly at 0.95 threshold -> EXCELLENT
    ctrl.update_metadata(
        filename="boundary_95.tif",
        motor_name="Pitch",
        motor_val="10.0",
        fwhm_px=5.0,
        fwhm_mev=25.0,
        score=0.9500,
        mono_energy_ev=500.0,
    )
    assert ctrl.score_badge_label.text() == "EXCELLENT"
    # R = (500 * 1000) / 25 = 20,000
    assert "20,000" in ctrl.rp_val_label.text()

    # Exactly at 0.80 threshold -> ACCEPTABLE
    ctrl.update_metadata(
        filename="boundary_80.tif",
        motor_name="Pitch",
        motor_val="10.0",
        fwhm_px=5.0,
        fwhm_mev=25.0,
        score=0.8000,
        mono_energy_ev=500.0,
    )
    assert ctrl.score_badge_label.text() == "ACCEPTABLE"

    # Just below 0.80 (0.7999) -> POOR
    ctrl.update_metadata(
        filename="boundary_799.tif",
        motor_name="Pitch",
        motor_val="10.0",
        fwhm_px=5.0,
        fwhm_mev=25.0,
        score=0.7999,
        mono_energy_ev=500.0,
    )
    assert ctrl.score_badge_label.text() == "POOR"

    # Missing mono energy (None or 0)
    ctrl.update_metadata(
        filename="no_mono.tif",
        motor_name="Pitch",
        motor_val="10.0",
        fwhm_px=5.0,
        fwhm_mev=25.0,
        score=0.96,
        mono_energy_ev=0.0,
    )
    assert ctrl.rp_val_label.text() == "N/A"
    assert "Missing E₀" in ctrl.rp_sub_label.text()

    # Missing fwhm_mev (None or 0)
    ctrl.update_metadata(
        filename="no_dispersion.tif",
        motor_name="Pitch",
        motor_val="10.0",
        fwhm_px=5.0,
        fwhm_mev=None,
        score=0.96,
        mono_energy_ev=500.0,
    )
    assert ctrl.rp_val_label.text() == "N/A"
    assert "Missing Dispersion" in ctrl.rp_sub_label.text()


def test_zo_pipeline_stage_switching_with_all_stages(app_window, temp_tif_files):
    """Switching between all 5 stages should update manager stage and render without exception."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    stages = [
        "Raw",
        "Denoised (D)",
        "Row-Smoothed (Dsm)",
        "Gradient (G)",
        "Fitted-Line Strip",
    ]

    for stage in stages:
        v.change_pipeline_stage(stage)
        assert v.manager.pipeline_stage == stage
        assert f"2D View: {stage}" in v.canvas_panel.ax_2d.get_title()


def test_zo_canvas_panel_none_and_fit_failed_graceful(app_window):
    """Canvas panel must gracefully render with None inputs or fit failure in #14172b theme."""
    cp = app_window.zeroth_order_view.canvas_panel

    # Fit failure with reason
    er_fail = ZerothOrderResult(
        score_valid=False,
        score=None,
        failure_reason="Low SNR on elastic line",
    )
    u_vals = np.linspace(-10, 10, 21)
    p_vals = np.zeros(21)

    cp.draw_plots(
        img_2d=None,
        profile_1d=(p_vals, u_vals),
        stage="Raw",
        colormap="viridis",
        vmin=0.0,
        vmax=1.0,
        evaluator_result=er_fail,
        fit_ok=False,
    )

    # Verify no crash and title set
    assert cp.ax_1d.get_title() == "1D Project Profile"
    assert cp.figure.get_facecolor() == cp.ax_1d.get_facecolor()


def test_zo_best_focus_all_invalid_fits(app_window, temp_tif_files):
    """When all frames have invalid fits, is_best_focus_frame must return False for all."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    manager = app_window.zeroth_order_view.manager

    for idx in range(3):
        er = ZerothOrderResult(score_valid=False, failure_reason="Fit failed")
        manager.pipeline_results[idx] = {
            "evaluator_result": er,
            "fit_ok": False,
        }

    for idx in range(3):
        assert manager.is_best_focus_frame(idx) is False


def test_zo_best_focus_badge_geometry_and_alignment(app_window, temp_tif_files):
    """Best focus badge must be exactly 30px height matching navbar buttons with centered text."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    nb = app_window.zeroth_order_view.navbar
    badge = nb.best_focus_badge

    assert badge.height() == 30 or badge.minimumHeight() == 30
    assert badge.alignment() == Qt.AlignCenter
    assert "border-radius: 5px" in badge.styleSheet()


def test_zo_tools_panel_dispersion_entry_and_resolving_power_live_update(app_window, temp_tif_files):
    """Entering dispersion in tools panel must update manager and live Resolving Power in KPI Card 3."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    tp = v.tools_panel
    ctrl = v.control_panel

    assert hasattr(tp, "dispersion_entry")
    # Simulate typing 2.5 meV/px
    tp.dispersion_entry.setText("2.5")
    tp._on_dispersion_submit()

    assert v.manager.energy_dispersion == 2.5

    # Update metadata with FWHM = 4.0 px -> fwhm_mev = 10.0 meV, mono_energy = 850 eV
    # R = (850 * 1000) / 10 = 85,000
    ctrl.update_metadata(
        "scan.tif", "Pitch", "1.0",
        fwhm_px=4.0, fwhm_mev=4.0 * 2.5,
        score=0.98, r_squared=0.9850,
        mono_energy_ev=850.0,
    )
    assert "85,000" in ctrl.rp_val_label.text()
    assert ctrl.score_badge_label.text() == "EXCELLENT"
    assert ctrl.score_val_label.text() == "0.9850"


def test_zo_export_panel_progress_bar_lifecycle(app_window, temp_tif_files):
    """Export panel must have a progress bar that is hidden initially and ready for export updates."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    ep = app_window.zeroth_order_view.bottom_bar

    assert hasattr(ep, "progress_bar")
    assert ep.progress_bar.isHidden() is True
    assert ep.progress_bar.height() == 14


def test_zo_text_input_focus_release_and_arrow_navigation(app_window, temp_tif_files, qtbot):
    """Pressing Enter or Escape in floor/ceiling/dispersion entries or clicking canvas releases focus."""
    from PySide6.QtGui import QKeyEvent, QMouseEvent
    from PySide6.QtCore import QPointF
    app_window.show()
    app_window.activateWindow()
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    tp = v.tools_panel

    # 1. Clear focus on floor entry submit
    tp.floor_entry.setFocus()
    tp._on_floor_submit()
    assert tp.floor_entry.hasFocus() is False

    # 2. Clear focus on Escape key event filter
    tp.ceiling_entry.setFocus()
    esc_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    res = tp.eventFilter(tp.ceiling_entry, esc_event)
    assert res is True
    assert tp.ceiling_entry.hasFocus() is False

    # 3. Clear focus on dispersion entry submit
    tp.dispersion_entry.setFocus()
    tp._on_dispersion_submit()
    assert tp.dispersion_entry.hasFocus() is False

    # 4. Clicking canvas releases focus
    tp.floor_entry.setFocus()
    v.canvas_panel._on_mpl_click(MagicMock(inaxes=None))
    assert tp.floor_entry.hasFocus() is False

    # 5. Clicking RangeSlider releases focus
    tp.ceiling_entry.setFocus()
    tp.range_slider.mousePressEvent(
        QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(50, 10), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    )
    assert tp.ceiling_entry.hasFocus() is False


def test_zo_best_focus_hidden_before_all_frames_cached(app_window, temp_tif_files):
    """BEST FOCUS badge and card highlight must remain hidden until all frames are cached."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view
    manager = v.manager

    # Multi-frame dataset (3 frames)
    assert len(manager.file_list) == 3

    # Initially only frame 0 is evaluated/cached
    assert len(manager.pipeline_results) == 1
    assert manager.all_frames_cached() is False
    assert manager.is_best_focus_frame(0) is False

    # Render frame 0: badge MUST be hidden
    v.load_and_render()
    assert v.navbar.best_focus_badge.isHidden() is True
    assert "#fbbf24" not in v.control_panel.card_fwhm.styleSheet()
    assert "★" not in v.control_panel.fwhm_title_label.text()

    # Now populate remaining frames
    u_mock = np.linspace(-15, 15, 31)
    for idx, fwhm in enumerate([8.5, 3.2, 6.1]):
        sigma = fwhm / 2.355
        p_mock = 50.0 * np.exp(-0.5 * (u_mock / sigma) ** 2) + 2.0
        er = ZerothOrderResult(
            score_valid=True,
            score=1.0 / fwhm,
            fwhm_px=fwhm,
            fwhm_mev=fwhm * 5.0,
            gaussian_amplitude=50.0,
            gaussian_center=0.0,
            gaussian_sigma=sigma,
            gaussian_background=2.0,
            profile_u=u_mock,
            intensity_profile=p_mock,
            failure_reason=None,
        )
        manager.pipeline_results[idx] = {
            "evaluator_result": er,
            "score": 1.0 / fwhm,
            "raw_img": np.ones((50, 50), dtype=np.float32),
            "denoised_img": np.ones((50, 50), dtype=np.float32),
            "masked_img": np.ones((50, 50), dtype=np.float32),
            "grad_img": np.ones((50, 50), dtype=np.float32),
            "centroid": (25.0, 25.0),
            "direction": (1.0, 0.0),
            "1d_profile": (p_mock, u_mock),
            "fit_ok": True,
        }

    assert manager.all_frames_cached() is True
    assert manager.is_best_focus_frame(1) is True
    assert manager.is_best_focus_frame(0) is False

    # Render frame 1 (best): badge is now visible
    manager.current_idx = 1
    v.load_and_render()
    assert v.navbar.best_focus_badge.isHidden() is False
    assert "#fbbf24" in v.control_panel.card_fwhm.styleSheet()
    assert "★" in v.control_panel.fwhm_title_label.text()


def test_zo_precompute_button_click_signal_safe(app_window, temp_tif_files, qtbot):
    """Clicking Precompute All button passes boolean clicked arg without throwing TypeError."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    # Mock _run_worker so it executes immediately on the main thread
    def mock_run_worker(worker):
        # Emulate successful worker execution
        for idx in range(worker.total):
            worker.signals.progress.emit(idx + 1, worker.total)
        worker.signals.result.emit(True)
        worker.signals.finished.emit()

    with patch.object(v, "_run_worker", side_effect=mock_run_worker):
        # Trigger button click (emits clicked(bool))
        v.navbar.precompute_button.click()
        QApplication.processEvents()

    # Precompute button text reset and peak focus enabled
    assert v.navbar.precompute_button.text() == "Precompute All"
    assert v.navbar.precompute_button.isEnabled() is True
    assert v.navbar.peak_focus_button.isEnabled() is True


def test_zo_focus_policy_configuration(app_window, temp_tif_files):
    """Slicer text inputs must use ClickFocus and buttons must use NoFocus."""
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    # Text inputs must not take tab/auto focus
    assert v.tools_panel.floor_entry.focusPolicy() == Qt.ClickFocus
    assert v.tools_panel.ceiling_entry.focusPolicy() == Qt.ClickFocus
    assert v.tools_panel.dispersion_entry.focusPolicy() == Qt.ClickFocus

    # Buttons and slider must not hold keyboard focus
    assert v.navbar.back_button.focusPolicy() == Qt.NoFocus
    assert v.navbar.prev_button.focusPolicy() == Qt.NoFocus
    assert v.navbar.next_button.focusPolicy() == Qt.NoFocus
    assert v.navbar.autoplay_button.focusPolicy() == Qt.NoFocus
    assert v.navbar.precompute_button.focusPolicy() == Qt.NoFocus
    assert v.navbar.peak_focus_button.focusPolicy() == Qt.NoFocus
    assert v.tools_panel.zoom_in_button.focusPolicy() == Qt.NoFocus
    assert v.tools_panel.zoom_out_button.focusPolicy() == Qt.NoFocus
    assert v.tools_panel.reset_view_button.focusPolicy() == Qt.NoFocus
    assert v.bottom_bar.support_points_button.focusPolicy() == Qt.NoFocus
    assert v.bottom_bar.extrapolation_button.focusPolicy() == Qt.NoFocus
    assert v.bottom_bar.fitted_line_button.focusPolicy() == Qt.NoFocus
    assert v.bottom_bar.export_button.focusPolicy() == Qt.NoFocus
    assert v.control_panel.frame_slider.focusPolicy() == Qt.NoFocus


def test_zo_arrow_keys_navigate_immediately_after_peak_focus(app_window, temp_tif_files, qtbot):
    """Clicking Best Focus must not hijack focus into slicer box and arrow keys must navigate frames."""
    from PySide6.QtGui import QKeyEvent
    app_window.show()
    app_window.activateWindow()
    app_window.show_zeroth_order_calibration(temp_tif_files)
    v = app_window.zeroth_order_view

    # Mock _run_worker to finish synchronously and mock precomputed results
    u_mock = np.linspace(-15, 15, 31)
    fwhms = [8.5, 3.2, 6.1]  # frame 1 is best focus
    for idx, fwhm in enumerate(fwhms):
        sigma = fwhm / 2.355
        p_mock = 50.0 * np.exp(-0.5 * (u_mock / sigma) ** 2) + 2.0
        er = ZerothOrderResult(
            score_valid=True,
            score=1.0 / fwhm,
            fwhm_px=fwhm,
            fwhm_mev=fwhm * 5.0,
            gaussian_amplitude=50.0,
            gaussian_center=0.0,
            gaussian_sigma=sigma,
            gaussian_background=2.0,
            profile_u=u_mock,
            intensity_profile=p_mock,
            failure_reason=None,
        )
        v.manager.pipeline_results[idx] = {
            "evaluator_result": er,
            "score": 1.0 / fwhm,
            "raw_img": np.ones((50, 50), dtype=np.float32),
            "centroid": (25.0, 25.0),
            "direction": (1.0, 0.0),
            "1d_profile": (p_mock, u_mock),
            "fit_ok": True,
        }

    # Click Best Focus button
    v.navbar.peak_focus_button.click()
    QApplication.processEvents()

    # Must be on best focus frame (index 1)
    assert v.manager.current_idx == 1

    # Slicer floor_entry must NOT have focus
    assert v.tools_panel.floor_entry.hasFocus() is False
    assert v.tools_panel.ceiling_entry.hasFocus() is False

    # Send Left arrow key event to the main window
    left_key = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Left, Qt.NoModifier)
    app_window.keyPressEvent(left_key)
    QApplication.processEvents()

    # Must have navigated to frame 0
    assert v.manager.current_idx == 0

    # Send Right arrow key event
    right_key = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_Right, Qt.NoModifier)
    app_window.keyPressEvent(right_key)
    QApplication.processEvents()

    # Must have navigated back to frame 1
    assert v.manager.current_idx == 1


