"""Comprehensive PySide6 GUI unit tests for Detector Dark Frame Calibration Studio.

Covers:
- DarkDiagnosticsWorker execution, signal emissions, error propagation, and progress.
- DarkCalibrationView UI initialization, drag-and-drop file/folder ingest, natural ordering.
- Dual log-scale diagnostic histogram rendering, axis dark theme styling.
- Interactive threshold sliders, live cutlines, and instant KPI recalculation.
- 1-Click calibration persistence to appdata/dark_calibration/ and manifest verification.
- Navigation back button callback and Co-Pilot button docking.
- Cleanup and figure disposal.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import weakref

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QThreadPool
from PySide6.QtWidgets import QApplication, QPushButton, QSlider

from rixs_app.core import calibration_store
from rixs_app.core.photon_clustering import DarkDiagnostics, compute_dark_diagnostics
from rixs_app.ui.dark_calibration.dark_cal_view import DarkCalibrationView
from rixs_app.ui.dark_calibration.workers import DarkDiagnosticsWorker, WorkerSignals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def synthetic_dark_frames(tmp_path):
    """Generate 6 synthetic 48x48 dark frame TIFF files with controlled noise and hot pixels."""
    dark_dir = tmp_path / "dark_run_01"
    dark_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    np.random.seed(101)
    base_dark = 100.0 + np.random.normal(0, 1.5, size=(48, 48)).astype(np.float32)

    for i in range(6):
        frame = base_dark + np.random.normal(0, 2.0, size=(48, 48)).astype(np.float32)
        # Hot pixel at (12, 12) with elevated stddev and excursion
        frame[12, 12] += 80.0
        # Warm pixel at (24, 24)
        frame[24, 24] += 25.0
        p = dark_dir / f"dark_frame_{i + 1:03d}.tif"
        tifffile.imwrite(str(p), frame)
        paths.append(str(p))

    return paths


@pytest.fixture
def populated_diagnostics():
    """Precomputed DarkDiagnostics fixture for direct view state testing."""
    np.random.seed(42)
    h, w = 32, 32
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    stddev = np.abs(np.random.normal(10.0, 5.0, size=(h, w))).astype(np.float32)
    residual = np.abs(np.random.normal(20.0, 10.0, size=(h, w))).astype(np.float32)

    # Inject out-of-bounds hot pixels
    stddev[0, 0] = 95.0
    residual[0, 1] = 120.0

    return DarkDiagnostics(
        med_dark=med_dark,
        per_pixel_stddev=stddev,
        pct93_residual=residual,
        dark_frame_count=10,
    )


# ---------------------------------------------------------------------------
# Worker Tests
# ---------------------------------------------------------------------------

def test_dark_diagnostics_worker_success(qapp, synthetic_dark_frames):
    """Verify DarkDiagnosticsWorker computes diagnostics and emits all expected Qt signals."""
    worker = DarkDiagnosticsWorker(synthetic_dark_frames, tail_pct=0.9333)

    results: list[DarkDiagnostics] = []
    progress_updates: list[tuple[int, int]] = []
    progress_messages: list[str] = []
    finished_called = []

    worker.signals.result.connect(results.append)
    worker.signals.progress.connect(lambda c, t: progress_updates.append((c, t)))
    worker.signals.progress_msg.connect(progress_messages.append)
    worker.signals.finished.connect(lambda: finished_called.append(True))

    worker.run()

    assert len(results) == 1
    diag = results[0]
    assert isinstance(diag, DarkDiagnostics)
    assert diag.dark_frame_count == len(synthetic_dark_frames)
    assert diag.med_dark.shape == (48, 48)
    assert diag.per_pixel_stddev.shape == (48, 48)
    assert diag.pct93_residual.shape == (48, 48)

    assert len(progress_updates) >= len(synthetic_dark_frames)
    assert progress_updates[-1][1] == 100
    assert any("[1/3]" in msg for msg in progress_messages)
    assert any("[2/3]" in msg for msg in progress_messages)
    assert len(finished_called) == 1


def test_dark_cal_view_generate_button_and_progress_flow(qapp, qtbot, synthetic_dark_frames):
    """Verify DarkCalibrationView updates generate button text and progress labels across stages."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()

    view._on_files_dropped(synthetic_dark_frames)
    assert view.generate_btn.text() == "▶ Generate Histograms"
    assert view.generate_btn.isEnabled()

    # Trigger generation
    view._on_generate_clicked()
    assert view.generate_btn.text() == "⏳ Processing Histograms..."
    assert not view.generate_btn.isEnabled()
    assert view.progress_bar.isVisible()
    assert "[1/3]" in view.progress_msg_label.text()

    # Worker progress updates
    view._on_worker_progress(50, 100)
    assert view.progress_bar.value() == 50
    view._on_worker_msg("[2/3] Computing noise statistics (chunk 1/1)...")
    assert "[2/3]" in view.progress_msg_label.text()

    # Diagnostics ready (Stage 3 executed)
    diag = compute_dark_diagnostics(synthetic_dark_frames)
    view._on_diagnostics_ready(diag)
    assert view.progress_bar.value() == 100
    assert view.generate_btn.text() == "▶ Generate Histograms"
    assert view.generate_btn.isEnabled()
    assert not view.progress_bar.isVisible()
    assert not view.progress_msg_label.isVisible()

    # Worker finished signal is safe no-op
    view._on_worker_finished()
    assert view._current_worker is None


def test_dark_diagnostics_worker_empty_paths(qapp):
    """Verify DarkDiagnosticsWorker emits error signal when dark_paths is empty."""
    worker = DarkDiagnosticsWorker([])

    errors: list[str] = []
    finished_called = []

    worker.signals.error.connect(errors.append)
    worker.signals.finished.connect(lambda: finished_called.append(True))

    worker.run()

    assert len(errors) == 1
    assert "No dark frame paths" in errors[0]
    assert len(finished_called) == 1


def test_dark_diagnostics_worker_mismatched_shapes(qapp, tmp_path):
    """Verify DarkDiagnosticsWorker emits error signal when frame shapes mismatch."""
    p1 = tmp_path / "f1.tif"
    p2 = tmp_path / "f2.tif"
    tifffile.imwrite(str(p1), np.zeros((32, 32), dtype=np.float32))
    tifffile.imwrite(str(p2), np.zeros((48, 48), dtype=np.float32))

    worker = DarkDiagnosticsWorker([str(p1), str(p2)])
    errors: list[str] = []
    finished = []

    worker.signals.error.connect(errors.append)
    worker.signals.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(errors) == 1
    assert "mismatch" in errors[0].lower() or "shape" in errors[0].lower()
    assert len(finished) == 1


# ---------------------------------------------------------------------------
# View Initialization and Layout Tests
# ---------------------------------------------------------------------------

def test_dark_cal_view_init_state(qapp, qtbot):
    """Verify DarkCalibrationView initial widget states and default parameters."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    assert view.dark_frame_count == 0
    assert not view.generate_btn.isEnabled()
    assert not view.save_btn.isEnabled()
    assert view.stddev_slider.value() == 40
    assert view.absdev_slider.value() == 60
    assert "0" in view.frame_count_label.text()
    assert "93.33%" in view.tail_ratio_label.text()


def test_dark_cal_view_back_navigation(qapp, qtbot):
    """Verify clicking ❮ Back to Home invokes the on_back callback."""
    mock_back = MagicMock()
    view = DarkCalibrationView(on_back=mock_back)
    qtbot.addWidget(view)

    view._back_btn.click()
    mock_back.assert_called_once()


def test_dark_cal_view_copilot_docking(qapp, qtbot):
    """Verify Co-Pilot button docking and reparenting in navbar."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    btn1 = QPushButton("🤖 Co-Pilot")
    qtbot.addWidget(btn1)
    view.set_copilot_button(btn1)
    assert view.isAncestorOf(btn1)
    assert not btn1.isHidden()

    # Reparenting a second button replaces the first cleanly
    btn2 = QPushButton("🤖 Co-Pilot New")
    qtbot.addWidget(btn2)
    view.set_copilot_button(btn2)
    assert view.isAncestorOf(btn2)


# ---------------------------------------------------------------------------
# File Ingestion Tests
# ---------------------------------------------------------------------------

def test_dark_cal_view_folder_drop(qapp, qtbot, synthetic_dark_frames):
    """Verify dropping a folder populates file list and enables generate button."""
    dark_dir = str(Path(synthetic_dark_frames[0]).parent)

    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_files_dropped([dark_dir])

    assert view.dark_frame_count == len(synthetic_dark_frames)
    assert view.file_list_widget.count() == len(synthetic_dark_frames)
    assert view.generate_btn.isEnabled()
    assert str(len(synthetic_dark_frames)) in view.frame_count_label.text()


def test_dark_cal_view_individual_files_drop(qapp, qtbot, synthetic_dark_frames):
    """Verify dropping individual file paths populates files in natural order."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    # Pass in reverse order to test natural sorting
    reversed_paths = list(reversed(synthetic_dark_frames))
    view._on_files_dropped(reversed_paths)

    assert view.dark_frame_count == len(synthetic_dark_frames)
    # First item in list widget should be frame 001
    assert "001" in view.file_list_widget.item(0).text()
    assert view.generate_btn.isEnabled()


def test_dark_cal_view_empty_and_invalid_drops(qapp, qtbot, tmp_path):
    """Verify dropping empty directory or non-TIFF files leaves view in disabled state."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a tif")

    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_files_dropped([str(empty_dir), str(text_file)])

    assert view.dark_frame_count == 0
    assert view.file_list_widget.count() == 0
    assert not view.generate_btn.isEnabled()


def test_dark_cal_view_clear_button(qapp, qtbot, synthetic_dark_frames):
    """Verify clear button resets file list and canvas state."""
    dark_dir = str(Path(synthetic_dark_frames[0]).parent)

    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_files_dropped([dark_dir])
    assert view.dark_frame_count > 0

    view.clear_btn.click()
    assert view.dark_frame_count == 0
    assert view.file_list_widget.count() == 0
    assert not view.generate_btn.isEnabled()
    assert not view.save_btn.isEnabled()


# ---------------------------------------------------------------------------
# Diagnostics & Histogram Interaction Tests
# ---------------------------------------------------------------------------

def test_dark_cal_view_diagnostics_ready_and_histograms(qapp, qtbot, populated_diagnostics):
    """Verify receiving DarkDiagnostics draws log-scale histograms and updates cutlines."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_diagnostics_ready(populated_diagnostics)

    assert view.save_btn.isEnabled()
    assert view._std_cutline is not None
    assert view._res_cutline is not None

    # Check axes labels and scale
    assert view.ax_std.get_yscale() == "log"
    assert view.ax_res.get_yscale() == "log"


def test_dark_cal_view_slider_changes_and_kpi_updates(qapp, qtbot, populated_diagnostics):
    """Verify slider adjustments dynamically recalculate survival percentages."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_diagnostics_ready(populated_diagnostics)

    # Tight thresholds: fewer surviving pixels
    view._on_stddev_slider_changed(5.0)
    view._on_absdev_slider_changed(10.0)

    t1_text = view.tier1_kpi_label.text()
    t2_text = view.tier2_kpi_label.text()
    final_text = view.final_mask_kpi_label.text()

    assert "% surviving" in t1_text
    assert "% surviving" in t2_text
    assert "Final Mask:" in final_text

    # Extremely high thresholds: 100% surviving
    view._on_stddev_slider_changed(150.0)
    view._on_absdev_slider_changed(200.0)

    assert "100.00%" in view.tier1_kpi_label.text()
    assert "100.00%" in view.tier2_kpi_label.text()
    assert "100.00%" in view.final_mask_kpi_label.text()


def test_dark_cal_view_slider_zero_boundary(qapp, qtbot, populated_diagnostics):
    """Verify 0 threshold boundary rejects all pixels without crashing."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_diagnostics_ready(populated_diagnostics)

    view._on_stddev_slider_changed(0.0)
    view._on_absdev_slider_changed(0.0)

    assert "0.00%" in view.tier1_kpi_label.text()
    assert "0.00%" in view.tier2_kpi_label.text()
    assert "0.00%" in view.final_mask_kpi_label.text()


# ---------------------------------------------------------------------------
# Worker Integration via GUI
# ---------------------------------------------------------------------------

def test_dark_cal_view_generate_clicked_worker_lifecycle(qapp, qtbot, synthetic_dark_frames):
    """Verify clicking generate button runs worker and updates view upon completion."""
    dark_dir = str(Path(synthetic_dark_frames[0]).parent)

    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_files_dropped([dark_dir])
    assert view.generate_btn.isEnabled()

    # Directly trigger generate
    view._on_generate_clicked()

    assert view._current_worker is not None

    # Wait for worker thread to complete and GUI signals to be processed
    qtbot.waitUntil(lambda: view._diagnostics is not None, timeout=5000)

    assert view.save_btn.isEnabled()
    assert view._current_worker is None


def test_dark_cal_view_worker_error_handling(qapp, qtbot):
    """Verify worker error displays formatted message in UI without crashing."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_worker_error("Simulated I/O failure")
    assert "Simulated I/O failure" in view.progress_msg_label.text()


# ---------------------------------------------------------------------------
# Calibration Persistence Tests
# ---------------------------------------------------------------------------

def test_dark_cal_view_save_calibration(qapp, qtbot, synthetic_dark_frames, tmp_path):
    """Verify 1-click save calibration writes valid TIFFs and manifest to calibration store."""
    cal_store_dir = tmp_path / "appdata" / "dark_calibration"

    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", cal_store_dir):
        view = DarkCalibrationView()
        qtbot.addWidget(view)

        view._on_files_dropped(synthetic_dark_frames)
        diag = compute_dark_diagnostics(synthetic_dark_frames)
        view._on_diagnostics_ready(diag)

        view._on_stddev_slider_changed(40.0)
        view._on_absdev_slider_changed(60.0)

        # Click save
        view.save_btn.click()

        # Check persistence
        assert calibration_store.has_calibration(cal_dir=cal_store_dir)
        med_dark, mask, record = calibration_store.load_calibration(cal_dir=cal_store_dir)

        assert med_dark.shape == (48, 48)
        assert mask.shape == (48, 48)
        assert record.dark_frame_count == len(synthetic_dark_frames)
        assert record.stddev_thresh == 40.0
        assert record.absdev_thresh == 60.0
        assert "Saved calibration" in view.save_status_label.text()


# ---------------------------------------------------------------------------
# Cleanup & Disposal Tests
# ---------------------------------------------------------------------------

def test_dark_cal_view_cleanup_and_close(qapp, qtbot, populated_diagnostics):
    """Verify cleanup and closeEvent properly tear down Matplotlib figures."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)

    view._on_diagnostics_ready(populated_diagnostics)
    assert view.figure is not None

    view.cleanup()
    assert view._current_worker is None
    view.close()


def test_dark_cal_canvas_hide_show_repaint(qapp, qtbot, populated_diagnostics):
    """Verify hiding and showing DarkCalibrationView keeps canvas alive and able to draw."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()

    # Initial load and draw
    view._on_diagnostics_ready(populated_diagnostics)
    assert view.canvas.isVisible()

    # Simulate navigating away (hideEvent)
    view.hide()
    assert getattr(view.canvas, "_destroyed", False) is False

    # Simulate navigating back (showEvent)
    view.show()
    assert getattr(view.canvas, "_destroyed", False) is False

    # Adjust thresholds; canvas should draw_idle without error
    view._on_stddev_slider_changed(30.0)
    assert view._std_cutline is not None

    view.cleanup()
    view.close()


def test_dark_cal_navigation_stack_transition(qapp, qtbot):
    """Verify RixsApp stacked navigation Home -> Dark Cal -> Home -> Dark Cal keeps canvas valid."""
    from rixs_app.main import RixsApp
    app_window = RixsApp(show_window=False)
    qtbot.addWidget(app_window)

    assert app_window._stack.currentIndex() == 0  # Home

    # Navigate to Dark Cal
    app_window.show_dark_calibration()
    assert app_window._stack.currentIndex() == 1
    assert getattr(app_window.dark_cal_view.canvas, "_destroyed", False) is False

    # Navigate back to Home
    app_window.show_home()
    assert app_window._stack.currentIndex() == 0

    # Navigate to Dark Cal again
    app_window.show_dark_calibration()
    assert app_window._stack.currentIndex() == 1
    assert getattr(app_window.dark_cal_view.canvas, "_destroyed", False) is False

    app_window.close()


# ---------------------------------------------------------------------------
# Consolidated Adversarial & Stress Tests
# ---------------------------------------------------------------------------

def test_adversarial_rapid_slider_updates(qapp, qtbot, populated_diagnostics):
    """Stress test rapid slider movements on StdDev and Residual sliders with KPI parity."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()

    view._on_diagnostics_ready(populated_diagnostics)
    qapp.processEvents()

    assert view._std_cutline is not None
    assert view._res_cutline is not None

    t0 = time.perf_counter()
    for i in range(50):
        std_val = (i % 80) + 5
        res_val = (i % 120) + 10
        view._on_stddev_slider_changed(float(std_val))
        view._on_absdev_slider_changed(float(res_val))

    qapp.processEvents()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0

    final_std = 35.0
    final_res = 55.0
    view._on_stddev_slider_changed(final_std)
    view._on_absdev_slider_changed(final_res)
    qapp.processEvents()

    m_std = (populated_diagnostics.per_pixel_stddev < final_std) & np.isfinite(populated_diagnostics.per_pixel_stddev)
    m_res = (populated_diagnostics.pct93_residual < final_res) & np.isfinite(populated_diagnostics.pct93_residual)
    m_final = m_std & m_res

    total_px = populated_diagnostics.per_pixel_stddev.size
    expected_final_pct = (np.count_nonzero(m_final) / total_px) * 100.0
    assert f"{expected_final_pct:.2f}%" in view.final_mask_kpi_label.text()
    assert view._std_cutline.get_xdata()[0] == final_std
    assert view._res_cutline.get_xdata()[0] == final_res


def test_adversarial_slider_extreme_and_negative_inputs(qapp, qtbot, populated_diagnostics):
    """Stress test boundary, negative, and extreme out-of-range slider inputs."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()
    view._on_diagnostics_ready(populated_diagnostics)

    view._on_stddev_slider_changed(-10.0)
    view._on_absdev_slider_changed(-5.0)
    qapp.processEvents()
    assert "0.00%" in view.final_mask_kpi_label.text()

    view._on_stddev_slider_changed(1_000_000.0)
    view._on_absdev_slider_changed(1_000_000.0)
    qapp.processEvents()
    assert "100.00%" in view.final_mask_kpi_label.text()


def test_adversarial_concurrent_worker_triggers(qapp, qtbot, synthetic_dark_frames):
    """Test rapid-fire triggering of generate histograms worker."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()

    view._on_files_dropped(synthetic_dark_frames)
    assert view.generate_btn.isEnabled()

    for _ in range(3):
        view._on_generate_clicked()

    qtbot.waitUntil(lambda: view._diagnostics is not None, timeout=8000)
    qapp.processEvents()

    assert view.save_btn.isEnabled()
    assert view._diagnostics.dark_frame_count == len(synthetic_dark_frames)


def test_adversarial_clear_and_cleanup_during_active_worker(qapp, qtbot, synthetic_dark_frames):
    """Verify calling clear_files() or cleanup() while worker is running is safe."""
    view = DarkCalibrationView()
    qtbot.addWidget(view)
    view.show()

    view._on_files_dropped(synthetic_dark_frames)
    view._on_generate_clicked()

    assert view._current_worker is not None

    view._clear_files()
    assert view.dark_frame_count == 0
    assert view._diagnostics is None

    QThreadPool.globalInstance().waitForDone(5000)
    qapp.processEvents()

    view.cleanup()
    assert view._current_worker is None


def test_adversarial_zero_surviving_pixels_calibration_save(qapp, qtbot, populated_diagnostics, tmp_path):
    """Stress test 0 surviving pixels (100% rejection) atomic persistence and loading."""
    cal_store_dir = tmp_path / "appdata" / "zero_surviving_cal"

    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", cal_store_dir):
        view = DarkCalibrationView()
        qtbot.addWidget(view)
        view.show()

        view._on_diagnostics_ready(populated_diagnostics)

        view._on_stddev_slider_changed(0.0)
        view._on_absdev_slider_changed(0.0)
        qapp.processEvents()

        assert "0.00%" in view.final_mask_kpi_label.text()

        view.save_btn.click()
        qapp.processEvents()

        assert "0.00% active pixels" in view.save_status_label.text()
        assert calibration_store.has_calibration(cal_dir=cal_store_dir)

        med_dark, mask, record = calibration_store.load_calibration(cal_dir=cal_store_dir)
        assert record.surviving_pixels == 0
        assert record.suppression_pct == 100.0
        assert np.count_nonzero(mask) == 0

        summary = calibration_store.get_calibration_summary(cal_dir=cal_store_dir)
        assert "0.00% pixels active" in summary


def test_adversarial_view_memory_lifecycle_and_gc(qapp, qtbot, populated_diagnostics):
    """Instantiate and destroy DarkCalibrationView objects; verify garbage collection."""
    refs = []

    for i in range(10):
        view = DarkCalibrationView()
        btn = QPushButton(f"Co-Pilot {i}")
        view.set_copilot_button(btn)
        view._on_diagnostics_ready(populated_diagnostics)
        view.stddev_slider.setValue((i % 50) + 10)
        view.absdev_slider.setValue((i % 50) + 20)

        refs.append(weakref.ref(view))
        refs.append(weakref.ref(view.canvas))
        refs.append(weakref.ref(view.figure))
        refs.append(weakref.ref(view.stddev_slider))
        refs.append(weakref.ref(btn))

        view.cleanup()
        view.close()
        view.deleteLater()
        del view
        del btn

    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()

    alive_count = sum(1 for r in refs if r() is not None)
    assert alive_count == 0, f"Memory leak detected: {alive_count}/{len(refs)} objects remained alive"


def test_adversarial_worker_signal_disconnection_and_gc(qapp, qtbot, synthetic_dark_frames):
    """Verify DarkDiagnosticsWorker instances are garbage collected upon completion."""
    worker_refs = []

    for _ in range(5):
        worker = DarkDiagnosticsWorker(synthetic_dark_frames)
        worker_refs.append(weakref.ref(worker))
        worker.run()
        del worker

    gc.collect()
    alive_workers = sum(1 for r in worker_refs if r() is not None)
    assert alive_workers == 0, f"Worker GC leak: {alive_workers}/5 workers remained alive"

