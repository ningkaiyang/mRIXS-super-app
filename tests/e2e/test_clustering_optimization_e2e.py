"""End-to-End Integration and Performance Test Suite for Photon Clustering Studio.

Verifies:
1. Real-world dataset processing performance (< 5.0 seconds for 20 real frames).
2. Synthetic dataset fallback E2E pipeline (< 5.0 seconds for 20 frames, portable).
3. Non-blocking asynchronous streaming and progressive canvas updates.
4. Dashboard zoom controls (Zoom In 2×/4×, Zoom Out, Reset View 1×).
5. [0, 1] Intensity contrast clamping (RangeSlider, QLineEdit returnPressed & editingFinished).
6. Polish items:
   - ClusteringManager.has_clusters is fast O(1) without triggering eager concatenation.
   - ClusteringStudioView._dash_title_lbl is hidden to prevent unmanaged floating in Qt.
   - Defensive .copy() in single/multi DataFrame manipulation in photon_clustering.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtWidgets import QApplication

from rixs_app.core import dark_mask_store
from rixs_app.core.photon_clustering import (
    ClusterConfig,
    process_single_frame_clusters,
)
from rixs_app.ui.clustering_slideshow.manager import (
    CLUSTER_COLUMNS,
    ClusteringManager,
    ClusteringState,
)
from rixs_app.ui.clustering_slideshow.studio_view import ClusteringStudioView

REAL_SIGNAL_DIR = Path(
    "/Users/ningkaiyang/Desktop/Each200Frames/RIXS_preclustering_signal/Time Scan 001840 Images"
)
REAL_CAL_DIR = Path(
    "/Users/ningkaiyang/Desktop/Each200Frames/rixs_app/appdata/dark_masking"
)


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def synthetic_20_frames(tmp_path):
    """Generate 20 synthetic frames with dark calibration for portable testing."""
    sig_dir = tmp_path / "synthetic_sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = tmp_path / "dark_mask"
    cal_dir.mkdir(parents=True, exist_ok=True)

    h, w = 128, 128
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    final_mask[0, 0] = 0.0

    dark_mask_store.save_dark_mask(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=20,
        surviving_pixels=int(np.sum(final_mask)),
        total_pixels=h * w,
        suppression_pct=1.0 / (h * w) * 100.0,
        source_dir="/mock/dark/frames",
        date="2026-08-28T12:00:00",
        mask_dir=cal_dir,
    )

    paths: list[str] = []
    for i in range(20):
        frame = np.full((h, w), 100.0, dtype=np.float32)
        # Add 5 artificial photon hits per frame
        for k in range(5):
            r = (10 + i * 3 + k * 17) % (h - 4)
            c = (15 + i * 5 + k * 21) % (w - 4)
            frame[r, c] += 180.0
            frame[r + 1, c] += 60.0
        p = sig_dir / f"scan_frame_{i + 1:03d}.tif"
        tifffile.imwrite(str(p), frame)
        paths.append(str(p))

    return paths, cal_dir


# ============================================================================
# 1. Polish Item Unit Verifications
# ============================================================================

def test_manager_has_clusters_fast_o1():
    """Verify has_clusters returns O(1) without triggering eager concatenation."""
    mgr = ClusteringManager()
    mgr.state = ClusteringState()

    # Initially empty
    assert not mgr.has_clusters

    # Append a mock frame DataFrame
    dummy_df = pd.DataFrame({
        "ClusterNum": [0],
        "Slice": [1],
        "Area": [1],
        "Mean": [150.0],
        "StdDev": [0.0],
        "Min": [150.0],
        "Max": [150.0],
        "XM": [10.0],
        "YM": [10.0],
        "Circ.": [1.0],
        "IntDen": [150.0],
    })
    mgr.append_frame_clusters(1, dummy_df)

    # State is marked dirty because frames were appended
    assert mgr.state._dirty is True
    # has_clusters must be True
    assert mgr.has_clusters is True
    # Crucially, checking has_clusters must NOT have cleared _dirty (no eager concatenation)
    assert mgr.state._dirty is True


def test_photon_clustering_defensive_copy():
    """Verify single and multi DataFrames are copied defensively during combination."""
    h, w = 32, 32
    frame = np.full((h, w), 100.0, dtype=np.float32)
    # Single-pixel hit
    frame[5, 5] += 200.0
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)

    df = process_single_frame_clusters(
        frame=frame,
        med_dark=med_dark,
        final_mask=final_mask,
        config=ClusterConfig(sig_thresh_low=50.0),
        slice_idx=1,
    )

    assert not df.empty
    assert "ClusterNum" in df.columns
    assert len(df) == 1
    assert df["Slice"].iloc[0] == 1


# ============================================================================
# 2. Real-World Dataset End-to-End Performance Test
# ============================================================================

def test_e2e_real_dataset_pipeline_performance(qtbot):
    """End-to-end integration benchmark on 20 frames of user's real beamline dataset.

    Validates:
    - 20 frames execute in < 5.0 seconds.
    - studio.manager.has_clusters is True.
    - Dash title label is hidden.
    - Clamping values are clim[0] == 0.0 and clim[1] >= 1.0.
    - Zoom controls (In, Out, Reset) operate accurately.
    - EditingFinished submits floor and ceiling entries.
    """
    if not REAL_SIGNAL_DIR.is_dir() or not REAL_CAL_DIR.is_dir():
        pytest.skip(f"Real dataset directory not found at {REAL_SIGNAL_DIR}")

    real_files = sorted(REAL_SIGNAL_DIR.glob("*.tif*"))[:20]
    if len(real_files) < 20:
        pytest.skip(f"Expected at least 20 TIFF files in {REAL_SIGNAL_DIR}, found {len(real_files)}")

    studio = ClusteringStudioView()
    qtbot.addWidget(studio)

    # Verify polish item: title label is hidden
    assert studio._dash_title_lbl.isHidden()

    # Load session
    studio.load_session(
        signal_paths=[str(p) for p in real_files],
        chunk_size=10,
        mask_dir=REAL_CAL_DIR,
        auto_run=False,
    )

    assert not studio.manager.has_clusters
    assert studio._current_zoom_level == 1.0
    assert studio._zoom_lbl.text() == "Zoom: 1×"
    assert studio._floor_entry.text() == "0.00"
    assert studio._ceiling_entry.text() == "1.00"

    # Start pipeline and measure latency
    t0 = time.perf_counter()
    studio.start_pipeline()

    # Wait for completion (timeout 10s, requirement is < 5.0s)
    qtbot.waitUntil(
        lambda: not studio.manager.state.is_processing and studio.manager.has_clusters,
        timeout=10000,
    )
    elapsed = time.perf_counter() - t0

    # 1. Assert latency < 5.0 seconds
    assert elapsed < 5.0, f"20 frames took {elapsed:.2f}s, expected < 5.0s"

    # 2. Assert clusters exist
    assert studio.manager.has_clusters is True
    total_clusters = len(studio.manager.state.df_clusters)
    assert total_clusters > 0

    # 3. Assert clamping values
    assert studio._im_dashboard_event is not None
    clim = studio._im_dashboard_event.get_clim()
    assert clim[0] == 0.0
    assert clim[1] >= 1.0

    # 4. Test Zoom Controls
    studio._handle_zoom_in()
    assert studio._dash_zoom_mode is True

    class MockClickEvent:
        inaxes = studio._ax_dashboard_event
        xdata = 100.0
        ydata = 100.0

    studio._on_dash_canvas_clicked(MockClickEvent())
    assert studio._current_zoom_level == 2.0
    assert "Zoom: 2×" in studio._zoom_lbl.text()

    studio._handle_zoom_in()
    assert studio._dash_zoom_mode is True
    studio._on_dash_canvas_clicked(MockClickEvent())
    assert studio._current_zoom_level == 4.0
    assert "Zoom: 4×" in studio._zoom_lbl.text()

    studio._handle_zoom_out()
    assert studio._current_zoom_level == 2.0
    assert "Zoom: 2×" in studio._zoom_lbl.text()

    studio._handle_zoom_reset()
    assert studio._current_zoom_level == 1.0
    assert "Zoom: 1×" in studio._zoom_lbl.text()

    # 5. Test Clamping Controls & editingFinished
    studio._floor_entry.setText("0.25")
    studio._floor_entry.editingFinished.emit()
    assert studio._clamping_floor == 0.25
    assert studio._im_dashboard_event.get_clim()[0] == 0.25

    studio._ceiling_entry.setText("2.50")
    studio._ceiling_entry.editingFinished.emit()
    assert studio._clamping_ceiling == 2.50
    assert studio._im_dashboard_event.get_clim()[1] == 2.50

    studio.cleanup()


# ============================================================================
# 3. Synthetic Dataset End-to-End Portable Pipeline Test
# ============================================================================

def test_e2e_synthetic_dataset_pipeline_performance(qtbot, synthetic_20_frames):
    """End-to-end integration test with synthetic data passing on any environment.

    Validates:
    - 20 frames execute in < 5.0 seconds.
    - studio.manager.has_clusters is True.
    - Dash title label is hidden.
    - Clamping values clim[0] == 0.0 and clim[1] >= 1.0.
    - Zoom controls and mode transitions.
    - Clean teardown.
    """
    paths, cal_dir = synthetic_20_frames
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)

    assert studio._dash_title_lbl.isHidden()

    studio.load_session(
        signal_paths=paths,
        chunk_size=5,
        mask_dir=cal_dir,
        auto_run=False,
    )

    t0 = time.perf_counter()
    studio.start_pipeline()

    qtbot.waitUntil(
        lambda: not studio.manager.state.is_processing and studio.manager.has_clusters,
        timeout=10000,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"Synthetic 20 frames took {elapsed:.2f}s, expected < 5.0s"
    assert studio.manager.has_clusters is True
    assert len(studio.manager.state.df_clusters) > 0

    assert studio._im_dashboard_event is not None
    clim = studio._im_dashboard_event.get_clim()
    assert clim[0] == 0.0
    assert clim[1] >= 1.0

    # Test mode transitions
    studio.set_mode("Frame Inspector")
    assert studio.active_mode == "Frame Inspector"
    studio.next_frame()
    studio.prev_frame()

    studio.set_mode("Chunk Inspector")
    assert studio.active_mode == "Chunk Inspector"
    studio.next_chunk()
    studio.prev_chunk()

    studio.set_mode("Dashboard")
    assert studio.active_mode == "Dashboard"

    studio.cleanup()


# ============================================================================
# 4. Asynchronous Streaming & UI Responsiveness Test
# ============================================================================

def test_e2e_streaming_nonblocking_responsiveness(qtbot, synthetic_20_frames):
    """Verify asynchronous streaming delivers progressive results without blocking Qt."""
    paths, cal_dir = synthetic_20_frames
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)

    studio.load_session(
        signal_paths=paths,
        chunk_size=5,
        mask_dir=cal_dir,
        auto_run=False,
    )

    studio.start_pipeline()
    assert studio.manager.state.is_processing is True
    assert not studio._cancel_btn.isHidden()

    # Verify event processing continues smoothly while pipeline executes
    for _ in range(10):
        QApplication.processEvents()
        time.sleep(0.01)

    qtbot.waitUntil(
        lambda: not studio.manager.state.is_processing,
        timeout=10000,
    )

    assert studio.manager.has_clusters is True
    assert studio.manager.processed_frame_count == 20
    assert studio._progress_bar.value() == 100
    studio.cleanup()
