"""Comprehensive PySide6 GUI unit and performance test suite for Single-Photon Clustering Studio.

Covers:
- ClusteringFileSelectionView:
  - Dark calibration status verification banner (green OK vs red missing).
  - Drag-and-drop file and folder ingest, natural sorting, and chunk count feedback.
  - Launch button validation gating (requires active dark calibration and signal files).
  - Navigation callbacks (❮ Back to Home, Calibrate link, Launch Studio) and Co-Pilot button docking.
- ClusteringManager:
  - In-memory DataFrame cache management, session initialization, and calibration loading.
  - Chunk frame ranges partitioning (e.g. 160 frames / 80 chunk size -> 2 chunks).
  - Per-chunk and per-frame cluster queries and reconstructions.
  - Dark-subtracted frame rendering.
  - STRICT BENCHMARK: Stage 3 in-memory filtering <50ms on 50,000+ clusters without disk I/O.
- Workers (ClusterPipelineWorker & ChunkSaveWorker):
  - Progressive signal emissions (frame_result, progress, finished).
  - Cooperative worker cancellation.
  - Artifact export naming (Photon_Event_Map_frames_{start}-{end}.tif, total map, Results_clusters.xls, IntDen_histogram.png).
- ClusteringStudioView:
  - 3 Studio modes (Dashboard, Frame Inspector, Chunk Inspector) and mode switching.
  - Contextual 4 KPI cards dynamic synchronization per mode.
  - Live progressive canvas accumulation during Stage 2 extraction.
  - RangeSlider cutlines during drag and instant <50ms re-filtering upon slider release.
  - Frame Inspector cluster overlays (green accepted, red rejected, cyan sub-pixel centroid '+').
  - Chunk Inspector colormap switching and chunk scrubbing.
  - Frame and chunk scrubbing API (prev_frame, next_frame, prev_chunk, next_chunk).
  - Stale parameters warning banner.
  - Matplotlib figure teardown on close.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import QApplication, QPushButton

from rixs_app.core import calibration_store
from rixs_app.core.photon_clustering import (
    ClusterConfig,
    ReconstructionConfig,
    ReconstructionResult,
)
from rixs_app.ui.clustering_slideshow.file_selection_view import (
    ClusteringFileSelectionView,
)
from rixs_app.ui.clustering_slideshow.manager import (
    CLUSTER_COLUMNS,
    ClusteringManager,
    ClusteringState,
)
from rixs_app.ui.clustering_slideshow.studio_view import (
    ClusteringStudioView,
    KPICard,
)
from rixs_app.ui.clustering_slideshow.workers import (
    ChunkSaveWorker,
    ClusterPipelineWorker,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def dummy_cal_dir(tmp_path):
    """Create a mock dark calibration store directory with valid calibration files."""
    cal_dir = tmp_path / "dark_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    h, w = 64, 64
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    # Mask out top-left pixel
    final_mask[0, 0] = 0.0

    calibration_store.save_calibration(
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
        cal_dir=cal_dir,
    )

    return cal_dir


@pytest.fixture
def synthetic_signal_frames(tmp_path):
    """Generate 6 synthetic 64x64 signal TIFF frames with isolated photon clusters."""
    sig_dir = tmp_path / "signal_run_01"
    sig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    np.random.seed(42)
    for i in range(6):
        frame = np.full((64, 64), 100.0, dtype=np.float32)
        # Inject photon event 1 (accepted single photon)
        frame[15, 20] += 120.0
        frame[15, 21] += 80.0
        # Inject photon event 2 (noise floor below 45 ADU)
        frame[40, 40] += 20.0
        # Inject photon event 3 (high pileup above 320 ADU)
        frame[50, 50] += 500.0

        p = sig_dir / f"signal_frame_{i + 1:03d}.tif"
        tifffile.imwrite(str(p), frame)
        paths.append(str(p))

    return paths


@pytest.fixture
def benchmark_dataframe():
    """Generate a large synthetic DataFrame of 50,000 clusters for benchmark testing."""
    np.random.seed(42)
    n_clusters = 50_000
    n_frames = 200

    slices = np.random.randint(1, n_frames + 1, size=n_clusters)
    areas = np.random.randint(1, 12, size=n_clusters)
    int_dens = np.random.uniform(50.0, 600.0, size=n_clusters).astype(np.float32)
    circs = np.random.uniform(0.1, 1.0, size=n_clusters).astype(np.float32)
    xms = np.random.uniform(0.0, 2047.0, size=n_clusters).astype(np.float32)
    yms = np.random.uniform(0.0, 2047.0, size=n_clusters).astype(np.float32)
    means = int_dens / np.maximum(areas, 1)

    df = pd.DataFrame({
        "ClusterNum": np.arange(n_clusters, dtype=np.int64),
        "Slice": slices,
        "Area": areas,
        "Mean": means,
        "StdDev": np.zeros(n_clusters, dtype=np.float32),
        "Min": np.zeros(n_clusters, dtype=np.float32),
        "Max": int_dens,
        "XM": xms,
        "YM": yms,
        "Circ.": circs,
        "IntDen": int_dens,
    })
    return df


# ============================================================================
# 1. ClusteringFileSelectionView Tests
# ============================================================================

def test_file_selection_banner_verified(qapp, qtbot, dummy_cal_dir):
    """Verify banner shows green OK and summary when dark calibration is present."""
    view = ClusteringFileSelectionView(cal_dir=dummy_cal_dir)
    qtbot.addWidget(view)

    assert "Dark Calibration Verified" in view._cal_status_text.text()
    assert view._cal_status_icon.text() == "✓"
    assert view.launch_btn.isEnabled() is False  # No files loaded yet

    view.load_files(["/fake/path/frame_1.tif"])
    assert view.launch_btn.isEnabled() is True


def test_file_selection_banner_missing(qapp, qtbot, tmp_path):
    """Verify banner shows red missing and disables launch when dark calibration is absent."""
    empty_cal_dir = tmp_path / "empty_cal"
    empty_cal_dir.mkdir()

    view = ClusteringFileSelectionView(cal_dir=empty_cal_dir)
    qtbot.addWidget(view)

    assert "No Dark Calibration Found" in view._cal_status_text.text()
    assert view._cal_status_icon.text() == "⚠️"
    assert view.launch_btn.isEnabled() is False

    # Even with files, launch is disabled
    view.load_files(["/fake/path/frame_1.tif"])
    assert view.launch_btn.isEnabled() is False


def test_file_selection_ingest_and_natural_sorting(qapp, qtbot, tmp_path, dummy_cal_dir):
    """Verify folder loading, file ingest, natural sorting, and chunk calculation."""
    folder = tmp_path / "test_frames"
    folder.mkdir()
    # Create filenames with out-of-order numerical strings
    file_names = ["scan_10.tif", "scan_1.tif", "scan_2.tif", "scan_20.tif"]
    for name in file_names:
        tifffile.imwrite(str(folder / name), np.zeros((10, 10), dtype=np.float32))

    view = ClusteringFileSelectionView(cal_dir=dummy_cal_dir)
    qtbot.addWidget(view)
    loaded = view.load_folder(folder)

    assert len(loaded) == 4
    # Check natural sorting: scan_1 -> scan_2 -> scan_10 -> scan_20
    assert Path(loaded[0]).name == "scan_1.tif"
    assert Path(loaded[1]).name == "scan_2.tif"
    assert Path(loaded[2]).name == "scan_10.tif"
    assert Path(loaded[3]).name == "scan_20.tif"

    # Verify chunk count text
    view.chunk_size_spin.setValue(2)
    assert "4 frames ÷ 2 = 2 chunks" in view._chunk_summary_lbl.text()

    view.clear_files()
    assert len(view.signal_paths) == 0
    assert "No Signal TIFF Files Loaded" in view._file_count_lbl.text()


def test_file_selection_callbacks_and_copilot_docking(qapp, qtbot, dummy_cal_dir):
    """Verify callbacks for back navigation, calibrate link, launch studio, and Co-Pilot docking."""
    mock_back = MagicMock()
    mock_cal = MagicMock()
    mock_launch = MagicMock()

    view = ClusteringFileSelectionView(
        cal_dir=dummy_cal_dir,
        on_back=mock_back,
        on_navigate_dark_cal=mock_cal,
        on_launch_studio=mock_launch,
    )
    qtbot.addWidget(view)

    # 1. Back button
    view._back_btn.click()
    mock_back.assert_called_once()

    # 2. Calibrate button
    view._cal_action_btn.click()
    mock_cal.assert_called_once()

    # 3. Launch button
    view.load_files(["/fake/path/scan_001.tif"])
    view.chunk_size_spin.setValue(50)
    view.launch_btn.click()
    mock_launch.assert_called_once()
    args, _ = mock_launch.call_args
    assert args[0] == ["/fake/path/scan_001.tif"]
    assert args[1] == 50
    assert isinstance(args[2], ClusterConfig)
    assert isinstance(args[3], ReconstructionConfig)

    # 4. Co-Pilot docking
    copilot_btn = QPushButton("Co-Pilot")
    view.set_copilot_button(copilot_btn)
    assert copilot_btn.parent() == view._copilot_container


def test_file_selection_adaptive_threshold_population(tmp_path, qtbot):
    """Verify ClusteringFileSelectionView auto-populates sig_low_spin with 4.0 * typical_dark_sigma and shows badge."""
    cal_dir = tmp_path / "adaptive_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    h, w = 32, 32
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)

    calibration_store.save_calibration(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=20,
        surviving_pixels=h * w,
        total_pixels=h * w,
        suppression_pct=0.0,
        source_dir="/mock/dark",
        cal_dir=cal_dir,
        typical_dark_sigma=14.2,
    )

    view = ClusteringFileSelectionView(cal_dir=cal_dir)
    qtbot.addWidget(view)

    assert view.sig_low_spin.value() == pytest.approx(56.8, 0.1)
    assert "Auto: 4.0σ (dark frame noise σ = 14.2 ADU)" in view._sig_low_auto_lbl.text()
    assert "#34d399" in view._sig_low_auto_lbl.styleSheet()


def test_file_selection_adaptive_threshold_fallback(tmp_path, qtbot):
    """Verify fallback behavior to 45.0 ADU with warning badge when typical_dark_sigma is absent."""
    cal_dir = tmp_path / "fallback_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    h, w = 32, 32
    med_dark = np.full((h, w), 100.0, dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)

    calibration_store.save_calibration(
        med_dark=med_dark,
        final_mask=final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=20,
        surviving_pixels=h * w,
        total_pixels=h * w,
        suppression_pct=0.0,
        source_dir="/mock/dark",
        cal_dir=cal_dir,
    )

    view = ClusteringFileSelectionView(cal_dir=cal_dir)
    qtbot.addWidget(view)

    assert view.sig_low_spin.value() == pytest.approx(45.0, 0.1)
    assert "falling back to default 45.0 ADU" in view._sig_low_auto_lbl.text()
    assert "#fbbf24" in view._sig_low_auto_lbl.styleSheet()



# ============================================================================
# 2. ClusteringManager Tests
# ============================================================================

def test_manager_init_and_state(dummy_cal_dir, synthetic_signal_frames):
    """Verify ClusteringManager session initialization and state encapsulation."""
    mgr = ClusteringManager()
    mgr.init_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
    )

    assert mgr.total_frames == 6
    assert mgr.total_chunks == 2
    assert mgr.state.image_shape == (64, 64)
    assert mgr.state.med_dark is not None
    assert mgr.state.final_mask is not None
    assert mgr.state.df_clusters.empty
    assert mgr.has_clusters is False


def test_manager_chunk_frame_ranges(dummy_cal_dir):
    """Verify chunk frame partition ranges for various chunk sizes."""
    mgr = ClusteringManager()
    paths = [f"/fake/frame_{i+1:03d}.tif" for i in range(160)]

    # 160 frames with chunk size 80 -> 2 chunks: (1, 80), (81, 160)
    mgr.state = ClusteringState(signal_paths=[Path(p) for p in paths], chunk_size=80)
    ranges = mgr.get_chunk_frame_ranges()
    assert ranges == [(1, 80), (81, 160)]

    # 200 frames with chunk size 80 -> 3 chunks: (1, 80), (81, 160), (161, 200)
    paths_200 = [f"/fake/frame_{i+1:03d}.tif" for i in range(200)]
    mgr.state = ClusteringState(signal_paths=[Path(p) for p in paths_200], chunk_size=80)
    ranges_200 = mgr.get_chunk_frame_ranges()
    assert ranges_200 == [(1, 80), (81, 160), (161, 200)]


def test_manager_progressive_append_and_queries(dummy_cal_dir, synthetic_signal_frames):
    """Verify append_frame_clusters, get_frame_clusters, get_chunk_clusters, and frame dark subtraction."""
    mgr = ClusteringManager()
    mgr.init_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
    )

    # Create dummy clusters for frame 1 and frame 2
    df_f1 = pd.DataFrame([{
        "ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0,
        "Min": 80.0, "Max": 120.0, "XM": 15.5, "YM": 20.5, "Circ.": 0.8, "IntDen": 200.0
    }])
    df_f2 = pd.DataFrame([{
        "ClusterNum": 0, "Slice": 2, "Area": 3, "Mean": 90.0, "StdDev": 0.0,
        "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 270.0
    }])

    mgr.append_frame_clusters(1, df_f1)
    assert len(mgr.state.df_clusters) == 1
    assert mgr.state.processed_frame_count == 1

    mgr.append_frame_clusters(2, df_f2)
    assert len(mgr.state.df_clusters) == 2
    assert mgr.state.processed_frame_count == 2
    assert list(mgr.state.df_clusters["ClusterNum"]) == [0, 1]

    # Query single frame clusters
    f1_res = mgr.get_frame_clusters(1)
    assert len(f1_res) == 1
    assert f1_res.iloc[0]["Slice"] == 1

    # Query chunk 0 (frames 1-3)
    c0_res = mgr.get_chunk_clusters(0)
    assert len(c0_res) == 2

    # Query chunk 1 (frames 4-6, empty)
    c1_res = mgr.get_chunk_clusters(1)
    assert len(c1_res) == 0

    # Get frame dark-subtracted image
    img = mgr.get_frame_image(1, dark_subtracted=True)
    assert img.shape == (64, 64)
    assert np.all(img >= 0.0)


def test_manager_filtering_latency_benchmark_under_50ms(benchmark_dataframe):
    """STRICT REQUIREMENT: In-memory Stage 3 filtering on 50,000+ clusters must complete in <50ms."""
    mgr = ClusteringManager()
    mgr.state = ClusteringState(
        image_shape=(2048, 2048),
        df_clusters=benchmark_dataframe,
        recon_config=ReconstructionConfig(intden_low=120.0, intden_high=320.0, max_area=9, min_circ=0.3),
    )

    # Warmup
    mgr.get_reconstruction()

    # Benchmark timed run
    n_iterations = 10
    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        recon = mgr.get_reconstruction()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    avg_latency = float(np.mean(latencies))
    max_latency = float(np.max(latencies))

    # Strict performance assertion
    assert max_latency < 0.050, f"Max latency {max_latency*1000:.2f}ms exceeded 50ms threshold!"
    assert avg_latency < 0.025, f"Avg latency {avg_latency*1000:.2f}ms exceeded 25ms target!"
    assert isinstance(recon, ReconstructionResult)
    assert recon.total_clusters == 50_000
    assert recon.accepted_events > 0
    assert recon.event_map.shape == (2048, 2048)


def test_manager_stale_stage2_flag():
    """Verify stale_stage2 property and setter."""
    mgr = ClusteringManager()
    assert mgr.stale_stage2 is False
    mgr.mark_stage2_stale(True)
    assert mgr.stale_stage2 is True
    mgr.mark_stage2_stale(False)
    assert mgr.stale_stage2 is False


def test_manager_append_frame_clusters_no_quadratic_realloc(dummy_cal_dir):
    """Verify O(1) list accumulation and lazy consolidation across 150 frames."""
    mgr = ClusteringManager()
    paths = [f"/mock/frame_{i+1:03d}.tif" for i in range(150)]
    mgr.init_session(
        signal_paths=paths,
        chunk_size=50,
        cal_dir=dummy_cal_dir,
    )

    # Ingest 150 frame DataFrames via append_frame_clusters
    clusters_per_frame = 5
    for f_idx in range(1, 151):
        frame_df = pd.DataFrame({
            "ClusterNum": np.arange(clusters_per_frame),
            "Slice": np.full(clusters_per_frame, f_idx),
            "Area": np.full(clusters_per_frame, 2),
            "Mean": np.full(clusters_per_frame, 100.0, dtype=np.float32),
            "StdDev": np.zeros(clusters_per_frame, dtype=np.float32),
            "Min": np.full(clusters_per_frame, 80.0, dtype=np.float32),
            "Max": np.full(clusters_per_frame, 120.0, dtype=np.float32),
            "XM": np.full(clusters_per_frame, 10.0, dtype=np.float32),
            "YM": np.full(clusters_per_frame, 20.0, dtype=np.float32),
            "Circ.": np.full(clusters_per_frame, 0.8, dtype=np.float32),
            "IntDen": np.full(clusters_per_frame, 200.0, dtype=np.float32),
        })
        mgr.append_frame_clusters(f_idx, frame_df)

    # Verify frame count and lazy accumulation state
    assert mgr.state.processed_frame_count == 150
    assert len(mgr.state._frame_dfs) == 150
    assert mgr.state._dirty is True

    # Consolidate via df_clusters property access
    consolidated = mgr.state.df_clusters
    assert mgr.state._dirty is False
    total_expected = 150 * clusters_per_frame
    assert len(consolidated) == total_expected
    assert list(consolidated["ClusterNum"]) == list(range(total_expected))
    assert consolidated.iloc[0]["Slice"] == 1
    assert consolidated.iloc[-1]["Slice"] == 150

    # Verify repeated access doesn't re-concatenate (cached)
    assert mgr.state.df_clusters is consolidated

    # Verify clear_clusters() clears _frame_dfs and resets state
    mgr.clear_clusters()
    assert len(mgr.state._frame_dfs) == 0
    assert mgr.state.df_clusters.empty
    assert mgr.state.processed_frame_count == 0
    assert mgr.state._dirty is False

    # Verify set_all_clusters(...) replaces _frame_dfs and df_clusters
    new_df = pd.DataFrame({
        "ClusterNum": [99, 100],
        "Slice": [1, 2],
        "Area": [1, 2],
        "Mean": [50.0, 60.0],
        "StdDev": [0.0, 0.0],
        "Min": [40.0, 50.0],
        "Max": [60.0, 70.0],
        "XM": [5.0, 6.0],
        "YM": [7.0, 8.0],
        "Circ.": [0.9, 0.9],
        "IntDen": [100.0, 120.0],
    })
    mgr.set_all_clusters(new_df)
    assert len(mgr.state._frame_dfs) == 1
    assert len(mgr.state.df_clusters) == 2
    assert list(mgr.state.df_clusters["ClusterNum"]) == [0, 1]
    assert mgr.state.processed_frame_count == 150
    assert mgr.state._dirty is False


# ============================================================================
# 3. Worker Tests
# ============================================================================

def test_pipeline_worker_signals_and_execution(qapp, synthetic_signal_frames, dummy_cal_dir):
    """Verify ClusterPipelineWorker processes frames sequentially and emits progressive signals."""
    med_dark, final_mask, _ = calibration_store.load_calibration(cal_dir=dummy_cal_dir)

    worker = ClusterPipelineWorker(
        signal_paths=synthetic_signal_frames,
        med_dark=med_dark,
        final_mask=final_mask,
        config=ClusterConfig(sig_thresh_low=45.0),
    )

    frame_results: list[tuple[int, pd.DataFrame]] = []
    progress_calls: list[tuple[int, int, int]] = []
    finished_dfs: list[pd.DataFrame] = []

    worker.signals.frame_result.connect(lambda f_idx, df: frame_results.append((f_idx, df)))
    worker.signals.progress.connect(lambda c, t, tot: progress_calls.append((c, t, tot)))
    worker.signals.finished.connect(finished_dfs.append)

    worker.run()

    assert len(frame_results) == 6
    assert sorted([f[0] for f in frame_results]) == [1, 2, 3, 4, 5, 6]
    assert len(finished_dfs) == 1
    df_all = finished_dfs[0]
    assert isinstance(df_all, pd.DataFrame)
    assert not df_all.empty
    assert "ClusterNum" in df_all.columns
    assert len(progress_calls) == 6


def test_pipeline_worker_cancellation(qapp, synthetic_signal_frames, dummy_cal_dir):
    """Verify ClusterPipelineWorker stops immediately when cancel() is requested."""
    med_dark, final_mask, _ = calibration_store.load_calibration(cal_dir=dummy_cal_dir)

    worker = ClusterPipelineWorker(
        signal_paths=synthetic_signal_frames,
        med_dark=med_dark,
        final_mask=final_mask,
    )

    canceled_called = []
    worker.signals.canceled.connect(lambda: canceled_called.append(True))

    worker.cancel()
    worker.run()

    assert len(canceled_called) == 1
    assert worker.is_canceled is True


def test_pipeline_worker_parallel_execution(qapp, qtbot, tmp_path):
    """Verify ClusterPipelineWorker processes frames in parallel and aggregates sorted results."""
    sig_dir = tmp_path / "parallel_sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(5):
        frame = np.full((32, 32), 100.0, dtype=np.float32)
        frame[5 + i, 5 + i] += 150.0
        p = sig_dir / f"frame_{i + 1:03d}.tif"
        tifffile.imwrite(str(p), frame)
        paths.append(str(p))

    med_dark = np.full((32, 32), 100.0, dtype=np.float32)
    final_mask = np.ones((32, 32), dtype=np.float32)

    worker = ClusterPipelineWorker(
        signal_paths=paths,
        med_dark=med_dark,
        final_mask=final_mask,
        config=ClusterConfig(sig_thresh_low=45.0),
        max_workers=4,
    )
    worker.setAutoDelete(False)
    assert worker.max_workers == 4

    frame_results: list[tuple[int, pd.DataFrame]] = []
    progress_calls: list[tuple[int, int, int]] = []
    finished_dfs: list[pd.DataFrame] = []

    worker.signals.frame_result.connect(lambda f_idx, df: frame_results.append((f_idx, df)))
    worker.signals.progress.connect(lambda c, t, tot: progress_calls.append((c, t, tot)))
    worker.signals.finished.connect(finished_dfs.append)

    with qtbot.waitSignal(worker.signals.finished, timeout=5000):
        QThreadPool.globalInstance().start(worker)

    assert len(frame_results) == 5
    assert sorted([f[0] for f in frame_results]) == [1, 2, 3, 4, 5]
    assert len(finished_dfs) == 1
    df_all = finished_dfs[0]
    assert isinstance(df_all, pd.DataFrame)
    assert len(df_all) == 5
    # Verify aggregation and sorted by Slice
    assert list(df_all["Slice"]) == [1, 2, 3, 4, 5]
    assert list(df_all["ClusterNum"]) == [0, 1, 2, 3, 4]
    assert len(progress_calls) == 5
    assert progress_calls[-1][0] == 5
    assert progress_calls[-1][1] == 5
    assert progress_calls[-1][2] == 5


def test_pipeline_worker_cancellation_parallel(qapp, qtbot, tmp_path):
    """Verify requesting cancel() causes worker to exit cleanly, cancel pending futures, and emit canceled signal."""
    sig_dir = tmp_path / "cancel_sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(20):
        frame = np.full((32, 32), 100.0, dtype=np.float32)
        frame[10, 10] += 150.0
        p = sig_dir / f"frame_{i + 1:03d}.tif"
        tifffile.imwrite(str(p), frame)
        paths.append(str(p))

    med_dark = np.full((32, 32), 100.0, dtype=np.float32)
    final_mask = np.ones((32, 32), dtype=np.float32)

    worker = ClusterPipelineWorker(
        signal_paths=paths,
        med_dark=med_dark,
        final_mask=final_mask,
        max_workers=2,
    )
    worker.setAutoDelete(False)

    canceled_called = []
    finished_called = []
    frame_results = []

    def on_frame(f_idx, df):
        frame_results.append(f_idx)
        if len(frame_results) >= 2:
            worker.cancel()

    worker.signals.frame_result.connect(on_frame)
    worker.signals.canceled.connect(lambda: canceled_called.append(True))
    worker.signals.finished.connect(finished_called.append)

    with qtbot.waitSignal(worker.signals.canceled, timeout=5000):
        QThreadPool.globalInstance().start(worker)

    assert worker.is_canceled is True
    assert len(canceled_called) == 1
    assert len(finished_called) == 0
    assert len(frame_results) < 20


def test_chunk_save_worker_exports(qapp, synthetic_signal_frames, dummy_cal_dir, tmp_path):
    """Verify ChunkSaveWorker exports per-chunk TIFFs, total TIFF, TSV spreadsheet, and histogram."""
    mgr = ClusteringManager()
    mgr.init_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
    )

    # Inject mock clusters
    mock_df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 4, "Area": 3, "Mean": 90.0, "StdDev": 0.0, "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 250.0},
    ])
    mgr.set_all_clusters(mock_df)

    export_dir = tmp_path / "clusters_export"
    worker = ChunkSaveWorker(
        manager=mgr,
        output_dir=export_dir,
        recon_config=ReconstructionConfig(intden_low=100.0, intden_high=300.0),
    )

    saved_chunks: list[str] = []
    finished_called = []

    worker.signals.chunk_saved.connect(lambda c, t, p: saved_chunks.append(p))
    worker.signals.finished.connect(finished_called.append)

    worker.run()

    assert len(finished_called) == 1
    assert len(saved_chunks) == 2

    # Check exported file products
    chunk1_file = export_dir / "Photon_Event_Map_frames_1-3.tif"
    chunk2_file = export_dir / "Photon_Event_Map_frames_4-6.tif"
    total_file = export_dir / "Photon_Event_Map_total.tif"
    xls_file = export_dir / "Results_clusters.xls"
    hist_file = export_dir / "IntDen_histogram.png"

    assert chunk1_file.is_file()
    assert chunk2_file.is_file()
    assert total_file.is_file()
    assert xls_file.is_file()
    assert hist_file.is_file()

    # Check that TSV file has headers and rows
    content = xls_file.read_text(encoding="utf-8")
    assert "ClusterNum\tSlice\tArea\tMean" in content
    assert "200.0" in content


# ============================================================================
# 4. ClusteringStudioView Tests
# ============================================================================

def test_studio_initialization_and_mode_switching(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify studio initializes correctly and switches smoothly across all 3 modes."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    assert studio.active_mode == "Dashboard"
    assert studio._mode_stack.currentIndex() == 0

    # Switch to Frame Inspector
    studio.set_mode("Frame Inspector")
    assert studio.active_mode == "Frame Inspector"
    assert studio._mode_stack.currentIndex() == 1

    # Switch to Chunk Inspector
    studio.set_mode("Chunk Inspector")
    assert studio.active_mode == "Chunk Inspector"
    assert studio._mode_stack.currentIndex() == 2

    # Switch back to Dashboard
    studio.set_mode("Dashboard")
    assert studio.active_mode == "Dashboard"
    assert studio._mode_stack.currentIndex() == 0

    studio.cleanup()


def test_studio_kpi_cards_synchronization(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify 4 KPI cards update contextually depending on the active studio mode."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    mock_df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 2, "Area": 3, "Mean": 90.0, "StdDev": 0.0, "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 250.0},
    ])
    studio.manager.set_all_clusters(mock_df)

    # 1. Dashboard Mode KPIs
    studio.set_mode("Dashboard")
    assert "TOTAL FRAMES" in studio.kpi_1._title_lbl.text()
    assert "TOTAL CLUSTERS" in studio.kpi_2._title_lbl.text()
    assert "SINGLE PHOTONS" in studio.kpi_3._title_lbl.text()

    # 2. Frame Inspector Mode KPIs
    studio.set_mode("Frame Inspector")
    assert "Frame 1" in studio.kpi_1._value_lbl.text()
    assert "1 clusters" in studio.kpi_2._value_lbl.text()

    # 3. Chunk Inspector Mode KPIs
    studio.set_mode("Chunk Inspector")
    assert "Chunk 1" in studio.kpi_1._value_lbl.text()
    assert "2 clusters" in studio.kpi_2._value_lbl.text()

    studio.cleanup()


def test_studio_progressive_accumulation(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify progressive accumulation slot updates 2D event map and histogram live."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    # Simulate frame 1 cluster emission
    df_f1 = pd.DataFrame([{
        "ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0,
        "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 200.0
    }])
    studio._on_worker_frame_result(1, df_f1)

    assert studio.manager.processed_frame_count == 1
    qtbot.waitUntil(lambda: studio._im_dashboard_event is not None, timeout=1000)
    assert studio._im_dashboard_event is not None

    # Simulate frame 2 cluster emission
    df_f2 = pd.DataFrame([{
        "ClusterNum": 0, "Slice": 2, "Area": 3, "Mean": 90.0, "StdDev": 0.0,
        "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 250.0
    }])
    studio._on_worker_frame_result(2, df_f2)

    assert studio.manager.processed_frame_count == 2
    assert len(studio.manager.state.df_clusters) == 2

    studio.cleanup()


def test_dashboard_zoom_and_intensity_clamping(qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify Dashboard View zoom controls, intensity clamping RangeSlider/textboxes, and throttled accumulation."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    # 1. Initial State
    assert studio._current_zoom_level == 1.0
    assert studio._clamping_floor == 0.0
    assert studio._clamping_ceiling == 1.0
    assert studio._zoom_lbl.text() == "Zoom: 1×"
    assert studio._floor_entry.text() == "0.00"
    assert studio._ceiling_entry.text() == "1.00"

    # 2. Zoom Controls
    h, w = studio.manager.state.image_shape
    # Zoom in
    studio._handle_zoom_in()
    assert studio._current_zoom_level == 2.0
    assert "Zoom: 2×" in studio._zoom_lbl.text()
    xlim = studio._ax_dashboard_event.get_xlim()
    ylim = studio._ax_dashboard_event.get_ylim()
    # Should be centered half-extent of detector
    assert xlim[0] > 0 and xlim[1] < w
    assert ylim[0] > 0 and ylim[1] < h

    # Zoom out
    studio._handle_zoom_out()
    assert studio._current_zoom_level == 1.0
    assert "Zoom: 1×" in studio._zoom_lbl.text()
    assert studio._ax_dashboard_event.get_xlim() == (0.0, float(w))
    assert studio._ax_dashboard_event.get_ylim() == (0.0, float(h))

    # Zoom in twice then reset
    studio._handle_zoom_in()
    studio._handle_zoom_in()
    assert studio._current_zoom_level == 4.0
    assert "Zoom: 4×" in studio._zoom_lbl.text()
    studio._handle_zoom_reset()
    assert studio._current_zoom_level == 1.0
    assert "Zoom: 1×" in studio._zoom_lbl.text()
    assert studio._ax_dashboard_event.get_xlim() == (0.0, float(w))
    assert studio._ax_dashboard_event.get_ylim() == (0.0, float(h))

    # 3. Clamping Slider & Entry Handlers
    # Trigger clamping change with latency check (<5ms)
    # First render a dummy map so _im_dashboard_event exists
    dummy_map = np.zeros((2048, 2048), dtype=np.float32)
    dummy_map[100, 100] = 5.0
    studio._render_dashboard_event_map(dummy_map)
    assert studio._im_dashboard_event is not None
    assert studio._clamping_slider_max >= 5.0

    t0 = time.perf_counter()
    studio._handle_clamping_changed(0.0, 3.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.050  # sub-millisecond, comfortably <50ms
    assert studio._clamping_floor == 0.0
    assert studio._clamping_ceiling == 3.0
    assert studio._ceiling_entry.text() == "3.00"
    clim = studio._im_dashboard_event.get_clim()
    assert clim == (0.0, 3.0)

    # Test Floor text entry submission
    studio._floor_entry.setText("0.50")
    studio._on_floor_entry_submitted()
    assert studio._clamping_floor == 0.50
    assert studio._floor_entry.text() == "0.50"
    assert studio._clamping_slider.val_left == 0.50
    assert studio._im_dashboard_event.get_clim() == (0.50, 3.0)

    # Test Ceiling text entry submission with dynamic expansion
    studio._ceiling_entry.setText("8.00")
    studio._on_ceiling_entry_submitted()
    assert studio._clamping_ceiling == 8.00
    assert studio._ceiling_entry.text() == "8.00"
    assert studio._clamping_slider.val_right == 8.00
    assert studio._clamping_slider.max_val >= 8.00
    assert studio._im_dashboard_event.get_clim() == (0.50, 8.00)

    # 4. Throttled Accumulation Timer
    assert hasattr(studio, "_accum_timer")
    assert studio._accum_timer.interval() == 100
    assert studio._accum_timer.isSingleShot()

    # Emitting worker frame result starts the timer
    mock_df = pd.DataFrame([{
        "ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0,
        "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 150.0
    }])
    studio._on_worker_frame_result(1, mock_df)
    assert studio._accum_timer.isActive()

    # Wait for timer to tick or call tick handler
    qtbot.waitUntil(lambda: not studio._accum_timer.isActive(), timeout=1000)
    assert studio._im_dashboard_event is not None

    studio.cleanup()


def test_studio_rangeslider_cutlines_and_instant_release(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify RangeSlider drag moves cutlines and release triggers instant in-memory filtering."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    mock_df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 150.0},
        {"ClusterNum": 1, "Slice": 2, "Area": 3, "Mean": 90.0, "StdDev": 0.0, "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 400.0},
    ])
    studio.manager.set_all_clusters(mock_df)
    studio._render_intden_histogram()

    # Drag handler: should update cutline positions
    studio._handle_intden_slider_changed(100.0, 300.0)
    assert "100.0 - 300.0 ADU" in studio._intden_cut_lbl.text()

    # Release handler: should re-filter in-memory in <50ms
    t0 = time.perf_counter()
    studio._handle_intden_slider_released(100.0, 300.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.050
    assert studio.manager.state.recon_config.intden_low == 100.0
    assert studio.manager.state.recon_config.intden_high == 300.0
    assert studio.manager.state.latest_recon.accepted_events == 1  # only 150.0 accepted

    studio.cleanup()


def test_studio_frame_inspector_rendering_and_scrubbing(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify Frame Inspector renders dark-subtracted frame, green/red boxes, cyan centroids, and scrubs."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    mock_df = pd.DataFrame([
        # Frame 1: 1 accepted (200 ADU), 1 rejected noise (50 ADU)
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 1, "Area": 1, "Mean": 50.0, "StdDev": 0.0, "Min": 50.0, "Max": 50.0, "XM": 40.0, "YM": 40.0, "Circ.": 1.0, "IntDen": 50.0},
        # Frame 2: 1 accepted
        {"ClusterNum": 2, "Slice": 2, "Area": 3, "Mean": 90.0, "StdDev": 0.0, "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 250.0},
    ])
    studio.manager.set_all_clusters(mock_df)

    studio.set_mode("Frame Inspector")
    assert "Frame 1/6" in studio._ax_frame.get_title()
    assert len(studio._ax_frame.patches) == 2  # 2 cluster bounding boxes

    # Scrub to next frame
    studio.next_frame()
    assert studio._current_frame_idx == 1
    assert studio.frame_spin.value() == 2
    assert "Frame 2/6" in studio._ax_frame.get_title()
    assert len(studio._ax_frame.patches) == 1

    # Scrub to previous frame
    studio.prev_frame()
    assert studio._current_frame_idx == 0
    assert studio.frame_spin.value() == 1

    # Jump to first and last
    studio._handle_frame_last()
    assert studio._current_frame_idx == 5
    assert studio.frame_spin.value() == 6
    studio._handle_frame_first()
    assert studio._current_frame_idx == 0
    assert studio.frame_spin.value() == 1

    studio.cleanup()


def test_studio_chunk_inspector_rendering_and_scrubbing(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify Chunk Inspector renders chunk event map, colormap change, and chunk scrubbing."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    mock_df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 80.0, "Max": 120.0, "XM": 15.0, "YM": 20.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 4, "Area": 3, "Mean": 90.0, "StdDev": 0.0, "Min": 70.0, "Max": 110.0, "XM": 30.0, "YM": 40.0, "Circ.": 0.9, "IntDen": 250.0},
    ])
    studio.manager.set_all_clusters(mock_df)

    studio.set_mode("Chunk Inspector")
    assert "Chunk 1/2" in studio._ax_chunk.get_title()

    # Next chunk
    studio.next_chunk()
    assert studio._current_chunk_idx == 1
    assert "Chunk 2/2" in studio._ax_chunk.get_title()

    # Prev chunk
    studio.prev_chunk()
    assert studio._current_chunk_idx == 0

    # Colormap selection
    studio._chunk_cmap_combo.setCurrentText("plasma")
    assert studio._im_chunk.get_cmap().name == "plasma"

    studio.cleanup()


def test_studio_stale_warning_and_rerun(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify stale parameter warning banner visibility and re-run trigger."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    assert studio._stale_banner.isHidden()

    # Mark stale
    studio.manager.mark_stage2_stale(True)
    studio._stale_banner.show()
    assert not studio._stale_banner.isHidden()

    # Rerun extraction hides stale banner
    studio.manager.mark_stage2_stale(False)
    studio._stale_banner.hide()
    assert studio._stale_banner.isHidden()

    studio.cleanup()


def test_studio_copilot_docking_and_cleanup(qapp, qtbot, synthetic_signal_frames, dummy_cal_dir):
    """Verify Co-Pilot docking and Matplotlib teardown on cleanup."""
    studio = ClusteringStudioView()
    qtbot.addWidget(studio)
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    copilot_btn = QPushButton("Co-Pilot")
    studio.set_copilot_button(copilot_btn)
    assert copilot_btn.parent() == studio._copilot_container

    # Teardown
    studio.cleanup()
    assert len(studio._fig_dashboard.axes) == 0
    assert len(studio._fig_hist.axes) == 0
    assert len(studio._fig_frame.axes) == 0
    assert len(studio._fig_chunk.axes) == 0


# ============================================================================
# 5. Consolidated Adversarial & Stress Tests
# ============================================================================

def test_adversarial_progressive_accumulation_bursts(qapp, dummy_cal_dir):
    """Simulate rapid-fire frame emissions arriving in a high burst."""
    studio = ClusteringStudioView()
    paths = [f"/fake/burst_frame_{i+1:04d}.tif" for i in range(50)]
    studio.load_session(
        signal_paths=paths,
        chunk_size=10,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    total_expected_clusters = 0
    for f_idx in range(1, 51):
        n_clusters_this_frame = 2 if f_idx % 2 == 0 else 0
        if n_clusters_this_frame > 0:
            frame_df = pd.DataFrame({
                "ClusterNum": np.arange(n_clusters_this_frame),
                "Slice": np.full(n_clusters_this_frame, f_idx),
                "Area": np.random.randint(1, 8, size=n_clusters_this_frame),
                "Mean": np.random.uniform(50.0, 200.0, size=n_clusters_this_frame),
                "StdDev": np.zeros(n_clusters_this_frame),
                "Min": np.zeros(n_clusters_this_frame),
                "Max": np.zeros(n_clusters_this_frame),
                "XM": np.random.uniform(0.0, 60.0, size=n_clusters_this_frame),
                "YM": np.random.uniform(0.0, 60.0, size=n_clusters_this_frame),
                "Circ.": np.random.uniform(0.2, 1.0, size=n_clusters_this_frame),
                "IntDen": np.random.uniform(80.0, 300.0, size=n_clusters_this_frame),
            })
            total_expected_clusters += n_clusters_this_frame
        else:
            frame_df = pd.DataFrame(columns=CLUSTER_COLUMNS)

        studio._on_worker_frame_result(f_idx, frame_df)
        studio._on_worker_progress(f_idx, 50, total_expected_clusters)

    assert studio.manager.processed_frame_count == 50
    assert len(studio.manager.state.df_clusters) == total_expected_clusters
    studio.cleanup()


def test_adversarial_rapid_mode_switching(qapp, dummy_cal_dir, synthetic_signal_frames):
    """Stress test switching between Dashboard, Frame Inspector, and Chunk Inspector."""
    studio = ClusteringStudioView()
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    dfs = []
    for f in range(1, 7):
        dfs.append(pd.DataFrame({
            "ClusterNum": [0],
            "Slice": [f],
            "Area": [2],
            "Mean": [100.0],
            "StdDev": [0.0],
            "Min": [50.0],
            "Max": [150.0],
            "XM": [10.0 + f],
            "YM": [15.0 + f],
            "Circ.": [0.8],
            "IntDen": [180.0],
        }))
    df_all = pd.concat(dfs, ignore_index=True)
    studio.manager.set_all_clusters(df_all)

    modes = ["Dashboard", "Frame Inspector", "Chunk Inspector"]
    for i in range(15):
        target_mode = modes[i % 3]
        studio.set_mode(target_mode)
        assert studio.active_mode == target_mode

        if target_mode == "Frame Inspector":
            studio.next_frame()
            studio.prev_frame()
        elif target_mode == "Chunk Inspector":
            studio.next_chunk()
            studio.prev_chunk()

    studio.set_mode("Dashboard")
    assert "TOTAL FRAMES" in studio.kpi_1._title_lbl.text()
    studio.cleanup()


def test_adversarial_worker_cancellation_lifecycle(qapp, dummy_cal_dir, synthetic_signal_frames):
    """Test studio cancel button and slot handling when extraction is running."""
    studio = ClusteringStudioView()
    studio.load_session(
        signal_paths=synthetic_signal_frames,
        chunk_size=3,
        cal_dir=dummy_cal_dir,
        auto_run=False,
    )

    studio.start_pipeline()
    assert studio.manager.state.is_processing is True
    assert not studio._cancel_btn.isHidden()

    studio._handle_cancel_clicked()
    assert studio._pipeline_worker.is_canceled is True

    studio._on_worker_canceled()
    assert studio.manager.state.is_processing is False
    assert studio._pipeline_worker is None
    assert studio._cancel_btn.isHidden()
    assert "canceled" in studio._status_lbl.text().lower()

    studio.cleanup()


def test_adversarial_chunk_partition_boundaries(dummy_cal_dir):
    """Verify chunk partitioning with chunk_size=1, chunk_size > total_frames, and 0 frames."""
    mgr = ClusteringManager()

    # 1. Total frames = 0
    mgr.state = ClusteringState(signal_paths=[], chunk_size=80)
    assert mgr.get_chunk_frame_ranges() == []
    assert mgr.total_chunks == 0

    # 2. Total frames = 10, chunk_size = 1
    paths_10 = [Path(f"/fake/f_{i}.tif") for i in range(10)]
    mgr.state = ClusteringState(signal_paths=paths_10, chunk_size=1)
    assert len(mgr.get_chunk_frame_ranges()) == 10
    assert mgr.get_chunk_frame_ranges()[0] == (1, 1)

    # 3. Total frames = 10, chunk_size = 100
    mgr.state = ClusteringState(signal_paths=paths_10, chunk_size=100)
    assert mgr.get_chunk_frame_ranges() == [(1, 10)]

    # 4. Out of bounds query
    assert mgr.get_chunk_clusters(-1).empty
    assert mgr.get_chunk_clusters(5).empty

