"""End-to-End Workflow Test Suite for Phase 2 (RIXS Super-App).

Simulates complete multi-step beamline workflows and cross-subsystem interactions:
1. Complete Beamline Dark Calibration & Single-Photon Clustering Run
2. Multi-Modal Beamline Session (Mirror Pitch + Drift Registration + Clustering)
3. Corrupted Calibration Detection & Re-calibration Recovery
4. High-Throughput Progressive Streaming & Chunked Accumulation
5. Core Diagnostics <-> Calibration Store Persistence Layer
6. Calibration Store <-> Home Launchpad Status Badging
7. Calibration Store <-> Clustering File Selection Verification Banner
8. Stage 2 Cluster DataFrame Cache <-> Stage 3 In-Memory RangeSlider Filtering (<50ms)
9. Multi-View Navigation Graph <-> Co-Pilot Sidebar Reparenting
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget, QWidget

from rixs_app.core import dark_mask_store
from rixs_app.core.alignment import phase_correlation_offset
from rixs_app.core.cli_utils import glob_tifs
from rixs_app.core.photon_clustering import (
    ClusterConfig,
    DarkDiagnostics,
    DarkMaskConfig,
    ReconstructionConfig,
    ReconstructionResult,
    Stage1Result,
    apply_dark_thresholds,
    compute_dark_diagnostics,
    compute_dark_mask,
    export_intden_histogram,
    process_signal_stack_clusters,
    process_single_frame_clusters,
    reconstruct_photon_event_map,
)
from rixs_app.core.txt_metadata_parser import parse_scan_log
from rixs_app.core.zeroth_order import run_zeroth_order_pipeline
from rixs_app.ui.clustering_slideshow.file_selection_view import ClusteringFileSelectionView
from rixs_app.ui.clustering_slideshow.manager import ClusteringManager
from rixs_app.ui.clustering_slideshow.studio_view import ClusteringStudioView
from rixs_app.ui.dark_masking.dark_mask_view import DarkMaskingView
from rixs_app.ui.home_launchpad import HomeLaunchpadView
from rixs_app.ui.sorting_view import SortingView, find_matching_scan_txt


# ============================================================================
# Synthetic Dataset Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless Qt testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


def create_beamline_dark_dataset(
    target_dir: Path,
    n_frames: int = 50,
    shape: tuple[int, int] = (64, 64),
    base_adu: float = 500.0,
    seed: int = 42,
) -> list[Path]:
    """Creates a realistic dark frame stack with thermal noise, hot pixels, and RTS pixels."""
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = shape
    paths = []

    for i in range(n_frames):
        frame = rng.normal(loc=base_adu, scale=5.0, size=(h, w)).astype(np.float32)
        # Hot pixel
        frame[min(10, h - 1), min(10, w - 1)] = rng.normal(loc=base_adu, scale=55.0)
        # RTS blinking pixel
        if i < 15:
            frame[min(20, h - 1), min(20, w - 1)] += 80.0
        # Dead pixel
        frame[min(30, h - 1), min(30, w - 1)] = 0.0

        file_path = target_dir / f"dark_{i:04d}.tif"
        tifffile.imwrite(file_path, frame)
        paths.append(file_path)

    return paths


def create_beamline_signal_dataset(
    target_dir: Path,
    n_frames: int = 30,
    shape: tuple[int, int] = (64, 64),
    base_adu: float = 500.0,
    seed: int = 2024,
) -> list[Path]:
    """Creates a realistic signal stack simulating Oxygen K-edge single-photon hits."""
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = shape
    paths = []

    for i in range(n_frames):
        frame = rng.normal(loc=base_adu, scale=5.0, size=(h, w)).astype(np.float32)
        # Photon hit 1: 2x2 charge cloud
        y1 = 10 + (i * 2) % max(1, h - 15)
        x1 = 10 + (i * 3) % max(1, w - 15)
        frame[y1, x1] += 90.0
        frame[y1 + 1, x1] += 50.0
        frame[y1, x1 + 1] += 50.0
        frame[y1 + 1, x1 + 1] += 30.0

        # Photon hit 2: isolated 1-pixel hit
        y2 = (y1 + 20) % (h - 5)
        x2 = (x1 + 20) % (w - 5)
        frame[y2, x2] += 210.0

        file_path = target_dir / f"signal_{i:04d}.tif"
        tifffile.imwrite(file_path, frame)
        paths.append(file_path)

    return paths


# ============================================================================
# Section 1: Realistic Beamline Scenarios
# ============================================================================

def test_scenario_1_beamline_dark_masking_and_clustering_lifecycle(tmp_path: Path):
    """Scenario 1: Complete ALS beamline dark masking & single-photon clustering lifecycle."""
    raw_darks_dir = tmp_path / "raw_darks"
    raw_signals_dir = tmp_path / "raw_signals"
    cal_store_dir = tmp_path / "appdata" / "dark_masking"
    clusters_out_dir = tmp_path / "clusters_output"

    raw_darks_dir.mkdir(parents=True, exist_ok=True)
    raw_signals_dir.mkdir(parents=True, exist_ok=True)
    cal_store_dir.mkdir(parents=True, exist_ok=True)
    clusters_out_dir.mkdir(parents=True, exist_ok=True)

    dark_paths = create_beamline_dark_dataset(raw_darks_dir, n_frames=30, shape=(64, 64))
    signal_paths = create_beamline_signal_dataset(raw_signals_dir, n_frames=20, shape=(64, 64))

    # Stage 1: Dark Calibration
    diag = compute_dark_diagnostics(dark_paths, tail_pct=0.9333)
    stage1 = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0, tail_ratio=0.9333)

    record = dark_mask_store.save_dark_mask(
        med_dark=stage1.med_dark,
        final_mask=stage1.final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=len(dark_paths),
        surviving_pixels=stage1.surviving_pixels,
        total_pixels=stage1.total_pixels,
        suppression_pct=stage1.suppression_pct,
        source_dir=raw_darks_dir,
        date="2026-08-28T12:00:00",
        mask_dir=cal_store_dir,
    )
    assert dark_mask_store.has_dark_mask(mask_dir=cal_store_dir)

    # Stage 2: Signal Frame Clustering
    med_dark, final_mask, _ = dark_mask_store.load_dark_mask(mask_dir=cal_store_dir)
    cluster_cfg = ClusterConfig(sig_thresh_low=45.0, sig_thresh_high=1e6, connectivity=8)
    df_clusters = process_signal_stack_clusters(
        signal_paths=signal_paths,
        med_dark=med_dark,
        final_mask=final_mask,
        config=cluster_cfg,
    )
    assert len(df_clusters) > 0
    assert "IntDen" in df_clusters.columns

    # Stage 3: Event Map Reconstruction & Histogram
    recon_cfg = ReconstructionConfig(intden_low=120.0, intden_high=320.0, max_area=9, min_circ=0.3, subpixel_factor=1)
    recon = reconstruct_photon_event_map(df_clusters=df_clusters, image_shape=(64, 64), config=recon_cfg)

    assert recon.total_clusters == len(df_clusters)
    assert recon.accepted_events > 0
    assert recon.event_map.shape == (64, 64)

    # Export
    hist_png = clusters_out_dir / "IntDen_histogram.png"
    export_intden_histogram(df_clusters, hist_png, intden_low=120.0, intden_high=320.0)
    assert hist_png.exists()


def test_scenario_2_multi_modal_beamline_session(tmp_path: Path):
    """Scenario 2: Multi-modal session verifying mirror pitch, drift alignment, and clustering."""
    # 1. Spatial alignment verification
    ref_frame = np.full((64, 64), 100.0, dtype=np.float32)
    ref_frame[30:34, 30:34] += 200.0
    shifted_frame = np.full((64, 64), 100.0, dtype=np.float32)
    shifted_frame[33:37, 34:38] += 200.0

    shift_y, shift_x = phase_correlation_offset(ref_frame, shifted_frame)
    assert abs(shift_y) > 0.0
    assert abs(shift_x) > 0.0

    # 2. Mirror pitch zeroth-order pipeline verification
    f = np.full((64, 64), 100.0, dtype=np.float32)
    f[30, :] += 50.0
    zo_res = run_zeroth_order_pipeline(f)
    assert zo_res is not None
    assert "score" in zo_res


def test_scenario_3_corrupted_calibration_detection_and_recovery(tmp_path: Path):
    """Scenario 3: Detection of corrupted calibration cache and automated re-calibration."""
    cal_dir = tmp_path / "corrupted_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Write corrupt JSON manifest
    (cal_dir / "mask_meta.json").write_text("{CORRUPTED_JSON_FILE")
    assert not dark_mask_store.has_dark_mask(mask_dir=cal_dir)

    # Re-calibration recovery
    dark_paths = create_beamline_dark_dataset(tmp_path / "darks", n_frames=10, shape=(32, 32))
    diag = compute_dark_diagnostics(dark_paths)
    stage1 = apply_dark_thresholds(diag)

    dark_mask_store.save_dark_mask(
        med_dark=stage1.med_dark,
        final_mask=stage1.final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=len(dark_paths),
        surviving_pixels=stage1.surviving_pixels,
        total_pixels=stage1.total_pixels,
        suppression_pct=stage1.suppression_pct,
        source_dir=tmp_path / "darks",
        mask_dir=cal_dir,
    )

    assert dark_mask_store.has_dark_mask(mask_dir=cal_dir)
    med_dark, mask, record = dark_mask_store.load_dark_mask(mask_dir=cal_dir)
    assert med_dark.shape == (32, 32)
    assert record.surviving_pixels > 0


def test_scenario_4_high_throughput_progressive_streaming_and_chunked_accumulation(tmp_path: Path):
    """Scenario 4: High-throughput progressive streaming and chunked event accumulation."""
    cal_dir = tmp_path / "stream_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    dark_paths = create_beamline_dark_dataset(tmp_path / "darks_stream", n_frames=10, shape=(32, 32))
    diag = compute_dark_diagnostics(dark_paths)
    stage1 = apply_dark_thresholds(diag)
    dark_mask_store.save_dark_mask(
        med_dark=stage1.med_dark,
        final_mask=stage1.final_mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.9333,
        dark_frame_count=len(dark_paths),
        surviving_pixels=stage1.surviving_pixels,
        total_pixels=stage1.total_pixels,
        suppression_pct=stage1.suppression_pct,
        source_dir=tmp_path / "darks_stream",
        mask_dir=cal_dir,
    )

    signal_paths = create_beamline_signal_dataset(tmp_path / "signals_stream", n_frames=20, shape=(32, 32))
    mgr = ClusteringManager()
    mgr.init_session(signal_paths=signal_paths, chunk_size=5, mask_dir=cal_dir)

    assert mgr.total_frames == 20
    assert mgr.total_chunks == 4

    med_dark, final_mask, _ = dark_mask_store.load_dark_mask(mask_dir=cal_dir)
    cfg = ClusterConfig()

    for idx, path in enumerate(signal_paths):
        frame = tifffile.imread(str(path))
        df_frame = process_single_frame_clusters(frame, med_dark, final_mask, config=cfg, slice_idx=idx + 1)
        mgr.append_frame_clusters(idx + 1, df_frame)

    assert mgr.processed_frame_count == 20
    assert mgr.has_clusters is True

    recon = mgr.get_reconstruction()
    assert recon.total_clusters == len(mgr.state.df_clusters)
    assert recon.event_map.shape == (32, 32)


# ============================================================================
# Section 2: Subsystem Interaction Contracts
# ============================================================================

def test_interaction_home_launchpad_status_badging(qapp, tmp_path: Path):
    """Interaction 2: Calibration Store <-> Home Launchpad Status Badging."""
    cal_dir = tmp_path / "lp_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    view = HomeLaunchpadView()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dark_mask_store, "DARK_MASK_DIR", cal_dir)
        view.refresh_mask_status()
        badge_text = view._card_dark_mask._badge_label.text()
        assert "No Mask" in badge_text or "Not calibrated" in badge_text

        med = np.full((32, 32), 100.0, dtype=np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        dark_mask_store.save_dark_mask(
            med_dark=med,
            final_mask=mask,
            stddev_thresh=40.0,
            absdev_thresh=60.0,
            tail_ratio=0.93,
            dark_frame_count=10,
            surviving_pixels=1020,
            total_pixels=1024,
            suppression_pct=0.39,
            source_dir="/source",
            date="2026-08-28T12:00:00",
            mask_dir=cal_dir,
        )

        view.refresh_mask_status()
        badge_text = view._card_dark_mask._badge_label.text()
        assert "Mask Generated" in badge_text or "Calibrated" in badge_text
        assert "mask_status_ok" == view._card_dark_mask._badge_label.objectName()


def test_interaction_clustering_file_selection_banner(qapp, tmp_path: Path):
    """Interaction 3: Calibration Store <-> Clustering File Selection Verification Banner."""
    empty_cal_dir = tmp_path / "empty_cal"
    empty_cal_dir.mkdir(parents=True, exist_ok=True)

    view_uncal = ClusteringFileSelectionView(mask_dir=empty_cal_dir)
    assert "No Dark Mask Found" in view_uncal._mask_status_text.text()
    assert not view_uncal.launch_btn.isEnabled()

    valid_cal_dir = tmp_path / "valid_cal"
    valid_cal_dir.mkdir(parents=True, exist_ok=True)
    dark_mask_store.save_dark_mask(
        med_dark=np.zeros((32, 32), dtype=np.float32),
        final_mask=np.ones((32, 32), dtype=np.float32),
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=10,
        surviving_pixels=1024,
        total_pixels=1024,
        suppression_pct=0.0,
        source_dir="/source",
        mask_dir=valid_cal_dir,
    )

    view_cal = ClusteringFileSelectionView(mask_dir=valid_cal_dir)
    assert "Dark Mask Verified" in view_cal._mask_status_text.text()
    view_cal.load_files(["/fake/frame_1.tif"])
    assert view_cal.launch_btn.isEnabled()


def test_interaction_stage3_in_memory_filtering_latency(tmp_path: Path):
    """Interaction 5: Stage 2 Cluster DataFrame Cache <-> Stage 3 In-Memory RangeSlider Filtering (<50ms)."""
    np.random.seed(42)
    n_clusters = 50_000
    df = pd.DataFrame({
        "ClusterNum": np.arange(n_clusters),
        "Slice": np.random.randint(1, 100, size=n_clusters),
        "Area": np.random.randint(1, 10, size=n_clusters),
        "Mean": np.random.uniform(50.0, 300.0, size=n_clusters),
        "StdDev": np.zeros(n_clusters),
        "Min": np.zeros(n_clusters),
        "Max": np.random.uniform(50.0, 300.0, size=n_clusters),
        "XM": np.random.uniform(0.0, 63.0, size=n_clusters),
        "YM": np.random.uniform(0.0, 63.0, size=n_clusters),
        "Circ.": np.random.uniform(0.1, 1.0, size=n_clusters),
        "IntDen": np.random.uniform(50.0, 400.0, size=n_clusters),
    })

    mgr = ClusteringManager()
    mgr.state.image_shape = (64, 64)
    mgr.set_all_clusters(df)

    t0 = time.perf_counter()
    recon = mgr.get_reconstruction(ReconstructionConfig(intden_low=100.0, intden_high=300.0))
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.050, f"Stage 3 in-memory filtering took {elapsed*1000:.2f}ms (>50ms)"
    assert recon.total_clusters == 50_000
    assert recon.accepted_events > 0


def test_interaction_8_view_navigation_routing_and_copilot_docking(qapp):
    """Interaction 10: Multi-View Navigation Graph <-> Co-Pilot Sidebar Reparenting."""
    from rixs_app.main import RixsApp

    app_window = RixsApp(show_window=False)

    # 1. Initial view: Home Launchpad (Index 0)
    assert app_window._stack.currentIndex() == 0
    assert app_window.home_view.isAncestorOf(app_window._sidebar_toggle)

    # 2. Navigate to Dark Masking (Index 1)
    app_window.show_dark_masking()
    assert app_window._stack.currentIndex() == 1
    assert app_window.dark_mask_view.isAncestorOf(app_window._sidebar_toggle)

    # 3. Navigate to Clustering File Selection (Index 2)
    app_window.show_clustering_files()
    assert app_window._stack.currentIndex() == 2
    assert app_window.clustering_file_view.isAncestorOf(app_window._sidebar_toggle)

    # 4. Navigate back to Home (Index 0)
    app_window.show_home()
    assert app_window._stack.currentIndex() == 0
    assert app_window.home_view.isAncestorOf(app_window._sidebar_toggle)

    app_window.close()
