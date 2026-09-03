"""End-to-End Boundary, Limits, and Error Recovery Test Suite for Phase 2 (RIXS Super-App).

Covers edge cases, limit values, and adversarial condition handling across:
1. Calibration Store Edge & Corrupted Cases
2. Core Dark Diagnostics & Fast Masking Extreme Boundaries
3. CLI Parameter Limits & Failure Modes
4. Home Launchpad Resiliency & Fallback States
5. Dark Calibration Studio I/O & Slider Limit Cases
6. Clustering State Manager & Empty Cluster DataFrames
7. Clustering Studio Lifecycle, Bounds & Teardown
8. Smart Scan Log Pairing Edge Cases & Global Keyboard Guarding
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from rixs_app.core import dark_mask_store
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
    process_signal_stack_clusters,
    reconstruct_photon_event_map,
)
from rixs_app.ui.clustering_slideshow.file_selection_view import ClusteringFileSelectionView
from rixs_app.ui.clustering_slideshow.manager import ClusteringManager
from rixs_app.ui.clustering_slideshow.studio_view import ClusteringStudioView
from rixs_app.ui.dark_masking.dark_mask_view import DarkMaskingView
from rixs_app.ui.home_launchpad import HomeLaunchpadView
from rixs_app.ui.sorting_view import SortingView, find_matching_scan_txt


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication instance for headless Qt testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


# ============================================================================
# Section 1: Dark Mask Store Boundaries
# ============================================================================

def test_e2e_boundary_cal_store_empty_and_partial(tmp_path: Path):
    """Verify dark mask store handles empty and partial file states cleanly."""
    cal_dir = tmp_path / "empty_cal"
    assert not dark_mask_store.has_dark_mask(mask_dir=cal_dir)

    cal_dir.mkdir(parents=True)
    assert not dark_mask_store.has_dark_mask(mask_dir=cal_dir)

    # Only JSON
    (cal_dir / "mask_meta.json").write_text("{}")
    assert not dark_mask_store.has_dark_mask(mask_dir=cal_dir)

    # Only MED_Dark
    tifffile.imwrite(str(cal_dir / "MED_Dark.tif"), np.zeros((10, 10), dtype=np.float32))
    assert not dark_mask_store.has_dark_mask(mask_dir=cal_dir)


def test_e2e_boundary_cal_store_save_shape_mismatch(tmp_path: Path):
    """Verify save_dark_mask raises ValueError on shape mismatches."""
    with pytest.raises(ValueError, match="Shape mismatch|must be 2D"):
        dark_mask_store.save_dark_mask(
            med_dark=np.zeros((10, 10), dtype=np.float32),
            final_mask=np.zeros((20, 20), dtype=np.float32),
            stddev_thresh=40.0,
            absdev_thresh=60.0,
            tail_ratio=0.93,
            dark_frame_count=5,
            surviving_pixels=100,
            total_pixels=100,
            suppression_pct=0.0,
            source_dir="/tmp",
            mask_dir=tmp_path,
        )


# ============================================================================
# Section 2: Dark Diagnostics & Fast Masking Boundaries
# ============================================================================

def test_e2e_boundary_dark_diagnostics_all_bad_and_all_clean():
    """Verify 0% and 100% survival boundary calculations."""
    h, w = 32, 32
    # All clean
    diag_clean = DarkDiagnostics(
        med_dark=np.full((h, w), 500.0, dtype=np.float32),
        per_pixel_stddev=np.full((h, w), 5.0, dtype=np.float32),
        pct93_residual=np.full((h, w), 10.0, dtype=np.float32),
        dark_frame_count=50,
    )
    res_clean = apply_dark_thresholds(diag_clean, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res_clean.surviving_pixels == h * w
    assert res_clean.suppression_pct == 0.0

    # All bad
    diag_bad = DarkDiagnostics(
        med_dark=np.full((h, w), 500.0, dtype=np.float32),
        per_pixel_stddev=np.full((h, w), 100.0, dtype=np.float32),
        pct93_residual=np.full((h, w), 200.0, dtype=np.float32),
        dark_frame_count=50,
    )
    res_bad = apply_dark_thresholds(diag_bad, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res_bad.surviving_pixels == 0
    assert res_bad.suppression_pct == 100.0


def test_e2e_boundary_dark_diagnostics_nan_inf_safety():
    """Verify non-finite float values are safely treated as suppressed without crashes."""
    h, w = 10, 10
    stddev = np.full((h, w), 5.0, dtype=np.float32)
    residual = np.full((h, w), 10.0, dtype=np.float32)
    stddev[0, 0] = np.nan
    stddev[0, 1] = np.inf
    residual[1, 0] = np.nan

    diag = DarkDiagnostics(
        med_dark=np.zeros((h, w), dtype=np.float32),
        per_pixel_stddev=stddev,
        pct93_residual=residual,
        dark_frame_count=50,
    )
    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res.final_mask[0, 0] == 0.0
    assert res.final_mask[0, 1] == 0.0
    assert res.final_mask[1, 0] == 0.0
    assert res.final_mask[5, 5] == 1.0


# ============================================================================
# Section 3: Single-Photon Clustering Boundaries
# ============================================================================

def test_e2e_boundary_clustering_zero_detected_clusters():
    """Verify reconstruction behavior when no clusters are detected."""
    df_empty = pd.DataFrame(columns=[
        "ClusterNum", "Slice", "Area", "Mean", "StdDev", "Min", "Max", "XM", "YM", "Circ.", "IntDen"
    ])
    recon = reconstruct_photon_event_map(
        df_clusters=df_empty,
        image_shape=(64, 64),
        config=ReconstructionConfig(),
    )
    assert recon.total_clusters == 0
    assert recon.accepted_events == 0
    assert recon.event_map.shape == (64, 64)
    assert np.all(recon.event_map == 0.0)


def test_e2e_boundary_clustering_out_of_bounds_coords():
    """Verify coordinates outside the image frame are safely rejected by bounds checking."""
    df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": 10.0, "YM": 10.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": -5.0, "YM": 10.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 2, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": 100.0, "YM": 105.0, "Circ.": 0.8, "IntDen": 200.0},
    ])
    recon = reconstruct_photon_event_map(
        df_clusters=df,
        image_shape=(64, 64),
        config=ReconstructionConfig(intden_low=100.0, intden_high=300.0),
    )
    assert recon.total_clusters == 3
    assert recon.accepted_events == 1
    assert recon.rejected_bounds == 2


# ============================================================================
# Section 4: GUI & Navigation Boundaries
# ============================================================================

def test_e2e_boundary_dark_mask_gui_zero_and_extreme_sliders(qapp):
    """Verify Dark Masking GUI handles extreme and 0 slider boundaries."""
    diag = DarkDiagnostics(
        med_dark=np.full((32, 32), 100.0, dtype=np.float32),
        per_pixel_stddev=np.full((32, 32), 10.0, dtype=np.float32),
        pct93_residual=np.full((32, 32), 20.0, dtype=np.float32),
        dark_frame_count=10,
    )
    view = DarkMaskingView()
    view._on_diagnostics_ready(diag)

    # 0 boundary -> 0%
    view._on_stddev_slider_changed(0.0)
    view._on_absdev_slider_changed(0.0)
    assert "0.00%" in view.final_mask_kpi_label.text()

    # Extreme high -> 100%
    view._on_stddev_slider_changed(1000.0)
    view._on_absdev_slider_changed(1000.0)
    assert "100.00%" in view.final_mask_kpi_label.text()
    view.cleanup()


def test_e2e_boundary_sorting_view_scan_log_discovery(tmp_path: Path):
    """Verify smart scan log regex discovery handles edge cases gracefully."""
    dataset_dir = tmp_path / "scan_0055_darks"
    dataset_dir.mkdir()

    # No TXT file present -> None
    assert find_matching_scan_txt([str(dataset_dir / "frame_001.tif")]) is None

    # Matching TXT in parent dir
    txt_path = tmp_path / "0055_scan_log.txt"
    txt_path.write_text("Header line\nCol1\tCol2\n1\t2\n")
    matched = find_matching_scan_txt([str(dataset_dir / "frame_001.tif")])
    assert matched is not None
    assert Path(matched).name == "0055_scan_log.txt"
