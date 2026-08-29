"""Unit tests for single-photon clustering engine (Stage 1, Stage 2, Stage 3)."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile

from rixs_app.core.photon_clustering import (
    DarkMaskConfig,
    ClusterConfig,
    ReconstructionConfig,
    Stage1Result,
    ReconstructionResult,
    compute_dark_mask,
    process_single_frame_clusters,
    process_signal_stack_clusters,
    reconstruct_photon_event_map,
    export_intden_histogram,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# ============================================================================
# Stage 1 Tests: Dark Mask Generation
# ============================================================================

def test_compute_dark_mask_basic(temp_dir: Path):
    """Test Stage 1 dark baseline, variance mask, and tail-count stability mask."""
    h, w = 100, 100
    n_frames = 150
    base_val = 500.0

    # Normal background: Gaussian noise around 500 ADU with stddev ~ 10 ADU
    rng = np.random.default_rng(42)
    dark_stack = rng.normal(loc=base_val, scale=10.0, size=(n_frames, h, w)).astype(np.float32)

    # Pixel (10, 10): Hot pixel with huge stddev (sigma > 40)
    dark_stack[:, 10, 10] = rng.normal(loc=base_val, scale=55.0, size=n_frames)

    # Pixel (20, 20): Blinking / RTS telegraph pixel (exceeds +/- 60 ADU in 20 frames = 130 good frames < 140)
    dark_stack[:25, 20, 20] += 80.0

    # Save to TIFF files
    dark_paths = []
    for i in range(n_frames):
        p = temp_dir / f"dark_{i:04d}.tif"
        tifffile.imwrite(p, dark_stack[i])
        dark_paths.append(p)

    config = DarkMaskConfig(
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_thresh_ratio=140.0 / 150.0,
    )

    result = compute_dark_mask(dark_paths, config=config)

    assert isinstance(result, Stage1Result)
    assert result.med_dark.shape == (h, w)
    assert result.final_mask.shape == (h, w)
    assert result.med_dark.dtype == np.float32
    assert result.final_mask.dtype == np.float32

    # Verify median is close to 500
    assert np.isclose(np.mean(result.med_dark), base_val, atol=2.0)

    # Hot pixel (10, 10) must be suppressed in stddev mask
    assert result.stddev_mask[10, 10] == 0.0
    assert result.final_mask[10, 10] == 0.0

    # Blinking pixel (20, 20) must be suppressed in tail mask
    assert result.tail_mask[20, 20] == 0.0
    assert result.final_mask[20, 20] == 0.0

    # Normal pixel (50, 50) must survive
    assert result.final_mask[50, 50] == 1.0
    assert result.surviving_pixels < result.total_pixels
    assert result.suppression_pct > 0.0


def test_compute_dark_mask_empty_raises(temp_dir: Path):
    """Empty list of dark paths should raise ValueError."""
    with pytest.raises(ValueError, match="No dark frame paths"):
        compute_dark_mask([])


# ============================================================================
# Stage 2 Tests: Cluster Extraction & 8-Connectivity
# ============================================================================

def test_process_single_frame_isolated_photons():
    """Test 8-connected component extraction and exact Center of Mass centroiding."""
    h, w = 100, 100
    med_dark = np.zeros((h, w), dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    frame = np.zeros((h, w), dtype=np.float32)

    # Photon 1: 2x2 cluster at (y=20..21, x=30..31)
    # [100,  50]
    # [ 50, 100] -> total IntDen = 300 ADU. Center of Mass:
    # X_M = (30*100 + 31*50 + 30*50 + 31*100) / 300 = (3000 + 1550 + 1500 + 3100) / 300 = 9150 / 300 = 30.5
    # Y_M = (20*100 + 20*50 + 21*50 + 21*100) / 300 = (2000 + 1000 + 1050 + 2100) / 300 = 6150 / 300 = 20.5
    frame[20, 30] = 100.0
    frame[20, 31] = 50.0
    frame[21, 30] = 50.0
    frame[21, 31] = 100.0

    # Photon 2: Single pixel hit at (y=60, x=70) with 200 ADU
    frame[60, 70] = 200.0

    config = ClusterConfig(sig_thresh_low=45.0, sig_thresh_high=1e6, connectivity=8)
    df = process_single_frame_clusters(frame, med_dark, final_mask, config=config, slice_idx=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    # Check columns
    expected_cols = ["ClusterNum", "Slice", "Area", "Mean", "StdDev", "Min", "Max", "XM", "YM", "Circ.", "IntDen"]
    for col in expected_cols:
        assert col in df.columns

    # Find the 2x2 cluster
    c_2x2 = df[df["Area"] == 4].iloc[0]
    assert np.isclose(c_2x2["IntDen"], 300.0)
    assert np.isclose(c_2x2["Mean"], 75.0)
    assert np.isclose(c_2x2["Min"], 50.0)
    assert np.isclose(c_2x2["Max"], 100.0)
    assert np.isclose(c_2x2["XM"], 30.5)
    assert np.isclose(c_2x2["YM"], 20.5)
    assert c_2x2["Slice"] == 1

    # Find the single-pixel hit
    c_1x1 = df[df["Area"] == 1].iloc[0]
    assert np.isclose(c_1x1["IntDen"], 200.0)
    assert np.isclose(c_1x1["XM"], 70.0)
    assert np.isclose(c_1x1["YM"], 60.0)
    assert np.isclose(c_1x1["Circ."], 1.0)


def test_8_connectivity_diagonal_contact():
    """Verify diagonally touching charge pixels are grouped as 1 cluster in 8-connectivity."""
    h, w = 50, 50
    med_dark = np.zeros((h, w), dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    frame = np.zeros((h, w), dtype=np.float32)

    # Diagonal contact: (10, 10) and (11, 11)
    frame[10, 10] = 80.0
    frame[11, 11] = 90.0

    config = ClusterConfig(sig_thresh_low=45.0, sig_thresh_high=1e6, connectivity=8)
    df = process_single_frame_clusters(frame, med_dark, final_mask, config=config, slice_idx=5)

    assert len(df) == 1
    assert df.iloc[0]["Area"] == 2
    assert np.isclose(df.iloc[0]["IntDen"], 170.0)
    assert df.iloc[0]["Slice"] == 5


def test_sub_threshold_and_mask_exclusion():
    """Verify pixels below 45 ADU or masked pixels are excluded from clustering."""
    h, w = 50, 50
    med_dark = np.zeros((h, w), dtype=np.float32)
    final_mask = np.ones((h, w), dtype=np.float32)
    final_mask[15, 15] = 0.0  # Dead pixel masked

    frame = np.zeros((h, w), dtype=np.float32)
    frame[10, 10] = 30.0  # Below 45 ADU cutoff
    frame[15, 15] = 200.0  # Hit on masked pixel

    config = ClusterConfig(sig_thresh_low=45.0, sig_thresh_high=1e6, connectivity=8)
    df = process_single_frame_clusters(frame, med_dark, final_mask, config=config, slice_idx=1)

    assert len(df) == 0


# ============================================================================
# Stage 3 Tests: Event Reconstruction & Histogram
# ============================================================================

def test_reconstruct_photon_event_map_cuts(temp_dir: Path):
    """Test IntDen and shape gating during event map reconstruction."""
    image_shape = (100, 100)

    # Create dummy clusters DataFrame
    records = [
        # Valid single-photon (O-K): IntDen = 220, Area = 4, Circ = 0.8 -> ACCEPT
        {"ClusterNum": 0, "Slice": 1, "Area": 4, "Mean": 55.0, "StdDev": 5.0, "Min": 50.0, "Max": 60.0, "XM": 25.4, "YM": 40.2, "Circ.": 0.8, "IntDen": 220.0},
        # Noise split: IntDen = 80 (< 120) -> REJECT NOISE
        {"ClusterNum": 1, "Slice": 1, "Area": 2, "Mean": 40.0, "StdDev": 2.0, "Min": 38.0, "Max": 42.0, "XM": 10.0, "YM": 10.0, "Circ.": 0.9, "IntDen": 80.0},
        # 2-photon pileup: IntDen = 450 (> 320) -> REJECT PILEUP
        {"ClusterNum": 2, "Slice": 1, "Area": 6, "Mean": 75.0, "StdDev": 10.0, "Min": 50.0, "Max": 90.0, "XM": 50.0, "YM": 50.0, "Circ.": 0.8, "IntDen": 450.0},
        # Large cosmic streak: Area = 15 (> 9) -> REJECT SHAPE
        {"ClusterNum": 3, "Slice": 1, "Area": 15, "Mean": 15.0, "StdDev": 5.0, "Min": 10.0, "Max": 20.0, "XM": 70.0, "YM": 70.0, "Circ.": 0.8, "IntDen": 225.0},
        # Low circularity artifact: Circ = 0.15 (< 0.3) -> REJECT SHAPE
        {"ClusterNum": 4, "Slice": 1, "Area": 5, "Mean": 40.0, "StdDev": 5.0, "Min": 30.0, "Max": 50.0, "XM": 80.0, "YM": 80.0, "Circ.": 0.15, "IntDen": 200.0},
    ]
    df_clusters = pd.DataFrame(records)

    config = ReconstructionConfig(
        intden_low=120.0,
        intden_high=320.0,
        max_area=9,
        min_circ=0.3,
        subpixel_factor=1,
    )

    result = reconstruct_photon_event_map(df_clusters, image_shape=image_shape, config=config)

    assert isinstance(result, ReconstructionResult)
    assert result.total_clusters == 5
    assert result.accepted_events == 1
    assert result.rejected_noise == 1
    assert result.rejected_pileup == 1
    assert result.rejected_shape == 2
    assert result.rejected_bounds == 0
    assert np.isclose(result.acceptance_pct, 20.0)

    # Check mapped coordinate: YM = 40.2 -> 40, XM = 25.4 -> 25
    assert result.event_map[40, 25] == 1.0
    assert np.sum(result.event_map) == 1.0


def test_subpixel_super_resolution_reconstruction():
    """Test 2x super-resolution grid accumulation."""
    image_shape = (50, 50)
    records = [
        {"ClusterNum": 0, "Slice": 1, "Area": 4, "Mean": 55.0, "StdDev": 5.0, "Min": 50.0, "Max": 60.0, "XM": 10.4, "YM": 20.6, "Circ.": 0.9, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 1, "Area": 4, "Mean": 55.0, "StdDev": 5.0, "Min": 50.0, "Max": 60.0, "XM": 10.7, "YM": 20.1, "Circ.": 0.9, "IntDen": 210.0},
    ]
    df_clusters = pd.DataFrame(records)

    # 2x subpixel factor: Grid is 100x100
    # Event 0: floor(10.4 * 2) = 20, floor(20.6 * 2) = 41 -> map[41, 20]
    # Event 1: floor(10.7 * 2) = 21, floor(20.1 * 2) = 40 -> map[40, 21]
    config = ReconstructionConfig(intden_low=100.0, intden_high=300.0, max_area=9, min_circ=0.3, subpixel_factor=2)
    result = reconstruct_photon_event_map(df_clusters, image_shape=image_shape, config=config)

    assert result.event_map.shape == (100, 100)
    assert result.event_map[41, 20] == 1.0
    assert result.event_map[40, 21] == 1.0
    assert np.sum(result.event_map) == 2.0


def test_export_intden_histogram(temp_dir: Path):
    """Test exporting diagnostic IntDen histogram to PNG."""
    df_clusters = pd.DataFrame({
        "IntDen": [50.0, 150.0, 220.0, 250.0, 480.0, 900.0],
        "Slice": [1, 1, 2, 2, 3, 3],
    })
    out_png = temp_dir / "test_histogram.png"

    export_intden_histogram(
        df_clusters,
        output_png=out_png,
        intden_low=120.0,
        intden_high=320.0,
        bins=50,
        hist_min=0.0,
        hist_max=1000.0,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_edge_case_zero_clusters_reconstruction():
    """Verify reconstruction handling when 0 clusters are detected across the session."""
    df_empty = pd.DataFrame(columns=[
        "ClusterNum", "Slice", "Area", "Mean", "StdDev", "Min", "Max", "XM", "YM", "Circ.", "IntDen"
    ])
    recon = reconstruct_photon_event_map(
        df_clusters=df_empty,
        image_shape=(100, 100),
        config=ReconstructionConfig(),
    )
    assert recon.total_clusters == 0
    assert recon.accepted_events == 0
    assert recon.acceptance_pct == 0.0
    assert recon.event_map.shape == (100, 100)
    assert np.all(recon.event_map == 0.0)


def test_edge_case_out_of_bounds_and_nan_cluster_coordinates():
    """Verify coordinates outside the image frame (negative or >= dimension) are safely rejected."""
    df_abnormal = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": 10.0, "YM": 10.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 1, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": -5.0, "YM": 10.0, "Circ.": 0.8, "IntDen": 200.0},
        {"ClusterNum": 2, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": 100.0, "YM": 105.0, "Circ.": 0.8, "IntDen": 200.0},
    ])

    recon = reconstruct_photon_event_map(
        df_clusters=df_abnormal,
        image_shape=(100, 100),
        config=ReconstructionConfig(intden_low=100.0, intden_high=300.0),
    )

    assert recon.total_clusters == 3
    assert recon.accepted_events == 1
    assert recon.rejected_bounds == 2
    assert recon.event_map[10, 10] == 1.0


def test_super_resolution_2x_and_4x_reconstruction():
    """Verify super-resolution reconstruction scaling at 2x and 4x."""
    df = pd.DataFrame([
        {"ClusterNum": 0, "Slice": 1, "Area": 2, "Mean": 100.0, "StdDev": 0.0, "Min": 0.0, "Max": 100.0, "XM": 10.25, "YM": 12.75, "Circ.": 0.9, "IntDen": 200.0}
    ])

    recon_2x = reconstruct_photon_event_map(
        df_clusters=df,
        image_shape=(50, 50),
        config=ReconstructionConfig(intden_low=100.0, intden_high=300.0, subpixel_factor=2),
    )
    assert recon_2x.event_map.shape == (100, 100)
    assert recon_2x.event_map[25, 20] == 1.0

    recon_4x = reconstruct_photon_event_map(
        df_clusters=df,
        image_shape=(50, 50),
        config=ReconstructionConfig(intden_low=100.0, intden_high=300.0, subpixel_factor=4),
    )
    assert recon_4x.event_map.shape == (200, 200)
    assert recon_4x.event_map[51, 41] == 1.0

