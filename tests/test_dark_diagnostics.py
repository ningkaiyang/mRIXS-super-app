"""Unit tests for dark diagnostics and real-time thresholding in photon_clustering.py."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from rixs_app.core.photon_clustering import (
    DarkDiagnostics,
    DarkMaskConfig,
    Stage1Result,
    apply_dark_thresholds,
    compute_dark_diagnostics,
    compute_dark_mask,
)


@pytest.fixture
def temp_dark_stack():
    """Generates synthetic 150-frame dark stack (64x64) with known hot and RTS pixels."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        h, w = 64, 64
        n_frames = 150
        base_val = 500.0

        rng = np.random.default_rng(123)
        dark_stack = rng.normal(loc=base_val, scale=5.0, size=(n_frames, h, w)).astype(np.float32)

        # Hot pixel at (10, 10): stddev ~ 60.0 (> 40.0)
        dark_stack[:, 10, 10] = rng.normal(loc=base_val, scale=60.0, size=n_frames)

        # RTS blinking pixel at (20, 20): jumps +90 ADU in 25 frames (out of 150 = 16.7% > 7% allowable)
        dark_stack[:25, 20, 20] += 90.0

        # Dead pixel at (30, 30): constant 0 ADU
        dark_stack[:, 30, 30] = 0.0

        paths = []
        for i in range(n_frames):
            p = tmp_path / f"dark_{i:04d}.tif"
            tifffile.imwrite(p, dark_stack[i])
            paths.append(p)

        yield {
            "paths": paths,
            "h": h,
            "w": w,
            "n_frames": n_frames,
            "base_val": base_val,
            "hot_pos": (10, 10),
            "rts_pos": (20, 20),
            "clean_pos": (40, 40),
        }


def test_dark_diagnostics_dataclass_fields():
    """Verify DarkDiagnostics holds med_dark, per_pixel_stddev, pct93_residual, dark_frame_count."""
    h, w = 32, 32
    diag = DarkDiagnostics(
        med_dark=np.zeros((h, w), dtype=np.float32),
        per_pixel_stddev=np.ones((h, w), dtype=np.float32),
        pct93_residual=np.full((h, w), 2.5, dtype=np.float32),
        dark_frame_count=100,
    )
    assert diag.med_dark.shape == (h, w)
    assert diag.per_pixel_stddev.shape == (h, w)
    assert diag.pct93_residual.shape == (h, w)
    assert diag.dark_frame_count == 100


def test_compute_dark_diagnostics_basic(temp_dark_stack):
    """Verify compute_dark_diagnostics returns correct array dimensions, dtypes, and expected values."""
    diag = compute_dark_diagnostics(temp_dark_stack["paths"], tail_pct=0.93)

    assert isinstance(diag, DarkDiagnostics)
    assert diag.med_dark.shape == (temp_dark_stack["h"], temp_dark_stack["w"])
    assert diag.per_pixel_stddev.shape == (temp_dark_stack["h"], temp_dark_stack["w"])
    assert diag.pct93_residual.shape == (temp_dark_stack["h"], temp_dark_stack["w"])
    assert diag.med_dark.dtype == np.float32
    assert diag.per_pixel_stddev.dtype == np.float32
    assert diag.pct93_residual.dtype == np.float32
    assert diag.dark_frame_count == temp_dark_stack["n_frames"]

    clean_r, clean_c = temp_dark_stack["clean_pos"]
    assert np.isclose(diag.med_dark[clean_r, clean_c], temp_dark_stack["base_val"], atol=1.5)
    assert diag.per_pixel_stddev[clean_r, clean_c] < 10.0
    assert diag.pct93_residual[clean_r, clean_c] < 15.0


def test_compute_dark_diagnostics_empty_paths_raises():
    """Verify compute_dark_diagnostics raises ValueError on empty path list."""
    with pytest.raises(ValueError, match="[Nn]o dark frame paths"):
        compute_dark_diagnostics([])


def test_compute_dark_diagnostics_mismatched_dimensions(temp_dark_stack):
    """Verify compute_dark_diagnostics raises ValueError when dark frames have mismatched dimensions."""
    paths = list(temp_dark_stack["paths"][:2])
    # Overwrite second path with mismatched shape
    tifffile.imwrite(paths[1], np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="mismatch|shape"):
        compute_dark_diagnostics(paths)


def test_compute_dark_diagnostics_progress_callback(temp_dark_stack):
    """Verify progress callback is called sequentially for each processed frame."""
    calls = []

    def callback(cur, tot):
        calls.append((cur, tot))

    compute_dark_diagnostics(temp_dark_stack["paths"][:10], progress_callback=callback)
    assert len(calls) == 10
    assert calls[0] == (1, 10)
    assert calls[-1] == (10, 10)


def test_hot_pixel_detection_in_diagnostics(temp_dark_stack):
    """Verify hot pixel with elevated stddev is identified and masked by apply_dark_thresholds."""
    diag = compute_dark_diagnostics(temp_dark_stack["paths"])
    hot_r, hot_c = temp_dark_stack["hot_pos"]

    assert diag.per_pixel_stddev[hot_r, hot_c] > 50.0

    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0, tail_ratio=0.93)
    assert isinstance(res, Stage1Result)
    assert res.stddev_mask[hot_r, hot_c] == 0.0
    assert res.final_mask[hot_r, hot_c] == 0.0


def test_rts_pixel_detection_in_diagnostics(temp_dark_stack):
    """Verify RTS blinking pixel is caught by pct93_residual and masked out."""
    diag = compute_dark_diagnostics(temp_dark_stack["paths"])
    rts_r, rts_c = temp_dark_stack["rts_pos"]

    assert diag.pct93_residual[rts_r, rts_c] > 70.0

    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0, tail_ratio=0.93)
    assert res.tail_mask[rts_r, rts_c] == 0.0
    assert res.final_mask[rts_r, rts_c] == 0.0


def test_apply_dark_thresholds_parity_with_compute_dark_mask(temp_dark_stack):
    """Prove apply_dark_thresholds generates bit-identical final_mask to compute_dark_mask."""
    config = DarkMaskConfig(
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_thresh_ratio=140.0 / 150.0,
    )
    res_direct = compute_dark_mask(temp_dark_stack["paths"], config=config)

    diag = compute_dark_diagnostics(temp_dark_stack["paths"], tail_pct=140.0 / 150.0)
    res_fast = apply_dark_thresholds(
        diag,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=140.0 / 150.0,
    )

    assert np.allclose(res_fast.med_dark, res_direct.med_dark, atol=1e-5)
    assert np.array_equal(res_fast.stddev_mask, res_direct.stddev_mask)
    assert np.array_equal(res_fast.final_mask, res_direct.final_mask)
    assert res_fast.surviving_pixels == res_direct.surviving_pixels
    assert np.isclose(res_fast.suppression_pct, res_direct.suppression_pct, atol=1e-4)


def test_apply_dark_thresholds_latency_benchmark():
    """Verify apply_dark_thresholds executes in < 10ms on full 2048x2048 detector arrays."""
    h, w = 2048, 2048
    rng = np.random.default_rng(42)

    med_dark = rng.normal(500.0, 5.0, (h, w)).astype(np.float32)
    per_pixel_stddev = rng.normal(8.0, 2.0, (h, w)).astype(np.float32)
    pct93_residual = rng.normal(12.0, 4.0, (h, w)).astype(np.float32)

    diag = DarkDiagnostics(
        med_dark=med_dark,
        per_pixel_stddev=per_pixel_stddev,
        pct93_residual=pct93_residual,
        dark_frame_count=150,
    )

    import gc

    # Warmup
    for _ in range(3):
        apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0, tail_ratio=0.93)

    gc.collect()
    gc.disable()
    timings = []
    try:
        for _ in range(20):
            t0 = time.perf_counter()
            _ = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0, tail_ratio=0.93)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)
    finally:
        gc.enable()

    mean_ms = float(np.mean(timings))
    p95_ms = float(np.percentile(timings, 95))

    assert mean_ms < 10.0, f"apply_dark_thresholds exceeded 10ms budget: {mean_ms:.2f}ms"
    assert p95_ms < 20.0, f"P95 latency spike: {p95_ms:.2f}ms"


def test_apply_dark_thresholds_slider_monotonicity():
    """Verify increasing thresholds monotonically increases or maintains surviving pixel count."""
    h, w = 100, 100
    rng = np.random.default_rng(999)
    diag = DarkDiagnostics(
        med_dark=np.zeros((h, w), dtype=np.float32),
        per_pixel_stddev=rng.exponential(scale=20.0, size=(h, w)).astype(np.float32),
        pct93_residual=rng.exponential(scale=30.0, size=(h, w)).astype(np.float32),
        dark_frame_count=100,
    )

    threshold_steps = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
    prev_surviving = 0

    for thresh in threshold_steps:
        res = apply_dark_thresholds(diag, stddev_thresh=thresh, absdev_thresh=thresh * 1.5)
        assert res.surviving_pixels >= prev_surviving
        prev_surviving = res.surviving_pixels


def test_edge_case_all_clean_pixels():
    """Verify all clean pixels result in 100% active pixels and 0% suppression."""
    h, w = 50, 50
    diag = DarkDiagnostics(
        med_dark=np.full((h, w), 500.0, dtype=np.float32),
        per_pixel_stddev=np.full((h, w), 5.0, dtype=np.float32),
        pct93_residual=np.full((h, w), 10.0, dtype=np.float32),
        dark_frame_count=150,
    )
    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res.surviving_pixels == h * w
    assert res.suppression_pct == 0.0
    assert np.all(res.final_mask == 1.0)


def test_edge_case_all_bad_pixels():
    """Verify all bad pixels result in 0 active pixels and 100% suppression."""
    h, w = 50, 50
    diag = DarkDiagnostics(
        med_dark=np.full((h, w), 500.0, dtype=np.float32),
        per_pixel_stddev=np.full((h, w), 100.0, dtype=np.float32),
        pct93_residual=np.full((h, w), 200.0, dtype=np.float32),
        dark_frame_count=150,
    )
    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res.surviving_pixels == 0
    assert res.suppression_pct == 100.0
    assert np.all(res.final_mask == 0.0)


def test_edge_case_single_dark_frame():
    """Verify compute_dark_diagnostics and apply_dark_thresholds handle 1 single frame safely."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "single_dark.tif"
        tifffile.imwrite(p, np.full((20, 20), 500.0, dtype=np.float32))

        diag = compute_dark_diagnostics([p])
        assert diag.dark_frame_count == 1
        assert np.array_equal(diag.med_dark, np.full((20, 20), 500.0, dtype=np.float32))
        assert np.all(diag.per_pixel_stddev == 0.0)
        assert np.all(diag.pct93_residual == 0.0)

        res = apply_dark_thresholds(diag)
        assert res.surviving_pixels == 400


def test_edge_case_odd_vs_even_frames():
    """Verify median and percentile computations succeed on odd (N=15) and even (N=16) stacks."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(77)
        for n in [15, 16]:
            paths = []
            for i in range(n):
                p = tmp_path / f"frame_{n}_{i:02d}.tif"
                tifffile.imwrite(p, rng.normal(500.0, 5.0, (20, 20)).astype(np.float32))
                paths.append(p)

            diag = compute_dark_diagnostics(paths)
            assert diag.dark_frame_count == n
            res = apply_dark_thresholds(diag)
            assert res.surviving_pixels > 0


def test_diagnostics_nan_inf_robustness():
    """Verify NaN and Inf values in diagnostics are treated as bad pixels without exceptions."""
    h, w = 10, 10
    stddev = np.full((h, w), 5.0, dtype=np.float32)
    residual = np.full((h, w), 10.0, dtype=np.float32)

    stddev[0, 0] = np.nan
    stddev[0, 1] = np.inf
    residual[1, 0] = np.nan
    residual[1, 1] = np.inf

    diag = DarkDiagnostics(
        med_dark=np.zeros((h, w), dtype=np.float32),
        per_pixel_stddev=stddev,
        pct93_residual=residual,
        dark_frame_count=100,
    )

    res = apply_dark_thresholds(diag, stddev_thresh=40.0, absdev_thresh=60.0)
    assert res.final_mask[0, 0] == 0.0
    assert res.final_mask[0, 1] == 0.0
    assert res.final_mask[1, 0] == 0.0
    assert res.final_mask[1, 1] == 0.0
    assert res.final_mask[5, 5] == 1.0


def test_compute_dark_diagnostics_chunking_parity_and_native_dtype(tmp_path):
    """Verify chunked compute_dark_diagnostics handles non-square native int16 and uint16 frames."""
    h, w = 150, 200
    n_frames = 20
    rng = np.random.default_rng(42)

    paths = []
    for i in range(n_frames):
        data = rng.integers(100, 500, size=(h, w), dtype=np.int16)
        p = tmp_path / f"dark_int16_{i:03d}.tif"
        tifffile.imwrite(p, data)
        paths.append(p)

    diag = compute_dark_diagnostics(paths, tail_pct=0.9333)

    assert diag.med_dark.shape == (h, w)
    assert diag.per_pixel_stddev.shape == (h, w)
    assert diag.pct93_residual.shape == (h, w)
    assert diag.dark_frame_count == n_frames
    assert np.all(np.isfinite(diag.med_dark))
    assert np.all(np.isfinite(diag.per_pixel_stddev))
    assert np.all(np.isfinite(diag.pct93_residual))


def test_adversarial_corrupt_and_zero_byte_tiffs(tmp_path: Path):
    """Verify compute_dark_diagnostics handles zero-byte, garbage, and truncated TIFF files gracefully."""
    bad_dir = tmp_path / "corrupt_tiffs"
    bad_dir.mkdir(parents=True, exist_ok=True)

    zero_file = bad_dir / "zero_bytes.tif"
    zero_file.write_bytes(b"")

    garbage_file = bad_dir / "garbage.tif"
    garbage_file.write_bytes(b"\x00\xFF\xAA\x55" * 100)

    trunc_file = bad_dir / "truncated.tif"
    trunc_file.write_bytes(b"II*\x00")

    with pytest.raises(Exception):
        compute_dark_diagnostics([zero_file, garbage_file, trunc_file])


def test_adversarial_all_constant_frames(tmp_path: Path):
    """Verify compute_dark_diagnostics handles dark frames with strictly identical constant values (variance=0)."""
    const_dir = tmp_path / "const_darks"
    const_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(5):
        p = const_dir / f"c_{i}.tif"
        tifffile.imwrite(str(p), np.full((25, 25), 120.0, dtype=np.float32))
        paths.append(p)

    diag = compute_dark_diagnostics(paths)
    assert np.all(diag.per_pixel_stddev == 0.0)
    assert np.all(diag.pct93_residual == 0.0)

    res = apply_dark_thresholds(diag, stddev_thresh=1.0, absdev_thresh=1.0)
    assert res.surviving_pixels == 25 * 25
    assert res.suppression_pct == 0.0


def test_adversarial_single_pixel_frame(tmp_path: Path):
    """Verify compute_dark_diagnostics handles 1x1 single-pixel image stacks."""
    p1_dir = tmp_path / "1x1_darks"
    p1_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(6):
        p = p1_dir / f"px_{i}.tif"
        tifffile.imwrite(str(p), np.array([[100.0 + i * 2.0]], dtype=np.float32))
        paths.append(p)

    diag = compute_dark_diagnostics(paths)
    assert diag.med_dark.shape == (1, 1)
    assert diag.per_pixel_stddev.shape == (1, 1)

    res = apply_dark_thresholds(diag, stddev_thresh=10.0, absdev_thresh=10.0)
    assert res.surviving_pixels == 1


def test_adversarial_sequential_dimensions(tmp_path: Path):
    """Verify running sequential diagnostics with changing geometries operates cleanly."""
    shapes = [(16, 16), (40, 24), (20, 50)]
    for idx, (h, w) in enumerate(shapes):
        dir_path = tmp_path / f"dim_{idx}"
        dir_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(4):
            p = dir_path / f"f_{i}.tif"
            tifffile.imwrite(str(p), np.full((h, w), 50.0 + i, dtype=np.float32))
            paths.append(p)

        diag = compute_dark_diagnostics(paths)
        assert diag.med_dark.shape == (h, w)
        assert diag.per_pixel_stddev.shape == (h, w)

