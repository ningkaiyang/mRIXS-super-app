"""High-performance single-photon event clustering engine for soft X-ray RIXS.

Implements:
- Stage 1: Dark baseline (median), variance (StdDev), and telegraph/stability (tail count) masking.
- Stage 2: Fast C++ 8-connected component analysis, intensity-weighted sub-pixel centroiding (XM, YM),
           integrated density (IntDen), and contour circularity.
- Stage 3: IntDen energy window gating, shape filtering, super-resolution event mapping (np.add.at),
           and diagnostic 1D log-scale IntDen histogram export.

Zero UI dependencies. Fully thread-safe and vectorized.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
import tifffile

logger = logging.getLogger(__name__)

_C_LIB = None


def _get_c_lib() -> ctypes.CDLL | None:
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB

    so_path = Path(__file__).parent / "_dark_thresh.so"
    c_path = Path(__file__).parent / "_dark_thresh.c"

    if not so_path.exists() and c_path.exists():
        for compiler in ("clang", "gcc"):
            try:
                subprocess.run(
                    [compiler, "-O3", "-shared", "-fPIC", str(c_path), "-o", str(so_path)],
                    capture_output=True,
                    check=True,
                )
                break
            except Exception:
                continue

    if so_path.exists():
        try:
            lib = ctypes.CDLL(str(so_path))
            lib.compute_masks_fast.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            if hasattr(lib, "compute_dark_stats_int16_c"):
                lib.compute_dark_stats_int16_c.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
            if hasattr(lib, "compute_dark_stats_float_c"):
                lib.compute_dark_stats_float_c.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
            _C_LIB = lib
            return _C_LIB
        except Exception:
            pass
    return None


# ============================================================================
# Configuration Data Structures
# ============================================================================

@dataclass(frozen=True)
class DarkMaskConfig:
    """Configuration for Stage 1 dark baseline & mask generation."""
    stddev_thresh: float = 40.0       # Max allowable per-pixel dark standard deviation (ADU)
    absdev_thresh: float = 60.0       # Max allowable single-frame dark excursion (ADU)
    tail_thresh_ratio: float = 0.9333 # Required fraction of stable frames (default 140/150 = 0.9333)
    chunk_size: int = 100             # Frames per streaming chunk to prevent OOM
    max_frames: int = 0               # Max dark frames to use (0 = all)


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for Stage 2 signal conditioning and connected component clustering."""
    sig_thresh_low: float = 45.0      # Lower signal cutoff (3-sigma noise floor in ADU)
    sig_thresh_high: float = 1e6      # Upper signal threshold ceiling (ADU)
    connectivity: int = 8             # 8-connectivity matching ImageJ and charge cloud physics


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for Stage 3 event filtering and 2D event map accumulation."""
    intden_low: float = 120.0         # Single-photon IntDen lower cut (ADU)
    intden_high: float = 320.0        # Single-photon IntDen upper cut (ADU)
    max_area: int = 9                 # Maximum cluster pixel area (pixels)
    min_circ: float = 0.3             # Minimum cluster circularity (4*pi*Area / P^2)
    subpixel_factor: int = 1          # 1 = standard detector grid, 2 = 2x super-resolution, etc.


@dataclass(frozen=True)
class DarkDiagnostics:
    """Diagnostic products computed from a raw dark frame stack.

    Attributes:
        med_dark: 2D float32 median dark image (H, W).
        per_pixel_stddev: 2D float32 per-pixel standard deviation (H, W).
        pct93_residual: 2D float32 93rd-percentile absolute residual (H, W).
        dark_frame_count: Total number of dark frames processed.
    """
    med_dark: np.ndarray              # 2D float32 median dark image (H, W)
    per_pixel_stddev: np.ndarray      # 2D float32 per-pixel standard deviation (H, W)
    pct93_residual: np.ndarray        # 2D float32 93rd-percentile absolute residual (H, W)
    dark_frame_count: int             # Total number of dark frames processed


@dataclass(frozen=True)
class Stage1Result:
    """Results from Stage 1 dark frame analysis."""
    med_dark: np.ndarray              # 2D float32 median dark image (H, W)
    final_mask: np.ndarray            # 2D float32 binary mask {0.0, 1.0} (H, W)
    stddev_mask: np.ndarray          # 2D float32 Tier 1 mask (H, W)
    tail_mask: np.ndarray            # 2D float32 Tier 2 mask (H, W)
    total_pixels: int
    surviving_pixels: int
    suppression_pct: float
    typical_dark_sigma: float = 0.0


@dataclass(frozen=True)
class ReconstructionResult:
    """Results from Stage 3 photon event map reconstruction."""
    event_map: np.ndarray             # 2D float32 photon event map (H * factor, W * factor)
    total_clusters: int
    accepted_events: int
    rejected_noise: int
    rejected_pileup: int
    rejected_shape: int
    rejected_bounds: int
    acceptance_pct: float

    @property
    def photons_accepted(self) -> int:
        """Alias for accepted_events."""
        return self.accepted_events

    @property
    def photons_rejected(self) -> int:
        """Total rejected events."""
        return self.rejected_noise + self.rejected_pileup + self.rejected_shape + self.rejected_bounds


# ============================================================================
# Stage 1: Dark Baseline & Noise Mask Generation
# ============================================================================

def compute_dark_diagnostics(
    dark_paths: Sequence[Path | str],
    tail_pct: float = 0.9333,
    progress_callback: Callable[[int, int], None] | None = None,
    stage_callback: Callable[[int, int, int, str], None] | None = None,
    max_frames: int = 0,
) -> DarkDiagnostics:
    """Compute temporal median dark frame, per-pixel stddev, and 93rd-percentile absolute residuals.

    Args:
        dark_paths: List of filepaths to raw dark TIFF frames.
        tail_pct: Percentile ratio for residual evaluation (default 0.9333 for 93.33rd percentile).
        progress_callback: Optional legacy callback receiving (current_frame, total_frames).
        stage_callback: Optional multi-stage callback receiving (stage_idx, current, total, message).
        max_frames: Max dark frames to process (0 = all).

    Returns:
        DarkDiagnostics containing med_dark, per_pixel_stddev, pct93_residual, and frame count.

    Raises:
        ValueError: If dark_paths is empty or frames have mismatched dimensions.
    """
    if not dark_paths:
        raise ValueError("No dark frame paths provided to compute_dark_diagnostics.")

    paths = [Path(p) for p in dark_paths]
    if max_frames > 0:
        paths = paths[:max_frames]

    n_dark = len(paths)
    if n_dark == 0:
        raise ValueError("No valid dark frame paths found.")

    first_frame = tifffile.imread(paths[0])
    h, w = first_frame.shape
    raw_dtype = first_frame.dtype if np.issubdtype(first_frame.dtype, np.number) else np.float32

    # Ingest dark frames in native dtype to minimize memory footprint
    dark_stack = np.empty((n_dark, h, w), dtype=raw_dtype)
    for i, path in enumerate(paths):
        frame = tifffile.imread(path)
        if frame.shape != (h, w):
            raise ValueError(
                f"Dark frame shape mismatch: expected {(h, w)}, got {frame.shape} for {path}"
            )
        dark_stack[i] = frame
        if progress_callback:
            progress_callback(i + 1, n_dark)
        if stage_callback:
            stage_callback(1, i + 1, n_dark, f"Ingesting dark frame {i + 1}/{n_dark}...")

    med_dark = np.empty((h, w), dtype=np.float32)
    per_pixel_stddev = np.empty((h, w), dtype=np.float32)
    pct93_residual = np.empty((h, w), dtype=np.float32)

    if n_dark == 1:
        med_dark[:, :] = dark_stack[0].astype(np.float32)
        per_pixel_stddev[:, :] = 0.0
        pct93_residual[:, :] = 0.0
        if stage_callback:
            stage_callback(2, 1, 1, "Computing noise statistics (chunk 1/1)...")
    else:
        q_ratio = float(tail_pct if tail_pct <= 1.0 else tail_pct / 100.0)
        c_lib = _get_c_lib()
        has_c_int16 = c_lib is not None and hasattr(c_lib, "compute_dark_stats_int16_c") and dark_stack.dtype == np.int16
        has_c_float = c_lib is not None and hasattr(c_lib, "compute_dark_stats_float_c") and dark_stack.dtype == np.float32

        chunk_size = 128
        chunk_ranges = [(r, min(r + chunk_size, h)) for r in range(0, h, chunk_size)]
        total_chunks = len(chunk_ranges)

        if has_c_int16 or has_c_float:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            completed_chunks = 0

            def _process_c_chunk(cr: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
                r_start, r_end = cr
                num_rows = r_end - r_start
                c_slice = np.ascontiguousarray(dark_stack[:, r_start:r_end, :])
                c_med = np.empty(num_rows * w, dtype=np.float32)
                c_std = np.empty(num_rows * w, dtype=np.float32)
                c_pct = np.empty(num_rows * w, dtype=np.float32)
                if has_c_int16:
                    c_lib.compute_dark_stats_int16_c(
                        c_slice.ctypes.data_as(ctypes.c_void_p),
                        n_dark,
                        num_rows * w,
                        ctypes.c_float(q_ratio),
                        c_med.ctypes.data_as(ctypes.c_void_p),
                        c_std.ctypes.data_as(ctypes.c_void_p),
                        c_pct.ctypes.data_as(ctypes.c_void_p),
                    )
                else:
                    c_lib.compute_dark_stats_float_c(
                        c_slice.ctypes.data_as(ctypes.c_void_p),
                        n_dark,
                        num_rows * w,
                        ctypes.c_float(q_ratio),
                        c_med.ctypes.data_as(ctypes.c_void_p),
                        c_std.ctypes.data_as(ctypes.c_void_p),
                        c_pct.ctypes.data_as(ctypes.c_void_p),
                    )
                return r_start, r_end, c_med.reshape((num_rows, w)), c_std.reshape((num_rows, w)), c_pct.reshape((num_rows, w))

            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as executor:
                futures = [executor.submit(_process_c_chunk, cr) for cr in chunk_ranges]
                for fut in as_completed(futures):
                    r_start, r_end, c_med_res, c_std_res, c_pct_res = fut.result()
                    med_dark[r_start:r_end, :] = c_med_res
                    per_pixel_stddev[r_start:r_end, :] = c_std_res
                    pct93_residual[r_start:r_end, :] = c_pct_res
                    completed_chunks += 1
                    if stage_callback:
                        stage_callback(
                            2,
                            completed_chunks,
                            total_chunks,
                            f"Computing noise statistics (chunk {completed_chunks}/{total_chunks})...",
                        )
        else:
            q = q_ratio * 100.0
            for chunk_idx, (r_start, r_end) in enumerate(chunk_ranges, 1):
                c_stack = dark_stack[:, r_start:r_end, :].astype(np.float32)
                c_med = np.median(c_stack, axis=0).astype(np.float32)
                med_dark[r_start:r_end, :] = c_med
                per_pixel_stddev[r_start:r_end, :] = np.std(c_stack, axis=0).astype(np.float32)
                c_res = np.abs(c_stack - c_med[np.newaxis, :, :])
                pct93_residual[r_start:r_end, :] = np.percentile(c_res, q=q, axis=0).astype(np.float32)
                if stage_callback:
                    stage_callback(
                        2,
                        chunk_idx,
                        total_chunks,
                        f"Computing noise statistics (chunk {chunk_idx}/{total_chunks})...",
                    )

    return DarkDiagnostics(
        med_dark=med_dark,
        per_pixel_stddev=per_pixel_stddev,
        pct93_residual=pct93_residual,
        dark_frame_count=n_dark,
    )


def _calc_typical_dark_sigma(s_arr: np.ndarray, final_mask: np.ndarray) -> float:
    """Calculate typical dark noise sigma from surviving pixels, subsampling large sensors."""
    total_pixels = final_mask.size
    if total_pixels > 65536:
        s_sub = s_arr[::8, ::8]
        m_sub = final_mask[::8, ::8]
        v_sub = s_sub[m_sub == 1.0]
        return float(np.median(v_sub)) if v_sub.size > 0 else 0.0
    v_pix = s_arr[final_mask == 1.0]
    return float(np.median(v_pix)) if v_pix.size > 0 else 0.0


def apply_dark_thresholds(
    diagnostics: DarkDiagnostics,
    stddev_thresh: float = 40.0,
    absdev_thresh: float = 60.0,
    tail_ratio: float = 0.9333,
) -> Stage1Result:
    """Apply variance and excursion thresholds to precomputed dark diagnostics in <10ms.

    Args:
        diagnostics: Precomputed DarkDiagnostics.
        stddev_thresh: Upper threshold for per-pixel standard deviation (ADU).
        absdev_thresh: Upper threshold for 93rd-percentile absolute residual (ADU).
        tail_ratio: Tail threshold ratio (for metadata/config parity).

    Returns:
        Stage1Result containing final binary mask and diagnostic metrics.
    """
    c_lib = _get_c_lib()
    s_arr = diagnostics.per_pixel_stddev
    r_arr = diagnostics.pct93_residual

    if (
        c_lib is not None
        and s_arr.flags.c_contiguous
        and r_arr.flags.c_contiguous
        and s_arr.dtype == np.float32
        and r_arr.dtype == np.float32
        and s_arr.shape == r_arr.shape
    ):
        h, w = s_arr.shape
        final_mask = np.empty((h, w), dtype=np.float32)
        stddev_mask = np.empty((h, w), dtype=np.float32)
        tail_mask = np.empty((h, w), dtype=np.float32)
        surviving = ctypes.c_int(0)
        c_lib.compute_masks_fast(
            s_arr.ctypes.data,
            r_arr.ctypes.data,
            ctypes.c_float(stddev_thresh),
            ctypes.c_float(absdev_thresh),
            final_mask.ctypes.data,
            stddev_mask.ctypes.data,
            tail_mask.ctypes.data,
            int(h * w),
            ctypes.byref(surviving),
        )
        total_pixels = int(final_mask.size)
        surviving_pixels = int(surviving.value)
        suppression_pct = (1.0 - (surviving_pixels / total_pixels)) * 100.0 if total_pixels > 0 else 0.0
        typical_dark_sigma = _calc_typical_dark_sigma(s_arr, final_mask)
        return Stage1Result(
            med_dark=diagnostics.med_dark,
            final_mask=final_mask,
            stddev_mask=stddev_mask,
            tail_mask=tail_mask,
            total_pixels=total_pixels,
            surviving_pixels=surviving_pixels,
            suppression_pct=suppression_pct,
            typical_dark_sigma=typical_dark_sigma,
        )

    # Pure NumPy fallback
    m_std = (diagnostics.per_pixel_stddev < stddev_thresh) & np.isfinite(diagnostics.per_pixel_stddev)
    m_tail = (diagnostics.pct93_residual < absdev_thresh) & np.isfinite(diagnostics.pct93_residual)
    m_final = m_std & m_tail

    stddev_mask = m_std.astype(np.float32)
    tail_mask = m_tail.astype(np.float32)
    final_mask = m_final.astype(np.float32)

    total_pixels = int(final_mask.size)
    surviving_pixels = int(np.count_nonzero(m_final))
    suppression_pct = (1.0 - (surviving_pixels / total_pixels)) * 100.0 if total_pixels > 0 else 0.0
    typical_dark_sigma = _calc_typical_dark_sigma(s_arr, final_mask)

    return Stage1Result(
        med_dark=diagnostics.med_dark,
        final_mask=final_mask,
        stddev_mask=stddev_mask,
        tail_mask=tail_mask,
        total_pixels=total_pixels,
        surviving_pixels=surviving_pixels,
        suppression_pct=suppression_pct,
        typical_dark_sigma=typical_dark_sigma,
    )


def compute_dark_mask(
    dark_paths: Sequence[Path | str],
    config: DarkMaskConfig | None = None,
    *,
    stddev_thresh: float | None = None,
    absdev_thresh: float | None = None,
    tail_ratio: float | None = None,
    tail_thresh_ratio: float | None = None,
    max_frames: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Stage1Result:
    """Compute temporal median dark frame and 2-tier bad/noisy pixel rejection mask.

    Tier 1: Standard deviation across frames (< stddev_thresh).
    Tier 2: Excursion stability score (|Delta D| < absdev_thresh in > tail_thresh frames).

    Args:
        dark_paths: List of filepaths to raw dark TIFF frames.
        config: Optional DarkMaskConfig containing threshold values.
        stddev_thresh: Optional direct override for stddev threshold.
        absdev_thresh: Optional direct override for absdev threshold.
        tail_ratio: Optional direct override for tail ratio cutoff.
        tail_thresh_ratio: Optional alias for tail_ratio.
        max_frames: Optional max frames override.
        progress_callback: Optional callback receiving (current_frame, total_frames).

    Returns:
        Stage1Result containing med_dark, final_mask, and diagnostic metrics.
    """
    cfg = config or DarkMaskConfig()
    s_thresh = stddev_thresh if stddev_thresh is not None else cfg.stddev_thresh
    a_thresh = absdev_thresh if absdev_thresh is not None else cfg.absdev_thresh
    t_ratio = (
        tail_ratio
        if tail_ratio is not None
        else (tail_thresh_ratio if tail_thresh_ratio is not None else cfg.tail_thresh_ratio)
    )
    m_frames = max_frames if max_frames is not None else cfg.max_frames

    diagnostics = compute_dark_diagnostics(
        dark_paths=dark_paths,
        tail_pct=t_ratio,
        progress_callback=progress_callback,
        max_frames=m_frames,
    )
    return apply_dark_thresholds(
        diagnostics=diagnostics,
        stddev_thresh=s_thresh,
        absdev_thresh=a_thresh,
        tail_ratio=t_ratio,
    )


# ============================================================================
# Stage 2: Signal Conditioning & 8-Connected Cluster Extraction
# ============================================================================

def process_single_frame_clusters(
    frame: np.ndarray,
    med_dark: np.ndarray,
    final_mask: np.ndarray,
    config: ClusterConfig = ClusterConfig(),
    slice_idx: int = 1,
) -> pd.DataFrame:
    """Condition a single signal frame and extract 8-connected photon charge clusters.

    Steps:
    1. Dark subtraction and mask multiplication: clean = max(0, (frame - med_dark) * final_mask)
    2. Threshold cutoff: clean[clean < sig_thresh_low] = 0
    3. 8-connected component analysis via cv2.connectedComponentsWithStats
    4. Exact intensity-weighted Center of Mass (XM, YM), IntDen, Area, and Circularity.

    Args:
        frame: Raw 2D signal frame (float32 or uint16).
        med_dark: 2D float32 median dark frame.
        final_mask: 2D float32 binary mask {0.0, 1.0}.
        config: ClusterConfig.
        slice_idx: Global 1-indexed frame index.

    Returns:
        pd.DataFrame matching Results_clusters.xls schema.
    """
    cols = [
        "ClusterNum", "Slice", "Area", "Mean", "StdDev", "Min", "Max", "XM", "YM", "Circ.", "IntDen"
    ]

    clean = (frame.astype(np.float32) - med_dark) * final_mask
    clean[clean < config.sig_thresh_low] = 0.0

    # Binary mask for connected components
    binary = ((clean >= config.sig_thresh_low) & (clean <= config.sig_thresh_high)).astype(np.uint8)

    # Fast C++ 8-connected component labeling
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=config.connectivity,
    )

    if n_labels <= 1:
        # No clusters detected
        return pd.DataFrame(columns=cols)

    # Label 0 is background; evaluate labels 1 to n_labels - 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    single_mask = (areas == 1)
    multi_mask = (areas > 1)

    # 1. Vectorized single-pixel cluster extraction (Area == 1, ~98.5% of all clusters)
    if np.any(single_mask):
        single_indices = np.where(single_mask)[0]
        single_label_ids = single_indices + 1
        xs_single = stats[single_label_ids, cv2.CC_STAT_LEFT]
        ys_single = stats[single_label_ids, cv2.CC_STAT_TOP]
        ints_single = clean[ys_single, xs_single].astype(np.float64)

        valid_single = ints_single > 0.0
        if not np.all(valid_single):
            single_label_ids = single_label_ids[valid_single]
            xs_single = xs_single[valid_single]
            ys_single = ys_single[valid_single]
            ints_single = ints_single[valid_single]

        n_single = len(single_label_ids)
        if n_single > 0:
            df_single = pd.DataFrame({
                "_order": single_label_ids,
                "Slice": np.full(n_single, slice_idx, dtype=np.int64),
                "Area": np.ones(n_single, dtype=np.int64),
                "Mean": ints_single,
                "StdDev": np.zeros(n_single, dtype=np.float64),
                "Min": ints_single,
                "Max": ints_single,
                "XM": xs_single.astype(np.float64),
                "YM": ys_single.astype(np.float64),
                "Circ.": np.ones(n_single, dtype=np.float64),
                "IntDen": ints_single,
            })
        else:
            df_single = pd.DataFrame()
    else:
        df_single = pd.DataFrame()

    # 2. Multi-pixel cluster extraction (Area > 1, ~1.5% of clusters)
    records_multi = []
    if np.any(multi_mask):
        multi_indices = np.where(multi_mask)[0]
        multi_label_ids = multi_indices + 1

        for label_id in multi_label_ids:
            x, y, w, h, area = stats[label_id]
            patch_clean = clean[y:y + h, x:x + w]
            patch_labels = labels[y:y + h, x:x + w]
            patch_mask = (patch_labels == label_id)

            intensities = patch_clean[patch_mask]
            int_den = float(np.sum(intensities))

            if int_den <= 0.0 or len(intensities) == 0:
                continue

            mean_val = float(int_den / area)
            min_val = float(np.min(intensities))
            max_val = float(np.max(intensities))
            std_val = float(np.std(intensities))

            # Sub-pixel Center of Mass (XM, YM)
            local_ys, local_xs = np.where(patch_mask)
            global_xs = local_xs + x
            global_ys = local_ys + y

            xm = float(np.sum(global_xs * intensities) / int_den)
            ym = float(np.sum(global_ys * intensities) / int_den)

            # Circularity: 4 * pi * Area / Perimeter^2
            contours, _ = cv2.findContours(
                patch_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours and len(contours) > 0:
                perimeter = float(cv2.arcLength(contours[0], True))
                if perimeter > 0.0:
                    circ = float(min(1.0, (4.0 * np.pi * area) / (perimeter ** 2)))
                else:
                    circ = 1.0
            else:
                circ = 1.0

            records_multi.append({
                "_order": label_id,
                "Slice": int(slice_idx),
                "Area": int(area),
                "Mean": mean_val,
                "StdDev": std_val,
                "Min": min_val,
                "Max": max_val,
                "XM": xm,
                "YM": ym,
                "Circ.": circ,
                "IntDen": int_den,
            })

    if records_multi:
        df_multi = pd.DataFrame(records_multi)
        df_multi["Slice"] = df_multi["Slice"].astype(np.int64)
        df_multi["Area"] = df_multi["Area"].astype(np.int64)
        df_multi["Mean"] = df_multi["Mean"].astype(np.float64)
        df_multi["StdDev"] = df_multi["StdDev"].astype(np.float64)
        df_multi["Min"] = df_multi["Min"].astype(np.float64)
        df_multi["Max"] = df_multi["Max"].astype(np.float64)
        df_multi["XM"] = df_multi["XM"].astype(np.float64)
        df_multi["YM"] = df_multi["YM"].astype(np.float64)
        df_multi["Circ."] = df_multi["Circ."].astype(np.float64)
        df_multi["IntDen"] = df_multi["IntDen"].astype(np.float64)
    else:
        df_multi = pd.DataFrame()

    # 3. Combine and restore original label ordering
    if df_single.empty and df_multi.empty:
        return pd.DataFrame(columns=cols)
    elif df_multi.empty:
        df = df_single.copy()
    elif df_single.empty:
        df = df_multi.copy()
    else:
        df = pd.concat([df_single, df_multi], ignore_index=True)
        df.sort_values("_order", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df.drop(columns=["_order"], inplace=True)
    df.insert(0, "ClusterNum", np.arange(len(df), dtype=np.int64))
    return df[cols]


def process_signal_stack_clusters(
    signal_paths: Sequence[Path | str],
    med_dark: np.ndarray,
    final_mask: np.ndarray,
    config: ClusterConfig = ClusterConfig(),
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> pd.DataFrame:
    """Process a stack of signal frames and accumulate all detected photon clusters.

    Args:
        signal_paths: Sequence of filepaths to raw signal TIFF files.
        med_dark: 2D float32 median dark frame.
        final_mask: 2D float32 binary mask.
        config: ClusterConfig.
        progress_callback: Optional callback receiving (frame_idx, total_frames, clusters_found).

    Returns:
        pd.DataFrame containing all clusters across all frames.
    """
    all_dfs = []
    total_clusters = 0
    n_frames = len(signal_paths)

    for i, path in enumerate(signal_paths):
        frame = tifffile.imread(path)
        df_frame = process_single_frame_clusters(
            frame=frame,
            med_dark=med_dark,
            final_mask=final_mask,
            config=config,
            slice_idx=i + 1,
        )
        if not df_frame.empty:
            all_dfs.append(df_frame)
            total_clusters += len(df_frame)

        if progress_callback:
            progress_callback(i + 1, n_frames, total_clusters)

    if not all_dfs:
        return pd.DataFrame(columns=[
            "ClusterNum", "Slice", "Area", "Mean", "StdDev", "Min", "Max", "XM", "YM", "Circ.", "IntDen"
        ])

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["ClusterNum"] = np.arange(len(df_all), dtype=np.int64)
    return df_all


# ============================================================================
# Stage 3: Event Map Reconstruction & IntDen Histogram
# ============================================================================

def reconstruct_photon_event_map(
    df_clusters: pd.DataFrame,
    image_shape: tuple[int, int],
    config: ReconstructionConfig = ReconstructionConfig(),
) -> ReconstructionResult:
    """Filter clusters by IntDen and shape cuts and accumulate onto a 2D event map.

    Supports sub-pixel super-resolution binning when config.subpixel_factor > 1.

    Args:
        df_clusters: Cluster DataFrame from Stage 2.
        image_shape: (height, width) of the original detector image.
        config: ReconstructionConfig containing cuts and subpixel factor.

    Returns:
        ReconstructionResult with 2D float32 event_map and diagnostic counts.
    """
    orig_h, orig_w = image_shape
    factor = config.subpixel_factor
    out_h = orig_h * factor
    out_w = orig_w * factor
    event_map = np.zeros((out_h, out_w), dtype=np.float32)

    total = len(df_clusters)
    if total == 0:
        return ReconstructionResult(
            event_map=event_map,
            total_clusters=0,
            accepted_events=0,
            rejected_noise=0,
            rejected_pileup=0,
            rejected_shape=0,
            rejected_bounds=0,
            acceptance_pct=0.0,
        )

    int_den = df_clusters["IntDen"].to_numpy()
    area = df_clusters["Area"].to_numpy()
    circ = df_clusters["Circ."].to_numpy()
    xm = df_clusters["XM"].to_numpy()
    ym = df_clusters["YM"].to_numpy()

    # Rejection boolean masks
    mask_noise = int_den < config.intden_low
    mask_pileup = int_den > config.intden_high
    mask_shape = (area > config.max_area) | (circ < config.min_circ)

    # Discretize sub-pixel coordinates
    px = np.floor(xm * factor).astype(np.int64)
    py = np.floor(ym * factor).astype(np.int64)

    mask_bounds = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)
    mask_accepted = (~mask_noise) & (~mask_pileup) & (~mask_shape) & mask_bounds

    # Diagnostic Tallies
    n_noise = int(np.sum(mask_noise))
    n_pileup = int(np.sum(mask_pileup))
    n_shape = int(np.sum((~mask_noise) & (~mask_pileup) & mask_shape))
    n_bounds = int(np.sum((~mask_noise) & (~mask_pileup) & (~mask_shape) & (~mask_bounds)))
    n_accepted = int(np.sum(mask_accepted))

    # Fast accumulation onto 2D grid
    if n_accepted > 0:
        acc_px = px[mask_accepted]
        acc_py = py[mask_accepted]
        np.add.at(event_map, (acc_py, acc_px), 1.0)

    acceptance_pct = (n_accepted / total * 100.0) if total > 0 else 0.0

    return ReconstructionResult(
        event_map=event_map,
        total_clusters=total,
        accepted_events=n_accepted,
        rejected_noise=n_noise,
        rejected_pileup=n_pileup,
        rejected_shape=n_shape,
        rejected_bounds=n_bounds,
        acceptance_pct=acceptance_pct,
    )


def export_intden_histogram(
    df_clusters: pd.DataFrame,
    output_png: Path | str,
    intden_low: float = 120.0,
    intden_high: float = 320.0,
    bins: int = 300,
    hist_min: float = 0.0,
    hist_max: float = 1500.0,
    dpi: int = 150,
) -> Path:
    """Export diagnostic 1D logarithmic IntDen histogram with cut markers to PNG.

    Args:
        df_clusters: Cluster results DataFrame.
        output_png: Path to save the PNG image.
        intden_low: Lower single-photon IntDen cut.
        intden_high: Upper single-photon IntDen cut.
        bins: Number of histogram bins.
        hist_min: Minimum X-axis ADU.
        hist_max: Maximum X-axis ADU.
        dpi: Output resolution.

    Returns:
        Path to saved PNG file.
    """
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5), facecolor="#14172b")
    _canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#16213e")

    int_den_values = df_clusters["IntDen"].to_numpy() if not df_clusters.empty else np.array([])

    if len(int_den_values) > 0:
        ax.hist(
            int_den_values,
            bins=bins,
            range=(hist_min, hist_max),
            color="#3b82f6",
            edgecolor="none",
            log=True,
            alpha=0.85,
        )

    # Add vertical cutlines
    ax.axvline(intden_low, color="#e11d48", linewidth=2.0, linestyle="--", label=f"Low Cut: {intden_low:.1f} ADU")
    ax.axvline(intden_high, color="#059669", linewidth=2.0, linestyle="--", label=f"High Cut: {intden_high:.1f} ADU")

    ax.set_xlabel("Integrated Density (ADU)", color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Counts (log scale)", color="#e2e8f0", fontsize=11)
    ax.set_title(f"Single-Photon IntDen Distribution ({len(int_den_values):,} clusters)", color="#f8fafc", fontsize=12)

    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#2d3561")

    ax.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax.legend(facecolor="#14172b", edgecolor="#2d3561", labelcolor="#f8fafc", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    fig.clear()

    return out_path


# ============================================================================
# Dark Diagnostics Data & Publication Plot Export
# ============================================================================

def export_dark_diagnostics_data(
    diagnostics: DarkDiagnostics,
    export_dir: str | Path,
    stddev_thresh: float = 40.0,
    absdev_thresh: float = 60.0,
    tail_ratio: float = 0.9333,
    bins: int = 60,
) -> dict[str, Path]:
    """Export dark diagnostics summary, histogram bin distributions, and per-pixel metrics to CSV.

    Args:
        diagnostics: DarkDiagnostics instance containing per_pixel_stddev and pct93_residual.
        export_dir: Destination directory path.
        stddev_thresh: Upper threshold for per-pixel standard deviation (ADU).
        absdev_thresh: Upper threshold for 93rd-percentile absolute residual (ADU).
        tail_ratio: Required fraction of stable frames.
        bins: Number of histogram bins.

    Returns:
        Dictionary mapping keys ('summary', 'stddev_bins', 'residual_bins', 'pixel_metrics') to Path objects.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    std_arr = diagnostics.per_pixel_stddev
    res_arr = diagnostics.pct93_residual

    total_pixels = int(std_arr.size)
    valid_mask = np.isfinite(std_arr) & np.isfinite(res_arr)
    valid_pixels = int(np.count_nonzero(valid_mask))

    std_finite = std_arr[np.isfinite(std_arr)]
    res_finite = res_arr[np.isfinite(res_arr)]

    # Dynamic histogram upper bounds matching UI logic
    std_max = max(100.0, float(np.percentile(std_finite, 99.9))) if len(std_finite) > 0 else 100.0
    res_max = max(150.0, float(np.percentile(res_finite, 99.9))) if len(res_finite) > 0 else 150.0

    # 1. Summary CSV
    m_t1 = (std_arr < stddev_thresh) & np.isfinite(std_arr)
    t1_surviving = int(np.count_nonzero(m_t1))
    t1_pct = (t1_surviving / total_pixels * 100.0) if total_pixels > 0 else 0.0

    m_t2 = (res_arr < absdev_thresh) & np.isfinite(res_arr)
    t2_surviving = int(np.count_nonzero(m_t2))
    t2_pct = (t2_surviving / total_pixels * 100.0) if total_pixels > 0 else 0.0

    m_final = m_t1 & m_t2
    final_active = int(np.count_nonzero(m_final))
    final_pct = (final_active / total_pixels * 100.0) if total_pixels > 0 else 0.0

    summary_rows = [
        {"Metric": "Dark Frames Count", "Value": str(diagnostics.dark_frame_count)},
        {"Metric": "Total Pixels", "Value": str(total_pixels)},
        {"Metric": "Valid Pixels", "Value": str(valid_pixels)},
        {"Metric": "StdDev Mean (ADU)", "Value": f"{float(np.mean(std_finite)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev Std (ADU)", "Value": f"{float(np.std(std_finite)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev Median (ADU)", "Value": f"{float(np.median(std_finite)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev Min (ADU)", "Value": f"{float(np.min(std_finite)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev Max (ADU)", "Value": f"{float(np.max(std_finite)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 25% (ADU)", "Value": f"{float(np.percentile(std_finite, 25)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 50% (ADU)", "Value": f"{float(np.percentile(std_finite, 50)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 75% (ADU)", "Value": f"{float(np.percentile(std_finite, 75)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 90% (ADU)", "Value": f"{float(np.percentile(std_finite, 90)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 95% (ADU)", "Value": f"{float(np.percentile(std_finite, 95)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 99% (ADU)", "Value": f"{float(np.percentile(std_finite, 99)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "StdDev 99.9% (ADU)", "Value": f"{float(np.percentile(std_finite, 99.9)):.4f}" if len(std_finite) > 0 else "0.0"},
        {"Metric": "Residual Mean (ADU)", "Value": f"{float(np.mean(res_finite)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual Std (ADU)", "Value": f"{float(np.std(res_finite)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual Median (ADU)", "Value": f"{float(np.median(res_finite)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual Min (ADU)", "Value": f"{float(np.min(res_finite)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual Max (ADU)", "Value": f"{float(np.max(res_finite)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 25% (ADU)", "Value": f"{float(np.percentile(res_finite, 25)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 50% (ADU)", "Value": f"{float(np.percentile(res_finite, 50)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 75% (ADU)", "Value": f"{float(np.percentile(res_finite, 75)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 90% (ADU)", "Value": f"{float(np.percentile(res_finite, 90)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 95% (ADU)", "Value": f"{float(np.percentile(res_finite, 95)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 99% (ADU)", "Value": f"{float(np.percentile(res_finite, 99)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Residual 99.9% (ADU)", "Value": f"{float(np.percentile(res_finite, 99.9)):.4f}" if len(res_finite) > 0 else "0.0"},
        {"Metric": "Tier 1 Threshold (ADU)", "Value": f"{stddev_thresh:.2f}"},
        {"Metric": "Tier 1 Surviving Pixels", "Value": str(t1_surviving)},
        {"Metric": "Tier 1 Surviving Pct (%)", "Value": f"{t1_pct:.2f}"},
        {"Metric": "Tier 2 Threshold (ADU)", "Value": f"{absdev_thresh:.2f}"},
        {"Metric": "Tier 2 Surviving Pixels", "Value": str(t2_surviving)},
        {"Metric": "Tier 2 Surviving Pct (%)", "Value": f"{t2_pct:.2f}"},
        {"Metric": "Final Mask Active Pixels", "Value": str(final_active)},
        {"Metric": "Final Mask Active Pct (%)", "Value": f"{final_pct:.2f}"},
    ]
    summary_path = out_dir / "dark_diagnostics_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # 2. StdDev Histogram Bins CSV
    std_counts, std_edges = np.histogram(std_finite, bins=bins, range=(0.0, std_max))
    std_starts = std_edges[:-1]
    std_ends = std_edges[1:]
    std_centers = (std_starts + std_ends) / 2.0
    std_log_counts = np.zeros(bins, dtype=np.float64)
    np.log10(std_counts, where=(std_counts > 0), out=std_log_counts)

    df_std_bins = pd.DataFrame({
        "bin_index": np.arange(bins, dtype=np.int64),
        "bin_start_adu": std_starts,
        "bin_end_adu": std_ends,
        "bin_center_adu": std_centers,
        "linear_count": std_counts,
        "log10_count": std_log_counts,
    })
    stddev_bins_path = out_dir / "dark_stddev_histogram_bins.csv"
    df_std_bins.to_csv(stddev_bins_path, index=False)

    # 3. Residual Histogram Bins CSV
    res_counts, res_edges = np.histogram(res_finite, bins=bins, range=(0.0, res_max))
    res_starts = res_edges[:-1]
    res_ends = res_edges[1:]
    res_centers = (res_starts + res_ends) / 2.0
    res_log_counts = np.zeros(bins, dtype=np.float64)
    np.log10(res_counts, where=(res_counts > 0), out=res_log_counts)

    df_res_bins = pd.DataFrame({
        "bin_index": np.arange(bins, dtype=np.int64),
        "bin_start_adu": res_starts,
        "bin_end_adu": res_ends,
        "bin_center_adu": res_centers,
        "linear_count": res_counts,
        "log10_count": res_log_counts,
    })
    residual_bins_path = out_dir / "dark_residual_histogram_bins.csv"
    df_res_bins.to_csv(residual_bins_path, index=False)

    # 4. Pixel Metrics CSV
    h, w = std_arr.shape
    rows_idx, cols_idx = np.indices((h, w))
    df_metrics = pd.DataFrame({
        "pixel_index": np.arange(total_pixels, dtype=np.int64),
        "row": rows_idx.ravel(),
        "col": cols_idx.ravel(),
        "stddev_adu": std_arr.ravel(),
        "residual_adu": res_arr.ravel(),
        "is_valid_mask": m_final.ravel().astype(np.int8),
    })
    pixel_metrics_path = out_dir / "dark_pixel_metrics.csv"
    df_metrics.to_csv(pixel_metrics_path, index=False)

    return {
        "summary": summary_path,
        "stddev_bins": stddev_bins_path,
        "residual_bins": residual_bins_path,
        "pixel_metrics": pixel_metrics_path,
    }


def export_dark_diagnostics_plots(
    diagnostics: DarkDiagnostics,
    export_dir: str | Path,
    bins: int = 60,
    dpi: int = 300,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Path]:
    """Export publication-grade scientific dark diagnostic histogram figures.

    Renders:
    - dark_histograms_publication.png (2-panel combined side-by-side)
    - dark_stddev_histogram_publication.png (standalone high-res StdDev plot)
    - dark_residual_histogram_publication.png (standalone high-res Residual plot)

    Publication styling:
    - Pure white background (facecolor="white")
    - Crisp spines (#334155), dark typography (#1e293b), subtle dashed grid (#cbd5e1)
    - Dual-axis: Log scale primary + Linear scale secondary overlay
    - Clean distribution without interactive cutlines or shaded spans

    Args:
        diagnostics: DarkDiagnostics instance containing per_pixel_stddev and pct93_residual.
        export_dir: Destination directory path.
        bins: Number of histogram bins.
        dpi: Output resolution (default 300 DPI for publication).
        progress_callback: Optional progress callback receiving (current_step, total_steps, message).

    Returns:
        Dictionary mapping keys ('combined_plot', 'stddev_plot', 'residual_plot') to Path objects.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    std_data = diagnostics.per_pixel_stddev.ravel()
    res_data = diagnostics.pct93_residual.ravel()

    std_finite = std_data[np.isfinite(std_data)]
    res_finite = res_data[np.isfinite(res_data)]

    std_max = max(100.0, float(np.percentile(std_finite, 99.9))) if len(std_finite) > 0 else 100.0
    res_max = max(150.0, float(np.percentile(res_finite, 99.9))) if len(res_finite) > 0 else 150.0

    def _draw_std_axis(ax_log, ax_lin) -> None:
        ax_log.set_facecolor("white")
        ax_log.set_title("Pixel Noise StdDev (σ)", color="#1e293b", fontsize=11, fontweight="bold", pad=8)
        ax_log.set_xlabel("Standard Deviation σ (ADU)", color="#1e293b", fontsize=10)
        ax_log.set_ylabel("Log Count", color="#0284c7", fontsize=10)
        ax_log.tick_params(colors="#0284c7", labelsize=9)
        for spine in ax_log.spines.values():
            spine.set_color("#334155")
        ax_log.grid(True, linestyle="--", color="#cbd5e1", alpha=0.5)

        ax_lin.set_ylabel("Linear Count", color="#4338ca", fontsize=10)
        ax_lin.tick_params(colors="#4338ca", labelsize=9)
        ax_lin.yaxis.tick_right()
        ax_lin.yaxis.set_label_position("right")
        ax_lin.yaxis.get_offset_text().set_color("#4338ca")
        ax_lin.yaxis.get_offset_text().set_fontsize(8)
        for spine in ax_lin.spines.values():
            spine.set_color("#334155")

        if len(std_finite) > 0:
            _, _, p_log = ax_log.hist(
                std_finite,
                bins=bins,
                range=(0, std_max),
                color="#0284c7",
                edgecolor="#0f172a",
                alpha=0.65,
                log=True,
                label="Log Count",
            )
            ax_lin.hist(
                std_finite,
                bins=bins,
                range=(0, std_max),
                histtype="stepfilled",
                color="#818cf8",
                alpha=0.25,
                log=False,
            )
            _, _, p_lin = ax_lin.hist(
                std_finite,
                bins=bins,
                range=(0, std_max),
                histtype="step",
                color="#4338ca",
                linewidth=1.5,
                log=False,
                label="Linear Count",
            )
            handles = []
            labels = []
            if len(p_log) > 0:
                handles.append(p_log[0])
                labels.append("Log Count")
            if len(p_lin) > 0:
                handles.append(p_lin[0])
                labels.append("Linear Count")
            if handles:
                ax_log.legend(
                    handles,
                    labels,
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    labelcolor="#1e293b",
                    fontsize=9,
                    loc="upper right",
                )

    def _draw_res_axis(ax_log, ax_lin) -> None:
        ax_log.set_facecolor("white")
        ax_log.set_title("93rd-Percentile Residual (Δ)", color="#1e293b", fontsize=11, fontweight="bold", pad=8)
        ax_log.set_xlabel("Excursion Residual Δ (ADU)", color="#1e293b", fontsize=10)
        ax_log.set_ylabel("Log Count", color="#d97706", fontsize=10)
        ax_log.tick_params(colors="#d97706", labelsize=9)
        for spine in ax_log.spines.values():
            spine.set_color("#334155")
        ax_log.grid(True, linestyle="--", color="#cbd5e1", alpha=0.5)

        ax_lin.set_ylabel("Linear Count", color="#c2410c", fontsize=10)
        ax_lin.tick_params(colors="#c2410c", labelsize=9)
        ax_lin.yaxis.tick_right()
        ax_lin.yaxis.set_label_position("right")
        ax_lin.yaxis.get_offset_text().set_color("#c2410c")
        ax_lin.yaxis.get_offset_text().set_fontsize(8)
        for spine in ax_lin.spines.values():
            spine.set_color("#334155")

        if len(res_finite) > 0:
            _, _, p_log = ax_log.hist(
                res_finite,
                bins=bins,
                range=(0, res_max),
                color="#d97706",
                edgecolor="#0f172a",
                alpha=0.65,
                log=True,
                label="Log Count",
            )
            ax_lin.hist(
                res_finite,
                bins=bins,
                range=(0, res_max),
                histtype="stepfilled",
                color="#fb923c",
                alpha=0.25,
                log=False,
            )
            _, _, p_lin = ax_lin.hist(
                res_finite,
                bins=bins,
                range=(0, res_max),
                histtype="step",
                color="#c2410c",
                linewidth=1.5,
                log=False,
                label="Linear Count",
            )
            handles = []
            labels = []
            if len(p_log) > 0:
                handles.append(p_log[0])
                labels.append("Log Count")
            if len(p_lin) > 0:
                handles.append(p_lin[0])
                labels.append("Linear Count")
            if handles:
                ax_log.legend(
                    handles,
                    labels,
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    labelcolor="#1e293b",
                    fontsize=9,
                    loc="upper right",
                )

    # 1. Standalone StdDev Plot
    fig_std = Figure(figsize=(6.5, 5), facecolor="white")
    _ = FigureCanvasAgg(fig_std)
    ax_std_solo = fig_std.add_subplot(111)
    ax_std_solo_lin = ax_std_solo.twinx()
    _draw_std_axis(ax_std_solo, ax_std_solo_lin)
    fig_std.tight_layout()
    stddev_path = out_dir / "dark_stddev_histogram_publication.png"
    fig_std.savefig(stddev_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig_std.clear()
    if progress_callback is not None:
        progress_callback(2, 4, "Rendering StdDev plot...")

    # 2. Standalone Residual Plot
    fig_res = Figure(figsize=(6.5, 5), facecolor="white")
    _ = FigureCanvasAgg(fig_res)
    ax_res_solo = fig_res.add_subplot(111)
    ax_res_solo_lin = ax_res_solo.twinx()
    _draw_res_axis(ax_res_solo, ax_res_solo_lin)
    fig_res.tight_layout()
    residual_path = out_dir / "dark_residual_histogram_publication.png"
    fig_res.savefig(residual_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig_res.clear()
    if progress_callback is not None:
        progress_callback(3, 4, "Rendering Residual plot...")

    # 3. 2-Panel Side-by-Side Combined Plot
    fig_comb = Figure(figsize=(12, 5), facecolor="white")
    _ = FigureCanvasAgg(fig_comb)
    ax_std_comb = fig_comb.add_subplot(121)
    ax_res_comb = fig_comb.add_subplot(122)
    ax_std_comb_lin = ax_std_comb.twinx()
    ax_res_comb_lin = ax_res_comb.twinx()

    _draw_std_axis(ax_std_comb, ax_std_comb_lin)
    _draw_res_axis(ax_res_comb, ax_res_comb_lin)

    fig_comb.tight_layout()
    combined_path = out_dir / "dark_histograms_publication.png"
    fig_comb.savefig(combined_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig_comb.clear()
    if progress_callback is not None:
        progress_callback(4, 4, "Rendering Combined plot...")

    return {
        "combined_plot": combined_path,
        "stddev_plot": stddev_path,
        "residual_plot": residual_path,
    }


def export_dark_diagnostics(
    diagnostics: DarkDiagnostics,
    export_dir: str | Path,
    stddev_thresh: float = 40.0,
    absdev_thresh: float = 60.0,
    tail_ratio: float = 0.9333,
    bins: int = 60,
    dpi: int = 300,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Path]:
    """Comprehensive export of dark frame diagnostics: summary metrics, histogram data, and publication plots.

    Args:
        diagnostics: DarkDiagnostics instance containing per_pixel_stddev and pct93_residual.
        export_dir: Destination directory path.
        stddev_thresh: Upper threshold for per-pixel standard deviation (ADU).
        absdev_thresh: Upper threshold for 93rd-percentile absolute residual (ADU).
        tail_ratio: Required fraction of stable frames.
        bins: Number of histogram bins.
        dpi: Output resolution for publication figures.
        progress_callback: Optional progress callback receiving (current_step, total_steps, message).

    Returns:
        Dictionary mapping all export keys ('summary', 'stddev_bins', 'residual_bins', 'pixel_metrics',
        'combined_plot', 'stddev_plot', 'residual_plot') to their respective Path objects.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_exports = export_dark_diagnostics_data(
        diagnostics=diagnostics,
        export_dir=out_dir,
        stddev_thresh=stddev_thresh,
        absdev_thresh=absdev_thresh,
        tail_ratio=tail_ratio,
        bins=bins,
    )
    if progress_callback is not None:
        progress_callback(1, 4, "Saving CSV data...")

    plot_exports = export_dark_diagnostics_plots(
        diagnostics=diagnostics,
        export_dir=out_dir,
        bins=bins,
        dpi=dpi,
        progress_callback=progress_callback,
    )

    return {**data_exports, **plot_exports}


