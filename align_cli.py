#!/usr/bin/env python3
"""Headless CLI for spectroscopy TIFF frame alignment.

This script provides a standalone, non-interactive interface to the alignment
pipeline that powers the GUI.  It processes one or more directories of TIFF
files, computes per-frame offsets with selectable engines (ECC, PCA,
PhaseCorrelation), and writes aligned sum images, offset logs, and optional
comparison PNGs — all without any Tkinter / CustomTkinter dependency.

Typical usage::

    python align_cli.py -d ./data -r -e ECC PCA --png
    python align_cli.py -d ./run42 -e PCA -t auto --overwrite
"""

# ── Headless Matplotlib backend ──────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # MUST be before any pyplot import
import matplotlib.pyplot as plt

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import tifffile
import cv2

# ── Project-internal (core only — no GUI) ────────────────────────────────────
from align_app.core import (
    natural_sort,
    find_peak_line,
    find_peak_line_fast,
    compute_line_based_offset,
    phase_correlation_offset,
    ecc_maximization_offset,
    precompute_ecc_reference,
    compute_alignment_priors,
    generate_aligned_sum,
    generate_direct_sum,
    PCAFitFailure,
    find_best_threshold,
)
from align_app.dataset import ZarrSequenceManager, CLIZarrSequenceManager, _frame_key



# ─────────────────────────────────────────────────────────────────────────────
# Comparison PNG helper
# ─────────────────────────────────────────────────────────────────────────────

def _contrast_vmax(img: np.ndarray) -> float:
    """Compute the 60th-percentile contrast ceiling for display.

    Filters out background pixels (those equal to the global minimum),
    then returns the 60th percentile of the remaining *active* pixels.
    Falls back to ``max(img)`` when there are no active pixels.

    Args:
        img: 2-D float array representing a summed image.

    Returns:
        Intensity value to use as ``vmax`` for display scaling.
    """
    img_min = np.min(img)
    active = img[img > img_min]
    if active.size == 0:
        return float(np.max(img))
    return float(np.percentile(active, 60))


def save_comparison_png(
    direct_sum: np.ndarray,
    aligned_sum: np.ndarray,
    engine_name: str,
    output_path: str,
) -> None:
    """Save a side-by-side comparison PNG of unaligned vs aligned sums.

    Uses the 60th-percentile contrast scaling (identical to the GUI)
    for each panel independently.  Saved at 150 DPI with tight bounding
    box.

    Args:
        direct_sum: 2-D float array of the direct (unaligned) sum.
        aligned_sum: 2-D float array of the engine-aligned sum.
        engine_name: Engine label used in the subplot title (e.g. "ECC").
        output_path: Absolute path for the output ``.png`` file.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    vmax_direct = _contrast_vmax(direct_sum)
    vmax_aligned = _contrast_vmax(aligned_sum)

    ax_left.imshow(
        direct_sum,
        cmap='gray',
        vmin=np.min(direct_sum),
        vmax=vmax_direct,
        aspect='equal',
    )
    ax_left.set_title("Direct Sum (Unaligned)")
    ax_left.axis('off')

    ax_right.imshow(
        aligned_sum,
        cmap='gray',
        vmin=np.min(aligned_sum),
        vmax=vmax_aligned,
        aspect='equal',
    )
    ax_right.set_title(f"Aligned Sum ({engine_name})")
    ax_right.axis('off')

    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Directory discovery
# ─────────────────────────────────────────────────────────────────────────────

def _has_tif_files(directory: Path, min_count: int = 2) -> bool:
    """Check whether *directory* contains at least *min_count* TIF files.

    Args:
        directory: Path object pointing to a directory.
        min_count: Minimum number of ``.tif`` / ``.tiff`` files required.

    Returns:
        ``True`` if the directory has enough TIF files, ``False`` otherwise.
    """
    count = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            count += 1
            if count >= min_count:
                return True
    return False


def discover_directories(root_dir: str, recursive: bool) -> list[str]:
    """Return directories under *root_dir* that contain ≥ 2 TIF files.

    In non-recursive mode only *root_dir* itself is considered.  In
    recursive mode every subdirectory (at any depth) is scanned too.

    Args:
        root_dir: Absolute path to the root scan directory.
        recursive: If ``True``, recurse into all subdirectories.

    Returns:
        Sorted list of absolute directory paths with sufficient TIFs.
    """
    root = Path(root_dir).resolve()
    result: list[str] = []

    if _has_tif_files(root):
        result.append(str(root))

    if recursive:
        for dirpath, dirnames, _filenames in os.walk(str(root)):
            # Modify dirnames in-place to ignore 'sum' and 'tif-cache'
            dirnames[:] = [d for d in dirnames if d.lower() not in ('sum', 'tif-cache')]
            dp = Path(dirpath).resolve()
            if dp == root:
                continue  # already checked
            if dp.name.lower() in ('sum', 'tif-cache'):
                continue
            if _has_tif_files(dp):
                result.append(str(dp))

    result.sort()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-directory processing
# ─────────────────────────────────────────────────────────────────────────────

def _glob_tifs(directory: str) -> list[str]:
    """Glob for .tif / .tiff files (case-insensitive) in *directory*.

    Args:
        directory: Absolute path to scan.

    Returns:
        Naturally sorted list of absolute TIFF paths.
    """
    d = Path(directory)
    files: list[str] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            files.append(str(p.resolve()))
    natural_sort(files)
    return files


def process_directory(
    dir_path: str,
    engines: list[str],
    threshold: str,
    save_png: bool,
    overwrite: bool,
    ephemeral_cache: bool = False,
    save_json: bool = False,
) -> None:
    """Run the full alignment pipeline on a single directory.

    Steps:
        1. Glob and sort TIF files.
        2. Create a :class:`CLIZarrSequenceManager` (synchronous caching).
        3. Select the reference frame (median, fallback to frame 0).
        4. Compute the direct (unaligned) sum.
        5. For each engine compute offsets, aligned sum, and save outputs.

    Args:
        dir_path: Absolute path to the directory containing TIF files.
        engines: List of engine names (``'ECC'``, ``'PCA'``,
            ``'PhaseCorrelation'``).
        threshold: PCA percentile threshold as a string (e.g. ``'99.9'``)
            or ``'auto'`` for automatic sweep.
        save_png: If ``True``, write a comparison PNG per engine.
        overwrite: If ``True``, overwrite existing output files.
        ephemeral_cache: If ``True``, delete the ``tif-cache/`` directory
            after processing completes.
        save_json: If ``True``, save computed offsets as a JSON file.
    """
    dir_name = os.path.basename(dir_path)
    print(f"\n{'='*60}")
    print(f"[{dir_name}] Scanning for TIF files…")

    tif_files = _glob_tifs(dir_path)
    if len(tif_files) < 2:
        print(f"[{dir_name}] Warning: fewer than 2 TIF files found — skipping.")
        return

    print(f"[{dir_name}] Found {len(tif_files)} TIF files.")

    # ── Zarr caching ─────────────────────────────────────────────────────
    print(f"[{dir_name}] Caching frames…")
    zarr_manager = CLIZarrSequenceManager(tif_files)

    ref_raw = zarr_manager.get_frame(0)
    ref_mode = "frame0"
    if ref_raw is None:
        print(f"[{dir_name}] Error: cannot load any frames — skipping.",
              file=sys.stderr)
        return

    ref_shape = ref_raw.shape

    # ── Output directory ─────────────────────────────────────────────────
    sum_dir = os.path.join(dir_path, "sum")
    os.makedirs(sum_dir, exist_ok=True)

    # ── Direct (unaligned) sum ───────────────────────────────────────────
    base_sum_path = os.path.join(sum_dir, "base_sum.tif")
    if not os.path.exists(base_sum_path) or overwrite:
        print(f"[{dir_name}] Computing direct sum…")

        def _get_raw(fpath):
            """Retrieve a raw frame from the Zarr cache or disk.

            Args:
                fpath: Absolute path to a TIFF file.

            Returns:
                2-D ``float32`` numpy array.
            """
            key = _frame_key(fpath)
            if key in zarr_manager.zarr_group:
                return zarr_manager.zarr_group[key][:]
            raw = tifffile.imread(fpath).astype(np.float32)
            return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        direct_sum = generate_direct_sum(tif_files, _get_raw, ref_shape)
        tifffile.imwrite(base_sum_path, direct_sum)
    else:
        print(f"[{dir_name}] Direct sum already exists — loading.")
        direct_sum = tifffile.imread(base_sum_path).astype(np.float32)

    # ── Helper to read a raw frame ───────────────────────────────────────
    def get_raw(fpath):
        """Retrieve a raw frame for alignment computation.

        Checks the Zarr group first, falling back to a live TIFF read.

        Args:
            fpath: Absolute path to a TIFF file.

        Returns:
            2-D ``float32`` numpy array, or ``None`` on failure.
        """
        key = _frame_key(fpath)
        if key in zarr_manager.zarr_group:
            return zarr_manager.zarr_group[key][:]
        try:
            raw = tifffile.imread(fpath).astype(np.float32)
            return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"  Error reading {os.path.basename(fpath)}: {e}",
                  file=sys.stderr)
            return None

    # ── Engine loop ──────────────────────────────────────────────────────
    for engine in engines:
        print(f"\n[{dir_name}] Running engine: {engine}")

        aligned_path = os.path.join(sum_dir, f"aligned_sum_{engine}.tif")
        json_path = os.path.join(sum_dir, f"aligned_offsets_{engine}.json")
        png_path = os.path.join(sum_dir, f"comparison_{engine}.png")

        if not overwrite:
            if os.path.exists(aligned_path) and (not save_json or os.path.exists(json_path)):
                print(f"[{dir_name}] Outputs for {engine} already exist — "
                      "skipping (use --overwrite to recompute).")
                continue

        n_frames = len(tif_files)
        offsets: dict[int, tuple[float, float]] = {}
        
        # ECC: pre-compute priors
        ecc_crop_bounds = None
        ecc_drift_vector = None
        ref_ecc_pyr = None
        if engine == 'ECC':
            print(f"[{dir_name}] Computing sample-agnostic alignment priors for ECC...")
            num_prior_frames = min(10, max(1, n_frames // 2))
            if num_prior_frames > 0 and n_frames > 1:
                early_frames = []
                late_frames = []
                for i in range(num_prior_frames):
                    f = get_raw(tif_files[i])
                    if f is not None: early_frames.append(f)
                for i in range(n_frames - num_prior_frames, n_frames):
                    f = get_raw(tif_files[i])
                    if f is not None: late_frames.append(f)
                ecc_crop_bounds, ecc_drift_vector = compute_alignment_priors(early_frames, late_frames)
                print(f"[{dir_name}] Computed Crop Bounds: {ecc_crop_bounds}")
                print(f"[{dir_name}] Computed Drift Vector: {ecc_drift_vector}")
            
            ref_ecc_pyr = precompute_ecc_reference(ref_raw, ecc_crop_bounds)


        # PCA: pre-compute reference line
        pca_ref_origin = None
        pca_ref_direction = None
        pca_threshold_val: float | None = None

        if engine == 'PCA':
            # Parse threshold
            if threshold == 'auto':
                pca_threshold_val = None  # will be computed per-frame
            else:
                pca_threshold_val = float(threshold)

            # Fit reference line with either auto or fixed threshold
            ref_thresh = pca_threshold_val
            if ref_thresh is None:
                ref_thresh = find_best_threshold(ref_raw)
                print(f"[{dir_name}] Auto-threshold for reference: {ref_thresh:.4f}%")

            try:
                pca_ref_origin, pca_ref_direction = find_peak_line(
                    ref_raw, ref_thresh
                )
            except PCAFitFailure as e:
                print(f"[{dir_name}] PCA reference fit failed: {e} — "
                      f"skipping {engine}.", file=sys.stderr)
                continue

        # Compute offsets for each frame (frame 0 is the reference)
        offsets[0] = (0.0, 0.0)
        
        cv2.setNumThreads(1)
        import concurrent.futures
        
        def process_frame(idx):
            raw = get_raw(tif_files[idx])
            if raw is None:
                print(f"[{dir_name}] Warning: frame {idx} unreadable — using (0, 0).", file=sys.stderr)
                return idx, (0.0, 0.0)

            # Shape mismatch guard
            if raw.shape != ref_shape:
                print(f"[{dir_name}] Warning: frame {idx} shape {raw.shape} != reference {ref_shape} — using (0, 0).", file=sys.stderr)
                return idx, (0.0, 0.0)

            try:
                if engine == 'ECC':
                    dx, dy = ecc_maximization_offset(ref_ecc_pyr, raw, crop_bounds=ecc_crop_bounds, drift_vector=ecc_drift_vector)

                elif engine == 'PhaseCorrelation':
                    dx, dy = phase_correlation_offset(ref_raw, raw)

                elif engine == 'PCA':
                    target_thresh = pca_threshold_val
                    if target_thresh is None:
                        target_thresh = find_best_threshold(raw)

                    dx, dy = compute_line_based_offset(
                        ref_raw,
                        raw,
                        pca_ref_direction,
                        pca_ref_origin,
                        ref_thresh if pca_threshold_val is None else pca_threshold_val,
                        target_thresh,
                    )
                else:
                    dx, dy = 0.0, 0.0

            except PCAFitFailure as e:
                print(f"[{dir_name}] PCA fit failure on frame {idx}: {e} — falling back to (0, 0).", file=sys.stderr)
                dx, dy = 0.0, 0.0
            except Exception as e:
                print(f"[{dir_name}] Error computing offset for frame {idx} ({engine}): {e} — falling back to (0, 0).", file=sys.stderr)
                dx, dy = 0.0, 0.0

            return idx, (float(dx), float(dy))

        max_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_frame, i): i for i in range(1, n_frames)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res_idx, offset = future.result()
                    offsets[res_idx] = offset
                    print(f"[{dir_name}] Finished frame {res_idx + 1}/{n_frames} ({engine})…")
                except Exception:
                    offsets[idx] = (0.0, 0.0)

        # ── Generate aligned sum ─────────────────────────────────────────
        print(f"[{dir_name}] Generating aligned sum ({engine})…")
        aligned_sum = generate_aligned_sum(
            tif_files, get_raw, offsets, ref_shape
        )
        tifffile.imwrite(aligned_path, aligned_sum)

        # ── Save offset JSON ─────────────────────────────────────────────
        if save_json:
            threshold_log = threshold
            if engine != 'PCA':
                threshold_log = "N/A"

            offsets_json = {
                "engine": engine,
                "threshold": threshold_log,
                "ref_mode": ref_mode,
                "timestamp": datetime.datetime.now().isoformat(),
                "offsets": {
                    str(k): [v[0], v[1]] for k, v in offsets.items()
                },
            }
            with open(json_path, 'w') as f:
                json.dump(offsets_json, f, indent=2)

            print(f"[{dir_name}] Saved: {os.path.basename(aligned_path)}")
            print(f"[{dir_name}] Saved: {os.path.basename(json_path)}")
        else:
            print(f"[{dir_name}] Saved: {os.path.basename(aligned_path)}")

        # ── Comparison PNG ───────────────────────────────────────────────
        if save_png:
            print(f"[{dir_name}] Saving comparison PNG ({engine})…")
            save_comparison_png(direct_sum, aligned_sum, engine, png_path)
            print(f"[{dir_name}] Saved: {os.path.basename(png_path)}")

    # ── Ephemeral cache cleanup ──────────────────────────────────────────
    if ephemeral_cache:
        cache_dir = os.path.join(dir_path, "tif-cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[{dir_name}] Removed tif-cache/")

    print(f"[{dir_name}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing & entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the alignment CLI.

    Args:
        argv: Optional list of argument strings.  Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Headless TIFF alignment CLI — align spectroscopy frames "
                    "using ECC, PCA, or Phase Correlation engines.",
    )
    parser.add_argument(
        '-d', '--dir',
        type=str,
        required=True,
        help="Root directory containing TIFF files or subdirectories.",
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        '-e', '--engines',
        nargs='+',
        choices=['ECC', 'PCA', 'PhaseCorrelation', 'all'],
        default=['ECC'],
        help="Alignment engine(s) to run.  'all' expands to all three.",
    )
    parser.add_argument(
        '-t', '--threshold',
        type=str,
        default='99.9',
        help="PCA percentile threshold.  Use 'auto' for automatic sweep.",
    )
    parser.add_argument(
        '--png',
        action='store_true',
        help="Save comparison PNGs (Direct Sum vs Aligned Sum).",
    )
    parser.add_argument(
        '--ephemeral-cache',
        action='store_true',
        help="Delete tif-cache/ directories after processing.",
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing sum/ output files.",
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Save offsets to a JSON file.",
    )

    args = parser.parse_args(argv)

    # ── Validate threshold ───────────────────────────────────────────────
    if args.threshold != 'auto':
        try:
            val = float(args.threshold)
            if not (0.0 <= val <= 100.0):
                parser.error(
                    f"--threshold must be between 0.0 and 100.0, got {val}"
                )
        except ValueError:
            parser.error(
                f"--threshold must be a number or 'auto', got '{args.threshold}'"
            )

    # ── Expand 'all' ─────────────────────────────────────────────────────
    if 'all' in args.engines:
        args.engines = ['ECC', 'PCA', 'PhaseCorrelation']

    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point for the headless CLI.

    Parses arguments, discovers directories, and runs
    :func:`process_directory` on each one.

    Args:
        argv: Optional argument list (for testing).  Defaults to
            ``sys.argv[1:]``.
    """
    args = _parse_args(argv)

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    dirs = discover_directories(root, args.recursive)
    if not dirs:
        print("No directories with ≥ 2 TIF files found.")
        sys.exit(0)

    print(f"Found {len(dirs)} directory(ies) to process.")

    for d in dirs:
        process_directory(
            dir_path=d,
            engines=args.engines,
            threshold=args.threshold,
            save_png=args.png,
            overwrite=args.overwrite,
            ephemeral_cache=args.ephemeral_cache,
            save_json=args.json,
        )

    print("\n✔ All directories processed.")


if __name__ == '__main__':
    main()
