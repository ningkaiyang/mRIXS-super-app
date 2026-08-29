#!/usr/bin/env python3
"""Headless CLI tool for single-photon event extraction (Stages 1, 2, 3) at ALS Beamline 6.0.2.

Provides 4 execution subcommands:
1. dark-mask: Stage 1 temporal dark median and 2-tier noise mask generation.
2. cluster:   Stage 2 8-connected component cluster analysis on raw signal frames.
3. reconstruct: Stage 3 event map filtering and super-resolution 2D reconstruction from TSV/XLS.
4. full:      End-to-end chained execution of Stages 1, 2, and 3.

Usage Examples:
    python cluster_cli.py dark-mask -d /path/to/dark -o /path/to/out
    python cluster_cli.py cluster -s /path/to/signal --dark-tif /path/to/MED_Dark.tif --mask-tif /path/to/Final_Mask.tif
    python cluster_cli.py reconstruct -c /path/to/Results_clusters.xls --intden-low 120 --intden-high 320
    python cluster_cli.py full -d /path/to/dark -s /path/to/signal --intden-low 120 --intden-high 320
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tifffile

from rixs_app.core.cli_utils import glob_tifs
from rixs_app.core.photon_clustering import (
    ClusterConfig,
    DarkMaskConfig,
    ReconstructionConfig,
    compute_dark_mask,
    export_intden_histogram,
    process_signal_stack_clusters,
    reconstruct_photon_event_map,
)

# Unbuffered real-time stdout streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def run_dark_mask(args: argparse.Namespace) -> int:
    """Execute Stage 1 dark baseline & mask generation."""
    dark_dir = Path(args.dark_dir)
    if not dark_dir.is_dir():
        print(f"Error: Dark directory not found: {dark_dir}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dark_dir / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    dark_files = glob_tifs(dark_dir)
    if not dark_files:
        print(f"Error: No TIFF files found in dark directory: {dark_dir}", file=sys.stderr)
        return 1

    print("========================================")
    print("Stage 1: Dark Mask Generation")
    print("========================================")
    print(f"Dark folder:        {dark_dir}")
    print(f"Dark frames count:  {len(dark_files)}")
    print(f"StdDev threshold:   {args.stddev_thresh} ADU")
    print(f"AbsDev threshold:   {args.absdev_thresh} ADU")
    print(f"Tail ratio cutoff:  {args.tail_thresh_ratio:.4f}")
    print("----------------------------------------")

    config = DarkMaskConfig(
        stddev_thresh=args.stddev_thresh,
        absdev_thresh=args.absdev_thresh,
        tail_thresh_ratio=args.tail_thresh_ratio,
        max_frames=args.max_frames,
    )

    t0 = time.perf_counter()
    result = compute_dark_mask(dark_files, config=config)
    elapsed = time.perf_counter() - t0

    label = args.label or "Dark"
    med_path = output_dir / f"MED_{label}.tif"
    mask_path = output_dir / f"Final_Mask_{label}.tif"
    med_std = output_dir / "MED_Dark.tif"
    mask_std = output_dir / "Final_Mask.tif"

    tifffile.imwrite(med_path, result.med_dark)
    tifffile.imwrite(mask_path, result.final_mask)
    if med_path != med_std:
        tifffile.imwrite(med_std, result.med_dark)
    if mask_path != mask_std:
        tifffile.imwrite(mask_std, result.final_mask)

    print(f"Stage 1 complete in {elapsed:.2f}s ({len(dark_files) / max(elapsed, 1e-6):.1f} fps).")
    print(f"  Surviving active pixels: {result.surviving_pixels:,} / {result.total_pixels:,}")
    print(f"  Suppression rate:        {result.suppression_pct:.2f}%")
    print(f"  Median dark saved:       {med_path}")
    print(f"  Final mask saved:        {mask_path}")
    print("========================================\n")
    return 0


def run_cluster(args: argparse.Namespace) -> int:
    """Execute Stage 2 8-connected cluster analysis on signal frames."""
    signal_dir = Path(args.signal_dir)
    if not signal_dir.is_dir():
        print(f"Error: Signal directory not found: {signal_dir}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = signal_dir / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    dark_path_str = getattr(args, "dark_tif", None) or getattr(args, "med_dark", None)
    mask_path_str = getattr(args, "mask_tif", None) or getattr(args, "final_mask", None)

    if not dark_path_str:
        candidate_dark = output_dir / "MED_Dark.tif"
        if candidate_dark.is_file():
            dark_path_str = str(candidate_dark)
        else:
            print("Error: Median dark TIFF not specified and not found in output directory", file=sys.stderr)
            return 1

    if not mask_path_str:
        candidate_mask = output_dir / "Final_Mask.tif"
        if not candidate_mask.is_file():
            candidate_mask = output_dir / "Final_Mask_Dark.tif"
        if candidate_mask.is_file():
            mask_path_str = str(candidate_mask)
        else:
            print("Error: Final mask TIFF not specified and not found in output directory", file=sys.stderr)
            return 1

    dark_tif = Path(dark_path_str)
    mask_tif = Path(mask_path_str)
    if not dark_tif.is_file():
        print(f"Error: Median dark TIFF not found: {dark_tif}", file=sys.stderr)
        return 1
    if not mask_tif.is_file():
        print(f"Error: Final mask TIFF not found: {mask_tif}", file=sys.stderr)
        return 1

    signal_files = glob_tifs(signal_dir)
    if not signal_files:
        print(f"Error: No TIFF files found in signal directory: {signal_dir}", file=sys.stderr)
        return 1

    med_dark = tifffile.imread(dark_tif).astype(np.float32)
    final_mask = tifffile.imread(mask_tif).astype(np.float32)

    print("========================================")
    print("Stage 2: 8-Connected Cluster Analysis")
    print("========================================")
    print(f"Signal folder:      {signal_dir}")
    print(f"Signal frames:      {len(signal_files)}")
    print(f"Signal cutoff:      {args.sig_thresh_low} - {args.sig_thresh_high} ADU")
    print("----------------------------------------")

    config = ClusterConfig(
        sig_thresh_low=args.sig_thresh_low,
        sig_thresh_high=args.sig_thresh_high,
        connectivity=8,
    )

    t0 = time.perf_counter()
    df_clusters = process_signal_stack_clusters(
        signal_paths=signal_files,
        med_dark=med_dark,
        final_mask=final_mask,
        config=config,
    )
    elapsed = time.perf_counter() - t0

    out_name = args.output_name or "Results_clusters.xls"
    out_path = output_dir / out_name
    # Save as Tab-Separated Values (TSV) matching ImageJ .xls format
    df_clusters.to_csv(out_path, sep="\t", index=False)

    print(f"Stage 2 complete in {elapsed:.2f}s ({len(signal_files) / max(elapsed, 1e-6):.1f} fps).")
    print(f"  Total clusters detected: {len(df_clusters):,}")
    print(f"  Cluster results saved:   {out_path}")
    print("========================================\n")
    return 0


def run_reconstruct(args: argparse.Namespace) -> int:
    """Execute Stage 3 photon event map reconstruction from cluster results."""
    clusters_path = Path(args.clusters_xls)
    if not clusters_path.is_file():
        print(f"Error: Cluster results spreadsheet not found: {clusters_path}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = clusters_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TSV / CSV
    sep = "\t" if clusters_path.suffix.lower() in [".xls", ".tsv"] else ","
    try:
        df_clusters = pd.read_csv(clusters_path, sep=sep)
    except Exception:
        df_clusters = pd.read_csv(clusters_path, sep=r"\s+")

    print("========================================")
    print("Stage 3: Photon Event Map Reconstruction")
    print("========================================")
    print(f"Cluster file:       {clusters_path} ({len(df_clusters):,} clusters)")
    print(f"IntDen cuts:        [{args.intden_low}, {args.intden_high}] ADU")
    print(f"Max Area:           {args.max_area} pixels")
    print(f"Min Circularity:    {args.min_circ}")
    print(f"Sub-pixel factor:   {args.subpixel}x")
    print("----------------------------------------")

    config = ReconstructionConfig(
        intden_low=args.intden_low,
        intden_high=args.intden_high,
        max_area=args.max_area,
        min_circ=args.min_circ,
        subpixel_factor=args.subpixel,
    )

    image_shape = (args.image_height, args.image_width)
    if getattr(args, "signal_dir", None):
        sig_dir = Path(args.signal_dir)
        if sig_dir.is_dir():
            s_files = glob_tifs(sig_dir)
            if s_files:
                first_sig = tifffile.imread(s_files[0])
                image_shape = first_sig.shape

    result = reconstruct_photon_event_map(df_clusters, image_shape=image_shape, config=config)

    map_name = args.output_name or "Photon_Event_Map.tif"
    map_path = output_dir / map_name
    map_total_path = output_dir / "Photon_Event_Map_total.tif"

    tifffile.imwrite(map_path, result.event_map)
    tifffile.imwrite(map_total_path, result.event_map)

    hist_path = output_dir / "IntDen_histogram.png"
    export_intden_histogram(
        df_clusters=df_clusters,
        output_png=hist_path,
        intden_low=args.intden_low,
        intden_high=args.intden_high,
    )

    # Chunk reconstruction if requested
    chunk_size = getattr(args, "chunk_size", 0)
    if chunk_size > 0 and not df_clusters.empty and "Slice" in df_clusters.columns:
        total_slices = int(df_clusters["Slice"].max())
        num_chunks = int(np.ceil(total_slices / chunk_size))
        for c in range(num_chunks):
            start_f = c * chunk_size + 1
            end_f = min((c + 1) * chunk_size, total_slices)
            df_chunk = df_clusters[
                (df_clusters["Slice"] >= start_f) & (df_clusters["Slice"] <= end_f)
            ]
            c_res = reconstruct_photon_event_map(df_chunk, image_shape=image_shape, config=config)
            c_path = output_dir / f"Photon_Event_Map_frames_{start_f}-{end_f}.tif"
            tifffile.imwrite(c_path, c_res.event_map)

    print("Stage 3 complete.")
    print(f"  Accepted events:  {result.accepted_events:,} ({result.acceptance_pct:.1f}%)")
    print(f"  Rejected noise:   {result.rejected_noise:,}")
    print(f"  Rejected pileup:  {result.rejected_pileup:,}")
    print(f"  Rejected shape:   {result.rejected_shape:,}")
    print(f"  Event map saved:  {map_path}")
    print(f"  Histogram saved:  {hist_path}")
    print("========================================\n")
    return 0


def run_full(args: argparse.Namespace) -> int:
    """Execute Stages 1, 2, and 3 end-to-end."""
    dark_dir = Path(args.dark_dir)
    signal_dir = Path(args.signal_dir)

    if not dark_dir.is_dir():
        print(f"Error: Dark directory not found: {dark_dir}", file=sys.stderr)
        return 1
    if not signal_dir.is_dir():
        print(f"Error: Signal directory not found: {signal_dir}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = signal_dir / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    dark_files = glob_tifs(dark_dir)
    signal_files = glob_tifs(signal_dir)
    if not dark_files:
        print(f"Error: No dark TIFF files in: {dark_dir}", file=sys.stderr)
        return 1
    if not signal_files:
        print(f"Error: No signal TIFF files in: {signal_dir}", file=sys.stderr)
        return 1

    print("============================================================")
    print("QERLIN Full 3-Stage Single-Photon Event Extraction Pipeline")
    print("============================================================")

    # 1. Stage 1
    dark_config = DarkMaskConfig(
        stddev_thresh=args.stddev_thresh,
        absdev_thresh=args.absdev_thresh,
        tail_thresh_ratio=args.tail_thresh_ratio,
        max_frames=args.max_frames,
    )
    t0 = time.perf_counter()
    stage1_res = compute_dark_mask(dark_files, config=dark_config)
    t1 = time.perf_counter()

    label = args.label or "Dark"
    med_path = output_dir / f"MED_{label}.tif"
    mask_path = output_dir / f"Final_Mask_{label}.tif"
    med_std = output_dir / "MED_Dark.tif"
    mask_std = output_dir / "Final_Mask.tif"
    mask_dark = output_dir / "Final_Mask_Dark.tif"

    tifffile.imwrite(med_path, stage1_res.med_dark)
    tifffile.imwrite(mask_path, stage1_res.final_mask)
    tifffile.imwrite(med_std, stage1_res.med_dark)
    tifffile.imwrite(mask_std, stage1_res.final_mask)
    tifffile.imwrite(mask_dark, stage1_res.final_mask)

    print(f"✓ Stage 1 complete in {t1 - t0:.2f}s: {stage1_res.surviving_pixels:,} surviving active pixels.")

    # 2. Stage 2
    cluster_config = ClusterConfig(
        sig_thresh_low=args.sig_thresh_low,
        sig_thresh_high=args.sig_thresh_high,
        connectivity=8,
    )
    df_clusters = process_signal_stack_clusters(
        signal_paths=signal_files,
        med_dark=stage1_res.med_dark,
        final_mask=stage1_res.final_mask,
        config=cluster_config,
    )
    t2 = time.perf_counter()

    cluster_xls = output_dir / "Results_clusters.xls"
    df_clusters.to_csv(cluster_xls, sep="\t", index=False)
    print(f"✓ Stage 2 complete in {t2 - t1:.2f}s: {len(df_clusters):,} clusters found across {len(signal_files)} frames.")

    # 3. Stage 3
    first_sig = tifffile.imread(signal_files[0])
    image_shape = first_sig.shape

    recon_config = ReconstructionConfig(
        intden_low=args.intden_low,
        intden_high=args.intden_high,
        max_area=args.max_area,
        min_circ=args.min_circ,
        subpixel_factor=args.subpixel,
    )
    recon_res = reconstruct_photon_event_map(df_clusters, image_shape=image_shape, config=recon_config)
    t3 = time.perf_counter()

    map_path = output_dir / (args.output_name or "Photon_Event_Map.tif")
    map_total = output_dir / "Photon_Event_Map_total.tif"
    tifffile.imwrite(map_path, recon_res.event_map)
    tifffile.imwrite(map_total, recon_res.event_map)

    hist_path = output_dir / "IntDen_histogram.png"
    export_intden_histogram(
        df_clusters=df_clusters,
        output_png=hist_path,
        intden_low=args.intden_low,
        intden_high=args.intden_high,
    )

    # Chunk reconstruction if requested
    chunk_size = getattr(args, "chunk_size", 0)
    if chunk_size > 0 and len(signal_files) > 0:
        total_frames = len(signal_files)
        num_chunks = int(np.ceil(total_frames / chunk_size))
        for c in range(num_chunks):
            start_f = c * chunk_size + 1
            end_f = min((c + 1) * chunk_size, total_frames)
            df_chunk = df_clusters[
                (df_clusters["Slice"] >= start_f) & (df_clusters["Slice"] <= end_f)
            ]
            c_res = reconstruct_photon_event_map(df_chunk, image_shape=image_shape, config=recon_config)
            c_path = output_dir / f"Photon_Event_Map_frames_{start_f}-{end_f}.tif"
            tifffile.imwrite(c_path, c_res.event_map)

    print(f"✓ Stage 3 complete in {t3 - t2:.2f}s: {recon_res.accepted_events:,} events mapped ({recon_res.acceptance_pct:.1f}%).")
    print(f"Total pipeline execution time: {t3 - t0:.2f}s")
    print("============================================================\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless Single-Photon Clustering & Reconstruction Tool (ALS BL 6.0.2 QERLIN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Processing subcommand")

    # --- Subcommand: dark-mask ---
    p_dark = subparsers.add_parser("dark-mask", help="Stage 1: Generate median dark and noise masks")
    p_dark.add_argument("--dark-dir", "-d", required=True, help="Folder containing raw dark TIFF frames")
    p_dark.add_argument("--output-dir", "-o", default=None, help="Directory to save MED_Dark.tif and Final_Mask.tif")
    p_dark.add_argument("--label", default="Dark", help="Output filename label tag")
    p_dark.add_argument("--stddev-thresh", type=float, default=40.0, help="StdDev threshold (ADU)")
    p_dark.add_argument("--absdev-thresh", type=float, default=60.0, help="AbsDev excursion threshold (ADU)")
    p_dark.add_argument("--tail-thresh-ratio", "--tail-ratio", dest="tail_thresh_ratio", type=float, default=0.9333, help="Fraction of required stable frames")
    p_dark.add_argument("--max-frames", type=int, default=0, help="Max dark frames to process (0 = all)")

    # --- Subcommand: cluster ---
    p_clust = subparsers.add_parser("cluster", help="Stage 2: Run 8-connected cluster analysis on signal frames")
    p_clust.add_argument("--signal-dir", "-s", required=True, help="Folder containing raw signal TIFF frames")
    p_clust.add_argument("--dark-tif", "--med-dark", dest="dark_tif", default=None, help="Path to precomputed MED_Dark.tif")
    p_clust.add_argument("--mask-tif", "--final-mask", dest="mask_tif", default=None, help="Path to precomputed Final_Mask.tif")
    p_clust.add_argument("--output-dir", "-o", default=None, help="Directory to save Results_clusters.xls")
    p_clust.add_argument("--output-name", default="Results_clusters.xls", help="Cluster spreadsheet filename")
    p_clust.add_argument("--sig-thresh-low", type=float, default=45.0, help="Signal noise floor cutoff (ADU)")
    p_clust.add_argument("--sig-thresh-high", type=float, default=1e6, help="Signal ceiling threshold (ADU)")
    p_clust.add_argument("--chunk-size", type=int, default=0, help="Chunk frame size")

    # --- Subcommand: reconstruct ---
    p_recon = subparsers.add_parser("reconstruct", help="Stage 3: Reconstruct 2D event map from cluster spreadsheet")
    p_recon.add_argument("--clusters-xls", "-c", required=True, help="Path to Results_clusters.xls")
    p_recon.add_argument("--signal-dir", "-s", default=None, help="Optional signal directory for detector shape")
    p_recon.add_argument("--output-dir", "-o", default=None, help="Directory to save Photon_Event_Map.tif")
    p_recon.add_argument("--output-name", default="Photon_Event_Map.tif", help="Output 2D TIFF map filename")
    p_recon.add_argument("--intden-low", type=float, default=120.0, help="Single-photon IntDen lower cutoff (ADU)")
    p_recon.add_argument("--intden-high", type=float, default=320.0, help="Single-photon IntDen upper cutoff (ADU)")
    p_recon.add_argument("--max-area", type=int, default=9, help="Max cluster area (pixels)")
    p_recon.add_argument("--min-circ", type=float, default=0.3, help="Min cluster circularity")
    p_recon.add_argument("--subpixel", type=int, default=1, help="Sub-pixel super-resolution binning multiplier")
    p_recon.add_argument("--image-width", type=int, default=2048, help="Original detector width")
    p_recon.add_argument("--image-height", type=int, default=2048, help="Original detector height")
    p_recon.add_argument("--chunk-size", type=int, default=0, help="Chunk frame size")

    # --- Subcommand: full ---
    p_full = subparsers.add_parser("full", help="Execute Stages 1, 2, and 3 end-to-end")
    p_full.add_argument("--dark-dir", "-d", required=True, help="Folder containing raw dark TIFF frames")
    p_full.add_argument("--signal-dir", "-s", required=True, help="Folder containing raw signal TIFF frames")
    p_full.add_argument("--output-dir", "-o", default=None, help="Output directory for all artifacts (defaults to <signal_dir>/clusters/)")
    p_full.add_argument("--label", default="Dark", help="Dark output label tag")
    p_full.add_argument("--output-name", default="Photon_Event_Map.tif", help="Final event map filename")
    p_full.add_argument("--stddev-thresh", type=float, default=40.0, help="StdDev threshold (ADU)")
    p_full.add_argument("--absdev-thresh", type=float, default=60.0, help="AbsDev excursion threshold (ADU)")
    p_full.add_argument("--tail-thresh-ratio", "--tail-ratio", dest="tail_thresh_ratio", type=float, default=0.9333, help="Fraction of required stable frames")
    p_full.add_argument("--max-frames", type=int, default=0, help="Max dark frames to process (0 = all)")
    p_full.add_argument("--sig-thresh-low", type=float, default=45.0, help="Signal noise floor cutoff (ADU)")
    p_full.add_argument("--sig-thresh-high", type=float, default=1e6, help="Signal ceiling threshold (ADU)")
    p_full.add_argument("--intden-low", type=float, default=120.0, help="Single-photon IntDen lower cutoff (ADU)")
    p_full.add_argument("--intden-high", type=float, default=320.0, help="Single-photon IntDen upper cutoff (ADU)")
    p_full.add_argument("--max-area", type=int, default=9, help="Max cluster area (pixels)")
    p_full.add_argument("--min-circ", type=float, default=0.3, help="Min cluster circularity")
    p_full.add_argument("--subpixel", type=int, default=1, help="Sub-pixel super-resolution binning multiplier")
    p_full.add_argument("--chunk-size", type=int, default=0, help="Chunk frame size for per-chunk event maps")

    args = parser.parse_args()

    if args.subcommand == "dark-mask":
        sys.exit(run_dark_mask(args))
    elif args.subcommand == "cluster":
        sys.exit(run_cluster(args))
    elif args.subcommand == "reconstruct":
        sys.exit(run_reconstruct(args))
    elif args.subcommand == "full":
        sys.exit(run_full(args))


if __name__ == "__main__":
    main()
