#!/usr/bin/env python3
"""CLI tool for spectroscopic image preprocessing and denoising.

This tool provides a frontend for the RIXS denoising pipeline, supporting both
single file and batch directory processing.
"""

import os
import sys
import argparse
import numpy as np
import scipy.ndimage
import cv2
import tifffile

from rixs_app.core.io import load_raw
from rixs_app.core.sharpness import denoise_image

def get_spike_count(img: np.ndarray, mad_threshold: float = 5.0) -> int:
    """Calculate the number of spike pixels in the image using MAD despiking threshold logic.

    Args:
        img: 2D numpy array representing the image.
        mad_threshold: MAD threshold multiplier.

    Returns:
        int: Number of detected spike pixels.
    """
    if img.ndim != 2 or img.size == 0:
        return 0
    median_img = scipy.ndimage.median_filter(img, size=3)
    dev = img - median_img
    mad = np.median(np.abs(dev))
    if mad < 1e-6:
        mad = np.std(dev)
    if mad > 1e-6:
        threshold = mad_threshold * 1.4826 * mad
        return int(np.sum(np.abs(dev) > threshold))
    return 0

def print_stats(img: np.ndarray, label: str, mad_threshold: float = 5.0) -> None:
    """Print image statistics (min, max, mean, std, negative count, spike count) to stdout.

    Args:
        img: 2D numpy array representing the image.
        label: Label string (e.g. 'Before' or 'After').
        mad_threshold: MAD threshold multiplier for spike count calculation.
    """
    val_min = float(np.min(img)) if img.size > 0 else 0.0
    val_max = float(np.max(img)) if img.size > 0 else 0.0
    val_mean = float(np.mean(img)) if img.size > 0 else 0.0
    val_std = float(np.std(img)) if img.size > 0 else 0.0
    neg_count = int(np.sum(img < 0.0))
    spike_count = get_spike_count(img, mad_threshold)

    print(f"Stats {label}:")
    print(f"  Min: {val_min:.4f}")
    print(f"  Max: {val_max:.4f}")
    print(f"  Mean: {val_mean:.4f}")
    print(f"  Std: {val_std:.4f}")
    print(f"  Negative count: {neg_count}")
    print(f"  Spike count: {spike_count}")

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Denoise 2D spectroscopic frame TIFF images."
    )
    # Target inputs/outputs
    parser.add_argument("-d", "--dir", type=str, help="Target directory containing TIFF images.")
    parser.add_argument("--input", type=str, help="Single input TIFF file.")
    parser.add_argument("--output", type=str, help="Single output TIFF file.")

    # Action flags
    parser.add_argument("--clip", action="store_true", help="Flag to run clipping.")
    parser.add_argument("--despike", action="store_true", help="Flag to run MAD despiking.")
    parser.add_argument(
        "--mad-threshold", type=float, default=5.0, help="Threshold multiplier (default 5.0)."
    )
    parser.add_argument("--anscombe", action="store_true", help="Flag to apply Anscombe VST.")
    parser.add_argument("--bilateral", action="store_true", help="Flag to apply bilateral filter.")
    parser.add_argument("--d", type=int, default=5, help="Bilateral diameter (default 5).")
    parser.add_argument(
        "--sigma-color", type=float, default=1.5, help="Bilateral sigmaColor (default 1.5)."
    )
    parser.add_argument(
        "--sigma-space", type=float, default=3.0, help="Bilateral sigmaSpace (default 3.0)."
    )
    parser.add_argument(
        "--inverse-anscombe", action="store_true", help="Flag to apply inverse Anscombe."
    )

    # Geometric Gradient parameters
    parser.add_argument(
        "--feature-low", type=float, default=0.0, help="Feature low threshold (default 0.0)."
    )
    parser.add_argument(
        "--feature-high", type=float, default=100.0, help="Feature high threshold (default 100.0)."
    )
    parser.add_argument(
        "--edge-margin", type=int, default=50, help="Edge margin pixels to mask (default 50)."
    )
    parser.add_argument(
        "--high-dilate", type=int, default=8, help="Dilation radius for high bad/edges (default 8)."
    )
    parser.add_argument(
        "--bg-sigma", type=float, default=None, help="Background blur sigma for high-pass (default None)."
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=1.2, help="Smoothing blur sigma (default 1.2)."
    )

    args = parser.parse_args()

    # --- Argument Validation ---

    # 1. Mutually exclusive input modes
    if args.dir is not None:
        if args.input is not None or args.output is not None:
            sys.stderr.write("Error: When --dir is specified, --input and --output must NOT be specified.\n")
            sys.exit(1)
    elif args.input is not None:
        if args.output is None:
            sys.stderr.write("Error: When --input is specified, --output MUST be specified.\n")
            sys.exit(1)
    else:
        # Neither --dir nor --input specified
        sys.stderr.write("Error: Either --dir or --input must be specified.\n")
        sys.exit(1)

    # 2. Non-negative numeric options
    if (args.d < 0 or args.sigma_color < 0.0 or args.sigma_space < 0.0 or args.mad_threshold < 0.0 or
            args.edge_margin < 0 or args.high_dilate < 0 or
            (args.bg_sigma is not None and args.bg_sigma < 0.0) or args.smooth_sigma < 0.0):
        sys.stderr.write("Error: All numeric options must be non-negative.\n")
        sys.exit(1)

    # 3. Path existence check
    if args.dir is not None:
        if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
            sys.stderr.write(f"Error: Target directory '{args.dir}' does not exist or is not a directory.\n")
            sys.exit(1)
    elif args.input is not None:
        if not os.path.exists(args.input):
            sys.stderr.write(f"Error: Input file '{args.input}' does not exist.\n")
            sys.exit(1)

    # --- Execution Logic setup ---

    action_flags = [args.clip, args.despike, args.anscombe, args.bilateral, args.inverse_anscombe]
    any_action = any(action_flags)

    if not any_action:
        run_clip = True
        run_despike = True
        run_anscombe = True
        run_bilateral = True
        run_inverse_anscombe = True
    else:
        run_clip = args.clip
        run_despike = args.despike
        run_anscombe = args.anscombe
        run_bilateral = args.bilateral
        run_inverse_anscombe = args.inverse_anscombe

    # --- Execute ---

    if args.dir is not None:
        # Directory Mode
        # Find all TIFF files in args.dir (ignoring files in "denoised" directory), sorted alphabetically
        files = sorted(os.listdir(args.dir))
        tiff_files = []
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                full_path = os.path.join(args.dir, f)
                if os.path.isfile(full_path):
                    tiff_files.append(full_path)

        if not tiff_files:
            print("No TIFF files found to process.")
            sys.exit(0)

        # Output subdirectory
        out_dir = os.path.join(args.dir, "denoised")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, ".denoise_version"), "w") as f:
            f.write("2.0-gradient-magnitude\n")

        for file_path in tiff_files:
            try:
                img = load_raw(file_path)
            except Exception as e:
                sys.stderr.write(f"Error loading image '{file_path}': {e}\n")
                sys.exit(1)

            if img.ndim != 2 or img.shape == (0, 0) or img.size == 0:
                sys.stderr.write(f"Error: Loaded image '{file_path}' is empty or zero-dimensional.\n")
                sys.exit(1)

            print(f"Preprocessing {file_path}...")
            print_stats(img, "before", args.mad_threshold)

            denoised_img = denoise_image(
                img,
                clip=run_clip,
                despike=run_despike,
                anscombe=run_anscombe,
                bilateral=run_bilateral,
                inverse_anscombe=run_inverse_anscombe,
                mad_threshold=args.mad_threshold,
                d=args.d,
                sigma_color=args.sigma_color,
                sigma_space=args.sigma_space
            )

            print_stats(denoised_img, "after", args.mad_threshold)

            # Save in subfolder named denoised/ with _denoised.tiff appended
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(out_dir, f"{base_name}_denoised.tiff")
            tifffile.imwrite(out_file, denoised_img.astype(np.float32))
            print(f"Saved denoised image to: {out_file}\n")

    else:
        # Single File Mode
        try:
            img = load_raw(args.input)
        except Exception as e:
            sys.stderr.write(f"Error loading input image '{args.input}': {e}\n")
            sys.exit(1)

        if img.ndim != 2 or img.shape == (0, 0) or img.size == 0:
            sys.stderr.write(f"Error: Loaded image '{args.input}' is empty or zero-dimensional.\n")
            sys.exit(1)

        print(f"Preprocessing {args.input}...")
        print_stats(img, "before", args.mad_threshold)

        denoised_img = denoise_image(
            img,
            clip=run_clip,
            despike=run_despike,
            anscombe=run_anscombe,
            bilateral=run_bilateral,
            inverse_anscombe=run_inverse_anscombe,
            mad_threshold=args.mad_threshold,
            d=args.d,
            sigma_color=args.sigma_color,
            sigma_space=args.sigma_space
        )

        print_stats(denoised_img, "after", args.mad_threshold)

        # Save single output
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, ".denoise_version"), "w") as f:
                f.write("2.0-gradient-magnitude\n")

        tifffile.imwrite(args.output, denoised_img.astype(np.float32))
        print(f"Saved denoised image to: {args.output}\n")

if __name__ == "__main__":
    main()
