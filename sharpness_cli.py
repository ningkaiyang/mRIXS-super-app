#!/usr/bin/env python3
"""
Command-line interface for running sharpness evaluation metrics
and calculating Spearman rank correlation with ground truth ranks.
"""

import os
import sys
import argparse
import json
import re
import numpy as np
import tifffile
from scipy.stats import spearmanr
from rixs_app.core.sharpness import denoise_image, evaluate_sharpness

def extract_frame_index(filename: str) -> int:
    """Extract the frame index from a filename.

    This function attempts to find and return the numeric index from a filename
    using two main regex matching phases, followed by a fallback mechanism.

    Matching Phases:
    1. CMOS Detector Prefix: Looks for the case-insensitive pattern 
       'CMOS[\\s_-]+?Detector[\\s_-]+?(-?\\d+)' to extract the index.
    2. Frame Prefix: Looks for the case-insensitive pattern 'frame[\\s_-]+?(-?\\d+)' 
       to extract the index.

    Fallback Behavior:
    If neither prefix is matched, the function removes the file extension and
    searches the remaining filename for any contiguous blocks of digits (possibly
    with a leading negative sign) using the pattern '-?\\d+'. If found, it returns
    the last digit block as the index. If no digit blocks are found, it raises a
    ValueError.

    Args:
        filename (str): The name of the file to parse.

    Returns:
        int: The parsed frame index.

    Raises:
        ValueError: If no digit index can be found using the regex patterns or 
            the fallback logic.
    """
    match = re.search(r'CMOS[\s_-]+?Detector[\s_-]+?(-?\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'frame[\s_-]+?(-?\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 2. If no prefix matches, split extension off, find all blocks of digits
    name_without_ext, _ = os.path.splitext(filename)
    digits = re.findall(r'-?\d+', name_without_ext)
    if digits:
        return int(digits[-1])
    raise ValueError(f"No digit index found in filename: {filename}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image sharpness and calculate Spearman rank correlation."
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        help="Path to the directory containing TIFF images and ground_truth.json (or subdirectories)."
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated list of metrics to run (e.g., dog_laplacian,directional_tenengrad)."
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Calculate and display Spearman's rank correlation."
    )
    parser.add_argument(
        "--print-scores",
        action="store_true",
        help="Print individual frame sharpness scores."
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Denoise each frame using denoise_image first, then evaluate sharpness."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate metrics
    VALID_METRICS = {"dog_laplacian", "directional_tenengrad", "fft_bandpass"}
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
        for m in metrics:
            if m not in VALID_METRICS:
                sys.stderr.write(f"Error: Invalid metric: '{m}'. Valid metrics are: {list(VALID_METRICS)}\n")
                sys.exit(1)
    else:
        metrics = ["dog_laplacian", "directional_tenengrad", "fft_bandpass"]
        
    base_dir = args.dir
    if not os.path.exists(base_dir):
        sys.stderr.write(f"Error: Directory does not exist: {base_dir}\n")
        sys.exit(1)
        
    # Discover scan directories
    scan_dirs = []
    
    def has_tiff_files(path):
        if not os.path.isdir(path):
            return False
        for f in os.listdir(path):
            if f.lower().endswith(('.tif', '.tiff')):
                return True
        return False

    # Check base_dir first
    denoised_sub = os.path.join(base_dir, "denoised")
    if has_tiff_files(denoised_sub):
        scan_dirs.append((base_dir, denoised_sub))
    elif has_tiff_files(base_dir):
        scan_dirs.append((base_dir, base_dir))
    else:
        # Check immediate subdirectories
        try:
            entries = os.listdir(base_dir)
        except Exception:
            entries = []
        for entry in entries:
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path) and entry != "denoised":
                sub_denoised = os.path.join(entry_path, "denoised")
                if has_tiff_files(sub_denoised):
                    scan_dirs.append((entry_path, sub_denoised))
                elif has_tiff_files(entry_path):
                    scan_dirs.append((entry_path, entry_path))
                    
    # If still none found, do recursive check
    if not scan_dirs:
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) == "denoised":
                continue
            sub_denoised = os.path.join(root, "denoised")
            if has_tiff_files(sub_denoised):
                scan_dirs.append((root, sub_denoised))
            elif has_tiff_files(root):
                scan_dirs.append((root, root))
                
    # Unique list while preserving order
    scan_dirs_set = []
    seen = set()
    for gt, img in scan_dirs:
        if (gt, img) not in seen:
            seen.add((gt, img))
            scan_dirs_set.append((gt, img))
    scan_dirs = scan_dirs_set
    
    if not scan_dirs:
        sys.stderr.write(f"Error: No TIFF images found in {base_dir} or its subdirectories.\n")
        sys.exit(1)
        
    table_rows = []
    
    for gt_dir, image_dir in scan_dirs:
        # Load TIFF files
        tifs = []
        try:
            files = os.listdir(image_dir)
        except Exception:
            files = []
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                tifs.append(os.path.join(image_dir, filename))
        tifs = sorted(tifs, key=lambda x: os.path.basename(x))
        
        # Extract frame indices
        frame_data = []
        for path in tifs:
            filename = os.path.basename(path)
            try:
                idx = extract_frame_index(filename)
                frame_data.append((idx, path))
            except ValueError:
                pass
                    
        # Load ground_truth.json
        gt_path = None
        for p in [os.path.join(gt_dir, "ground_truth.json"), os.path.join(image_dir, "ground_truth.json")]:
            if os.path.exists(p):
                gt_path = p
                break
                
        ground_truth = None
        if gt_path:
            try:
                with open(gt_path, 'r') as f:
                    ground_truth = json.load(f)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load ground truth from {gt_path}: {e}\n")
                
        # Ground truth missing logic
        if ground_truth is None:
            if args.correlation or not args.print_scores:
                sys.stderr.write(f"Error: ground_truth.json is missing in {gt_dir}.\n")
                sys.exit(1)
            else:
                sys.stderr.write(f"Warning: ground_truth.json is missing in {gt_dir}. Skipping correlation calculation.\n")
                
        fractional_ranks = {}
        if ground_truth:
            fractional_ranks = ground_truth.get("fractional_ranks", {})
            
        for metric in metrics:
            scores = []
            ranks = []
            for idx, path in frame_data:
                img = tifffile.imread(path)
                if args.denoise:
                    img = denoise_image(img)
                score = evaluate_sharpness(img, metric)
                
                if args.print_scores:
                    print(f"Frame {idx} ({metric}): {score} (rounded: {score:.2f})")
                    
                rank = None
                for key in [str(idx), idx]:
                    if key in fractional_ranks:
                        rank = fractional_ranks[key]
                        break
                        
                if rank is not None:
                    scores.append(score)
                    ranks.append(rank)
                    
            # Compute correlation
            correlation = float('nan')
            if ground_truth:
                if len(scores) < 2:
                    sys.stderr.write(f"Warning: Insufficient frames ({len(scores)}) to calculate correlation for {metric} in {gt_dir}.\n")
                else:
                    corr, _ = spearmanr(scores, -np.array(ranks))
                    correlation = corr
                    
            table_rows.append((gt_dir, metric, correlation))
            
    # Print markdown table
    if not args.print_scores or args.correlation:
        print("| Directory | Metric | Spearman Correlation |")
        print("|---|---|---|")
        for directory, metric, corr in table_rows:
            if np.isnan(corr):
                corr_str = "NaN"
            else:
                corr_str = f"{corr:.4f}"
            print(f"| {directory} | {metric} | {corr_str} |")

if __name__ == "__main__":
    main()
