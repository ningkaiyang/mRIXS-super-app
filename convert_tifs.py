#!/usr/bin/env python3
"""
TIFF to JPEG Converter for Visual Inspection
This script processes float32 TIFF files (typically from photon-counting or high-dynamic-range imaging),
applies custom scaling/colormapping to make sparse details visible, downsamples them for fast previewing,
and saves them as JPEG files.

Dependencies:
- numpy
- pillow
- tifffile
- matplotlib
"""

import os
import glob
import argparse
import numpy as np
import tifffile
import matplotlib
from matplotlib import colormaps
from PIL import Image

def process_image(tif_path, output_path, cmap_name='inferno', target_width=960, clip_percentile=99.9, vmax=None):
    """
    Reads a single float32 TIFF image, normalizes, resizes, and saves it.
    """
    try:
        # Load image data
        data = tifffile.imread(tif_path)
        
        # Calculate dynamic range / stats
        dmin = float(data.min())
        dmax = float(data.max())
        
        # Determine vmax for normalization
        if vmax is None:
            if clip_percentile >= 100.0:
                vmax = dmax
            else:
                vmax = float(np.percentile(data, clip_percentile))
                # Ensure vmax is not <= dmin to avoid division by zero
                if vmax <= dmin:
                    vmax = dmax
        
        # Clip data to [dmin, vmax] and scale to [0, 1]
        norm_range = vmax - dmin
        if norm_range > 1e-8:
            scaled_data = np.clip((data - dmin) / norm_range, 0.0, 1.0)
        else:
            scaled_data = np.zeros_like(data)
            
        # Apply colormapping
        if cmap_name.lower() == 'gray' or cmap_name.lower() == 'grayscale':
            # Direct 8-bit grayscale mapping
            img_array = (scaled_data * 255.0).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        else:
            # Map values through a Matplotlib colormap
            try:
                cmap = colormaps[cmap_name]
            except KeyError:
                print(f"Colormap '{cmap_name}' not found. Defaulting to 'inferno'.")
                cmap = colormaps['inferno']
                
            colored = cmap(scaled_data) # Returns H x W x 4 float array
            rgb_array = (colored[:, :, :3] * 255.0).astype(np.uint8)
            img = Image.fromarray(rgb_array, mode='RGB')
            
        # Resize maintaining aspect ratio
        orig_w, orig_h = img.size
        if target_width and orig_w > target_width:
            target_height = int(orig_h * (target_width / orig_w))
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            img_resized = img
            
        # Save as JPEG
        img_resized.save(output_path, "JPEG", quality=85)
        print(f"  Converted: {os.path.basename(tif_path)}")
        print(f"    Raw Range: [{dmin:.2f}, {dmax:.2f}] -> Normalized Range: [{dmin:.2f}, {vmax:.2f}]")
        print(f"    Output: {output_path} ({img_resized.width}x{img_resized.height})")
        return True
    except Exception as e:
        print(f"  Error processing {tif_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert float32 TIFF files to lightweight JPEGs.")
    parser.add_argument("--input_dir", default=".", help="Directory containing .tif files.")
    parser.add_argument("--output_dir", default="./jpgs", help="Directory where JPEGs will be saved.")
    parser.add_argument("--cmap", default="inferno", choices=["inferno", "viridis", "plasma", "magma", "gray"],
                        help="Colormap to apply (default: inferno).")
    parser.add_argument("--width", type=int, default=960, help="Target width of output JPEGs (default: 960).")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="Percentile to clip intensity values at (default: 99.9). Set to 100 for min-max scaling.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Override percentile clipping with a specific maximum value.")
    
    args = parser.parse_args()
    
    # Resolve directories
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .tif and .tiff files
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff")))
    
    if not tif_files:
        print(f"No .tif or .tiff files found in {input_dir}")
        return
        
    print(f"Found {len(tif_files)} files to convert.")
    print(f"Output directory: {output_dir}")
    print(f"Applying colormap: {args.cmap} (clipped at {args.vmax if args.vmax else f'{args.percentile}%'})")
    
    success_count = 0
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_{args.cmap}.jpg" if args.cmap != "inferno" else f"{base_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        if process_image(
            tif_path=tif_path,
            output_path=output_path,
            cmap_name=args.cmap,
            target_width=args.width,
            clip_percentile=args.percentile,
            vmax=args.vmax
        ):
            success_count += 1
            
    print(f"\nProcessing complete: {success_count}/{len(tif_files)} files successfully converted.")

if __name__ == "__main__":
    main()
