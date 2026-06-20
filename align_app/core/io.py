import os
import numpy as np
import tifffile

def load_raw(image_path: str) -> np.ndarray:
    """
    Load a float32 2D array from a TIFF file, using a persistent disk cache for fast loads.

    Performance Caching Architecture:
    1. Disk Caching: On first access, reads the TIFF image, converts it to float32, and cleans 
       NaN/Inf values. The clean array is saved as a binary `.npy` file under `tif-cache/`.
    2. Zero-Copy Loading: On subsequent accesses, loads the cached file using `np.load` with 
       `mmap_mode='r'`. This creates a memory-mapped array, utilizing virtual memory to map the file 
       directly into RAM without double-buffering, significantly reducing loading time.
    3. Cache Invalidation: Compares modification times (`os.path.getmtime`). If the source `.tif` 
       file is newer than the cached `.npy` file, the cache is automatically regenerated.

    Args:
        image_path: Path to the source TIFF file.

    Returns:
        np.ndarray: 2D float32 array containing raw pixel intensities.

    Raises:
        FileNotFoundError: If the source image path does not exist.
        ValueError: If the TIFF cannot be squeezed to a 2D array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    tif_dir = os.path.dirname(os.path.abspath(image_path))
    cache_dir = os.path.join(tif_dir, "tif-cache")
    basename = os.path.splitext(os.path.basename(image_path))[0]
    npy_path = os.path.join(cache_dir, basename + ".npy")

    # Check for a valid cache hit
    if os.path.exists(npy_path):
        tif_mtime = os.path.getmtime(image_path)
        npy_mtime = os.path.getmtime(npy_path)
        if npy_mtime >= tif_mtime:
            return np.load(npy_path, mmap_mode="r")

    # Cache miss or stale: read TIFF and convert
    raw = tifffile.imread(image_path).astype(np.float32)
    if raw.ndim != 2:
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError("TIFF image must be 2D")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Save to cache (create directory if needed)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy_path, raw)

    return raw
