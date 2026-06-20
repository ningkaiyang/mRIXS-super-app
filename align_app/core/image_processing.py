import numpy as np
import cv2
import matplotlib
from align_app.core.io import load_raw

def warp_image(image_data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate a 2D or 3D image by sub-pixel offsets (dx, dy) using affine warping.
    
    Mathematics & Alignment Context:
      - Sub-pixel shift is necessary to align RIXS beam profiles precisely without rounding errors.
      - Constructs a 2x3 affine translation matrix: M = [[1, 0, dx], [0, 1, dy]].
      - Applies OpenCV's `warpAffine` with bilinear interpolation (`cv2.INTER_LINEAR`).
      - Bilinear interpolation reconstructs intensities at fractional pixel locations, preventing 
        high-frequency sampling artifacts.
      - Boundary values are padded with 0 (constant padding) to signify empty detector regions.

    Args:
        image_data: 2D (grayscale/raw) or 3D (RGB) numpy array.
        dx: Sub-pixel translation along the horizontal axis.
        dy: Sub-pixel translation along the vertical axis.

    Returns:
        np.ndarray: Translated image copy of the same shape and data type.
    """
    if image_data.ndim not in (2, 3):
        raise ValueError("image_data must be 2D or 3D")
        
    if image_data.shape[0] == 0 or image_data.shape[1] == 0:
        return image_data.copy()
        
    if dx == 0.0 and dy == 0.0:
        return image_data.copy()
        
    h, w = image_data.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    warped = cv2.warpAffine(
        image_data,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def preprocess_image(image_path: str, cmap_name: str, percentile_threshold: float, floor: float = None, ceiling: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a raw TIFF file, normalize intensities, apply a colormap, and generate an RGB image.

    Mathematics Context:
    1. Value Clamping: Clamps raw intensities between the `floor` and `ceiling` limits. If not specified, 
       clamps between `min(raw)` and the intensity at the given `percentile_threshold` (e.g., 99.9%). 
       Percentile-based clipping protects contrast against cosmic rays (which introduce hot pixels far 
       above the true line intensity).
    2. Linear Rescaling: Rescales the clamped range linearly to [0.0, 1.0]:
       scaled = (raw - vmin) / (vmax - vmin)
    3. Colormap Mapping: Maps the normalized array into a 3-channel RGB uint8 image using a Matplotlib 
       colormap (or falls back to a 3-channel grayscale mapping).

    Args:
        image_path: Path to the TIFF file on disk.
        cmap_name: Name of the colormap to apply (e.g. 'viridis', 'inferno', 'grayscale').
        percentile_threshold: Contrast ceiling percentile in [0, 100].
        floor: Optional explicit intensity floor for contrast scaling.
        ceiling: Optional explicit intensity ceiling for contrast scaling.

    Returns:
        tuple[np.ndarray, np.ndarray]: (RGB uint8 image, raw float32 intensities).
    """
    if not (0.0 <= percentile_threshold <= 100.0):
        raise ValueError("percentile_threshold must be between 0.0 and 100.0")

    raw = load_raw(image_path)  # Handles FileNotFoundError, 2D check, and NaN cleanup

    if floor is not None and ceiling is not None:
        vmin = float(floor)
        vmax = float(ceiling)
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin = np.min(raw)
        if percentile_threshold < 100.0:
            vmax = np.percentile(raw, percentile_threshold)
        else:
            vmax = np.max(raw)
        if vmax <= vmin:
            vmax = np.max(raw)
        
    norm_range = vmax - vmin
    if norm_range > 1e-8:
        scaled = np.clip((raw - vmin) / norm_range, 0.0, 1.0)
    else:
        scaled = np.zeros_like(raw)
        
    is_grayscale = False
    if isinstance(cmap_name, str):
        try:
            cmap_name_lower = cmap_name.lower()
            if cmap_name_lower in ["grayscale", "gray"]:
                is_grayscale = True
        except Exception:
            is_grayscale = True
    else:
        is_grayscale = True
        
    if is_grayscale:
        scaled_8bit = (scaled * 255.0).astype(np.uint8)
        rgb_image = np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)
    else:
        try:
            cmap_name_lower = cmap_name.lower()
            if hasattr(matplotlib, 'colormaps'):
                cmap = matplotlib.colormaps[cmap_name_lower]
            else:
                import matplotlib.cm as cm
                cmap = cm.get_cmap(cmap_name_lower)
            rgba_float = cmap(scaled)
            rgb_image = (rgba_float[:, :, :3] * 255.0).astype(np.uint8)
        except Exception:
            # Fall back to standard grayscale 3-channel
            scaled_8bit = (scaled * 255.0).astype(np.uint8)
            rgb_image = np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)
            
    return rgb_image, raw

def apply_colormap(raw: np.ndarray, cmap_name: str,
                   display_floor: float = None,
                   display_ceiling: float = None) -> np.ndarray:
    """
    Apply a colormap to a raw intensity matrix in memory, yielding a 3-channel RGB image.

    Interactive GUI Performance:
    Operates strictly on the raw numpy array already cached in memory. By avoiding disk IO, 
    this routine achieves extremely low latency (~5-50ms), allowing real-time visualization 
    updates as the user drags GUI clamping sliders. Clamps values to the specified floor/ceiling 
    range before mapping to RGB.

    Args:
        raw: 2D float32 array of raw intensity values.
        cmap_name: Colormap name ('grayscale', 'viridis', 'inferno', etc.).
        display_floor: Minimum display intensity. Defaults to raw.min().
        display_ceiling: Maximum display intensity. Defaults to raw.max().

    Returns:
        np.ndarray: RGB uint8 numpy array of shape (H, W, 3).
    """
    if raw.ndim != 2:
        raise ValueError("raw must be a 2D numpy array")

    vmin = float(display_floor) if display_floor is not None else float(np.min(raw))
    vmax = float(display_ceiling) if display_ceiling is not None else float(np.max(raw))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm_range = vmax - vmin
    if norm_range > 1e-8:
        scaled = np.clip((raw - vmin) / norm_range, 0.0, 1.0)
    else:
        scaled = np.zeros_like(raw)

    is_grayscale = False
    if isinstance(cmap_name, str):
        try:
            if cmap_name.lower() in ("grayscale", "gray"):
                is_grayscale = True
        except Exception:
            is_grayscale = True
    else:
        is_grayscale = True

    if is_grayscale:
        scaled_8bit = (scaled * 255.0).astype(np.uint8)
        return np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)

    try:
        cmap_lower = cmap_name.lower()
        if hasattr(matplotlib, 'colormaps'):
            cmap = matplotlib.colormaps[cmap_lower]
        else:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(cmap_lower)
        rgba_float = cmap(scaled)
        return (rgba_float[:, :, :3] * 255.0).astype(np.uint8)
    except Exception:
        scaled_8bit = (scaled * 255.0).astype(np.uint8)
        return np.stack([scaled_8bit, scaled_8bit, scaled_8bit], axis=-1)

def generate_aligned_sum(file_list: list[str], get_raw_fn, offsets: dict,
                         ref_shape: tuple,
                         progress_callback=None) -> np.ndarray:
    """
    Warp and accumulate all spectroscopic frames into a single high-signal-to-noise ratio (SNR) aligned sum.
    
    Physics & Alignment Context:
      - RIXS measurements involve multiple short-exposure frames to prevent detector saturation.
      - Individual frames have low SNR. Summing them directly without alignment blurs spectral features 
        due to beamline drift.
      - For each frame, retrieves the alignment offset (dx, dy) and applies the negative (-dx, -dy) 
        using bilinear interpolation to warp it back to the reference frame's coordinate system.
      - Accumulates warped frames into a float32 array to preserve full bit-depth and sub-pixel alignment quality.

    Args:
        file_list: List of paths to TIFF files.
        get_raw_fn: Function mapping filepath -> 2D numpy array (raw float32 intensity).
        offsets: Dictionary mapping frame index -> (dx, dy) alignment offsets.
        ref_shape: (H, W) shape of the reference frame.
        progress_callback: Optional function called with (current_index, total_frames).

    Returns:
        np.ndarray: 2D float32 array of the summed, aligned spectral image.

    Raises:
        ValueError: If any frame file fails to load.
    """
    accum = np.zeros(ref_shape, dtype=np.float32)
    n_frames = len(file_list)

    for idx, filepath in enumerate(file_list):
        raw = get_raw_fn(filepath)
        if raw is None:
            raise ValueError(f"Could not load frame {idx}: {filepath}")

        if idx == 0:
            accum += raw
        else:
            dx, dy = offsets.get(idx, (0.0, 0.0))
            warped = warp_image(raw, -dx, -dy)
            accum += warped

        if progress_callback is not None:
            progress_callback(idx + 1, n_frames)

    return accum
