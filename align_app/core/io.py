"""Single-frame TIFF loading utilities.

This module provides low-level image I/O for the align_app pipeline. The
``load_raw`` function is the sole entry point for reading raw float32 frames
from disk.  It performs TIFF decoding, converts to ``float32``, and sanitises
NaN / inf values.  Persistent caching is handled at a higher level by
:class:`~align_app.dataset.ZarrSequenceManager`.
"""

import os
import numpy as np
import tifffile


def load_raw(image_path: str) -> np.ndarray:
    """Load a float32 2D array from a TIFF file.

    Reads the TIFF via ``tifffile``, converts to ``float32``, and cleans
    ``NaN`` / ``inf`` values before returning.  This function intentionally
    performs **no disk caching** — persistent caching is the responsibility of
    :class:`~align_app.dataset.ZarrSequenceManager`.

    Args:
        image_path: Absolute or relative path to the source ``.tif`` /
            ``.tiff`` file.

    Returns:
        2D ``float32`` numpy array of shape ``(H, W)`` containing the raw
        pixel intensities.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist on disk.
        ValueError: If the TIFF cannot be squeezed to a 2-D array after
            loading (e.g., a multi-channel or volumetric file).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    raw = tifffile.imread(image_path).astype(np.float32)
    if raw.ndim != 2:
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            raise ValueError("TIFF image must be 2D")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw
