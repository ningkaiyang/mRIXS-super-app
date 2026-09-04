"""In-memory sequence manager for fast frame and derived array access.

:class:`SequenceManager` handles loading and caching TIFF frames in RAM
using :class:`~rixs_app.core.frame_cache.CompressedFrameCache` (np.float16 + Zstd).
Derived filter products (e.g., denoised or row-smoothed frames in Zeroth-Order)
are also cached strictly in memory during an active session.

Zero on-disk cache files, zero disk clutter, zero lock contention.
"""

from __future__ import annotations

import threading
import numpy as np

from rixs_app.core.frame_cache import CompressedFrameCache
from rixs_app.core.io import load_raw


class SequenceManager:
    """Manages an in-memory sequence of TIFF frames and derived arrays.

    Frames are loaded on-demand via :func:`~rixs_app.core.io.load_raw` and cached
    in an in-memory :class:`~rixs_app.core.frame_cache.CompressedFrameCache` to ensure
    sub-millisecond scrubbing with minimal RAM overhead.

    Args:
        file_list: Ordered list of absolute paths to the TIFF source files.
            The order determines the frame index mapping used by :meth:`get_frame`.
        capacity: Maximum number of raw frames to retain in the LRU cache (default 128).

    Attributes:
        file_list (list[str]): The ordered source file paths.
        n_frames (int): Total number of frames in the sequence.
        median_frame (np.ndarray | None): 2-D float32 array containing the temporal
            median across loaded frames, or None if not computed.
    """

    def __init__(self, file_list: list[str], capacity: int = 128):
        self.file_list = [str(f) for f in file_list]
        self.n_frames = len(self.file_list)
        self.median_frame: np.ndarray | None = None
        self._mem_cache = CompressedFrameCache(capacity=capacity)
        self._derived_mem_cache: dict[str, CompressedFrameCache] = {}
        self._lock = threading.Lock()
        self._loading_done = threading.Event()
        self._loading_done.set()

    def get_frame(self, index: int) -> np.ndarray | None:
        """Retrieve a single frame as a float32 numpy array.

        Returns the frame from the in-memory cache if present, falling back
        to reading the source TIFF directly via :func:`load_raw`.

        Args:
            index: Zero-based frame index in ``[0, n_frames - 1]``.

        Returns:
            2-D ``float32`` array of shape ``(H, W)``, or ``None`` if
            ``index`` is out of range or the source file cannot be read.
        """
        if index < 0 or index >= self.n_frames:
            return None

        cached = self._mem_cache.get(index)
        if cached is not None:
            return cached

        filepath = self.file_list[index]
        try:
            raw = load_raw(filepath)
        except Exception:
            return None

        self._mem_cache.put(index, raw)
        return raw

    def set_frame(self, index: int, data: np.ndarray) -> None:
        """Store a frame directly in the in-memory cache.

        Args:
            index: Zero-based frame index.
            data: 2-D numpy array to store.
        """
        if index < 0:
            return
        self._mem_cache.put(index, data)

    def get_derived_frame(
        self,
        index: int,
        name: str | None = None,
        *,
        suffix: str | None = None,
    ) -> np.ndarray | None:
        """Retrieve a derived preprocessed frame from in-memory cache.

        Args:
            index: Zero-based frame index.
            name: Identifier for the derived frame (e.g. 'denoised_img').
            suffix: Alias parameter for name.

        Returns:
            2-D float32 numpy array or None if not cached.
        """
        key_name = name if name is not None else suffix
        if key_name is None or index < 0:
            return None

        with self._lock:
            cache = self._derived_mem_cache.get(key_name)
        if cache is None:
            return None
        return cache.get(index)

    def set_derived_frame(
        self,
        index: int,
        name: str | None = None,
        data: np.ndarray | None = None,
        *,
        suffix: str | None = None,
    ) -> None:
        """Store a derived preprocessed frame in the in-memory cache.

        Args:
            index: Zero-based frame index.
            name: Identifier for the derived frame.
            data: 2-D numpy array to store.
            suffix: Alias parameter for name.
        """
        key_name = name if name is not None else suffix
        if key_name is None or data is None or index < 0:
            return

        with self._lock:
            if key_name not in self._derived_mem_cache:
                self._derived_mem_cache[key_name] = CompressedFrameCache(capacity=128)
            cache = self._derived_mem_cache[key_name]

        cache.put(index, data)

    def compute_median(self) -> None:
        """Compute the pixel-wise temporal median over the current sequence into :attr:`median_frame`."""
        frames = []
        for idx in range(self.n_frames):
            frame = self.get_frame(idx)
            if frame is not None:
                frames.append(frame)
        if frames:
            self.median_frame = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
