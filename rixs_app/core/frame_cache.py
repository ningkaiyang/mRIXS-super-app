"""Compressed frame cache for sparse RIXS detector frames.

Detector frames are typically 2048x3840 float32 (31.5 MB uncompressed), but
dark-subtracted RIXS frames are ~99.8% background zeros. By casting to np.float16
and compressing using numcodecs.Zstd(level=1), frames compress ~2000x down to
~16-50 KB each.

This component provides thread-safe LRU caching allowing 128+ frames to reside
in < 5 MB of RAM while decompressing back to float32 in < 1 ms.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Dict, Optional, Tuple

import numcodecs
import numpy as np


class CompressedFrameCache:
    """Thread-safe LRU in-memory cache for compressed detector frames."""

    def __init__(self, capacity: int = 128, compression_level: int = 1) -> None:
        """Initialize cache with capacity limit and Zstd compression level.

        Args:
            capacity: Maximum number of frames retained before LRU eviction. Must be > 0.
            compression_level: Zstandard compression level (1 is fastest and optimal for speed).

        Raises:
            ValueError: If capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be greater than 0, got {capacity}")

        self._capacity = int(capacity)
        self._compression_level = int(compression_level)
        self._codec = numcodecs.Zstd(level=self._compression_level)
        self._lock = threading.Lock()

        # Cache storage: index -> (compressed_bytes, original_shape)
        self._cache: OrderedDict[int, Tuple[bytes, Tuple[int, ...]]] = OrderedDict()
        self._total_compressed_bytes: int = 0

    @property
    def capacity(self) -> int:
        """Maximum frame capacity of this cache."""
        return self._capacity

    def put(self, index: int, frame: np.ndarray) -> None:
        """Compress and insert or update a frame in the cache with LRU eviction.

        Args:
            index: Frame index identifier.
            frame: Detector frame array (any numeric dtype, converted to float16 for compression).
        """
        shape = frame.shape
        f16 = np.ascontiguousarray(frame, dtype=np.float16)
        compressed = bytes(self._codec.encode(f16))
        comp_len = len(compressed)

        with self._lock:
            if index in self._cache:
                old_bytes, _ = self._cache.pop(index)
                self._total_compressed_bytes -= len(old_bytes)
            elif len(self._cache) >= self._capacity:
                _, (evicted_bytes, _) = self._cache.popitem(last=False)
                self._total_compressed_bytes -= len(evicted_bytes)

            self._cache[index] = (compressed, shape)
            self._total_compressed_bytes += comp_len

    def get(self, index: int) -> Optional[np.ndarray]:
        """Retrieve and decompress a frame as a float32 array.

        Args:
            index: Frame index identifier.

        Returns:
            Decompressed frame as float32 ndarray, or None if not present.
        """
        with self._lock:
            if index not in self._cache:
                return None
            self._cache.move_to_end(index)
            compressed, shape = self._cache[index]

        decompressed = self._codec.decode(compressed)
        arr_f16 = np.frombuffer(decompressed, dtype=np.float16).reshape(shape)
        return arr_f16.astype(np.float32)

    def preload_batch(self, frames: Dict[int, np.ndarray]) -> None:
        """Preload a dictionary of frames into the cache.

        Args:
            frames: Mapping of frame index to frame ndarray.
        """
        for idx, frame in frames.items():
            self.put(idx, frame)

    def has(self, index: int) -> bool:
        """Check whether a frame index is present in the cache.

        Args:
            index: Frame index identifier.

        Returns:
            True if present, False otherwise.
        """
        with self._lock:
            return index in self._cache

    def __contains__(self, index: int) -> bool:
        """Enable `index in cache` syntax."""
        return self.has(index)

    def clear(self) -> None:
        """Clear all cached frames and reset memory tracking."""
        with self._lock:
            self._cache.clear()
            self._total_compressed_bytes = 0

    def memory_usage_mb(self) -> float:
        """Return total memory footprint of cached compressed frames in megabytes."""
        with self._lock:
            return self._total_compressed_bytes / (1024.0 * 1024.0)

    def __len__(self) -> int:
        """Return current number of frames in cache."""
        with self._lock:
            return len(self._cache)
