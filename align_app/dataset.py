"""Zarr-backed sequence manager for memory-efficient frame caching.

:class:`ZarrSequenceManager` speeds up the loading of a TIFF sequence
(e.g. 10 x 30 MB .tif files = 300 MB raw data) by storing each frame as
an individual array inside a shared ``zarr.Group`` on disk.

Cache design
------------
A single ``tif-cache/frames.zarr`` group is created (or re-opened) inside the
same directory as the source TIF files.  Each frame is stored under a key
derived from an MD5 hash of its ``filepath + mtime``, so:

* Reordering files — no re-encoding needed; keys are reused.
* Adding / removing files — only genuinely new frames are written.
* Modifying a file — mtime changes → new key → old entry is silently ignored.

This means the ``.zarr`` group grows only when *new* data are encountered, and
a 100 % cache-hit rate is achieved whenever the same set of source files (in
any order) is reloaded.

In addition to frame-level access, the manager computes the **temporal
median** of all frames (once, in a background thread) and exposes it as
:attr:`median_frame`.  The median is a noise-resistant "average" image that
:class:`~align_app.ui.slideshow.view.SlideshowView` uses as the alignment
reference so that no single noisy frame dominates the registration.
"""

import zarr
import tifffile
import numpy as np
import os
import pathlib
import hashlib
import threading


def _frame_key(filepath: str) -> str:
    """Compute a stable cache key from a file's path and modification time.

    Args:
        filepath: Absolute path to the TIFF source file.

    Returns:
        12-character hex MD5 digest of ``filepath + mtime``.
    """
    try:
        mtime = str(os.path.getmtime(filepath))
    except OSError:
        mtime = ""
    return hashlib.md5((filepath + mtime).encode()).hexdigest()[:12]


class ZarrSequenceManager:
    """Manages a Zarr-backed frame cache and a temporal median reference image.

    On construction the manager opens (or creates) a ``frames.zarr`` group
    inside the ``tif-cache/`` directory adjacent to the source TIFFs.  Each
    frame is stored under its own content-addressed key so that cache entries
    survive reordering, additions, or deletions of files in the sequence.

    Frame reads via :meth:`get_frame` return the cached array when available,
    or fall back to live TIFF decoding (writing the result into the group for
    future accesses).  All frames are also populated asynchronously in a
    background daemon thread.

    Args:
        file_list: Ordered list of absolute paths to the TIFF source files.
            The order determines the frame index mapping used by
            :meth:`get_frame`.
        chunk_size: Number of frames per Zarr chunk along the time axis.
            Kept for API compatibility; not used in the per-frame group
            design (each frame has its own 2-D array).  Defaults to ``10``.

    Attributes:
        file_list (list[str]): The ordered source file paths.
        chunk_size (int): Kept for backwards compatibility.
        n_frames (int): Total number of frames in the sequence.
        zarr_group (zarr.Group | None): Open Zarr group acting as the
            key-value store for all cached frames.  ``None`` until
            :meth:`_init_zarr` succeeds.
        median_frame (np.ndarray | None): 2-D ``float32`` array of shape
            ``(H, W)`` containing the pixel-wise temporal median across all
            frames.  ``None`` until :meth:`compute_median` completes.
    """

    def __init__(self, file_list: list[str], chunk_size: int = 10):
        self.file_list = file_list
        self.chunk_size = chunk_size
        self.n_frames = len(file_list)
        self.zarr_group = None
        self.median_frame = None
        self._loading_done = threading.Event()
        self._init_zarr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_zarr(self) -> None:
        """Open or create the shared ``frames.zarr`` group cache.

        The group lives at ``<tif_dir>/tif-cache/frames.zarr`` and is shared
        by all :class:`ZarrSequenceManager` instances that point at the same
        directory.  Individual frames are keyed by :func:`_frame_key`.
        """
        if not self.file_list:
            self._loading_done.set()
            return

        tif_dir = os.path.dirname(os.path.abspath(self.file_list[0]))
        cache_dir = os.path.join(tif_dir, "tif-cache")
        os.makedirs(cache_dir, exist_ok=True)
        group_path = os.path.join(cache_dir, "frames.zarr")

        # Open in append mode: creates the group if absent, re-opens if present.
        # Wrap in pathlib.Path to bypass URL/URI parsing and correctly support '#' in directory paths.
        self.zarr_group = zarr.open_group(pathlib.Path(group_path), mode="a")
        self._load_all_async()

    def _load_all_async(self) -> None:
        """Spawn a daemon thread to populate uncached frames and compute the median.

        Each frame is checked by its content-addressed key.  Only frames whose
        key is not yet present in the group are read from the source TIFF and
        written into the cache.  After all frames are confirmed present,
        :meth:`compute_median` is called.
        """
        def _worker():
            try:
                for filepath in self.file_list:
                    key = _frame_key(filepath)
                    if key not in self.zarr_group:
                        try:
                            raw = tifffile.imread(filepath).astype(np.float32)
                            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                            self.zarr_group[key] = raw
                        except Exception:
                            pass
                self.compute_median()
            finally:
                self._loading_done.set()

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self, index: int) -> np.ndarray | None:
        """Retrieve a single frame as a float32 numpy array.

        Returns the frame from the Zarr group cache if its key is present, or
        falls back to reading the source TIFF directly and writing the result
        into the group for future accesses.

        Args:
            index: Zero-based frame index in ``[0, n_frames - 1]``.

        Returns:
            2-D ``float32`` array of shape ``(H, W)``, or ``None`` if
            ``index`` is out of range or the source file cannot be read.
        """
        if index < 0 or index >= self.n_frames:
            return None

        filepath = self.file_list[index]
        key = _frame_key(filepath)

        if key in self.zarr_group:
            return self.zarr_group[key][:]

        # Cache miss: read from disk and write into cache for next time.
        try:
            raw = tifffile.imread(filepath).astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            self.zarr_group[key] = raw
            return raw
        except Exception:
            return None

    def compute_median(self) -> None:
        """Compute the pixel-wise temporal median over the current sequence.

        Gathers the cached arrays for all frames in :attr:`file_list`, stacks
        them in memory, and computes ``np.median`` along axis 0.  The result
        is stored in :attr:`median_frame` and used by
        :class:`~align_app.ui.slideshow.view.SlideshowView` as the
        noise-resistant alignment reference.

        Note:
            This method blocks the calling thread until computation is
            complete.  It should only be called from a daemon thread (as done
            by :meth:`_load_all_async`) to avoid freezing the UI.
        """
        frames = []
        for filepath in self.file_list:
            key = _frame_key(filepath)
            if key in self.zarr_group:
                frames.append(self.zarr_group[key][:])
            else:
                # Frame not yet cached — fall back to live read for median.
                try:
                    raw = tifffile.imread(filepath).astype(np.float32)
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    frames.append(raw)
                except Exception:
                    pass
        if frames:
            self.median_frame = np.median(np.stack(frames, axis=0), axis=0)
