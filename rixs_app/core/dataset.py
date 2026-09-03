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
:class:`~rixs_app.ui.alignment_slideshow.slideshow_view.SlideshowView` uses as the alignment
reference so that no single noisy frame dominates the registration.
"""

import zarr
import tifffile
import numpy as np
import os
import pathlib
import hashlib
import threading
import datetime


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


def _write_cache_readme(cache_dir: str, tif_dir: str, file_list: list[str]) -> None:
    """Write or update a human-readable README_CACHE.txt and .gitignore inside the cache folder.

    Args:
        cache_dir: Path to the cache directory (e.g. ``<tif_dir>/tif-cache``).
        tif_dir: Path to the source TIFF directory.
        file_list: List of source TIFF file paths.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        gitignore_path = os.path.join(cache_dir, ".gitignore")
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("*\n")

        readme_path = os.path.join(cache_dir, "README_CACHE.txt")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 80,
            "QERLIN Beamline 6.0.2 RIXS Super-App — Frame Performance Cache",
            "=" * 80,
            "This directory (tif-cache/) was automatically generated to accelerate GUI",
            "slider scrubbing, temporal median calculations, and alignment processing.",
            "",
            f"Source Directory : {tif_dir}",
            f"Last Updated     : {now_str}",
            "",
            "SAFETY NOTE:",
            "- This directory contains ONLY cached binary data.",
            "- It is 100% SAFE TO DELETE at any time.",
            "- If deleted, the application will automatically regenerate it on the next run.",
            "",
            "Cached Files:",
        ]

        for idx, filepath in enumerate(file_list, 1):
            fname = os.path.basename(filepath)
            key = _frame_key(filepath)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                mtime_str = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"  - [{idx:03d}] {fname} (Key: {key}, Size: {size_mb:.1f} MB, Modified: {mtime_str})")
            except OSError:
                lines.append(f"  - [{idx:03d}] {fname} (Key: {key})")

        lines.append("=" * 80)
        lines.append("")

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


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

    Attributes:
        file_list (list[str]): The ordered source file paths.
        n_frames (int): Total number of frames in the sequence.
        zarr_group (zarr.Group | None): Open Zarr group acting as the
            key-value store for all cached frames.  ``None`` until
            :meth:`_init_zarr` succeeds.
        median_frame (np.ndarray | None): 2-D ``float32`` array of shape
            ``(H, W)`` containing the pixel-wise temporal median across all
            frames.  ``None`` until :meth:`compute_median` completes.
        cache_dir (str | None): Path to the on-disk cache directory.
        tif_dir (str | None): Path to the directory containing source TIFF files.
    """

    def __init__(self, file_list: list[str]):
        self.file_list = file_list
        self.n_frames = len(file_list)
        self.zarr_group = None
        self.median_frame = None
        self.cache_dir = None
        self.tif_dir = None
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
        if not self.file_list or not any(os.path.exists(f) for f in self.file_list):
            self._loading_done.set()
            return

        self.tif_dir = os.path.dirname(os.path.abspath(self.file_list[0]))
        try:
            self.cache_dir = os.path.join(self.tif_dir, "tif-cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            _write_cache_readme(self.cache_dir, self.tif_dir, self.file_list)
            group_path = os.path.join(self.cache_dir, "frames.zarr")
            self.zarr_group = zarr.open_group(pathlib.Path(group_path), mode="a")
        except (PermissionError, OSError):
            try:
                import tempfile
                dir_hash = hashlib.md5(self.tif_dir.encode("utf-8")).hexdigest()
                self.cache_dir = os.path.join(tempfile.gettempdir(), f"rixs_cache_{dir_hash}")
                group_path = self.cache_dir
                self.zarr_group = zarr.open_group(pathlib.Path(group_path), mode="a")
                _write_cache_readme(self.cache_dir, self.tif_dir, self.file_list)
            except Exception:
                self.zarr_group = None
                self.cache_dir = None
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
                    try:
                        key = _frame_key(filepath)
                        if self.zarr_group is not None and key not in self.zarr_group:
                            raw = tifffile.imread(filepath).astype(np.float32)
                            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                            if self.zarr_group is not None:
                                self.zarr_group[key] = raw
                    except Exception:
                        pass
                try:
                    self.compute_median()
                    if self.cache_dir and self.tif_dir:
                        _write_cache_readme(self.cache_dir, self.tif_dir, self.file_list)
                except Exception:
                    pass
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

        if self.zarr_group is not None and key in self.zarr_group:
            return self.zarr_group[key][:]

        # Cache miss: read from disk and write into cache for next time.
        try:
            raw = tifffile.imread(filepath).astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

        if self.zarr_group is not None:
            try:
                self.zarr_group[key] = raw
            except Exception:
                pass

        return raw

    def get_derived_frame(self, index: int, suffix: str) -> np.ndarray | None:
        """Retrieve a derived preprocessed frame (e.g. 'denoised', 'masked') from cache.

        Args:
            index: Zero-based frame index.
            suffix: Suffix for the key (e.g. 'denoised', 'masked').

        Returns:
            2-D float32 numpy array or None if not cached.
        """
        if index < 0 or index >= self.n_frames or self.zarr_group is None:
            return None
        filepath = self.file_list[index]
        raw_key = _frame_key(filepath)
        key = f"{raw_key}_{suffix}"
        if key in self.zarr_group:
            return self.zarr_group[key][:]
        return None

    def set_derived_frame(self, index: int, suffix: str, data: np.ndarray) -> None:
        """Store a derived preprocessed frame in the cache.

        Args:
            index: Zero-based frame index.
            suffix: Suffix for the key (e.g. 'denoised', 'masked').
            data: 2-D numpy array to store.
        """
        if index < 0 or index >= self.n_frames or self.zarr_group is None:
            return
        filepath = self.file_list[index]
        raw_key = _frame_key(filepath)
        key = f"{raw_key}_{suffix}"
        try:
            self.zarr_group[key] = data
        except Exception:
            pass

    def compute_median(self) -> None:
        """Compute the pixel-wise temporal median over the current sequence.

        Gathers the cached arrays for all frames in :attr:`file_list`, stacks
        them in memory, and computes ``np.median`` along axis 0.  The result
        is stored in :attr:`median_frame` and used by
        :class:`~rixs_app.ui.alignment_slideshow.slideshow_view.SlideshowView` as the
        noise-resistant alignment reference.

        Note:
            This method blocks the calling thread until computation is
            complete.  It should only be called from a daemon thread (as done
            by :meth:`_load_all_async`) to avoid freezing the UI.
        """
        frames = []
        if self.zarr_group is None:
            for filepath in self.file_list:
                try:
                    raw = tifffile.imread(filepath).astype(np.float32)
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    frames.append(raw)
                except Exception:
                    pass
            if frames:
                self.median_frame = np.median(np.stack(frames, axis=0), axis=0)
            return

        for filepath in self.file_list:
            key = _frame_key(filepath)
            if self.zarr_group is not None and key in self.zarr_group:
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


class CLIZarrSequenceManager(ZarrSequenceManager):
    """Synchronous variant of ZarrSequenceManager for headless CLI operation.

    Overrides the asynchronous background loading with a synchronous loop
    to ensure all frames are cached and the temporal median is computed
    before alignment begins.
    """

    def _load_all_async(self):
        """Load all frames synchronously and compute the temporal median.

        Iterates over every file in *file_list*, caching each one in the
        Zarr group if it is not already present.  After all frames are
        cached, :meth:`compute_median` is called so that
        :attr:`median_frame` is available immediately.
        """
        if self.zarr_group is not None:
            for filepath in self.file_list:
                key = _frame_key(filepath)
                if key not in self.zarr_group:
                    try:
                        raw = tifffile.imread(filepath).astype(np.float32)
                        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                        if self.zarr_group is not None:
                            self.zarr_group[key] = raw
                    except Exception as e:
                        import sys
                        print(
                            f"  Warning: Failed to cache {os.path.basename(filepath)}: {e}",
                            file=sys.stderr,
                        )
        self.compute_median()
        if self.cache_dir and self.tif_dir:
            _write_cache_readme(self.cache_dir, self.tif_dir, self.file_list)
        self._loading_done.set()

