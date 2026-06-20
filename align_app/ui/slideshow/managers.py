# align_app/ui/slideshow/managers.py

import os
import queue
import threading
import numpy as np

from align_app.core import (
    load_raw,
    apply_colormap,
    generate_aligned_sum,
    find_peak_line,
    find_peak_line_fast,
    compute_line_based_offset,
    phase_correlation_offset
)

class SlideshowManager:
    """
    State manager and business logic controller for the slideshow application.

    GUI Controller-Manager-View (CMV) Architecture:
      - Decouples UI rendering (View) from backend alignment logic (Manager).
      - Thread Safety: Employs a thread-safe `Queue` to post UI callback updates from background threads.
      - Background Preloading: Spawns a daemon thread on startup to preload raw TIFF arrays into memory to 
        eliminate lag during frame navigation.
      - Multilevel Caching:
        - `raw_cache`: Caches raw 2D float32 intensity arrays loaded from disk.
        - `rgb_cache`: Caches colormapped RGB images based on current clamping levels.
        - `offset_cache`: Caches calculated sub-pixel translation vectors (dx, dy) to avoid repeating SVD.
      - State Management: Keeps track of current frame indexes, zoom/pan parameters, colormaps, thresholds, 
        and manual alignment overrides.
      - Threshold Optimization: Runs a multi-stage search (coarse, fine, ultra-fine) on background worker threads
        to minimize line fit perpendicular spread for automated snapping.
    """
    def __init__(self, result_queue: queue.Queue):
        """Initializes the SlideshowManager state and configurations.

        Args:
            result_queue (queue.Queue): Thread-safe queue used to dispatch callbacks to the main GUI thread.
        """
        self.result_queue = result_queue
        self.file_list = []
        self.current_idx = 0
        self.pca_threshold = 99.9
        self.colormap = "viridis"
        self.warp_enabled = True

        # Clamping state
        self.intensity_min = 0.0
        self.intensity_max = 1.0
        self.clamping_floor = 0.0
        self.clamping_ceiling = 1.0

        # Reference line info
        self.ref_raw = None
        self.ref_origin = None
        self.ref_direction = None
        self.ref_threshold = 99.9

        # Per-frame states
        self.per_frame_threshold = {}
        self.per_frame_manual = {}
        self.per_frame_manual_dir = {}
        self.per_frame_origin = {}

        # Caching
        self.raw_cache = {}
        self.rgb_cache = {}
        self.offset_cache = {}

        # Current frame data (for compatibility and logic)
        self.current_raw = None
        self.current_rgb = None

        # Zoom & Panning
        self.zoom_level = 0
        self.zoom_steps = [1, 2, 4, 8, 16]
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Autoplay Config
        self.autoplay_active = False
        self.autoplay_speed_ms = 500

        # Manual alignment state
        self.manual_mode = False
        self.manual_clicks = []

    def start(self, file_list: list[str]):
        """Starts a new slideshow session with the provided list of files.

        Resets caching, zooming, and state tracking, then loads the first image 
        as the reference and spawns a background thread to preload remaining images.

        Args:
            file_list (list[str]): List of absolute file paths to the TIFF images.
        """
        self.file_list = file_list
        self.current_idx = 0
        self.ref_raw = None
        self.ref_origin = None
        self.ref_direction = None
        self.raw_cache.clear()
        self.rgb_cache.clear()
        self.offset_cache.clear()
        self.per_frame_threshold.clear()
        self.per_frame_manual.clear()
        self.per_frame_manual_dir.clear()
        self.per_frame_origin.clear()
        self.zoom_level = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.autoplay_active = False
        self.manual_mode = False
        self.manual_clicks = []

        self._load_reference()
        self._preload_all_in_background()

    def _load_reference(self):
        """Loads and processes the first frame as the global reference frame.

        Initializes intensity bounds, clamping limits, and computes the 
        origin/direction peak line vectors for the reference image.
        """
        if not self.file_list:
            return
        ref_raw = self.get_raw(self.file_list[0])
        if ref_raw is None:
            return
        self.ref_raw = ref_raw

        if ref_raw.size > 0:
            self.intensity_min = float(np.min(ref_raw))
            self.intensity_max = float(np.max(ref_raw))
            
            # Compute 95th percentile strictly on active (non-background) pixels
            active_pixels = ref_raw[ref_raw > self.intensity_min]
            if active_pixels.size > 0:
                p95 = float(np.percentile(active_pixels, 95.0))
            else:
                p95 = self.intensity_max
            
            # Final fallback: if still invalid or flat, default to maximum
            if p95 <= self.intensity_min:
                self.clamping_ceiling = self.intensity_max
            else:
                self.clamping_ceiling = p95
        else:
            self.intensity_min = 0.0
            self.intensity_max = 1.0
            self.clamping_ceiling = 1.0

        if self.intensity_max <= self.intensity_min:
            self.intensity_max = self.intensity_min + 1.0
            self.clamping_ceiling = self.intensity_max

        self.clamping_floor = self.intensity_min

        t = self.per_frame_threshold.get(0, self.pca_threshold)
        self.ref_threshold = t
        self.ref_origin, self.ref_direction = find_peak_line(ref_raw, t)
        self.per_frame_origin[0] = self.ref_origin.copy()

    def _preload_all_in_background(self):
        """Spawns a daemon thread to asynchronously load all raw image files into cache."""
        def _worker():
            for path in self.file_list:
                self.get_raw(path)
        threading.Thread(target=_worker, daemon=True).start()

    def get_raw(self, filepath: str) -> np.ndarray:
        """Retrieves the raw float32 image array for the given file, utilizing cache if available.

        Args:
            filepath (str): The absolute path to the image file.

        Returns:
            np.ndarray: The 2D image array, or None if loading fails.
        """
        if filepath in self.raw_cache:
            return self.raw_cache[filepath]
        try:
            raw = load_raw(filepath)
            self.raw_cache[filepath] = raw
            return raw
        except Exception:
            return None

    def get_rgb(self, filepath: str, colormap: str) -> np.ndarray:
        """Retrieves the colormapped RGB image array for the given file.

        Args:
            filepath (str): The absolute path to the image file.
            colormap (str): The matplotlib colormap name to apply.

        Returns:
            np.ndarray: The RGB image array (H, W, 3) mapped within clamping limits, or None if loading fails.
        """
        cache_key = (filepath, colormap)
        if cache_key in self.rgb_cache:
            return self.rgb_cache[cache_key]
        try:
            raw = self.get_raw(filepath)
            if raw is None:
                return None
            rgb = apply_colormap(raw, colormap,
                                 display_floor=self.clamping_floor,
                                 display_ceiling=self.clamping_ceiling)
            self.rgb_cache[cache_key] = rgb
            return rgb
        except Exception:
            return None

    @property
    def global_ref_origin(self) -> np.ndarray:
        return self.per_frame_manual.get(0, self.ref_origin)

    @property
    def global_ref_direction(self) -> np.ndarray:
        return self.per_frame_manual_dir.get(0, self.ref_direction)

    def get_offset(self, frame_idx: int) -> tuple[float, float]:
        """Calculates the x and y pixel translation offset of a frame relative to the reference frame.

        Args:
            frame_idx (int): The index of the frame in the file list.

        Returns:
            tuple[float, float]: A tuple containing the (dx, dy) translation offsets.
        """
        filepath = self.file_list[frame_idx]
        if frame_idx in self.per_frame_manual:
            manual_centroid = self.per_frame_manual[frame_idx]
            if self.global_ref_origin is None:
                return (0.0, 0.0)
            delta = manual_centroid - self.global_ref_origin
            return float(delta[0]), float(delta[1])

        cache_key = (filepath, frame_idx)
        if cache_key in self.offset_cache:
            return self.offset_cache[cache_key]

        raw = self.get_raw(filepath)
        if raw is None or self.ref_raw is None or self.global_ref_direction is None:
            return (0.0, 0.0)

        target_threshold = self.per_frame_threshold.get(frame_idx, self.ref_threshold)
        try:
            dx, dy = compute_line_based_offset(
                self.ref_raw, raw,
                self.global_ref_direction, self.global_ref_origin,
                self.ref_threshold, target_threshold
            )
            self.offset_cache[cache_key] = (dx, dy)
            return (dx, dy)
        except Exception:
            try:
                dx, dy = phase_correlation_offset(self.ref_raw, raw)
                return (dx, dy)
            except Exception:
                return (0.0, 0.0)

    def get_current_threshold(self) -> float:
        """Gets the PCA threshold percentage currently active for the active frame.

        Returns:
            float: The active PCA threshold (e.g., 99.9).
        """
        return self.per_frame_threshold.get(self.current_idx, self.ref_threshold)

    def set_current_threshold(self, val: float):
        """Sets a custom PCA threshold for the current frame and invalidates relevant caches.

        Args:
            val (float): The new threshold percentage.
        """
        self.per_frame_threshold[self.current_idx] = val
        if self.current_idx < len(self.file_list):
            filepath = self.file_list[self.current_idx]
            self.offset_cache.pop((filepath, self.current_idx), None)
        self.per_frame_origin.pop(self.current_idx, None)

    def apply_pca_change(self):
        """Applies the globally modified PCA threshold to the current frame.

        If the current frame is the reference frame (index 0), reloads the reference state 
        and resets all cached offsets and origins.
        """
        t = self.pca_threshold
        self.set_current_threshold(t)
        if self.current_idx == 0:
            self._load_reference()
            self.offset_cache.clear()
            self.per_frame_origin.clear()
            if self.ref_origin is not None:
                self.per_frame_origin[0] = self.ref_origin.copy()

    def zoom_in(self, canvas_w: int, canvas_h: int):
        """Increases the zoom level and updates panning offsets to center the view.

        Args:
            canvas_w (int): Current width of the display canvas.
            canvas_h (int): Current height of the display canvas.
        """
        if self.zoom_level < len(self.zoom_steps) - 1:
            self.zoom_level += 1
            if self.zoom_level > 0:
                self.compute_centroid_pan(canvas_w, canvas_h)

    def zoom_out(self, canvas_w: int, canvas_h: int):
        """Decreases the zoom level and updates panning offsets to center the view.

        Args:
            canvas_w (int): Current width of the display canvas.
            canvas_h (int): Current height of the display canvas.
        """
        if self.zoom_level > 0:
            self.zoom_level -= 1
            if self.zoom_level == 0:
                self.pan_offset_x = 0
                self.pan_offset_y = 0
            else:
                self.compute_centroid_pan(canvas_w, canvas_h)

    def reset_view(self):
        """Resets the zoom level and panning offsets to their default initial state."""
        self.zoom_level = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

    def get_display_origin(self) -> np.ndarray:
        """Retrieves the origin pixel coordinates used for centering the display.

        Returns:
            np.ndarray: The [x, y] coordinates of the peak line origin or manual centroid.
        """
        if self.current_idx in self.per_frame_manual:
            return self.per_frame_manual[self.current_idx]
        if self.current_idx in self.per_frame_origin:
            return self.per_frame_origin[self.current_idx]
        return self.ref_origin

    def get_display_direction(self) -> np.ndarray:
        """Retrieves the direction vector used for drawing the line."""
        if self.current_idx == 0 and 0 in self.per_frame_manual_dir:
            return self.per_frame_manual_dir[0]
        return self.global_ref_direction

    def compute_centroid_pan(self, canvas_w: int, canvas_h: int, image_w=3840, image_h=2048):
        """Computes panning offsets required to keep the spectral peak centered on screen.

        Args:
            canvas_w (int): Width of the display canvas.
            canvas_h (int): Height of the display canvas.
            image_w (int, optional): Original image width. Defaults to 3840.
            image_h (int, optional): Original image height. Defaults to 2048.
        """
        origin = self.get_display_origin()
        if origin is None or canvas_w <= 1 or canvas_h <= 1:
            return
        base_scale = min(canvas_w / image_w, canvas_h / image_h)
        scale = base_scale * self.zoom_steps[self.zoom_level]
        nw = int(image_w * scale)
        nh = int(image_h * scale)
        base_dx = (canvas_w - nw) // 2
        base_dy = (canvas_h - nh) // 2
        cx = base_dx + origin[0] * scale
        cy = base_dy + origin[1] * scale
        self.pan_offset_x = int(canvas_w / 2 - cx)
        self.pan_offset_y = int(canvas_h / 2 - cy)

    def process_manual_click(self, clicks: list[tuple[int, int]], lb_dx: int, lb_dy: int, lb_scale: float):
        """Processes two manual user clicks to set a custom reference midpoint for a frame.

        Args:
            clicks (list[tuple[int, int]]): List of (x, y) canvas click coordinates.
            lb_dx (int): The horizontal padding/offset of the displayed image on canvas.
            lb_dy (int): The vertical padding/offset of the displayed image on canvas.
            lb_scale (float): The current display scale factor of the image on canvas.

        Returns:
            bool: True if clicks were successfully processed, False otherwise.
        """
        if len(clicks) < 2:
            return False
        cx1, cy1 = clicks[0]
        cx2, cy2 = clicks[1]
        ix1 = (cx1 - lb_dx) / lb_scale
        iy1 = (cy1 - lb_dy) / lb_scale
        ix2 = (cx2 - lb_dx) / lb_scale
        iy2 = (cy2 - lb_dy) / lb_scale

        midpoint = np.array([(ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0])
        
        direction = np.array([ix2 - ix1, iy2 - iy1], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm
        else:
            direction = self.ref_direction
            
        if self.warp_enabled and self.current_idx > 0 and self.ref_raw is not None:
            dx, dy = self.get_offset(self.current_idx)
            midpoint = midpoint + np.array([dx, dy])

        self.per_frame_manual[self.current_idx] = midpoint
        if self.current_idx == 0:
            self.per_frame_manual_dir[0] = direction
            self.offset_cache.clear()
            
        if self.current_idx < len(self.file_list):
            filepath = self.file_list[self.current_idx]
            self.offset_cache.pop((filepath, self.current_idx), None)
        return True

    def clear_manual_line(self):
        """Clears any manual alignment override applied to the current frame."""
        self.per_frame_manual.pop(self.current_idx, None)
        self.per_frame_manual_dir.pop(self.current_idx, None)
        if self.current_idx == 0:
            self.offset_cache.clear()
        elif self.current_idx < len(self.file_list):
            filepath = self.file_list[self.current_idx]
            self.offset_cache.pop((filepath, self.current_idx), None)

    def run_auto_snap(self, on_complete):
        """Asynchronously searches for the optimal PCA threshold for the current frame.

        Args:
            on_complete (callable): Callback executed on the main thread with the optimal threshold float.
        """
        idx = self.current_idx
        if idx >= len(self.file_list):
            return
        raw = self.get_raw(self.file_list[idx])
        if raw is None:
            return

        def _worker():
            best = self._find_best_threshold(raw)
            self.result_queue.put(lambda: on_complete(best))
        threading.Thread(target=_worker, daemon=True).start()

    def run_auto_snap_all(self, on_progress, on_complete):
        """Asynchronously searches for the optimal PCA threshold for all frames.

        Args:
            on_progress (callable): Callback executed on the main thread with current and total frame counts.
            on_complete (callable): Callback executed on the main thread with a dict of {frame_idx: best_threshold}.
        """
        n_frames = len(self.file_list)
        def _worker():
            results = {}
            for idx in range(n_frames):
                raw = self.get_raw(self.file_list[idx])
                if raw is not None:
                    results[idx] = self._find_best_threshold(raw)
                self.result_queue.put(lambda idx_c=idx+1: on_progress(idx_c, n_frames))
            
            self.result_queue.put(lambda: on_complete(results))
        threading.Thread(target=_worker, daemon=True).start()

    def _find_best_threshold(self, raw: np.ndarray) -> float:
        """Finds the PCA threshold that minimizes the perpendicular spread of the peak line.

        Args:
            raw (np.ndarray): The raw 2D intensity array.

        Returns:
            float: The optimal PCA threshold percentage.
        """
        h, w = raw.shape
        flat = raw.ravel()
        sorted_idx = np.argsort(flat)
        sorted_rows = sorted_idx // w
        sorted_cols = sorted_idx % w
        n_total = len(flat)

        best_threshold = 99.9
        best_spread = float('inf')

        def _eval_at(t_pct):
            cutoff = int(t_pct / 100.0 * n_total)
            if n_total - cutoff < 5:
                return None
            rows = sorted_rows[cutoff:]
            cols = sorted_cols[cutoff:]
            points = np.column_stack((cols, rows)).astype(np.float64)
            weights = raw[rows, cols].astype(np.float64)
            w_min, w_max = np.min(weights), np.max(weights)
            if w_max - w_min > 1e-9:
                weights = ((weights - w_min) / (w_max - w_min)) ** 2
            else:
                weights = np.ones(len(points))
            weights = np.clip(weights, 1e-6, None)
            _, _, spread = find_peak_line_fast(points, weights)
            return spread

        # Coarse pass
        for t_int in range(9800, 10000):
            t = t_int / 100.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        # Fine pass
        fine_lo = max(98.0, best_threshold - 0.1)
        fine_hi = min(99.999, best_threshold + 0.1)
        for t_int in range(int(fine_lo * 1000), int(fine_hi * 1000) + 1):
            t = t_int / 1000.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        # Ultra-fine pass
        uf_lo = max(98.0, best_threshold - 0.005)
        uf_hi = min(99.9999, best_threshold + 0.005)
        for t_int in range(int(uf_lo * 10000), int(uf_hi * 10000) + 1):
            t = t_int / 10000.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        return best_threshold

    def compute_all_offsets_for_export(self, progress_callback) -> dict[int, tuple[float, float]]:
        """Synchronously computes sub-pixel translation offsets for all frames relative to the reference.

        Args:
            progress_callback (callable): Function to call with current progress updates.

        Returns:
            dict[int, tuple[float, float]]: Dictionary mapping frame index to (dx, dy) translation vectors.
        """
        n_frames = len(self.file_list)
        offsets = {0: (0.0, 0.0)}
        for idx in range(1, n_frames):
            progress_callback(idx, n_frames)
            offsets[idx] = self.get_offset(idx)
        return offsets

    def start_export(self, save_path: str, offsets: dict[int, tuple[float, float]], on_progress, on_complete):
        """Starts a background thread to generate and save the aligned sum image.

        Args:
            save_path (str): The absolute file path to save the resulting TIFF.
            offsets (dict[int, tuple[float, float]]): Dictionary of precomputed frame offsets.
            on_progress (callable): Callback executed with current frame processing progress.
            on_complete (callable): Callback executed upon success or failure.
        """
        first_file = self.file_list[0]
        ref_raw = self.get_raw(first_file)
        if ref_raw is None:
            on_complete(False, "Could not load reference frame.")
            return

        def _progress(current, total):
            self.result_queue.put(lambda: on_progress(current, total))

        def _worker():
            try:
                result = generate_aligned_sum(
                    self.file_list, self.get_raw, offsets,
                    ref_raw.shape, progress_callback=_progress
                )
                import tifffile
                tifffile.imwrite(save_path, result)
                self.result_queue.put(lambda: on_complete(True, save_path))
            except Exception as e:
                self.result_queue.put(lambda err=str(e): on_complete(False, err))

        threading.Thread(target=_worker, daemon=True).start()
