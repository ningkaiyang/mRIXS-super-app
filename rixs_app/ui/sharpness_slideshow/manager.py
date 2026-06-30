"""Logical manager driving the state, Zarr caching, and calculations of the sharpness slideshow."""

import os
import queue
import threading
import numpy as np
import hashlib
import inspect
from rixs_app.core.dataset import ZarrSequenceManager
from rixs_app.core.sharpness import run_sharpness_pipeline, denoise_image

def _call_on_complete(on_complete, success, err_msg=None):
    try:
        sig = inspect.signature(on_complete)
        sig.bind(success, err_msg)
        on_complete(success, err_msg)
    except (TypeError, ValueError):
        try:
            on_complete()
        except TypeError:
            on_complete(success, err_msg)

class SharpnessManager:
    """Logical model encapsulating sharpness pipeline computations, state caching, and threads."""

    def __init__(self, result_queue: queue.Queue):
        self.result_queue = result_queue
        self._local = threading.local()
        self._file_list = []
        self.lock = threading.Lock()

    @property
    def file_list(self):
        if hasattr(self._local, "file_list") and self._local.file_list is not None:
            return self._local.file_list
        return self._file_list

    @file_list.setter
    def file_list(self, value):
        self._local.file_list = None
        self._file_list = value
        self.current_idx = 0
        self.colormap = "viridis"
        self.pipeline_stage = "Raw"
        self.engine = "Line-finding and Scoring"
        self.metric = "score"
        self.autoplay_active = False

        # Slicing
        self.intensity_min = 0.0
        self.intensity_max = 1.0
        self.slicing_floor = 0.0
        self.slicing_ceiling = 1.0

        # Precomputed/cached values in RAM
        self.scores = {}
        self.centroids = {}
        self.directions = {}
        self.profiles = {}

        # Reference peak line to bypass PCA fit on high noise files
        self.global_ref_line = None
        self.global_raw_std = None
        self._global_ref_line_calculated = False
        self._global_ref_line_calculating = False
        self._global_ref_line_event = threading.Event()

        # Zarr cache
        self.zarr_manager = None
        self.session_id = None

    def start(self, file_list: list[str]):
        """Resets the manager state and launches data preloading."""
        manager = ZarrSequenceManager(file_list)
        self.file_list = file_list
        self.current_idx = 0
        self.autoplay_active = False
        with self.lock:
            self.zarr_manager = manager
            self.session_id = object()
            self.scores.clear()
            self.centroids.clear()
            self.directions.clear()
            self.profiles.clear()
        self.global_ref_line = None
        self.global_raw_std = None
        self._global_ref_line_calculated = False
        self._global_ref_line_calculating = False
        self._global_ref_line_event = threading.Event()
        self._load_reference_bounds()

    def _load_reference_bounds(self):
        """Loads first frame to establish display thresholds and noise metrics."""
        current_zarr_manager = self.zarr_manager
        if current_zarr_manager is None or not self.file_list:
            return
        ref_raw = None
        for idx in range(len(self.file_list)):
            frame = current_zarr_manager.get_frame(idx)
            if frame is not None:
                ref_raw = frame
                break

        if ref_raw is None:
            self.intensity_min = 0.0
            self.intensity_max = 1.0
            self.slicing_floor = 0.0
            self.slicing_ceiling = 1.0
            self.global_raw_std = 1.0
            return

        self.intensity_min = float(np.min(ref_raw))
        self.intensity_max = float(np.max(ref_raw))

        # Compute 60th percentile on active pixels
        active_pixels = ref_raw[ref_raw > self.intensity_min]
        if active_pixels.size > 0:
            self.slicing_ceiling = float(np.percentile(active_pixels, 60.0))
        else:
            self.slicing_ceiling = self.intensity_max

        self.slicing_floor = self.intensity_min
        self.global_raw_std = float(np.std(ref_raw))

    def get_frame_pipeline_data(self, idx: int) -> dict:
        """Retrieves or calculates the sharpness pipeline breakdown dictionary for a frame."""
        self._local.file_list = None
        current_zarr_manager = self.zarr_manager
        current_session = self.session_id
        session_file_list = self._file_list
        if idx < 0 or idx >= len(session_file_list):
            return {}

        # 1. Return cached metadata if present
        with self.lock:
            if self.session_id is not current_session:
                return None
            cached = (
                idx in self.scores
                and idx in self.centroids
                and idx in self.directions
                and idx in self.profiles
            )
            if cached:
                cached_data = {
                    "score": self.scores[idx],
                    "centroid": self.centroids[idx],
                    "direction": self.directions[idx],
                    "1d_profile": self.profiles[idx]
                }
            else:
                cached_data = None
        
        if self.session_id is not current_session:
            return None

        # 2. Get frames from Zarr cache
        if current_zarr_manager is None:
            return None
        denoised_img = current_zarr_manager.get_derived_frame(idx, "denoised_img")
        masked_img = current_zarr_manager.get_derived_frame(idx, "masked_img")
        grad_img = current_zarr_manager.get_derived_frame(idx, "grad_img")
        raw_img = current_zarr_manager.get_frame(idx)

        if raw_img is None:
            return None

        if cached_data is not None and denoised_img is not None and masked_img is not None:
            if self.session_id is not current_session:
                return None
            self._local.file_list = session_file_list
            res = {
                "raw_img": raw_img,
                "denoised_img": denoised_img,
                "masked_img": masked_img,
                "score": cached_data["score"],
                "centroid": cached_data["centroid"],
                "direction": cached_data["direction"],
                "1d_profile": cached_data["1d_profile"]
            }
            if grad_img is not None:
                res["grad_img"] = grad_img
            return res

        # 3. Cache Miss: Run processing pipeline
        if self.global_raw_std is not None and self.global_raw_std > 500.0 and not self._global_ref_line_calculated:
            if self.session_id is not current_session:
                return None
            is_calculating_thread = False
            with self.lock:
                if not self._global_ref_line_calculated:
                    if not self._global_ref_line_calculating:
                        self._global_ref_line_calculating = True
                        is_calculating_thread = True

            if is_calculating_thread:
                try:
                    sum_img = None
                    for idx_ref in range(len(session_file_list)):
                        if self.session_id is not current_session:
                            return None
                        frame = current_zarr_manager.get_frame(idx_ref)
                        if frame is not None:
                            denoised = denoise_image(frame, despike=False, bilateral=False)
                            if sum_img is None:
                                sum_img = denoised
                            else:
                                sum_img += denoised
                    
                    calculated_ref_line = None
                    if sum_img is not None:
                        calculated_ref_line = fit_line_robustly(sum_img, crop_y=200)

                    with self.lock:
                        if self.session_id is current_session:
                            self.global_ref_line = calculated_ref_line
                            self._global_ref_line_calculated = True
                finally:
                    with self.lock:
                        self._global_ref_line_calculating = False
                        self._global_ref_line_event.set()
            else:
                if threading.current_thread() is threading.main_thread():
                    self._global_ref_line_event.wait(timeout=0.05)
                else:
                    self._global_ref_line_event.wait()

        if self.session_id is not current_session:
            return None

        is_uncalibrated = (
            self.global_raw_std is not None
            and self.global_raw_std > 500.0
            and not self._global_ref_line_calculated
        )

        res = run_sharpness_pipeline(
            raw_img,
            metric=self.metric,
            ref_line=self.global_ref_line,
            raw_std=self.global_raw_std
        )

        if not is_uncalibrated:
            if self.session_id is current_session:
                # Save derived frames to disk cache
                if current_zarr_manager is not None:
                    current_zarr_manager.set_derived_frame(idx, "denoised_img", res["denoised_img"])
                    current_zarr_manager.set_derived_frame(idx, "masked_img", res["masked_img"])
                    current_zarr_manager.set_derived_frame(idx, "grad_img", res["grad_img"])
                    # Update in-memory metadata
                with self.lock:
                    if self.session_id is current_session:
                        self.centroids[idx] = res["centroid"]
                        self.directions[idx] = res["direction"]
                        self.profiles[idx] = res["1d_profile"]
                        # Populate scores last as the triggering dictionary
                        self.scores[idx] = res["score"]

        if self.session_id is not current_session:
            return None

        self._local.file_list = session_file_list
        return res

    def run_precompute_worker(self, on_progress, on_complete):
        """Spawns background thread worker to evaluate and cache all frames."""
        total = len(self.file_list)
        current_session = self.session_id
        def _worker():
            try:
                for idx in range(total):
                    if self.session_id is not current_session:
                        return
                    data = self.get_frame_pipeline_data(idx)
                    if not data:
                        raise ValueError(f"Frame {idx} data is missing or corrupted.")
                    if self.session_id is not current_session:
                        return
                    self.result_queue.put(lambda c=idx+1: on_progress(c, total))
                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, True, None))
            except Exception as e:
                err_str = str(e)
                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, False, err_str))
        threading.Thread(target=_worker, daemon=True).start()

    def run_export_worker(self, export_dir: str, vmin: float, vmax: float, on_progress, on_complete):
        """Spawns background thread to write Matplotlib figures to disk using Agg backend."""
        total = len(self.file_list)
        current_session = self.session_id
        def _worker():
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                # Map colormap if needed
                matplotlib_cmap = self.colormap
                if self.colormap == "grayscale":
                    matplotlib_cmap = "gray"

                for idx in range(total):
                    if self.session_id is not current_session:
                        return
                    data = self.get_frame_pipeline_data(idx)
                    if not data or data.get("raw_img") is None:
                        import logging
                        logging.warning(f"Frame {idx} data is missing or corrupted. Skipping export.")
                        if self.session_id is not current_session:
                            return
                        self.result_queue.put(lambda c=idx+1: on_progress(c, total))
                        continue
                    
                    fig = Figure(figsize=(10, 8))
                    canvas = FigureCanvasAgg(fig)

                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)

                    # Sliced images for display (only values in [vmin, vmax] are kept, rest zeroed out)
                    raw_disp = np.where((data["raw_img"] >= vmin) & (data["raw_img"] <= vmax), data["raw_img"], 0.0) if data.get("raw_img") is not None else None

                    # Plot Raw
                    ax1.imshow(raw_disp, cmap=matplotlib_cmap, vmin=vmin, vmax=vmax, aspect='auto')
                    ax1.set_title("Raw Image")
                    
                    from rixs_app.core.sharpness import detect_elastic_line_bottom_right
                    line_res = detect_elastic_line_bottom_right(data["raw_img"])
                    
                    if line_res is not None:
                        p_start = line_res['endpoints'][0]
                        p_end = line_res['endpoints'][1]
                        ax1.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r-', linewidth=2)
                    else:
                        dx, dy = data["direction"]
                        if abs(dx) > 1e-5:
                            ax1.axline((data["centroid"][0], data["centroid"][1]), slope=dy/dx, color="red", linestyle="--")
                    ax1.axis("off")

                    # Plot Denoised Image
                    denoised = data.get("denoised_img")
                    if denoised is not None:
                        p99 = np.percentile(denoised, 99.5)
                        if p99 == 0: p99 = 1.0
                        ax2.imshow(denoised, cmap=matplotlib_cmap, vmin=0, vmax=p99, aspect='auto')
                        if line_res is not None:
                            ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'w-', linewidth=2)
                    else:
                        ax2.text(0.5, 0.5, "No Denoised Image", ha='center', va='center')
                    ax2.set_title("Denoised Image")
                    ax2.axis("off")

                    # Plot Masked Gradient
                    masked_img = data.get("masked_img")
                    if masked_img is not None:
                        p99_g = np.percentile(masked_img, 99.9)
                        if p99_g == 0: p99_g = 1.0
                        ax3.imshow(masked_img, cmap=matplotlib_cmap, vmin=0, vmax=p99_g, aspect='auto')
                    else:
                        ax3.text(0.5, 0.5, "No Masked Image", ha='center', va='center')
                    ax3.set_title("Masked Gradient")
                    ax3.axis("off")

                    # Plot 1D Profile + Fit
                    P, u = data["1d_profile"]
                    ax4.plot(u, P, 'k-', linewidth=2, label='1D Profile')
                    
                    ax4.set_title(f"1D Profile (Peak Sharpness: {data['score']:.2f})")
                    ax4.set_xlabel("Perpendicular Distance (u)")
                    ax4.set_ylabel("Gradient Sum")
                    ax4.legend(fontsize=8)

                    fig.tight_layout()
                    save_path = os.path.join(export_dir, f"frame_{idx:03d}_diagnostic.png")
                    fig.savefig(save_path, dpi=150)
                    
                    if self.session_id is not current_session:
                        return
                    self.result_queue.put(lambda c=idx+1: on_progress(c, total))

                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, True, None))
            except Exception as e:
                err_str = str(e)
                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, False, err_str))
        threading.Thread(target=_worker, daemon=True).start()

