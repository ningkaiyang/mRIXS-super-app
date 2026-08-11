"""Logical manager driving the state, Zarr caching, and calculations of the zeroth-order calibration slideshow."""

import os
import queue
import threading
import numpy as np
import hashlib
import inspect
from rixs_app.core.dataset import ZarrSequenceManager
from rixs_app.core.zeroth_order import run_zeroth_order_pipeline

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

class ZerothOrderManager:
    """Logical model encapsulating zeroth-order calibration pipeline computations, state caching, and threads."""

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

        # Zeroth-order calibration metadata
        self.txt_metadata = None   # dict from parse_scan_log(), or None
        self.energy_dispersion = 0.0   # meV/px — set from tools panel
        self.mono_energy_ev = 0.0      # eV — set from txt metadata or manual

        # Precomputed/cached values in RAM
        self.scores = {}
        self.centroids = {}
        self.directions = {}
        self.profiles = {}
        self.pipeline_results = {}

        # Slicing configuration tracking
        self._cache_generation = 0
        self._active_config_fingerprint = None

        # Zarr cache
        self.zarr_manager = None
        self.session_id = None

    def start(self, file_list: list[str], txt_metadata=None):
        """Resets the manager state and launches data preloading."""
        manager = ZarrSequenceManager(file_list)
        self.file_list = file_list
        self.current_idx = 0
        self.autoplay_active = False
        self.txt_metadata = txt_metadata
        self.energy_dispersion = 0.0
        self.mono_energy_ev = 0.0

        with self.lock:
            self.zarr_manager = manager
            self.session_id = object()
            self.scores.clear()
            self.centroids.clear()
            self.directions.clear()
            self.profiles.clear()
            self.pipeline_results.clear()
        self._cache_generation = 0
        self._active_config_fingerprint = None
        self._load_reference_bounds()

    def cancel(self):
        """Cancel any running background computation or export worker threads."""
        with self.lock:
            self.session_id = object()

    def _load_reference_bounds(self):
        """Loads first frame to establish display thresholds using 20th/98th percentile."""
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
            return

        self.intensity_min = float(np.min(ref_raw))
        self.intensity_max = float(np.max(ref_raw))

        # Establish default slicing floor (cut background noise) and ceiling (line peak top)
        self.slicing_floor = float(np.percentile(ref_raw, 88))
        self.slicing_ceiling = float(np.percentile(ref_raw, 99.8))
        if self.slicing_ceiling <= self.slicing_floor:
            self.slicing_ceiling = float(self.intensity_max)

    def get_frame_pipeline_data(self, idx: int) -> dict:
        """Retrieves or calculates the zeroth-order pipeline breakdown dictionary for a frame."""
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
            if idx in self.pipeline_results:
                cached_meta = dict(self.pipeline_results[idx])
            else:
                cached_meta = None

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

        if cached_meta is not None and denoised_img is not None and masked_img is not None:
            if self.session_id is not current_session:
                return None
            self._local.file_list = session_file_list
            res = dict(cached_meta)
            res["raw_img"] = raw_img
            res["denoised_img"] = denoised_img
            res["masked_img"] = masked_img
            if grad_img is not None:
                res["grad_img"] = grad_img
            return res

        # 3. Cache Miss: Run processing pipeline
        if self.session_id is not current_session:
            return None

        res = run_zeroth_order_pipeline(
            raw_img,
            metric=self.metric,
            energy_dispersion=self.energy_dispersion,
        )

        if self.session_id is current_session:
            # Save derived frames to disk cache
            if current_zarr_manager is not None:
                current_zarr_manager.set_derived_frame(idx, "denoised_img", res["denoised_img"])
                current_zarr_manager.set_derived_frame(idx, "masked_img", res["masked_img"])
                current_zarr_manager.set_derived_frame(idx, "grad_img", res["grad_img"])
            with self.lock:
                if self.session_id is current_session:
                    self.centroids[idx] = res["centroid"]
                    self.directions[idx] = res["direction"]
                    self.profiles[idx] = res["1d_profile"]
                    self.scores[idx] = res["score"]
                    meta = {k: v for k, v in res.items() if k not in ("raw_img", "denoised_img", "masked_img", "grad_img")}
                    self.pipeline_results[idx] = meta

        if self.session_id is not current_session:
            return None

        self._local.file_list = session_file_list
        return res

    def get_peak_focus_index(self) -> int:
        """Return the index of the frame with the minimum (best) FWHM."""
        best_idx, best_fwhm = 0, float('inf')
        for idx, meta in self.pipeline_results.items():
            er = meta.get('evaluator_result')
            if er is not None and er.score_valid and er.fwhm_px is not None and er.fwhm_px < best_fwhm:
                best_fwhm = er.fwhm_px
                best_idx = idx
        return best_idx

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
        txt_metadata = self.txt_metadata
        energy_dispersion = self.energy_dispersion
        mono_energy_ev = self.mono_energy_ev

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

                    # Sliced images for display
                    raw_disp = np.where((data["raw_img"] >= vmin) & (data["raw_img"] <= vmax), data["raw_img"], 0.0) if data.get("raw_img") is not None else None

                    # Plot Raw
                    ax1.imshow(raw_disp, cmap=matplotlib_cmap, vmin=vmin, vmax=vmax, aspect='auto')
                    ax1.set_title("Raw Image")

                    # Use metadata for line
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
                        if "direction" in data:
                            c = data["centroid"]
                            d = data["direction"]
                            ax2.axline((c[0], c[1]), slope=d[1]/d[0], color="white", linestyle="-")
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
                    ax4.set_title(f"1D Profile (Score: {data['score']:.2f})")
                    ax4.set_xlabel("Perpendicular Distance (u)")
                    ax4.set_ylabel("Gradient Sum")
                    ax4.legend(fontsize=8)

                    fig.tight_layout()
                    save_path = os.path.join(export_dir, f"frame_{idx:03d}_diagnostic.png")
                    fig.savefig(save_path, dpi=150)

                    if self.session_id is not current_session:
                        return
                    self.result_queue.put(lambda c=idx+1: on_progress(c, total))

                # Generate focus curve (always — falls back to frame index when txt_metadata is None)
                self._export_focus_curve(export_dir, txt_metadata, energy_dispersion, mono_energy_ev)

                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, True, None))
            except Exception as e:
                err_str = str(e)
                if self.session_id is not current_session:
                    return
                self.result_queue.put(lambda: _call_on_complete(on_complete, False, err_str))
        threading.Thread(target=_worker, daemon=True).start()

    def _export_focus_curve(
        self,
        export_dir: str,
        txt_metadata: dict | None,
        energy_dispersion: float,
        mono_energy_ev: float,
    ) -> None:
        """Generate focus_curve.png by delegating to the shared CLI utility.

        When *txt_metadata* is available, the X-axis represents the motor
        goal position (e.g. Mirror Pitch in mrad) sourced from the parsed scan
        log.  When *txt_metadata* is ``None``, the X-axis falls back to frame
        index so that a focus curve is always produced when FWHM data exists.

        Args:
            export_dir: Absolute path to the directory where the PNG is written.
            txt_metadata: Parsed scan-log metadata dict from
                ``txt_metadata_parser.parse_scan_log()``, or ``None`` when no
                scan log is available.
            energy_dispersion: Energy dispersion in meV/px for resolving-power
                calculation.  Pass 0.0 to skip resolving-power annotation.
            mono_energy_ev: Monochromator energy in eV.  Pass 0.0 to skip
                resolving-power annotation.
        """
        from rixs_app.core.cli_utils import export_focus_curve

        x_values: list[float] = []
        fwhms: list[float] = []
        resolving_powers: list[float | None] = []

        for idx in range(len(self._file_list)):
            basename = os.path.basename(self._file_list[idx])
            cached = self.pipeline_results.get(idx, {})
            er = cached.get('evaluator_result')
            if er is None or not er.score_valid or er.fwhm_px is None:
                continue

            # Determine X value
            if txt_metadata is not None:
                frame_meta = txt_metadata['frames'].get(basename)
                if frame_meta is None:
                    continue
                x_values.append(frame_meta['motor_goal'])
            else:
                x_values.append(float(idx))

            fwhms.append(er.fwhm_px)

            if energy_dispersion > 0 and mono_energy_ev > 0:
                fwhm_mev = er.fwhm_px * energy_dispersion
                R = mono_energy_ev / (fwhm_mev * 1e-3)
                resolving_powers.append(R)
            else:
                resolving_powers.append(None)

        if len(x_values) < 2:
            return

        x_label = txt_metadata.get('motor_name', 'Motor Pitch') if txt_metadata is not None else 'Frame Index'

        export_focus_curve(
            export_dir=export_dir,
            x_values=x_values,
            fwhms=fwhms,
            x_label=x_label,
            energy_dispersion=energy_dispersion,
            mono_energy_ev=mono_energy_ev,
            resolving_powers=resolving_powers,
        )
