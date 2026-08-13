"""Alignment slideshow main view — PySide6 port.

SlideshowView is the controller/view for the alignment slideshow. It:
- Owns a ``SlideshowManager`` (pure Python state/business logic, unchanged)
- Wires all UI panels (NavBar, ControlPanel, ToolsPanel, ClampingPanel,
  ExportPanel, CanvasPanel) together
- Replaces the Tkinter ``queue``-polling loop with a ``QTimer``
- Thread-safety: background workers still post lambdas into
  ``_result_queue``; a ``QTimer`` running at 50 ms pops them on the
  main GUI thread.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel

from rixs_app.ui.alignment_slideshow.alignment_manager import SlideshowManager
from rixs_app.ui.alignment_slideshow.navbar import SlideshowNavBar
from rixs_app.ui.alignment_slideshow.control_panel import SlideshowControlPanel
from rixs_app.ui.alignment_slideshow.tools_panel import SlideshowToolsPanel
from rixs_app.ui.alignment_slideshow.clamping_panel import SlideshowClampingPanel
from rixs_app.ui.alignment_slideshow.export_panel import SlideshowExportPanel
from rixs_app.ui.alignment_slideshow.canvas_panel import SlideshowCanvasPanel
from rixs_app.ui.alignment_slideshow.workers import (
    AutoSnapWorker,
    AutoSnapAllWorker,
    PrecomputeOffsetsWorker,
    ExportSumsWorker,
)
from rixs_app.ui.theme import set_play_btn, set_active_btn, set_tool_btn


class SlideshowView(QWidget):
    """Main controller/view for the alignment slideshow.

    Coordinates all sub-panels, owns the ``SlideshowManager`` state object,
    and dispatches background workers via ``QThreadPool``.

    Args:
        parent: Parent widget.
        on_back_to_sorting: Callback invoked when the user clicks 'Back'.
        on_show_export_comparison: Callback invoked with
            (aligned_sum, direct_sum, initial_dir) to navigate to the
            comparison view.
    """

    def __init__(
        self,
        parent=None,
        *,
        on_back_to_sorting=None,
        on_show_export_comparison=None,
    ):
        """Initialise the slideshow view.

        Args:
            parent: Parent QWidget.
            on_back_to_sorting: Back-navigation callback.
            on_show_export_comparison: Comparison-view navigation callback.
        """
        super().__init__(parent)
        self.on_back_to_sorting = on_back_to_sorting
        self.on_show_export_comparison = on_show_export_comparison

        # Logical state manager (no GUI dependencies)
        self.manager = SlideshowManager()

        # Active workers set (prevents Python GC from destroying workers before signal delivery)
        self._workers: set = set()

        # Debounce timer IDs
        self._pca_debounce_timer: QTimer | None = None
        self._frame_debounce_timer: QTimer | None = None
        self._clamping_debounce_timer: QTimer | None = None
        self._autoplay_timer: QTimer | None = None

        self._build_ui()

    def _run_worker(self, worker) -> None:
        """Keep a strong reference to worker so signals are delivered safely before GC."""
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and lay out all child panels."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 1. Navigation bar
        self.navbar = SlideshowNavBar(self, controller=self)
        outer.addWidget(self.navbar)

        # 2. Control panel (engine settings + frame slider)
        self.control_panel = SlideshowControlPanel(self, controller=self)
        outer.addWidget(self.control_panel)

        # 3. Tools panel
        self.tools_panel = SlideshowToolsPanel(self, controller=self)
        outer.addWidget(self.tools_panel)

        # 4. Metadata label
        self.metadata_label = QLabel(
            "Filename: - | Frame Index: - | Offset: (0.00, 0.00)"
        )
        self.metadata_label.setObjectName("dim_label")
        self.metadata_label.setContentsMargins(8, 0, 8, 0)
        outer.addWidget(self.metadata_label)

        # 5. Bottom bar (clamping + export) — fixed height
        bottom_bar = QFrame()
        bottom_bar.setFixedHeight(46)
        bb_layout = QHBoxLayout(bottom_bar)
        bb_layout.setContentsMargins(4, 2, 4, 2)
        bb_layout.setSpacing(0)

        self.clamping_panel = SlideshowClampingPanel(bottom_bar, controller=self)
        bb_layout.addWidget(self.clamping_panel, stretch=1)

        self.export_panel = SlideshowExportPanel(bottom_bar, controller=self)
        bb_layout.addWidget(self.export_panel)

        outer.addWidget(bottom_bar)

        # 6. Canvas (greedy — fills remaining vertical space)
        self.canvas_panel = SlideshowCanvasPanel(self, controller=self)
        outer.addWidget(self.canvas_panel, stretch=1)

    # ------------------------------------------------------------------
    # Queue polling (replaces Tkinter .after loop)
    # ------------------------------------------------------------------

    def _start_queue_poll(self) -> None:
        """Start a 50 ms QTimer to drain the result queue on the GUI thread."""
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_queue)
        self._poll_timer.start()

    def _poll_queue(self) -> None:
        """Drain all pending callbacks from the worker result queue."""
        try:
            while True:
                callback = self._result_queue.get_nowait()
                callback()
        except Empty:
            pass

    # ------------------------------------------------------------------
    # Show-line switch compatibility shim
    # ------------------------------------------------------------------

    @property
    def show_line_switch(self):
        """Forwards to the PCA panel's show-ref-line checkbox."""
        return self.control_panel.pca_panel.show_line_switch

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start(self, file_list: list[str]) -> None:
        """Initialise the manager and UI for a new set of files.

        Args:
            file_list: Absolute paths to the TIFF images.
        """
        self.manager.start(file_list)

        n = len(self.manager.file_list)
        if n > 1:
            slider = self.control_panel.frame_slider
            slider.setMinimum(0)
            slider.setMaximum(n - 1)
            slider.setValue(0)
            slider.setEnabled(True)
        else:
            self.control_panel.frame_slider.setEnabled(False)

        self.clamping_panel.setup_clamping_limits(
            self.manager.intensity_min, self.manager.intensity_max
        )
        self.clamping_panel.sync_clamping_inputs(
            self.manager.clamping_floor, self.manager.clamping_ceiling
        )
        self.load_and_render()

    def get_current_clamping(self) -> tuple[float, float]:
        """Return the current (floor, ceiling) clamping tuple.

        Returns:
            Tuple of (clamping_floor, clamping_ceiling).
        """
        return self.manager.clamping_floor, self.manager.clamping_ceiling

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def load_and_render(self) -> None:
        """Synchronise UI state and render the current frame to the canvas."""
        if not self.manager.file_list:
            self.canvas_panel.clear()
            self.control_panel.frame_label.setText("Frame: 0/0")
            self.metadata_label.setText(
                "Filename: - | Frame Index: - | Offset: (0.00, 0.00)"
            )
            return

        self.control_panel.sync_timeline_label(
            self.manager.current_idx + 1, len(self.manager.file_list)
        )
        self._sync_slider_to_frame()

        img_path = self.manager.file_list[self.manager.current_idx]
        raw = self.manager.get_raw(img_path)
        rgb = self.manager.get_rgb(img_path, self.manager.colormap)

        if raw is None or rgb is None:
            self.canvas_panel.set_cached_image(None)
            self.canvas_panel.render_error(img_path)
            return

        self.manager.current_raw = raw
        self.manager.current_rgb = rgb

        # Generate per-frame PCA origins if not cached
        if (
            self.manager.current_idx not in self.manager.per_frame_origin
            and self.manager.current_idx not in self.manager.per_frame_manual
        ):
            t = self.manager.get_current_threshold()
            try:
                from rixs_app.core import find_peak_line
                origin, _ = find_peak_line(raw, t)
                self.manager.per_frame_origin[self.manager.current_idx] = origin
            except Exception:
                self.manager.per_frame_origin[self.manager.current_idx] = (
                    self.manager.ref_origin
                )

        dx, dy = 0.0, 0.0
        if self.manager.current_idx > 0 and self.manager.ref_raw is not None:
            dx, dy = self.manager.get_offset(self.manager.current_idx)

        self.metadata_label.setText(
            f"Filename: {os.path.basename(img_path)} "
            f"| Frame Index: {self.manager.current_idx} "
            f"| Offset: ({dx:.2f}, {dy:.2f})"
        )

        disp_rgb = rgb.copy()
        if (
            self.manager.warp_enabled
            and self.manager.current_idx > 0
            and self.manager.ref_raw is not None
        ):
            try:
                from rixs_app.core import warp_image
                disp_rgb = warp_image(rgb, -dx, -dy)
            except Exception:
                pass

        self.canvas_panel.set_cached_image(disp_rgb)
        self._render_display()

    def _render_display(self, *_args) -> None:
        """Push the cached display image and overlay vectors to the canvas."""
        if self.canvas_panel.cached_disp_rgb is None:
            return

        is_warped = (
            self.manager.warp_enabled
            and self.manager.current_idx > 0
            and self.manager.ref_raw is not None
        )
        if is_warped:
            origin = self.manager.global_ref_origin
            direction = self.manager.global_ref_direction
        else:
            origin = self.manager.get_display_origin()
            direction = self.manager.get_display_direction()

        self.canvas_panel.draw_canvas(
            self.canvas_panel.cached_disp_rgb, origin, direction
        )

    def _sync_slider_to_frame(self) -> None:
        """Synchronise threshold and info label to the current frame."""
        t = self.manager.get_current_threshold()
        self.control_panel.sync_pca_elements(t)

        info_parts = []
        if self.manager.current_idx in self.manager.per_frame_threshold:
            val = self.manager.per_frame_threshold[self.manager.current_idx]
            info_parts.append(f"Custom threshold: {val:.4f}%")
        if self.manager.current_idx in self.manager.per_frame_manual:
            info_parts.append("Manual line set")
        self.tools_panel.frame_info_label.setText(" | ".join(info_parts))

    # ------------------------------------------------------------------
    # Event handlers forwarded from panels
    # ------------------------------------------------------------------

    def handle_pca_slider_drag(self, val: float) -> None:
        """Handle live drag of the PCA threshold slider.

        Args:
            val: New threshold value.
        """
        self.manager.pca_threshold = val
        self.control_panel.sync_pca_label_and_entry(val)

        if self._pca_debounce_timer is not None:
            self._pca_debounce_timer.stop()
        self._pca_debounce_timer = QTimer.singleShot(
            80, self._apply_pca_change
        )

    def handle_pca_entry_submit(self, text_val: str) -> None:
        """Handle numeric entry submission for PCA threshold.

        Args:
            text_val: String value typed by the user.
        """
        try:
            val = float(text_val)
            val = max(95.0, min(99.9999, val))
            self.manager.pca_threshold = val
            self.control_panel.sync_pca_elements(val)
            self._apply_pca_change()
        except ValueError:
            pass

    def _apply_pca_change(self) -> None:
        """Apply debounced PCA threshold changes and re-render."""
        self.manager.apply_pca_change()
        self.canvas_panel.clear_photo_cache()
        self._sync_slider_to_frame()
        self.load_and_render()

    def change_engine(self, choice: str) -> None:
        """Switch alignment engine and re-render.

        Also toggles visibility of manual-line buttons (PCA-only feature).

        Args:
            choice: Engine name: 'PCA', 'ECC', or 'Phase Correlation'.
        """
        self.manager.active_engine = choice
        self.control_panel.switch_engine(choice)
        self.tools_panel.show_manual_buttons(choice == "PCA")
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def trigger_auto_snap(self) -> None:
        """Find optimal threshold for the current frame (async)."""
        if self.manager.active_engine in ("ECC", "Phase Correlation"):
            self.manager._invalidate_offset_cache(self.manager.current_idx)
            self.canvas_panel.clear_photo_cache()
            self.load_and_render()
            return

        pca_panel = self.control_panel.pca_panel
        pca_panel.auto_snap_button.setText("...")
        pca_panel.auto_snap_button.setEnabled(False)

        idx = self.manager.current_idx
        if idx >= len(self.manager.file_list):
            return
        raw = self.manager.get_raw(self.manager.file_list[idx])
        if raw is None:
            return

        worker = AutoSnapWorker(raw)
        def _on_result(best_t: float) -> None:
            pca_panel.auto_snap_button.setText("Auto")
            pca_panel.auto_snap_button.setEnabled(True)
            self.manager.set_current_threshold(best_t)
            self.canvas_panel.clear_photo_cache()
            self._sync_slider_to_frame()
            self.load_and_render()

        def _on_error(_err: str) -> None:
            pca_panel.auto_snap_button.setText("Auto")
            pca_panel.auto_snap_button.setEnabled(True)

        worker.signals.result.connect(_on_result)
        worker.signals.error.connect(_on_error)
        self._run_worker(worker)

    def trigger_auto_snap_all(self) -> None:
        """Find optimal threshold for all frames (async)."""
        n_frames = len(self.manager.file_list)
        active_panel = self.control_panel.active_engine_panel

        if self.manager.active_engine in ("ECC", "Phase Correlation"):
            precomp_btn = active_panel.precompute_button
            precomp_btn.setText(f"0/{n_frames}...")
            precomp_btn.setEnabled(False)

            worker = PrecomputeOffsetsWorker(n_frames, self.manager.get_offset)
            def _on_progress(current: int, total: int) -> None:
                precomp_btn.setText(f"{current}/{total}...")
            def _on_result(offsets: dict) -> None:
                precomp_btn.setText("Precompute All")
                precomp_btn.setEnabled(True)
                self.load_and_render()
            def _on_error(_err: str) -> None:
                precomp_btn.setText("Precompute All")
                precomp_btn.setEnabled(True)

            worker.signals.progress.connect(_on_progress)
            worker.signals.result.connect(_on_result)
            worker.signals.error.connect(_on_error)
            self._run_worker(worker)
            return

        auto_all_btn = active_panel.auto_all_button
        auto_snap_btn = active_panel.auto_snap_button
        auto_all_btn.setText(f"0/{n_frames}...")
        auto_all_btn.setEnabled(False)
        auto_snap_btn.setEnabled(False)

        worker = AutoSnapAllWorker(self.manager.file_list, self.manager.get_raw)
        def _on_progress(current: int, total: int) -> None:
            auto_all_btn.setText(f"{current}/{total}...")
        def _on_result(results: dict) -> None:
            auto_all_btn.setText("Auto All")
            auto_all_btn.setEnabled(True)
            auto_snap_btn.setEnabled(True)
            for idx, val in results.items():
                self.manager.per_frame_threshold[idx] = val
            if 0 in self.manager.per_frame_threshold:
                self.manager.ref_threshold = self.manager.per_frame_threshold[0]
            self.manager._load_reference()
            self.manager.offset_cache.clear()
            self.manager.per_frame_origin.clear()
            self.canvas_panel.clear_photo_cache()
            self._sync_slider_to_frame()
            self.load_and_render()
        def _on_error(_err: str) -> None:
            auto_all_btn.setText("Auto All")
            auto_all_btn.setEnabled(True)
            auto_snap_btn.setEnabled(True)

        worker.signals.progress.connect(_on_progress)
        worker.signals.result.connect(_on_result)
        worker.signals.error.connect(_on_error)
        self._run_worker(worker)

    def handle_frame_slider_move(self, val: int) -> None:
        """Handle frame timeline slider movement.

        Args:
            val: New slider integer position (frame index).
        """
        idx = int(val)
        if 0 <= idx < len(self.manager.file_list) and idx != self.manager.current_idx:
            self.manager.current_idx = idx
            QTimer.singleShot(50, self._apply_frame_change)

    def _apply_frame_change(self) -> None:
        """Apply debounced frame change."""
        self.load_and_render()

    def prev_frame(self) -> None:
        """Navigate to the previous frame if available."""
        if self.manager.current_idx > 0:
            self.manager.current_idx -= 1
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(self.manager.current_idx)
            slider.blockSignals(False)
            self.load_and_render()

    def next_frame(self) -> None:
        """Navigate to the next frame if available."""
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(self.manager.current_idx)
            slider.blockSignals(False)
            self.load_and_render()

    def toggle_autoplay(self) -> None:
        """Toggle autoplay between active and paused."""
        if self.manager.autoplay_active:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self) -> None:
        """Start automatic frame cycling."""
        self.manager.autoplay_active = True
        self.navbar.autoplay_button.setText("\u23f8 Pause")
        set_active_btn(self.navbar.autoplay_button)
        self._autoplay_timer = QTimer(self)
        self._autoplay_timer.setInterval(self.manager.autoplay_speed_ms)
        self._autoplay_timer.timeout.connect(self._autoplay_tick)
        self._autoplay_timer.start()

    def stop_autoplay(self) -> None:
        """Stop automatic frame cycling."""
        self.manager.autoplay_active = False
        self.navbar.autoplay_button.setText("\u25ba Play")
        set_play_btn(self.navbar.autoplay_button)
        if self._autoplay_timer is not None:
            self._autoplay_timer.stop()
            self._autoplay_timer = None

    def _autoplay_tick(self) -> None:
        """Advance by one frame in the autoplay loop."""
        if not self.manager.autoplay_active:
            return
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
        else:
            self.manager.current_idx = 0
        slider = self.control_panel.frame_slider
        slider.blockSignals(True)
        slider.setValue(self.manager.current_idx)
        slider.blockSignals(False)
        self.load_and_render()

    def change_colormap(self, val: str) -> None:
        """Change the active colormap and re-render.

        Args:
            val: Colormap name (matplotlib name or 'grayscale').
        """
        self.manager.colormap = val
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def toggle_warp(self) -> None:
        """Toggle dynamic alignment warping on/off."""
        self.manager.warp_enabled = not getattr(self.manager, 'warp_enabled', True)
        if self.manager.warp_enabled:
            self.navbar.warp_button.setText("Warp: ON")
            from rixs_app.ui.theme import set_success_btn
            set_success_btn(self.navbar.warp_button)
        else:
            self.navbar.warp_button.setText("Warp: OFF")
            from rixs_app.ui.theme import set_tool_btn
            set_tool_btn(self.navbar.warp_button)
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def toggle_zoom_mode(self) -> None:
        """Toggle click-to-zoom mode on/off."""
        self.manager.zoom_mode = not getattr(self.manager, 'zoom_mode', False)
        if self.manager.zoom_mode:
            if getattr(self.manager, 'manual_mode', False):
                self.toggle_manual_mode()
            self.tools_panel.zoom_in_button.setText("\U0001f50d Click Zoom...")
            set_active_btn(self.tools_panel.zoom_in_button)
            self.canvas_panel.setCursor(Qt.CrossCursor)
        else:
            self.tools_panel.zoom_in_button.setText("\U0001f50d+ Zoom In")
            set_tool_btn(self.tools_panel.zoom_in_button)
            self.canvas_panel.unsetCursor()

    def toggle_manual_mode(self) -> None:
        """Toggle 2-click manual line alignment mode on/off."""
        self.manager.manual_mode = not getattr(self.manager, 'manual_mode', False)
        if self.manager.manual_mode:
            if getattr(self.manager, 'zoom_mode', False):
                self.toggle_zoom_mode()
            self.tools_panel.manual_line_button.setText("\u270f Click 2 pts...")
            set_active_btn(self.tools_panel.manual_line_button)
            self.manager.manual_clicks.clear()
            self.canvas_panel.setCursor(Qt.CrossCursor)
        else:
            self.tools_panel.manual_line_button.setText("\u270f Manual Line")
            set_tool_btn(self.tools_panel.manual_line_button)
            self.canvas_panel.unsetCursor()

    def clear_manual_line(self) -> None:
        """Clear the manual alignment override for the current frame."""
        self.manager.clear_manual_line()
        if getattr(self.manager, 'manual_mode', False):
            self.toggle_manual_mode()
        self.load_and_render()

    def handle_canvas_click(self, event) -> None:
        """Process mouse click events on the canvas.

        Args:
            event: ``QMouseEvent`` from the canvas panel.
        """
        if getattr(self.manager, 'zoom_mode', False):
            cw = self.canvas_panel.width()
            ch = self.canvas_panel.height()
            scale = self.canvas_panel._lb_scale
            dx = self.canvas_panel._lb_dx
            dy = self.canvas_panel._lb_dy

            ex = event.position().toPoint().x()
            ey = event.position().toPoint().y()

            ix = (ex - dx) / scale
            iy = (ey - dy) / scale

            iw = self.canvas_panel._img_w or 3840
            ih = self.canvas_panel._img_h or 2048

            self.manager.zoom_in_on_point(cw, ch, ix, iy, iw, ih)
            self.tools_panel.sync_zoom_label(
                self.manager.zoom_steps[self.manager.zoom_level]
            )
            self.toggle_zoom_mode()
            self._render_display()
            return

        if not getattr(self.manager, 'manual_mode', False):
            return

        ex = event.position().toPoint().x()
        ey = event.position().toPoint().y()
        self.manager.manual_clicks.append((ex, ey))

        if len(self.manager.manual_clicks) == 1:
            self.canvas_panel.draw_marker(ex, ey)
        elif len(self.manager.manual_clicks) >= 2:
            self.manager.process_manual_click(
                self.manager.manual_clicks,
                self.canvas_panel._lb_dx,
                self.canvas_panel._lb_dy,
                self.canvas_panel._lb_scale,
            )
            self.toggle_manual_mode()
            self.load_and_render()

    def zoom_in(self) -> None:
        """Activate zoom mode if not already at maximum zoom."""
        if self.manager.zoom_level < len(self.manager.zoom_steps) - 1:
            self.toggle_zoom_mode()

    def zoom_out(self) -> None:
        """Decrease canvas zoom level, centred on the current canvas centre."""
        cw = self.canvas_panel.width()
        ch = self.canvas_panel.height()
        iw = self.canvas_panel._img_w or 3840
        ih = self.canvas_panel._img_h or 2048
        self.manager.zoom_out(cw, ch, iw, ih)
        self.tools_panel.sync_zoom_label(
            self.manager.zoom_steps[self.manager.zoom_level]
        )
        self._render_display()

    def reset_view(self) -> None:
        """Reset the canvas to the default unzoomed state."""
        self.manager.reset_view()
        self.tools_panel.sync_zoom_label(1)
        self._render_display()

    def handle_clamping_change(self, floor: float, ceiling: float) -> None:
        """Handle range slider clamping changes.

        Args:
            floor: New lower clamp value.
            ceiling: New upper clamp value.
        """
        self.manager.clamping_floor = floor
        self.manager.clamping_ceiling = ceiling
        self.clamping_panel.sync_clamping_inputs(floor, ceiling)
        QTimer.singleShot(80, self._apply_clamping_change)

    def _apply_clamping_change(self) -> None:
        """Apply debounced clamping change and clear caches."""
        self.manager.rgb_cache.clear()
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def handle_floor_entry_submit(self, val_str: str) -> None:
        """Handle numeric entry for floor clamping limit.

        Args:
            val_str: String entered by the user.
        """
        try:
            val = float(val_str)
            val = max(
                self.manager.intensity_min,
                min(self.manager.clamping_ceiling - 1e-4, val),
            )
            self.manager.clamping_floor = val
            self.clamping_panel.range_slider.set_values(
                self.manager.clamping_floor, self.manager.clamping_ceiling
            )
            self._apply_clamping_change()
        except ValueError:
            self.clamping_panel.floor_entry.setText(
                f"{self.manager.clamping_floor:.4f}"
            )

    def handle_ceiling_entry_submit(self, val_str: str) -> None:
        """Handle numeric entry for ceiling clamping limit.

        Args:
            val_str: String entered by the user.
        """
        try:
            val = float(val_str)
            val = max(
                self.manager.clamping_floor + 1e-4,
                min(self.manager.intensity_max, val),
            )
            self.manager.clamping_ceiling = val
            self.clamping_panel.range_slider.set_values(
                self.manager.clamping_floor, self.manager.clamping_ceiling
            )
            self._apply_clamping_change()
        except ValueError:
            self.clamping_panel.ceiling_entry.setText(
                f"{self.manager.clamping_ceiling:.4f}"
            )

    def trigger_export(self) -> None:
        """Compute alignment offsets and launch the background export worker."""
        self.stop_autoplay()
        self._set_export_ui_state(False)
        self.export_panel.progress_label.setText("Computing offsets...")

        n_frames = len(self.manager.file_list)
        first_file = self.manager.file_list[0]
        initial_dir = os.path.dirname(first_file)

        offset_worker = PrecomputeOffsetsWorker(n_frames, self.manager.get_offset)

        def _on_offset_progress(idx: int, total: int) -> None:
            self.export_panel.progress_label.setText(f"Computing offsets: {idx}/{total}...")

        def _on_offset_result(offsets: dict) -> None:
            ref_raw = self.manager.get_raw(first_file)
            if ref_raw is None:
                self._finish_export(False, "Could not load reference frame.")
                return

            sums_worker = ExportSumsWorker(
                self.manager.file_list, self.manager.get_raw, offsets, ref_raw.shape
            )

            def _on_msg(msg: str) -> None:
                self.export_panel.progress_label.setText(msg)

            def _on_sums_result(result_tuple) -> None:
                self._finish_export(True, result_tuple, initial_dir)

            def _on_error(err: str) -> None:
                self._finish_export(False, err)

            sums_worker.signals.progress_msg.connect(_on_msg)
            sums_worker.signals.result.connect(_on_sums_result)
            sums_worker.signals.error.connect(_on_error)
            self._run_worker(sums_worker)

        def _on_offset_error(err: str) -> None:
            self._finish_export(False, err)

        offset_worker.signals.progress.connect(_on_offset_progress)
        offset_worker.signals.result.connect(_on_offset_result)
        offset_worker.signals.error.connect(_on_offset_error)
        self._run_worker(offset_worker)

    def _finish_export(
        self,
        success: bool,
        result,
        initial_dir: str = "",
    ) -> None:
        """Handle export completion on the GUI thread.

        Args:
            success: True if export succeeded.
            result: (aligned_sum, direct_sum) tuple on success, or error string.
            initial_dir: Default save directory for the comparison view.
        """
        self._set_export_ui_state(True)
        self.export_panel.progress_label.setText("")
        if success:
            aligned_sum, direct_sum = result
            if self.on_show_export_comparison:
                self.on_show_export_comparison(aligned_sum, direct_sum, initial_dir)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, "Export Failed",
                f"An error occurred during export:\n{result}"
            )

    def _set_export_ui_state(self, enabled: bool) -> None:
        """Enable or disable UI elements during an export operation.

        Args:
            enabled: True to enable, False to disable.
        """
        for w in [
            self.navbar.back_button, self.navbar.prev_button,
            self.navbar.next_button, self.navbar.autoplay_button,
            self.tools_panel.manual_line_button, self.tools_panel.clear_manual_button,
            self.tools_panel.zoom_in_button, self.tools_panel.zoom_out_button,
            self.tools_panel.reset_view_button,
            self.navbar.colormap_menu, self.control_panel.warp_button,
            self.export_panel.export_button,
        ]:
            w.setEnabled(enabled)
        self.control_panel.set_ui_state(enabled)

    def back_to_sorting(self) -> None:
        """Navigate back to the sorting view."""
        self.stop_autoplay()
        if self.on_back_to_sorting:
            self.on_back_to_sorting()

    def _teardown_mpl(self) -> None:
        """Stop timers on window close (called by main window closeEvent)."""
        self.stop_autoplay()
        if hasattr(self, '_poll_timer'):
            self._poll_timer.stop()

    # ------------------------------------------------------------------
    # Backwards-compat properties (for existing unit tests)
    # ------------------------------------------------------------------

    @property
    def canvas(self):
        """Alias for canvas_panel (backwards compat)."""
        return self.canvas_panel

    @property
    def current_idx(self) -> int:
        """Current frame index (proxy to manager)."""
        return self.manager.current_idx

    @current_idx.setter
    def current_idx(self, val: int) -> None:
        self.manager.current_idx = val

    @property
    def file_list(self):
        """File list (proxy to manager)."""
        return self.manager.file_list

    @file_list.setter
    def file_list(self, val) -> None:
        self.manager.file_list = val

    @property
    def colormap(self) -> str:
        """Active colormap name."""
        return self.manager.colormap

    @colormap.setter
    def colormap(self, val: str) -> None:
        self.manager.colormap = val

    @property
    def pca_threshold(self) -> float:
        """PCA threshold (proxy to manager)."""
        return self.manager.pca_threshold

    @pca_threshold.setter
    def pca_threshold(self, val: float) -> None:
        self.manager.pca_threshold = val

    @property
    def warp_enabled(self) -> bool:
        """Warp enabled flag (proxy to manager)."""
        return self.manager.warp_enabled

    @warp_enabled.setter
    def warp_enabled(self, val: bool) -> None:
        self.manager.warp_enabled = val

    @property
    def current_rgb(self):
        """Current RGB display array (proxy to manager)."""
        return self.manager.current_rgb

    @property
    def back_button(self):
        """Back button (proxy to navbar)."""
        return self.navbar.back_button

    @property
    def prev_button(self):
        """Prev button (proxy to navbar)."""
        return self.navbar.prev_button

    @property
    def next_button(self):
        """Next button (proxy to navbar)."""
        return self.navbar.next_button

    @property
    def autoplay_button(self):
        """Autoplay button (proxy to navbar)."""
        return self.navbar.autoplay_button

    @property
    def colormap_menu(self):
        """Colormap dropdown (proxy to navbar)."""
        return self.navbar.colormap_menu

    @property
    def warp_switch(self):
        """Warp button (proxy to control_panel warp_button)."""
        return self.control_panel.warp_button

    @property
    def pca_slider(self):
        """PCA slider (proxy to pca_panel)."""
        return self.control_panel.pca_panel.pca_slider

    @property
    def pca_label(self):
        """PCA label (proxy to pca_panel)."""
        return self.control_panel.pca_panel.pca_label

    @property
    def pca_entry(self):
        """PCA entry (proxy to pca_panel)."""
        return self.control_panel.pca_panel.pca_entry

    @property
    def frame_slider(self):
        """Frame slider (proxy to control_panel)."""
        return self.control_panel.frame_slider

    @property
    def manual_line_button(self):
        """Manual line button (proxy to tools_panel)."""
        return self.tools_panel.manual_line_button

    @property
    def clear_manual_button(self):
        """Clear manual button (proxy to tools_panel)."""
        return self.tools_panel.clear_manual_button

    @property
    def zoom_in_button(self):
        """Zoom-in button (proxy to tools_panel)."""
        return self.tools_panel.zoom_in_button

    @property
    def zoom_out_button(self):
        """Zoom-out button (proxy to tools_panel)."""
        return self.tools_panel.zoom_out_button

    @property
    def reset_view_button(self):
        """Reset-view button (proxy to tools_panel)."""
        return self.tools_panel.reset_view_button

    @property
    def zoom_label(self):
        """Zoom label (proxy to tools_panel)."""
        return self.tools_panel.zoom_label

    @property
    def frame_info_label(self):
        """Frame info label (proxy to tools_panel)."""
        return self.tools_panel.frame_info_label

    @property
    def export_button(self):
        """Export button (proxy to export_panel)."""
        return self.export_panel.export_button

    @property
    def progress_label(self):
        """Progress label (proxy to export_panel)."""
        return self.export_panel.progress_label

    @property
    def photo_img(self):
        """The most recently rendered QPixmap (proxy to canvas_panel)."""
        return self.canvas_panel.photo_img

    @property
    def intensity_min(self) -> float:
        """Intensity min (proxy to manager)."""
        return self.manager.intensity_min

    @intensity_min.setter
    def intensity_min(self, val: float) -> None:
        self.manager.intensity_min = val

    @property
    def intensity_max(self) -> float:
        """Intensity max (proxy to manager)."""
        return self.manager.intensity_max

    @intensity_max.setter
    def intensity_max(self, val: float) -> None:
        self.manager.intensity_max = val

    @property
    def clamping_floor(self) -> float:
        """Clamping floor (proxy to manager)."""
        return self.manager.clamping_floor

    @clamping_floor.setter
    def clamping_floor(self, val: float) -> None:
        self.manager.clamping_floor = val

    @property
    def clamping_ceiling(self) -> float:
        """Clamping ceiling (proxy to manager)."""
        return self.manager.clamping_ceiling

    @clamping_ceiling.setter
    def clamping_ceiling(self, val: float) -> None:
        self.manager.clamping_ceiling = val

    @property
    def ref_raw(self):
        """Reference frame raw array (proxy to manager)."""
        return self.manager.ref_raw

    @ref_raw.setter
    def ref_raw(self, val) -> None:
        self.manager.ref_raw = val

    @property
    def ref_origin(self):
        """Reference frame PCA origin (proxy to manager)."""
        return self.manager.ref_origin

    @ref_origin.setter
    def ref_origin(self, val) -> None:
        self.manager.ref_origin = val

    @property
    def ref_direction(self):
        """Reference frame PCA direction (proxy to manager)."""
        return self.manager.ref_direction

    @ref_direction.setter
    def ref_direction(self, val) -> None:
        self.manager.ref_direction = val

    @property
    def ref_threshold(self) -> float:
        """Reference frame threshold (proxy to manager)."""
        return self.manager.ref_threshold

    @ref_threshold.setter
    def ref_threshold(self, val: float) -> None:
        self.manager.ref_threshold = val

    @property
    def per_frame_threshold(self):
        """Per-frame threshold dict (proxy to manager)."""
        return self.manager.per_frame_threshold

    @property
    def per_frame_manual(self):
        """Per-frame manual centroid dict (proxy to manager)."""
        return self.manager.per_frame_manual

    @property
    def per_frame_origin(self):
        """Per-frame PCA origin dict (proxy to manager)."""
        return self.manager.per_frame_origin

    @property
    def current_raw(self):
        """Current raw array (proxy to manager)."""
        return self.manager.current_raw

    @current_raw.setter
    def current_raw(self, val) -> None:
        self.manager.current_raw = val

    @property
    def rgb_cache(self):
        """RGB cache dict (proxy to manager)."""
        return self.manager.rgb_cache

    @rgb_cache.setter
    def rgb_cache(self, val) -> None:
        self.manager.rgb_cache = val

    @property
    def offset_cache(self):
        """Offset cache dict (proxy to manager)."""
        return self.manager.offset_cache

    @offset_cache.setter
    def offset_cache(self, val) -> None:
        self.manager.offset_cache = val

    @property
    def cached_disp_rgb(self):
        """Cached display RGB array (proxy to canvas_panel)."""
        return self.canvas_panel.cached_disp_rgb

    @cached_disp_rgb.setter
    def cached_disp_rgb(self, val) -> None:
        self.canvas_panel.cached_disp_rgb = val

    @property
    def manual_mode(self) -> bool:
        """Manual mode flag (proxy to manager)."""
        return self.manager.manual_mode

    @manual_mode.setter
    def manual_mode(self, val: bool) -> None:
        self.manager.manual_mode = val

    @property
    def manual_clicks(self):
        """Manual clicks list (proxy to manager)."""
        return self.manager.manual_clicks

    @manual_clicks.setter
    def manual_clicks(self, val) -> None:
        self.manager.manual_clicks = val

    @property
    def zoom_level(self) -> int:
        """Zoom level (proxy to manager)."""
        return self.manager.zoom_level

    @zoom_level.setter
    def zoom_level(self, val: int) -> None:
        self.manager.zoom_level = val

    @property
    def zoom_steps(self):
        """Zoom steps list (proxy to manager)."""
        return self.manager.zoom_steps

    @property
    def zoom_mode(self) -> bool:
        """Zoom mode flag (proxy to manager)."""
        return self.manager.zoom_mode

    @zoom_mode.setter
    def zoom_mode(self, val: bool) -> None:
        self.manager.zoom_mode = val

    @property
    def pan_offset_x(self) -> int:
        """Pan x offset (proxy to manager)."""
        return self.manager.pan_offset_x

    @pan_offset_x.setter
    def pan_offset_x(self, val: int) -> None:
        self.manager.pan_offset_x = val

    @property
    def pan_offset_y(self) -> int:
        """Pan y offset (proxy to manager)."""
        return self.manager.pan_offset_y

    @pan_offset_y.setter
    def pan_offset_y(self, val: int) -> None:
        self.manager.pan_offset_y = val

    @property
    def autoplay_active(self) -> bool:
        """Autoplay active flag (proxy to manager)."""
        return self.manager.autoplay_active

    @autoplay_active.setter
    def autoplay_active(self, val: bool) -> None:
        self.manager.autoplay_active = val

    @property
    def autoplay_speed_ms(self) -> int:
        """Autoplay speed in milliseconds (proxy to manager)."""
        return self.manager.autoplay_speed_ms

    @autoplay_speed_ms.setter
    def autoplay_speed_ms(self, val: int) -> None:
        self.manager.autoplay_speed_ms = val

    @property
    def photo_cache(self):
        """Photo/pixmap cache (proxy to canvas_panel)."""
        return self.canvas_panel.photo_cache

    @photo_cache.setter
    def photo_cache(self, val) -> None:
        self.canvas_panel.photo_cache = val

    # Proxy methods
    def on_resize(self, event=None) -> None:
        """Forward resize to canvas panel.

        Args:
            event: Optional resize event (ignored).
        """
        return self.canvas_panel.resizeEvent(event)

    def draw_canvas(self, rgb, origin, direction) -> None:
        """Forward draw to canvas panel.

        Args:
            rgb: RGB numpy array.
            origin: Centroid origin or None.
            direction: Direction vector or None.
        """
        return self.canvas_panel.draw_canvas(rgb, origin, direction)

    def jump_to_frame(self, idx: int) -> None:
        """Jump directly to a given frame index.

        Args:
            idx: Target frame index.
        """
        self.manager.current_idx = idx
        slider = self.control_panel.frame_slider
        slider.blockSignals(True)
        slider.setValue(idx)
        slider.blockSignals(False)
        self.load_and_render()

    def change_pca_threshold(self, val: float) -> None:
        """Set PCA threshold programmatically and re-render.

        Args:
            val: New PCA threshold value.
        """
        self.manager.pca_threshold = val
        self.control_panel.pca_panel.sync_pca_elements(val)
        self.load_and_render()
