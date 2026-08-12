"""Zeroth-order calibration main view — PySide6 port.

Replaces the Tkinter/CustomTkinter ``ZerothOrderSlideshowView``.
All tkinter.messagebox / filedialog calls are replaced with Qt equivalents.
Timer polling uses QTimer instead of ``self.after``.
"""

from __future__ import annotations

import os
from queue import Queue, Empty

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox,
)

from rixs_app.ui.zeroth_order_slideshow.navbar import ZerothOrderNavBar
from rixs_app.ui.zeroth_order_slideshow.canvas_panel import ZerothOrderCanvasPanel
from rixs_app.ui.zeroth_order_slideshow.control_panel import ZerothOrderControlPanel
from rixs_app.ui.zeroth_order_slideshow.tools_panel import ZerothOrderToolsPanel
from rixs_app.ui.zeroth_order_slideshow.export_panel import ZerothOrderExportPanel
from rixs_app.ui.zeroth_order_slideshow.manager import ZerothOrderManager


class ZerothOrderSlideshowView(QWidget):
    """Main controller/view for the zeroth-order calibration slideshow.

    Manages navigation, pipeline rendering, and UI state for the
    zeroth-order focus & FWHM calibration screen.

    Args:
        parent: Parent widget.
        on_back_to_sorting: Callback to navigate back to the sorting view.
    """

    def __init__(self, parent=None, *, on_back_to_sorting=None):
        """Initialise the zeroth-order slideshow view.

        Args:
            parent: Parent QWidget.
            on_back_to_sorting: Back-navigation callback.
        """
        super().__init__(parent)
        self.on_back_to_sorting = on_back_to_sorting
        self._result_queue: Queue = Queue()
        self.manager = ZerothOrderManager(self._result_queue)

        # State
        self.autoplay_speed_ms: int = 600
        self.zoom_factor: float = 1.0
        self.zoom_center = None
        self.zoom_mode: bool = False
        self._autoplay_timer: QTimer | None = None

        self._build_ui()
        self._start_queue_poll()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and lay out all child panels."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.navbar = ZerothOrderNavBar(self, controller=self)
        outer.addWidget(self.navbar)

        self.control_panel = ZerothOrderControlPanel(self, controller=self)
        outer.addWidget(self.control_panel)

        self.tools_panel = ZerothOrderToolsPanel(self, controller=self)
        outer.addWidget(self.tools_panel)

        self.bottom_bar = ZerothOrderExportPanel(self, controller=self)
        outer.addWidget(self.bottom_bar)

        self.canvas_panel = ZerothOrderCanvasPanel(self, controller=self)
        outer.addWidget(self.canvas_panel, stretch=1)

    # ------------------------------------------------------------------
    # Queue polling (replaces tkinter after loop)
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
    # Session management
    # ------------------------------------------------------------------

    def start(self, file_list: list[str], txt_path: str | None = None) -> None:
        """Start zeroth-order calibration with file list and optional scan log.

        Args:
            file_list: Absolute paths to TIFF images.
            txt_path: Optional path to a scan log TXT file.
        """
        txt_metadata = None
        if txt_path:
            from rixs_app.core.txt_metadata_parser import parse_scan_log, validate_tif_coverage
            try:
                txt_metadata = parse_scan_log(txt_path)
            except (ValueError, FileNotFoundError) as e:
                QMessageBox.warning(self, "Scan Log Error", str(e))
                txt_metadata = None
            if txt_metadata:
                matched, unmatched = validate_tif_coverage(file_list, txt_metadata)
                if unmatched:
                    names = "\n".join(os.path.basename(p) for p in unmatched[:15])
                    extra = f"\n... and {len(unmatched) - 15} more" if len(unmatched) > 15 else ""
                    QMessageBox.warning(
                        self, "Unmatched Frames",
                        f"{len(unmatched)} TIF file(s) not found in scan log:\n{names}{extra}\n\n"
                        "These frames will not have motor pitch data."
                    )

        self.manager.start(file_list, txt_metadata=txt_metadata)

        total = len(file_list)
        slider = self.control_panel.frame_slider
        if total > 1:
            slider.setMinimum(0)
            slider.setMaximum(total - 1)
            slider.setEnabled(True)
        else:
            slider.setMinimum(0)
            slider.setMaximum(1)
            slider.setEnabled(False)
        slider.blockSignals(True)
        slider.setValue(0)
        slider.blockSignals(False)

        self.zoom_factor = 1.0
        self.zoom_center = None
        self.zoom_mode = False
        self.tools_panel.zoom_in_button.setText("\U0001f50d+ Zoom In")
        self.tools_panel.zoom_in_button.setStyleSheet(
            "background-color: #555; color: white;"
        )
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.tools_panel.range_slider.configure_range(
            self.manager.intensity_min, self.manager.intensity_max
        )
        self.tools_panel.sync_slicing_inputs(
            self.manager.slicing_floor, self.manager.slicing_ceiling
        )
        self.manager.pipeline_stage = "Raw"

        self.load_and_render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def load_and_render(self) -> None:
        """Fetch pipeline data for the current frame and update all panels."""
        idx = self.manager.current_idx
        total = len(self.manager.file_list)
        self.control_panel.frame_label.setText(f"Frame: {idx + 1}/{total}")
        slider = self.control_panel.frame_slider
        slider.blockSignals(True)
        slider.setValue(idx)
        slider.blockSignals(False)

        data = self.manager.get_frame_pipeline_data(idx)
        if not data:
            return

        stage = getattr(self.manager, "pipeline_stage", "Raw")
        if stage == "Raw":
            img_2d = data["raw_img"]
        elif stage == "Denoised (D)":
            img_2d = data.get("denoised_img", data["raw_img"])
        elif stage == "Row-Smoothed (Dsm)":
            img_2d = data.get("dsm_img", data.get("denoised_img", data["raw_img"]))
        elif stage == "Gradient (G)":
            img_2d = data.get("grad_img", data["raw_img"])
        elif stage == "Fitted-Line Strip":
            img_2d = data.get("masked_img", data["raw_img"])
        else:
            img_2d = data["raw_img"]

        show_pts = self.tools_panel.show_support_points_var.get()
        show_extrap = self.tools_panel.show_extrapolation_var.get()
        show_line = self.tools_panel.show_fitted_line_var.get()

        self.canvas_panel.draw_plots(
            img_2d, data.get("1d_profile"), stage,
            self.manager.colormap,
            self.manager.slicing_floor, self.manager.slicing_ceiling,
            centroid=data.get("centroid"),
            direction=data.get("direction"),
            fit_ok=data.get("fit_ok", False),
            candidates_xy=data.get("candidates_xy"),
            inliers_xy=data.get("inliers_xy"),
            segment_endpoints=data.get("endpoints"),
            detected_support_y_range=data.get("detected_support_y_range"),
            show_support_points=show_pts,
            show_extrapolation=show_extrap,
            evaluator_result=data.get("evaluator_result"),
            show_fitted_line=show_line,
        )

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def handle_frame_slider_move(self, val: int) -> None:
        """Handle frame slider drag.

        Args:
            val: New slider integer value (frame index).
        """
        idx = int(val)
        if 0 <= idx < len(self.manager.file_list) and idx != self.manager.current_idx:
            self.manager.current_idx = idx
            QTimer.singleShot(50, self.load_and_render)

    def prev_frame(self) -> None:
        """Navigate to the previous frame."""
        if self.manager.current_idx > 0:
            self.manager.current_idx -= 1
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(self.manager.current_idx)
            slider.blockSignals(False)
            self.load_and_render()

    def next_frame(self) -> None:
        """Navigate to the next frame."""
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(self.manager.current_idx)
            slider.blockSignals(False)
            self.load_and_render()

    def jump_to_peak_focus(self) -> None:
        """Jump to the frame with the sharpest focus (minimum FWHM)."""
        total = len(self.manager.file_list)
        if total == 0:
            return
        if len(self.manager.pipeline_results) >= total:
            best_idx = self.manager.get_peak_focus_index()
            self.manager.current_idx = best_idx
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(best_idx)
            slider.blockSignals(False)
            self.load_and_render()
        else:
            def _after_precompute():
                best_idx = self.manager.get_peak_focus_index()
                self.manager.current_idx = best_idx
                slider = self.control_panel.frame_slider
                slider.blockSignals(True)
                slider.setValue(best_idx)
                slider.blockSignals(False)
                self.load_and_render()
            self.trigger_precompute(on_complete_extra=_after_precompute)

    def trigger_precompute(self, on_complete_extra=None) -> None:
        """Run background precomputation for all frames.

        Args:
            on_complete_extra: Optional extra callback once precompute finishes.
        """
        self.navbar.prev_button.setEnabled(False)
        self.navbar.next_button.setEnabled(False)

        def on_progress(current: int, total: int) -> None:
            pass  # Could update a status bar here

        def on_complete(success=True, err_msg=None) -> None:
            self.navbar.prev_button.setEnabled(True)
            self.navbar.next_button.setEnabled(True)
            if not success:
                QMessageBox.critical(
                    self, "Precompute Error",
                    f"An error occurred during precomputation:\n{err_msg}"
                )
            else:
                self.load_and_render()
                if on_complete_extra is not None:
                    on_complete_extra()

        self.manager.run_precompute_worker(on_progress, on_complete)

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
        self.navbar.autoplay_button.setStyleSheet("background-color: #cc5500; color: white;")
        self._autoplay_timer = QTimer(self)
        self._autoplay_timer.setInterval(self.autoplay_speed_ms)
        self._autoplay_timer.timeout.connect(self._autoplay_tick)
        self._autoplay_timer.start()

    def stop_autoplay(self) -> None:
        """Stop automatic frame cycling."""
        self.manager.autoplay_active = False
        self.navbar.autoplay_button.setText("\u25ba Play")
        from rixs_app.ui.theme import PALETTE
        self.navbar.autoplay_button.setStyleSheet(
            f"background-color: {PALETTE['accent_green']}; color: white;"
        )
        if self._autoplay_timer is not None:
            self._autoplay_timer.stop()
            self._autoplay_timer = None

    def _autoplay_tick(self) -> None:
        """Advance to the next frame in the autoplay loop."""
        if not self.manager.autoplay_active:
            return
        if self.manager.current_idx >= len(self.manager.file_list) - 1:
            self.manager.current_idx = 0
        else:
            self.manager.current_idx += 1
        slider = self.control_panel.frame_slider
        slider.blockSignals(True)
        slider.setValue(self.manager.current_idx)
        slider.blockSignals(False)
        self.load_and_render()

    def change_colormap(self, val: str) -> None:
        """Change colormap and re-render.

        Args:
            val: Matplotlib colormap name or 'grayscale'.
        """
        self.manager.colormap = val
        self.load_and_render()

    def change_pipeline_stage(self, val: str) -> None:
        """Change visible pipeline stage and re-render.

        Args:
            val: Stage name ('Raw', 'Denoised (D)', etc.).
        """
        self.manager.pipeline_stage = val
        self.load_and_render()

    def handle_slicing_change(self, floor: float, ceiling: float) -> None:
        """Handle range slider slicing change (debounced).

        Args:
            floor: New lower slicing bound.
            ceiling: New upper slicing bound.
        """
        self.manager.slicing_floor = floor
        self.manager.slicing_ceiling = ceiling
        self.tools_panel.sync_slicing_inputs(floor, ceiling)
        QTimer.singleShot(80, self.load_and_render)

    def handle_floor_entry_submit(self, val_str: str) -> None:
        """Handle return-key submission of the floor slicing entry.

        Args:
            val_str: String value typed by the user.
        """
        try:
            val = float(val_str)
            val = max(self.manager.intensity_min, min(self.manager.slicing_ceiling - 1e-4, val))
            self.manager.slicing_floor = val
            self.tools_panel.range_slider.set_values(val, self.manager.slicing_ceiling)
            self.load_and_render()
        except ValueError:
            pass

    def handle_ceiling_entry_submit(self, val_str: str) -> None:
        """Handle return-key submission of the ceiling slicing entry.

        Args:
            val_str: String value typed by the user.
        """
        try:
            val = float(val_str)
            val = max(self.manager.slicing_floor + 1e-4, min(self.manager.intensity_max, val))
            self.manager.slicing_ceiling = val
            self.tools_panel.range_slider.set_values(self.manager.slicing_floor, val)
            self.load_and_render()
        except ValueError:
            pass

    def toggle_zoom_mode(self) -> None:
        """Toggle click-to-zoom mode."""
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.tools_panel.zoom_in_button.setText("\U0001f50d Click Zoom...")
            self.tools_panel.zoom_in_button.setStyleSheet(
                "background-color: #cc5500; color: white;"
            )
        else:
            self.tools_panel.zoom_in_button.setText("\U0001f50d+ Zoom In")
            self.tools_panel.zoom_in_button.setStyleSheet(
                "background-color: #555; color: white;"
            )

    def handle_canvas_click(self, xdata: float, ydata: float) -> None:
        """Process canvas click (zoom to point).

        Args:
            xdata: Image X coordinate.
            ydata: Image Y coordinate.
        """
        if self.zoom_mode:
            self.zoom_center = (xdata, ydata)
            self.zoom_factor = min(10.0, self.zoom_factor * 1.5)
            self.tools_panel.sync_zoom_label(self.zoom_factor)
            self.toggle_zoom_mode()
            self.load_and_render()

    def zoom_in(self) -> None:
        """Activate zoom mode."""
        self.toggle_zoom_mode()

    def zoom_out(self) -> None:
        """Decrease zoom level."""
        self.zoom_factor = max(1.0, self.zoom_factor / 1.5)
        if self.zoom_factor == 1.0:
            self.zoom_center = None
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.load_and_render()

    def reset_view(self) -> None:
        """Reset zoom to 1×."""
        self.zoom_factor = 1.0
        self.zoom_center = None
        if self.zoom_mode:
            self.toggle_zoom_mode()
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.load_and_render()

    def trigger_export(self) -> None:
        """Open directory picker and run the export worker."""
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not export_dir:
            return

        self.bottom_bar.export_button.setEnabled(False)

        def on_progress(current: int, total: int) -> None:
            self.bottom_bar.status_label.setText(f"Exporting: {current}/{total}...")

        def on_complete(success=True, err_msg=None) -> None:
            self.bottom_bar.export_button.setEnabled(True)
            self.bottom_bar.status_label.setText("")
            if not success:
                QMessageBox.critical(
                    self, "Export Error",
                    f"An error occurred during export:\n{err_msg}"
                )
            else:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Diagnostic multi-plots exported to:\n{export_dir}"
                )

        self.manager.run_export_worker(
            export_dir=export_dir,
            vmin=self.manager.slicing_floor,
            vmax=self.manager.slicing_ceiling,
            on_progress=on_progress,
            on_complete=on_complete
        )

    def set_energy_dispersion(self, value: float) -> None:
        """Update energy dispersion and re-render.

        Args:
            value: Energy dispersion in meV/px.
        """
        self.manager.energy_dispersion = value
        self.load_and_render()

    def back_to_sorting(self) -> None:
        """Navigate back to the sorting view."""
        self.stop_autoplay()
        if self.on_back_to_sorting:
            self.on_back_to_sorting()

    def _teardown_mpl(self) -> None:
        """Stop timers and teardown Matplotlib resources."""
        self.stop_autoplay()
        if hasattr(self, '_poll_timer'):
            self._poll_timer.stop()
        if hasattr(self.manager, 'cancel'):
            self.manager.cancel()
        if hasattr(self, 'canvas_panel') and hasattr(self.canvas_panel, '_teardown_mpl'):
            self.canvas_panel._teardown_mpl()

    # ------------------------------------------------------------------
    # Stub callbacks for buttons in control_panel that don't have matching
    # manager methods in this simplified port
    # ------------------------------------------------------------------

    def handle_motor_entry_submit(self) -> None:
        """Placeholder — motor entry submission (no-op unless manager supports it)."""
        pass

    def add_calibration_point(self) -> None:
        """Placeholder — add calibration point (no-op unless manager supports it)."""
        pass

    def clear_calibration_points(self) -> None:
        """Placeholder — clear calibration points (no-op unless manager supports it)."""
        pass
