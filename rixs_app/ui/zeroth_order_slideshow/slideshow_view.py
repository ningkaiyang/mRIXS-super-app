"""Zeroth-order calibration main view — PySide6 port.

Replaces the Tkinter/CustomTkinter ``ZerothOrderSlideshowView``.
All tkinter.messagebox / filedialog calls are replaced with Qt equivalents.
Timer polling uses QTimer instead of ``self.after``.
"""

from __future__ import annotations

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox,
)

from rixs_app.ui.zeroth_order_slideshow.navbar import ZerothOrderNavBar
from rixs_app.ui.zeroth_order_slideshow.canvas_panel import ZerothOrderCanvasPanel
from rixs_app.ui.zeroth_order_slideshow.control_panel import ZerothOrderControlPanel
from rixs_app.ui.zeroth_order_slideshow.tools_panel import ZerothOrderToolsPanel
from rixs_app.ui.zeroth_order_slideshow.export_panel import ZerothOrderExportPanel
from rixs_app.ui.zeroth_order_slideshow.manager import ZerothOrderManager
from rixs_app.ui.zeroth_order_slideshow.workers import (
    PrecomputeFramesWorker,
    ExportDiagnosticWorker,
)


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
        self.manager = ZerothOrderManager()

        # Active workers set (prevents Python GC from destroying workers before signal delivery)
        self._workers: set = set()

        # State
        self.autoplay_speed_ms: int = 600
        self.zoom_factor: float = 1.0
        self.zoom_center = None
        self.zoom_mode: bool = False
        self._autoplay_timer: QTimer | None = None

        self._slicing_timer = QTimer(self)
        self._slicing_timer.setSingleShot(True)
        self._slicing_timer.timeout.connect(self.load_and_render)

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
    # Session management
    # ------------------------------------------------------------------

    def start(self, file_list: list[str], txt_path: str | None = None) -> None:
        """Start zeroth-order calibration with file list and optional scan log.

        Args:
            file_list: Absolute paths to TIFF images.
            txt_path: Optional path to a scan log TXT file.
        """
        txt_metadata = None
        if txt_path and os.path.exists(txt_path):
            from rixs_app.core.txt_metadata_parser import parse_scan_log, validate_tif_coverage
            try:
                txt_metadata = parse_scan_log(txt_path)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Scan Log Parse Error",
                    f"Failed to parse scan log file '{os.path.basename(txt_path)}':\n{e}\n\n"
                    "Proceeding with frame index ordering."
                )
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
        self.tools_panel.sync_dispersion_input(self.manager.energy_dispersion)
        self.manager.pipeline_stage = "Raw"

        self.load_and_render()
        self._clear_text_focus()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def load_and_render(self) -> None:
        """Fetch pipeline data for the current frame and update all panels."""
        idx = self.manager.current_idx
        total = len(self.manager.file_list)
        if total == 0 or idx >= total:
            return

        filename = os.path.basename(self.manager.file_list[idx])
        self.control_panel.frame_label.setText(f"Frame: {idx + 1}/{total}")
        slider = self.control_panel.frame_slider
        slider.blockSignals(True)
        slider.setValue(idx)
        slider.blockSignals(False)

        # Update metadata summary KPI cards
        motor_name, motor_val = self.manager.get_motor_info(idx)

        data = self.manager.get_frame_pipeline_data(idx)
        if not data:
            self.control_panel.update_metadata(
                filename, motor_name, motor_val,
                is_best_focus=False,
                mono_energy_ev=self.manager.mono_energy_ev,
            )
            if hasattr(self.navbar, "set_best_focus_active"):
                self.navbar.set_best_focus_active(False)
            return

        is_best_focus = self.manager.is_best_focus_frame(idx)

        er = data.get("evaluator_result")
        fwhm_px = er.fwhm_px if (er is not None and er.fwhm_px is not None) else None
        fwhm_mev = (fwhm_px * self.manager.energy_dispersion) if (fwhm_px is not None and self.manager.energy_dispersion > 0) else None
        score = data.get("score")
        r_squared = getattr(er, "r_squared", None) if er is not None else data.get("r_squared")

        self.control_panel.update_metadata(
            filename, motor_name, motor_val,
            fwhm_px=fwhm_px, fwhm_mev=fwhm_mev,
            score=score,
            r_squared=r_squared,
            is_best_focus=is_best_focus,
            mono_energy_ev=self.manager.mono_energy_ev,
        )
        if hasattr(self.navbar, "set_best_focus_active"):
            self.navbar.set_best_focus_active(is_best_focus)

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

        show_pts = self.bottom_bar.show_support_points
        show_extrap = self.bottom_bar.show_extrapolation
        show_line = self.bottom_bar.show_fitted_line

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

    def _clear_text_focus(self) -> None:
        """Clear focus from all text inputs and restore focus to slideshow controller."""
        if hasattr(self, "tools_panel"):
            if hasattr(self.tools_panel, "floor_entry"):
                self.tools_panel.floor_entry.clearFocus()
            if hasattr(self.tools_panel, "ceiling_entry"):
                self.tools_panel.ceiling_entry.clearFocus()
            if hasattr(self.tools_panel, "dispersion_entry"):
                self.tools_panel.dispersion_entry.clearFocus()
        self.setFocus()

    def jump_to_peak_focus(self) -> None:
        """Jump to the frame with the sharpest focus (minimum FWHM)."""
        self._clear_text_focus()
        total = len(self.manager.file_list)
        if total == 0:
            return
        if self.manager.all_frames_cached():
            best_idx = self.manager.get_peak_focus_index()
            self.manager.current_idx = best_idx
            slider = self.control_panel.frame_slider
            slider.blockSignals(True)
            slider.setValue(best_idx)
            slider.blockSignals(False)
            self.load_and_render()
            self._clear_text_focus()
        else:
            def _after_precompute():
                best_idx = self.manager.get_peak_focus_index()
                self.manager.current_idx = best_idx
                slider = self.control_panel.frame_slider
                slider.blockSignals(True)
                slider.setValue(best_idx)
                slider.blockSignals(False)
                self.load_and_render()
                self._clear_text_focus()
            self.trigger_precompute(on_complete_extra=_after_precompute)

    def trigger_precompute(self, on_complete_extra=None, *args, **kwargs) -> None:
        """Run background precomputation for all frames.

        Args:
            on_complete_extra: Optional extra callback once precompute finishes.
        """
        self._clear_text_focus()
        extra_callback = on_complete_extra if callable(on_complete_extra) else None
        btn = self.navbar.precompute_button
        btn.setEnabled(False)
        self.navbar.peak_focus_button.setEnabled(False)
        self._clear_text_focus()

        worker = PrecomputeFramesWorker(self.manager, len(self.manager.file_list))

        def _on_progress(current: int, total: int) -> None:
            btn.setText(f"{current}/{total}...")

        def _on_result(success: bool) -> None:
            btn.setText("Precompute All")
            btn.setEnabled(True)
            self.navbar.peak_focus_button.setEnabled(True)
            self.load_and_render()
            self._clear_text_focus()
            if extra_callback is not None:
                extra_callback()

        def _on_error(err_msg: str) -> None:
            btn.setText("Precompute All")
            btn.setEnabled(True)
            self.navbar.peak_focus_button.setEnabled(True)
            QMessageBox.critical(
                self, "Precompute Error",
                f"An error occurred during precomputation:\n{err_msg}"
            )

        worker.signals.progress.connect(_on_progress)
        worker.signals.result.connect(_on_result)
        worker.signals.error.connect(_on_error)
        self._run_worker(worker)

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
        """Handle range slider slicing change (debounced live updates).

        Args:
            floor: New lower slicing bound.
            ceiling: New upper slicing bound.
        """
        self.manager.slicing_floor = floor
        self.manager.slicing_ceiling = ceiling
        self.tools_panel.sync_slicing_inputs(floor, ceiling)
        self._slicing_timer.start(150)

    def handle_slicing_release(self, floor: float, ceiling: float) -> None:
        """Handle mouse release on range slider (immediate render).

        Args:
            floor: Final lower slicing bound.
            ceiling: Final upper slicing bound.
        """
        self._slicing_timer.stop()
        self.load_and_render()

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
        total = len(self.manager.file_list)
        if hasattr(self.bottom_bar, "progress_bar"):
            self.bottom_bar.progress_bar.setRange(0, total)
            self.bottom_bar.progress_bar.setValue(0)
            self.bottom_bar.progress_bar.setVisible(True)

        worker = ExportDiagnosticWorker(
            self.manager,
            export_dir,
            self.manager.slicing_floor,
            self.manager.slicing_ceiling,
        )

        def _on_progress(current: int, total: int) -> None:
            if hasattr(self.bottom_bar, "progress_bar"):
                self.bottom_bar.progress_bar.setValue(current)
            self.bottom_bar.status_label.setText(f"Exporting: {current}/{total}...")

        def _on_result(success: bool) -> None:
            self.bottom_bar.export_button.setEnabled(True)
            if hasattr(self.bottom_bar, "progress_bar"):
                self.bottom_bar.progress_bar.setVisible(False)
            self.bottom_bar.status_label.setText("")
            QMessageBox.information(
                self, "Export Complete",
                f"Diagnostic multi-plots exported to:\n{export_dir}"
            )

        def _on_error(err_msg: str) -> None:
            self.bottom_bar.export_button.setEnabled(True)
            if hasattr(self.bottom_bar, "progress_bar"):
                self.bottom_bar.progress_bar.setVisible(False)
            self.bottom_bar.status_label.setText("")
            QMessageBox.critical(
                self, "Export Error",
                f"An error occurred during export:\n{err_msg}"
            )

        worker.signals.progress.connect(_on_progress)
        worker.signals.result.connect(_on_result)
        worker.signals.error.connect(_on_error)
        self._run_worker(worker)

    def set_energy_dispersion(self, value: float) -> None:
        """Update energy dispersion and re-render.

        Args:
            value: Energy dispersion in meV/px.
        """
        self.manager.energy_dispersion = max(0.0, float(value))
        if hasattr(self, "tools_panel") and hasattr(self.tools_panel, "sync_dispersion_input"):
            self.tools_panel.sync_dispersion_input(self.manager.energy_dispersion)
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

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Release focus from any text inputs when clicking on view background."""
        focused = self.window().focusWidget() if self.window() else self.focusWidget()
        if focused and isinstance(focused, QLineEdit):
            focused.clearFocus()
        self.setFocus()
        super().mousePressEvent(event)


