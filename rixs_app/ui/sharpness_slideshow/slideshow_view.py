"""Main container view for the sharpness slideshow GUI."""

import queue
import os
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import customtkinter

from rixs_app.ui.sharpness_slideshow.navbar import SharpnessNavBar
from rixs_app.ui.sharpness_slideshow.canvas_panel import SharpnessCanvasPanel
from rixs_app.ui.sharpness_slideshow.control_panel import SharpnessControlPanel
from rixs_app.ui.sharpness_slideshow.tools_panel import SharpnessToolsPanel
from rixs_app.ui.sharpness_slideshow.export_panel import SharpnessExportPanel
from rixs_app.ui.sharpness_slideshow.manager import SharpnessManager

class SharpnessSlideshowView(customtkinter.CTkFrame):
    """Main view class orchestrating modules and callbacks."""

    def __init__(self, parent, on_back_to_sorting=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_back_to_sorting = on_back_to_sorting
        self._result_queue = queue.Queue()
        self.manager = SharpnessManager(self._result_queue)

        # Debouncing and timers
        self._frame_debounce_id = None
        self._clamping_debounce_id = None
        self._autoplay_job = None
        self.autoplay_speed_ms = 600
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.zoom_mode = False

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        self.navbar = SharpnessNavBar(self, controller=self)
        self.navbar.pack(fill="x", pady=5)

        self.control_panel = SharpnessControlPanel(self, controller=self)
        self.control_panel.pack(fill="x", pady=5)

        self.tools_panel = SharpnessToolsPanel(self, controller=self)
        self.tools_panel.pack(fill="x", pady=2)

        # Bottom Bar for exporting
        self.bottom_bar = SharpnessExportPanel(self, controller=self)
        self.bottom_bar.pack(fill="x", side="bottom", pady=5)

        # Left / Right Split plot Canvas (greedy)
        self.canvas_panel = SharpnessCanvasPanel(self, controller=self)
        self.canvas_panel.pack(fill="both", expand=True, pady=5)

    def _poll_queue(self):
        try:
            while True:
                callback = self._result_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def start(self, file_list):
        self.manager.start(file_list)

        # Configure Frame Timeline bounds
        total = len(file_list)
        if total > 1:
            self.control_panel.frame_slider.configure(
                from_=0, to=total - 1, number_of_steps=total - 1, state="normal"
            )
        else:
            self.control_panel.frame_slider.configure(from_=0, to=1, number_of_steps=1, state="disabled")

        self.control_panel.frame_slider.set(0)
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.zoom_mode = False
        self.tools_panel.zoom_in_button.configure(fg_color="#555", text="🔍+ Zoom In")
        self.canvas_panel.canvas.get_tk_widget().configure(cursor="")
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.tools_panel.range_slider.configure_range(self.manager.intensity_min, self.manager.intensity_max)
        self.tools_panel.sync_slicing_inputs(self.manager.slicing_floor, self.manager.slicing_ceiling)

        self.manager.pipeline_stage = getattr(self.manager, "pipeline_stage", "Denoised (D)")
        if self.manager.pipeline_stage not in ["Raw", "Denoised (D)", "Row-Smoothed (Dsm)", "Gradient (G)", "Fitted-Line Strip"]:
            self.manager.pipeline_stage = "Denoised (D)"
        self.control_panel.set_stage_description(self.manager.pipeline_stage)

        self.load_and_render()

    def load_and_render(self):
        idx = self.manager.current_idx
        total = len(self.manager.file_list)
        self.control_panel.sync_timeline_label(idx + 1, total)

        # Fetch pipeline information
        data = self.manager.get_frame_pipeline_data(idx)
        if not data:
            print(f"Warning: Frame data at index {idx} is empty or corrupted.")
            return

        eval_result = data.get("evaluator_result")
        self.control_panel.sync_score_with_evaluator(
            eval_result, data.get("profile_score_fallback", data.get("score"))
        )

        # Determine which 2D matrix to plot
        stage = getattr(self.manager, "pipeline_stage", "Denoised (D)")
        if stage == "Raw":
            img_2d = data["raw_img"]
        elif stage == "Denoised (D)":
            img_2d = data.get("denoised_img", data["raw_img"])
        elif stage == "Row-Smoothed (Dsm)":
            img_2d = data.get("denoised_img", data["raw_img"])
        elif stage == "Gradient (G)":
            img_2d = data.get("grad_img", data["raw_img"])
        elif stage == "Fitted-Line Strip":
            img_2d = data.get("masked_img", data["raw_img"])
        else:
            img_2d = data["raw_img"]

        show_pts = self.tools_panel.show_support_points_var.get()
        show_extrap = self.tools_panel.show_extrapolation_var.get()

        self.canvas_panel.draw_plots(
            img_2d, data.get("1d_profile"), stage,
            self.manager.colormap, self.manager.slicing_floor, self.manager.slicing_ceiling,
            centroid=data.get("centroid"),
            direction=data.get("direction"),
            fit_ok=data.get("fit_ok", False),
            candidates_xy=data.get("candidates_xy"),
            inliers_xy=data.get("inliers_xy"),
            segment_endpoints=data.get("endpoints"),
            detected_support_y_range=data.get("detected_support_y_range"),
            show_support_points=show_pts,
            show_extrapolation=show_extrap,
            evaluator_result=eval_result,
        )

        self.control_panel.sync_detection_status(data)

    # --- Callbacks ---
    def handle_frame_slider_move(self, val):
        idx = int(float(val))
        if 0 <= idx < len(self.manager.file_list) and idx != self.manager.current_idx:
            self.manager.current_idx = idx
            if self._frame_debounce_id is not None:
                self.after_cancel(self._frame_debounce_id)
            self._frame_debounce_id = self.after(50, self._apply_frame_change)

    def _apply_frame_change(self):
        self._frame_debounce_id = None
        self.load_and_render()

    def prev_frame(self):
        if self.manager.current_idx > 0:
            self.manager.current_idx -= 1
            self.control_panel.frame_slider.set(self.manager.current_idx)
            self.load_and_render()

    def next_frame(self):
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
            self.control_panel.frame_slider.set(self.manager.current_idx)
            self.load_and_render()

    def toggle_autoplay(self):
        if self.manager.autoplay_active:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self):
        self.manager.autoplay_active = True
        self.navbar.autoplay_button.configure(text="⏸ Pause", fg_color="#cc5500")
        self._autoplay_tick()

    def stop_autoplay(self):
        self.manager.autoplay_active = False
        self.navbar.autoplay_button.configure(text="▶ Play", fg_color="#2FA572")
        if self._autoplay_job is not None:
            self.after_cancel(self._autoplay_job)
            self._autoplay_job = None

    def _autoplay_tick(self):
        if not self.manager.autoplay_active:
            return
        if self.manager.current_idx >= len(self.manager.file_list) - 1:
            self.manager.current_idx = 0
        else:
            self.manager.current_idx += 1
        self.control_panel.frame_slider.set(self.manager.current_idx)
        self.load_and_render()
        self._autoplay_job = self.after(self.autoplay_speed_ms, self._autoplay_tick)

    def change_colormap(self, val):
        self.manager.colormap = val
        self.load_and_render()

    def change_pipeline_stage(self, val):
        self.manager.pipeline_stage = val
        self.control_panel.set_stage_description(val)
        self.load_and_render()

    def handle_slicing_change(self, floor, ceiling):
        self.manager.slicing_floor = floor
        self.manager.slicing_ceiling = ceiling
        self.tools_panel.sync_slicing_inputs(floor, ceiling)

        if self._clamping_debounce_id is not None:
            self.after_cancel(self._clamping_debounce_id)
        self._clamping_debounce_id = self.after(80, self._apply_slicing_change)

    def _apply_slicing_change(self):
        self._clamping_debounce_id = None
        self.load_and_render()

    def handle_floor_entry_submit(self, val_str):
        try:
            val = float(val_str)
            val = max(self.manager.intensity_min, min(self.manager.slicing_ceiling - 1e-4, val))
            self.manager.slicing_floor = val
            self.tools_panel.range_slider.set_values(val, self.manager.slicing_ceiling)
            self._apply_slicing_change()
        except ValueError:
            pass

    def handle_ceiling_entry_submit(self, val_str):
        try:
            val = float(val_str)
            val = max(self.manager.slicing_floor + 1e-4, min(self.manager.intensity_max, val))
            self.manager.slicing_ceiling = val
            self.tools_panel.range_slider.set_values(self.manager.slicing_floor, val)
            self._apply_slicing_change()
        except ValueError:
            pass

    def toggle_zoom_mode(self):
        self.zoom_mode = not getattr(self, 'zoom_mode', False)
        if self.zoom_mode:
            self.tools_panel.zoom_in_button.configure(fg_color="#cc5500", text="🔍 Click Zoom...")
            self.canvas_panel.canvas.get_tk_widget().configure(cursor="crosshair")
        else:
            self.tools_panel.zoom_in_button.configure(fg_color="#555", text="🔍+ Zoom In")
            self.canvas_panel.canvas.get_tk_widget().configure(cursor="")

    def handle_canvas_click(self, xdata, ydata):
        if getattr(self, 'zoom_mode', False):
            self.zoom_center = (xdata, ydata)
            self.zoom_factor = min(10.0, self.zoom_factor * 1.5)
            self.tools_panel.sync_zoom_label(self.zoom_factor)
            self.toggle_zoom_mode()
            self.load_and_render()

    def zoom_in(self):
        self.toggle_zoom_mode()

    def zoom_out(self):
        self.zoom_factor = max(1.0, self.zoom_factor / 1.5)
        if self.zoom_factor == 1.0:
            self.zoom_center = None
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.load_and_render()

    def reset_view(self):
        self.zoom_factor = 1.0
        self.zoom_center = None
        if getattr(self, 'zoom_mode', False):
            self.toggle_zoom_mode()
        self.tools_panel.sync_zoom_label(self.zoom_factor)
        self.load_and_render()

    def trigger_precompute(self):
        self.navbar.prev_button.configure(state="disabled")
        self.navbar.next_button.configure(state="disabled")
        self.control_panel.precompute_button.configure(state="disabled")

        def on_progress(current, total):
            self.control_panel.precompute_button.configure(text=f"{current}/{total}...")

        def on_complete(success=True, err_msg=None):
            self.navbar.prev_button.configure(state="normal")
            self.navbar.next_button.configure(state="normal")
            self.control_panel.precompute_button.configure(text="Precompute All", state="normal")
            if not success:
                tk.messagebox.showerror(
                    "Precompute Error",
                    f"An error occurred during precomputation:\n{err_msg}",
                    parent=self.winfo_toplevel()
                )
            else:
                self.load_and_render()

        self.manager.run_precompute_worker(on_progress, on_complete)

    def trigger_export(self):
        export_dir = tk.filedialog.askdirectory(parent=self.winfo_toplevel())
        if not export_dir:
            return

        self.bottom_bar.export_button.configure(state="disabled")

        def on_progress(current, total):
            self.bottom_bar.progress_label.configure(text=f"Exporting: {current}/{total}...")

        def on_complete(success=True, err_msg=None):
            self.bottom_bar.export_button.configure(state="normal")
            self.bottom_bar.progress_label.configure(text="")
            if not success:
                tk.messagebox.showerror(
                    "Export Error",
                    f"An error occurred during export:\n{err_msg}",
                    parent=self.winfo_toplevel()
                )
            else:
                tk.messagebox.showinfo(
                    "Export Complete",
                    f"Diagnostic multi-plots exported to:\n{export_dir}",
                    parent=self.winfo_toplevel()
                )

        self.manager.run_export_worker(
            export_dir=export_dir,
            vmin=self.manager.slicing_floor,
            vmax=self.manager.slicing_ceiling,
            on_progress=on_progress,
            on_complete=on_complete
        )

    def back_to_sorting(self):
        self.stop_autoplay()
        if self.on_back_to_sorting:
            self.on_back_to_sorting()
