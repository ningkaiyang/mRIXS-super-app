# rixs_app/slideshow_view.py

import customtkinter
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import queue
import os
import numpy as np

from rixs_app.ui.slideshow import (
    SlideshowManager,
    SlideshowNavBar,
    SlideshowControlPanel,
    SlideshowToolsPanel,
    SlideshowClampingPanel,
    SlideshowExportPanel,
    SlideshowCanvasPanel
)

class SlideshowView(customtkinter.CTkFrame):
    """Main view component for the slideshow application.
    
    Coordinates the UI panels (NavBar, ControlPanel, ToolsPanel, etc.) and forwards
    user actions to the SlideshowManager. Observes state changes and re-renders the canvas.
    """
    def __init__(self, parent, on_back_to_sorting=None, on_show_export_comparison=None, **kwargs):
        """Initializes the slideshow view and its sub-panels.

        Args:
            parent: The parent tkinter/customtkinter widget.
            on_back_to_sorting (callable, optional): Callback invoked when returning to sorting mode.
            on_show_export_comparison (callable, optional): Callback invoked with
                (aligned_sum, direct_sum, initial_dir) to transition to the
                in-app comparison view.
            **kwargs: Additional keyword arguments passed to CTkFrame.
        """
        super().__init__(parent, **kwargs)
        self.on_back_to_sorting = on_back_to_sorting
        self.on_show_export_comparison = on_show_export_comparison
        
        # Thread-safe result queue and poll scheduler
        self._result_queue = queue.Queue()
        
        # Logical Manager
        self.manager = SlideshowManager(self._result_queue)

        # Debounce tracking
        self._pca_debounce_id = None
        self._frame_debounce_id = None
        self._clamping_debounce_id = None
        self._autoplay_job = None

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        """Constructs and packs all sub-panels within the main view."""
        # 1. Navigation Bar
        self.navbar = SlideshowNavBar(self, controller=self)
        self.navbar.pack(fill="x", pady=5)

        # 2. Control/Sliders Panel
        self.control_panel = SlideshowControlPanel(self, controller=self)
        self.control_panel.pack(fill="x", pady=5)

        # 3. Tools Panel
        self.tools_panel = SlideshowToolsPanel(self, controller=self)
        self.tools_panel.pack(fill="x", pady=2)

        # 4. Metadata Label
        self.metadata_label = customtkinter.CTkLabel(
            self, text="Filename: - | Frame Index: - | Offset: (0.00, 0.00)"
        )
        self.metadata_label.pack(fill="x", pady=2)

        # 5. Bottom Panel (Clamping & Export)
        self.bottom_bar = customtkinter.CTkFrame(self)
        self.bottom_bar.pack(fill="x", side="bottom", pady=5)

        self.clamping_panel = SlideshowClampingPanel(self.bottom_bar, controller=self)
        self.clamping_panel.pack(side="left", fill="x", expand=True, padx=10)

        self.export_panel = SlideshowExportPanel(self.bottom_bar, controller=self)
        self.export_panel.pack(side="right", padx=10)

        # 6. Canvas (Greedy widget, packed last)
        self.canvas_panel = SlideshowCanvasPanel(self, controller=self)
        self.canvas_panel.pack(fill="both", expand=True, pady=5)

    def _poll_queue(self):
        """Periodically polls the thread-safe queue for callbacks from background threads."""
        try:
            while True:
                callback = self._result_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def start(self, file_list):
        """Initializes the view and manager with a new list of files.

        Args:
            file_list (list[str]): List of absolute file paths to load.
        """
        self.manager.start(file_list)

        # Setup frame slider range
        if len(self.manager.file_list) > 1:
            self.control_panel.frame_slider.configure(
                from_=0, to=len(self.manager.file_list) - 1,
                number_of_steps=len(self.manager.file_list) - 1, state="normal"
            )
        else:
            self.control_panel.frame_slider.configure(from_=0, to=1, number_of_steps=1, state="disabled")
        self.control_panel.frame_slider.set(0)

        # Update Clamping Sliders Limits
        self.clamping_panel.setup_clamping_limits(self.manager.intensity_min, self.manager.intensity_max)
        self.clamping_panel.sync_clamping_inputs(self.manager.clamping_floor, self.manager.clamping_ceiling)

        self.load_and_render()

    def get_current_clamping(self):
        """Retrieves the current intensity clamping levels.

        Returns:
            tuple[float, float]: The current floor and ceiling clamping values.
        """
        return self.manager.clamping_floor, self.manager.clamping_ceiling

    def load_and_render(self):
        """Synchronizes UI states and renders the current frame to the canvas.

        Fetches active frame data, determines offsets, constructs the image for display,
        and triggers a redraw operation.
        """
        if not self.manager.file_list:
            self.canvas_panel.clear()
            self.control_panel.frame_label.configure(text="Frame: 0/0")
            self.metadata_label.configure(text="Filename: - | Frame Index: - | Offset: (0.00, 0.00)")
            return

        # 1. Sync sliders & label values
        self.control_panel.sync_timeline_label(self.manager.current_idx + 1, len(self.manager.file_list))
        self._sync_slider_to_frame()

        # 2. Extract active filepath and perform loads
        img_path = self.manager.file_list[self.manager.current_idx]
        raw = self.manager.get_raw(img_path)
        rgb = self.manager.get_rgb(img_path, self.manager.colormap)

        if raw is None or rgb is None:
            self.canvas_panel.set_cached_image(None)
            self.canvas_panel.render_error(img_path)
            return

        # 3. Store reference arrays locally or in manager
        self.manager.current_raw = raw
        self.manager.current_rgb = rgb

        # 4. Generate per-frame PCA origins if uncached and not manual override
        if (self.manager.current_idx not in self.manager.per_frame_origin and
            self.manager.current_idx not in self.manager.per_frame_manual):
            t = self.manager.get_current_threshold()
            try:
                from rixs_app.core import find_peak_line
                origin, _ = find_peak_line(raw, t)
                self.manager.per_frame_origin[self.manager.current_idx] = origin
            except Exception:
                self.manager.per_frame_origin[self.manager.current_idx] = self.manager.ref_origin

        # 5. Extract displacement offsets
        dx, dy = 0.0, 0.0
        if self.manager.current_idx > 0 and self.manager.ref_raw is not None:
            dx, dy = self.manager.get_offset(self.manager.current_idx)

        # 6. Push Metadata
        self.metadata_label.configure(
            text=f"Filename: {os.path.basename(img_path)} | Frame Index: {self.manager.current_idx} | Offset: ({dx:.2f}, {dy:.2f})"
        )

        # 7. Dynamic alignment warping
        disp_rgb = rgb.copy()
        if self.manager.warp_enabled and self.manager.current_idx > 0 and self.manager.ref_raw is not None:
            try:
                from rixs_app.core import warp_image
                disp_rgb = warp_image(rgb, -dx, -dy)
            except Exception:
                pass

        self.canvas_panel.set_cached_image(disp_rgb)
        self._render_display()

    def _render_display(self):
        """Pushes the cached display image and vectors to the canvas panel."""
        if self.canvas_panel.cached_disp_rgb is None:
            return
        
        is_warped = self.manager.warp_enabled and self.manager.current_idx > 0 and self.manager.ref_raw is not None
        
        if is_warped:
            # If warped, the image has been transformed so its feature sits at the global reference
            origin = self.manager.global_ref_origin
            direction = self.manager.global_ref_direction
        else:
            # If not warped, or it's the first frame, use the unwarped coordinates
            origin = self.manager.get_display_origin()
            direction = self.manager.get_display_direction()
            
        # Ensure we have valid vectors before drawing
        if origin is None or direction is None:
            origin = None
            direction = None

        self.canvas_panel.draw_canvas(self.canvas_panel.cached_disp_rgb, origin, direction)

    def _sync_slider_to_frame(self):
        """Synchronizes threshold and tools panel information to match the current frame state."""
        t = self.manager.get_current_threshold()
        self.control_panel.sync_pca_elements(t)

        info_parts = []
        if self.manager.current_idx in self.manager.per_frame_threshold:
            info_parts.append(f"Custom threshold: {self.manager.per_frame_threshold[self.manager.current_idx]:.4f}%")
        if self.manager.current_idx in self.manager.per_frame_manual:
            info_parts.append("Manual line set")
        self.tools_panel.frame_info_label.configure(text=" | ".join(info_parts))

    # --- Event Handlers & Forwarding Calls ---
    def handle_pca_slider_drag(self, val):
        """Handles user interaction with the PCA threshold slider.

        Args:
            val (float): The updated slider value.
        """
        self.manager.pca_threshold = float(val)
        self.control_panel.sync_pca_label_and_entry(self.manager.pca_threshold)

        if self._pca_debounce_id is not None:
            self.after_cancel(self._pca_debounce_id)
        self._pca_debounce_id = self.after(80, self._apply_pca_change)

    def handle_pca_entry_submit(self, text_val):
        """Handles user submission of a manual PCA threshold via the entry box.

        Args:
            text_val (str): The string value entered by the user.
        """
        try:
            val = float(text_val)
            val = max(95.0, min(99.9999, val))
            self.manager.pca_threshold = val
            self.control_panel.sync_pca_elements(val)
            self._apply_pca_change()
        except ValueError:
            pass

    def _apply_pca_change(self):
        """Applies debounced PCA threshold changes."""
        self._pca_debounce_id = None
        self.manager.apply_pca_change()
        self.canvas_panel.clear_photo_cache()
        self._sync_slider_to_frame()
        self.load_and_render()

    def change_engine(self, choice):
        self.manager.active_engine = choice
        self.control_panel.switch_engine(choice)
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def trigger_auto_snap(self):
        """Triggers the auto-snap operation to find the best threshold for the current frame."""
        if self.manager.active_engine in ("ECC", "Phase Correlation"):
            self.manager._invalidate_offset_cache(self.manager.current_idx)
            self.canvas_panel.clear_photo_cache()
            self.load_and_render()
            return

        self.control_panel.active_engine_panel.auto_snap_button.configure(text="...", state="disabled")
        def on_complete(best_t):
            self.control_panel.active_engine_panel.auto_snap_button.configure(text="Auto", state="normal")
            self.manager.set_current_threshold(best_t)
            self.canvas_panel.clear_photo_cache()
            self._sync_slider_to_frame()
            self.load_and_render()
        self.manager.run_auto_snap(on_complete)

    def trigger_auto_snap_all(self):
        """Triggers the auto-snap operation for all loaded frames asynchronously."""
        n_frames = len(self.manager.file_list)
        
        if self.manager.active_engine in ("ECC", "Phase Correlation"):
            self.control_panel.active_engine_panel.precompute_button.configure(text=f"0/{n_frames}...", state="disabled")
            def _worker():
                for idx in range(n_frames):
                    self.manager.get_offset(idx)
                    self.manager.result_queue.put(lambda current=idx+1, total=n_frames: self.control_panel.active_engine_panel.precompute_button.configure(text=f"{current}/{total}..."))
                self.manager.result_queue.put(lambda: self.control_panel.active_engine_panel.precompute_button.configure(text="Precompute All", state="normal"))
                self.manager.result_queue.put(self.load_and_render)
            import threading
            threading.Thread(target=_worker, daemon=True).start()
            return

        self.control_panel.active_engine_panel.auto_all_button.configure(text=f"0/{n_frames}...", state="disabled")
        self.control_panel.active_engine_panel.auto_snap_button.configure(state="disabled")

        def on_progress(current, total):
            self.control_panel.active_engine_panel.auto_all_button.configure(text=f"{current}/{total}...")

        def on_complete(results):
            self.control_panel.active_engine_panel.auto_all_button.configure(text="Auto All", state="normal")
            self.control_panel.active_engine_panel.auto_snap_button.configure(state="normal")
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

        self.manager.run_auto_snap_all(on_progress, on_complete)

    def handle_frame_slider_move(self, val):
        """Handles user interaction with the frame timeline slider.

        Args:
            val (float): The selected frame index.
        """
        idx = int(float(val))
        if 0 <= idx < len(self.manager.file_list) and idx != self.manager.current_idx:
            self.manager.current_idx = idx
            if self._frame_debounce_id is not None:
                self.after_cancel(self._frame_debounce_id)
            self._frame_debounce_id = self.after(50, self._apply_frame_change)

    def _apply_frame_change(self):
        """Applies a debounced frame change and triggers a re-render."""
        self._frame_debounce_id = None
        self.load_and_render()

    def prev_frame(self):
        """Navigates to the previous frame, if available."""
        if self.manager.current_idx > 0:
            self.manager.current_idx -= 1
            self.control_panel.frame_slider.set(self.manager.current_idx)
            self.load_and_render()

    def next_frame(self):
        """Navigates to the next frame, if available."""
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
            self.control_panel.frame_slider.set(self.manager.current_idx)
            self.load_and_render()

    def toggle_autoplay(self):
        """Toggles the autoplay state between active and paused."""
        if self.manager.autoplay_active:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self):
        """Starts the automatic playback of frames."""
        self.manager.autoplay_active = True
        self.navbar.autoplay_button.configure(text="⏸ Pause", fg_color="#cc5500")
        self._autoplay_tick()

    def stop_autoplay(self):
        """Stops the automatic playback of frames."""
        self.manager.autoplay_active = False
        self.navbar.autoplay_button.configure(text="▶ Play", fg_color="#2FA572")
        if self._autoplay_job is not None:
            self.after_cancel(self._autoplay_job)
            self._autoplay_job = None

    def _autoplay_tick(self):
        """Executes a single step of the autoplay loop and schedules the next tick."""
        if not self.manager.autoplay_active:
            return
        if self.manager.current_idx < len(self.manager.file_list) - 1:
            self.manager.current_idx += 1
        else:
            self.manager.current_idx = 0
        self.control_panel.frame_slider.set(self.manager.current_idx)
        self.load_and_render()
        self._autoplay_job = self.after(self.manager.autoplay_speed_ms, self._autoplay_tick)

    def change_colormap(self, val):
        """Updates the active colormap and re-renders the display.

        Args:
            val (str): The new colormap name.
        """
        self.manager.colormap = val
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def toggle_warp(self):
        """Toggles the dynamic alignment warping functionality."""
        self.manager.warp_enabled = bool(self.navbar.warp_switch.get())
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def toggle_zoom_mode(self):
        """Toggles the interactive click-to-zoom mode."""
        self.manager.zoom_mode = not getattr(self.manager, 'zoom_mode', False)
        if self.manager.zoom_mode:
            if getattr(self.manager, 'manual_mode', False):
                self.toggle_manual_mode()
            self.tools_panel.zoom_in_button.configure(fg_color="#cc5500", text="🔍 Click Zoom...")
            self.canvas_panel.configure(cursor="crosshair")
        else:
            self.tools_panel.zoom_in_button.configure(fg_color="#555", text="🔍+ Zoom In")
            self.canvas_panel.configure(cursor="")

    def toggle_manual_mode(self):
        """Toggles the manual 2-point line alignment mode."""
        self.manager.manual_mode = not getattr(self.manager, 'manual_mode', False)
        if self.manager.manual_mode:
            if getattr(self.manager, 'zoom_mode', False):
                self.toggle_zoom_mode()
            self.tools_panel.manual_line_button.configure(fg_color="#cc5500", text="✏ Click 2 pts...")
            self.manager.manual_clicks.clear()
            self.canvas_panel.configure(cursor="crosshair")
        else:
            self.tools_panel.manual_line_button.configure(fg_color="#555", text="✏ Manual Line")
            self.canvas_panel.configure(cursor="")

    def clear_manual_line(self):
        """Clears any user-defined manual alignment for the current frame."""
        self.manager.clear_manual_line()
        if getattr(self.manager, 'manual_mode', False):
            self.toggle_manual_mode()
        self.load_and_render()

    def handle_canvas_click(self, event):
        """Processes mouse click events on the canvas during interactive modes."""
        if getattr(self.manager, 'zoom_mode', False):
            cw = self.canvas_panel.winfo_width()
            ch = self.canvas_panel.winfo_height()
            scale = self.canvas_panel._lb_scale
            dx = self.canvas_panel._lb_dx
            dy = self.canvas_panel._lb_dy
            
            ix = (event.x - dx) / scale
            iy = (event.y - dy) / scale
            
            iw = self.canvas_panel._img_w or 3840
            ih = self.canvas_panel._img_h or 2048
            
            self.manager.zoom_in_on_point(cw, ch, ix, iy, iw, ih)
            self.tools_panel.sync_zoom_label(self.manager.zoom_steps[self.manager.zoom_level])
            self.toggle_zoom_mode()
            self._render_display()
            return

        if not getattr(self.manager, 'manual_mode', False):
            return
        self.manager.manual_clicks.append((event.x, event.y))
        
        if len(self.manager.manual_clicks) == 1:
            self.canvas_panel.draw_marker(event.x, event.y)
        elif len(self.manager.manual_clicks) >= 2:
            self.manager.process_manual_click(
                self.manager.manual_clicks,
                self.canvas_panel._lb_dx,
                self.canvas_panel._lb_dy,
                self.canvas_panel._lb_scale
            )
            self.toggle_manual_mode()
            self.load_and_render()

    def zoom_in(self):
        """Activates zoom mode if we aren't at max zoom."""
        if self.manager.zoom_level < len(self.manager.zoom_steps) - 1:
            self.toggle_zoom_mode()

    def zoom_out(self):
        """Decreases the canvas zoom level, centered on the current canvas center."""
        cw = self.canvas_panel.winfo_width()
        ch = self.canvas_panel.winfo_height()
        iw = self.canvas_panel._img_w or 3840
        ih = self.canvas_panel._img_h or 2048
        self.manager.zoom_out(cw, ch, iw, ih)
        self.tools_panel.sync_zoom_label(self.manager.zoom_steps[self.manager.zoom_level])
        self._render_display()

    def reset_view(self):
        """Resets the canvas view to its default unzoomed state."""
        self.manager.reset_view()
        self.tools_panel.sync_zoom_label(1)
        self._render_display()

    def handle_clamping_change(self, floor, ceiling):
        """Handles changes made to the display clamping levels.

        Args:
            floor (float): The new lower clamp bound.
            ceiling (float): The new upper clamp bound.
        """
        self.manager.clamping_floor = floor
        self.manager.clamping_ceiling = ceiling
        self.clamping_panel.sync_clamping_inputs(floor, ceiling)

        if self._clamping_debounce_id is not None:
            self.after_cancel(self._clamping_debounce_id)
        self._clamping_debounce_id = self.after(80, self._apply_clamping_change)

    def _apply_clamping_change(self):
        """Applies debounced clamping changes and clears relevant image caches."""
        self._clamping_debounce_id = None
        self.manager.rgb_cache.clear()
        self.canvas_panel.clear_photo_cache()
        self.load_and_render()

    def handle_floor_entry_submit(self, val_str):
        """Handles user submission of a new floor clamping limit via entry box.

        Args:
            val_str (str): The text entered by the user.
        """
        try:
            val = float(val_str)
            val = max(self.manager.intensity_min, min(self.manager.clamping_ceiling - 1e-4, val))
            self.manager.clamping_floor = val
            self.clamping_panel.range_slider.set_values(self.manager.clamping_floor, self.manager.clamping_ceiling)
            self._apply_clamping_change()
        except ValueError:
            self.clamping_panel.floor_entry.delete(0, "end")
            self.clamping_panel.floor_entry.insert(0, f"{self.manager.clamping_floor:.4f}")

    def handle_ceiling_entry_submit(self, val_str):
        """Handles user submission of a new ceiling clamping limit via entry box.

        Args:
            val_str (str): The text entered by the user.
        """
        try:
            val = float(val_str)
            val = max(self.manager.clamping_floor + 1e-4, min(self.manager.intensity_max, val))
            self.manager.clamping_ceiling = val
            self.clamping_panel.range_slider.set_values(self.manager.clamping_floor, self.manager.clamping_ceiling)
            self._apply_clamping_change()
        except ValueError:
            self.clamping_panel.ceiling_entry.delete(0, "end")
            self.clamping_panel.ceiling_entry.insert(0, f"{self.manager.clamping_ceiling:.4f}")

    def trigger_export(self):
        """Initiates the process to compute final offsets and export the aligned sum."""
        self.stop_autoplay()
        self.export_panel.progress_label.configure(text="Computing offsets...")
        self.update_idletasks()

        def _progress_offsets(idx, total):
            self.export_panel.progress_label.configure(text=f"Computing offsets: {idx}/{total}...")
            self.update_idletasks()

        offsets = self.manager.compute_all_offsets_for_export(_progress_offsets)
        self.export_panel.progress_label.configure(text="")

        first_file = self.manager.file_list[0]
        initial_dir = os.path.dirname(first_file)

        self._set_export_ui_state("disabled")

        def on_progress(msg):
            self.export_panel.progress_label.configure(text=msg)

        def on_complete(success, result):
            self._set_export_ui_state("normal")
            self.export_panel.progress_label.configure(text="")
            if success:
                aligned_sum, direct_sum = result
                if self.on_show_export_comparison:
                    self.on_show_export_comparison(aligned_sum, direct_sum, initial_dir)
            else:
                tk.messagebox.showerror("Export Failed", f"An error occurred during export:\n{result}")

        self.manager.compute_both_sums(offsets, on_progress, on_complete)

    def _set_export_ui_state(self, state):
        """Sets the active state of UI elements during an export operation.

        Args:
            state (str): 'normal' or 'disabled'.
        """
        widgets = [
            self.navbar.back_button, self.navbar.prev_button, self.navbar.next_button, self.navbar.autoplay_button,
            self.tools_panel.manual_line_button, self.tools_panel.clear_manual_button,
            self.tools_panel.zoom_in_button, self.tools_panel.zoom_out_button, self.tools_panel.reset_view_button,
            self.navbar.colormap_menu, self.navbar.warp_switch,
            self.export_panel.export_button
        ]
        for w in widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass
        self.control_panel.set_ui_state(state)

    def back_to_sorting(self):
        """Handles the user request to navigate back to the sorting view."""
        self.stop_autoplay()
        if self.on_back_to_sorting:
            self.on_back_to_sorting()

    # --- Backwards Compatibility Properties & Methods (for E2E Test Compatibility) ---
    @property
    def frame_label(self):
        return self.control_panel.frame_label

    @property
    def canvas(self):
        return self.canvas_panel

    @property
    def current_idx(self):
        return self.manager.current_idx

    @current_idx.setter
    def current_idx(self, val):
        self.manager.current_idx = val

    @property
    def file_list(self):
        return self.manager.file_list

    @file_list.setter
    def file_list(self, val):
        self.manager.file_list = val

    @property
    def colormap(self):
        return self.manager.colormap

    @colormap.setter
    def colormap(self, val):
        self.manager.colormap = val

    @property
    def pca_threshold(self):
        return self.manager.pca_threshold

    @pca_threshold.setter
    def pca_threshold(self, val):
        self.manager.pca_threshold = val

    @property
    def warp_enabled(self):
        return self.manager.warp_enabled

    @warp_enabled.setter
    def warp_enabled(self, val):
        self.manager.warp_enabled = val

    @property
    def current_rgb(self):
        return self.manager.current_rgb

    @property
    def back_button(self):
        return self.navbar.back_button

    @property
    def prev_button(self):
        return self.navbar.prev_button

    @property
    def next_button(self):
        return self.navbar.next_button

    @property
    def autoplay_button(self):
        return self.navbar.autoplay_button

    @property
    def colormap_menu(self):
        return self.navbar.colormap_menu

    @property
    def warp_switch(self):
        return self.navbar.warp_switch

    @property
    def show_line_switch(self):
        return self.control_panel.pca_panel.show_line_switch

    @property
    def pca_slider(self):
        return self.control_panel.pca_panel.pca_slider

    @property
    def pca_label(self):
        return self.control_panel.pca_panel.pca_label

    @property
    def pca_entry(self):
        return self.control_panel.pca_panel.pca_entry

    @property
    def auto_snap_button(self):
        return self.control_panel.active_engine_panel.auto_snap_button

    @property
    def auto_all_button(self):
        return self.control_panel.active_engine_panel.auto_all_button

    @property
    def frame_slider(self):
        return self.control_panel.frame_slider

    @property
    def manual_line_button(self):
        return self.tools_panel.manual_line_button

    @property
    def clear_manual_button(self):
        return self.tools_panel.clear_manual_button

    @property
    def zoom_in_button(self):
        return self.tools_panel.zoom_in_button

    @property
    def zoom_out_button(self):
        return self.tools_panel.zoom_out_button

    @property
    def reset_view_button(self):
        return self.tools_panel.reset_view_button

    @property
    def zoom_label(self):
        return self.tools_panel.zoom_label

    @property
    def frame_info_label(self):
        return self.tools_panel.frame_info_label

    @property
    def export_button(self):
        return self.export_panel.export_button

    @property
    def progress_label(self):
        return self.export_panel.progress_label

    @property
    def photo_img(self):
        return self.canvas_panel.photo_img

    # Manager / State properties forwarders
    @property
    def intensity_min(self):
        return self.manager.intensity_min
    @intensity_min.setter
    def intensity_min(self, val):
        self.manager.intensity_min = val

    @property
    def intensity_max(self):
        return self.manager.intensity_max
    @intensity_max.setter
    def intensity_max(self, val):
        self.manager.intensity_max = val

    @property
    def clamping_floor(self):
        return self.manager.clamping_floor
    @clamping_floor.setter
    def clamping_floor(self, val):
        self.manager.clamping_floor = val

    @property
    def clamping_ceiling(self):
        return self.manager.clamping_ceiling
    @clamping_ceiling.setter
    def clamping_ceiling(self, val):
        self.manager.clamping_ceiling = val

    @property
    def ref_raw(self):
        return self.manager.ref_raw
    @ref_raw.setter
    def ref_raw(self, val):
        self.manager.ref_raw = val

    @property
    def ref_origin(self):
        return self.manager.ref_origin
    @ref_origin.setter
    def ref_origin(self, val):
        self.manager.ref_origin = val

    @property
    def ref_direction(self):
        return self.manager.ref_direction
    @ref_direction.setter
    def ref_direction(self, val):
        self.manager.ref_direction = val

    @property
    def ref_threshold(self):
        return self.manager.ref_threshold
    @ref_threshold.setter
    def ref_threshold(self, val):
        self.manager.ref_threshold = val

    @property
    def per_frame_threshold(self):
        return self.manager.per_frame_threshold
    @per_frame_threshold.setter
    def per_frame_threshold(self, val):
        self.manager.per_frame_threshold = val

    @property
    def per_frame_manual(self):
        return self.manager.per_frame_manual
    @per_frame_manual.setter
    def per_frame_manual(self, val):
        self.manager.per_frame_manual = val

    @property
    def per_frame_origin(self):
        return self.manager.per_frame_origin
    @per_frame_origin.setter
    def per_frame_origin(self, val):
        self.manager.per_frame_origin = val

    @property
    def current_raw(self):
        return self.manager.current_raw
    @current_raw.setter
    def current_raw(self, val):
        self.manager.current_raw = val


    @property
    def rgb_cache(self):
        return self.manager.rgb_cache
    @rgb_cache.setter
    def rgb_cache(self, val):
        self.manager.rgb_cache = val

    @property
    def offset_cache(self):
        return self.manager.offset_cache
    @offset_cache.setter
    def offset_cache(self, val):
        self.manager.offset_cache = val

    @property
    def cached_disp_rgb(self):
        return self.canvas_panel.cached_disp_rgb
    @cached_disp_rgb.setter
    def cached_disp_rgb(self, val):
        self.canvas_panel.cached_disp_rgb = val

    @property
    def manual_mode(self):
        return self.manager.manual_mode
    @manual_mode.setter
    def manual_mode(self, val):
        self.manager.manual_mode = val

    @property
    def manual_clicks(self):
        return self.manager.manual_clicks
    @manual_clicks.setter
    def manual_clicks(self, val):
        self.manager.manual_clicks = val

    @property
    def zoom_level(self):
        return self.manager.zoom_level
    @zoom_level.setter
    def zoom_level(self, val):
        self.manager.zoom_level = val

    @property
    def zoom_steps(self):
        return self.manager.zoom_steps
    @zoom_steps.setter
    def zoom_steps(self, val):
        self.manager.zoom_steps = val

    @property
    def pan_offset_x(self):
        return self.manager.pan_offset_x
    @pan_offset_x.setter
    def pan_offset_x(self, val):
        self.manager.pan_offset_x = val

    @property
    def pan_offset_y(self):
        return self.manager.pan_offset_y
    @pan_offset_y.setter
    def pan_offset_y(self, val):
        self.manager.pan_offset_y = val

    @property
    def autoplay_active(self):
        return self.manager.autoplay_active
    @autoplay_active.setter
    def autoplay_active(self, val):
        self.manager.autoplay_active = val

    @property
    def autoplay_speed_ms(self):
        return self.manager.autoplay_speed_ms
    @autoplay_speed_ms.setter
    def autoplay_speed_ms(self, val):
        self.manager.autoplay_speed_ms = val

    @property
    def photo_cache(self):
        return self.canvas_panel.photo_cache
    @photo_cache.setter
    def photo_cache(self, val):
        self.canvas_panel.photo_cache = val

    # Proxy Methods
    def on_resize(self, event):
        return self.canvas_panel.on_resize(event)

    def draw_canvas(self, rgb, origin, direction):
        return self.canvas_panel.draw_canvas(rgb, origin, direction)

    def jump_to_frame(self, idx):
        self.manager.current_idx = idx
        self.control_panel.frame_slider.set(idx)
        return self._apply_frame_change()

    def change_pca_threshold(self, val):
        self.manager.pca_threshold = val
        self.control_panel.pca_panel.pca_slider.set(val)
        return self._apply_pca_change()
