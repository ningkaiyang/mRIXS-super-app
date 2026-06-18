import customtkinter
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import queue

from align_app.core import (
    preprocess_image,
    load_raw,
    apply_colormap,
    generate_aligned_sum,
    find_peak_line,
    find_peak_line_fast,
    phase_correlation_offset,
    compute_line_based_offset,
    warp_image,
    _weighted_pca
)
from align_app.widgets import RangeSlider


class SlideshowView(customtkinter.CTkFrame):
    def __init__(self, parent, on_back_to_sorting=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_back_to_sorting = on_back_to_sorting
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

        # Reference line (computed once from frame 1, held static)
        self.ref_raw = None
        self.ref_origin = None
        self.ref_direction = None
        self.ref_threshold = 99.9

        # Per-frame state
        self.per_frame_threshold = {}    # {frame_idx: float}
        self.per_frame_manual = {}       # {frame_idx: np.ndarray (centroid)}
        # Per-frame line results (cached)
        self.per_frame_origin = {}       # {frame_idx: np.ndarray}

        # Current frame data
        self.current_raw = None
        self.current_rgb = None

        # Caches for performance
        self.raw_cache = {}       # {filepath: np.ndarray}
        self.rgb_cache = {}       # {(filepath, colormap): np.ndarray}
        self.offset_cache = {}    # {filepath: (dx, dy)}

        # Display cache (for resize redraws without recomputation)
        self.cached_disp_rgb = None

        # Manual line mode
        self.manual_mode = False
        self.manual_clicks = []

        # Zoom state
        self.zoom_level = 0
        self.zoom_steps = [1, 2, 4, 8, 16]
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        # Autoplay state
        self.autoplay_active = False
        self.autoplay_speed_ms = 500
        self._autoplay_job = None

        # Debounce IDs
        self._pca_debounce_id = None
        self._frame_debounce_id = None

        # Letterbox transform params
        self._lb_scale = 1.0
        self._lb_dx = 0
        self._lb_dy = 0
        self._img_w = 0
        self._img_h = 0

        # Thread-safe result queue (fixes self.after() from background threads)
        self._result_queue = queue.Queue()

        # PhotoImage cache for rendering performance
        self.photo_cache = {}          # {cache_key: ImageTk.PhotoImage}
        self._photo_cache_order = []   # LRU order tracking
        self._photo_cache_max = 20     # Max cached PhotoImages

        self._build_ui()
        self._poll_queue()  # Start the queue poll loop

    def _build_ui(self):
        # === Navigation Bar ===
        self.nav_frame = customtkinter.CTkFrame(self)
        self.nav_frame.pack(fill="x", pady=5)

        self.back_button = customtkinter.CTkButton(
            self.nav_frame, text="◀ Back", command=self.back_to_sorting, width=80
        )
        self.back_button.pack(side="left", padx=5)

        self.prev_button = customtkinter.CTkButton(
            self.nav_frame, text="◀ Prev", command=self.prev_frame, width=80
        )
        self.prev_button.pack(side="left", padx=5)

        self.next_button = customtkinter.CTkButton(
            self.nav_frame, text="Next ▶", command=self.next_frame, width=80
        )
        self.next_button.pack(side="left", padx=5)

        self.autoplay_button = customtkinter.CTkButton(
            self.nav_frame, text="▶ Play", command=self.toggle_autoplay,
            width=80, fg_color="#2FA572", hover_color="#238a5a"
        )
        self.autoplay_button.pack(side="left", padx=5)

        self.show_line_switch = customtkinter.CTkSwitch(
            self.nav_frame, text="Show Ref Line", command=self._render_display
        )
        self.show_line_switch.select()
        self.show_line_switch.pack(side="right", padx=5)

        self.warp_switch = customtkinter.CTkSwitch(
            self.nav_frame, text="Warp Image", command=self.toggle_warp
        )
        self.warp_switch.select()
        self.warp_switch.pack(side="right", padx=5)

        self.colormap_menu = customtkinter.CTkOptionMenu(
            self.nav_frame,
            values=["viridis", "inferno", "plasma", "magma", "grayscale"],
            command=self.change_colormap
        )
        self.colormap_menu.set("viridis")
        self.colormap_menu.pack(side="right", padx=5)

        # === Sliders Frame ===
        self.slider_frame = customtkinter.CTkFrame(self)
        self.slider_frame.pack(fill="x", pady=5)

        # PCA threshold row
        self.pca_frame = customtkinter.CTkFrame(self.slider_frame)
        self.pca_frame.pack(fill="x", pady=2)

        self.pca_label = customtkinter.CTkLabel(self.pca_frame, text="PCA Threshold: 99.9000%")
        self.pca_label.pack(side="left", padx=5)

        self.pca_slider = customtkinter.CTkSlider(
            self.pca_frame, from_=95.0, to=99.9999,
            number_of_steps=4999, command=self._on_pca_slider_move
        )
        self.pca_slider.set(99.9)
        self.pca_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.pca_entry = customtkinter.CTkEntry(self.pca_frame, width=80, placeholder_text="99.9000")
        self.pca_entry.pack(side="left", padx=2)
        self.pca_entry.insert(0, "99.9000")
        self.pca_entry.bind("<Return>", self._on_pca_entry_submit)

        self.auto_snap_button = customtkinter.CTkButton(
            self.pca_frame, text="Auto", command=self._auto_snap_threshold,
            width=50, fg_color="#555", hover_color="#777"
        )
        self.auto_snap_button.pack(side="left", padx=2)

        self.auto_all_button = customtkinter.CTkButton(
            self.pca_frame, text="Auto All", command=self._auto_snap_all_frames,
            width=65, fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.auto_all_button.pack(side="left", padx=2)

        # Frame slider row
        self.frame_nav_frame = customtkinter.CTkFrame(self.slider_frame)
        self.frame_nav_frame.pack(fill="x", pady=2)

        self.frame_label = customtkinter.CTkLabel(self.frame_nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side="left", padx=5)

        self.frame_slider = customtkinter.CTkSlider(
            self.frame_nav_frame, from_=0, to=1,
            number_of_steps=1, command=self._on_frame_slider_move
        )
        self.frame_slider.set(0)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)

        # === Tools Frame ===
        self.tools_frame = customtkinter.CTkFrame(self)
        self.tools_frame.pack(fill="x", pady=2)

        self.manual_line_button = customtkinter.CTkButton(
            self.tools_frame, text="✏ Manual Line", command=self.toggle_manual_mode,
            width=110, fg_color="#555", hover_color="#777"
        )
        self.manual_line_button.pack(side="left", padx=5)

        self.clear_manual_button = customtkinter.CTkButton(
            self.tools_frame, text="Clear Manual", command=self.clear_manual_line,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.clear_manual_button.pack(side="left", padx=2)

        self.zoom_in_button = customtkinter.CTkButton(
            self.tools_frame, text="🔍+ Zoom In", command=self.zoom_in,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.zoom_in_button.pack(side="left", padx=5)

        self.zoom_out_button = customtkinter.CTkButton(
            self.tools_frame, text="🔍- Zoom Out", command=self.zoom_out,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.zoom_out_button.pack(side="left", padx=2)

        self.reset_view_button = customtkinter.CTkButton(
            self.tools_frame, text="⟲ Reset View", command=self.reset_view,
            width=100, fg_color="#555", hover_color="#777"
        )
        self.reset_view_button.pack(side="left", padx=2)

        self.zoom_label = customtkinter.CTkLabel(self.tools_frame, text="Zoom: 1×")
        self.zoom_label.pack(side="left", padx=5)

        # Per-frame info label
        self.frame_info_label = customtkinter.CTkLabel(
            self.tools_frame, text="", text_color="#88aacc"
        )
        self.frame_info_label.pack(side="right", padx=10)

        # === Metadata Label ===
        self.metadata_label = customtkinter.CTkLabel(
            self, text="Filename: - | Frame Index: - | Offset: (0.00, 0.00)"
        )
        self.metadata_label.pack(fill="x", pady=2)

        # === Bottom Bar ===
        self.bottom_bar = customtkinter.CTkFrame(self)
        self.bottom_bar.pack(fill="x", side="bottom", pady=5)

        # Bottom Bar Columns
        self.clamping_frame = customtkinter.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.clamping_frame.pack(side="left", fill="x", expand=True, padx=10)

        self.export_frame = customtkinter.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.export_frame.pack(side="right", padx=10)

        # Clamping section widgets
        self.clamping_label = customtkinter.CTkLabel(self.clamping_frame, text="Intensity Clamping:")
        self.clamping_label.pack(side="left", padx=5)

        self.floor_entry = customtkinter.CTkEntry(self.clamping_frame, width=80)
        self.floor_entry.pack(side="left", padx=5)
        self.floor_entry.bind("<Return>", self._on_floor_entry_submit)

        self.range_slider = RangeSlider(
            self.clamping_frame, height=25, command=self._on_clamping_change
        )
        self.range_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.ceiling_entry = customtkinter.CTkEntry(self.clamping_frame, width=80)
        self.ceiling_entry.pack(side="left", padx=5)
        self.ceiling_entry.bind("<Return>", self._on_ceiling_entry_submit)

        # Export section widgets
        self.progress_label = customtkinter.CTkLabel(self.export_frame, text="", text_color="#aaaaaa")
        self.progress_label.pack(side="left", padx=5)

        self.export_button = customtkinter.CTkButton(
            self.export_frame, text="💾 Export Aligned Sum",
            command=self.export_aligned_sum,
            fg_color="#2F72A5", hover_color="#1F5A85"
        )
        self.export_button.pack(side="left", padx=5)

        # === Canvas === (Greedy widget, packed last)
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=5)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.photo_img = None

    # ─── Lifecycle ────────────────────────────────────────────

    def start(self, file_list):
        self.file_list = file_list
        self.current_idx = 0

        # Clear all state
        self.ref_raw = None
        self.ref_origin = None
        self.ref_direction = None
        self.raw_cache.clear()
        self.rgb_cache.clear()
        self.offset_cache.clear()
        self.cached_disp_rgb = None
        self.photo_cache.clear()
        self._photo_cache_order.clear()
        self.per_frame_threshold.clear()
        self.per_frame_manual.clear()
        self.per_frame_origin.clear()
        self.manual_clicks.clear()
        self.zoom_level = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0

        if len(self.file_list) > 1:
            self.frame_slider.configure(
                from_=0, to=len(self.file_list) - 1,
                number_of_steps=len(self.file_list) - 1, state="normal"
            )
        else:
            self.frame_slider.configure(from_=0, to=1, number_of_steps=1, state="disabled")
        self.frame_slider.set(0)

        self._load_reference()
        self._preload_all_in_background()
        self.load_and_render()

    def _load_reference(self):
        """Compute the reference line from frame 1."""
        if not self.file_list:
            return
        ref_raw = self._get_raw(self.file_list[0])
        if ref_raw is None:
            return
        self.ref_raw = ref_raw

        # Initialize display intensity boundaries once
        if ref_raw is not None and ref_raw.size > 0:
            self.intensity_min = float(np.min(ref_raw))
            self.intensity_max = float(np.max(ref_raw))
        else:
            self.intensity_min = 0.0
            self.intensity_max = 1.0
        if self.intensity_max <= self.intensity_min:
            self.intensity_max = self.intensity_min + 1.0

        self.clamping_floor = self.intensity_min
        self.clamping_ceiling = self.intensity_max

        self.range_slider.configure_range(self.intensity_min, self.intensity_max)
        self.range_slider.set_values(self.clamping_floor, self.clamping_ceiling)

        self.floor_entry.delete(0, "end")
        self.floor_entry.insert(0, f"{self.clamping_floor:.4f}")
        self.ceiling_entry.delete(0, "end")
        self.ceiling_entry.insert(0, f"{self.clamping_ceiling:.4f}")

        t = self.per_frame_threshold.get(0, self.pca_threshold)
        self.ref_threshold = t
        self.ref_origin, self.ref_direction = find_peak_line(ref_raw, t)
        self.per_frame_origin[0] = self.ref_origin.copy()

    def _preload_all_in_background(self):
        def _worker():
            for path in self.file_list:
                self._get_raw(path)
        threading.Thread(target=_worker, daemon=True).start()

    # ─── Data Loading (Cached) ────────────────────────────────

    def _get_raw(self, filepath):
        if filepath in self.raw_cache:
            return self.raw_cache[filepath]
        try:
            raw = load_raw(filepath)
            self.raw_cache[filepath] = raw
            return raw
        except Exception:
            return None

    def _get_rgb(self, filepath, colormap):
        """Get RGB display image using apply_colormap() on cached raw data.

        This avoids re-reading the TIFF from disk — the raw array is already
        in raw_cache from _get_raw(). Only the colormap + clamping math runs,
        making slider dragging near-instant (~5-50ms).
        """
        cache_key = (filepath, colormap)
        if cache_key in self.rgb_cache:
            return self.rgb_cache[cache_key]
        try:
            raw = self._get_raw(filepath)
            if raw is None:
                return None
            floor, ceiling = self.get_current_clamping()
            rgb = apply_colormap(raw, colormap,
                                display_floor=floor, display_ceiling=ceiling)
            self.rgb_cache[cache_key] = rgb
            return rgb
        except Exception:
            return None

    def _get_offset(self, frame_idx):
        """Get alignment offset for a frame using line-based alignment."""
        filepath = self.file_list[frame_idx]
        
        # Check if manual centroid exists for this frame
        if frame_idx in self.per_frame_manual:
            manual_centroid = self.per_frame_manual[frame_idx]
            # Compute offset from reference origin to manual centroid
            delta = manual_centroid - self.ref_origin
            return float(delta[0]), float(delta[1])

        # Use line-based alignment with per-frame threshold
        cache_key = (filepath, frame_idx)
        if cache_key in self.offset_cache:
            return self.offset_cache[cache_key]

        raw = self._get_raw(filepath)
        if raw is None or self.ref_raw is None or self.ref_direction is None:
            return (0.0, 0.0)

        target_threshold = self.per_frame_threshold.get(frame_idx, self.ref_threshold)
        try:
            dx, dy = compute_line_based_offset(
                self.ref_raw, raw,
                self.ref_direction, self.ref_origin,
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

    # ─── Navigation ───────────────────────────────────────────

    def back_to_sorting(self):
        self.stop_autoplay()
        if self.on_back_to_sorting:
            self.on_back_to_sorting()

    def prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.frame_slider.set(self.current_idx)
            self.load_and_render()

    def next_frame(self):
        if self.current_idx < len(self.file_list) - 1:
            self.current_idx += 1
            self.frame_slider.set(self.current_idx)
            self.load_and_render()

    def change_colormap(self, val):
        self.colormap = val
        self.photo_cache.clear()
        self._photo_cache_order.clear()
        self.load_and_render()

    def toggle_warp(self):
        self.warp_enabled = bool(self.warp_switch.get())
        self.photo_cache.clear()
        self._photo_cache_order.clear()
        self.load_and_render()

    # ─── PCA Threshold (per-frame) ────────────────────────────

    def _get_current_threshold(self):
        """Get the threshold for the current frame."""
        return self.per_frame_threshold.get(self.current_idx, self.ref_threshold)

    def _set_current_threshold(self, val):
        """Set the threshold for the current frame."""
        self.per_frame_threshold[self.current_idx] = val
        # Invalidate caches for this frame
        filepath = self.file_list[self.current_idx] if self.current_idx < len(self.file_list) else None
        if filepath:
            cache_key = (filepath, self.current_idx)
            self.offset_cache.pop(cache_key, None)
        self.per_frame_origin.pop(self.current_idx, None)

    def _format_threshold(self, t):
        s = f"{t:.4f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    def _sync_slider_to_frame(self):
        """Update slider/label/entry to show current frame's threshold."""
        t = self._get_current_threshold()
        self.pca_threshold = t
        self.pca_slider.set(min(t, 99.9999))
        self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(t)}%")
        self.pca_entry.delete(0, "end")
        self.pca_entry.insert(0, f"{t:.4f}")

        # Update per-frame info
        info_parts = []
        if self.current_idx in self.per_frame_threshold:
            info_parts.append(f"Custom threshold: {self.per_frame_threshold[self.current_idx]:.4f}%")
        if self.current_idx in self.per_frame_manual:
            info_parts.append("Manual line set")
        self.frame_info_label.configure(text=" | ".join(info_parts))

    def _on_pca_slider_move(self, val):
        self.pca_threshold = float(val)
        self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(self.pca_threshold)}%")
        self.pca_entry.delete(0, "end")
        self.pca_entry.insert(0, f"{self.pca_threshold:.4f}")
        if self._pca_debounce_id is not None:
            self.after_cancel(self._pca_debounce_id)
        self._pca_debounce_id = self.after(80, self._apply_pca_change)

    def _on_pca_entry_submit(self, event=None):
        try:
            val = float(self.pca_entry.get())
            val = max(95.0, min(99.9999, val))
            self.pca_threshold = val
            self.pca_slider.set(val)
            self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(val)}%")
            self._apply_pca_change()
        except ValueError:
            pass

    def _apply_pca_change(self):
        self._pca_debounce_id = None
        self._set_current_threshold(self.pca_threshold)

        if self.current_idx == 0:
            # Recompute reference line
            self._load_reference()
            # Clear all offset caches (reference changed)
            self.offset_cache.clear()
            self.per_frame_origin.clear()
            self.per_frame_origin[0] = self.ref_origin.copy()

        self.load_and_render()

    def _poll_queue(self):
        """Drain all pending results from background threads (runs on main thread)."""
        try:
            while True:
                callback = self._result_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        # Re-schedule every 50ms
        self.after(50, self._poll_queue)

    def _auto_snap_threshold(self):
        """Auto-snap threshold for the CURRENT frame (runs in background thread)."""
        idx = self.current_idx
        raw = self._get_raw(self.file_list[idx]) if idx < len(self.file_list) else None
        if raw is None:
            return

        self.auto_snap_button.configure(text="...", state="disabled")

        def _worker():
            best = self._find_best_threshold(raw)
            self._result_queue.put(lambda: self._finish_auto_snap(best))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_auto_snap(self, best_threshold):
        self.auto_snap_button.configure(text="Auto", state="normal")
        self.pca_threshold = best_threshold
        self.pca_slider.set(min(best_threshold, 99.9999))
        self.pca_label.configure(text=f"PCA Threshold: {self._format_threshold(best_threshold)}%")
        self.pca_entry.delete(0, "end")
        self.pca_entry.insert(0, f"{best_threshold:.4f}")
        self._apply_pca_change()

    def _auto_snap_all_frames(self):
        """Auto-snap threshold for ALL frames (runs in background thread)."""
        n_frames = len(self.file_list)
        self.auto_all_button.configure(text=f"0/{n_frames}...", state="disabled")
        self.auto_snap_button.configure(state="disabled")

        def _worker():
            results = {}
            for idx in range(n_frames):
                raw = self._get_raw(self.file_list[idx])
                if raw is not None:
                    results[idx] = self._find_best_threshold(raw)
                # Post progress update via queue (thread-safe)
                count = idx + 1
                self._result_queue.put(
                    lambda c=count: self.auto_all_button.configure(
                        text=f"{c}/{n_frames}..."
                    )
                )
            # Post completion via queue
            self._result_queue.put(lambda: self._finish_auto_snap_all(results))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_auto_snap_all(self, results):
        self.auto_all_button.configure(text="Auto All", state="normal")
        self.auto_snap_button.configure(state="normal")

        for idx, threshold in results.items():
            self.per_frame_threshold[idx] = threshold

        if 0 in self.per_frame_threshold:
            self.ref_threshold = self.per_frame_threshold[0]
        self._load_reference()
        self.offset_cache.clear()
        self.per_frame_origin.clear()

        self._sync_slider_to_frame()
        self.load_and_render()

    def _find_best_threshold(self, raw):
        """Find optimal threshold using pre-sorted pixel indices for speed.
        
        Pre-sorts all pixels by intensity once (~140ms), then each threshold
        evaluation is a simple array slice + fast PCA (~3ms each).
        Total: ~1.5s per frame instead of ~38s.
        """
        # Pre-sort all pixel indices by intensity (ascending) — done ONCE
        h, w = raw.shape
        flat = raw.ravel()
        sorted_idx = np.argsort(flat)
        sorted_rows = sorted_idx // w
        sorted_cols = sorted_idx % w
        n_total = len(flat)

        best_threshold = 99.9
        best_spread = float('inf')

        def _eval_at(t_pct):
            """Evaluate spread at percentile t_pct using pre-sorted indices."""
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

        # Coarse pass: 98.00 to 99.99 in 0.01 steps
        for t_int in range(9800, 10000):
            t = t_int / 100.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        # Fine pass: best ± 0.1 in 0.001 steps
        fine_lo = max(98.0, best_threshold - 0.1)
        fine_hi = min(99.999, best_threshold + 0.1)
        for t_int in range(int(fine_lo * 1000), int(fine_hi * 1000) + 1):
            t = t_int / 1000.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        # Ultra-fine pass: best ± 0.005 in 0.0001 steps
        uf_lo = max(98.0, best_threshold - 0.005)
        uf_hi = min(99.9999, best_threshold + 0.005)
        for t_int in range(int(uf_lo * 10000), int(uf_hi * 10000) + 1):
            t = t_int / 10000.0
            spread = _eval_at(t)
            if spread is not None and spread < best_spread:
                best_spread = spread
                best_threshold = t

        return best_threshold

    # ─── Frame Slider ─────────────────────────────────────────

    def _on_frame_slider_move(self, val):
        idx = int(float(val))
        if 0 <= idx < len(self.file_list) and idx != self.current_idx:
            self.current_idx = idx
            if self._frame_debounce_id is not None:
                self.after_cancel(self._frame_debounce_id)
            self._frame_debounce_id = self.after(50, self._apply_frame_change)

    def _apply_frame_change(self):
        self._frame_debounce_id = None
        self.load_and_render()

    # ─── Autoplay ─────────────────────────────────────────────

    def toggle_autoplay(self):
        if self.autoplay_active:
            self.stop_autoplay()
        else:
            self.start_autoplay()

    def start_autoplay(self):
        self.autoplay_active = True
        self.autoplay_button.configure(text="⏸ Pause", fg_color="#cc5500")
        self._autoplay_tick()

    def stop_autoplay(self):
        self.autoplay_active = False
        self.autoplay_button.configure(text="▶ Play", fg_color="#2FA572")
        if self._autoplay_job is not None:
            self.after_cancel(self._autoplay_job)
            self._autoplay_job = None

    def _autoplay_tick(self):
        if not self.autoplay_active:
            return
        if self.current_idx < len(self.file_list) - 1:
            self.current_idx += 1
        else:
            self.current_idx = 0
        self.frame_slider.set(self.current_idx)
        self.load_and_render()
        self._autoplay_job = self.after(self.autoplay_speed_ms, self._autoplay_tick)

    # ─── Zoom ─────────────────────────────────────────────────

    def _compute_centroid_pan(self):
        origin = self._get_display_origin()
        if origin is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        ih, iw = self.cached_disp_rgb.shape[:2] if self.cached_disp_rgb is not None else (2048, 3840)
        base_scale = min(cw / iw, ch / ih)
        zoom_factor = self.zoom_steps[self.zoom_level]
        scale = base_scale * zoom_factor
        nw = int(iw * scale)
        nh = int(ih * scale)
        base_dx = (cw - nw) // 2
        base_dy = (ch - nh) // 2
        cx = base_dx + origin[0] * scale
        cy = base_dy + origin[1] * scale
        self.pan_offset_x = int(cw / 2 - cx)
        self.pan_offset_y = int(ch / 2 - cy)

    def zoom_in(self):
        if self.zoom_level < len(self.zoom_steps) - 1:
            self.zoom_level += 1
            self.zoom_label.configure(text=f"Zoom: {self.zoom_steps[self.zoom_level]}×")
            if self.zoom_level > 0:
                self._compute_centroid_pan()
            self._render_display()

    def zoom_out(self):
        if self.zoom_level > 0:
            self.zoom_level -= 1
            self.zoom_label.configure(text=f"Zoom: {self.zoom_steps[self.zoom_level]}×")
            if self.zoom_level == 0:
                self.pan_offset_x = 0
                self.pan_offset_y = 0
            else:
                self._compute_centroid_pan()
            self._render_display()

    def reset_view(self):
        self.zoom_level = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.zoom_label.configure(text="Zoom: 1×")
        self._render_display()

    # ─── Manual Line Mode ─────────────────────────────────────

    def toggle_manual_mode(self):
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            self.manual_line_button.configure(fg_color="#cc5500", text="✏ Click 2 pts...")
            self.manual_clicks.clear()
            self.canvas.configure(cursor="crosshair")
        else:
            self.manual_line_button.configure(fg_color="#555", text="✏ Manual Line")
            self.canvas.configure(cursor="")

    def clear_manual_line(self):
        """Clear manual line for the current frame."""
        self.per_frame_manual.pop(self.current_idx, None)
        self.manual_clicks.clear()
        # Invalidate offset cache for this frame
        if self.current_idx < len(self.file_list):
            filepath = self.file_list[self.current_idx]
            self.offset_cache.pop((filepath, self.current_idx), None)
        if self.manual_mode:
            self.toggle_manual_mode()
        self.load_and_render()

    def _on_canvas_click(self, event):
        if not self.manual_mode:
            return

        canvas_x, canvas_y = event.x, event.y
        self.manual_clicks.append((canvas_x, canvas_y))

        if len(self.manual_clicks) == 1:
            self.canvas.create_oval(
                canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5,
                fill="lime", outline="white", width=2, tags="manual_marker"
            )
        elif len(self.manual_clicks) >= 2:
            cx1, cy1 = self.manual_clicks[0]
            cx2, cy2 = self.manual_clicks[1]

            # Convert canvas coords to image coords
            ix1 = (cx1 - self._lb_dx) / self._lb_scale
            iy1 = (cy1 - self._lb_dy) / self._lb_scale
            ix2 = (cx2 - self._lb_dx) / self._lb_scale
            iy2 = (cy2 - self._lb_dy) / self._lb_scale

            # Compute midpoint in the displayed (possibly warped) image coords
            midpoint = np.array([(ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0])

            # If warp is enabled, back-calculate the un-warped centroid.
            # The displayed image was shifted by (-dx, -dy), so the original
            # position is midpoint + (dx, dy) = midpoint - (-dx, -dy).
            if self.warp_enabled and self.current_idx > 0 and self.ref_raw is not None:
                # Get the current warp offset that was applied
                existing_dx, existing_dy = self._get_offset(self.current_idx)
                # Undo the warp: the display was shifted by (-dx, -dy),
                # so original coords = displayed coords + (dx, dy)
                midpoint = midpoint + np.array([existing_dx, existing_dy])

            # Store the manual centroid for THIS frame (uses reference slope)
            self.per_frame_manual[self.current_idx] = midpoint

            # Invalidate offset cache for this frame
            if self.current_idx < len(self.file_list):
                filepath = self.file_list[self.current_idx]
                self.offset_cache.pop((filepath, self.current_idx), None)

            self.toggle_manual_mode()
            self.load_and_render()

    # ─── Display Origin Helper ────────────────────────────────

    def _get_display_origin(self):
        """Get the line origin to display (manual override > per-frame PCA > reference)."""
        if self.current_idx in self.per_frame_manual:
            return self.per_frame_manual[self.current_idx]
        if self.current_idx in self.per_frame_origin:
            return self.per_frame_origin[self.current_idx]
        if self.ref_origin is not None:
            return self.ref_origin
        return None

    # ─── Core Render Pipeline ─────────────────────────────────

    def load_and_render(self):
        if not self.file_list:
            self.canvas.delete("all")
            self.frame_label.configure(text="Frame: 0/0")
            self.metadata_label.configure(text="Filename: - | Frame Index: - | Offset: (0.00, 0.00)")
            return

        self.frame_label.configure(text=f"Frame: {self.current_idx + 1}/{len(self.file_list)}")
        self._sync_slider_to_frame()

        img_path = self.file_list[self.current_idx]

        raw = self._get_raw(img_path)
        rgb = self._get_rgb(img_path, self.colormap)
        if raw is None or rgb is None:
            self.cached_disp_rgb = None
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() / 2 or 300,
                self.canvas.winfo_height() / 2 or 200,
                text=f"Error loading image:\n{img_path}",
                fill="red", tags="error"
            )
            return

        self.current_raw = raw
        self.current_rgb = rgb

        # Compute per-frame line origin (if not cached and not manual)
        if (self.current_idx not in self.per_frame_origin and
            self.current_idx not in self.per_frame_manual):
            t = self._get_current_threshold()
            try:
                origin, _ = find_peak_line(raw, t)
                self.per_frame_origin[self.current_idx] = origin
            except Exception:
                pass

        # Get offset
        dx, dy = 0.0, 0.0
        if self.current_idx > 0 and self.ref_raw is not None:
            dx, dy = self._get_offset(self.current_idx)

        self.metadata_label.configure(
            text=f"Filename: {os.path.basename(img_path)} | Frame Index: {self.current_idx} | Offset: ({dx:.2f}, {dy:.2f})"
        )

        # Apply warp
        disp_rgb = rgb.copy()
        if self.warp_enabled and self.current_idx > 0 and self.ref_raw is not None:
            try:
                disp_rgb = warp_image(rgb, -dx, -dy)
            except Exception:
                pass

        self.cached_disp_rgb = disp_rgb
        self._render_display()

    def _render_display(self):
        if self.cached_disp_rgb is None:
            return

        # Always draw with the REFERENCE line (origin + direction)
        # The origin shown depends on whether we're viewing frame 1 or a warped frame
        if self.ref_origin is not None and self.ref_direction is not None:
            origin = self.ref_origin
            direction = self.ref_direction
        else:
            origin = None
            direction = None

        self._draw_canvas(self.cached_disp_rgb, origin, direction)

    def _get_photo_cache_key(self, rgb, nw, nh):
        """Build a cache key for PhotoImage based on current display state."""
        # Use id(rgb) as a fast proxy — rgb arrays are cached in rgb_cache
        # so the same frame+colormap always returns the same object
        return (id(rgb), nw, nh, self.zoom_level)

    def _evict_photo_cache(self):
        """LRU eviction: remove oldest entries beyond max cache size."""
        while len(self._photo_cache_order) > self._photo_cache_max:
            old_key = self._photo_cache_order.pop(0)
            self.photo_cache.pop(old_key, None)

    def _draw_canvas(self, rgb, origin, direction):
        if rgb is None:
            return
        ih, iw = rgb.shape[:2]
        if iw <= 0 or ih <= 0:
            return

        self.canvas.delete("all")

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw = 600
            ch = 400

        base_scale = min(cw / iw, ch / ih)
        zoom_factor = self.zoom_steps[self.zoom_level]
        scale = base_scale * zoom_factor

        nw = int(iw * scale)
        nh = int(ih * scale)
        if nw <= 0 or nh <= 0:
            return

        dx = (cw - nw) // 2 + self.pan_offset_x
        dy = (ch - nh) // 2 + self.pan_offset_y

        self._lb_scale = scale
        self._lb_dx = dx
        self._lb_dy = dy
        self._img_w = iw
        self._img_h = ih

        # Check PhotoImage cache first
        cache_key = self._get_photo_cache_key(rgb, nw, nh)
        if cache_key in self.photo_cache:
            self.photo_img = self.photo_cache[cache_key]
            # Move to end of LRU order
            if cache_key in self._photo_cache_order:
                self._photo_cache_order.remove(cache_key)
            self._photo_cache_order.append(cache_key)
        else:
            # Create new PhotoImage with appropriate resampling
            pil_img = Image.fromarray(rgb)
            if zoom_factor > 1:
                # NEAREST for zoomed views: sharp pixels, very fast
                pil_img_resized = pil_img.resize((nw, nh), Image.Resampling.NEAREST)
            else:
                # BILINEAR at 1×: good quality, ~3× faster than LANCZOS
                pil_img_resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
            self.photo_img = ImageTk.PhotoImage(pil_img_resized)
            # Cache it
            self.photo_cache[cache_key] = self.photo_img
            self._photo_cache_order.append(cache_key)
            self._evict_photo_cache()

        self.canvas.create_image(dx, dy, image=self.photo_img, anchor="nw", tags="image")

        # Draw reference line
        if self.show_line_switch.get() and origin is not None and direction is not None:
            ox_canvas = dx + origin[0] * scale
            oy_canvas = dy + origin[1] * scale

            self.canvas.create_oval(
                ox_canvas - 4, oy_canvas - 4, ox_canvas + 4, oy_canvas + 4,
                fill="red", outline="white", tags="centroid"
            )

            dir_x, dir_y = direction
            if abs(dir_x) > 1e-5 or abs(dir_y) > 1e-5:
                extent = max(nw, nh) * 2
                p1_x = ox_canvas - dir_x * extent
                p1_y = oy_canvas - dir_y * extent
                p2_x = ox_canvas + dir_x * extent
                p2_y = oy_canvas + dir_y * extent

                is_manual = self.current_idx in self.per_frame_manual
                line_color = "lime" if is_manual else "red"
                self.canvas.create_line(
                    p1_x, p1_y, p2_x, p2_y,
                    fill=line_color, width=2, tags="peak_line"
                )

    def _on_resize(self, event):
        if self.cached_disp_rgb is not None:
            self._render_display()
        else:
            self.load_and_render()

    # ─── Display Intensity Clamping ───────────────────────────

    def get_current_clamping(self):
        floor = getattr(self, "clamping_floor", None)
        ceiling = getattr(self, "clamping_ceiling", None)
        return floor, ceiling

    def _on_clamping_change(self, floor, ceiling):
        self.clamping_floor = floor
        self.clamping_ceiling = ceiling

        # Update text entries in real-time
        self.floor_entry.delete(0, "end")
        self.floor_entry.insert(0, f"{floor:.4f}")
        self.ceiling_entry.delete(0, "end")
        self.ceiling_entry.insert(0, f"{ceiling:.4f}")

        # Debounce the heavy rendering updates
        if hasattr(self, "_clamping_debounce_id") and self._clamping_debounce_id is not None:
            self.after_cancel(self._clamping_debounce_id)
        self._clamping_debounce_id = self.after(80, self._apply_clamping_change)

    def _apply_clamping_change(self):
        self._clamping_debounce_id = None
        self.rgb_cache.clear()
        self.photo_cache.clear()
        self._photo_cache_order.clear()
        self.load_and_render()

    def _on_floor_entry_submit(self, event=None):
        try:
            val = float(self.floor_entry.get())
            val = max(self.intensity_min, min(self.clamping_ceiling - 1e-4, val))
            self.clamping_floor = val
            self.range_slider.set_values(self.clamping_floor, self.clamping_ceiling)
            self._apply_clamping_change()
        except ValueError:
            self.floor_entry.delete(0, "end")
            self.floor_entry.insert(0, f"{self.clamping_floor:.4f}")

    def _on_ceiling_entry_submit(self, event=None):
        try:
            val = float(self.ceiling_entry.get())
            val = max(self.clamping_floor + 1e-4, min(self.intensity_max, val))
            self.clamping_ceiling = val
            self.range_slider.set_values(self.clamping_floor, self.clamping_ceiling)
            self._apply_clamping_change()
        except ValueError:
            self.ceiling_entry.delete(0, "end")
            self.ceiling_entry.insert(0, f"{self.clamping_ceiling:.4f}")

    # ─── Aligned Sum Export ───────────────────────────────────

    def export_aligned_sum(self):
        """Export the aligned sum of all frames as a single TIFF file.

        Pre-computes all alignment offsets, opens a save dialog, then spawns
        a background thread using generate_aligned_sum() with progress updates.
        """
        if not self.file_list:
            return

        self.stop_autoplay()

        # Pre-compute ALL offsets on the main thread (may trigger PCA for
        # unvisited frames) so the background thread doesn't touch UI state.
        n_frames = len(self.file_list)
        self.progress_label.configure(text="Computing offsets...")
        self.update_idletasks()  # Force label redraw before blocking loop

        offsets = {0: (0.0, 0.0)}
        for idx in range(1, n_frames):
            self.progress_label.configure(text=f"Computing offsets: {idx}/{n_frames}...")
            self.update_idletasks()
            offsets[idx] = self._get_offset(idx)
        self.progress_label.configure(text="")

        # Open save dialog
        first_file = self.file_list[0]
        initial_dir = os.path.dirname(first_file)
        save_path = tk.filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile="aligned_sum.tif",
            defaultextension=".tif",
            filetypes=[("TIFF Files", "*.tif;*.tiff")]
        )
        if not save_path:
            return

        self._set_export_ui_state("disabled")

        ref_raw = self._get_raw(first_file)
        if ref_raw is None:
            self._finish_export(False, "Could not load reference frame.")
            return

        def _progress(current, total):
            self._result_queue.put(
                lambda c=current, t=total: self.progress_label.configure(
                    text=f"Summing frames: {c}/{t}..."
                )
            )

        def _worker():
            try:
                result = generate_aligned_sum(
                    self.file_list, self._get_raw, offsets,
                    ref_raw.shape, progress_callback=_progress
                )
                import tifffile
                tifffile.imwrite(save_path, result)
                self._result_queue.put(lambda: self._finish_export(True, save_path))
            except Exception as e:
                self._result_queue.put(lambda err=str(e): self._finish_export(False, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_export_ui_state(self, state):
        """Enable or disable all interactive UI widgets during export."""
        widgets = [
            self.back_button, self.prev_button, self.next_button, self.autoplay_button,
            self.pca_slider, self.pca_entry, self.auto_snap_button, self.auto_all_button,
            self.frame_slider, self.manual_line_button, self.clear_manual_button,
            self.zoom_in_button, self.zoom_out_button, self.reset_view_button,
            self.colormap_menu, self.warp_switch, self.show_line_switch,
            self.export_button
        ]
        for w in widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _finish_export(self, success, info):
        """Called on main thread when export completes."""
        self._set_export_ui_state("normal")
        self.progress_label.configure(text="")
        if success:
            tk.messagebox.showinfo("Export Successful", f"Aligned sum saved to:\n{info}")
        else:
            tk.messagebox.showerror("Export Failed", f"An error occurred during export:\n{info}")

    # ─── Public Wrapper Methods (for Test Compatibility) ─────

    def on_resize(self, event):
        return self._on_resize(event)

    def draw_canvas(self, rgb, origin, direction):
        return self._draw_canvas(rgb, origin, direction)

    def jump_to_frame(self, idx):
        self.current_idx = idx
        self.frame_slider.set(idx)
        return self._apply_frame_change()

    def change_pca_threshold(self, val):
        self.pca_threshold = val
        self.pca_slider.set(val)
        return self._apply_pca_change()
