# align_app/ui/slideshow/canvas_panel.py

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

class SlideshowCanvasPanel(tk.Canvas):
    """
    A custom Tkinter Canvas panel responsible for drawing images, fit vectors, and manual alignment markers.

    GUI Architecture & Coordinate Transformations:
    1. Scaling & Letterboxing: Fits the raw image to the canvas dimensions while preserving aspect ratio. 
       Computes scaling factor `scale = min(canvas_w / img_w, canvas_h / img_h)` and top-left displacement 
       offsets `(dx, dy)` to center the image.
    2. Dynamic Zoom & Pan: Integrates zoom factors and pan offsets into the coordinate transformation:
       canvas_coord = (raw_coord * scale * zoom_factor) + offset
    3. PhotoImage Caching (LRU): To prevent memory leaks and garbage collection overhead during rapid frame 
       navigation, a Least Recently Used (LRU) cache holds up to 20 rendered `ImageTk.PhotoImage` references.
    4. Resampling Logic: Uses `Image.Resampling.BILINEAR` for default displays. When zoom factor is greater 
       than 1, uses `Image.Resampling.NEAREST` to display sharp pixels without blurring.
    5. Overlay Rendering: Superimposes red centroids and line fit vectors. Manual overrides are rendered 
       in lime green.
    """
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, bg="black", highlightthickness=0, **kwargs)
        self.controller = controller

        self.bind("<Configure>", self.on_resize)
        self.bind("<Button-1>", self.controller.handle_canvas_click)

        self.photo_img = None
        self.photo_cache = {}          # {cache_key: ImageTk.PhotoImage}
        self._photo_cache_order = []   # LRU order tracking
        self._photo_cache_max = 20     # Max cached PhotoImages

        # Letterbox transform params
        self._lb_scale = 1.0
        self._lb_dx = 0
        self._lb_dy = 0
        self._img_w = 0
        self._img_h = 0

        self.cached_disp_rgb = None

    def on_resize(self, event=None):
        if self.cached_disp_rgb is not None:
            self.controller._render_display()
        else:
            self.controller.load_and_render()

    def draw_canvas(self, rgb, origin, direction):
        if rgb is None:
            return
        ih, iw = rgb.shape[:2]
        if iw <= 0 or ih <= 0:
            return

        self.delete("all")

        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            cw = 600
            ch = 400

        base_scale = min(cw / iw, ch / ih)
        zoom_factor = self.controller.manager.zoom_steps[self.controller.manager.zoom_level]
        scale = base_scale * zoom_factor

        nw = int(iw * scale)
        nh = int(ih * scale)
        if nw <= 0 or nh <= 0:
            return

        dx = (cw - nw) // 2 + self.controller.manager.pan_offset_x
        dy = (ch - nh) // 2 + self.controller.manager.pan_offset_y

        self._lb_scale = scale
        self._lb_dx = dx
        self._lb_dy = dy
        self._img_w = iw
        self._img_h = ih

        # Check PhotoImage cache first
        cache_key = (id(rgb), nw, nh, self.controller.manager.zoom_level)
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

        self.create_image(dx, dy, image=self.photo_img, anchor="nw", tags="image")

        # Draw reference line
        show_line = bool(self.controller.navbar.show_line_switch.get())
        if show_line and origin is not None and direction is not None:
            ox_canvas = dx + origin[0] * scale
            oy_canvas = dy + origin[1] * scale

            self.create_oval(
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

                is_manual = self.controller.manager.current_idx in self.controller.manager.per_frame_manual
                line_color = "lime" if is_manual else "red"
                self.create_line(
                    p1_x, p1_y, p2_x, p2_y,
                    fill=line_color, width=2, tags="peak_line"
                )

    def draw_marker(self, cx, cy):
        self.create_oval(
            cx - 5, cy - 5, cx + 5, cy + 5,
            fill="lime", outline="white", width=2, tags="manual_marker"
        )

    def clear(self):
        self.delete("all")

    def render_error(self, img_path):
        self.delete("all")
        cw = self.winfo_width()
        ch = self.winfo_height()
        self.create_text(
            cw / 2 or 300,
            ch / 2 or 200,
            text=f"Error loading image:\n{img_path}",
            fill="red", tags="error"
        )

    def set_cached_image(self, disp_rgb):
        self.cached_disp_rgb = disp_rgb

    def clear_photo_cache(self):
        self.photo_cache.clear()
        self._photo_cache_order.clear()

    def _evict_photo_cache(self):
        while len(self._photo_cache_order) > self._photo_cache_max:
            old_key = self._photo_cache_order.pop(0)
            self.photo_cache.pop(old_key, None)
