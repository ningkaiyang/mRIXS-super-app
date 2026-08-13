"""Alignment slideshow canvas panel — PySide6 port.

Replaces the former Tkinter ``tk.Canvas``-based ``SlideshowCanvasPanel``.
All image rendering is done via a ``QLabel`` that holds a ``QPixmap``
derived from the numpy RGB array.  Line / centroid overlays are drawn
with ``QPainter`` on top of that pixmap.

Key design decisions
- **Letterbox scaling** (same formula as Tkinter version: scale = min(cw/iw, ch/ih)).
- **Zoom × pan** parameters live on the manager so they are shared with
  any other panel that needs them.
- **LRU PhotoImage cache** replaced by a simple dict keyed on
  (frame_idx, id(rgb), nw, nh) limited to 20 entries.
- **Mouse click** forwarded to the controller via ``handle_canvas_click``.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget, QSizePolicy


class SlideshowCanvasPanel(QWidget):
    """PySide6 canvas widget for rendering RIXS alignment slideshow frames.

    Responsibilities:
    - Convert numpy RGB arrays to ``QPixmap`` with letterbox scaling.
    - Overlay centroid circles and direction lines for PCA mode.
    - Draw manual-alignment click markers.
    - Forward left-button press events to the controller.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` controller.
    """

    _MAX_CACHE = 20  # max LRU cached pixmaps

    def __init__(self, parent=None, *, controller):
        """Initialise the canvas panel.

        Args:
            parent: Parent QWidget.
            controller: ``SlideshowView`` instance — the view/controller.
        """
        super().__init__(parent)
        self.controller = controller
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 100)
        self.setStyleSheet("background-color: #000000;")

        # Current rendered pixmap (letterboxed & zoomed)
        self._pixmap: QPixmap | None = None
        # LRU cache: OrderedDict keyed (frame_idx, rgb_id, nw, nh)
        self._pixmap_cache: OrderedDict = OrderedDict()

        # Letterbox transform state (updated in paintEvent)
        self._lb_scale: float = 1.0
        self._lb_dx: int = 0
        self._lb_dy: int = 0
        self._img_w: int = 0
        self._img_h: int = 0

        # Pending render data (set before calling update())
        self._pending_rgb: np.ndarray | None = None
        self._pending_origin: np.ndarray | None = None
        self._pending_direction: np.ndarray | None = None

        # Cached display RGB (mirrors old .cached_disp_rgb)
        self.cached_disp_rgb: np.ndarray | None = None

        # Manual markers list [(cx, cy), ...]
        self._markers: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Public API (matches old Tkinter interface)
    # ------------------------------------------------------------------

    def draw_canvas(self, rgb: np.ndarray, origin, direction) -> None:
        """Schedule a repaint with a new RGB frame and optional overlay vectors.

        Args:
            rgb: H×W×3 uint8 numpy array.
            origin: (x, y) centroid as np.ndarray or None.
            direction: (dx, dy) unit vector as np.ndarray or None.
        """
        if rgb is None:
            return
        self._pending_rgb = rgb
        self._pending_origin = origin
        self._pending_direction = direction
        self._markers.clear()
        self.update()

    def draw_marker(self, cx: float, cy: float) -> None:
        """Add a manual-alignment click marker and repaint.

        Args:
            cx: Canvas x coordinate.
            cy: Canvas y coordinate.
        """
        self._markers.append((int(cx), int(cy)))
        self.update()

    def clear(self) -> None:
        """Clear the canvas to solid black."""
        self._pending_rgb = None
        self._pending_origin = None
        self._pending_direction = None
        self._markers.clear()
        self.cached_disp_rgb = None
        self.update()

    def render_error(self, img_path: str) -> None:
        """Display an error message on the canvas.

        Args:
            img_path: The path that failed to load.
        """
        self._pending_rgb = None
        self._err_msg = f"Error loading image:\n{img_path}"
        self.update()

    def set_cached_image(self, disp_rgb: np.ndarray | None) -> None:
        """Store the prepared RGB display image in the cache slot.

        Args:
            disp_rgb: RGB numpy array or None.
        """
        self.cached_disp_rgb = disp_rgb

    def clear_photo_cache(self) -> None:
        """Clear the internal pixmap LRU cache."""
        self._pixmap_cache.clear()



    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Trigger re-render on resize.

        Args:
            event: The QResizeEvent.
        """
        super().resizeEvent(event)
        if self.cached_disp_rgb is not None:
            self.controller._render_display()
        else:
            self.controller.load_and_render()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Forward left-button clicks to the controller.

        Args:
            event: The QMouseEvent.
        """
        self.setFocus()
        if event.button() == Qt.LeftButton:
            self.controller.handle_canvas_click(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the letterboxed image plus overlay.

        Args:
            event: The QPaintEvent.
        """
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))

        rgb = self._pending_rgb
        if rgb is None:
            # Draw error message if set
            err = getattr(self, '_err_msg', None)
            if err:
                p.setPen(QColor("red"))
                p.drawText(self.rect(), Qt.AlignCenter, err)
            p.end()
            return

        ih, iw = rgb.shape[:2]
        if iw <= 0 or ih <= 0:
            p.end()
            return

        cw, ch = self.width(), self.height()
        if cw <= 1 or ch <= 1:
            p.end()
            return

        mgr = self.controller.manager
        base_scale = min(cw / iw, ch / ih)
        zoom_factor = mgr.zoom_steps[mgr.zoom_level]
        scale = base_scale * zoom_factor

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        dx = (cw - nw) // 2 + mgr.pan_offset_x
        dy = (ch - nh) // 2 + mgr.pan_offset_y

        # Store transform for hit-testing in click handler
        self._lb_scale = scale
        self._lb_dx = dx
        self._lb_dy = dy
        self._img_w = iw
        self._img_h = ih

        # Build / fetch pixmap ------------------------------------------------
        cache_key = (mgr.current_idx, id(rgb), nw, nh, mgr.zoom_level)
        if cache_key in self._pixmap_cache:
            pix = self._pixmap_cache[cache_key]
            # Move to end (LRU hit)
            self._pixmap_cache.move_to_end(cache_key)
        else:
            # Convert numpy → QPixmap
            h, w, ch3 = rgb.shape
            bytes_per_line = w * 3
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            raw_pix = QPixmap.fromImage(img)
            # Scale — nearest for zoom > 1 (sharp pixels), smooth otherwise
            mode = (
                Qt.FastTransformation
                if zoom_factor > 1
                else Qt.SmoothTransformation
            )
            pix = raw_pix.scaled(nw, nh, Qt.IgnoreAspectRatio, mode)
            self._pixmap_cache[cache_key] = pix
            if len(self._pixmap_cache) > self._MAX_CACHE:
                self._pixmap_cache.popitem(last=False)  # evict oldest

        self._pixmap = pix
        p.drawPixmap(dx, dy, pix)

        # Overlay: centroid + line --------------------------------------------
        # Respect "Show Ref Line" switch and PCA-only visibility
        try:
            show_line = (
                self.controller.show_line_switch.isChecked()
                and mgr.active_engine == "PCA"
            )
        except Exception:
            show_line = False

        origin = self._pending_origin
        direction = self._pending_direction
        if show_line and origin is not None and direction is not None:
            ox = dx + origin[0] * scale
            oy = dy + origin[1] * scale

            # Centroid circle
            p.setPen(QPen(QColor("white"), 1))
            p.setBrush(QBrush(QColor("red")))
            p.drawEllipse(QPoint(int(ox), int(oy)), 4, 4)

            # Direction line
            dir_x, dir_y = direction
            if abs(dir_x) > 1e-5 or abs(dir_y) > 1e-5:
                extent = max(nw, nh) * 2
                p1 = QPoint(int(ox - dir_x * extent), int(oy - dir_y * extent))
                p2 = QPoint(int(ox + dir_x * extent), int(oy + dir_y * extent))
                is_manual = mgr.current_idx in mgr.per_frame_manual
                line_color = QColor("lime") if is_manual else QColor("red")
                p.setPen(QPen(line_color, 2))
                p.drawLine(p1, p2)

        # Manual click markers -----------------------------------------------
        for cx, cy in self._markers:
            p.setPen(QPen(QColor("white"), 2))
            p.setBrush(QBrush(QColor("lime")))
            p.drawEllipse(QPoint(cx, cy), 5, 5)

        p.end()
