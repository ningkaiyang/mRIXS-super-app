"""Shared custom PySide6 widgets for the RIXS Super-App.

Contains the RangeSlider — a dual-handle intensity clamping slider widget
with hardware-accelerated integer QPainter rendering, live floating callouts,
and middle-span window dragging.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPen,
    QCursor,
    QLinearGradient,
    QFont,
)
from PySide6.QtWidgets import QWidget


class RangeSlider(QWidget):
    """A custom dual-handle range slider built with QPainter.

    Features:
    - Recessed track groove (4px height, #1f274a bg, #2d3561 border, 2px radius).
    - Active linear gradient span (#3b82f6 to #60a5fa).
    - Concentric tactile handles (16px outer ring, 10px inner white circle, hover halo, drop shadow).
    - Live floating value callout pills above handles during hover and drag.
    - Middle-span window dragging translating [floor, ceiling] preserving window span.
    - Pixel-crisp integer coordinate drawing for macOS Retina / High-DPI displays.

    Emits the ``range_changed`` signal whenever either handle or window is dragged,
    carrying the current (left_value, right_value) tuple as arguments.

    Args:
        parent: Parent widget.
    """

    range_changed = Signal(float, float)
    slider_released = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the RangeSlider.

        Args:
            parent: Parent QWidget.
        """
        super().__init__(parent)

        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.val_left: float = 0.0
        self.val_right: float = 1.0

        self._padding: int = 14
        self._handle_radius: int = 8       # 16px outer diameter
        self._inner_radius: int = 5        # 10px inner diameter
        self._halo_radius: int = 11        # 22px hover halo diameter
        self._track_h: int = 4             # 4px groove height

        # Palette
        self._track_bg = QColor("#1f274a")
        self._track_border = QColor("#2d3561")
        self._gradient_start = QColor("#3b82f6")
        self._gradient_end = QColor("#60a5fa")
        self._handle_idle = QColor("#3b82f6")
        self._handle_active = QColor("#60a5fa")
        self._handle_inner = QColor("#ffffff")
        self._halo_color = QColor(59, 130, 246, 64)       # 25% opacity
        self._shadow_color = QColor(0, 0, 0, 80)
        self._callout_bg = QColor("#0f172a")
        self._callout_border = QColor("#3b82f6")
        self._callout_text = QColor("#ffffff")

        self._active_handle: str | None = None  # 'left' | 'right' | 'window' | None
        self._hover_target: str | None = None   # 'left' | 'right' | 'window' | None

        self._drag_start_x: int = 0
        self._drag_start_left: float = 0.0
        self._drag_start_right: float = 0.0

        self.setMinimumHeight(32)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure_range(self, min_val: float, max_val: float) -> None:
        """Set the absolute minimum and maximum boundaries for the slider.

        Args:
            min_val: Lower bound.
            max_val: Upper bound (must be > min_val).
        """
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1.0
        self.val_left = max(self.min_val, min(self.max_val, self.val_left))
        self.val_right = max(self.min_val, min(self.max_val, self.val_right))
        if self.val_right < self.val_left:
            self.val_right = self.val_left
        self.update()

    def set_values(self, val_left: float, val_right: float) -> None:
        """Set the current left and right handle positions.

        Values are clamped to [min_val, max_val].

        Args:
            val_left: Left handle value.
            val_right: Right handle value.
        """
        self.val_left = max(self.min_val, min(self.max_val, float(val_left)))
        self.val_right = max(self.min_val, min(self.max_val, float(val_right)))
        if self.val_right < self.val_left:
            self.val_right = self.val_left
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _x_for_val(self, val: float) -> int:
        """Convert a value to an integer x-pixel coordinate on the track.

        Args:
            val: The value to convert.

        Returns:
            The x pixel coordinate (integer).
        """
        w = self.width()
        usable = max(w - 2 * self._padding, 1)
        span = max(self.max_val - self.min_val, 1e-12)
        frac = (val - self.min_val) / span
        return int(round(self._padding + frac * usable))

    def _val_for_x(self, x: int) -> float:
        """Convert an x-pixel coordinate to a slider value.

        Args:
            x: The x pixel coordinate.

        Returns:
            The corresponding slider value, clamped to [min_val, max_val].
        """
        w = self.width()
        usable = max(w - 2 * self._padding, 1)
        frac = (x - self._padding) / usable
        frac = max(0.0, min(1.0, frac))
        val = self.min_val + frac * (self.max_val - self.min_val)
        return max(self.min_val, min(self.max_val, val))

    def _format_value(self, val: float) -> str:
        """Format a slider value cleanly for the callout pill.

        Args:
            val: Slider value.

        Returns:
            Formatted string representation.
        """
        span = abs(self.max_val - self.min_val)
        if span >= 500:
            return f"{val:.0f}"
        elif span >= 50:
            return f"{val:.1f}"
        elif span >= 1:
            return f"{val:.2f}"
        elif span >= 0.01:
            return f"{val:.3f}"
        else:
            return f"{val:.4f}"

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the recessed groove, gradient span, handles, halos, and callouts.

        Args:
            event: The QPaintEvent (unused directly).
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        cy = h - 13 if h >= 30 else h // 2 + 2
        x_min = self._padding
        x_max = self.width() - self._padding
        track_y = cy - self._track_h // 2

        # 1. Recessed track groove
        p.setPen(QPen(self._track_border, 1))
        p.setBrush(QBrush(self._track_bg))
        p.drawRoundedRect(x_min, track_y, x_max - x_min, self._track_h, 2, 2)

        # 2. Active gradient span
        xl = self._x_for_val(self.val_left)
        xr = self._x_for_val(self.val_right)
        if xr > xl:
            grad = QLinearGradient(xl, cy, xr, cy)
            grad.setColorAt(0.0, self._gradient_start)
            grad.setColorAt(1.0, self._gradient_end)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(xl, track_y, xr - xl, self._track_h, 2, 2)

        # 3. Handle interaction states
        left_active = self._active_handle in ('left', 'window')
        right_active = self._active_handle in ('right', 'window')
        left_hover = (self._hover_target in ('left', 'window') or left_active)
        right_hover = (self._hover_target in ('right', 'window') or right_active)

        # Draw handles (left then right)
        self._draw_handle(p, xl, cy, is_active=left_active, is_hover=left_hover)
        self._draw_handle(p, xr, cy, is_active=right_active, is_hover=right_hover)

        # 4. Floating value callout pills
        show_left_pill = left_hover or left_active
        show_right_pill = right_hover or right_active

        pill_y_top = max(1, cy - self._handle_radius - 17)
        if show_left_pill and show_right_pill and abs(xl - xr) < 40:
            # Overlapping handles: render single combined pill or offset
            combined_text = f"{self._format_value(self.val_left)} - {self._format_value(self.val_right)}"
            mid_x = (xl + xr) // 2
            self._draw_pill_with_text(p, mid_x, pill_y_top, combined_text)
        else:
            if show_left_pill:
                self._draw_callout_pill(p, xl, pill_y_top, self.val_left)
            if show_right_pill:
                self._draw_callout_pill(p, xr, pill_y_top, self.val_right)

        p.end()

    def _draw_handle(
        self, p: QPainter, x: int, cy: int, *, is_active: bool, is_hover: bool
    ) -> None:
        """Draw concentric tactile handle with drop shadow and hover halo.

        Args:
            p: Active QPainter.
            x: Center x-coordinate.
            cy: Center y-coordinate.
            is_active: Whether handle is pressed.
            is_hover: Whether handle is hovered.
        """
        # Hover halo (22px diameter)
        if is_hover:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self._halo_color))
            p.drawEllipse(QPoint(x, cy), self._halo_radius, self._halo_radius)

        # Subtle drop shadow
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._shadow_color))
        p.drawEllipse(QPoint(x, cy + 1), self._handle_radius, self._handle_radius)

        # Outer ring (16px diameter)
        outer_color = self._handle_active if is_active else self._handle_idle
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(outer_color))
        p.drawEllipse(QPoint(x, cy), self._handle_radius, self._handle_radius)

        # Inner white knob (10px diameter)
        p.setBrush(QBrush(self._handle_inner))
        p.drawEllipse(QPoint(x, cy), self._inner_radius, self._inner_radius)

    def _draw_callout_pill(
        self, p: QPainter, x: int, y_top: int, val: float
    ) -> None:
        """Draw a floating value callout pill above a handle.

        Args:
            p: Active QPainter.
            x: Center x-coordinate of the handle knob.
            y_top: Top y-coordinate for the pill.
            val: Numeric value to format.
        """
        self._draw_pill_with_text(p, x, y_top, self._format_value(val))

    def _draw_pill_with_text(
        self, p: QPainter, x: int, y_top: int, text: str
    ) -> None:
        """Draw a styled floating pill callout containing text.

        Args:
            p: Active QPainter.
            x: Center x-coordinate.
            y_top: Top y-coordinate.
            text: Text string to display.
        """
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        p.setFont(font)

        fm = p.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()

        pill_w = text_w + 10
        pill_h = text_h + 2
        pill_x = int(round(x - pill_w / 2))
        pill_x = max(2, min(self.width() - pill_w - 2, pill_x))
        pill_y = int(y_top)

        p.setPen(QPen(self._callout_border, 1))
        p.setBrush(QBrush(self._callout_bg))
        p.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 4, 4)

        p.setPen(QPen(self._callout_text))
        p.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, text)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _pick_target(self, x: int) -> str | None:
        """Determine which element (left handle, right handle, or window) is at x.

        Args:
            x: Mouse x-pixel position.

        Returns:
            'left', 'right', 'window', or None.
        """
        xl = self._x_for_val(self.val_left)
        xr = self._x_for_val(self.val_right)
        dl = abs(x - xl)
        dr = abs(x - xr)
        threshold = self._handle_radius + 4  # 12px grab tolerance

        if dl <= threshold and dr <= threshold:
            if dl < dr:
                return 'left'
            elif dr < dl:
                return 'right'
            else:
                return 'left' if x <= xl else 'right'

        if dl <= threshold:
            return 'left'
        if dr <= threshold:
            return 'right'

        min_x = min(xl, xr)
        max_x = max(xl, xr)
        if min_x <= x <= max_x:
            return 'window'

        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Detect handle or window click and initiate drag.

        Args:
            event: The QMouseEvent.
        """
        if event.button() == Qt.LeftButton:
            self.setFocus()
            x = event.position().toPoint().x()
            target = self._pick_target(x)
            self._active_handle = target
            self._drag_start_x = x
            self._drag_start_left = self.val_left
            self._drag_start_right = self.val_right

            if target == 'window':
                self.setCursor(QCursor(Qt.ClosedHandCursor))
            elif target in ('left', 'right'):
                self.setCursor(QCursor(Qt.SizeHorCursor))
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Update cursor feedback and translate handles / middle window.

        Args:
            event: The QMouseEvent.
        """
        x = event.position().toPoint().x()

        if self._active_handle is None:
            target = self._pick_target(x)
            if target != self._hover_target:
                self._hover_target = target
                self.update()
            if target in ('left', 'right'):
                self.setCursor(QCursor(Qt.SizeHorCursor))
            elif target == 'window':
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                self.unsetCursor()
            return

        if self._active_handle == 'left':
            val = self._val_for_x(x)
            self.val_left = min(val, self.val_right)
            self.update()
            self.range_changed.emit(self.val_left, self.val_right)

        elif self._active_handle == 'right':
            val = self._val_for_x(x)
            self.val_right = max(val, self.val_left)
            self.update()
            self.range_changed.emit(self.val_left, self.val_right)

        elif self._active_handle == 'window':
            usable = max(self.width() - 2 * self._padding, 1)
            delta_x = x - self._drag_start_x
            delta_val = delta_x / usable * (self.max_val - self.min_val)
            window_span = self._drag_start_right - self._drag_start_left

            new_left = self._drag_start_left + delta_val
            new_right = self._drag_start_right + delta_val

            if new_left < self.min_val:
                new_left = self.min_val
                new_right = self.min_val + window_span
            elif new_right > self.max_val:
                new_right = self.max_val
                new_left = self.max_val - window_span

            self.val_left = max(self.min_val, min(self.max_val, new_left))
            self.val_right = max(self.min_val, min(self.max_val, new_right))
            self.update()
            self.range_changed.emit(self.val_left, self.val_right)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """End handle/window dragging and emit slider_released signal.

        Args:
            event: The QMouseEvent.
        """
        was_active = self._active_handle is not None
        self._active_handle = None
        target = self._pick_target(event.position().toPoint().x())
        self._hover_target = target
        if target in ('left', 'right'):
            self.setCursor(QCursor(Qt.SizeHorCursor))
        elif target == 'window':
            self.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            self.unsetCursor()
        self.update()
        if was_active:
            self.slider_released.emit(self.val_left, self.val_right)

    def leaveEvent(self, event) -> None:  # noqa: N802
        """Reset hover state when mouse exits the widget.

        Args:
            event: The QEvent.
        """
        if self._active_handle is None:
            self._hover_target = None
            self.unsetCursor()
            self.update()


try:
    import shiboken6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class SafeFigureCanvasQTAgg(FigureCanvasQTAgg):
        """FigureCanvasQTAgg subclass guarding against deferred draw_idle and paint on deleted C++ Qt objects."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._destroyed = False
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        def draw_idle(self) -> None:
            try:
                if getattr(self, "_destroyed", False) or not shiboken6.isValid(self):
                    return
                if not self.isVisible() or self.isHidden() or self.height() <= 0 or self.width() <= 0:
                    return
                super().draw_idle()
            except Exception:
                pass

        def draw(self) -> None:
            try:
                if getattr(self, "_destroyed", False) or not shiboken6.isValid(self):
                    return
                if self.height() <= 0 or self.width() <= 0 or not self.isVisible() or self.isHidden():
                    return
                super().draw()
            except Exception:
                pass

        def paintEvent(self, event) -> None:  # noqa: N802
            try:
                if getattr(self, "_destroyed", False) or not shiboken6.isValid(self):
                    return
                if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
                    return
                super().paintEvent(event)
            except Exception:
                pass

        def resizeEvent(self, event) -> None:  # noqa: N802
            try:
                if getattr(self, "_destroyed", False) or not shiboken6.isValid(self):
                    return
                if self.width() <= 0 or self.height() <= 0:
                    return
                super().resizeEvent(event)
            except Exception:
                pass

        def showEvent(self, event) -> None:  # noqa: N802
            self._destroyed = False
            super().showEvent(event)
            self.draw_idle()

        def hideEvent(self, event) -> None:  # noqa: N802
            super().hideEvent(event)

        def cleanup(self) -> None:
            self._destroyed = True
            try:
                if hasattr(self, "figure") and self.figure is not None:
                    self.figure.clear()
            except Exception:
                pass

        def closeEvent(self, event) -> None:  # noqa: N802
            self.cleanup()
            super().closeEvent(event)

        def destroy(self, destroyWindow=True, destroySubWindows=True) -> None:  # noqa: N802
            self.cleanup()
            super().destroy(destroyWindow, destroySubWindows)
except ImportError:
    SafeFigureCanvasQTAgg = None  # type: ignore
