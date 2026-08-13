"""Shared custom PySide6 widgets for the mRIXS Super-App.

Contains the RangeSlider — a dual-handle intensity clamping slider widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QCursor
from PySide6.QtWidgets import QWidget


class RangeSlider(QWidget):
    """A custom dual-handle range slider built with QPainter.

    Emits the ``range_changed`` signal whenever either handle is dragged,
    carrying the current (left_value, right_value) tuple as arguments.
    Also invokes an optional *command* callable with the same signature for
    backwards-compatibility with code that passed ``command=`` on construction.

    Args:
        parent: Parent widget.
        command: Optional callable called with (left_val, right_val) on change.
    """

    range_changed = Signal(float, float)
    slider_released = Signal(float, float)

    def __init__(self, parent=None, *, command=None, **kwargs):
        """Initialise the RangeSlider.

        Args:
            parent: Parent QWidget.
            command: Optional callable invoked as command(left, right) on change.
            **kwargs: Absorbed silently (height= etc. from Tkinter callers).
        """
        super().__init__(parent)
        self._command = command

        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.val_left: float = 0.0
        self.val_right: float = 1.0

        self._padding: int = 12
        self._handle_radius: int = 8
        self._track_h: int = 6

        # Colors
        self._track_bg = QColor("#2d3561")
        self._track_active = QColor("#2196f3")
        self._handle_idle = QColor("#c5cae9")
        self._handle_active = QColor("#2196f3")

        self._active_handle: str | None = None  # 'left' | 'right' | None
        self.setMinimumHeight(28)
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
        """Convert a value to an x-pixel coordinate on the track.

        Args:
            val: The value to convert.

        Returns:
            The x pixel coordinate (integer).
        """
        w = self.width()
        usable = w - 2 * self._padding
        frac = (val - self.min_val) / max(self.max_val - self.min_val, 1e-12)
        return int(self._padding + frac * usable)

    def _val_for_x(self, x: int) -> float:
        """Convert an x-pixel coordinate to a slider value.

        Args:
            x: The x pixel coordinate.

        Returns:
            The corresponding slider value, clamped to [min_val, max_val].
        """
        w = self.width()
        usable = w - 2 * self._padding
        frac = (x - self._padding) / max(usable, 1)
        val = self.min_val + frac * (self.max_val - self.min_val)
        return max(self.min_val, min(self.max_val, val))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the track and two handles.

        Args:
            event: The QPaintEvent (unused directly).
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        cy = self.height() // 2
        x_min = self._padding
        x_max = self.width() - self._padding

        # Background track
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._track_bg))
        p.drawRoundedRect(x_min, cy - self._track_h // 2,
                          x_max - x_min, self._track_h, 3, 3)

        # Active range
        xl = self._x_for_val(self.val_left)
        xr = self._x_for_val(self.val_right)
        p.setBrush(QBrush(self._track_active))
        p.drawRoundedRect(xl, cy - self._track_h // 2,
                          xr - xl, self._track_h, 3, 3)

        # Left handle
        c = self._handle_active if self._active_handle == 'left' else self._handle_idle
        p.setBrush(QBrush(c))
        p.setPen(QPen(QColor("#555"), 1))
        p.drawEllipse(QPoint(xl, cy), self._handle_radius, self._handle_radius)

        # Right handle
        c = self._handle_active if self._active_handle == 'right' else self._handle_idle
        p.setBrush(QBrush(c))
        p.drawEllipse(QPoint(xr, cy), self._handle_radius, self._handle_radius)

        p.end()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _pick_handle(self, x: int) -> str | None:
        """Determine which handle (if any) is closest to x within grab radius.

        Args:
            x: Mouse x-pixel position.

        Returns:
            'left', 'right', or None.
        """
        xl = self._x_for_val(self.val_left)
        xr = self._x_for_val(self.val_right)
        dl = abs(x - xl)
        dr = abs(x - xr)
        threshold = self._handle_radius + 4
        if dl < threshold or dr < threshold:
            if dl < dr:
                return 'left'
            elif dr < dl:
                return 'right'
            else:
                return 'left' if x < xl else 'right'
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Detect which handle was clicked and activate it.

        Args:
            event: The QMouseEvent.
        """
        if event.button() == Qt.LeftButton:
            self._active_handle = self._pick_handle(event.position().toPoint().x())
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Update cursor and drag the active handle.

        Args:
            event: The QMouseEvent.
        """
        x = event.position().toPoint().x()
        # Cursor feedback
        if self._pick_handle(x) is not None:
            self.setCursor(QCursor(Qt.SizeHorCursor))
        else:
            self.unsetCursor()

        if self._active_handle is None:
            return

        val = self._val_for_x(x)
        if self._active_handle == 'left':
            self.val_left = min(val, self.val_right - 1e-9)
        else:
            self.val_right = max(val, self.val_left + 1e-9)
        self.update()
        self.range_changed.emit(self.val_left, self.val_right)
        if self._command:
            self._command(self.val_left, self.val_right)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Deactivate the dragged handle and emit release signal.

        Args:
            event: The QMouseEvent.
        """
        was_active = self._active_handle is not None
        self._active_handle = None
        self.update()
        if was_active:
            self.slider_released.emit(self.val_left, self.val_right)
