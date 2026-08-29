"""Unit tests for shared custom widgets (RangeSlider).

Tests tactile drawing, integer coordinate conversions, handle picking,
left/right handle dragging, middle-span window translation, live floating
callout pills, and cursor state management.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent, QCursor
from PySide6.QtWidgets import QApplication

from rixs_app.ui.widgets import RangeSlider


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def slider(qapp, qtbot):
    """Instantiate a RangeSlider widget with fixed geometry for deterministic testing."""
    widget = RangeSlider()
    widget.resize(200, 36)
    widget.configure_range(0.0, 100.0)
    widget.set_values(20.0, 80.0)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def make_mouse_event(
    event_type: QMouseEvent.Type,
    pos: QPointF,
    button: Qt.MouseButton = Qt.MouseButton.NoButton,
    buttons: Qt.MouseButton = Qt.MouseButton.NoButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> QMouseEvent:
    """Construct a modern QMouseEvent without deprecation warnings."""
    return QMouseEvent(event_type, pos, pos, button, buttons, modifiers)


# ---------------------------------------------------------------------------
# Initialization & Configuration Tests
# ---------------------------------------------------------------------------

def test_rangeslider_initial_bounds(qapp):
    slider = RangeSlider()
    assert slider.min_val == 0.0
    assert slider.max_val == 1.0
    assert slider.val_left == 0.0
    assert slider.val_right == 1.0
    assert slider.minimumHeight() >= 30


def test_rangeslider_configure_range(slider):
    slider.configure_range(10.0, 50.0)
    assert slider.min_val == 10.0
    assert slider.max_val == 50.0
    assert slider.val_left == 20.0
    assert slider.val_right == 50.0


def test_rangeslider_configure_range_invalid_inverted(slider):
    slider.configure_range(50.0, 50.0)
    assert slider.min_val == 50.0
    assert slider.max_val == 51.0


def test_rangeslider_set_values_clamping(slider):
    slider.set_values(-10.0, 150.0)
    assert slider.val_left == 0.0
    assert slider.val_right == 100.0


def test_rangeslider_set_values_ordering(slider):
    slider.set_values(70.0, 30.0)
    assert slider.val_left == 70.0
    assert slider.val_right == 70.0


# ---------------------------------------------------------------------------
# Coordinate Conversion & Value Formatting Tests
# ---------------------------------------------------------------------------

def test_rangeslider_coordinate_mapping(slider):
    slider.configure_range(0.0, 100.0)
    x_min = slider._x_for_val(0.0)
    x_max = slider._x_for_val(100.0)
    x_mid = slider._x_for_val(50.0)

    assert x_min == slider._padding
    assert x_max == slider.width() - slider._padding
    assert abs(x_mid - (slider._padding + (slider.width() - 2 * slider._padding) // 2)) <= 1

    val_min = slider._val_for_x(x_min)
    val_max = slider._val_for_x(x_max)
    val_mid = slider._val_for_x(x_mid)

    assert pytest.approx(val_min, rel=1e-3) == 0.0
    assert pytest.approx(val_max, rel=1e-3) == 100.0
    assert pytest.approx(val_mid, abs=1.0) == 50.0


def test_rangeslider_format_value(slider):
    slider.configure_range(0.0, 1000.0)
    assert slider._format_value(123.456) == "123"

    slider.configure_range(0.0, 100.0)
    assert slider._format_value(12.345) == "12.3"

    slider.configure_range(0.0, 5.0)
    assert slider._format_value(1.234) == "1.23"

    slider.configure_range(0.0, 0.05)
    assert slider._format_value(0.01234) == "0.012"


# ---------------------------------------------------------------------------
# Handle Picking & Target Detection Tests
# ---------------------------------------------------------------------------

def test_rangeslider_pick_targets(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(25.0, 75.0)
    xl = slider._x_for_val(25.0)
    xr = slider._x_for_val(75.0)

    # Click exactly on left handle
    assert slider._pick_target(xl) == "left"
    # Click exactly on right handle
    assert slider._pick_target(xr) == "right"
    # Click in middle window
    mid_x = (xl + xr) // 2
    assert slider._pick_target(mid_x) == "window"
    # Click outside track (far left/right)
    assert slider._pick_target(0) is None
    assert slider._pick_target(slider.width()) is None


def test_rangeslider_pick_handle_legacy_shim(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(25.0, 75.0)
    xl = slider._x_for_val(25.0)
    xr = slider._x_for_val(75.0)
    mid_x = (xl + xr) // 2

    assert slider._pick_handle(xl) == "left"
    assert slider._pick_handle(xr) == "right"
    assert slider._pick_handle(mid_x) is None


# ---------------------------------------------------------------------------
# Interactive Left / Right Handle Dragging Tests
# ---------------------------------------------------------------------------

def test_rangeslider_drag_left_handle(slider, qtbot):
    slider.configure_range(0.0, 100.0)
    slider.set_values(20.0, 80.0)
    xl = slider._x_for_val(20.0)
    cy = slider.height() // 2

    range_spy = []
    release_spy = []
    slider.range_changed.connect(lambda l, r: range_spy.append((l, r)))
    slider.slider_released.connect(lambda l, r: release_spy.append((l, r)))

    # Press on left handle
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(xl, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "left"

    # Drag left handle to 40.0
    x_new = slider._x_for_val(40.0)
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)
    assert pytest.approx(slider.val_left, abs=1.0) == 40.0
    assert len(range_spy) >= 1

    # Release
    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)
    assert slider._active_handle is None
    assert len(release_spy) == 1


def test_rangeslider_drag_right_handle(slider, qtbot):
    slider.configure_range(0.0, 100.0)
    slider.set_values(20.0, 80.0)
    xr = slider._x_for_val(80.0)
    cy = slider.height() // 2

    command_mock = MagicMock()
    slider._command = command_mock

    # Press on right handle
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(xr, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "right"

    # Drag right handle to 60.0
    x_new = slider._x_for_val(60.0)
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)
    assert pytest.approx(slider.val_right, abs=1.0) == 60.0
    command_mock.assert_called()

    # Release
    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)
    assert slider._active_handle is None


# ---------------------------------------------------------------------------
# Middle-Span Window Translation Tests
# ---------------------------------------------------------------------------

def test_rangeslider_window_drag_preserves_span(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(20.0, 40.0)  # span width = 20.0
    span_width = slider.val_right - slider.val_left
    xl = slider._x_for_val(20.0)
    xr = slider._x_for_val(40.0)
    mid_x = (xl + xr) // 2
    cy = slider.height() // 2

    # Click in middle span
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(mid_x, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "window"
    assert slider.cursor().shape() == Qt.ClosedHandCursor

    # Drag rightwards by ~20 value units
    x_target = slider._x_for_val(50.0)
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)

    # Verify both left and right shifted together, preserving span width
    assert pytest.approx(slider.val_right - slider.val_left, abs=1e-3) == span_width
    assert slider.val_left > 20.0
    assert slider.val_right > 40.0

    # Release
    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)
    assert slider._active_handle is None


def test_rangeslider_window_drag_clamps_min_bound(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(10.0, 30.0)  # span = 20.0
    xl = slider._x_for_val(10.0)
    xr = slider._x_for_val(30.0)
    mid_x = (xl + xr) // 2
    cy = slider.height() // 2

    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(mid_x, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)

    # Drag way to the left (beyond min bound)
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(0, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)

    assert slider.val_left == 0.0
    assert pytest.approx(slider.val_right, abs=1e-3) == 20.0


def test_rangeslider_window_drag_clamps_max_bound(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(70.0, 90.0)  # span = 20.0
    xl = slider._x_for_val(70.0)
    xr = slider._x_for_val(90.0)
    mid_x = (xl + xr) // 2
    cy = slider.height() // 2

    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(mid_x, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)

    # Drag way to the right (beyond max bound)
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(slider.width() + 50, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)

    assert slider.val_right == 100.0
    assert pytest.approx(slider.val_left, abs=1e-3) == 80.0


# ---------------------------------------------------------------------------
# Hover & Cursor State Tests
# ---------------------------------------------------------------------------

def test_rangeslider_hover_cursors(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(25.0, 75.0)
    xl = slider._x_for_val(25.0)
    xr = slider._x_for_val(75.0)
    mid_x = (xl + xr) // 2
    cy = slider.height() // 2

    # Hover over left handle
    move_l = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(xl, cy),
    )
    slider.mouseMoveEvent(move_l)
    assert slider._hover_target == "left"
    assert slider.cursor().shape() == Qt.SizeHorCursor

    # Hover over middle window
    move_m = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(mid_x, cy),
    )
    slider.mouseMoveEvent(move_m)
    assert slider._hover_target == "window"
    assert slider.cursor().shape() == Qt.OpenHandCursor

    # Mouse leave resets hover
    slider.leaveEvent(None)
    assert slider._hover_target is None


# ---------------------------------------------------------------------------
# Paint Event & Visual Rendering Tests
# ---------------------------------------------------------------------------

def test_rangeslider_paint_event_does_not_crash(slider):
    slider.configure_range(0.0, 100.0)
    slider.set_values(30.0, 70.0)
    slider._hover_target = "left"
    slider.repaint()

    slider._hover_target = "window"
    slider._active_handle = "window"
    slider.repaint()

    slider._hover_target = None
    slider._active_handle = None
    slider.repaint()
