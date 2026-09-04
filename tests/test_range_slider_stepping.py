"""Unit tests for discrete step support in RangeSlider (TASK-03).

Tests integer and float stepping, snapping during handle and window dragging,
integer callout pill formatting, and boundary preservation.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from rixs_app.ui.widgets import RangeSlider


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication for offscreen widget testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def stepped_slider(qapp, qtbot):
    """Instantiate a RangeSlider configured with integer stepping (1 to 20, step=1.0)."""
    widget = RangeSlider(step=1.0)
    widget.resize(200, 36)
    widget.configure_range(1.0, 20.0, step=1.0)
    widget.set_values(1.0, 9.0)
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
# Initialization & Configuration
# ---------------------------------------------------------------------------

def test_rangeslider_init_with_step(qapp):
    """Verify __init__ accepts optional step parameter."""
    s_default = RangeSlider()
    assert s_default.step is None

    s_stepped = RangeSlider(step=1.0)
    assert s_stepped.step == 1.0

    s_float = RangeSlider(step=0.5)
    assert s_float.step == 0.5


def test_rangeslider_configure_range_step(stepped_slider):
    """Verify configure_range can set or update step."""
    slider = stepped_slider
    assert slider.step == 1.0

    # configure_range without step keeps existing step
    slider.configure_range(1.0, 50.0)
    assert slider.step == 1.0
    assert slider.min_val == 1.0
    assert slider.max_val == 50.0

    # configure_range with new step updates it
    slider.configure_range(0.0, 10.0, step=2.0)
    assert slider.step == 2.0

    # configure_range with step=None clears stepping
    slider.configure_range(0.0, 10.0, step=None)
    assert slider.step is None


def test_rangeslider_set_step_method(stepped_slider):
    """Verify set_step method updates step and snaps current values."""
    slider = stepped_slider
    slider.configure_range(0.0, 10.0, step=None)
    slider.set_values(2.3, 7.7)
    assert slider.val_left == pytest.approx(2.3)
    assert slider.val_right == pytest.approx(7.7)

    slider.set_step(1.0)
    assert slider.step == 1.0
    assert slider.val_left == 2.0
    assert slider.val_right == 8.0

    slider.set_step(None)
    assert slider.step is None


# ---------------------------------------------------------------------------
# Value Snapping via set_values
# ---------------------------------------------------------------------------

def test_rangeslider_set_values_snapping(stepped_slider):
    """Verify set_values snaps inputs to nearest step multiple from min_val."""
    slider = stepped_slider  # min=1.0, max=20.0, step=1.0
    slider.set_values(2.3, 7.8)
    assert slider.val_left == 2.0
    assert slider.val_right == 8.0

    slider.set_values(1.4, 9.6)
    assert slider.val_left == 1.0
    assert slider.val_right == 10.0

    # Clamp boundaries
    slider.set_values(-5.0, 25.0)
    assert slider.val_left == 1.0
    assert slider.val_right == 20.0


def test_rangeslider_val_for_x_snapping(stepped_slider):
    """Verify _val_for_x snaps pixel positions to step boundaries."""
    slider = stepped_slider  # min=1.0, max=20.0, step=1.0
    # Collect values for all x across the track
    values = set()
    for x in range(slider._padding, slider.width() - slider._padding + 1):
        val = slider._val_for_x(x)
        values.add(val)
        # Every returned value must be an exact integer between 1 and 20
        assert val == round(val)
        assert 1.0 <= val <= 20.0

    # There should be discrete values across the range
    assert len(values) > 1
    assert 1.0 in values
    assert 20.0 in values


# ---------------------------------------------------------------------------
# Interactive Dragging with Step Snapping
# ---------------------------------------------------------------------------

def test_rangeslider_drag_left_handle_snaps(stepped_slider):
    """Verify dragging the left handle snaps to discrete steps."""
    slider = stepped_slider  # min=1.0, max=20.0, step=1.0, [1.0, 9.0]
    xl = slider._x_for_val(1.0)
    cy = slider.height() // 2

    range_emissions = []
    slider.range_changed.connect(lambda l, r: range_emissions.append((l, r)))

    # Press left handle
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(xl, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "left"

    # Move to approximate position of 4.3 (should snap to 4.0)
    x_target = slider._padding + int(round((4.3 - 1.0) / 19.0 * (slider.width() - 2 * slider._padding)))
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)
    assert slider.val_left == 4.0
    assert slider.val_left == round(slider.val_left)

    # Release
    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)
    assert slider._active_handle is None

    # All emitted left values must be integers
    for l_val, r_val in range_emissions:
        assert l_val == round(l_val)
        assert r_val == round(r_val)


def test_rangeslider_drag_right_handle_snaps(stepped_slider):
    """Verify dragging the right handle snaps to discrete steps."""
    slider = stepped_slider  # min=1.0, max=20.0, step=1.0, [1.0, 9.0]
    xr = slider._x_for_val(9.0)
    cy = slider.height() // 2

    # Press right handle
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(xr, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "right"

    # Move to approximate position of 14.7 (should snap to 15.0)
    x_target = slider._padding + int(round((14.7 - 1.0) / 19.0 * (slider.width() - 2 * slider._padding)))
    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)
    assert slider.val_right == 15.0
    assert slider.val_right == round(slider.val_right)

    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_target, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)


# ---------------------------------------------------------------------------
# Middle-Span Window Dragging with Step Snapping
# ---------------------------------------------------------------------------

def test_rangeslider_window_drag_snaps_steps(stepped_slider):
    """Verify dragging the middle window shifts by discrete step increments and preserves span."""
    slider = stepped_slider
    slider.set_values(2.0, 6.0)  # span = 4.0
    initial_span = slider.val_right - slider.val_left
    assert initial_span == 4.0

    xl = slider._x_for_val(2.0)
    xr = slider._x_for_val(6.0)
    mid_x = (xl + xr) // 2
    cy = slider.height() // 2

    # Press in middle window
    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(mid_x, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)
    assert slider._active_handle == "window"

    # Drag rightward by an amount corresponding to ~3.2 units -> should snap to 3.0 delta
    usable = slider.width() - 2 * slider._padding
    delta_x = int(round(3.2 / 19.0 * usable))
    x_new = mid_x + delta_x

    move_ev = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_ev)

    # Shifted by 3.0 units: [2.0 + 3.0, 6.0 + 3.0] = [5.0, 9.0]
    assert slider.val_left == 5.0
    assert slider.val_right == 9.0
    assert slider.val_right - slider.val_left == initial_span

    # Release
    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(x_new, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)


def test_rangeslider_window_drag_clamps_at_bounds_with_step(stepped_slider):
    """Verify window dragging snaps and clamps at bounds while preserving span."""
    slider = stepped_slider
    slider.set_values(3.0, 7.0)  # span = 4.0
    cy = slider.height() // 2
    mid_x = (slider._x_for_val(3.0) + slider._x_for_val(7.0)) // 2

    press_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(mid_x, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mousePressEvent(press_ev)

    # Drag far right past max_val
    move_r = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(slider.width() + 100, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_r)
    assert slider.val_right == 20.0
    assert slider.val_left == 16.0
    assert slider.val_right - slider.val_left == 4.0

    # Drag far left past min_val
    move_l = make_mouse_event(
        QMouseEvent.Type.MouseMove,
        QPointF(-100, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    slider.mouseMoveEvent(move_l)
    assert slider.val_left == 1.0
    assert slider.val_right == 5.0
    assert slider.val_right - slider.val_left == 4.0

    rel_ev = make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(0, cy),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    slider.mouseReleaseEvent(rel_ev)


# ---------------------------------------------------------------------------
# Callout Pill Value Formatting
# ---------------------------------------------------------------------------

def test_rangeslider_format_value_integer_step(stepped_slider):
    """Verify _format_value formats as clean integer strings when step is integer."""
    slider = stepped_slider  # step=1.0
    assert slider._format_value(1.0) == "1"
    assert slider._format_value(9.0) == "9"
    assert slider._format_value(20.0) == "20"
    assert slider._format_value(14.0) == "14"

    # step=2.0
    slider.set_step(2.0)
    assert slider._format_value(2.0) == "2"
    assert slider._format_value(10.0) == "10"


def test_rangeslider_format_value_continuous_step(stepped_slider):
    """Verify _format_value preserves decimal formatting when step is None or fractional."""
    slider = stepped_slider
    slider.set_step(None)
    slider.configure_range(1.0, 20.0)
    # span is 19.0, span >= 1 -> 2 decimal places
    assert slider._format_value(1.234) == "1.23"

    # fractional step e.g. 0.5
    slider.set_step(0.5)
    # int(0.5) != 0.5, so falls back to standard float formatting
    assert slider._format_value(1.5) == "1.50"


# ---------------------------------------------------------------------------
# Repaint / Visual Rendering with Step
# ---------------------------------------------------------------------------

def test_rangeslider_paint_event_with_step(stepped_slider):
    """Verify paintEvent executes cleanly with step configured and callouts visible."""
    slider = stepped_slider
    slider.set_values(1.0, 9.0)
    slider._hover_target = "left"
    slider.repaint()

    slider._hover_target = "right"
    slider.repaint()

    slider._hover_target = "window"
    slider._active_handle = "window"
    slider.repaint()

    slider._hover_target = None
    slider._active_handle = None
    slider.repaint()
