"""Unit and integration tests for Milestone 4: Alignment Slideshow enhancements.

Tests the Drift Stability Pill Badge, Autoplay Speed Cycling, and
Dynamic Sequence Boundary Disabling in the navigation bar.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from unittest.mock import patch, MagicMock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from rixs_app.ui.alignment_slideshow.slideshow_view import SlideshowView


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app


@pytest.fixture
def synthetic_tifs(tmp_path):
    """Generate 4 small synthetic TIFF frames."""
    paths = []
    for i in range(4):
        p = tmp_path / f"frame_{i + 1}.tif"
        data = np.zeros((64, 64), dtype=np.float32)
        data[:, 30 + i] = 100.0
        tifffile.imwrite(str(p), data)
        paths.append(str(p))
    return paths


@pytest.fixture
def slideshow(qapp, synthetic_tifs, qtbot):
    """Instantiate SlideshowView with mocked dataset for fast, synchronous UI tests."""
    with patch("rixs_app.ui.alignment_slideshow.alignment_manager.SequenceManager") as mock_ds:
        mock_instance = MagicMock()
        mock_instance.get_frame.return_value = np.zeros((64, 64), dtype=np.float32)
        mock_ds.return_value = mock_instance

        view = SlideshowView()
        qtbot.addWidget(view)
        view.start(synthetic_tifs)
        yield view
        view.stop_autoplay()
        view.close()


# ---------------------------------------------------------------------------
# Drift Stability Pill Badge Tests
# ---------------------------------------------------------------------------

def test_drift_pill_stable_state(slideshow):
    """Shift magnitude <= 1.0 px displays 'Stable ✓' with emerald styling."""
    slideshow._update_drift_pill(0.2)
    assert slideshow.drift_pill.text() == "Stable \u2713"
    style = slideshow.drift_pill.styleSheet()
    assert "#065f46" in style
    assert "#34d399" in style
    assert "#059669" in style

    slideshow._update_drift_pill(1.0)
    assert slideshow.drift_pill.text() == "Stable \u2713"


def test_drift_pill_moderate_drift_state(slideshow):
    """Shift magnitude 1.0 < d <= 5.0 px displays 'Moderate Drift' with amber styling."""
    slideshow._update_drift_pill(1.01)
    assert slideshow.drift_pill.text() == "Moderate Drift"
    style = slideshow.drift_pill.styleSheet()
    assert "#78350f" in style
    assert "#fbbf24" in style
    assert "#d97706" in style

    slideshow._update_drift_pill(5.0)
    assert slideshow.drift_pill.text() == "Moderate Drift"


def test_drift_pill_severe_drift_state(slideshow):
    """Shift magnitude > 5.0 px displays 'Severe Drift ⚠️' with rose/red styling."""
    slideshow._update_drift_pill(5.01)
    assert "Severe Drift" in slideshow.drift_pill.text()
    style = slideshow.drift_pill.styleSheet()
    assert "#881337" in style
    assert "#f87171" in style
    assert "#e11d48" in style


def test_drift_pill_and_canvas_panel_canonical(slideshow):
    """Verify canonical drift_pill and canvas_panel are present."""
    assert slideshow.drift_pill is not None
    assert slideshow.canvas_panel is not None


def test_drift_pill_updates_on_frame_offset(slideshow):
    """load_and_render dynamically updates drift pill based on computed offset."""
    with patch.object(slideshow.manager, "get_offset", return_value=(4.0, 4.0)):
        slideshow.current_idx = 1
        slideshow.load_and_render()
        # sqrt(4^2 + 4^2) = sqrt(32) ≈ 5.657 > 5.0 -> Severe Drift
        assert "Severe Drift" in slideshow.drift_pill.text()

    with patch.object(slideshow.manager, "get_offset", return_value=(0.5, 0.5)):
        slideshow.current_idx = 2
        slideshow.load_and_render()
        # sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707 <= 1.0 -> Stable
        assert slideshow.drift_pill.text() == "Stable \u2713"


# ---------------------------------------------------------------------------
# Autoplay Speed Cycling Tests
# ---------------------------------------------------------------------------

def test_autoplay_speed_cycling_sequence(slideshow, qtbot):
    """Clicking speed button cycles 1× -> 2× -> 5× -> 0.5× -> 1×."""
    navbar = slideshow.navbar
    speed_btn = navbar.speed_button

    assert speed_btn.text() == "1×"
    assert slideshow.manager.autoplay_speed_ms == 500

    # 1st Click -> 2×
    qtbot.mouseClick(speed_btn, Qt.LeftButton)
    assert speed_btn.text() == "2×"
    assert slideshow.manager.autoplay_speed_ms == 250

    # 2nd Click -> 5×
    qtbot.mouseClick(speed_btn, Qt.LeftButton)
    assert speed_btn.text() == "5×"
    assert slideshow.manager.autoplay_speed_ms == 100

    # 3rd Click -> 0.5×
    qtbot.mouseClick(speed_btn, Qt.LeftButton)
    assert speed_btn.text() == "0.5×"
    assert slideshow.manager.autoplay_speed_ms == 1000

    # 4th Click -> 1× (loops back)
    qtbot.mouseClick(speed_btn, Qt.LeftButton)
    assert speed_btn.text() == "1×"
    assert slideshow.manager.autoplay_speed_ms == 500


def test_autoplay_speed_set_speed_explicit(slideshow):
    """set_speed method accepts string labels or millisecond integers."""
    navbar = slideshow.navbar

    navbar.set_speed("5×")
    assert navbar.speed_button.text() == "5×"
    assert slideshow.manager.autoplay_speed_ms == 100

    navbar.set_speed(250)
    assert navbar.speed_button.text() == "2×"
    assert slideshow.manager.autoplay_speed_ms == 250


def test_autoplay_speed_change_during_active_playback(slideshow, qtbot):
    """Changing speed while autoplay is active updates the running QTimer interval immediately."""
    slideshow.start_autoplay()
    assert slideshow._autoplay_timer is not None
    assert slideshow._autoplay_timer.isActive()
    assert slideshow._autoplay_timer.interval() == 500

    qtbot.mouseClick(slideshow.navbar.speed_button, Qt.LeftButton)
    assert slideshow.navbar.speed_button.text() == "2×"
    assert slideshow._autoplay_timer.interval() == 250

    slideshow.stop_autoplay()
    assert not slideshow.manager.autoplay_active


# ---------------------------------------------------------------------------
# Sequence Boundary Disabling Tests
# ---------------------------------------------------------------------------

def test_sequence_boundary_disabling_at_index_zero(slideshow):
    """At index 0: prev_button and first_button are disabled; next and last are enabled."""
    slideshow.jump_to_frame(0)
    navbar = slideshow.navbar

    assert not navbar.prev_button.isEnabled()
    assert not navbar.first_button.isEnabled()
    assert navbar.next_button.isEnabled()
    assert navbar.last_button.isEnabled()
    assert navbar.autoplay_button.isEnabled()
    assert navbar.speed_button.isEnabled()


def test_sequence_boundary_disabling_at_middle_index(slideshow):
    """At middle index: prev, first, next, last are all enabled."""
    slideshow.jump_to_frame(1)
    navbar = slideshow.navbar

    assert navbar.prev_button.isEnabled()
    assert navbar.first_button.isEnabled()
    assert navbar.next_button.isEnabled()
    assert navbar.last_button.isEnabled()


def test_sequence_boundary_disabling_at_last_index(slideshow):
    """At last index (N-1): next_button and last_button are disabled; prev and first are enabled."""
    n = len(slideshow.manager.file_list)
    slideshow.jump_to_frame(n - 1)
    navbar = slideshow.navbar

    assert navbar.prev_button.isEnabled()
    assert navbar.first_button.isEnabled()
    assert not navbar.next_button.isEnabled()
    assert not navbar.last_button.isEnabled()


def test_sequence_boundary_single_frame(qapp, qtbot):
    """When only 1 frame is loaded, all navigation and autoplay buttons are disabled."""
    view = SlideshowView()
    qtbot.addWidget(view)
    view.start([])  # empty
    navbar = view.navbar

    assert not navbar.prev_button.isEnabled()
    assert not navbar.next_button.isEnabled()
    assert not navbar.first_button.isEnabled()
    assert not navbar.last_button.isEnabled()
    assert not navbar.autoplay_button.isEnabled()
    assert not navbar.speed_button.isEnabled()
    view.close()


# ---------------------------------------------------------------------------
# First and Last Frame Navigation Jump Tests
# ---------------------------------------------------------------------------

def test_first_and_last_button_actions(slideshow, qtbot):
    """Clicking last_button jumps to N-1; clicking first_button jumps to 0."""
    navbar = slideshow.navbar
    n = len(slideshow.manager.file_list)

    assert slideshow.current_idx == 0
    qtbot.mouseClick(navbar.last_button, Qt.LeftButton)
    assert slideshow.current_idx == n - 1

    qtbot.mouseClick(navbar.first_button, Qt.LeftButton)
    assert slideshow.current_idx == 0


# ---------------------------------------------------------------------------
# UI Disabling During Export
# ---------------------------------------------------------------------------

def test_export_ui_state_disables_navbar_controls(slideshow):
    """_set_export_ui_state toggles all navbar controls including first/last/speed."""
    navbar = slideshow.navbar

    slideshow._set_export_ui_state(False)
    assert not navbar.first_button.isEnabled()
    assert not navbar.last_button.isEnabled()
    assert not navbar.speed_button.isEnabled()
    assert not navbar.autoplay_button.isEnabled()
    assert not navbar.back_button.isEnabled()

    slideshow._set_export_ui_state(True)
    assert navbar.first_button.isEnabled()
    assert navbar.last_button.isEnabled()
    assert navbar.speed_button.isEnabled()
    assert navbar.autoplay_button.isEnabled()
    assert navbar.back_button.isEnabled()
