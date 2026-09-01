"""Unit tests for responsive splitter, stacked widget, sidebar sizing, and tooltip states.

Tests cover:
1. RixsStackedWidget sizeHint and minimumSizeHint calculation based on active page.
2. RixsSplitter bounded minimumSizeHint and non-collapsible children.
3. Main window dimension stability across sidebar toggle lifecycle.
4. Sidebar toggle dimension stability across all 8 application views.
5. Co-Pilot toggle button tooltip transitions ("Open..." -> "Close..." -> "Open...").
6. AgentSidebarWidget toolbar responsiveness and model_combo configuration.
7. apply_dark_palette integration in main window.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QSizePolicy,
    QComboBox,
    QStackedWidget,
    QSplitter,
)

from rixs_app.main import (
    RixsApp,
    RixsStackedWidget,
    RixsSplitter,
    _IDX_HOME,
    _IDX_DARK_CAL,
    _IDX_CLUSTERING_FILES,
    _IDX_CLUSTERING_STUDIO,
    _IDX_SORTING,
    _IDX_SLIDESHOW,
    _IDX_COMPARISON,
    _IDX_ZEROTH_ORDER,
)
from rixs_app.ui.agent_sidebar.sidebar_widget import AgentSidebarWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


# ===========================================================================
# 1. RixsStackedWidget & RixsSplitter Component Tests
# ===========================================================================

def test_rixs_stacked_widget_size_hints(qapp):
    """Verify RixsStackedWidget sizeHint and minimumSizeHint adapt to currentWidget."""
    stack = RixsStackedWidget()
    assert stack.minimumSizeHint() == QSize(400, 300)
    assert stack.sizeHint() == QSize(800, 600)

    w1 = QWidget()
    w1.setMinimumSize(500, 400)
    stack.addWidget(w1)
    stack.setCurrentWidget(w1)

    min_hint = stack.minimumSizeHint()
    assert min_hint.width() <= 800
    assert min_hint.height() <= 600
    assert min_hint.width() >= 400
    assert min_hint.height() >= 300

    w2 = QWidget()
    w2.setMinimumSize(1200, 1000)
    stack.addWidget(w2)
    stack.setCurrentWidget(w2)

    min_hint2 = stack.minimumSizeHint()
    assert min_hint2.width() <= 800
    assert min_hint2.height() <= 600


def test_rixs_splitter_properties(qapp):
    """Verify RixsSplitter bounded minimum size hint and non-collapsible configuration."""
    splitter = RixsSplitter(Qt.Orientation.Horizontal)
    assert splitter.minimumSizeHint() == QSize(600, 400)
    assert not splitter.childrenCollapsible()


# ===========================================================================
# 2. Window Dimension Stability & Sidebar Toggle
# ===========================================================================

def test_sidebar_toggle_preserves_main_window_geometry(qapp, qtbot):
    """Opening and closing the sidebar must not expand or modify win.width() or win.height()."""
    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)
    app_win.resize(1200, 800)
    app_win.show()
    qapp.processEvents()

    # Mock the sidebar widget so we don't need real API key / auth
    mock_sidebar = QWidget()
    mock_sidebar.setMinimumWidth(280)
    app_win._sidebar = mock_sidebar
    app_win._splitter.addWidget(mock_sidebar)
    mock_sidebar.hide()

    initial_size = app_win.size()

    # Open sidebar
    app_win._show_sidebar()
    qapp.processEvents()
    assert app_win.size() == initial_size
    assert app_win.width() == initial_size.width()
    assert app_win.height() == initial_size.height()

    # Close sidebar
    app_win._hide_sidebar()
    qapp.processEvents()
    assert app_win.size() == initial_size
    assert app_win.width() == initial_size.width()
    assert app_win.height() == initial_size.height()


def test_sidebar_toggle_across_all_views(qapp, qtbot):
    """Cycling through all primary views and toggling the sidebar keeps window dimensions stable."""
    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)
    app_win.resize(1200, 800)
    app_win.show()
    qapp.processEvents()

    mock_sidebar = QWidget()
    mock_sidebar.setMinimumWidth(280)
    app_win._sidebar = mock_sidebar
    app_win._splitter.addWidget(mock_sidebar)
    mock_sidebar.hide()

    import numpy as np

    dummy_img = np.zeros((10, 10), dtype=np.float32)
    view_switches = [
        ("Home", app_win.show_home, _IDX_HOME),
        ("Dark Mask", app_win.show_dark_calibration, _IDX_DARK_CAL),
        ("Clustering Files", app_win.show_clustering_files, _IDX_CLUSTERING_FILES),
        ("Clustering Studio", lambda: app_win.show_clustering_studio(), _IDX_CLUSTERING_STUDIO),
        ("Sorting", app_win.show_sorting, _IDX_SORTING),
        ("Slideshow", lambda: app_win.show_slideshow([]), _IDX_SLIDESHOW),
        ("Comparison", lambda: app_win.show_export_comparison(dummy_img, dummy_img, ""), _IDX_COMPARISON),
        ("Zeroth Order", lambda: app_win.show_zeroth_order_calibration([]), _IDX_ZEROTH_ORDER),
    ]

    target_size = app_win.size()

    for name, switch_fn, expected_idx in view_switches:
        switch_fn()
        qapp.processEvents()
        assert app_win._stack.currentIndex() == expected_idx, f"Failed navigating to {name}"

        # Toggle open
        app_win._show_sidebar()
        qapp.processEvents()
        assert app_win.size() == target_size, f"Size changed on open in view {name}: {app_win.size()} vs {target_size}"

        # Toggle close
        app_win._hide_sidebar()
        qapp.processEvents()
        assert app_win.size() == target_size, f"Size changed on close in view {name}: {app_win.size()} vs {target_size}"


# ===========================================================================
# 3. Tooltip States
# ===========================================================================

def test_copilot_button_tooltip_states(qapp, qtbot):
    """Tooltip starts as 'Open...', transitions to 'Close...', and reverts to 'Open...'."""
    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)

    mock_sidebar = QWidget()
    app_win._sidebar = mock_sidebar
    app_win._splitter.addWidget(mock_sidebar)
    mock_sidebar.hide()

    # Initial tooltip
    assert app_win._sidebar_toggle.toolTip() == "Open RIXS Co-Pilot Agentic AI Side Panel"

    # Open sidebar
    app_win._show_sidebar()
    assert app_win._sidebar_toggle.toolTip() == "Close RIXS Co-Pilot Agentic AI Side Panel"

    # Close sidebar
    app_win._hide_sidebar()
    assert app_win._sidebar_toggle.toolTip() == "Open RIXS Co-Pilot Agentic AI Side Panel"


# ===========================================================================
# 4. AgentSidebarWidget Toolbar Responsiveness
# ===========================================================================

def test_agent_sidebar_model_combo_responsiveness(qapp, qtbot):
    """Verify model_combo has Expanding size policy and AdjustToMinimumContentsLengthWithIcon."""
    mock_bridge = MagicMock()
    mock_main = QWidget()
    sidebar = AgentSidebarWidget(bridge=mock_bridge, main_window_ref=mock_main)
    qtbot.addWidget(sidebar)

    policy = sidebar.model_combo.sizePolicy()
    assert policy.horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert policy.verticalPolicy() == QSizePolicy.Policy.Fixed

    assert sidebar.model_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert sidebar.model_combo.minimumContentsLength() == 6


# ===========================================================================
# 5. Palette Integration
# ===========================================================================

def test_main_app_applies_dark_palette(qapp, qtbot):
    """Verify RixsApp applies dark palette with dark ToolTipBase and Window roles."""
    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)

    palette = app_win.palette()
    assert palette.color(QPalette.ColorRole.ToolTipBase).name().lower() == "#16213e"
    assert palette.color(QPalette.ColorRole.Window).name().lower() == "#1a1a2e"


# ===========================================================================
# 6. Smooth First-Toggle Sidebar Initialization
# ===========================================================================

def test_sidebar_first_init_hidden_and_smooth_sizing(qapp, qtbot, monkeypatch):
    """Verify _init_sidebar hides sidebar before adding to splitter and _show_sidebar calculates sizes smoothly."""
    monkeypatch.setattr("rixs_app.agent.auth.resolve_api_key", lambda: "mock_key")
    monkeypatch.setattr("rixs_app.main.RixsApp._load_models_async", lambda self, key: None)

    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)
    app_win.resize(1200, 800)
    app_win.show()
    qapp.processEvents()

    # Pre-initialized during startup in hidden state to prevent first-toggle layout jump
    assert app_win._sidebar is not None
    assert app_win._sidebar.isHidden() or not app_win._sidebar.isVisible()
    assert not app_win._sidebar_visible

    # Show sidebar and verify sizes are set smoothly
    app_win._show_sidebar()
    qapp.processEvents()
    assert app_win._sidebar_visible
    assert app_win._sidebar.isVisible()

    sizes = app_win._splitter.sizes()
    assert len(sizes) == 2
    assert sizes[1] > 0
    assert sizes[0] >= 300
    assert sizes[1] <= app_win._sidebar_cached_width


def test_show_sidebar_total_width_fallback(qapp, qtbot, monkeypatch):
    """Verify _show_sidebar gracefully falls back to centralWidget or main window width if splitter width is 0."""
    app_win = RixsApp(show_window=False)
    qtbot.addWidget(app_win)
    app_win.resize(900, 600)
    app_win.show()
    qapp.processEvents()

    mock_sidebar = QWidget()
    app_win._sidebar = mock_sidebar
    mock_sidebar.hide()
    app_win._splitter.addWidget(mock_sidebar)

    # Mock splitter.width to return 0 to test fallback to centralWidget / window width
    monkeypatch.setattr(app_win._splitter, "width", lambda: 0)

    set_sizes_called_with = None
    orig_set_sizes = app_win._splitter.setSizes

    def mock_set_sizes(sizes):
        nonlocal set_sizes_called_with
        set_sizes_called_with = list(sizes)
        orig_set_sizes(sizes)

    monkeypatch.setattr(app_win._splitter, "setSizes", mock_set_sizes)

    app_win._show_sidebar()
    assert app_win._sidebar_visible
    assert set_sizes_called_with is not None
    assert len(set_sizes_called_with) == 2
    assert set_sizes_called_with[1] > 0
    assert set_sizes_called_with[0] >= 300


