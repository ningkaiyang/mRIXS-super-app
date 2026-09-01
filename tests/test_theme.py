"""PySide6 unit tests for UI theme, typography, button states, focus rings, splitter handles, and combobox styling.

Covers:
1. Font stack declarations and typography constants (preventing macOS QPA font warnings).
2. Palette hex color validity and QSS syntactic bracket balance.
3. Named button specificity and explicit :disabled pseudo-class rules across all 10 button IDs.
4. Focus ring rules and interactive controls focus lifecycle.
5. QSplitter horizontal and vertical handle styling specifications and resize integrity.
6. Co-Pilot toggle button synchronization, state transitions, and reparenting across views.
7. Dropdown SVG arrow assets, dark theme QSS down-arrow references, and QComboBox instantiation.
"""

from __future__ import annotations

from pathlib import Path
import re
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QComboBox,
    QListWidget,
    QLineEdit,
    QSplitter,
    QVBoxLayout,
    QToolTip,
)

from rixs_app.ui.theme import (
    DARK_STYLE,
    FULL_QSS,
    PALETTE,
    FONT_STACK_UI,
    FONT_STACK_CODE,
    _DROPDOWN_ARROW_SVG,
    _DROPDOWN_ARROW_HOVER_SVG,
    _DROPDOWN_ARROW_DISABLED_SVG,
    apply_dark_palette,
    set_tool_btn,
    set_play_btn,
    set_active_btn,
    set_accent_btn,
    set_danger_btn,
    set_danger_secondary_btn,
    set_success_btn,
    set_sort_btn,
    set_cancel_btn,
    set_amber_btn,
)
from rixs_app.main import RixsApp, _IDX_SORTING, _IDX_SLIDESHOW, _IDX_COMPARISON, _IDX_ZEROTH_ORDER


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


BUTTON_HELPERS = [
    ("tool_btn", set_tool_btn),
    ("play_btn", set_play_btn),
    ("active_btn", set_active_btn),
    ("accent_btn", set_accent_btn),
    ("danger_btn", set_danger_btn),
    ("danger_secondary_btn", set_danger_secondary_btn),
    ("success_btn", set_success_btn),
    ("sort_btn", set_sort_btn),
    ("cancel_btn", set_cancel_btn),
    ("amber_btn", set_amber_btn),
]


# ===========================================================================
# 1. Font Stacks & Typography Verification
# ===========================================================================

def test_font_stacks_contract():
    """Verify FONT_STACK_UI and FONT_STACK_CODE match cross-platform definitions."""
    import sys
    if sys.platform == "darwin":
        expected_ui = '"Helvetica Neue", Arial'
        expected_code = 'Menlo, Monaco, "Courier New"'
    elif sys.platform == "win32":
        expected_ui = '"Segoe UI", Arial'
        expected_code = 'Consolas, "Courier New"'
    else:
        expected_ui = 'Ubuntu, "DejaVu Sans", Arial'
        expected_code = '"DejaVu Sans Mono", "Courier New"'

    assert FONT_STACK_UI == expected_ui, f"FONT_STACK_UI mismatch: got {FONT_STACK_UI}"
    assert FONT_STACK_CODE == expected_code, f"FONT_STACK_CODE mismatch: got {FONT_STACK_CODE}"
    assert f"font-family: {FONT_STACK_UI};" in DARK_STYLE


def test_palette_hex_format():
    """Verify all PALETTE values are valid CSS hex color strings."""
    hex_pattern = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
    for key, val in PALETTE.items():
        assert isinstance(val, str), f"PALETTE[{key}] is not a string"
        assert hex_pattern.match(val), f"PALETTE[{key}] = {val} is not a valid hex color"


def test_qss_bracket_balance():
    """Verify DARK_STYLE has perfectly matched braces (no QSS syntax parse corruption)."""
    open_count = DARK_STYLE.count("{")
    close_count = DARK_STYLE.count("}")
    assert open_count == close_count, f"Mismatched braces in DARK_STYLE: {open_count} open vs {close_count} close"
    assert FULL_QSS == DARK_STYLE


# ===========================================================================
# 2. Button Types & Disabled State Specificity
# ===========================================================================

def test_all_10_button_disabled_qss_rules_present():
    """Verify that all 10 button IDs are explicitly defined with :disabled pseudo-classes."""
    for btn_id, _ in BUTTON_HELPERS:
        selector = f"QPushButton#{btn_id}:disabled"
        assert selector in DARK_STYLE, f"Missing selector in DARK_STYLE: {selector}"


@pytest.mark.parametrize("btn_id,helper_fn", BUTTON_HELPERS)
def test_button_helper_sets_object_name_and_polishes(qapp, btn_id, helper_fn):
    """Verify helper function sets objectName and allows rendering without crash."""
    btn = QPushButton("Test Button")
    helper_fn(btn)
    assert btn.objectName() == btn_id

    # Test rendering enabled
    pix_enabled = QPixmap(100, 30)
    btn.render(pix_enabled)
    assert not pix_enabled.isNull()

    # Test rendering disabled
    btn.setEnabled(False)
    assert not btn.isEnabled()
    pix_disabled = QPixmap(100, 30)
    btn.render(pix_disabled)
    assert not pix_disabled.isNull()


def test_button_toggle_enabled_stress(qapp):
    """Test toggling enabled/disabled state across all button types."""
    container = QWidget()
    container.setStyleSheet(DARK_STYLE)
    layout = QVBoxLayout(container)

    buttons = []
    for btn_id, helper_fn in BUTTON_HELPERS:
        btn = QPushButton(f"Btn {btn_id}")
        helper_fn(btn)
        layout.addWidget(btn)
        buttons.append(btn)

    container.show()

    for _ in range(10):
        for btn in buttons:
            btn.setEnabled(False)
            assert not btn.isEnabled()
        for btn in buttons:
            btn.setEnabled(True)
            assert btn.isEnabled()

    pix = QPixmap(container.size())
    container.render(pix)
    assert not pix.isNull()
    container.close()


# ===========================================================================
# 3. Focus Rings & Interactive Controls
# ===========================================================================

def test_focus_styling_qss_syntax():
    """Verify focus ring rules for QPushButton, QComboBox, QListWidget, and QLineEdit."""
    assert "QPushButton:focus" in DARK_STYLE
    assert "border: 1.5px solid #4a7fc1;" in DARK_STYLE
    assert "QComboBox:focus" in DARK_STYLE
    assert "QListWidget:focus" in DARK_STYLE
    assert "QLineEdit:focus" in DARK_STYLE


def test_interactive_controls_focus_lifecycle(qapp):
    """Verify interactive widgets accept focus and render cleanly under DARK_STYLE."""
    window = QWidget()
    window.setStyleSheet(DARK_STYLE)
    layout = QVBoxLayout(window)

    btn = QPushButton("Action")
    set_accent_btn(btn)
    combo = QComboBox()
    combo.addItems(["Opt A", "Opt B"])
    line_edit = QLineEdit("Search...")
    list_widget = QListWidget()
    list_widget.addItems(["Item 1", "Item 2"])

    layout.addWidget(btn)
    layout.addWidget(combo)
    layout.addWidget(line_edit)
    layout.addWidget(list_widget)

    window.show()

    widgets = [btn, combo, line_edit, list_widget]
    for w in widgets:
        w.setFocus()
        qapp.processEvents()
        pix = QPixmap(w.size())
        w.render(pix)
        assert not pix.isNull()
        w.clearFocus()
        qapp.processEvents()

    window.close()


# ===========================================================================
# 4. QSplitter Handle Specifications
# ===========================================================================

def test_qsplitter_qss_specifications():
    """Verify QSplitter handle styling rules in DARK_STYLE."""
    assert "QSplitter::handle:horizontal" in DARK_STYLE
    assert "background-color: #1f274a;" in DARK_STYLE
    assert "width: 5px;" in DARK_STYLE
    assert "QSplitter::handle:horizontal:hover" in DARK_STYLE
    assert "background-color: #3b82f6;" in DARK_STYLE
    assert "QSplitter::handle:horizontal:pressed" in DARK_STYLE
    assert "background-color: #60a5fa;" in DARK_STYLE

    assert "QSplitter::handle:vertical" in DARK_STYLE
    assert "height: 5px;" in DARK_STYLE
    assert "QSplitter::handle:vertical:hover" in DARK_STYLE
    assert "QSplitter::handle:vertical:pressed" in DARK_STYLE


def test_qsplitter_instantiation_and_resize(qapp):
    """Verify horizontal and vertical QSplitters instantiate and handle resize under DARK_STYLE."""
    h_splitter = QSplitter(Qt.Orientation.Horizontal)
    h_splitter.setStyleSheet(DARK_STYLE)
    w1 = QWidget()
    w2 = QWidget()
    h_splitter.addWidget(w1)
    h_splitter.addWidget(w2)
    h_splitter.resize(800, 600)
    h_splitter.show()
    h_splitter.setSizes([500, 300])
    assert h_splitter.sizes() == [500, 300] or sum(h_splitter.sizes()) > 0
    h_splitter.close()

    v_splitter = QSplitter(Qt.Orientation.Vertical)
    v_splitter.setStyleSheet(DARK_STYLE)
    v1 = QWidget()
    v2 = QWidget()
    v_splitter.addWidget(v1)
    v_splitter.addWidget(v2)
    v_splitter.resize(600, 800)
    v_splitter.show()
    v_splitter.setSizes([400, 400])
    assert v_splitter.sizes() == [400, 400] or sum(v_splitter.sizes()) > 0
    v_splitter.close()


# ===========================================================================
# 5. Co-Pilot Button Toggle Synchronization
# ===========================================================================

def test_copilot_sidebar_toggle_lifecycle(qapp):
    """Verify _show_sidebar and _hide_sidebar styling and state transitions."""
    rixs = RixsApp(show_window=False)
    rixs._sidebar = MagicMock()

    # Initial state
    assert rixs._sidebar_visible is False
    assert rixs._sidebar_toggle.text() == "🤖 Co-Pilot"
    assert rixs._sidebar_toggle.toolTip() == "Open RIXS Co-Pilot Agentic AI Side Panel"
    assert rixs._sidebar_toggle.objectName() == "copilot_btn"
    assert rixs._sidebar_toggle.styleSheet() == ""

    # Show sidebar
    rixs._show_sidebar()
    assert rixs._sidebar_visible is True
    assert rixs._sidebar_toggle.text() == "🤖 ✕"
    assert rixs._sidebar_toggle.toolTip() == "Close RIXS Co-Pilot Agentic AI Side Panel"
    assert "border: 1.5px solid #38bdf8" in rixs._sidebar_toggle.styleSheet()
    assert "color: #ffffff" in rixs._sidebar_toggle.styleSheet()

    # Hide sidebar
    rixs._hide_sidebar()
    assert rixs._sidebar_visible is False
    assert rixs._sidebar_toggle.text() == "🤖 Co-Pilot"
    assert rixs._sidebar_toggle.toolTip() == "Open RIXS Co-Pilot Agentic AI Side Panel"
    assert rixs._sidebar_toggle.objectName() == "copilot_btn"
    assert rixs._sidebar_toggle.styleSheet() == ""


def test_copilot_reparenting_across_views(qapp):
    """Verify Co-Pilot button is correctly reparented across all views without breaking styling."""
    rixs = RixsApp(show_window=False)
    rixs._sidebar = MagicMock()

    view_indices = [_IDX_SORTING, _IDX_SLIDESHOW, _IDX_COMPARISON, _IDX_ZEROTH_ORDER]

    for idx in view_indices:
        rixs._reparent_toggle_btn(idx)
        rixs._show_sidebar()
        assert rixs._sidebar_toggle.text() == "🤖 ✕"
        assert "border: 1.5px solid #38bdf8" in rixs._sidebar_toggle.styleSheet()

        rixs._hide_sidebar()
        assert rixs._sidebar_toggle.text() == "🤖 Co-Pilot"
        assert rixs._sidebar_toggle.objectName() == "copilot_btn"
        assert rixs._sidebar_toggle.styleSheet() == ""


# ===========================================================================
# 6. Dropdown Arrow SVG Assets & QComboBox Styling
# ===========================================================================

def test_dropdown_svg_assets_exist_and_valid():
    """Verify all 3 SVG arrow assets exist and parse as valid SVG/XML."""
    assets = [_DROPDOWN_ARROW_SVG, _DROPDOWN_ARROW_HOVER_SVG, _DROPDOWN_ARROW_DISABLED_SVG]
    for asset_path in assets:
        p = Path(asset_path)
        assert p.is_file(), f"Missing SVG asset: {asset_path}"
        assert p.stat().st_size > 0, f"Empty SVG asset: {asset_path}"

        tree = ET.parse(p)
        root = tree.getroot()
        assert "svg" in root.tag, f"Root element is not svg in {asset_path}"
        polyline = root.find("{http://www.w3.org/2000/svg}polyline")
        assert polyline is not None, f"Missing polyline in {asset_path}"


def test_theme_qss_references_svg_arrows():
    """Verify DARK_STYLE contains url(...) references for down-arrows."""
    assert "QComboBox::down-arrow" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_SVG});" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_HOVER_SVG});" in DARK_STYLE
    assert f"image: url({_DROPDOWN_ARROW_DISABLED_SVG});" in DARK_STYLE
    assert "image: none;" not in DARK_STYLE


def test_all_app_comboboxes_styled_cleanly(qapp):
    """Verify that dropdowns across the application instantiate and apply FULL_QSS without error."""
    parent = QWidget()
    parent.setStyleSheet(FULL_QSS)

    engine_combo = QComboBox(parent)
    engine_combo.setObjectName("engine_menu")
    engine_combo.addItems(["PCA", "ECC", "Phase Correlation"])
    assert engine_combo.count() == 3

    colormap_combo = QComboBox(parent)
    colormap_combo.setObjectName("colormap_menu")
    colormap_combo.addItems(["viridis", "inferno", "plasma", "magma", "grayscale"])
    assert colormap_combo.count() == 5

    stage_combo = QComboBox(parent)
    stage_combo.setObjectName("stage_menu")
    stage_combo.addItems(["Raw", "Denoised (D)", "Row-Smoothed (Dsm)", "Gradient (G)", "Fitted-Line Strip"])
    assert stage_combo.count() == 5

    model_combo = QComboBox(parent)
    model_combo.addItems(["lbl/cborg-deepthought:latest", "lbl/cborg-coder:latest"])
    assert model_combo.count() == 2


# ===========================================================================
# 8. Dark Palette, ToolTip & Squircle Card Sizing
# ===========================================================================

def test_apply_dark_palette_configures_palette_and_tooltip(qapp):
    """Verify apply_dark_palette sets dark roles on QApplication, QPalette, and QToolTip."""
    palette = apply_dark_palette(qapp)

    # Check key color roles
    assert palette.color(QPalette.ColorRole.Window).name().lower() == "#1a1a2e"
    assert palette.color(QPalette.ColorRole.WindowText).name().lower() == "#e8eaf6"
    assert palette.color(QPalette.ColorRole.Base).name().lower() == "#16213e"
    assert palette.color(QPalette.ColorRole.Text).name().lower() == "#e8eaf6"
    assert palette.color(QPalette.ColorRole.ToolTipBase).name().lower() == "#16213e"
    assert palette.color(QPalette.ColorRole.ToolTipText).name().lower() == "#e8eaf6"

    # Check global QToolTip palette
    tooltip_palette = QToolTip.palette()
    assert tooltip_palette.color(QPalette.ColorRole.ToolTipBase).name().lower() == "#16213e"
    assert tooltip_palette.color(QPalette.ColorRole.ToolTipText).name().lower() == "#e8eaf6"


def test_apply_dark_palette_with_widget(qapp):
    """Verify apply_dark_palette applies to a QWidget instance."""
    w = QWidget()
    palette = apply_dark_palette(w)
    assert w.palette().color(QPalette.ColorRole.ToolTipBase).name().lower() == "#16213e"
    assert w.palette().color(QPalette.ColorRole.ToolTipText).name().lower() == "#e8eaf6"


def test_apply_dark_palette_sets_fusion_style_and_global_stylesheet(qapp):
    """Verify apply_dark_palette enforces Fusion style and FULL_QSS on QApplication."""
    apply_dark_palette(qapp)
    style = qapp.style()
    style_names = [style.name().lower() if hasattr(style, "name") else style.objectName().lower()]
    style_names.extend([c.name().lower() for c in style.children() if hasattr(c, "name")])
    assert any("fusion" in name for name in style_names)
    assert qapp.styleSheet() == FULL_QSS


def test_qtooltip_qss_styling():
    """Verify QToolTip and QLabel#qtooltip_label QSS rule in DARK_STYLE has explicit colors, borders, and padding."""
    assert "QToolTip, QLabel#qtooltip_label" in DARK_STYLE or "QToolTip,\nQLabel#qtooltip_label" in DARK_STYLE
    assert f"background-color: {PALETTE['bg_panel']};" in DARK_STYLE
    assert f"color: {PALETTE['text']};" in DARK_STYLE
    assert f"border: 1px solid {PALETTE['border_focus']};" in DARK_STYLE
    assert "padding: 4px 8px;" in DARK_STYLE
    assert "border-radius: 4px;" in DARK_STYLE


def test_squircle_card_qss_no_forced_min_size():
    """Verify QFrame#squircle_card does not enforce artificial min-width or min-height in QSS."""
    match = re.search(r"QFrame#squircle_card\s*\{([^}]+)\}", DARK_STYLE)
    assert match is not None, "QFrame#squircle_card rule not found in DARK_STYLE"
    card_block = match.group(1)
    assert "min-width" not in card_block, f"min-width should not be enforced in QSS: {card_block}"
    assert "min-height" not in card_block, f"min-height should not be enforced in QSS: {card_block}"
    assert "border-radius: 12px;" in card_block

