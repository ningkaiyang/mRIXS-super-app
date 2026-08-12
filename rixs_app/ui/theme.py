"""Centralized dark theme definitions for the mRIXS Super-App PySide6 GUI.

All color constants, stylesheet strings, and helper factories live here so that
every widget can be styled consistently without scattering magic hex values
throughout the codebase.

Usage::

    from rixs_app.ui.theme import DARK_STYLE, PALETTE, make_button_style
    app.setStyleSheet(DARK_STYLE)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

PALETTE = {
    # Backgrounds
    "bg_base":     "#1a1a2e",   # Deepest background (main window)
    "bg_panel":    "#16213e",   # Panel / sidebar background
    "bg_widget":   "#0f3460",   # Widget / input background (darker blue)
    "bg_hover":    "#1a4a8a",   # Hover state
    "bg_selected": "#1f6aa5",   # Selected / active
    "bg_disabled": "#2a2a3e",   # Disabled element background

    # Borders
    "border":      "#2d3561",   # Default border
    "border_focus":"#4a7fc1",   # Focused border

    # Text
    "text":        "#e8eaf6",   # Primary text
    "text_dim":    "#9fa8da",   # Dim / secondary text
    "text_muted":  "#5c6bc0",   # Muted / placeholder text
    "text_error":  "#ef5350",   # Error / warning text

    # Accents
    "accent_blue":  "#2196f3",  # Primary action blue
    "accent_green": "#2fa572",  # Success / play green
    "accent_orange":"#cc5500",  # Active mode indicator
    "accent_red":   "#aa3333",  # Destructive action red
    "accent_teal":  "#1a8c6e",  # Secondary action teal

    # Scrollbars
    "scroll_bg":   "#1e1e2e",
    "scroll_handle":"#3d4a7a",
    "scroll_hover": "#5060a0",
}

# ---------------------------------------------------------------------------
# Global application stylesheet (Qt QSS)
# ---------------------------------------------------------------------------

DARK_STYLE = f"""
/* ── Global base ────────────────────────────────────────── */
QWidget {{
    background-color: {PALETTE['bg_base']};
    color: {PALETTE['text']};
    font-family: "Helvetica Neue", "Arial";
    font-size: 13px;
}}

QMainWindow {{
    background-color: {PALETTE['bg_base']};
}}

/* ── Frames / containers ────────────────────────────────── */
QFrame, QGroupBox {{
    background-color: {PALETTE['bg_panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {PALETTE['text_dim']};
    font-weight: bold;
}}

/* ── Push buttons (default fallback) ────────────────────── */
QPushButton {{
    background-color: {PALETTE['bg_selected']};
    color: {PALETTE['text']};
    border: none;
    border-radius: 5px;
    padding: 5px 12px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {PALETTE['bg_hover']};
}}

QPushButton:pressed {{
    background-color: {PALETTE['accent_blue']};
}}

QPushButton:disabled {{
    background-color: {PALETTE['bg_disabled']};
    color: {PALETTE['text_muted']};
}}

/* ── Named button variants (hover + pressed states) ────── */

/* Tool / neutral buttons (zoom, manual line, help, etc.) */
QPushButton#tool_btn {{
    background-color: #2d3558;
    color: #ffffff;
    border: 1px solid #3f4b78;
}}
QPushButton#tool_btn:hover {{
    background-color: #3d4875;
    border: 1px solid #5667a0;
}}
QPushButton#tool_btn:pressed {{
    background-color: #4a578c;
}}

/* Play / start button (green) */
QPushButton#play_btn {{
    background-color: {PALETTE['accent_green']};
    color: white;
}}
QPushButton#play_btn:hover {{
    background-color: #3ac085;
}}
QPushButton#play_btn:pressed {{
    background-color: #238a5a;
}}

/* Active mode indicator (orange — autoplay active, manual mode on) */
QPushButton#active_btn {{
    background-color: {PALETTE['accent_orange']};
    color: white;
}}
QPushButton#active_btn:hover {{
    background-color: #dd6611;
}}
QPushButton#active_btn:pressed {{
    background-color: #bb4400;
}}

/* Accent / primary action (blue — Precompute, Auto All, Compare) */
QPushButton#accent_btn {{
    background-color: {PALETTE['accent_blue']};
    color: white;
}}
QPushButton#accent_btn:hover {{
    background-color: #42a5f5;
}}
QPushButton#accent_btn:pressed {{
    background-color: #1976d2;
}}

/* Danger / destructive (red — Remove, Clear) */
QPushButton#danger_btn {{
    background-color: {PALETTE['accent_red']};
    color: white;
}}
QPushButton#danger_btn:hover {{
    background-color: #cc4444;
}}
QPushButton#danger_btn:pressed {{
    background-color: #882222;
}}

/* Danger secondary (darker red — Clear All) */
QPushButton#danger_secondary_btn {{
    background-color: #883333;
    color: white;
}}
QPushButton#danger_secondary_btn:hover {{
    background-color: #994444;
}}
QPushButton#danger_secondary_btn:pressed {{
    background-color: #662222;
}}

/* Success / launch button (green, bold — Start Alignment) */
QPushButton#success_btn {{
    background-color: {PALETTE['accent_green']};
    color: white;
    font-size: 14px;
    font-weight: bold;
}}
QPushButton#success_btn:hover {{
    background-color: #3ac085;
}}
QPushButton#success_btn:pressed {{
    background-color: #238a5a;
}}

/* Sort / secondary accent (blue — Sort Files, Zeroth-Order) */
QPushButton#sort_btn {{
    background-color: #1F6AA5;
    color: white;
    font-size: 14px;
    font-weight: bold;
}}
QPushButton#sort_btn:hover {{
    background-color: #2878b8;
}}
QPushButton#sort_btn:pressed {{
    background-color: #165a8a;
}}

/* Muted cancel / dismiss button */
QPushButton#cancel_btn {{
    background-color: #666677;
    color: white;
}}
QPushButton#cancel_btn:hover {{
    background-color: #777788;
}}
QPushButton#cancel_btn:pressed {{
    background-color: #555566;
}}

/* Header title label */
QLabel#header_title {{
    font-size: 26px;
    font-weight: bold;
    color: {PALETTE['text']};
}}

/* ── Sliders ────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {PALETTE['bg_widget']};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {PALETTE['accent_blue']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {PALETTE['text']};
    border: 2px solid {PALETTE['border_focus']};
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -4px 0;
}}

QSlider::handle:horizontal:hover {{
    background: {PALETTE['accent_blue']};
}}

QSlider:disabled {{
    opacity: 0.4;
}}

/* ── Line edits / inputs ────────────────────────────────── */
QLineEdit {{
    background-color: {PALETTE['bg_widget']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    padding: 3px 8px;
    selection-background-color: {PALETTE['accent_blue']};
}}

QLineEdit:focus {{
    border: 1px solid {PALETTE['border_focus']};
}}

QLineEdit:disabled {{
    background-color: {PALETTE['bg_disabled']};
    color: {PALETTE['text_muted']};
}}

/* ── ComboBox ───────────────────────────────────────────── */
QComboBox {{
    combobox-popup: 0;
    background-color: #262c4a;
    color: #ffffff;
    border: 1px solid #3d4566;
    border-radius: 4px;
    padding: 3px 8px;
    min-width: 80px;
}}

QComboBox#engine_menu {{
    min-width: 185px;
}}

QComboBox:hover {{
    border: 1px solid #54659e;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #3d4566;
}}

QComboBox::down-arrow {{
    image: none;
    width: 10px;
    height: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: #1a1f36;
    border: 1px solid #3d4566;
    selection-background-color: #2196f3;
    selection-color: #ffffff;
    color: #ffffff;
    padding: 4px;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    min-height: 24px;
    padding: 4px 8px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: #2a3356;
    color: #ffffff;
}}

/* ── Labels ─────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {PALETTE['text']};
    border: none;
}}

QLabel#dim_label {{
    color: {PALETTE['text_dim']};
    font-size: 12px;
}}

QLabel#muted_label {{
    color: {PALETTE['text_muted']};
    font-size: 11px;
}}

QLabel#accent_blue {{
    color: #64b5f6;
    font-weight: bold;
}}

QLabel#accent_orange {{
    color: #ffb74d;
    font-weight: bold;
}}

QLabel#accent_green {{
    color: #81c784;
    font-weight: bold;
}}

QLabel#badge_best_focus {{
    color: #22cc66;
    font-weight: bold;
    font-size: 12px;
}}

/* ── CheckBox ───────────────────────────────────────────── */
QCheckBox {{
    spacing: 6px;
    color: {PALETTE['text']};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {PALETTE['border']};
    border-radius: 3px;
    background-color: {PALETTE['bg_widget']};
}}

QCheckBox::indicator:checked {{
    background-color: {PALETTE['accent_blue']};
    border-color: {PALETTE['accent_blue']};
}}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {PALETTE['scroll_bg']};
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {PALETTE['scroll_handle']};
    min-height: 30px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: {PALETTE['scroll_hover']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {PALETTE['scroll_bg']};
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {PALETTE['scroll_handle']};
    min-width: 30px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {PALETTE['scroll_hover']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Splitter ───────────────────────────────────────────── */
QSplitter::handle {{
    background: {PALETTE['border']};
}}

QSplitter::handle:horizontal {{
    width: 3px;
}}

QSplitter::handle:vertical {{
    height: 3px;
}}

/* ── Toolbar / nav bar frame ────────────────────────────── */
QFrame#navbar_frame {{
    background-color: {PALETTE['bg_panel']};
    border: none;
    border-bottom: 1px solid {PALETTE['border']};
    border-radius: 0px;
}}

/* ── Stacked widget pages ───────────────────────────────── */
QStackedWidget {{
    background: transparent;
    border: none;
}}

/* ── List widgets ───────────────────────────────────────── */
QListWidget {{
    background-color: {PALETTE['bg_widget']};
    border: 1px solid {PALETTE['border']};
    border-radius: 5px;
    padding: 4px;
    alternate-background-color: {PALETTE['bg_panel']};
}}

QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {PALETTE['bg_selected']};
    color: {PALETTE['text']};
}}

QListWidget::item:hover {{
    background-color: {PALETTE['bg_hover']};
}}

/* ── MessageBox / Dialogs ───────────────────────────────── */
QMessageBox {{
    background-color: {PALETTE['bg_panel']};
    color: {PALETTE['text']};
}}

QMessageBox QPushButton {{
    min-width: 80px;
}}

/* ── ToolTips ───────────────────────────────────────────── */
QToolTip {{
    background-color: {PALETTE['bg_panel']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border_focus']};
    padding: 4px 6px;
    border-radius: 4px;
}}
"""

# ---------------------------------------------------------------------------
# Per-button style helpers (legacy — prefer set_*_btn helpers below)
# ---------------------------------------------------------------------------


def danger_style() -> str:
    """Return a QSS snippet for a destructive / danger button."""
    return f"background-color: {PALETTE['accent_red']}; color: white;"


def success_style() -> str:
    """Return a QSS snippet for a success / start button."""
    return f"background-color: {PALETTE['accent_green']}; color: white;"


def accent_style() -> str:
    """Return a QSS snippet for a primary accent button."""
    return f"background-color: {PALETTE['accent_blue']}; color: white;"


def neutral_style() -> str:
    """Return a QSS snippet for a neutral / tool button."""
    return "background-color: #444455; color: white;"


def active_mode_style() -> str:
    """Return a QSS snippet for a button that is in an active 'mode on' state."""
    return f"background-color: {PALETTE['accent_orange']}; color: white;"


# ---------------------------------------------------------------------------
# Object-name helpers — apply these to QPushButtons so the global QSS
# `:hover` / `:pressed` pseudo-states fire correctly.  Using objectName
# instead of inline setStyleSheet is essential because Qt inline styles
# override the cascade and prevent pseudo-state matching.
# ---------------------------------------------------------------------------


def set_tool_btn(btn) -> None:
    """Mark *btn* as a neutral 'tool' button (zoom, manual line, help, etc.)."""
    btn.setObjectName("tool_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_play_btn(btn) -> None:
    """Mark *btn* as the green play / start button."""
    btn.setObjectName("play_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_active_btn(btn) -> None:
    """Mark *btn* as an orange 'mode active' indicator."""
    btn.setObjectName("active_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_accent_btn(btn) -> None:
    """Mark *btn* as a blue primary-action button."""
    btn.setObjectName("accent_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_danger_btn(btn) -> None:
    """Mark *btn* as a red destructive-action button."""
    btn.setObjectName("danger_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_danger_secondary_btn(btn) -> None:
    """Mark *btn* as a darker-red secondary danger button."""
    btn.setObjectName("danger_secondary_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_success_btn(btn) -> None:
    """Mark *btn* as a green success / launch button."""
    btn.setObjectName("success_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_sort_btn(btn) -> None:
    """Mark *btn* as a blue sort / secondary-accent button."""
    btn.setObjectName("sort_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


def set_cancel_btn(btn) -> None:
    """Mark *btn* as a muted cancel / dismiss button."""
    btn.setObjectName("cancel_btn")
    btn.style().unpolish(btn)
    btn.style().polish(btn)


# Alias used by main.py for the full application stylesheet
FULL_QSS = DARK_STYLE
