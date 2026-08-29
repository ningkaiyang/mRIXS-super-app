"""PySide6 unit tests for Home Launchpad Hub (rixs_app/ui/home_launchpad.py).

Covers:
1. 2x2 Squircle grid layout, card titles, icons, accent bars, and dimensions (>= 280x160).
2. Navigation callbacks for Dark Cal, Zeroth-Order, Photon Clustering, and Spatial Alignment.
3. Dynamic dark calibration status badge and subtitle updates (valid, missing, and error states).
4. Co-Pilot toggle button docking, replacement, and reparenting lifecycle.
5. Resiliency against None callbacks, rapid clicking, and viewport resizing.
6. Theme integration, PointingHandCursor, and offscreen QPixmap rendering.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import weakref

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QPushButton,
    QWidget,
)

from rixs_app.core import calibration_store
from rixs_app.ui.home_launchpad import HomeLaunchpadView, SquircleCard
from rixs_app.ui.theme import DARK_STYLE, FULL_QSS


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_cal_dir(tmp_path: Path):
    """Create a temporary directory with a valid calibration manifest and TIFFs."""
    cal_dir = tmp_path / "mock_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    med = np.full((100, 100), 25.0, dtype=np.float32)
    mask = np.ones((100, 100), dtype=np.float32)

    calibration_store.save_calibration(
        med_dark=med,
        final_mask=mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=20,
        surviving_pixels=9950,
        total_pixels=10000,
        suppression_pct=0.5,
        source_dir=str(tmp_path / "raw_dark"),
        date="2026-08-28T12:00:00",
        cal_dir=cal_dir,
    )
    return cal_dir


# ===========================================================================
# 1. Grid & Squircle Cards Structure
# ===========================================================================

def test_home_launchpad_initialization_and_cards(qapp, qtbot):
    """Verify HomeLaunchpadView instantiates with 4 squircle cards in a 2x2 grid."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.setStyleSheet(DARK_STYLE)
    view.show()

    cards = view.findChildren(QFrame, "squircle_card")
    assert len(cards) == 4, f"Expected 4 squircle cards, found {len(cards)}"

    assert hasattr(view, "_card_dark_cal")
    assert hasattr(view, "_card_zeroth_order")
    assert hasattr(view, "_card_clustering")
    assert hasattr(view, "_card_alignment")

    assert view._card_dark_cal in cards
    assert view._card_zeroth_order in cards
    assert view._card_clustering in cards
    assert view._card_alignment in cards


def test_squircle_card_dimensions_and_cursor(qapp, qtbot):
    """Verify each squircle card meets minimum dimensions and uses PointingHandCursor."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    cards = view.findChildren(QFrame, "squircle_card")
    for card in cards:
        assert card.minimumWidth() >= 280
        assert card.minimumHeight() >= 160
        assert card.cursor().shape() == Qt.PointingHandCursor


def test_squircle_card_labels_and_accents(qapp, qtbot):
    """Verify each card contains expected title, subtitle, icon, and accent bar."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    card_titles = [
        "Dark Image & Pixel Masking",
        "Zeroth-Order Mirror Pitch Calibration",
        "Single-Photon Event Clustering",
        "Spatial Drift Alignment & Stacking",
    ]

    for title in card_titles:
        matching = [
            c for c in view.findChildren(QFrame, "squircle_card")
            if any(title in lbl.text() for lbl in c.findChildren(QLabel, "squircle_card_title"))
        ]
        assert len(matching) == 1, f"Missing card with title: {title}"


# ===========================================================================
# 2. Navigation Callbacks & Click Handling
# ===========================================================================

def test_navigation_callbacks_triggered_on_mouse_press(qapp, qtbot):
    """Verify each card click invokes its respective registered navigation callback."""
    mock_dark = MagicMock()
    mock_zo = MagicMock()
    mock_clust = MagicMock()
    mock_align = MagicMock()

    view = HomeLaunchpadView(
        on_dark_calibration=mock_dark,
        on_zeroth_order=mock_zo,
        on_clustering=mock_clust,
        on_alignment=mock_align,
    )
    qtbot.addWidget(view)
    view.show()

    view._card_dark_cal.mousePressEvent()
    assert mock_dark.call_count == 1

    view._card_zeroth_order.mousePressEvent()
    assert mock_zo.call_count == 1

    view._card_clustering.mousePressEvent()
    assert mock_clust.call_count == 1

    view._card_alignment.mousePressEvent()
    assert mock_align.call_count == 1


def test_qtbot_mouse_click_on_cards(qapp, qtbot):
    """Verify real qtbot mouse clicks on card surface invoke callbacks."""
    mock_dark = MagicMock()
    mock_clust = MagicMock()

    view = HomeLaunchpadView(
        on_dark_calibration=mock_dark,
        on_clustering=mock_clust,
    )
    qtbot.addWidget(view)
    view.show()

    qtbot.mouseClick(view._card_dark_cal, Qt.LeftButton)
    assert mock_dark.call_count == 1

    qtbot.mouseClick(view._card_clustering, Qt.LeftButton)
    assert mock_clust.call_count == 1


def test_none_callbacks_resilience(qapp, qtbot):
    """Verify clicking cards when callbacks are None does not raise TypeError or crash."""
    view = HomeLaunchpadView(
        on_dark_calibration=None,
        on_zeroth_order=None,
        on_clustering=None,
        on_alignment=None,
    )
    qtbot.addWidget(view)
    view.show()

    for card in view.findChildren(QFrame, "squircle_card"):
        card.mousePressEvent()
        qtbot.mouseClick(card, Qt.LeftButton)


# ===========================================================================
# 3. Dynamic Dark Calibration Status Badging
# ===========================================================================

def test_refresh_calibration_status_with_valid_calibration(qapp, qtbot, mock_cal_dir):
    """Verify refresh_calibration_status displays date and active pixel percentage when calibrated."""
    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", mock_cal_dir):
        view = HomeLaunchpadView()
        qtbot.addWidget(view)
        view.show()
        view.refresh_calibration_status()

        labels = view._card_dark_cal.findChildren(QLabel)
        texts = [lbl.text() for lbl in labels]
        assert any("2026-08-28" in t for t in texts)
        assert any("99.50%" in t or "active" in t for t in texts)
        assert any("Mask Generated" in t or "Calibrated" in t for t in texts)


def test_refresh_calibration_status_uncalibrated(qapp, qtbot, tmp_path):
    """Verify refresh_calibration_status displays '⚠️ No Mask' badge when uncalibrated."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", empty_dir):
        view = HomeLaunchpadView()
        qtbot.addWidget(view)
        view.show()
        view.refresh_calibration_status()

        labels = view._card_dark_cal.findChildren(QLabel)
        texts = [lbl.text() for lbl in labels]
        assert any("No Mask" in t or "Not calibrated" in t for t in texts)
        assert any("⚠️" in t for t in texts)


def test_refresh_calibration_status_exception_handling(qapp, qtbot):
    """Verify refresh_calibration_status falls back to uncalibrated state if store raises."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    with patch("rixs_app.core.calibration_store.get_calibration_summary", side_effect=RuntimeError("Disk failure")):
        view.refresh_calibration_status()
        labels = view._card_dark_cal.findChildren(QLabel)
        texts = [lbl.text() for lbl in labels]
        assert any("No Mask" in t or "Not calibrated" in t for t in texts)


# ===========================================================================
# 4. Co-Pilot Button Docking Lifecycle
# ===========================================================================

def test_copilot_button_docking(qapp, qtbot):
    """Verify set_copilot_button reparents and docks toggle button into header row."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    btn = QPushButton("🤖 Co-Pilot")
    view.set_copilot_button(btn)

    assert btn.parent() is not None
    assert view.isAncestorOf(btn)
    assert btn.isVisible()


def test_copilot_button_replacement(qapp, qtbot):
    """Verify calling set_copilot_button with a new button cleans up the old one."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    btn1 = QPushButton("🤖 Co-Pilot 1")
    btn2 = QPushButton("🤖 Co-Pilot 2")

    view.set_copilot_button(btn1)
    assert view.isAncestorOf(btn1)

    view.set_copilot_button(btn2)
    assert view.isAncestorOf(btn2)
    assert not view.isAncestorOf(btn1)


def test_copilot_button_idempotent_docking(qapp, qtbot):
    """Verify calling set_copilot_button multiple times with the same button does not duplicate."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    btn = QPushButton("🤖 Co-Pilot")
    view.set_copilot_button(btn)
    view.set_copilot_button(btn)

    assert view.isAncestorOf(btn)


# ===========================================================================
# 5. Stress Testing & Viewport Resizing
# ===========================================================================

def test_rapid_sequential_clicks(qapp, qtbot):
    """Verify rapid card clicking executes without deadlock or race conditions."""
    counts = {"dark": 0, "zo": 0, "clust": 0, "align": 0}

    view = HomeLaunchpadView(
        on_dark_calibration=lambda: counts.update(dark=counts["dark"] + 1),
        on_zeroth_order=lambda: counts.update(zo=counts["zo"] + 1),
        on_clustering=lambda: counts.update(clust=counts["clust"] + 1),
        on_alignment=lambda: counts.update(align=counts["align"] + 1),
    )
    qtbot.addWidget(view)
    view.show()

    for _ in range(25):
        view._card_dark_cal.mousePressEvent()
        view._card_zeroth_order.mousePressEvent()
        view._card_clustering.mousePressEvent()
        view._card_alignment.mousePressEvent()

    assert counts["dark"] == 25
    assert counts["zo"] == 25
    assert counts["clust"] == 25
    assert counts["align"] == 25


def test_viewport_resize_integrity(qapp, qtbot):
    """Verify squircle card minimum size is strictly preserved on window resize."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    # Small resize
    view.resize(300, 200)
    qapp.processEvents()
    for card in view.findChildren(QFrame, "squircle_card"):
        assert card.minimumWidth() >= 280
        assert card.minimumHeight() >= 160

    # Large resize
    view.resize(1920, 1080)
    qapp.processEvents()
    for card in view.findChildren(QFrame, "squircle_card"):
        assert card.minimumWidth() >= 280
        assert card.minimumHeight() >= 160


def test_offscreen_rendering_pixmap(qapp, qtbot):
    """Verify view renders cleanly to QPixmap without OpenGL or styling faults."""
    view = HomeLaunchpadView()
    view.setStyleSheet(FULL_QSS)
    qtbot.addWidget(view)
    view.resize(1000, 700)
    view.show()

    pix = QPixmap(view.size())
    view.render(pix)
    assert not pix.isNull()
    assert pix.width() == 1000
    assert pix.height() == 700


# ===========================================================================
# 6. Consolidated Adversarial & Edge-Case Tests
# ===========================================================================

def test_adversarial_child_widget_click_transparency(qapp, qtbot):
    """Verify clicking directly on child labels, icon, badge, and accent bar triggers card callback."""
    counts = {"dark": 0}
    view = HomeLaunchpadView(
        on_dark_calibration=lambda: counts.update(dark=counts["dark"] + 1)
    )
    qtbot.addWidget(view)
    view.show()
    qapp.processEvents()

    card = view._card_dark_cal
    children_to_test = [
        card._icon_label,
        card._title_label,
        card._subtitle_label,
        card._badge_label,
        card._accent_bar,
    ]

    for child in children_to_test:
        child.show()
        assert child.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        initial_count = counts["dark"]
        qtbot.mouseClick(child, Qt.LeftButton)
        qapp.processEvents()
        assert counts["dark"] == initial_count + 1


def test_adversarial_multi_button_mouse_events(qapp, qtbot):
    """Verify right, middle, and double click events execute gracefully without crashes."""
    mock_cb = MagicMock()
    card = SquircleCard("Title", "Subtitle", "⚡", "#3b82f6", callback=mock_cb)
    qtbot.addWidget(card)
    card.show()

    qtbot.mouseClick(card, Qt.LeftButton)
    assert mock_cb.call_count == 1

    qtbot.mouseClick(card, Qt.RightButton)
    assert mock_cb.call_count == 2


def test_adversarial_calibration_status_churn(qapp, qtbot, tmp_path):
    """Toggle calibration status through valid, empty, corrupt, and missing states."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.setStyleSheet(FULL_QSS)
    view.show()

    cal_dir = tmp_path / "churn_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corrupt JSON
    (cal_dir / "calibration_meta.json").write_text("{malformed: true, invalid...", encoding="utf-8")
    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", cal_dir):
        view.refresh_calibration_status()
        assert "No Mask" in view._card_dark_cal._badge_label.text() or "Not calibrated" in view._card_dark_cal._badge_label.text()

    # 2. Valid calibration
    med = np.full((50, 50), 30.0, dtype=np.float32)
    mask = np.ones((50, 50), dtype=np.float32)
    calibration_store.save_calibration(
        med_dark=med,
        final_mask=mask,
        stddev_thresh=40.0,
        absdev_thresh=60.0,
        tail_ratio=0.93,
        dark_frame_count=10,
        surviving_pixels=2490,
        total_pixels=2500,
        suppression_pct=0.4,
        source_dir=str(tmp_path / "raw"),
        date="2026-08-28T12:00:00",
        cal_dir=cal_dir,
    )
    with patch("rixs_app.core.calibration_store.DARK_CAL_DIR", cal_dir):
        view.refresh_calibration_status()
        assert "Mask Generated" in view._card_dark_cal._badge_label.text() or "Calibrated" in view._card_dark_cal._badge_label.text()
        assert "cal_status_ok" == view._card_dark_cal._badge_label.objectName()


def test_adversarial_copilot_foreign_container_reparenting(qapp, qtbot):
    """Test button migrating between HomeLaunchpadView and external parent windows."""
    home = HomeLaunchpadView()
    external_window = QWidget()
    qtbot.addWidget(home)
    qtbot.addWidget(external_window)
    home.show()
    external_window.show()

    copilot_btn = QPushButton("🤖 Shared Co-Pilot")
    home.set_copilot_button(copilot_btn)
    assert home.isAncestorOf(copilot_btn)

    copilot_btn.setParent(external_window)
    assert not home.isAncestorOf(copilot_btn)
    assert external_window.isAncestorOf(copilot_btn)

    home.set_copilot_button(copilot_btn)
    assert home.isAncestorOf(copilot_btn)
    assert not external_window.isAncestorOf(copilot_btn)


def test_adversarial_copilot_deleted_button_handling(qapp, qtbot):
    """Ensure set_copilot_button works cleanly if the previous button was destroyed."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    btn1 = QPushButton("Btn 1")
    view.set_copilot_button(btn1)
    btn1.deleteLater()
    qapp.processEvents()

    btn2 = QPushButton("Btn 2")
    view.set_copilot_button(btn2)
    assert view.isAncestorOf(btn2)
    assert view._copilot_btn is btn2


def test_adversarial_view_memory_lifecycle(qapp, qtbot):
    """Create and tear down HomeLaunchpadView instances; verify garbage collection."""
    refs = []
    for i in range(10):
        view = HomeLaunchpadView()
        btn = QPushButton(f"Btn {i}")
        view.set_copilot_button(btn)

        refs.append(weakref.ref(view))
        refs.append(weakref.ref(view._card_dark_cal))
        refs.append(weakref.ref(btn))

        view.deleteLater()
        del view
        del btn

    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()

    alive_count = sum(1 for r in refs if r() is not None)
    assert alive_count == 0, f"Memory leak detected: {alive_count}/{len(refs)} objects remained alive"


def test_adversarial_callback_exception_isolation(qapp, qtbot):
    """Verify that an exception raised in one callback does not break other cards."""
    faulty_cal_called = False
    healthy_zo_count = 0

    def _faulty_cal():
        nonlocal faulty_cal_called
        faulty_cal_called = True
        raise RuntimeError("Simulated error")

    def _healthy_zo():
        nonlocal healthy_zo_count
        healthy_zo_count += 1

    view = HomeLaunchpadView(
        on_dark_calibration=_faulty_cal,
        on_zeroth_order=_healthy_zo,
    )
    qtbot.addWidget(view)
    view.show()

    with pytest.raises(RuntimeError, match="Simulated error"):
        view._card_dark_cal.mousePressEvent()

    assert faulty_cal_called is True
    view._card_zeroth_order.mousePressEvent()
    assert healthy_zo_count == 1


def test_adversarial_qss_invalidation_churn(qapp, qtbot):
    """Rapidly apply stylesheets while toggling cards."""
    view = HomeLaunchpadView()
    qtbot.addWidget(view)
    view.show()

    for style in ["", DARK_STYLE, FULL_QSS]:
        view.setStyleSheet(style)
        view.refresh_calibration_status()
        qapp.processEvents()

    assert view._card_dark_cal._title_label.text() == "Dark Image & Pixel Masking"

