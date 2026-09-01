"""Home Launchpad Hub view for RIXS Super-App.

Provides an iOS-inspired 2x2 squircle grid dashboard directing users to:
1. Detector Dark Frame Calibration (Amber #fbbf24)
2. Zeroth-Order Mirror Pitch Calibration (Teal #14b8a6)
3. Single-Photon Event Clustering (Blue #3b82f6)
4. Spatial Drift Alignment & Stacking (Green #059669)

Features dynamic dark calibration status badging and Co-Pilot button docking.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from rixs_app.core import dark_mask_store, calibration_store


class SquircleCard(QFrame):
    """An iOS-inspired squircle card container for the Home Launchpad."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        icon: str,
        accent_color: str,
        badge_text: str | None = None,
        badge_ok: bool = True,
        callback: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("squircle_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(280, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._callback = callback
        self._accent_color = accent_color

        # Card layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        # Header row: Icon on left, Badge on right
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self._icon_label = QLabel(icon, self)
        self._icon_label.setObjectName("squircle_card_icon")
        self._icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_layout.addWidget(self._icon_label)

        header_layout.addStretch(1)

        self._badge_label = QLabel(self)
        self._badge_label.setObjectName("squircle_card_badge")
        self._badge_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if badge_text:
            self.set_badge(badge_text, is_ok=badge_ok)
        else:
            self._badge_label.hide()
        header_layout.addWidget(self._badge_label)

        layout.addLayout(header_layout)

        # Title
        self._title_label = QLabel(title, self)
        self._title_label.setObjectName("squircle_card_title")
        self._title_label.setWordWrap(True)
        self._title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._title_label)

        # Subtitle
        self._subtitle_label = QLabel(subtitle, self)
        self._subtitle_label.setObjectName("squircle_card_subtitle")
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._subtitle_label)

        layout.addStretch(1)

        # Accent bar indicator along the bottom
        self._accent_bar = QFrame(self)
        self._accent_bar.setFixedHeight(3)
        self._accent_bar.setStyleSheet(
            f"background-color: {accent_color}; border-radius: 1.5px; border: none;"
        )
        self._accent_bar.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._accent_bar)

    def set_subtitle(self, text: str) -> None:
        """Update subtitle text dynamically."""
        self._subtitle_label.setText(text)

    def set_badge(self, text: str, is_ok: bool = True) -> None:
        """Update badge text and status styling."""
        self._badge_label.setText(text)
        self._badge_label.show()
        if is_ok:
            self._badge_label.setObjectName("cal_status_ok")
        else:
            self._badge_label.setObjectName("cal_status_missing")
        self._badge_label.style().unpolish(self._badge_label)
        self._badge_label.style().polish(self._badge_label)

    def mousePressEvent(self, event: QMouseEvent | None = None) -> None:  # noqa: N802
        if self._callback is not None:
            self._callback()
        if event is not None:
            try:
                super().mousePressEvent(event)
            except Exception:
                pass


class HomeLaunchpadView(QWidget):
    """Home Launchpad Hub 2x2 grid dashboard.

    Args:
        parent: Optional parent QWidget.
        on_dark_calibration: Callback for Card 1 (Dark Image & Pixel Masking).
        on_zeroth_order: Callback for Card 2 (Zeroth-Order Calibration).
        on_clustering: Callback for Card 3 (Single-Photon Event Clustering).
        on_alignment: Callback for Card 4 (Spatial Drift Alignment).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        on_dark_calibration: Callable[[], None] | None = None,
        on_zeroth_order: Callable[[], None] | None = None,
        on_clustering: Callable[[], None] | None = None,
        on_alignment: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("home_launchpad_view")
        self.on_dark_calibration = on_dark_calibration
        self.on_zeroth_order = on_zeroth_order
        self.on_clustering = on_clustering
        self.on_alignment = on_alignment

        self._copilot_btn: QPushButton | None = None

        self._init_ui()
        self.refresh_calibration_status()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        # ── Header Row ──
        self._header_layout = QHBoxLayout()
        self._header_layout.setContentsMargins(0, 0, 0, 0)

        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        app_title = QLabel("RIXS Super-App", self)
        app_title.setObjectName("header_title")

        app_subtitle = QLabel(
            "Offline Spectroscopy Image Processing & Calibration Suite", self
        )
        app_subtitle.setObjectName("dim_label")

        title_col.addWidget(app_title)
        title_col.addWidget(app_subtitle)
        self._header_layout.addLayout(title_col)

        self._header_layout.addStretch(1)

        # Docking placeholder container for Co-Pilot button
        self._copilot_container = QWidget(self)
        self._copilot_container_layout = QHBoxLayout(self._copilot_container)
        self._copilot_container_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.addWidget(self._copilot_container)

        main_layout.addLayout(self._header_layout)

        # ── 2x2 Grid ──
        grid_container = QWidget(self)
        self._grid_layout = QGridLayout(grid_container)
        self._grid_layout.setContentsMargins(0, 8, 0, 8)
        self._grid_layout.setSpacing(20)

        # 1. Dark Image & Pixel Masking (Amber #fbbf24)
        self._card_dark_cal = SquircleCard(
            title="Dark Image & Pixel Masking",
            subtitle="⚠️ No Mask Generated — Run dark masking first",
            icon="🛡️",
            accent_color="#fbbf24",
            badge_text="⚠️ No Mask",
            badge_ok=False,
            callback=self._handle_dark_calibration,
            parent=self,
        )

        # 2. Zeroth-Order Mirror Pitch Calibration (Teal #14b8a6)
        self._card_zeroth_order = SquircleCard(
            title="Zeroth-Order Mirror Pitch Calibration",
            subtitle="Mirror pitch focus evaluation",
            icon="📐",
            accent_color="#14b8a6",
            badge_text="FWHM Focus",
            badge_ok=True,
            callback=self._handle_zeroth_order,
            parent=self,
        )

        # 3. Single-Photon Event Clustering (Blue #3b82f6)
        self._card_clustering = SquircleCard(
            title="Single-Photon Event Clustering",
            subtitle="Photon clustering & energy gating",
            icon="⚡",
            accent_color="#3b82f6",
            badge_text="3-Mode Studio",
            badge_ok=True,
            callback=self._handle_clustering,
            parent=self,
        )

        # 4. Spatial Drift Alignment & Stacking (Green #059669)
        self._card_alignment = SquircleCard(
            title="Spatial Drift Alignment & Stacking",
            subtitle="Sub-pixel 2D image registration via SVD/ECC",
            icon="🔄",
            accent_color="#059669",
            badge_text="SVD / ECC / Phase",
            badge_ok=True,
            callback=self._handle_alignment,
            parent=self,
        )

        # Add to 2x2 grid
        self._grid_layout.addWidget(self._card_dark_cal, 0, 0)
        self._grid_layout.addWidget(self._card_zeroth_order, 0, 1)
        self._grid_layout.addWidget(self._card_clustering, 1, 0)
        self._grid_layout.addWidget(self._card_alignment, 1, 1)

        main_layout.addWidget(grid_container)
        main_layout.addStretch(1)

    def _handle_dark_calibration(self) -> None:
        if self.on_dark_calibration is not None:
            self.on_dark_calibration()

    def _handle_zeroth_order(self) -> None:
        if self.on_zeroth_order is not None:
            self.on_zeroth_order()

    def _handle_clustering(self) -> None:
        if self.on_clustering is not None:
            self.on_clustering()

    def _handle_alignment(self) -> None:
        if self.on_alignment is not None:
            self.on_alignment()

    def refresh_calibration_status(self) -> None:
        """Query dark_mask_store/calibration_store and dynamically update the Dark Mask card subtitle and badge."""
        summary = None
        try:
            cal_dir = getattr(calibration_store, "DARK_CAL_DIR", None)
            mask_dir = getattr(dark_mask_store, "DARK_MASK_DIR", None)
            if cal_dir is not None:
                summary = calibration_store.get_calibration_summary(cal_dir=cal_dir)
            elif mask_dir is not None:
                summary = dark_mask_store.get_mask_summary(mask_dir=mask_dir)
        except Exception:
            summary = None

        if summary:
            self._card_dark_cal.set_subtitle(summary)
            self._card_dark_cal.set_badge("✓ Mask Generated", is_ok=True)
        else:
            self._card_dark_cal.set_subtitle(
                "⚠️ No Mask Generated — Run dark masking first"
            )
            self._card_dark_cal.set_badge("⚠️ No Mask", is_ok=False)

    def set_copilot_button(self, btn: QPushButton) -> None:
        """Dock the Co-Pilot toggle button into the header row."""
        if self._copilot_btn is not None and self._copilot_btn is not btn:
            self._copilot_container_layout.removeWidget(self._copilot_btn)
            self._copilot_btn.setParent(None)

        self._copilot_btn = btn
        self._copilot_container_layout.addWidget(btn)
        btn.setParent(self._copilot_container)
        btn.show()
