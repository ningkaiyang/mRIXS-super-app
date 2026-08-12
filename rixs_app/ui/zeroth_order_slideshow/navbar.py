"""Zeroth-order slideshow navigation bar — PySide6 port.

Provides back/prev/next navigation, autoplay toggle, pipeline-stage selector,
colormap selector, precompute button, and peak-focus jump button.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QComboBox, QLabel,
)

from rixs_app.ui.theme import PALETTE, success_style, neutral_style


class ZerothOrderNavBar(QFrame):
    """Top navigation bar for the zeroth-order calibration slideshow.

    Args:
        parent: Parent widget.
        controller: The ``ZerothOrderSlideshowView`` controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the zeroth-order navigation bar.

        Args:
            parent: Parent QWidget.
            controller: ZerothOrderSlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("navbar_frame")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # --- Navigation ---
        self.back_button = QPushButton("\u25c4 Back")
        self.back_button.setFixedWidth(80)
        self.back_button.clicked.connect(self.controller.back_to_sorting)
        layout.addWidget(self.back_button)

        self.prev_button = QPushButton("\u25c4 Prev")
        self.prev_button.setFixedWidth(80)
        self.prev_button.clicked.connect(self.controller.prev_frame)
        layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next \u25ba")
        self.next_button.setFixedWidth(80)
        self.next_button.clicked.connect(self.controller.next_frame)
        layout.addWidget(self.next_button)

        # --- Autoplay ---
        self.autoplay_button = QPushButton("\u25ba Play")
        self.autoplay_button.setFixedWidth(80)
        self.autoplay_button.setStyleSheet(success_style())
        self.autoplay_button.clicked.connect(self.controller.toggle_autoplay)
        layout.addWidget(self.autoplay_button)

        # --- Precompute ---
        self.precompute_button = QPushButton("Precompute All")
        self.precompute_button.setFixedWidth(130)
        self.precompute_button.setStyleSheet(neutral_style())
        self.precompute_button.clicked.connect(self.controller.trigger_precompute)
        layout.addWidget(self.precompute_button)

        # --- Peak focus ---
        self.peak_focus_button = QPushButton("\u26a1 Best Focus")
        self.peak_focus_button.setFixedWidth(110)
        self.peak_focus_button.setStyleSheet(neutral_style())
        self.peak_focus_button.clicked.connect(self.controller.jump_to_peak_focus)
        layout.addWidget(self.peak_focus_button)

        layout.addStretch()

        # --- Pipeline stage ---
        layout.addWidget(QLabel("Stage:"))
        self.stage_menu = QComboBox()
        self.stage_menu.addItems([
            "Raw", "Denoised (D)", "Row-Smoothed (Dsm)", "Gradient (G)", "Fitted-Line Strip"
        ])
        self.stage_menu.setCurrentText("Raw")
        self.stage_menu.currentTextChanged.connect(self.controller.change_pipeline_stage)
        layout.addWidget(self.stage_menu)

        # --- Colormap ---
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_menu = QComboBox()
        self.colormap_menu.addItems(["viridis", "inferno", "plasma", "magma", "grayscale"])
        self.colormap_menu.setCurrentText("viridis")
        self.colormap_menu.currentTextChanged.connect(self.controller.change_colormap)
        layout.addWidget(self.colormap_menu)
