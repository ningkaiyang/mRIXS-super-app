"""Alignment slideshow navigation bar — PySide6 port.

Provides back/prev/next navigation, autoplay toggle, warp image switch,
colormap selector, and alignment engine selector in a compact horizontal
toolbar row.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QComboBox, QCheckBox, QLabel
from PySide6.QtCore import Qt

from rixs_app.ui.theme import set_play_btn


class SlideshowNavBar(QFrame):
    """Top navigation bar for the alignment slideshow view.

    Contains frame navigation buttons, autoplay toggle, warp toggle,
    colormap dropdown, and alignment engine dropdown.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` instance used to dispatch actions.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the navigation bar.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("navbar_frame")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 16, 4)
        layout.setSpacing(12)

        # --- Left controls ---
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

        self.autoplay_button = QPushButton("\u25ba Play")
        self.autoplay_button.setFixedWidth(80)
        set_play_btn(self.autoplay_button)
        self.autoplay_button.clicked.connect(self.controller.toggle_autoplay)
        layout.addWidget(self.autoplay_button)

        layout.addStretch()

        # --- Right controls ---
        layout.addWidget(QLabel("Engine:"))

        self.engine_menu = QComboBox()
        self.engine_menu.setObjectName("engine_menu")
        engine_options = ["PCA", "ECC", "Phase Correlation"]
        self.engine_menu.addItems(engine_options)
        for idx, text in enumerate(engine_options):
            self.engine_menu.setItemData(idx, text, Qt.ToolTipRole)
        self.engine_menu.setCurrentText("ECC")
        self.engine_menu.setMinimumWidth(185)
        self.engine_menu.view().setMinimumWidth(185)
        self.engine_menu.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.engine_menu.currentTextChanged.connect(self.controller.change_engine)
        layout.addWidget(self.engine_menu)

        layout.addWidget(QLabel("Colormap:"))

        self.colormap_menu = QComboBox()
        colormap_options = ["viridis", "inferno", "plasma", "magma", "grayscale"]
        self.colormap_menu.addItems(colormap_options)
        for idx, text in enumerate(colormap_options):
            self.colormap_menu.setItemData(idx, text, Qt.ToolTipRole)
        self.colormap_menu.setCurrentText("viridis")
        self.colormap_menu.setMinimumWidth(100)
        self.colormap_menu.currentTextChanged.connect(self.controller.change_colormap)
        layout.addWidget(self.colormap_menu)

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    @property
    def warp_button(self):
        """Proxy to control_panel.warp_button."""
        return self.controller.control_panel.warp_button


