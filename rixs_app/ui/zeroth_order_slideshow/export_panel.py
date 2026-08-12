"""Zeroth-order slideshow export panel — PySide6 port."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from rixs_app.ui.theme import accent_style


class ZerothOrderExportPanel(QFrame):
    """Bottom export bar for the zeroth-order slideshow.

    Args:
        parent: Parent widget.
        controller: ZerothOrderSlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the export panel.

        Args:
            parent: Parent QWidget.
            controller: ZerothOrderSlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setObjectName("muted_label")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.export_button = QPushButton("\U0001f4be Export Calibration")
        self.export_button.setFixedSize(200, 35)
        self.export_button.setStyleSheet(accent_style())
        self.export_button.clicked.connect(self.controller.trigger_export)
        layout.addWidget(self.export_button)
