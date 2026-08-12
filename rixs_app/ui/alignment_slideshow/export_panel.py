"""Alignment slideshow export panel — PySide6 port.

A compact bottom bar showing export progress and a compare-and-save button.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from rixs_app.ui.theme import PALETTE, accent_style


class SlideshowExportPanel(QFrame):
    """Bottom action bar for triggering the export comparison workflow.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the export panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("muted_label")
        layout.addWidget(self.progress_label)

        layout.addStretch()

        self.export_button = QPushButton("\U0001f4be Compare and Save")
        self.export_button.setFixedSize(200, 35)
        self.export_button.setStyleSheet(accent_style())
        self.export_button.clicked.connect(self.controller.trigger_export)
        layout.addWidget(self.export_button)
