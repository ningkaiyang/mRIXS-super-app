"""Alignment slideshow tools panel — PySide6 port.

Houses manual-alignment controls, zoom in/out/reset buttons, a zoom level
label, and a per-frame info label.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from rixs_app.ui.theme import neutral_style


class SlideshowToolsPanel(QFrame):
    """Horizontal toolbar providing manual-line and zoom tools.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the tools panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.manual_line_button = QPushButton("\u270f Manual Line")
        self.manual_line_button.setFixedWidth(120)
        self.manual_line_button.setStyleSheet(neutral_style())
        self.manual_line_button.clicked.connect(self.controller.toggle_manual_mode)
        layout.addWidget(self.manual_line_button)

        self.clear_manual_button = QPushButton("Clear Manual")
        self.clear_manual_button.setFixedWidth(110)
        self.clear_manual_button.setStyleSheet(neutral_style())
        self.clear_manual_button.clicked.connect(self.controller.clear_manual_line)
        layout.addWidget(self.clear_manual_button)

        self.zoom_in_button = QPushButton("\U0001f50d+ Zoom In")
        self.zoom_in_button.setFixedWidth(110)
        self.zoom_in_button.setStyleSheet(neutral_style())
        self.zoom_in_button.clicked.connect(self.controller.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("\U0001f50d- Zoom Out")
        self.zoom_out_button.setFixedWidth(110)
        self.zoom_out_button.setStyleSheet(neutral_style())
        self.zoom_out_button.clicked.connect(self.controller.zoom_out)
        layout.addWidget(self.zoom_out_button)

        self.reset_view_button = QPushButton("\u27f2 Reset View")
        self.reset_view_button.setFixedWidth(110)
        self.reset_view_button.setStyleSheet(neutral_style())
        self.reset_view_button.clicked.connect(self.controller.reset_view)
        layout.addWidget(self.reset_view_button)

        self.zoom_label = QLabel("Zoom: 1\u00d7")
        layout.addWidget(self.zoom_label)

        layout.addStretch()

        self.frame_info_label = QLabel("")
        self.frame_info_label.setObjectName("dim_label")
        layout.addWidget(self.frame_info_label)

    def sync_zoom_label(self, factor) -> None:
        """Update the zoom multiplier display.

        Args:
            factor: Current zoom factor (e.g. 2 for 2\u00d7).
        """
        self.zoom_label.setText(f"Zoom: {factor}\u00d7")
