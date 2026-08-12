"""Alignment slideshow tools panel — PySide6 port.

Houses zoom in/out/reset buttons, a zoom level label, and manual-alignment
controls that are shown only when the PCA engine is active.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from rixs_app.ui.theme import set_tool_btn


class SlideshowToolsPanel(QFrame):
    """Horizontal toolbar providing zoom and manual-line tools.

    Manual-line buttons are positioned at the right end and hidden by
    default so they can appear/disappear when the engine changes without
    disturbing the zoom controls.

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

        # --- Zoom controls (always visible) ---
        self.zoom_in_button = QPushButton("\U0001f50d+ Zoom In")
        self.zoom_in_button.setFixedWidth(110)
        set_tool_btn(self.zoom_in_button)
        self.zoom_in_button.clicked.connect(self.controller.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("\U0001f50d- Zoom Out")
        self.zoom_out_button.setFixedWidth(110)
        set_tool_btn(self.zoom_out_button)
        self.zoom_out_button.clicked.connect(self.controller.zoom_out)
        layout.addWidget(self.zoom_out_button)

        self.reset_view_button = QPushButton("\u27f2 Reset View")
        self.reset_view_button.setFixedWidth(110)
        set_tool_btn(self.reset_view_button)
        self.reset_view_button.clicked.connect(self.controller.reset_view)
        layout.addWidget(self.reset_view_button)

        self.zoom_label = QLabel("Zoom: 1\u00d7")
        layout.addWidget(self.zoom_label)

        layout.addStretch()

        # --- Manual-line controls (right side, hidden by default for non-PCA) ---
        self.manual_line_button = QPushButton("\u270f Manual Line")
        self.manual_line_button.setFixedWidth(120)
        set_tool_btn(self.manual_line_button)
        self.manual_line_button.clicked.connect(self.controller.toggle_manual_mode)
        layout.addWidget(self.manual_line_button)

        self.clear_manual_button = QPushButton("Clear Manual")
        self.clear_manual_button.setFixedWidth(110)
        set_tool_btn(self.clear_manual_button)
        self.clear_manual_button.clicked.connect(self.controller.clear_manual_line)
        layout.addWidget(self.clear_manual_button)

        # Per-frame info label (far right)
        self.frame_info_label = QLabel("")
        self.frame_info_label.setObjectName("dim_label")
        layout.addWidget(self.frame_info_label)

        # Default: hide manual buttons (ECC is default engine)
        self.show_manual_buttons(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_manual_buttons(self, visible: bool) -> None:
        """Show or hide the manual-line alignment buttons.

        Called when the alignment engine changes — only PCA mode uses
        manual line overrides.

        Args:
            visible: True to show, False to hide.
        """
        self.manual_line_button.setVisible(visible)
        self.clear_manual_button.setVisible(visible)

    def sync_zoom_label(self, factor) -> None:
        """Update the zoom multiplier display.

        Args:
            factor: Current zoom factor (e.g. 2 for 2\u00d7).
        """
        self.zoom_label.setText(f"Zoom: {factor}\u00d7")
