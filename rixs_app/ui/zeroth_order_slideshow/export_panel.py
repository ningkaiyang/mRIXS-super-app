"""Zeroth-order slideshow export and display settings panel — PySide6 port."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QLabel
from rixs_app.ui.theme import set_accent_btn, set_play_btn, set_danger_btn


class ZerothOrderExportPanel(QFrame):
    """Bottom bar for zeroth-order slideshow housing display toggles and export button.

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

        # Display toggle states (default: line on, points & extrap off)
        self.show_support_points: bool = False
        self.show_extrapolation: bool = False
        self.show_fitted_line: bool = True

        self.setFixedHeight(46)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setObjectName("muted_label")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # --- Display toggle buttons ---
        self.support_points_button = QPushButton("Support Points: OFF")
        self.support_points_button.setFixedSize(155, 32)
        self.support_points_button.clicked.connect(self._toggle_support_points)
        layout.addWidget(self.support_points_button)

        self.extrapolation_button = QPushButton("Extrapolation: OFF")
        self.extrapolation_button.setFixedSize(155, 32)
        self.extrapolation_button.clicked.connect(self._toggle_extrapolation)
        layout.addWidget(self.extrapolation_button)

        self.fitted_line_button = QPushButton("Fitted Line: ON")
        self.fitted_line_button.setFixedSize(145, 32)
        self.fitted_line_button.clicked.connect(self._toggle_fitted_line)
        layout.addWidget(self.fitted_line_button)

        self.update_button_styles()

        # --- Export Calibration button ---
        self.export_button = QPushButton("\U0001f4be Export Calibration")
        self.export_button.setFixedSize(185, 32)
        set_accent_btn(self.export_button)
        self.export_button.clicked.connect(self.controller.trigger_export)
        layout.addWidget(self.export_button)

    def update_button_styles(self) -> None:
        """Update text and green/red state styles for the 3 display toggle buttons."""
        # Support Points
        if self.show_support_points:
            self.support_points_button.setText("Support Points: ON")
            set_play_btn(self.support_points_button)
        else:
            self.support_points_button.setText("Support Points: OFF")
            set_danger_btn(self.support_points_button)

        # Extrapolation
        if self.show_extrapolation:
            self.extrapolation_button.setText("Extrapolation: ON")
            set_play_btn(self.extrapolation_button)
        else:
            self.extrapolation_button.setText("Extrapolation: OFF")
            set_danger_btn(self.extrapolation_button)

        # Fitted Line
        if self.show_fitted_line:
            self.fitted_line_button.setText("Fitted Line: ON")
            set_play_btn(self.fitted_line_button)
        else:
            self.fitted_line_button.setText("Fitted Line: OFF")
            set_danger_btn(self.fitted_line_button)

    def _toggle_support_points(self) -> None:
        """Toggle support points display and re-render canvas."""
        self.show_support_points = not self.show_support_points
        self.update_button_styles()
        self.controller.load_and_render()

    def _toggle_extrapolation(self) -> None:
        """Toggle extrapolation display and re-render canvas."""
        self.show_extrapolation = not self.show_extrapolation
        self.update_button_styles()
        self.controller.load_and_render()

    def _toggle_fitted_line(self) -> None:
        """Toggle fitted line display and re-render canvas."""
        self.show_fitted_line = not self.show_fitted_line
        self.update_button_styles()
        self.controller.load_and_render()
