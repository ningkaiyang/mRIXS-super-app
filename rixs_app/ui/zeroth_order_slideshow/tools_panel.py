"""Zeroth-order slideshow tools panel — PySide6 port.

Houses zoom controls and intensity slicing RangeSlider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QLabel, QLineEdit,
)
from PySide6.QtCore import Qt
from rixs_app.ui.theme import set_tool_btn
from rixs_app.ui.widgets import RangeSlider


class ZerothOrderToolsPanel(QFrame):
    """Middle toolbar for zeroth-order calibration view.

    Provides zoom controls and intensity slicing (floor/ceiling via RangeSlider +
    text entries).

    Args:
        parent: Parent widget.
        controller: ZerothOrderSlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the tools panel.

        Args:
            parent: Parent QWidget.
            controller: ZerothOrderSlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # --- Zoom controls ---
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

        self.zoom_label = QLabel("Zoom: 1.0\u00d7")
        layout.addWidget(self.zoom_label)

        # --- Intensity slicing ---
        layout.addWidget(QLabel("Slicing:"))

        self.floor_entry = QLineEdit()
        self.floor_entry.setFixedWidth(80)
        self.floor_entry.returnPressed.connect(self._on_floor_submit)
        layout.addWidget(self.floor_entry)

        self.range_slider = RangeSlider(self, command=self.controller.handle_slicing_change)
        self.range_slider.slider_released.connect(self.controller.handle_slicing_release)
        self.range_slider.setMinimumWidth(120)
        layout.addWidget(self.range_slider, stretch=1)

        self.ceiling_entry = QLineEdit()
        self.ceiling_entry.setFixedWidth(80)
        self.ceiling_entry.returnPressed.connect(self._on_ceiling_submit)
        layout.addWidget(self.ceiling_entry)

    # ------------------------------------------------------------------
    # Compatibility properties for legacy code/tests
    # ------------------------------------------------------------------

    @property
    def support_points_cb(self):
        """Backwards compatibility proxy object for support points toggle."""
        panel = self
        class _Proxy:
            def isChecked(self):
                return panel.controller.bottom_bar.show_support_points
        return _Proxy()

    @property
    def extrapolation_cb(self):
        """Backwards compatibility proxy object for extrapolation toggle."""
        panel = self
        class _Proxy:
            def isChecked(self):
                return panel.controller.bottom_bar.show_extrapolation
        return _Proxy()

    @property
    def fitted_line_cb(self):
        """Backwards compatibility proxy object for fitted line toggle."""
        panel = self
        class _Proxy:
            def isChecked(self):
                return panel.controller.bottom_bar.show_fitted_line
        return _Proxy()

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def sync_zoom_label(self, val) -> None:
        """Update the zoom factor display."""
        self.zoom_label.setText(f"Zoom: {float(val):.1f}\u00d7")

    def sync_slicing_inputs(self, floor: float, ceiling: float) -> None:
        """Synchronise the floor/ceiling entries and range slider."""
        self.floor_entry.setText(f"{floor:.4f}")
        self.ceiling_entry.setText(f"{ceiling:.4f}")
        self.range_slider.set_values(floor, ceiling)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_floor_submit(self) -> None:
        """Handle return-key in the floor entry."""
        self.controller.handle_floor_entry_submit(self.floor_entry.text())

    def _on_ceiling_submit(self) -> None:
        """Handle return-key in the ceiling entry."""
        self.controller.handle_ceiling_entry_submit(self.ceiling_entry.text())
