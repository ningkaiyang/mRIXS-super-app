"""Alignment slideshow intensity clamping panel — PySide6 port.

Provides a dual-handle ``RangeSlider`` with flanking ``QLineEdit`` inputs
for numeric entry of floor and ceiling clamping values.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit
from PySide6.QtCore import Qt

from rixs_app.ui.widgets import RangeSlider


class SlideshowClampingPanel(QFrame):
    """Horizontal intensity clamping control bar.

    Provides a ``RangeSlider`` for interactive clamping with synchronised
    text-entry fields at each end.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the clamping panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Intensity Clamping:"))

        self.floor_entry = QLineEdit()
        self.floor_entry.setFixedWidth(90)
        self.floor_entry.returnPressed.connect(self._on_floor_submit)
        layout.addWidget(self.floor_entry)

        self.range_slider = RangeSlider(self, command=self.controller.handle_clamping_change)
        self.range_slider.setMinimumWidth(120)
        layout.addWidget(self.range_slider, stretch=1)

        self.ceiling_entry = QLineEdit()
        self.ceiling_entry.setFixedWidth(90)
        self.ceiling_entry.returnPressed.connect(self._on_ceiling_submit)
        layout.addWidget(self.ceiling_entry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_clamping_limits(self, intensity_min: float, intensity_max: float) -> None:
        """Configure the absolute min/max boundaries of the range slider.

        Args:
            intensity_min: Minimum valid intensity value.
            intensity_max: Maximum valid intensity value.
        """
        self.range_slider.configure_range(intensity_min, intensity_max)

    def sync_clamping_inputs(self, floor: float, ceiling: float) -> None:
        """Synchronise slider and entry fields to the given clamping values.

        Args:
            floor: Current lower clamp value.
            ceiling: Current upper clamp value.
        """
        self.floor_entry.setText(f"{floor:.4f}")
        self.ceiling_entry.setText(f"{ceiling:.4f}")
        self.range_slider.set_values(floor, ceiling)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_floor_submit(self) -> None:
        """Handle return-key submission of the floor entry."""
        self.controller.handle_floor_entry_submit(self.floor_entry.text())
        self.floor_entry.clearFocus()

    def _on_ceiling_submit(self) -> None:
        """Handle return-key submission of the ceiling entry."""
        self.controller.handle_ceiling_entry_submit(self.ceiling_entry.text())
        self.ceiling_entry.clearFocus()
