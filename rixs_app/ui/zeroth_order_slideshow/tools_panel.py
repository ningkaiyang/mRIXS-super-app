"""Zeroth-order slideshow tools panel — PySide6 port.

Houses zoom controls and intensity slicing RangeSlider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QLabel, QLineEdit,
)
from PySide6.QtCore import Qt, QEvent
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
        self.zoom_in_button.setFocusPolicy(Qt.NoFocus)
        set_tool_btn(self.zoom_in_button)
        self.zoom_in_button.clicked.connect(self.controller.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("\U0001f50d- Zoom Out")
        self.zoom_out_button.setFixedWidth(110)
        self.zoom_out_button.setFocusPolicy(Qt.NoFocus)
        set_tool_btn(self.zoom_out_button)
        self.zoom_out_button.clicked.connect(self.controller.zoom_out)
        layout.addWidget(self.zoom_out_button)

        self.reset_view_button = QPushButton("\u27f2 Reset View")
        self.reset_view_button.setFixedWidth(110)
        self.reset_view_button.setFocusPolicy(Qt.NoFocus)
        set_tool_btn(self.reset_view_button)
        self.reset_view_button.clicked.connect(self.controller.reset_view)
        layout.addWidget(self.reset_view_button)

        self.zoom_label = QLabel("Zoom: 1.0\u00d7")
        layout.addWidget(self.zoom_label)

        # --- Intensity slicing ---
        layout.addWidget(QLabel("Slicing:"))

        self.floor_entry = QLineEdit()
        self.floor_entry.setFixedWidth(80)
        self.floor_entry.setFocusPolicy(Qt.ClickFocus)
        self.floor_entry.returnPressed.connect(self._on_floor_submit)
        self.floor_entry.installEventFilter(self)
        layout.addWidget(self.floor_entry)

        self.range_slider = RangeSlider(self)
        self.range_slider.range_changed.connect(self.controller.handle_slicing_change)
        self.range_slider.slider_released.connect(self.controller.handle_slicing_release)
        self.range_slider.setMinimumWidth(120)
        layout.addWidget(self.range_slider, stretch=1)

        self.ceiling_entry = QLineEdit()
        self.ceiling_entry.setFixedWidth(80)
        self.ceiling_entry.setFocusPolicy(Qt.ClickFocus)
        self.ceiling_entry.returnPressed.connect(self._on_ceiling_submit)
        self.ceiling_entry.installEventFilter(self)
        layout.addWidget(self.ceiling_entry)

        # --- Energy Dispersion ---
        layout.addWidget(QLabel("Dispersion:"))
        self.dispersion_entry = QLineEdit()
        self.dispersion_entry.setFixedWidth(70)
        self.dispersion_entry.setFocusPolicy(Qt.ClickFocus)
        self.dispersion_entry.setPlaceholderText("0.00")
        self.dispersion_entry.setToolTip("Energy dispersion scale (meV/pixel) to compute Resolving Power R")
        self.dispersion_entry.returnPressed.connect(self._on_dispersion_submit)
        self.dispersion_entry.editingFinished.connect(self._on_dispersion_submit)
        self.dispersion_entry.installEventFilter(self)
        layout.addWidget(self.dispersion_entry)
        layout.addWidget(QLabel("meV/px"))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Intercept Escape key to clear focus and restore frame navigation."""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter):
                if isinstance(watched, QLineEdit):
                    watched.clearFocus()
                    if hasattr(self.controller, "setFocus"):
                        self.controller.setFocus()
                    if event.key() == Qt.Key_Escape:
                        return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Clear text input focus when clicking tools panel background."""
        self.setFocus()
        if hasattr(self.controller, "setFocus"):
            self.controller.setFocus()
        super().mousePressEvent(event)

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

    def sync_dispersion_input(self, val: float) -> None:
        """Synchronise the energy dispersion input field."""
        if val > 0:
            self.dispersion_entry.setText(f"{val:.2f}")
        else:
            self.dispersion_entry.setText("")

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_floor_submit(self) -> None:
        """Handle return-key in the floor entry."""
        self.floor_entry.clearFocus()
        self.controller.handle_floor_entry_submit(self.floor_entry.text())
        if hasattr(self.controller, "setFocus"):
            self.controller.setFocus()

    def _on_ceiling_submit(self) -> None:
        """Handle return-key in the ceiling entry."""
        self.ceiling_entry.clearFocus()
        self.controller.handle_ceiling_entry_submit(self.ceiling_entry.text())
        if hasattr(self.controller, "setFocus"):
            self.controller.setFocus()

    def _on_dispersion_submit(self) -> None:
        """Handle return-key or editingFinished in the dispersion entry."""
        text = self.dispersion_entry.text().strip()
        try:
            val = float(text) if text else 0.0
            val = max(0.0, val)
        except ValueError:
            val = 0.0
        if val > 0:
            self.dispersion_entry.setText(f"{val:.2f}")
        else:
            self.dispersion_entry.setText("")
        self.dispersion_entry.clearFocus()
        self.controller.set_energy_dispersion(val)
        if hasattr(self.controller, "setFocus"):
            self.controller.setFocus()
