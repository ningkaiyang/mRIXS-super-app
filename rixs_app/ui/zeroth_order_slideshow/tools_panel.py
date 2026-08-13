"""Zeroth-order slideshow tools panel — PySide6 port.

Houses zoom controls, intensity slicing RangeSlider, fitted-line toggle,
support-points toggle, extrapolation toggle, and energy dispersion input.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QCheckBox,
)
from PySide6.QtCore import Qt
from rixs_app.ui.theme import neutral_style
from rixs_app.ui.widgets import RangeSlider


class ZerothOrderToolsPanel(QFrame):
    """Full tools panel for zeroth-order calibration view.

    Provides zoom controls, intensity slicing (floor/ceiling via RangeSlider +
    text entries), and display toggle checkboxes (support points, extrapolation,
    fitted line).

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

        # --- Toggle checkboxes ---
        self.show_support_points_var = False  # backing value
        self.support_points_cb = QCheckBox("Show support points")
        self.support_points_cb.setChecked(False)
        self.support_points_cb.stateChanged.connect(
            lambda _: self.controller.load_and_render()
        )
        layout.addWidget(self.support_points_cb)

        self.show_extrapolation_var = False
        self.extrapolation_cb = QCheckBox("Show extrapolation")
        self.extrapolation_cb.setChecked(False)
        self.extrapolation_cb.stateChanged.connect(
            lambda _: self.controller.load_and_render()
        )
        layout.addWidget(self.extrapolation_cb)

        self.show_fitted_line_var = True
        self.fitted_line_cb = QCheckBox("Show fitted line")
        self.fitted_line_cb.setChecked(True)
        self.fitted_line_cb.stateChanged.connect(
            lambda _: self.controller.load_and_render()
        )
        layout.addWidget(self.fitted_line_cb)

    # ------------------------------------------------------------------
    # Checkbox value shims (old code accessed .get() on CTkBooleanVar)
    # ------------------------------------------------------------------

    @property
    def show_support_points_var(self):
        """Boolean property for support points visibility."""
        return _CheckboxShim(self, 'support_points_cb')

    @show_support_points_var.setter
    def show_support_points_var(self, val):
        """Setter ignored (backing value stored in checkbox widget)."""
        pass  # backing value stored in the QCheckBox

    @property
    def show_extrapolation_var(self):
        """Boolean property for extrapolation visibility."""
        return _CheckboxShim(self, 'extrapolation_cb')

    @show_extrapolation_var.setter
    def show_extrapolation_var(self, val):
        """Setter ignored (backing value stored in checkbox widget)."""
        pass

    @property
    def show_fitted_line_var(self):
        """Boolean property for fitted line visibility."""
        return _CheckboxShim(self, 'fitted_line_cb')

    @show_fitted_line_var.setter
    def show_fitted_line_var(self, val):
        """Setter ignored (backing value stored in checkbox widget)."""
        pass

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def sync_zoom_label(self, val) -> None:
        """Update the zoom factor display.

        Args:
            val: Current zoom factor (float or int).
        """
        self.zoom_label.setText(f"Zoom: {float(val):.1f}\u00d7")

    def sync_slicing_inputs(self, floor: float, ceiling: float) -> None:
        """Synchronise the floor/ceiling entries and range slider.

        Args:
            floor: Current floor intensity value.
            ceiling: Current ceiling intensity value.
        """
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


class _CheckboxShim:
    """Thin shim to make ``cb.get()`` work like a ``BooleanVar`` from Tkinter.

    The old zeroth-order controller reads
    ``self.tools_panel.show_support_points_var.get()``.
    This shim wraps the underlying ``QCheckBox`` so the same call works.

    Args:
        panel: The parent ZerothOrderToolsPanel.
        attr: String name of the QCheckBox attribute on the panel.
    """

    def __init__(self, panel, attr: str):
        """Initialise the shim.

        Args:
            panel: Parent ZerothOrderToolsPanel widget.
            attr: Name of the QCheckBox attribute.
        """
        self._panel = panel
        self._attr = attr

    def get(self) -> bool:
        """Return the current check state of the underlying QCheckBox.

        Returns:
            True if checked, False otherwise.
        """
        cb = getattr(self._panel, self._attr, None)
        if cb is None:
            return False
        return cb.isChecked()
