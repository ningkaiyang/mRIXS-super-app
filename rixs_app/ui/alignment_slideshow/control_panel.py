"""Alignment slideshow control panel — PySide6 port.

Contains engine-specific settings panels (PCA, ECC, Phase Correlation)
and the timeline frame scrub slider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSlider,
    QLineEdit, QCheckBox, QStackedWidget, QWidget,
)
from PySide6.QtCore import Qt
from rixs_app.ui.theme import PALETTE, set_tool_btn, set_accent_btn, set_success_btn


class PcaSettingsPanel(QWidget):
    """Settings panel for the PCA alignment engine.

    Provides a threshold slider, numeric entry, auto-snap buttons,
    and a 'Show Ref Line' toggle checkbox.

    Args:
        parent: Parent widget.
        controller: SlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the PCA settings panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.pca_label = QLabel("PCA Threshold: 99.9000%")
        layout.addWidget(self.pca_label)

        self.pca_slider = QSlider(Qt.Horizontal)
        self.pca_slider.setMinimum(95000)
        self.pca_slider.setMaximum(99999)
        self.pca_slider.setValue(99900)
        self.pca_slider.valueChanged.connect(self._on_slider_move)
        layout.addWidget(self.pca_slider, stretch=1)

        self.pca_entry = QLineEdit("99.9000")
        self.pca_entry.setFixedWidth(90)
        self.pca_entry.returnPressed.connect(self._on_entry_submit)
        layout.addWidget(self.pca_entry)

        self.auto_snap_button = QPushButton("Auto")
        self.auto_snap_button.setFixedWidth(55)
        set_tool_btn(self.auto_snap_button)
        self.auto_snap_button.clicked.connect(self.controller.trigger_auto_snap)
        layout.addWidget(self.auto_snap_button)

        self.auto_all_button = QPushButton("Auto All")
        self.auto_all_button.setFixedWidth(75)
        set_accent_btn(self.auto_all_button)
        self.auto_all_button.clicked.connect(self.controller.trigger_auto_snap_all)
        layout.addWidget(self.auto_all_button)

        # Show line toggle (right-aligned)
        self.show_line_switch = QCheckBox("Show Ref Line")
        self.show_line_switch.setChecked(True)
        self.show_line_switch.stateChanged.connect(self.controller._render_display)
        layout.addWidget(self.show_line_switch)

    # ------------------------------------------------------------------
    # Slider value <-> float threshold conversion
    # Slider range: 95000..99999 represents 95.000..99.999 %
    # ------------------------------------------------------------------

    @staticmethod
    def _slider_to_float(val: int) -> float:
        """Convert slider integer position to float percentage.

        Args:
            val: Slider integer in range [95000, 99999].

        Returns:
            Float threshold percentage, e.g. 99.9.
        """
        return val / 1000.0

    @staticmethod
    def _float_to_slider(t: float) -> int:
        """Convert float percentage threshold to slider integer.

        Args:
            t: Float threshold, e.g. 99.9.

        Returns:
            Slider integer.
        """
        return int(min(99999, max(95000, round(t * 1000))))

    def _on_slider_move(self, val: int) -> None:
        """Handle PCA slider drag without applying (debounced by controller).

        Args:
            val: Slider integer position.
        """
        t = self._slider_to_float(val)
        self.controller.handle_pca_slider_drag(t)

    def _on_entry_submit(self) -> None:
        """Handle PCA entry box return-key submission."""
        self.controller.handle_pca_entry_submit(self.pca_entry.text())

    # ------------------------------------------------------------------
    # Public sync helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(t: float) -> str:
        """Format threshold for display, stripping trailing zeros.

        Args:
            t: Float threshold value.

        Returns:
            Formatted string like '99.9' or '95.05'.
        """
        s = f"{t:.4f}".rstrip('0')
        return s if not s.endswith('.') else s + '0'

    def sync_pca_elements(self, t: float) -> None:
        """Synchronise slider, label, and entry to threshold t.

        Args:
            t: New threshold value.
        """
        self.pca_slider.blockSignals(True)
        self.pca_slider.setValue(self._float_to_slider(t))
        self.pca_slider.blockSignals(False)
        self.pca_label.setText(f"PCA Threshold: {self._fmt(t)}%")
        self.pca_entry.setText(f"{t:.4f}")

    def sync_pca_label_and_entry(self, t: float) -> None:
        """Synchronise only the label and entry (during live drag).

        Args:
            t: New threshold value.
        """
        self.pca_label.setText(f"PCA Threshold: {self._fmt(t)}%")
        self.pca_entry.setText(f"{t:.4f}")

    def set_ui_state(self, enabled: bool) -> None:
        """Enable or disable all interactive elements.

        Args:
            enabled: True to enable, False to disable.
        """
        self.pca_slider.setEnabled(enabled)
        self.pca_entry.setEnabled(enabled)
        self.auto_snap_button.setEnabled(enabled)
        self.auto_all_button.setEnabled(enabled)
        self.show_line_switch.setEnabled(enabled)


class EccSettingsPanel(QWidget):
    """Settings panel for the ECC alignment engine.

    Shows an informational label and a 'Precompute All' button.

    Args:
        parent: Parent widget.
        controller: SlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the ECC settings panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(
            QLabel("ECC uses automatic coarse-to-fine pyramiding."), stretch=1
        )

        self.precompute_button = QPushButton("Precompute All")
        self.precompute_button.setFixedWidth(130)
        set_accent_btn(self.precompute_button)
        self.precompute_button.clicked.connect(self.controller.trigger_auto_snap_all)
        layout.addWidget(self.precompute_button)

    def set_ui_state(self, enabled: bool) -> None:
        """Enable or disable the precompute button.

        Args:
            enabled: True to enable, False to disable.
        """
        self.precompute_button.setEnabled(enabled)


class PhaseCorrelationSettingsPanel(QWidget):
    """Settings panel for the Phase Correlation alignment engine.

    Args:
        parent: Parent widget.
        controller: SlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the phase correlation settings panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(
            QLabel(
                "Phase Correlation uses Fourier-domain cross-correlation for "
                "sub-pixel drift estimation."
            ),
            stretch=1,
        )

        self.precompute_button = QPushButton("Precompute All")
        self.precompute_button.setFixedWidth(130)
        set_accent_btn(self.precompute_button)
        self.precompute_button.clicked.connect(self.controller.trigger_auto_snap_all)
        layout.addWidget(self.precompute_button)

    def set_ui_state(self, enabled: bool) -> None:
        """Enable or disable the precompute button.

        Args:
            enabled: True to enable, False to disable.
        """
        self.precompute_button.setEnabled(enabled)


class SlideshowControlPanel(QFrame):
    """Compound control panel with engine settings and frame scrub slider.

    Uses a ``QStackedWidget`` to swap engine-specific panels (PCA / ECC /
    Phase Correlation) without recreating them.

    Args:
        parent: Parent widget.
        controller: SlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the control panel.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        # Engine settings stack + Warp button row -------------------------------
        engine_row = QFrame()
        engine_layout = QHBoxLayout(engine_row)
        engine_layout.setContentsMargins(0, 0, 0, 0)
        engine_layout.setSpacing(8)

        self.pca_panel = PcaSettingsPanel(self, controller=controller)
        self.ecc_panel = EccSettingsPanel(self, controller=controller)
        self.phase_correlation_panel = PhaseCorrelationSettingsPanel(
            self, controller=controller
        )

        self._engine_stack = QStackedWidget()
        self._engine_stack.addWidget(self.pca_panel)            # index 0
        self._engine_stack.addWidget(self.ecc_panel)            # index 1
        self._engine_stack.addWidget(self.phase_correlation_panel)  # index 2
        self._engine_stack.setCurrentIndex(1)  # default: ECC
        engine_layout.addWidget(self._engine_stack, stretch=1)

        self.warp_button = QPushButton("Warp: ON")
        self.warp_button.setFixedWidth(100)
        set_success_btn(self.warp_button)
        self.warp_button.clicked.connect(self.controller.toggle_warp)
        engine_layout.addWidget(self.warp_button)

        outer.addWidget(engine_row)

        # Frame slider row -----------------------------------------------------
        slider_row = QFrame()
        slider_layout = QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setObjectName("dim_label")
        slider_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(
            lambda v: self.controller.handle_frame_slider_move(v)
        )
        slider_layout.addWidget(self.frame_slider, stretch=1)

        outer.addWidget(slider_row)

        # Track current active panel for set_ui_state
        self._active_engine_panel = self.ecc_panel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_engine_panel(self):
        """The currently visible engine-specific settings panel."""
        return self._active_engine_panel

    @property
    def show_line_switch(self):
        """Shortcut to the PCA panel's show-line checkbox."""
        return self.pca_panel.show_line_switch

    def switch_engine(self, engine_name: str) -> None:
        """Switch the visible engine settings panel.

        Args:
            engine_name: One of 'PCA', 'ECC', or 'Phase Correlation'.

        Raises:
            ValueError: If engine_name is unknown.
        """
        mapping = {
            "PCA": (0, self.pca_panel),
            "ECC": (1, self.ecc_panel),
            "Phase Correlation": (2, self.phase_correlation_panel),
        }
        if engine_name not in mapping:
            raise ValueError(f"Unknown engine: {engine_name}")
        idx, panel = mapping[engine_name]
        self._engine_stack.setCurrentIndex(idx)
        self._active_engine_panel = panel

    def set_ui_state(self, enabled: bool) -> None:
        """Enable or disable interactive elements across all panels.

        Args:
            enabled: True to enable, False to disable.
        """
        self._active_engine_panel.set_ui_state(enabled)
        self.warp_button.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)

    def sync_timeline_label(self, current: int, total: int) -> None:
        """Update the frame counter label.

        Args:
            current: 1-based current frame index.
            total: Total number of frames.
        """
        self.frame_label.setText(f"Frame: {current}/{total}")

    def sync_pca_elements(self, t: float) -> None:
        """Synchronise PCA slider, label, and entry to threshold t.

        Args:
            t: Threshold value to display.
        """
        self.pca_panel.sync_pca_elements(t)

    def sync_pca_label_and_entry(self, t: float) -> None:
        """Synchronise PCA label and entry only.

        Args:
            t: Threshold value to display.
        """
        self.pca_panel.sync_pca_label_and_entry(t)
