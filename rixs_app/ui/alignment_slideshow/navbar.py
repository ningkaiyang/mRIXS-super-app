"""Alignment slideshow navigation bar — PySide6 port.

Provides back/first/prev/next/last navigation, autoplay toggle with speed cycling,
warp image switch, colormap selector, and alignment engine selector in a compact
horizontal toolbar row.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
)
from PySide6.QtCore import Qt

from rixs_app.ui.theme import set_play_btn, set_tool_btn


class SlideshowNavBar(QFrame):
    """Top navigation bar for the alignment slideshow view.

    Contains frame navigation buttons (first/prev/next/last), autoplay toggle,
    autoplay speed cycling button, warp toggle, colormap dropdown,
    and alignment engine dropdown.

    Args:
        parent: Parent widget.
        controller: The ``SlideshowView`` instance used to dispatch actions.
    """

    SPEED_CYCLE = [
        ("1×", 500),
        ("2×", 250),
        ("5×", 100),
        ("0.5×", 1000),
    ]

    def __init__(self, parent=None, *, controller):
        """Initialise the navigation bar.

        Args:
            parent: Parent QWidget.
            controller: SlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller
        self._speed_idx = 0
        self.setObjectName("navbar_frame")
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 16, 4)
        layout.setSpacing(8)

        # --- Navigation controls ---
        self.back_button = QPushButton("\u25c4 Back")
        self.back_button.setFixedWidth(75)
        set_tool_btn(self.back_button)
        self.back_button.clicked.connect(self.controller.back_to_sorting)
        layout.addWidget(self.back_button)

        self.first_button = QPushButton("\u23ee")
        self.first_button.setFixedWidth(38)
        self.first_button.setToolTip("Jump to first frame")
        set_tool_btn(self.first_button)
        self.first_button.clicked.connect(self._on_first_clicked)
        layout.addWidget(self.first_button)

        self.prev_button = QPushButton("\u25c4 Prev")
        self.prev_button.setFixedWidth(75)
        set_tool_btn(self.prev_button)
        self.prev_button.clicked.connect(self.controller.prev_frame)
        layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next \u25ba")
        self.next_button.setFixedWidth(75)
        set_tool_btn(self.next_button)
        self.next_button.clicked.connect(self.controller.next_frame)
        layout.addWidget(self.next_button)

        self.last_button = QPushButton("\u23ed")
        self.last_button.setFixedWidth(38)
        self.last_button.setToolTip("Jump to last frame")
        set_tool_btn(self.last_button)
        self.last_button.clicked.connect(self._on_last_clicked)
        layout.addWidget(self.last_button)

        # --- Autoplay & Speed cycling ---
        self.autoplay_button = QPushButton("\u25ba Play")
        self.autoplay_button.setFixedWidth(75)
        set_play_btn(self.autoplay_button)
        self.autoplay_button.clicked.connect(self.controller.toggle_autoplay)
        layout.addWidget(self.autoplay_button)

        self.speed_button = QPushButton("1×")
        self.speed_button.setFixedWidth(52)
        set_tool_btn(self.speed_button)
        self.speed_button.setToolTip("Playback Speed: 1× (500 ms/frame) — Click to cycle")
        self.speed_button.clicked.connect(self.cycle_speed)
        layout.addWidget(self.speed_button)

        layout.addStretch()

        # --- Right controls ---
        layout.addWidget(QLabel("Engine:"))

        self.engine_menu = QComboBox()
        self.engine_menu.setObjectName("engine_menu")
        engine_options = ["PCA", "ECC", "Phase Correlation"]
        self.engine_menu.addItems(engine_options)
        for idx, text in enumerate(engine_options):
            self.engine_menu.setItemData(idx, text, Qt.ToolTipRole)
        self.engine_menu.setCurrentText("ECC")
        self.engine_menu.setMinimumWidth(185)
        self.engine_menu.view().setMinimumWidth(185)
        self.engine_menu.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.engine_menu.currentTextChanged.connect(self.controller.change_engine)
        layout.addWidget(self.engine_menu)

        layout.addWidget(QLabel("Colormap:"))

        self.colormap_menu = QComboBox()
        colormap_options = ["viridis", "inferno", "plasma", "magma", "grayscale"]
        self.colormap_menu.addItems(colormap_options)
        for idx, text in enumerate(colormap_options):
            self.colormap_menu.setItemData(idx, text, Qt.ToolTipRole)
        self.colormap_menu.setCurrentText("viridis")
        self.colormap_menu.setMinimumWidth(100)
        self.colormap_menu.currentTextChanged.connect(self.controller.change_colormap)
        layout.addWidget(self.colormap_menu)

    # ------------------------------------------------------------------
    # Speed Cycling API
    # ------------------------------------------------------------------

    def cycle_speed(self) -> None:
        """Cycle through autoplay speeds (1× -> 2× -> 5× -> 0.5× -> 1×)."""
        self._speed_idx = (self._speed_idx + 1) % len(self.SPEED_CYCLE)
        label, ms = self.SPEED_CYCLE[self._speed_idx]
        self.speed_button.setText(label)
        self.speed_button.setToolTip(f"Playback Speed: {label} ({ms} ms/frame) — Click to cycle")

        if hasattr(self.controller, "manager"):
            self.controller.manager.autoplay_speed_ms = ms
        if hasattr(self.controller, "_autoplay_timer") and self.controller._autoplay_timer is not None:
            if self.controller._autoplay_timer.isActive():
                self.controller._autoplay_timer.setInterval(ms)

    def set_speed(self, label_or_ms: str | int) -> None:
        """Set a specific speed by label string (e.g. '2×') or interval ms (e.g. 250).

        Args:
            label_or_ms: Speed label string or millisecond integer.
        """
        for idx, (label, ms) in enumerate(self.SPEED_CYCLE):
            if label == label_or_ms or ms == label_or_ms:
                self._speed_idx = idx
                self.speed_button.setText(label)
                self.speed_button.setToolTip(f"Playback Speed: {label} ({ms} ms/frame) — Click to cycle")
                if hasattr(self.controller, "manager"):
                    self.controller.manager.autoplay_speed_ms = ms
                if hasattr(self.controller, "_autoplay_timer") and self.controller._autoplay_timer is not None:
                    if self.controller._autoplay_timer.isActive():
                        self.controller._autoplay_timer.setInterval(ms)
                return

    # ------------------------------------------------------------------
    # Sequence Boundary Disabling
    # ------------------------------------------------------------------

    def update_navigation_state(self, current_idx: int, total_frames: int) -> None:
        """Dynamically update button enabled states at sequence boundaries.

        Args:
            current_idx: Current frame index (0-based).
            total_frames: Total number of frames loaded.
        """
        if total_frames <= 1:
            self.first_button.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.last_button.setEnabled(False)
            self.autoplay_button.setEnabled(False)
            self.speed_button.setEnabled(False)
        else:
            self.first_button.setEnabled(current_idx > 0)
            self.prev_button.setEnabled(current_idx > 0)
            self.next_button.setEnabled(current_idx < total_frames - 1)
            self.last_button.setEnabled(current_idx < total_frames - 1)
            self.autoplay_button.setEnabled(True)
            self.speed_button.setEnabled(True)

    # ------------------------------------------------------------------
    # First / Last Navigation Helpers
    # ------------------------------------------------------------------

    def _on_first_clicked(self) -> None:
        """Jump to index 0."""
        if hasattr(self.controller, "first_frame"):
            self.controller.first_frame()
        elif hasattr(self.controller, "jump_to_frame"):
            self.controller.jump_to_frame(0)

    def _on_last_clicked(self) -> None:
        """Jump to the last frame."""
        if hasattr(self.controller, "last_frame"):
            self.controller.last_frame()
        elif hasattr(self.controller, "jump_to_frame") and hasattr(self.controller, "manager"):
            n = len(self.controller.manager.file_list)
            if n > 0:
                self.controller.jump_to_frame(n - 1)

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    @property
    def warp_button(self):
        """Proxy to control_panel.warp_button."""
        return self.controller.control_panel.warp_button

    # ------------------------------------------------------------------
    # Co-Pilot button integration
    # ------------------------------------------------------------------

    def set_copilot_button(self, btn) -> None:
        """Append the Co-Pilot toggle button to the right end of the navbar.

        Args:
            btn: The Co-Pilot toggle QPushButton to reparent here.
        """
        self.layout().addWidget(btn)
