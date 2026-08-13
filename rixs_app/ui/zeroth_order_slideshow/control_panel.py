"""Zeroth-order slideshow control panel — PySide6 port.

Contains the read-only frame metadata info bar and the frame scrub slider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
)
from PySide6.QtCore import Qt


class ZerothOrderControlPanel(QFrame):
    """Control panel for zeroth-order calibration.

    Displays a read-only metadata summary bar (filename, scan log motor position,
    FWHM, and score) and a frame scrub slider.

    Args:
        parent: Parent widget.
        controller: ZerothOrderSlideshowView controller.
    """

    def __init__(self, parent=None, *, controller):
        """Initialise the zeroth-order control panel.

        Args:
            parent: Parent QWidget.
            controller: ZerothOrderSlideshowView controller.
        """
        super().__init__(parent)
        self.controller = controller

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)

        # Read-only metadata summary row
        info_row = QFrame()
        info_layout = QHBoxLayout(info_row)
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(8)

        self.metadata_label = QLabel()
        self.metadata_label.setStyleSheet("font-size: 17px; font-weight: 500; background: transparent;")
        info_layout.addWidget(self.metadata_label, stretch=1)

        outer.addWidget(info_row)

        # Frame slider row
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

    def update_metadata(
        self,
        filename: str,
        motor_name: str,
        motor_val: str,
        fwhm_px: float | None = None,
        fwhm_mev: float | None = None,
        score: float | None = None,
    ) -> None:
        """Update the metadata summary label with rich HTML styling.

        Args:
            filename: TIF image basename.
            motor_name: Parsed motor variable name (e.g. 'SM3 Mirror Pitch' or 'Sample X').
            motor_val: Parsed motor position string or 'N/A'.
            fwhm_px: Optional line fit FWHM in pixels.
            fwhm_mev: Optional line fit FWHM in meV.
            score: Optional FWHM focus score (1/FWHM).
        """
        if fwhm_px is not None:
            fwhm_str = f"{fwhm_px:.2f} px"
            if fwhm_mev is not None and fwhm_mev > 0:
                fwhm_str += f" ({fwhm_mev:.1f} meV)"
        else:
            fwhm_str = "—"

        score_str = f"{score:.4f}" if score is not None else "—"
        sep = "&nbsp;&nbsp;<span style='color: #4c5d73;'>|</span>&nbsp;&nbsp;"

        html = (
            f"<div style='font-size: 17px; font-weight: 500;'>"
            f"<span style='color: #88aacc;'>Filename:</span> "
            f"<b style='color: #ffffff;'>{filename}</b>"
            f"{sep}"
            f"<span style='color: #88aacc;'>{motor_name}:</span> "
            f"<b style='color: #38bdf8;'>{motor_val}</b>"
            f"{sep}"
            f"<span style='color: #88aacc;'>FWHM:</span> "
            f"<b style='color: #fbbf24;'>{fwhm_str}</b>"
            f"{sep}"
            f"<span style='color: #88aacc;'>Score:</span> "
            f"<b style='color: #4ade80;'>{score_str}</b>"
            f"</div>"
        )

        self.metadata_label.setText(html)
