"""Zeroth-order slideshow control panel — PySide6 port.

Contains the measured X position display, SMU3 motor entry fields,
and the frame scrub slider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QLineEdit, QPushButton,
)
from PySide6.QtCore import Qt
from rixs_app.ui.theme import neutral_style


class ZerothOrderControlPanel(QFrame):
    """Control panel for zeroth-order calibration.

    Displays measured centroid X positions, motor value inputs,
    and a frame scrub slider.

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
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        # Measurement display row
        meas_row = QFrame()
        meas_layout = QHBoxLayout(meas_row)
        meas_layout.setContentsMargins(0, 0, 0, 0)
        meas_layout.setSpacing(8)

        meas_layout.addWidget(QLabel("Measured X:"))
        self.measured_x_label = QLabel("N/A")
        self.measured_x_label.setObjectName("value_label")
        meas_layout.addWidget(self.measured_x_label)

        meas_layout.addWidget(QLabel("SM3 Motor:"))
        self.motor_entry = QLineEdit()
        self.motor_entry.setFixedWidth(100)
        self.motor_entry.setPlaceholderText("Motor value")
        self.motor_entry.returnPressed.connect(self.controller.handle_motor_entry_submit)
        meas_layout.addWidget(self.motor_entry)

        self.add_point_button = QPushButton("+ Add Calibration Point")
        self.add_point_button.setFixedWidth(190)
        self.add_point_button.setStyleSheet(neutral_style())
        self.add_point_button.clicked.connect(self.controller.add_calibration_point)
        meas_layout.addWidget(self.add_point_button)

        self.clear_points_button = QPushButton("Clear Points")
        self.clear_points_button.setFixedWidth(110)
        self.clear_points_button.setStyleSheet(neutral_style())
        self.clear_points_button.clicked.connect(self.controller.clear_calibration_points)
        meas_layout.addWidget(self.clear_points_button)

        meas_layout.addStretch()
        outer.addWidget(meas_row)

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
