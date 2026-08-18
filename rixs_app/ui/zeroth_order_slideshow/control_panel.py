"""Zeroth-order slideshow control panel — PySide6 port.

Contains 4 modular elevated KPI diagnostic cards and the frame scrub slider.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
)
from PySide6.QtCore import Qt


CARD_STYLE_DEFAULT = """
QFrame#kpi_card {
    background-color: #16213e;
    border: 1px solid #2d3561;
    border-radius: 8px;
}
"""

CARD_STYLE_BEST_FOCUS = """
QFrame#kpi_card {
    background-color: #1c233d;
    border: 1.5px solid #fbbf24;
    border-radius: 8px;
}
"""


def _create_kpi_card(title: str):
    """Create a modular elevated KPI diagnostic card widget.

    Args:
        title: Uppercase title for the KPI card.

    Returns:
        tuple: (card_frame, title_label, val_label, sub_label, badge_label)
    """
    card = QFrame()
    card.setObjectName("kpi_card")
    card.setStyleSheet(CARD_STYLE_DEFAULT)

    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.setSpacing(2)

    title_label = QLabel(title)
    title_label.setStyleSheet(
        "font-size: 10px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px; border: none; background: transparent;"
    )
    layout.addWidget(title_label)

    val_row = QHBoxLayout()
    val_row.setContentsMargins(0, 0, 0, 0)
    val_row.setSpacing(6)

    val_label = QLabel("—")
    val_label.setStyleSheet(
        "font-size: 16px; font-weight: bold; color: #e2e8f0; border: none; background: transparent;"
    )
    val_row.addWidget(val_label)

    badge_label = QLabel("")
    badge_label.setVisible(False)
    val_row.addWidget(badge_label)
    val_row.addStretch()

    layout.addLayout(val_row)

    sub_label = QLabel("")
    sub_label.setStyleSheet(
        "font-size: 11px; color: #64748b; border: none; background: transparent;"
    )
    layout.addWidget(sub_label)

    return card, title_label, val_label, sub_label, badge_label


class ZerothOrderControlPanel(QFrame):
    """Control panel for zeroth-order calibration.

    Displays 4 modular elevated KPI diagnostic cards (Motor Pitch Position,
    FWHM Resolution, Resolving Power R, Gaussian Fit Score) and a frame scrub slider.

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
        outer.setSpacing(6)

        # 4 Modular Elevated KPI Diagnostic Cards
        cards_row = QFrame()
        cards_row.setStyleSheet("background: transparent; border: none;")
        cards_layout = QHBoxLayout(cards_row)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)

        # Card 1: Motor Pitch Position
        self.card_motor, self.motor_title_label, self.motor_val_label, self.motor_sub_label, _ = _create_kpi_card("MOTOR POSITION")
        cards_layout.addWidget(self.card_motor, stretch=1)

        # Card 2: FWHM Resolution
        self.card_fwhm, self.fwhm_title_label, self.fwhm_val_label, self.fwhm_sub_label, _ = _create_kpi_card("FWHM RESOLUTION")
        cards_layout.addWidget(self.card_fwhm, stretch=1)

        # Card 3: Resolving Power R
        self.card_rp, self.rp_title_label, self.rp_val_label, self.rp_sub_label, _ = _create_kpi_card("RESOLVING POWER (R)")
        cards_layout.addWidget(self.card_rp, stretch=1)

        # Card 4: Gaussian Fit Score
        self.card_score, self.score_title_label, self.score_val_label, self.score_sub_label, self.score_badge_label = _create_kpi_card("FIT SCORE (R²)")
        cards_layout.addWidget(self.card_score, stretch=1)

        outer.addWidget(cards_row)

        # Hidden / legacy metadata label for full backward compatibility
        self.metadata_label = QLabel()
        self.metadata_label.setVisible(False)
        outer.addWidget(self.metadata_label)

        # Frame slider row
        slider_row = QFrame()
        slider_row.setStyleSheet("background: transparent; border: none;")
        slider_layout = QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setObjectName("dim_label")
        slider_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setFocusPolicy(Qt.NoFocus)
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
        *,
        r_squared: float | None = None,
        is_best_focus: bool = False,
        mono_energy_ev: float | None = None,
    ) -> None:
        """Update the 4 elevated KPI diagnostic cards with live values and styling.

        Args:
            filename: TIF image basename.
            motor_name: Parsed motor variable name (e.g. 'SM3 Mirror Pitch' or 'Sample X').
            motor_val: Parsed motor position string or 'N/A'.
            fwhm_px: Optional line fit FWHM in pixels.
            fwhm_mev: Optional line fit FWHM in meV.
            score: Optional FWHM focus score (1/FWHM).
            r_squared: Optional Gaussian fit R² score (0.0 to 1.0).
            is_best_focus: Whether this frame is the optimal focus frame.
            mono_energy_ev: Monochromator photon energy in eV.
        """
        # --- Card 1: Motor Position ---
        title_text = motor_name.upper() if motor_name else "MOTOR POSITION"
        self.motor_title_label.setText(title_text)
        self.motor_val_label.setText(str(motor_val))
        self.motor_sub_label.setText(filename)

        # --- Card 2: FWHM Resolution ---
        if fwhm_px is not None:
            self.fwhm_val_label.setText(f"{fwhm_px:.2f} px")
            if fwhm_mev is not None and fwhm_mev > 0:
                self.fwhm_sub_label.setText(f"{fwhm_mev:.1f} meV")
            else:
                self.fwhm_sub_label.setText("Uncalibrated (meV)")
        else:
            self.fwhm_val_label.setText("—")
            self.fwhm_sub_label.setText("No line fit")

        if is_best_focus:
            self.card_fwhm.setStyleSheet(CARD_STYLE_BEST_FOCUS)
            self.fwhm_title_label.setText("FWHM RESOLUTION ★")
            self.fwhm_title_label.setStyleSheet(
                "font-size: 10px; font-weight: 700; color: #fbbf24; letter-spacing: 0.5px; border: none; background: transparent;"
            )
        else:
            self.card_fwhm.setStyleSheet(CARD_STYLE_DEFAULT)
            self.fwhm_title_label.setText("FWHM RESOLUTION")
            self.fwhm_title_label.setStyleSheet(
                "font-size: 10px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px; border: none; background: transparent;"
            )

        # --- Card 3: Resolving Power R ---
        if (
            mono_energy_ev is not None
            and mono_energy_ev > 0
            and fwhm_mev is not None
            and fwhm_mev > 0
        ):
            r_val = (mono_energy_ev * 1000.0) / fwhm_mev
            self.rp_val_label.setText(f"{int(round(r_val)):,}")
            self.rp_sub_label.setText(f"E₀ = {mono_energy_ev:.1f} eV")
        else:
            self.rp_val_label.setText("N/A")
            if fwhm_mev is None or fwhm_mev <= 0:
                self.rp_sub_label.setText("Missing Dispersion")
            else:
                self.rp_sub_label.setText("Missing E₀")

        # --- Card 4: Gaussian Fit Score (R²) ---
        val_r2 = r_squared if r_squared is not None else score
        if val_r2 is not None:
            self.score_val_label.setText(f"{val_r2:.4f}")
            self.score_badge_label.setVisible(True)
            if val_r2 >= 0.95:
                self.score_badge_label.setText("EXCELLENT")
                self.score_badge_label.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
                    "border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: bold;"
                )
                self.score_sub_label.setText("R² ≥ 0.95")
            elif val_r2 >= 0.80:
                self.score_badge_label.setText("ACCEPTABLE")
                self.score_badge_label.setStyleSheet(
                    "background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; "
                    "border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: bold;"
                )
                self.score_sub_label.setText("0.80 ≤ R² < 0.95")
            else:
                self.score_badge_label.setText("POOR")
                self.score_badge_label.setStyleSheet(
                    "background-color: #4c0519; color: #f87171; border: 1px solid #dc2626; "
                    "border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: bold;"
                )
                self.score_sub_label.setText("R² < 0.80")
        else:
            self.score_val_label.setText("—")
            self.score_badge_label.setVisible(True)
            self.score_badge_label.setText("NO FIT")
            self.score_badge_label.setStyleSheet(
                "background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; "
                "border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: bold;"
            )
            self.score_sub_label.setText("No valid Gaussian fit")

        # Synchronize legacy HTML metadata label
        fwhm_str = f"{fwhm_px:.2f} px" if fwhm_px is not None else "—"
        if fwhm_mev is not None and fwhm_mev > 0 and fwhm_px is not None:
            fwhm_str += f" ({fwhm_mev:.1f} meV)"
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
