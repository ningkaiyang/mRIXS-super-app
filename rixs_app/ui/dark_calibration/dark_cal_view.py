"""Detector Dark Frame Calibration Studio GUI View.

Provides an interactive workstation for:
1. Drag-and-drop dark frame TIFF folder ingestion with validation.
2. Background asynchronous computation of temporal median, per-pixel standard deviation,
   and 93rd-percentile excursion residual distributions via DarkDiagnosticsWorker.
3. Dual dark-themed log-scale Matplotlib diagnostic histograms with interactive vertical cutlines.
4. Real-time Tier 1 and Tier 2 threshold sliders with instant (<10ms) live KPI survival badges.
5. 1-Click calibration persistence to appdata/dark_calibration/ via calibration_store.
6. Navigation navbar with ❮ Back to Home button and Co-Pilot docking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from rixs_app.ui.widgets import SafeFigureCanvasQTAgg
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rixs_app.core import calibration_store
from rixs_app.core.cli_utils import glob_tifs
from rixs_app.core.photon_clustering import (
    DarkDiagnostics,
    apply_dark_thresholds,
    compute_dark_diagnostics,
)
from rixs_app.core.utils import natural_sort
from rixs_app.ui import theme
from rixs_app.ui.dark_calibration.workers import DarkDiagnosticsWorker

logger = logging.getLogger(__name__)


class DropZoneFrame(QFrame):
    """A container supporting drag-and-drop ingestion of TIFF folders/files."""

    def __init__(self, on_files_dropped: Callable[[list[str]], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_files_dropped_cb = on_files_dropped
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self._on_files_dropped_cb(paths)
        else:
            super().dropEvent(event)


class DarkCalibrationView(QWidget):
    """Detector Dark Frame Calibration Studio interactive workstation.

    Args:
        parent: Optional parent QWidget.
        on_back: Optional callback invoked when the user clicks '❮ Back to Home'.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_back = on_back
        self.setAcceptDrops(True)

        self.dark_paths: list[str] = []
        self.dark_frame_count: int = 0
        self._tail_ratio: float = 0.9333
        self._stddev_thresh: float = 40.0
        self._absdev_thresh: float = 60.0

        self._diagnostics: DarkDiagnostics | None = None
        self._current_worker: DarkDiagnosticsWorker | None = None
        self._copilot_btn: QPushButton | None = None

        # Hist cutline objects
        self._std_cutline = None
        self._res_cutline = None

        self._init_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # ── 1. Navbar Header ──
        navbar_layout = QHBoxLayout()
        navbar_layout.setContentsMargins(0, 0, 0, 0)
        navbar_layout.setSpacing(12)

        self._back_btn = QPushButton("❮ Back to Home", self)
        theme.set_tool_btn(self._back_btn)
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.clicked.connect(self._handle_back_clicked)
        navbar_layout.addWidget(self._back_btn)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        header_title = QLabel("Detector Dark Frame Calibration Studio", self)
        header_title.setObjectName("header_title")
        header_title.setStyleSheet("font-size: 17px; font-weight: bold; color: #e8eaf6;")

        header_sub = QLabel(
            "Noise variance analysis, bad-pixel thresholding & baseline persistence", self
        )
        header_sub.setObjectName("dim_label")
        header_sub.setStyleSheet("font-size: 11px; color: #9fa8da;")
        title_layout.addWidget(header_title)
        title_layout.addWidget(header_sub)
        navbar_layout.addLayout(title_layout)

        navbar_layout.addStretch(1)

        # Co-Pilot docking container
        self._copilot_container = QWidget(self)
        self._copilot_container_layout = QHBoxLayout(self._copilot_container)
        self._copilot_container_layout.setContentsMargins(0, 0, 0, 0)
        navbar_layout.addWidget(self._copilot_container)

        main_layout.addLayout(navbar_layout)

        # ── 2. Splitter Body (Left Ingest, Right Histograms/Controls) ──
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)

        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 7)

        main_layout.addWidget(splitter, 1)

    def _create_left_panel(self) -> QWidget:
        """Build the left ingest panel with drag-drop, file list, and generate button."""
        panel = DropZoneFrame(on_files_dropped=self._on_files_dropped, parent=self)
        panel.setObjectName("dark_cal_left_panel")
        panel.setStyleSheet(
            "DropZoneFrame#dark_cal_left_panel { background-color: #16213e; "
            "border: 1px solid #2d3561; border-radius: 8px; }"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Section Header
        lbl_title = QLabel("1. Dark Frame Ingest", panel)
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel("Drag & drop dark TIFF folder here or browse:", panel)
        lbl_desc.setStyleSheet("color: #9fa8da; font-size: 11px;")
        layout.addWidget(lbl_desc)

        # Ingest action buttons
        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("📁 Browse Dark Folder", panel)
        theme.set_tool_btn(self.browse_btn)
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._browse_folder)
        btn_row.addWidget(self.browse_btn)

        self.clear_btn = QPushButton("✕ Clear", panel)
        theme.set_cancel_btn(self.clear_btn)
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_files)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        # File list display
        self.file_list_widget = QListWidget(panel)
        self.file_list_widget.setStyleSheet(
            f"QListWidget {{ background-color: #0f172a; border: 1px solid #2d3561; "
            f"border-radius: 6px; color: #e8eaf6; font-size: 11px; font-family: {theme.FONT_STACK_CODE}; }}"
        )
        self.file_list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.file_list_widget, 1)

        # Metadata stats card
        stats_box = QFrame(panel)
        stats_box.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #2d3561; border-radius: 6px; padding: 6px;"
        )
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setSpacing(4)

        self.frame_count_label = QLabel("Dark Frames: 0", stats_box)
        self.frame_count_label.setStyleSheet("font-weight: bold; color: #e8eaf6; font-size: 12px;")
        stats_layout.addWidget(self.frame_count_label)

        self.tail_ratio_label = QLabel(f"Tail Stability Cutoff: {self._tail_ratio * 100:.2f}%", stats_box)
        self.tail_ratio_label.setStyleSheet("color: #9fa8da; font-size: 11px;")
        stats_layout.addWidget(self.tail_ratio_label)

        self.source_dir_label = QLabel("Source: None", stats_box)
        self.source_dir_label.setStyleSheet("color: #64748b; font-size: 10px;")
        self.source_dir_label.setWordWrap(True)
        stats_layout.addWidget(self.source_dir_label)

        layout.addWidget(stats_box)

        # Progress bar and status label
        self.progress_bar = QProgressBar(panel)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1e293b; border-radius: 4px; border: none; }"
            "QProgressBar::chunk { background-color: #38bdf8; border-radius: 4px; }"
        )
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.progress_msg_label = QLabel("", panel)
        self.progress_msg_label.setStyleSheet("color: #38bdf8; font-size: 11px;")
        self.progress_msg_label.hide()
        layout.addWidget(self.progress_msg_label)

        # Generate Button
        self.generate_btn = QPushButton("▶ Generate Histograms", panel)
        theme.set_play_btn(self.generate_btn)
        self.generate_btn.setCursor(Qt.PointingHandCursor)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setFixedHeight(36)
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.generate_btn)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Build the right panel with dual log-scale histograms, sliders, KPI badges, and save."""
        panel = QFrame(self)
        panel.setObjectName("dark_cal_right_panel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setAutoFillBackground(True)
        panel.setStyleSheet(
            "QFrame#dark_cal_right_panel { background-color: #16213e; "
            "border: 1px solid #2d3561; border-radius: 8px; }"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header Row
        header_row = QHBoxLayout()
        lbl_title = QLabel("2. Diagnostic Distributions & Threshold Tuning", panel)
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        header_row.addWidget(lbl_title)
        header_row.addStretch(1)

        layout.addLayout(header_row)

        # Matplotlib Canvas Container
        self.figure = Figure(figsize=(8, 4.2), dpi=100, facecolor="#14172b")
        canvas_cls = SafeFigureCanvasQTAgg or FigureCanvasQTAgg
        self.canvas = canvas_cls(self.figure)
        self.canvas.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.canvas.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.canvas.setAutoFillBackground(True)
        self.canvas.setStyleSheet("background-color: #14172b; border-radius: 6px;")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_std = self.figure.add_subplot(121, facecolor="#14172b")
        self.ax_res = self.figure.add_subplot(122, facecolor="#14172b")
        self._init_empty_axes()
        self.figure.tight_layout(pad=2.0)

        layout.addWidget(self.canvas, 1)

        # ── Sliders & Controls Area ──
        controls_frame = QFrame(panel)
        controls_frame.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #2d3561; border-radius: 8px; padding: 10px;"
        )
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(8)

        # Slider 1: Tier 1 StdDev Threshold
        std_row = QHBoxLayout()
        lbl_std = QLabel("Tier 1 Max StdDev (\u03c3):", controls_frame)
        lbl_std.setStyleSheet("font-weight: bold; color: #e8eaf6; font-size: 12px;")
        lbl_std.setFixedWidth(160)
        std_row.addWidget(lbl_std)

        self.stddev_slider = QSlider(Qt.Horizontal, controls_frame)
        self.stddev_slider.setRange(1, 200)
        self.stddev_slider.setValue(int(self._stddev_thresh))
        self.stddev_slider.valueChanged.connect(self._handle_stddev_slider_moved)
        std_row.addWidget(self.stddev_slider, 1)

        self.stddev_val_label = QLabel(f"{self._stddev_thresh:.1f} ADU", controls_frame)
        self.stddev_val_label.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        self.stddev_val_label.setFixedWidth(70)
        self.stddev_val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        std_row.addWidget(self.stddev_val_label)
        controls_layout.addLayout(std_row)

        # Slider 2: Tier 2 Residual / Excursion Threshold
        res_row = QHBoxLayout()
        lbl_res = QLabel("Tier 2 Max Residual (\u0394):", controls_frame)
        lbl_res.setStyleSheet("font-weight: bold; color: #e8eaf6; font-size: 12px;")
        lbl_res.setFixedWidth(160)
        res_row.addWidget(lbl_res)

        self.absdev_slider = QSlider(Qt.Horizontal, controls_frame)
        self.absdev_slider.setRange(1, 300)
        self.absdev_slider.setValue(int(self._absdev_thresh))
        self.absdev_slider.valueChanged.connect(self._handle_absdev_slider_moved)
        res_row.addWidget(self.absdev_slider, 1)

        self.absdev_val_label = QLabel(f"{self._absdev_thresh:.1f} ADU", controls_frame)
        self.absdev_val_label.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 12px;")
        self.absdev_val_label.setFixedWidth(70)
        self.absdev_val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        res_row.addWidget(self.absdev_val_label)
        controls_layout.addLayout(res_row)

        # ── KPI Badges Row ──
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)

        # Tier 1 KPI badge
        self.tier1_kpi_label = QLabel("Tier 1 (\u03c3 < 40.0): - % surviving", controls_frame)
        self.tier1_kpi_label.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1px solid #3b82f6; "
            "border-radius: 6px; padding: 5px 10px; font-weight: 600; font-size: 11px;"
        )
        kpi_row.addWidget(self.tier1_kpi_label)

        # Tier 2 KPI badge
        self.tier2_kpi_label = QLabel("Tier 2 (\u0394 < 60.0): - % surviving", controls_frame)
        self.tier2_kpi_label.setStyleSheet(
            "background-color: #1e293b; color: #fbbf24; border: 1px solid #d97706; "
            "border-radius: 6px; padding: 5px 10px; font-weight: 600; font-size: 11px;"
        )
        kpi_row.addWidget(self.tier2_kpi_label)

        # Composite Final Mask KPI badge
        self.final_mask_kpi_label = QLabel("Final Mask: - % active", controls_frame)
        self.final_mask_kpi_label.setStyleSheet(
            "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
            "border-radius: 6px; padding: 5px 10px; font-weight: 600; font-size: 11px;"
        )
        kpi_row.addWidget(self.final_mask_kpi_label)

        controls_layout.addLayout(kpi_row)
        layout.addWidget(controls_frame)

        # ── Bottom Action & Persistence Row ──
        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        self.save_status_label = QLabel("", panel)
        self.save_status_label.setStyleSheet("color: #34d399; font-weight: bold; font-size: 12px;")
        action_row.addWidget(self.save_status_label)
        action_row.addStretch(1)

        self.save_btn = QPushButton("💾 Save Calibration", panel)
        theme.set_success_btn(self.save_btn)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedHeight(36)
        self.save_btn.clicked.connect(self._on_save_clicked)
        action_row.addWidget(self.save_btn)

        layout.addLayout(action_row)

        return panel

    def _init_empty_axes(self) -> None:
        """Setup dark-styled empty placeholder axes before data generation."""
        for ax, title, xlabel in [
            (self.ax_std, "Pixel Noise StdDev (\u03c3)", "Standard Deviation \u03c3 (ADU)"),
            (self.ax_res, "93rd-Percentile Residual (\u0394)", "Excursion Residual \u0394 (ADU)"),
        ]:
            ax.clear()
            ax.set_facecolor("#14172b")
            ax.set_title(title, color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(xlabel, color="#9fa8da", fontsize=10)
            ax.set_ylabel("Pixel Count (Log Scale)", color="#9fa8da", fontsize=10)
            ax.tick_params(colors="#9fa8da", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2d3561")
            ax.grid(True, linestyle=":", color="#2d3561", alpha=0.6)
            ax.text(
                0.5,
                0.5,
                "No data loaded\nClick [▶ Generate Histograms]",
                color="#64748b",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )

    # ------------------------------------------------------------------
    # Drag and Drop & File Management
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self._on_files_dropped(paths)
        else:
            super().dropEvent(event)

    def _browse_folder(self) -> None:
        """Open folder dialog to select dark frame TIFF directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Dark Frame TIFF Directory")
        if folder:
            self._on_files_dropped([folder])

    def _clear_files(self) -> None:
        """Clear loaded file list and reset diagnostic state."""
        self.dark_paths.clear()
        self.dark_frame_count = 0
        self.file_list_widget.clear()
        self.frame_count_label.setText("Dark Frames: 0")
        self.source_dir_label.setText("Source: None")
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.save_status_label.setText("")
        self._diagnostics = None
        self._init_empty_axes()
        self.canvas.draw_idle()
        self.tier1_kpi_label.setText("Tier 1 (\u03c3 < 40.0): - % surviving")
        self.tier2_kpi_label.setText("Tier 2 (\u0394 < 60.0): - % surviving")
        self.final_mask_kpi_label.setText("Final Mask: - % active")

    def _on_files_dropped(self, file_paths: list[str]) -> None:
        """Ingest dropped files or folders, discover TIFF files, and populate file list."""
        discovered: list[str] = []
        for p in file_paths:
            path_obj = Path(p)
            if path_obj.is_dir():
                discovered.extend(glob_tifs(str(path_obj)))
            elif path_obj.is_file() and path_obj.suffix.lower() in (".tif", ".tiff"):
                discovered.append(str(path_obj.resolve()))

        if not discovered:
            self.dark_paths = []
            self.dark_frame_count = 0
            self.file_list_widget.clear()
            self.frame_count_label.setText("Dark Frames: 0")
            self.generate_btn.setEnabled(False)
            return

        # Deduplicate while preserving natural order
        unique_paths = list(dict.fromkeys(discovered))
        natural_sort(unique_paths)

        self.dark_paths = unique_paths
        self.dark_frame_count = len(unique_paths)

        # Update UI elements
        self.file_list_widget.clear()
        for p in self.dark_paths:
            self.file_list_widget.addItem(Path(p).name)

        self.frame_count_label.setText(f"Dark Frames: {self.dark_frame_count}")
        source_dir = str(Path(self.dark_paths[0]).parent)
        self.source_dir_label.setText(f"Source: {source_dir}")
        self.generate_btn.setEnabled(self.dark_frame_count > 0)
        self.save_status_label.setText("")

    # ------------------------------------------------------------------
    # Worker Execution
    # ------------------------------------------------------------------

    def _on_generate_clicked(self) -> None:
        """Launch asynchronous background diagnostics computation."""
        if not self.dark_paths:
            return

        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_msg_label.setText("Starting analysis...")
        self.progress_msg_label.show()

        worker = DarkDiagnosticsWorker(
            dark_paths=self.dark_paths,
            tail_pct=self._tail_ratio,
        )
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.progress_msg.connect(self._on_worker_msg)
        worker.signals.result.connect(self._on_diagnostics_ready)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)

        # Keep strong reference to prevent premature garbage collection
        self._current_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_worker_progress(self, current: int, total: int) -> None:
        """Update progress bar during background processing."""
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)

    def _on_worker_msg(self, msg: str) -> None:
        """Update progress status label."""
        self.progress_msg_label.setText(msg)

    def _on_worker_error(self, err: str) -> None:
        """Handle background worker failure."""
        logger.error("Dark calibration worker error: %s", err)
        self.progress_msg_label.setText(f"Error: {err}")
        self.progress_msg_label.setStyleSheet("color: #ef4444; font-size: 11px;")

    def _on_worker_finished(self) -> None:
        """Handle background worker completion."""
        self.generate_btn.setEnabled(self.dark_frame_count > 0)
        self.progress_bar.hide()
        self.progress_msg_label.hide()
        self._current_worker = None

    def _on_diagnostics_ready(self, diag: DarkDiagnostics) -> None:
        """Receive completed diagnostics, render histograms, and update KPIs."""
        self._diagnostics = diag
        self.save_btn.setEnabled(True)
        self._render_histograms()
        self._update_kpis_and_cutlines()

    # ------------------------------------------------------------------
    # Histogram Rendering & Cutlines
    # ------------------------------------------------------------------

    def _render_histograms(self) -> None:
        """Draw dark-themed log-scale histograms of StdDev and residual distributions."""
        if self._diagnostics is None:
            return

        std_data = self._diagnostics.per_pixel_stddev.ravel()
        res_data = self._diagnostics.pct93_residual.ravel()

        std_finite = std_data[np.isfinite(std_data)]
        res_finite = res_data[np.isfinite(res_data)]

        # --- Subplot 1: StdDev ---
        self.ax_std.clear()
        self.ax_std.set_facecolor("#14172b")
        self.ax_std.set_title("Pixel Noise StdDev (\u03c3)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_std.set_xlabel("Standard Deviation \u03c3 (ADU)", color="#9fa8da", fontsize=10)
        self.ax_std.set_ylabel("Pixel Count (Log Scale)", color="#9fa8da", fontsize=10)
        self.ax_std.tick_params(colors="#9fa8da", labelsize=9)
        for spine in self.ax_std.spines.values():
            spine.set_color("#2d3561")
        self.ax_std.grid(True, linestyle=":", color="#2d3561", alpha=0.6)

        if len(std_finite) > 0:
            std_max = max(100.0, float(np.percentile(std_finite, 99.9)))
            self.ax_std.hist(
                std_finite,
                bins=60,
                range=(0, std_max),
                color="#38bdf8",
                edgecolor="#0284c7",
                alpha=0.85,
                log=True,
            )
            self.stddev_slider.setMaximum(max(200, int(std_max * 1.2)))

        self._std_cutline = self.ax_std.axvline(
            x=self._stddev_thresh,
            color="#ef4444",
            linestyle="--",
            linewidth=1.8,
            label=f"Cut: {self._stddev_thresh:.1f}",
        )
        self.ax_std.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=9)

        # --- Subplot 2: Residual ---
        self.ax_res.clear()
        self.ax_res.set_facecolor("#14172b")
        self.ax_res.set_title("93rd-Percentile Residual (\u0394)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_res.set_xlabel("Excursion Residual \u0394 (ADU)", color="#9fa8da", fontsize=10)
        self.ax_res.set_ylabel("Pixel Count (Log Scale)", color="#9fa8da", fontsize=10)
        self.ax_res.tick_params(colors="#9fa8da", labelsize=9)
        for spine in self.ax_res.spines.values():
            spine.set_color("#2d3561")
        self.ax_res.grid(True, linestyle=":", color="#2d3561", alpha=0.6)

        if len(res_finite) > 0:
            res_max = max(150.0, float(np.percentile(res_finite, 99.9)))
            self.ax_res.hist(
                res_finite,
                bins=60,
                range=(0, res_max),
                color="#f59e0b",
                edgecolor="#d97706",
                alpha=0.85,
                log=True,
            )
            self.absdev_slider.setMaximum(max(300, int(res_max * 1.2)))

        self._res_cutline = self.ax_res.axvline(
            x=self._absdev_thresh,
            color="#ef4444",
            linestyle="--",
            linewidth=1.8,
            label=f"Cut: {self._absdev_thresh:.1f}",
        )
        self.ax_res.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=9)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slider & KPI Updates
    # ------------------------------------------------------------------

    def _handle_stddev_slider_moved(self, val: int) -> None:
        """Handle user moving the StdDev slider."""
        self._on_stddev_slider_changed(float(val))

    def _handle_absdev_slider_moved(self, val: int) -> None:
        """Handle user moving the Residual slider."""
        self._on_absdev_slider_changed(float(val))

    def _on_stddev_slider_changed(self, val: float | int) -> None:
        """Update StdDev threshold value, sync slider position, cutlines, and KPIs."""
        self._stddev_thresh = float(val)
        self.stddev_val_label.setText(f"{self._stddev_thresh:.1f} ADU")

        if self.stddev_slider.value() != int(val):
            self.stddev_slider.blockSignals(True)
            self.stddev_slider.setValue(int(val))
            self.stddev_slider.blockSignals(False)

        if self._std_cutline is not None:
            self._std_cutline.set_xdata([self._stddev_thresh, self._stddev_thresh])
            self._std_cutline.set_label(f"Cut: {self._stddev_thresh:.1f}")
            self.ax_std.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=9)
            self.canvas.draw_idle()

        self._update_kpis_and_cutlines()

    def _on_absdev_slider_changed(self, val: float | int) -> None:
        """Update Residual threshold value, sync slider position, cutlines, and KPIs."""
        self._absdev_thresh = float(val)
        self.absdev_val_label.setText(f"{self._absdev_thresh:.1f} ADU")

        if self.absdev_slider.value() != int(val):
            self.absdev_slider.blockSignals(True)
            self.absdev_slider.setValue(int(val))
            self.absdev_slider.blockSignals(False)

        if self._res_cutline is not None:
            self._res_cutline.set_xdata([self._absdev_thresh, self._absdev_thresh])
            self._res_cutline.set_label(f"Cut: {self._absdev_thresh:.1f}")
            self.ax_res.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=9)
            self.canvas.draw_idle()

        self._update_kpis_and_cutlines()

    def _update_kpis_and_cutlines(self) -> None:
        """Recalculate survival percentages and update KPI badge labels."""
        if self._diagnostics is None:
            return

        total_pixels = int(self._diagnostics.per_pixel_stddev.size)
        if total_pixels == 0:
            return

        m_std = (self._diagnostics.per_pixel_stddev < self._stddev_thresh) & np.isfinite(
            self._diagnostics.per_pixel_stddev
        )
        m_tail = (self._diagnostics.pct93_residual < self._absdev_thresh) & np.isfinite(
            self._diagnostics.pct93_residual
        )
        m_final = m_std & m_tail

        surv_std = int(np.count_nonzero(m_std))
        surv_tail = int(np.count_nonzero(m_tail))
        surv_final = int(np.count_nonzero(m_final))

        pct_std = (surv_std / total_pixels) * 100.0
        pct_tail = (surv_tail / total_pixels) * 100.0
        pct_final = (surv_final / total_pixels) * 100.0

        self.tier1_kpi_label.setText(
            f"Tier 1 (\u03c3 < {self._stddev_thresh:.1f}): {pct_std:.2f}% surviving ({surv_std:,} px)"
        )
        self.tier2_kpi_label.setText(
            f"Tier 2 (\u0394 < {self._absdev_thresh:.1f}): {pct_tail:.2f}% surviving ({surv_tail:,} px)"
        )
        self.final_mask_kpi_label.setText(
            f"Final Mask: {pct_final:.2f}% active ({surv_final:,} / {total_pixels:,} px)"
        )

    # ------------------------------------------------------------------
    # 1-Click Save Calibration Persistence
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        """Apply current thresholds and persist calibration products to calibration_store."""
        if self._diagnostics is None:
            return

        stage1 = apply_dark_thresholds(
            diagnostics=self._diagnostics,
            stddev_thresh=self._stddev_thresh,
            absdev_thresh=self._absdev_thresh,
            tail_ratio=self._tail_ratio,
        )

        source_dir = (
            str(Path(self.dark_paths[0]).parent)
            if self.dark_paths
            else "unknown"
        )

        cal_dir = getattr(calibration_store, "DARK_CAL_DIR", None)

        record = calibration_store.save_calibration(
            med_dark=stage1.med_dark,
            final_mask=stage1.final_mask,
            stddev_thresh=self._stddev_thresh,
            absdev_thresh=self._absdev_thresh,
            tail_ratio=self._tail_ratio,
            dark_frame_count=self._diagnostics.dark_frame_count,
            surviving_pixels=stage1.surviving_pixels,
            total_pixels=stage1.total_pixels,
            suppression_pct=stage1.suppression_pct,
            source_dir=source_dir,
            cal_dir=cal_dir,
        )

        active_pct = (
            (record.surviving_pixels / record.total_pixels) * 100.0
            if record.total_pixels > 0
            else 100.0 - record.suppression_pct
        )
        self.save_status_label.setText(
            f"✓ Saved calibration ({active_pct:.2f}% active pixels) to appdata/dark_calibration/"
        )
        logger.info("Saved dark calibration record: %s", record)

    # ------------------------------------------------------------------
    # Navbar & Co-Pilot Docking
    # ------------------------------------------------------------------

    def _handle_back_clicked(self) -> None:
        """Handle ❮ Back to Home button press."""
        if self.on_back is not None:
            self.on_back()

    def set_copilot_button(self, btn: QPushButton) -> None:
        """Dock Co-Pilot toggle button into navbar header."""
        if self._copilot_btn is not None and self._copilot_btn is not btn:
            self._copilot_container_layout.removeWidget(self._copilot_btn)
            self._copilot_btn.setParent(None)

        self._copilot_btn = btn
        self._copilot_container_layout.addWidget(btn)
        btn.setParent(self._copilot_container)
        btn.show()

    # ------------------------------------------------------------------
    # Lifecycle & Teardown
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        """Ensure canvas redraws and repaints cleanly when view is displayed."""
        super().showEvent(event)
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.draw_idle()

    def cleanup(self) -> None:
        """Clean up Matplotlib figures and background workers."""
        if self._current_worker is not None:
            try:
                self._current_worker.cancel()
            except Exception:
                pass
            self._current_worker = None
        if hasattr(self, "canvas") and self.canvas is not None:
            try:
                self.canvas._destroyed = True
            except Exception:
                pass
            if hasattr(self.canvas, "_idle_timer") and self.canvas._idle_timer is not None:
                try:
                    self.canvas._idle_timer.stop()
                except Exception:
                    pass
        if hasattr(self, "figure") and self.figure is not None:
            try:
                self.figure.clear()
            except Exception:
                pass

    def _teardown_mpl(self) -> None:
        """Idempotent teardown of Matplotlib resources."""
        self.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Handle window close event with resource cleanup."""
        self.cleanup()
        super().closeEvent(event)
