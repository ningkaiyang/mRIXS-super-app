"""Dark Image & Pixel Masking Studio GUI View.

Provides an interactive workstation for:
1. Clickable drag-and-drop dark frame TIFF folder ingestion with validation.
2. Background asynchronous computation of temporal median, per-pixel standard deviation,
   and 93rd-percentile excursion residual distributions via DarkDiagnosticsWorker.
3. Dual-axis log/linear Matplotlib diagnostic histograms with interactive vertical cutlines
   and dynamic transparent red shaded regions highlighting masked pixels.
4. Real-time StdDev (σ) and Excursion Residual (Δ) threshold sliders with instant (<10ms)
   incremental KPI badges showing exact marginal pixel suppression.
5. 1-Click dark mask persistence to appdata/dark_masking/ via dark_mask_store.
6. Navigation navbar with ❮ Back to Home button and Co-Pilot docking.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from matplotlib.figure import Figure
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

from rixs_app.core import dark_mask_store, calibration_store
from rixs_app.core.cli_utils import glob_tifs
from rixs_app.core.photon_clustering import (
    DarkDiagnostics,
    apply_dark_thresholds,
    compute_dark_diagnostics,
)
from rixs_app.core.utils import natural_sort
from rixs_app.ui import theme
from rixs_app.ui.dark_masking.workers import DarkDiagnosticsWorker
from rixs_app.ui.widgets import SafeFigureCanvasQTAgg

logger = logging.getLogger(__name__)


class DropZoneFrame(QFrame):
    """A container supporting drag-and-drop ingestion and click-to-browse of TIFF folders/files."""

    def __init__(
        self,
        on_files_dropped: Callable[[list[str]], None],
        on_clicked: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_files_dropped_cb = on_files_dropped
        self._on_clicked_cb = on_clicked
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._on_clicked_cb is not None:
            self._on_clicked_cb()
        super().mousePressEvent(event)

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


class DarkMaskingView(QWidget):
    """Dark Image & Pixel Masking Studio interactive workstation.

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

        # Hist cutline & span objects
        self._std_cutline = None
        self._res_cutline = None
        self._std_span = None
        self._res_span = None
        self._std_max = 100.0
        self._res_max = 150.0

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
        header_title = QLabel("Dark Image & Pixel Masking Studio", self)
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

        # Co-Pilot docking container (transparent)
        self._copilot_container = QWidget(self)
        self._copilot_container.setStyleSheet("background: transparent;")
        self._copilot_container_layout = QHBoxLayout(self._copilot_container)
        self._copilot_container_layout.setContentsMargins(0, 0, 0, 0)
        navbar_layout.addWidget(self._copilot_container)

        main_layout.addLayout(navbar_layout)

        # ── 2. Splitter Body (Left Ingest, Right Histograms/Controls) ──
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 7)

        main_layout.addWidget(splitter, 1)

    def _create_left_panel(self) -> QWidget:
        """Build the left ingest panel with clickable drag-drop, file list, and generate button."""
        panel = QFrame(self)
        panel.setObjectName("dark_cal_left_panel")
        panel.setStyleSheet(
            "QFrame#dark_cal_left_panel { background-color: #16213e; "
            "border: 1px solid #2d3561; border-radius: 10px; }"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Section Header
        lbl_title = QLabel("1. Dark Frame Ingest", panel)
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_title)

        # Clickable Drag & Drop Zone Box
        self.drop_zone = DropZoneFrame(
            on_files_dropped=self._on_files_dropped,
            on_clicked=self._browse_folder,
            parent=panel,
        )
        self.drop_zone.setObjectName("drop_zone")
        self.drop_zone.setMinimumHeight(100)
        self.drop_zone.setStyleSheet(
            "QFrame#drop_zone { background-color: #10162a; border: 1.5px dashed #38bdf8; "
            "border-radius: 8px; } QFrame#drop_zone:hover { background-color: #1a2344; "
            "border-color: #60a5fa; }"
        )

        dz_layout = QVBoxLayout(self.drop_zone)
        dz_layout.setAlignment(Qt.AlignCenter)
        dz_layout.setContentsMargins(10, 10, 10, 10)
        dz_layout.setSpacing(4)

        dz_icon = QLabel("📥", self.drop_zone)
        dz_icon.setStyleSheet("font-size: 24px;")
        dz_icon.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(dz_icon)

        dz_title = QLabel("Click to Browse or Drag & Drop Dark TIFF Folder", self.drop_zone)
        dz_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff;")
        dz_title.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(dz_title)

        dz_hint = QLabel("Supports raw .tif / .tiff sequence directories", self.drop_zone)
        dz_hint.setStyleSheet("font-size: 10px; color: #94a3b8;")
        dz_hint.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(dz_hint)

        layout.addWidget(self.drop_zone)

        # File list & controls
        list_header_layout = QHBoxLayout()
        list_header_layout.setContentsMargins(0, 0, 0, 0)
        list_title = QLabel("Loaded Dark Frames:", panel)
        list_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #e8eaf6;")
        list_header_layout.addWidget(list_title)
        list_header_layout.addStretch(1)

        self.clear_btn = QPushButton("✕ Clear", panel)
        self.clear_btn.setFixedHeight(24)
        self.clear_btn.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        theme.set_cancel_btn(self.clear_btn)
        self.clear_btn.clicked.connect(self._clear_files)
        list_header_layout.addWidget(self.clear_btn)
        layout.addLayout(list_header_layout)

        self.file_list_widget = QListWidget(panel)
        self.file_list_widget.setStyleSheet(
            "QListWidget { background-color: #12182b; border: 1px solid #2d3561; "
            "border-radius: 6px; color: #e8eaf6; padding: 4px; font-size: 11px; }"
        )
        layout.addWidget(self.file_list_widget, 1)

        # Metadata info box (rounded card)
        self.info_box = QFrame(panel)
        self.info_box.setStyleSheet(
            "QFrame { background-color: #12182b; border: 1px solid #2d3561; border-radius: 6px; padding: 6px; }"
        )
        info_layout = QVBoxLayout(self.info_box)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(4)

        self.frame_count_label = QLabel("Dark Frames: 0", self.info_box)
        self.frame_count_label.setStyleSheet("font-size: 11px; color: #e8eaf6; font-weight: 600;")
        info_layout.addWidget(self.frame_count_label)

        self.tail_ratio_label = QLabel("Tail Stability Cutoff: 93.33%", self.info_box)
        self.tail_ratio_label.setStyleSheet("font-size: 11px; color: #9fa8da;")
        info_layout.addWidget(self.tail_ratio_label)

        self.source_dir_label = QLabel("Source: None", self.info_box)
        self.source_dir_label.setStyleSheet("font-size: 11px; color: #9fa8da;")
        self.source_dir_label.setWordWrap(True)
        info_layout.addWidget(self.source_dir_label)

        layout.addWidget(self.info_box)

        # Progress bar
        self.progress_bar = QProgressBar(panel)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1a1f36; border: none; border-radius: 4px; } "
            "QProgressBar::chunk { background-color: #38bdf8; border-radius: 4px; }"
        )
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.progress_msg_label = QLabel("", panel)
        self.progress_msg_label.setStyleSheet("font-size: 11px; color: #38bdf8;")
        self.progress_msg_label.hide()
        layout.addWidget(self.progress_msg_label)

        # Action: Generate Histograms
        self.generate_btn = QPushButton("▶ Generate Histograms", panel)
        self.generate_btn.setFixedHeight(36)
        theme.set_accent_btn(self.generate_btn)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.generate_btn)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Build the right panel with dual-axis histograms, threshold sliders, and KPI badges."""
        panel = QFrame(self)
        panel.setObjectName("dark_cal_right_panel")
        panel.setStyleSheet(
            "QFrame#dark_cal_right_panel { background-color: #16213e; "
            "border: 1px solid #2d3561; border-radius: 10px; }"
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Section Header
        lbl_title = QLabel("2. Diagnostic Distributions & Threshold Tuning", panel)
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_title)

        self.figure = Figure(facecolor="#14172b")
        self.canvas = SafeFigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(240)

        self.ax_std = None
        self.ax_res = None
        self.ax_std_linear = None
        self.ax_res_linear = None

        self._init_empty_axes()
        layout.addWidget(self.canvas, 1)

        # Sliders & KPI Container
        sliders_box = QFrame(panel)
        sliders_box.setStyleSheet(
            "QFrame { background-color: #12182b; border: 1px solid #2d3561; border-radius: 8px; padding: 8px; }"
        )
        sliders_layout = QVBoxLayout(sliders_box)
        sliders_layout.setContentsMargins(12, 10, 12, 10)
        sliders_layout.setSpacing(10)

        # Row 1: StdDev Threshold Slider
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        lbl_std = QLabel("Tier 1 Max StdDev (σ):", sliders_box)
        lbl_std.setFixedWidth(160)
        lbl_std.setStyleSheet("font-weight: bold; font-size: 12px; color: #38bdf8;")
        row1.addWidget(lbl_std)

        self.stddev_slider = QSlider(Qt.Horizontal, sliders_box)
        self.stddev_slider.setRange(1, 200)
        self.stddev_slider.setValue(int(self._stddev_thresh))
        self.stddev_slider.valueChanged.connect(self._handle_stddev_slider_moved)
        row1.addWidget(self.stddev_slider, 1)

        self.stddev_val_label = QLabel(f"{self._stddev_thresh:.1f} ADU", sliders_box)
        self.stddev_val_label.setFixedWidth(85)
        self.stddev_val_label.setAlignment(Qt.AlignCenter)
        self.stddev_val_label.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-weight: bold; "
            "font-size: 12px; border: 1px solid #334155; border-radius: 6px; padding: 3px 6px;"
        )
        row1.addWidget(self.stddev_val_label)
        sliders_layout.addLayout(row1)

        # Row 2: Residual Threshold Slider
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        lbl_res = QLabel("Tier 2 Max Residual (Δ):", sliders_box)
        lbl_res.setFixedWidth(160)
        lbl_res.setStyleSheet("font-weight: bold; font-size: 12px; color: #f59e0b;")
        row2.addWidget(lbl_res)

        self.absdev_slider = QSlider(Qt.Horizontal, sliders_box)
        self.absdev_slider.setRange(1, 300)
        self.absdev_slider.setValue(int(self._absdev_thresh))
        self.absdev_slider.valueChanged.connect(self._handle_absdev_slider_moved)
        row2.addWidget(self.absdev_slider, 1)

        self.absdev_val_label = QLabel(f"{self._absdev_thresh:.1f} ADU", sliders_box)
        self.absdev_val_label.setFixedWidth(85)
        self.absdev_val_label.setAlignment(Qt.AlignCenter)
        self.absdev_val_label.setStyleSheet(
            "background-color: #1e293b; color: #f59e0b; font-weight: bold; "
            "font-size: 12px; border: 1px solid #334155; border-radius: 6px; padding: 3px 6px;"
        )
        row2.addWidget(self.absdev_val_label)
        sliders_layout.addLayout(row2)

        # KPI Badges Row (Incremental Tiering Contributions)
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)

        self.tier1_kpi_label = QLabel("Tier 1 (σ < 40.0): - % surviving", sliders_box)
        self.tier1_kpi_label.setStyleSheet(
            "background-color: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; "
            "color: #38bdf8; border-radius: 6px; padding: 4px 8px; font-weight: bold; font-size: 11px;"
        )
        kpi_row.addWidget(self.tier1_kpi_label, 1)

        self.tier2_kpi_label = QLabel("Tier 2 (Δ < 60.0): - % surviving", sliders_box)
        self.tier2_kpi_label.setStyleSheet(
            "background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; "
            "color: #f59e0b; border-radius: 6px; padding: 4px 8px; font-weight: bold; font-size: 11px;"
        )
        kpi_row.addWidget(self.tier2_kpi_label, 1)

        self.final_mask_kpi_label = QLabel("Final Mask: - % active", sliders_box)
        self.final_mask_kpi_label.setStyleSheet(
            "background-color: rgba(5, 150, 105, 0.15); border: 1px solid #059669; "
            "color: #34d399; border-radius: 6px; padding: 4px 8px; font-weight: bold; font-size: 11px;"
        )
        kpi_row.addWidget(self.final_mask_kpi_label, 1)

        sliders_layout.addLayout(kpi_row)
        layout.addWidget(sliders_box)

        # Bottom Action Bar: Save Status & Save Button
        save_bar = QHBoxLayout()
        save_bar.setSpacing(10)

        self.save_status_label = QLabel("", panel)
        self.save_status_label.setStyleSheet("font-size: 12px; color: #34d399; font-weight: bold;")
        save_bar.addWidget(self.save_status_label, 1)

        self.save_btn = QPushButton("💾 Save Dark Mask", panel)
        self.save_btn.setFixedHeight(36)
        self.save_btn.setStyleSheet("font-size: 13px; font-weight: bold; padding: 6px 16px;")
        theme.set_tool_btn(self.save_btn)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_clicked)
        save_bar.addWidget(self.save_btn)

        layout.addLayout(save_bar)
        return panel

    def _init_empty_axes(self) -> None:
        """Draw placeholder styled axes prior to data loading."""
        self.figure.clf()
        self.ax_std = self.figure.add_subplot(121)
        self.ax_res = self.figure.add_subplot(122)
        self.ax_std_linear = None
        self.ax_res_linear = None
        self._std_span = None
        self._res_span = None

        self.ax_std.set_facecolor("#14172b")
        self.ax_std.set_title("Pixel Noise StdDev (σ)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_std.set_xlabel("Standard Deviation σ (ADU)", color="#9fa8da", fontsize=10)
        self.ax_std.set_ylabel("Log Count", color="#38bdf8", fontsize=10)
        self.ax_std.tick_params(colors="#38bdf8", labelsize=9)
        for spine in self.ax_std.spines.values():
            spine.set_color("#2d3561")
        self.ax_std.grid(True, linestyle=":", color="#2d3561", alpha=0.6)
        self.ax_std.text(
            0.5, 0.5, "No data loaded\nClick [▶ Generate Histograms]",
            ha="center", va="center", color="#5c6bc0", transform=self.ax_std.transAxes, fontsize=10,
        )

        self.ax_res.set_facecolor("#14172b")
        self.ax_res.set_title("93rd-Percentile Residual (Δ)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_res.set_xlabel("Excursion Residual Δ (ADU)", color="#9fa8da", fontsize=10)
        self.ax_res.set_ylabel("Log Count", color="#f59e0b", fontsize=10)
        self.ax_res.tick_params(colors="#f59e0b", labelsize=9)
        for spine in self.ax_res.spines.values():
            spine.set_color("#2d3561")
        self.ax_res.grid(True, linestyle=":", color="#2d3561", alpha=0.6)
        self.ax_res.text(
            0.5, 0.5, "No data loaded\nClick [▶ Generate Histograms]",
            ha="center", va="center", color="#5c6bc0", transform=self.ax_res.transAxes, fontsize=10,
        )

        self.figure.tight_layout(pad=2.0)

    # ------------------------------------------------------------------
    # Drag-and-Drop & File Ingest
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
        self.tier1_kpi_label.setText("Tier 1 (σ < 40.0): - % surviving")
        self.tier2_kpi_label.setText("Tier 2 (Δ < 60.0): - % surviving")
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

        unique_paths = list(dict.fromkeys(discovered))
        natural_sort(unique_paths)

        self.dark_paths = unique_paths
        self.dark_frame_count = len(unique_paths)

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

        self._current_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_worker_progress(self, current: int, total: int) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)

    def _on_worker_msg(self, msg: str) -> None:
        self.progress_msg_label.setText(msg)

    def _on_worker_error(self, err: str) -> None:
        logger.error("Dark mask worker error: %s", err)
        self.progress_msg_label.setText(f"Error: {err}")
        self.progress_msg_label.setStyleSheet("color: #ef4444; font-size: 11px;")

    def _on_worker_finished(self) -> None:
        self.generate_btn.setEnabled(self.dark_frame_count > 0)
        self.progress_bar.hide()
        self.progress_msg_label.hide()
        self._current_worker = None

    def _on_diagnostics_ready(self, diag: DarkDiagnostics) -> None:
        self._diagnostics = diag
        self.save_btn.setEnabled(True)
        self._render_histograms()
        self._update_kpis_and_cutlines()

    # ------------------------------------------------------------------
    # Histogram Rendering & Cutlines
    # ------------------------------------------------------------------

    def _render_histograms(self) -> None:
        """Draw dual-axis log-scale and transparent linear histograms with dynamic masked spans."""
        if self._diagnostics is None:
            return

        std_data = self._diagnostics.per_pixel_stddev.ravel()
        res_data = self._diagnostics.pct93_residual.ravel()

        std_finite = std_data[np.isfinite(std_data)]
        res_finite = res_data[np.isfinite(res_data)]

        self.figure.clf()
        self.ax_std = self.figure.add_subplot(121)
        self.ax_res = self.figure.add_subplot(122)
        self.ax_std_linear = self.ax_std.twinx()
        self.ax_res_linear = self.ax_res.twinx()

        # --- Subplot 1: StdDev ---
        self.ax_std.set_facecolor("#14172b")
        self.ax_std.set_title("Pixel Noise StdDev (σ)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_std.set_xlabel("Standard Deviation σ (ADU)", color="#9fa8da", fontsize=10)
        self.ax_std.set_ylabel("Log Count", color="#38bdf8", fontsize=10)
        self.ax_std_linear.set_ylabel("Linear Count", color="#9fa8da", fontsize=9, alpha=0.7)
        self.ax_std.tick_params(colors="#38bdf8", labelsize=9)
        self.ax_std_linear.tick_params(colors="#9fa8da", labelsize=8)
        for spine in self.ax_std.spines.values():
            spine.set_color("#2d3561")
        for spine in self.ax_std_linear.spines.values():
            spine.set_color("#2d3561")
        self.ax_std.grid(True, linestyle=":", color="#2d3561", alpha=0.6)

        std_max = 100.0
        if len(std_finite) > 0:
            std_max = max(100.0, float(np.percentile(std_finite, 99.9)))
            # Primary axis: Log scale
            self.ax_std.hist(
                std_finite,
                bins=60,
                range=(0, std_max),
                color="#38bdf8",
                edgecolor="#0284c7",
                alpha=0.75,
                log=True,
                label="Log Count",
            )
            # Secondary axis: Linear scale overlay
            self.ax_std_linear.hist(
                std_finite,
                bins=60,
                range=(0, std_max),
                color="#38bdf8",
                edgecolor="none",
                alpha=0.25,
                log=False,
                label="Linear Count",
            )
            self.stddev_slider.setMaximum(max(200, int(std_max * 1.2)))

        self._std_max = std_max

        # Cutline & Dynamic Masked Pixel Span
        self._std_cutline = self.ax_std.axvline(
            x=self._stddev_thresh,
            color="#ef4444",
            linestyle="--",
            linewidth=1.8,
            label=f"Cut: {self._stddev_thresh:.1f}",
        )
        self._std_span = self.ax_std.axvspan(
            self._stddev_thresh,
            std_max,
            color="#ef4444",
            alpha=0.22,
            label="Masked Pixels",
        )
        self.ax_std.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=8, loc="upper right")

        # --- Subplot 2: Residual ---
        self.ax_res.clear()
        self.ax_res_linear.clear()
        self.ax_res.set_facecolor("#14172b")
        self.ax_res.set_title("93rd-Percentile Residual (Δ)", color="#e8eaf6", fontsize=11, fontweight="bold", pad=8)
        self.ax_res.set_xlabel("Excursion Residual Δ (ADU)", color="#9fa8da", fontsize=10)
        self.ax_res.set_ylabel("Log Count", color="#f59e0b", fontsize=10)
        self.ax_res_linear.set_ylabel("Linear Count", color="#9fa8da", fontsize=9, alpha=0.7)
        self.ax_res.tick_params(colors="#f59e0b", labelsize=9)
        self.ax_res_linear.tick_params(colors="#9fa8da", labelsize=8)
        for spine in self.ax_res.spines.values():
            spine.set_color("#2d3561")
        for spine in self.ax_res_linear.spines.values():
            spine.set_color("#2d3561")
        self.ax_res.grid(True, linestyle=":", color="#2d3561", alpha=0.6)

        res_max = 150.0
        if len(res_finite) > 0:
            res_max = max(150.0, float(np.percentile(res_finite, 99.9)))
            # Primary axis: Log scale
            self.ax_res.hist(
                res_finite,
                bins=60,
                range=(0, res_max),
                color="#f59e0b",
                edgecolor="#d97706",
                alpha=0.75,
                log=True,
                label="Log Count",
            )
            # Secondary axis: Linear scale overlay
            self.ax_res_linear.hist(
                res_finite,
                bins=60,
                range=(0, res_max),
                color="#f59e0b",
                edgecolor="none",
                alpha=0.25,
                log=False,
                label="Linear Count",
            )
            self.absdev_slider.setMaximum(max(300, int(res_max * 1.2)))

        self._res_max = res_max

        # Cutline & Dynamic Masked Pixel Span
        self._res_cutline = self.ax_res.axvline(
            x=self._absdev_thresh,
            color="#ef4444",
            linestyle="--",
            linewidth=1.8,
            label=f"Cut: {self._absdev_thresh:.1f}",
        )
        self._res_span = self.ax_res.axvspan(
            self._absdev_thresh,
            res_max,
            color="#ef4444",
            alpha=0.22,
            label="Masked Pixels",
        )
        self.ax_res.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=8, loc="upper right")

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slider & KPI Updates
    # ------------------------------------------------------------------

    def _handle_stddev_slider_moved(self, val: int) -> None:
        self._on_stddev_slider_changed(float(val))

    def _handle_absdev_slider_moved(self, val: int) -> None:
        self._on_absdev_slider_changed(float(val))

    def _on_stddev_slider_changed(self, val: float | int) -> None:
        self._stddev_thresh = float(val)
        self.stddev_val_label.setText(f"{self._stddev_thresh:.1f} ADU")

        if self.stddev_slider.value() != int(val):
            self.stddev_slider.blockSignals(True)
            self.stddev_slider.setValue(int(val))
            self.stddev_slider.blockSignals(False)

        if self._std_cutline is not None:
            self._std_cutline.set_xdata([self._stddev_thresh, self._stddev_thresh])
            self._std_cutline.set_label(f"Cut: {self._stddev_thresh:.1f}")
            if self._std_span is not None:
                self._std_span.remove()
                std_max = getattr(self, "_std_max", max(200.0, self._stddev_thresh * 1.5))
                self._std_span = self.ax_std.axvspan(
                    self._stddev_thresh,
                    std_max,
                    color="#ef4444",
                    alpha=0.22,
                    label="Masked Pixels",
                )
            self.ax_std.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=8, loc="upper right")
            self.canvas.draw_idle()

        self._update_kpis_and_cutlines()

    def _on_absdev_slider_changed(self, val: float | int) -> None:
        self._absdev_thresh = float(val)
        self.absdev_val_label.setText(f"{self._absdev_thresh:.1f} ADU")

        if self.absdev_slider.value() != int(val):
            self.absdev_slider.blockSignals(True)
            self.absdev_slider.setValue(int(val))
            self.absdev_slider.blockSignals(False)

        if self._res_cutline is not None:
            self._res_cutline.set_xdata([self._absdev_thresh, self._absdev_thresh])
            self._res_cutline.set_label(f"Cut: {self._absdev_thresh:.1f}")
            if self._res_span is not None:
                self._res_span.remove()
                res_max = getattr(self, "_res_max", max(300.0, self._absdev_thresh * 1.5))
                self._res_span = self.ax_res.axvspan(
                    self._absdev_thresh,
                    res_max,
                    color="#ef4444",
                    alpha=0.22,
                    label="Masked Pixels",
                )
            self.ax_res.legend(facecolor="#16213e", edgecolor="#2d3561", labelcolor="#e8eaf6", fontsize=8, loc="upper right")
            self.canvas.draw_idle()

        self._update_kpis_and_cutlines()

    def _update_kpis_and_cutlines(self) -> None:
        """Recalculate survival percentages with true incremental marginal contribution."""
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
        # Pixels that passed StdDev filter but are eliminated specifically by Residual filter
        removed_by_tail_marginal = int(np.count_nonzero(m_std & (~m_tail)))
        surv_final = int(np.count_nonzero(m_final))

        pct_std = (surv_std / total_pixels) * 100.0
        pct_tail = (surv_tail / total_pixels) * 100.0
        pct_tail_marginal = (removed_by_tail_marginal / total_pixels) * 100.0
        pct_final = (surv_final / total_pixels) * 100.0

        self.tier1_kpi_label.setText(
            f"Tier 1 (σ < {self._stddev_thresh:.1f}): {pct_std:.2f}% surviving ({surv_std:,} px)"
        )
        if removed_by_tail_marginal > 0:
            self.tier2_kpi_label.setText(
                f"Tier 2 (Δ < {self._absdev_thresh:.1f}): {pct_tail:.2f}% surviving (+{pct_tail_marginal:.2f}% marginal, {removed_by_tail_marginal:,} px)"
            )
        else:
            self.tier2_kpi_label.setText(
                f"Tier 2 (Δ < {self._absdev_thresh:.1f}): {pct_tail:.2f}% surviving (+0 px marginal)"
            )
        self.final_mask_kpi_label.setText(
            f"Final Mask: {pct_final:.2f}% active ({surv_final:,} px)"
        )

    # ------------------------------------------------------------------
    # Persistence & Saving
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        """Apply current thresholds and persist dark mask products to dark_mask_store."""
        if self._diagnostics is None:
            return

        stage1_res = apply_dark_thresholds(
            self._diagnostics,
            stddev_thresh=self._stddev_thresh,
            absdev_thresh=self._absdev_thresh,
        )
        final_mask = stage1_res.final_mask if hasattr(stage1_res, "final_mask") else stage1_res

        total_pixels = int(final_mask.size)
        surviving = int(np.count_nonzero(final_mask))
        suppression_pct = ((total_pixels - surviving) / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        source_dir = str(Path(self.dark_paths[0]).parent) if self.dark_paths else "synthetic"
        mask_dir = getattr(dark_mask_store, "DARK_MASK_DIR", None)
        if hasattr(calibration_store, "DARK_CAL_DIR") and calibration_store.DARK_CAL_DIR != dark_mask_store.DEFAULT_MASK_DIR:
            mask_dir = calibration_store.DARK_CAL_DIR

        record = dark_mask_store.save_dark_mask(
            med_dark=self._diagnostics.med_dark,
            final_mask=final_mask,
            stddev_thresh=self._stddev_thresh,
            absdev_thresh=self._absdev_thresh,
            tail_ratio=self._tail_ratio,
            dark_frame_count=self.dark_frame_count or self._diagnostics.dark_frame_count,
            surviving_pixels=surviving,
            total_pixels=total_pixels,
            suppression_pct=suppression_pct,
            source_dir=source_dir,
            mask_dir=mask_dir,
        )

        pct_active = (surviving / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        self.save_status_label.setText(
            f"✓ Saved calibration / dark mask: {pct_active:.2f}% active pixels ({surviving:,} px active)"
        )
        self.save_status_label.setStyleSheet("color: #34d399; font-weight: bold; font-size: 12px;")

    # ------------------------------------------------------------------
    # Cleanup & Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Explicitly dispose of Matplotlib canvas and subplots."""
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cleanup()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Navbar & Co-Pilot Docking
    # ------------------------------------------------------------------

    def _handle_back_clicked(self) -> None:
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


# Backward compatibility alias
DarkCalibrationView = DarkMaskingView
