"""Single-Photon Clustering Studio View.

Provides a unified 3-mode scientific analysis workstation:
1. Dashboard Mode:
   - Live accumulating 2D photon event map during Stage 2 extraction.
   - 1D logarithmic IntDen histogram with interactive cutlines.
   - RangeSlider for instant (<50ms) in-memory single-photon gating.
2. Frame Inspector Mode:
   - Dark-subtracted frame browser with green/red cluster bounding boxes.
   - Cyan '+' centroid markers at exact sub-pixel coordinates.
   - Interactive frame scrubbing and playback controls.
3. Chunk Inspector Mode:
   - Per-chunk reconstructed event map sequence browser.
   - Custom colormaps and dynamic intensity contrast clamping.
   - Chunk scrubbing controls.

Includes contextual KPI cards, stale parameter tracking, background chunk exports,
and clean Matplotlib figure teardown on close.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from rixs_app.ui.widgets import SafeFigureCanvasQTAgg
from PySide6.QtCore import Qt, QThreadPool, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rixs_app.core.dark_mask_store import load_dark_mask
from rixs_app.core.photon_clustering import (
    ClusterConfig,
    ReconstructionConfig,
    ReconstructionResult,
)
from rixs_app.ui import theme

CanvasCls = SafeFigureCanvasQTAgg or FigureCanvasQTAgg
from rixs_app.ui.clustering_slideshow.manager import ClusteringManager
from rixs_app.ui.clustering_slideshow.workers import (
    ChunkSaveWorker,
    ClusterPipelineWorker,
)
from rixs_app.ui.widgets import RangeSlider

logger = logging.getLogger(__name__)


class KPICard(QFrame):
    """Elevated dark-themed KPI summary card."""

    def __init__(
        self,
        title: str,
        value: str = "--",
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        theme.set_squircle_card(self)
        self.setMinimumHeight(70)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title, self)
        self._title_lbl.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: bold;")
        layout.addWidget(self._title_lbl)

        self._value_lbl = QLabel(value, self)
        self._value_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(self._value_lbl)

        self._sub_lbl = QLabel(subtitle, self)
        self._sub_lbl.setStyleSheet("font-size: 10px; color: #64748b;")
        layout.addWidget(self._sub_lbl)

    def set_content(self, value: str, subtitle: str = "", color: str = "#38bdf8") -> None:
        """Update KPI value and subtitle."""
        self._value_lbl.setText(value)
        self._value_lbl.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        self._sub_lbl.setText(subtitle)


class ClusteringStudioView(QWidget):
    """Single-Photon Clustering Studio multi-mode workstation.

    Args:
        parent: Optional parent QWidget.
        on_back: Optional callback when clicking '❮ Back to Home'.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_back = on_back
        self.manager = ClusteringManager()

        self._current_mode = "Dashboard"
        self._current_frame_idx = 0
        self._current_chunk_idx = 0

        self._pipeline_worker: ClusterPipelineWorker | None = None
        self._save_worker: ChunkSaveWorker | None = None
        self._copilot_btn: QPushButton | None = None

        # Matplotlib references
        self._fig_dashboard: Figure | None = None
        self._fig_frame: Figure | None = None
        self._fig_chunk: Figure | None = None

        self._im_dashboard_event = None
        self._im_frame = None
        self._im_chunk = None

        self._line_low = None
        self._line_high = None

        self._current_zoom_level: float = 1.0
        self._clamping_floor: float = 0.0
        self._clamping_ceiling: float = 1.0
        self._clamping_slider_max: float = 1.0

        self._accum_timer = QTimer(self)
        self._accum_timer.setInterval(100)
        self._accum_timer.setSingleShot(True)
        self._accum_timer.timeout.connect(self._on_accum_timer_tick)

        self._running_cluster_count = 0
        self._progressive_event_map = np.zeros((2048, 2048), dtype=np.float32)

        self._init_ui()
        self._mode_combo = self.mode_combo
        self._kpi_cards = [self.kpi_1, self.kpi_2, self.kpi_3, self.kpi_4]
        self._kpi_labels = [self.kpi_1._title_lbl, self.kpi_2._title_lbl, self.kpi_3._title_lbl, self.kpi_4._title_lbl]
        self._stale_warning_label = self._stale_lbl

        self._sig_thresh_low_spin = QDoubleSpinBox(self)
        self._sig_thresh_low_spin.setRange(1.0, 10000.0)
        self._sig_thresh_low_spin.setValue(45.0)
        self._sig_thresh_low_spin.valueChanged.connect(self._on_stage2_param_changed)
        self._sig_thresh_low_spin.hide()

    def _on_stage2_param_changed(self) -> None:
        self.manager.mark_stage2_stale(True)
        self._stale_banner.show()
        self._stale_warning_label.show()

    # ------------------------------------------------------------------
    # UI Initialization
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 10, 14, 12)
        main_layout.setSpacing(10)

        # 1. Header Navbar
        navbar = self._create_navbar()
        main_layout.addWidget(navbar)

        # 2. Stale Parameters Warning Banner
        self._stale_banner = self._create_stale_banner()
        main_layout.addWidget(self._stale_banner)
        self._stale_banner.hide()

        # 3. Contextual KPI Cards
        kpi_row = self._create_kpi_row()
        main_layout.addWidget(kpi_row)

        # 4. Mode Stack (Dashboard, Frame Inspector, Chunk Inspector)
        self._mode_stack = QStackedWidget(self)
        self._dashboard_view = self._create_dashboard_view()
        self._frame_view = self._create_frame_view()
        self._chunk_view = self._create_chunk_view()

        self._mode_stack.addWidget(self._dashboard_view)    # Index 0
        self._mode_stack.addWidget(self._frame_view)        # Index 1
        self._mode_stack.addWidget(self._chunk_view)        # Index 2

        main_layout.addWidget(self._mode_stack, stretch=1)

        # 5. Bottom Status / Progress Bar
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_navbar(self) -> QWidget:
        navbar = QWidget(self)
        layout = QHBoxLayout(navbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._back_btn = QPushButton("❮ Back to Home", navbar)
        theme.set_tool_btn(self._back_btn)
        self._back_btn.clicked.connect(self._handle_back_clicked)
        layout.addWidget(self._back_btn)

        # Mode Selector
        layout.addWidget(QLabel("Mode:", navbar))
        self.mode_combo = QComboBox(navbar)
        theme.set_mode_selector(self.mode_combo)
        self.mode_combo.addItems([
            "Dashboard",
            "Frame Inspector",
            "Chunk Inspector",
        ])
        self.mode_combo.currentIndexChanged.connect(self._handle_mode_changed)
        layout.addWidget(self.mode_combo)

        layout.addStretch(1)

        # Action Buttons
        self._rerun_btn = QPushButton("🔄 Re-run Extraction", navbar)
        theme.set_tool_btn(self._rerun_btn)
        self._rerun_btn.clicked.connect(self._handle_rerun_clicked)
        layout.addWidget(self._rerun_btn)

        self._save_chunks_btn = QPushButton("💾 Export Chunks...", navbar)
        theme.set_accent_btn(self._save_chunks_btn)
        self._save_chunks_btn.clicked.connect(self._handle_save_chunks_clicked)
        layout.addWidget(self._save_chunks_btn)

        # Co-Pilot docking container
        self._copilot_container = QWidget(navbar)
        self._copilot_container_layout = QHBoxLayout(self._copilot_container)
        self._copilot_container_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._copilot_container)

        return navbar

    def _create_stale_banner(self) -> QFrame:
        banner = QFrame(self)
        theme.set_stale_warning(banner)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self._stale_lbl = QLabel(
            "⚠️ Stage 2 extraction parameters changed (stale). Re-run cluster extraction to synchronize results.",
            banner,
        )
        self._stale_lbl.setStyleSheet("color: #fbbf24; font-weight: bold;")
        layout.addWidget(self._stale_lbl, stretch=1)

        rerun_action_btn = QPushButton("Re-run Extraction Now", banner)
        theme.set_amber_btn(rerun_action_btn)
        rerun_action_btn.clicked.connect(self._handle_rerun_clicked)
        layout.addWidget(rerun_action_btn)

        return banner

    def _create_kpi_row(self) -> QWidget:
        kpi_container = QWidget(self)
        layout = QHBoxLayout(kpi_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.kpi_1 = KPICard("TOTAL FRAMES", "0 / 0", "0% processed", kpi_container)
        self.kpi_2 = KPICard("TOTAL CLUSTERS", "0", "0 clusters / frame", kpi_container)
        self.kpi_3 = KPICard("SINGLE PHOTONS", "0 (0.0%)", "Stage 3 accepted", kpi_container)
        self.kpi_4 = KPICard("REJECTED EVENTS", "0 (0.0%)", "Noise / Pileup / Shape", kpi_container)

        layout.addWidget(self.kpi_1)
        layout.addWidget(self.kpi_2)
        layout.addWidget(self.kpi_3)
        layout.addWidget(self.kpi_4)

        return kpi_container

    # ------------------------------------------------------------------
    # Mode 1: Dashboard View (2D Event Map + 1D IntDen Cutlines)
    # ------------------------------------------------------------------

    def _create_dashboard_view(self) -> QWidget:
        view = QWidget()
        layout = QHBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal, view)

        # Left Panel: 2D Event Map Canvas
        left_panel = QFrame(splitter)
        theme.set_squircle_card(left_panel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        self._dash_title_lbl = QLabel("2D Single-Photon Event Map", left_panel)
        self._dash_title_lbl.hide()

        # Row 1: Zoom In, Zoom Out, Reset View, Zoom label, spacer, colormap combo.
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self._zoom_in_btn = QPushButton("🔍+ Zoom In", left_panel)
        self._zoom_out_btn = QPushButton("🔍- Zoom Out", left_panel)
        self._zoom_reset_btn = QPushButton("↺ Reset View", left_panel)
        theme.set_tool_btn(self._zoom_in_btn)
        theme.set_tool_btn(self._zoom_out_btn)
        theme.set_tool_btn(self._zoom_reset_btn)

        self._zoom_lbl = QLabel("Zoom: 1×", left_panel)
        self._zoom_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; margin-left: 4px;")

        self._zoom_in_btn.clicked.connect(self._handle_zoom_in)
        self._zoom_out_btn.clicked.connect(self._handle_zoom_out)
        self._zoom_reset_btn.clicked.connect(self._handle_zoom_reset)

        row1.addWidget(self._zoom_in_btn)
        row1.addWidget(self._zoom_out_btn)
        row1.addWidget(self._zoom_reset_btn)
        row1.addWidget(self._zoom_lbl)
        row1.addStretch(1)

        self._dash_cmap_combo = QComboBox(left_panel)
        self._dash_cmap_combo.addItems(["inferno", "viridis", "plasma", "magma", "hot", "gray"])
        self._dash_cmap_combo.currentTextChanged.connect(self._handle_dash_cmap_changed)
        row1.addWidget(self._dash_cmap_combo)

        left_layout.addLayout(row1)

        # Row 2: Intensity Clamping label, Floor QLineEdit, RangeSlider [0.0, 1.0], Ceiling QLineEdit.
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        clamp_lbl = QLabel("Intensity Clamping:", left_panel)
        clamp_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        row2.addWidget(clamp_lbl)

        self._floor_entry = QLineEdit("0.00", left_panel)
        self._floor_entry.setFixedWidth(55)
        self._floor_entry.setAlignment(Qt.AlignCenter)
        self._floor_entry.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; "
            "border-radius: 4px; font-size: 11px; padding: 2px;"
        )
        row2.addWidget(self._floor_entry)

        self._clamping_slider = RangeSlider(left_panel)
        self._clamping_slider.configure_range(0.0, 1.0)
        self._clamping_slider.set_values(0.0, 1.0)
        row2.addWidget(self._clamping_slider, stretch=1)

        self._ceiling_entry = QLineEdit("1.00", left_panel)
        self._ceiling_entry.setFixedWidth(55)
        self._ceiling_entry.setAlignment(Qt.AlignCenter)
        self._ceiling_entry.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; "
            "border-radius: 4px; font-size: 11px; padding: 2px;"
        )
        row2.addWidget(self._ceiling_entry)

        self._clamping_slider.range_changed.connect(self._handle_clamping_changed)
        self._floor_entry.returnPressed.connect(self._on_floor_entry_submitted)
        self._floor_entry.editingFinished.connect(self._on_floor_entry_submitted)
        self._ceiling_entry.returnPressed.connect(self._on_ceiling_entry_submitted)
        self._ceiling_entry.editingFinished.connect(self._on_ceiling_entry_submitted)

        left_layout.addLayout(row2)

        self._fig_dashboard = Figure(figsize=(6, 5), facecolor="#14172b")
        self._canvas_dashboard = CanvasCls(self._fig_dashboard)
        self._ax_dashboard_event = self._fig_dashboard.add_subplot(111)
        self._ax_dashboard_event.set_facecolor("#16213e")
        self._ax_dashboard_event.set_title("Reconstructed Photon Map", color="#f8fafc", fontsize=11)
        self._ax_dashboard_event.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_dashboard_event.spines.values():
            spine.set_color("#2d3561")
        self._fig_dashboard.tight_layout()

        left_layout.addWidget(self._canvas_dashboard, stretch=1)
        splitter.addWidget(left_panel)

        # Right Panel: 1D Histogram & Stage 3 Cuts
        right_panel = QFrame(splitter)
        theme.set_squircle_card(right_panel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        right_header = QLabel("IntDen Distribution & Energy Window Gating", right_panel)
        right_header.setStyleSheet("font-weight: bold; color: #e2e8f0; font-size: 13px;")
        right_layout.addWidget(right_header)

        self._fig_hist = Figure(figsize=(5, 3.5), facecolor="#14172b")
        self._canvas_hist = CanvasCls(self._fig_hist)
        self._ax_hist = self._fig_hist.add_subplot(111)
        self._ax_hist.set_facecolor("#16213e")
        self._ax_hist.set_title("Cluster Integrated Density (IntDen)", color="#f8fafc", fontsize=11)
        self._ax_hist.set_xlabel("IntDen (ADU)", color="#e2e8f0", fontsize=9)
        self._ax_hist.set_ylabel("Counts (log)", color="#e2e8f0", fontsize=9)
        self._ax_hist.tick_params(colors="#94a3b8", labelsize=8)
        for spine in self._ax_hist.spines.values():
            spine.set_color("#2d3561")
        self._fig_hist.tight_layout()

        right_layout.addWidget(self._canvas_hist, stretch=1)

        # Interactive RangeSlider for IntDen Cuts
        slider_box = QWidget(right_panel)
        s_layout = QVBoxLayout(slider_box)
        s_layout.setContentsMargins(4, 0, 4, 0)
        s_layout.setSpacing(4)

        s_label_row = QHBoxLayout()
        self._intden_cut_lbl = QLabel("Single-Photon Window: 120.0 - 320.0 ADU", slider_box)
        self._intden_cut_lbl.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 11px;")
        s_label_row.addWidget(self._intden_cut_lbl)
        s_label_row.addStretch(1)
        s_layout.addLayout(s_label_row)

        self.intden_slider = RangeSlider(slider_box)
        self.intden_slider.configure_range(0.0, 1500.0)
        self.intden_slider.set_values(120.0, 320.0)
        self.intden_slider.range_changed.connect(self._handle_intden_slider_changed)
        self.intden_slider.slider_released.connect(self._handle_intden_slider_released)
        s_layout.addWidget(self.intden_slider)
        right_layout.addWidget(slider_box)

        # Shape Filters Box
        shape_box = QGroupBox("Shape & Resolution Filtering", right_panel)
        shape_grid = QGridLayout(shape_box)
        shape_grid.setContentsMargins(10, 8, 10, 8)
        shape_grid.setSpacing(8)

        shape_grid.addWidget(QLabel("Max Area (px):", shape_box), 0, 0)
        self.max_area_spin = QSpinBox(shape_box)
        self.max_area_spin.setRange(1, 100)
        self.max_area_spin.setValue(9)
        self.max_area_spin.valueChanged.connect(self._handle_shape_filter_changed)
        shape_grid.addWidget(self.max_area_spin, 0, 1)

        shape_grid.addWidget(QLabel("Min Circ:", shape_box), 0, 2)
        self.min_circ_spin = QDoubleSpinBox(shape_box)
        self.min_circ_spin.setRange(0.0, 1.0)
        self.min_circ_spin.setSingleStep(0.05)
        self.min_circ_spin.setValue(0.3)
        self.min_circ_spin.valueChanged.connect(self._handle_shape_filter_changed)
        shape_grid.addWidget(self.min_circ_spin, 0, 3)

        shape_grid.addWidget(QLabel("Subpixel Factor:", shape_box), 1, 0)
        self.subpixel_spin = QSpinBox(shape_box)
        self.subpixel_spin.setRange(1, 4)
        self.subpixel_spin.setValue(1)
        self.subpixel_spin.valueChanged.connect(self._handle_shape_filter_changed)
        shape_grid.addWidget(self.subpixel_spin, 1, 1)

        right_layout.addWidget(shape_box)
        splitter.addWidget(right_panel)

        splitter.setSizes([550, 450])
        layout.addWidget(splitter)
        return view

    # ------------------------------------------------------------------
    # Mode 2: Frame Inspector View
    # ------------------------------------------------------------------

    def _create_frame_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Toolbar / Controls
        ctrl_bar = QFrame(view)
        theme.set_squircle_card(ctrl_bar)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 6, 12, 6)
        ctrl_layout.setSpacing(8)

        self._frame_first_btn = QPushButton("⏮ First", ctrl_bar)
        theme.set_tool_btn(self._frame_first_btn)
        self._frame_first_btn.clicked.connect(self._handle_frame_first)
        ctrl_layout.addWidget(self._frame_first_btn)

        self._frame_prev_btn = QPushButton("◀ Prev", ctrl_bar)
        theme.set_tool_btn(self._frame_prev_btn)
        self._frame_prev_btn.clicked.connect(self.prev_frame)
        ctrl_layout.addWidget(self._frame_prev_btn)

        self.frame_spin = QSpinBox(ctrl_bar)
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.valueChanged.connect(self._handle_frame_spin_changed)
        ctrl_layout.addWidget(self.frame_spin)

        self._frame_max_lbl = QLabel("/ 0", ctrl_bar)
        ctrl_layout.addWidget(self._frame_max_lbl)

        self.frame_slider = QSlider(Qt.Horizontal, ctrl_bar)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        self.frame_slider.valueChanged.connect(self.frame_spin.setValue)
        ctrl_layout.addWidget(self.frame_slider, stretch=1)

        self._frame_next_btn = QPushButton("Next ▶", ctrl_bar)
        theme.set_tool_btn(self._frame_next_btn)
        self._frame_next_btn.clicked.connect(self.next_frame)
        ctrl_layout.addWidget(self._frame_next_btn)

        self._frame_last_btn = QPushButton("Last ⏭", ctrl_bar)
        theme.set_tool_btn(self._frame_last_btn)
        self._frame_last_btn.clicked.connect(self._handle_frame_last)
        ctrl_layout.addWidget(self._frame_last_btn)

        layout.addWidget(ctrl_bar)
        # Frame Canvas with Cluster Overlays
        canvas_card = QFrame(view)
        theme.set_squircle_card(canvas_card)
        c_layout = QVBoxLayout(canvas_card)
        c_layout.setContentsMargins(8, 8, 8, 8)

        self._fig_frame = Figure(figsize=(8, 6), facecolor="#14172b")
        self._canvas_frame = CanvasCls(self._fig_frame)
        self._ax_frame = self._fig_frame.add_subplot(111)
        self._ax_frame.set_facecolor("#16213e")
        self._ax_frame.set_title("Dark-Subtracted Frame with Photon Cluster Overlays", color="#f8fafc", fontsize=11)
        self._ax_frame.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_frame.spines.values():
            spine.set_color("#2d3561")
        self._fig_frame.tight_layout()

        c_layout.addWidget(self._canvas_frame, stretch=1)
        layout.addWidget(canvas_card, stretch=1)

        return view

    # ------------------------------------------------------------------
    # Mode 3: Chunk Inspector View
    # ------------------------------------------------------------------

    def _create_chunk_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Chunk Navigation Bar
        ctrl_bar = QFrame(view)
        theme.set_squircle_card(ctrl_bar)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 6, 12, 6)
        ctrl_layout.setSpacing(8)

        self._chunk_first_btn = QPushButton("⏮ First Chunk", ctrl_bar)
        theme.set_tool_btn(self._chunk_first_btn)
        self._chunk_first_btn.clicked.connect(self._handle_chunk_first)
        ctrl_layout.addWidget(self._chunk_first_btn)

        self._chunk_prev_btn = QPushButton("◀ Prev Chunk", ctrl_bar)
        theme.set_tool_btn(self._chunk_prev_btn)
        self._chunk_prev_btn.clicked.connect(self.prev_chunk)
        ctrl_layout.addWidget(self._chunk_prev_btn)

        self.chunk_spin = QSpinBox(ctrl_bar)
        self.chunk_spin.setRange(1, 1)
        self.chunk_spin.setValue(1)
        self.chunk_spin.valueChanged.connect(self._handle_chunk_spin_changed)
        ctrl_layout.addWidget(self.chunk_spin)

        self._chunk_max_lbl = QLabel("/ 0", ctrl_bar)
        ctrl_layout.addWidget(self._chunk_max_lbl)

        self.chunk_slider = QSlider(Qt.Horizontal, ctrl_bar)
        self.chunk_slider.setRange(1, 1)
        self.chunk_slider.setValue(1)
        self.chunk_slider.valueChanged.connect(self.chunk_spin.setValue)
        ctrl_layout.addWidget(self.chunk_slider, stretch=1)

        self._chunk_next_btn = QPushButton("Next Chunk ▶", ctrl_bar)
        theme.set_tool_btn(self._chunk_next_btn)
        self._chunk_next_btn.clicked.connect(self.next_chunk)
        ctrl_layout.addWidget(self._chunk_next_btn)

        self._chunk_last_btn = QPushButton("Last Chunk ⏭", ctrl_bar)
        theme.set_tool_btn(self._chunk_last_btn)
        self._chunk_last_btn.clicked.connect(self._handle_chunk_last)
        ctrl_layout.addWidget(self._chunk_last_btn)

        # Colormap selector
        self._chunk_cmap_combo = QComboBox(ctrl_bar)
        self._chunk_cmap_combo.addItems(["inferno", "viridis", "plasma", "magma", "hot", "gray"])
        self._chunk_cmap_combo.currentTextChanged.connect(self._handle_chunk_cmap_changed)
        ctrl_layout.addWidget(self._chunk_cmap_combo)

        layout.addWidget(ctrl_bar)

        # Chunk Canvas
        canvas_card = QFrame(view)
        theme.set_squircle_card(canvas_card)
        c_layout = QVBoxLayout(canvas_card)
        c_layout.setContentsMargins(8, 8, 8, 8)

        self._fig_chunk = Figure(figsize=(8, 6), facecolor="#14172b")
        self._canvas_chunk = CanvasCls(self._fig_chunk)
        self._ax_chunk = self._fig_chunk.add_subplot(111)
        self._ax_chunk.set_facecolor("#16213e")
        self._ax_chunk.set_title("Chunk Single-Photon Event Map", color="#f8fafc", fontsize=11)
        self._ax_chunk.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_chunk.spines.values():
            spine.set_color("#2d3561")
        self._fig_chunk.tight_layout()

        c_layout.addWidget(self._canvas_chunk, stretch=1)
        layout.addWidget(canvas_card, stretch=1)

        return view

    def _create_bottom_bar(self) -> QWidget:
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._progress_bar = QProgressBar(bar)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(18)
        layout.addWidget(self._progress_bar, stretch=1)

        self._status_lbl = QLabel("Ready", bar)
        self._status_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        layout.addWidget(self._status_lbl)

        self._cancel_btn = QPushButton("Cancel", bar)
        theme.set_cancel_btn(self._cancel_btn)
        self._cancel_btn.clicked.connect(self._handle_cancel_clicked)
        self._cancel_btn.hide()
        layout.addWidget(self._cancel_btn)

        return bar

    # ------------------------------------------------------------------
    # Session Loading & Pipeline Execution
    # ------------------------------------------------------------------

    def load_session(
        self,
        signal_paths: Sequence[Path | str],
        chunk_size: int = 80,
        cluster_config: ClusterConfig | None = None,
        recon_config: ReconstructionConfig | None = None,
        mask_dir: Path | str | None = None,
        auto_run: bool = True,
    ) -> None:
        """Load a new dataset session into the studio and optionally start cluster extraction.

        Args:
            signal_paths: Sequence of filepaths to raw signal TIFF files.
            chunk_size: Number of frames per chunk.
            cluster_config: Optional Stage 2 cluster config.
            recon_config: Optional Stage 3 reconstruction config.
            mask_dir: Optional custom dark mask directory.
            auto_run: If True, automatically launches Stage 2 background extraction.
        """
        self.manager.init_session(
            signal_paths=signal_paths,
            chunk_size=chunk_size,
            cluster_config=cluster_config,
            recon_config=recon_config,
            mask_dir=mask_dir,
        )

        n_frames = self.manager.total_frames
        n_chunks = self.manager.total_chunks

        # Update Frame Inspector ranges
        self.frame_spin.blockSignals(True)
        self.frame_slider.blockSignals(True)
        self.frame_spin.setRange(1, max(1, n_frames))
        self.frame_slider.setRange(1, max(1, n_frames))
        self.frame_spin.setValue(1)
        self.frame_slider.setValue(1)
        self._frame_max_lbl.setText(f"/ {n_frames}")
        self.frame_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)

        # Update Chunk Inspector ranges
        self.chunk_spin.blockSignals(True)
        self.chunk_slider.blockSignals(True)
        self.chunk_spin.setRange(1, max(1, n_chunks))
        self.chunk_slider.setRange(1, max(1, n_chunks))
        self.chunk_spin.setValue(1)
        self.chunk_slider.setValue(1)
        self._chunk_max_lbl.setText(f"/ {n_chunks}")
        self.chunk_spin.blockSignals(False)
        self.chunk_slider.blockSignals(False)

        # Sync Stage 3 sliders & controls
        cfg = self.manager.state.recon_config
        self.intden_slider.set_values(cfg.intden_low, cfg.intden_high)
        self.max_area_spin.setValue(cfg.max_area)
        self.min_circ_spin.setValue(cfg.min_circ)
        self.subpixel_spin.setValue(cfg.subpixel_factor)

        self._current_frame_idx = 0
        self._current_chunk_idx = 0
        self._stale_banner.hide()

        self._update_kpi_cards()

        if auto_run and n_frames > 0:
            self.start_pipeline()

    def start_pipeline(self) -> None:
        """Start or re-run Stage 2 single-photon cluster extraction worker."""
        if self._pipeline_worker is not None and not self._pipeline_worker.is_canceled:
            self._pipeline_worker.cancel()

        self.manager.clear_clusters()
        self.manager.state.is_processing = True
        self._stale_banner.hide()

        self._progress_bar.setValue(0)
        self._cancel_btn.show()
        self._status_lbl.setText("Extracting photon clusters...")

        # Reset canvases
        self._reset_dashboard_canvas()

        worker = ClusterPipelineWorker(
            signal_paths=self.manager.state.signal_paths,
            med_dark=self.manager.state.med_dark,
            final_mask=self.manager.state.final_mask,
            config=self.manager.state.cluster_config,
        )
        self._pipeline_worker = worker

        worker.signals.frame_result.connect(self._on_worker_frame_result)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.progress_msg.connect(self._on_worker_progress_msg)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.canceled.connect(self._on_worker_canceled)

        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Progressive Canvas Accumulation & Worker Slots
    # ------------------------------------------------------------------

    @Slot(int, object)
    def _on_worker_frame_result(self, frame_idx: int, frame_df: pd.DataFrame) -> None:
        """Progressive accumulation: append clusters from each frame as it finishes."""
        self.manager.append_frame_clusters(frame_idx, frame_df)
        self._update_kpi_cards()

        if not self._accum_timer.isActive():
            self._accum_timer.start()

        # If user is in Frame Inspector and currently viewing this frame, refresh
        if self._current_mode == "Frame Inspector" and self._current_frame_idx == frame_idx:
            self._render_frame_inspector()

    @Slot()
    def _on_accum_timer_tick(self) -> None:
        """Throttled update of the 2D event map and IntDen histogram."""
        if self.manager.has_clusters:
            recon = self.manager.get_reconstruction()
            self._render_dashboard_event_map(recon.event_map)
            self._render_intden_histogram()

    @Slot(int, int, int)
    def _on_worker_progress(self, current: int, total: int, clusters: int) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self.kpi_1.set_content(
            f"{current} / {total}",
            f"{pct}% extracted",
            color="#38bdf8",
        )
        self.kpi_2.set_content(
            f"{clusters:,}",
            f"{clusters / max(1, current):.1f} / frame",
            color="#38bdf8",
        )

    @Slot(str)
    def _on_worker_progress_msg(self, msg: str) -> None:
        self._status_lbl.setText(msg)

    @Slot(object)
    def _on_worker_finished(self, df_all: pd.DataFrame) -> None:
        self._accum_timer.stop()
        self.manager.set_all_clusters(df_all)
        self.manager.state.is_processing = False
        self._pipeline_worker = None
        self._cancel_btn.hide()
        self._progress_bar.setValue(100)
        self._status_lbl.setText(f"Extraction complete ({len(df_all):,} clusters).")

        # Full refresh across views
        self._on_accum_timer_tick()
        self._update_kpi_cards()

        if self._current_mode == "Frame Inspector":
            self._render_frame_inspector()
        elif self._current_mode == "Chunk Inspector":
            self._render_chunk_inspector()

    @Slot(str)
    def _on_worker_error(self, err_msg: str) -> None:
        self.manager.state.is_processing = False
        self._pipeline_worker = None
        self._cancel_btn.hide()
        self._status_lbl.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Cluster Extraction Error", f"Extraction failed:\n{err_msg}")

    @Slot()
    def _on_worker_canceled(self) -> None:
        self.manager.state.is_processing = False
        self._pipeline_worker = None
        self._cancel_btn.hide()
        self._status_lbl.setText("Cluster extraction canceled by user.")

    # ------------------------------------------------------------------
    # Mode Switching & Contextual KPI Synchronization
    # ------------------------------------------------------------------

    @property
    def active_mode(self) -> str:
        """Currently active studio mode ('Dashboard', 'Frame Inspector', 'Chunk Inspector')."""
        return self._current_mode

    def set_mode(self, mode: str) -> None:
        """Switch the studio view mode programmatically."""
        mode_map = {
            "Dashboard": 0,
            "Frame Inspector": 1,
            "Chunk Inspector": 2,
        }
        if mode in mode_map:
            self.mode_combo.setCurrentIndex(mode_map[mode])

    def _handle_mode_changed(self, index: int) -> None:
        modes = ["Dashboard", "Frame Inspector", "Chunk Inspector"]
        self._current_mode = modes[index] if index < len(modes) else "Dashboard"
        self._mode_stack.setCurrentIndex(index)

        self._update_kpi_cards()

        if self._current_mode == "Dashboard":
            if self.manager.has_clusters:
                recon = self.manager.get_reconstruction()
                self._render_dashboard_event_map(recon.event_map)
                self._render_intden_histogram()
        elif self._current_mode == "Frame Inspector":
            self._render_frame_inspector()
        elif self._current_mode == "Chunk Inspector":
            self._render_chunk_inspector()

    def _on_frame_processed(
        self,
        frame_idx: int = 0,
        total: int = 0,
        running_clusters: int = 0,
        frame_df: pd.DataFrame | None = None,
    ) -> None:
        """Progressively accumulate event map and update live metrics."""
        if frame_df is not None and not frame_df.empty:
            slice_num = frame_idx + 1 if (frame_idx == 0 and "Slice" in frame_df.columns) else frame_idx
            self.manager.append_frame_clusters(slice_num, frame_df)
            if self._progressive_event_map is None or self._progressive_event_map.shape != self.manager.state.image_shape:
                self._progressive_event_map = np.zeros(self.manager.state.image_shape, dtype=np.float32)
            for _, row in frame_df.iterrows():
                xm, ym = int(round(float(row["XM"]))), int(round(float(row["YM"])))
                if 0 <= ym < self._progressive_event_map.shape[0] and 0 <= xm < self._progressive_event_map.shape[1]:
                    self._progressive_event_map[ym, xm] += 1.0
        self._running_cluster_count = len(self.manager.state.df_clusters)
        self._update_kpi_cards()

    def _update_kpi_cards(self) -> None:
        """Update KPI cards dynamically based on active mode."""
        df = self.manager.state.df_clusters
        total_clusters = len(df)
        total_frames = self.manager.total_frames
        processed = self.manager.state.processed_frame_count

        if self._current_mode == "Dashboard":
            self.kpi_1._title_lbl.setText("TOTAL FRAMES")
            self.kpi_2._title_lbl.setText("TOTAL CLUSTERS")
            self.kpi_3._title_lbl.setText("SINGLE PHOTONS")
            self.kpi_4._title_lbl.setText("REJECTED EVENTS")
            recon = self.manager.state.latest_recon or self.manager.get_reconstruction()
            pct_frames = (processed / total_frames * 100.0) if total_frames > 0 else 0.0

            self.kpi_1.set_content(
                f"{processed} / {total_frames}",
                f"{pct_frames:.0f}% frames processed",
            )
            self.kpi_2.set_content(
                f"{total_clusters:,}",
                f"{total_clusters / max(1, processed):.1f} clusters / frame",
            )
            self.kpi_3.set_content(
                f"{recon.accepted_events:,} ({recon.acceptance_pct:.1f}%)",
                "Single-photon events",
                color="#34d399",
            )
            rejected_total = recon.rejected_noise + recon.rejected_pileup + recon.rejected_shape
            self.kpi_4.set_content(
                f"{rejected_total:,}",
                f"Noise:{recon.rejected_noise} Shape:{recon.rejected_shape}",
                color="#f87171",
            )

        elif self._current_mode == "Frame Inspector":
            self.kpi_1._title_lbl.setText("FRAME")
            self.kpi_2._title_lbl.setText("FRAME CLUSTERS")
            self.kpi_3._title_lbl.setText("FRAME PHOTONS")
            self.kpi_4._title_lbl.setText("FRAME REJECTED")
            slice_num = self._current_frame_idx + 1
            frame_df = self.manager.get_frame_clusters(slice_num)
            n_frame_clusters = len(frame_df)

            cfg = self.manager.state.recon_config
            if not frame_df.empty:
                int_dens = frame_df["IntDen"].to_numpy()
                areas = frame_df["Area"].to_numpy()
                circs = frame_df["Circ."].to_numpy()
                accepted_mask = (
                    (int_dens >= cfg.intden_low)
                    & (int_dens <= cfg.intden_high)
                    & (areas <= cfg.max_area)
                    & (circs >= cfg.min_circ)
                )
                n_accepted = int(np.count_nonzero(accepted_mask))
                n_rejected = n_frame_clusters - n_accepted
            else:
                n_accepted = 0
                n_rejected = 0

            filename = ""
            if 0 <= self._current_frame_idx < len(self.manager.state.signal_paths):
                filename = Path(self.manager.state.signal_paths[self._current_frame_idx]).name

            self.kpi_1.set_content(
                f"Frame {self._current_frame_idx + 1} / {total_frames}",
                filename,
            )
            self.kpi_2.set_content(
                f"{n_frame_clusters} clusters",
                "Detected in current frame",
            )
            self.kpi_3.set_content(
                f"{n_accepted} accepted",
                f"{(n_accepted / max(1, n_frame_clusters) * 100):.1f}% single photons",
                color="#34d399",
            )
            self.kpi_4.set_content(
                f"{n_rejected} rejected",
                "Filtered out",
                color="#f87171",
            )

        elif self._current_mode == "Chunk Inspector":
            self.kpi_1._title_lbl.setText("CHUNK")
            self.kpi_2._title_lbl.setText("CHUNK CLUSTERS")
            self.kpi_3._title_lbl.setText("CHUNK PHOTONS")
            self.kpi_4._title_lbl.setText("CHUNK REJECTED")
            ranges = self.manager.get_chunk_frame_ranges()
            total_chunks = len(ranges)
            if 0 <= self._current_chunk_idx < total_chunks:
                start_f, end_f = ranges[self._current_chunk_idx]
                chunk_recon = self.manager.get_chunk_reconstruction(self._current_chunk_idx)

                self.kpi_1.set_content(
                    f"Chunk {self._current_chunk_idx + 1} / {total_chunks}",
                    f"Frames {start_f} - {end_f}",
                )
                self.kpi_2.set_content(
                    f"{chunk_recon.total_clusters:,} clusters",
                    f"In frames {start_f} to {end_f}",
                )
                self.kpi_3.set_content(
                    f"{chunk_recon.accepted_events:,} ({chunk_recon.acceptance_pct:.1f}%)",
                    "Accepted in chunk",
                    color="#34d399",
                )
                rejected = chunk_recon.total_clusters - chunk_recon.accepted_events
                self.kpi_4.set_content(
                    f"{rejected:,}",
                    f"{100.0 - chunk_recon.acceptance_pct:.1f}% rejection rate",
                    color="#f87171",
                )

    # ------------------------------------------------------------------
    # Rendering: Dashboard Canvas & IntDen Histogram
    # ------------------------------------------------------------------

    def _reset_dashboard_canvas(self) -> None:
        self._ax_dashboard_event.clear()
        self._ax_dashboard_event.set_facecolor("#16213e")
        self._ax_dashboard_event.set_title("Reconstructed Photon Map", color="#f8fafc", fontsize=11)
        self._ax_dashboard_event.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_dashboard_event.spines.values():
            spine.set_color("#2d3561")
        self._im_dashboard_event = None
        self._canvas_dashboard.draw_idle()

    def _render_dashboard_event_map(self, event_map: np.ndarray) -> None:
        if event_map is None:
            return
        try:
            event_arr = np.asarray(event_map, dtype=np.float32)
            if event_arr.ndim != 2 or event_arr.size == 0:
                return
        except Exception:
            return

        vmax = float(np.max(event_arr)) if event_arr.size > 0 else 1.0
        if vmax > self._clamping_slider_max:
            self._clamping_slider_max = max(1.0, vmax)
            self._clamping_slider.configure_range(0.0, self._clamping_slider_max)

        cmap = self._dash_cmap_combo.currentText()
        if self._im_dashboard_event is None:
            self._ax_dashboard_event.clear()
            self._ax_dashboard_event.set_facecolor("#16213e")
            self._im_dashboard_event = self._ax_dashboard_event.imshow(
                event_arr,
                cmap=cmap,
                origin="lower",
                vmin=self._clamping_floor,
                vmax=self._clamping_ceiling,
            )
            self._ax_dashboard_event.set_title(
                f"2D Photon Event Map ({int(np.sum(event_arr)):,} photons)",
                color="#f8fafc",
                fontsize=11,
            )
            self._ax_dashboard_event.tick_params(colors="#94a3b8", labelsize=9)
            for spine in self._ax_dashboard_event.spines.values():
                spine.set_color("#2d3561")
            self._fig_dashboard.tight_layout()
            if self._current_zoom_level > 1.0:
                self._apply_zoom()
            else:
                self._ax_dashboard_event.set_xlim(0.0, float(event_arr.shape[1]))
                self._ax_dashboard_event.set_ylim(0.0, float(event_arr.shape[0]))
        else:
            self._im_dashboard_event.set_data(event_arr)
            self._im_dashboard_event.set_clim(self._clamping_floor, self._clamping_ceiling)
            self._ax_dashboard_event.set_title(
                f"2D Photon Event Map ({int(np.sum(event_arr)):,} photons)",
                color="#f8fafc",
                fontsize=11,
            )
            if self._current_zoom_level > 1.0:
                self._apply_zoom()

        self._canvas_dashboard.draw_idle()

    def _render_intden_histogram(self) -> None:
        df = self.manager.state.df_clusters
        self._ax_hist.clear()
        self._ax_hist.set_facecolor("#16213e")

        if not df.empty:
            int_dens = df["IntDen"].to_numpy()
            self._ax_hist.hist(
                int_dens,
                bins=150,
                range=(0.0, 1500.0),
                color="#3b82f6",
                edgecolor="none",
                log=True,
                alpha=0.85,
            )

        cfg = self.manager.state.recon_config
        self._line_low = self._ax_hist.axvline(
            cfg.intden_low, color="#e11d48", linewidth=1.8, linestyle="--", label=f"Low: {cfg.intden_low:.1f}"
        )
        self._line_high = self._ax_hist.axvline(
            cfg.intden_high, color="#059669", linewidth=1.8, linestyle="--", label=f"High: {cfg.intden_high:.1f}"
        )

        self._ax_hist.set_title(
            f"IntDen Distribution ({len(df):,} clusters)",
            color="#f8fafc",
            fontsize=11,
        )
        self._ax_hist.set_xlabel("IntDen (ADU)", color="#e2e8f0", fontsize=9)
        self._ax_hist.set_ylabel("Counts (log)", color="#e2e8f0", fontsize=9)
        self._ax_hist.tick_params(colors="#94a3b8", labelsize=8)
        for spine in self._ax_hist.spines.values():
            spine.set_color("#2d3561")
        self._ax_hist.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")

        self._fig_hist.tight_layout()
        self._canvas_hist.draw_idle()

    # ------------------------------------------------------------------
    # Instant <50ms Stage 3 In-Memory RangeSlider Cuts
    # ------------------------------------------------------------------

    def _handle_intden_slider_changed(self, low_val: float, high_val: float) -> None:
        """Fast drag handler: updates visual cutlines only without full refilter."""
        self._intden_cut_lbl.setText(f"Single-Photon Window: {low_val:.1f} - {high_val:.1f} ADU")
        if self._line_low is not None:
            self._line_low.set_xdata([low_val, low_val])
        if self._line_high is not None:
            self._line_high.set_xdata([high_val, high_val])
        self._canvas_hist.draw_idle()

    def _handle_intden_slider_released(self, low_val: float, high_val: float) -> None:
        """Release handler: triggers in-memory Stage 3 filtering (<50ms benchmark)."""
        current_cfg = self.manager.state.recon_config
        new_cfg = ReconstructionConfig(
            intden_low=float(low_val),
            intden_high=float(high_val),
            max_area=current_cfg.max_area,
            min_circ=current_cfg.min_circ,
            subpixel_factor=current_cfg.subpixel_factor,
        )

        # In-memory execution strictly <50ms
        recon = self.manager.get_reconstruction(new_cfg)
        self._render_dashboard_event_map(recon.event_map)
        self._update_kpi_cards()

        if self._current_mode == "Frame Inspector":
            self._render_frame_inspector()
        elif self._current_mode == "Chunk Inspector":
            self._render_chunk_inspector()

    def _on_intden_slider_released(self, low_val: float, high_val: float) -> None:
        """Alias for slider release event."""
        self._handle_intden_slider_released(low_val, high_val)

    def _on_intden_slider_moved(self, low_val: float, high_val: float) -> None:
        """Alias for slider move event."""
        self._handle_intden_slider_changed(low_val, high_val)

    @property
    def _current_reconstruction(self) -> ReconstructionResult | None:
        """Latest Stage 3 reconstruction result."""
        return self.manager.state.latest_recon

    def _handle_shape_filter_changed(self) -> None:
        """Handler for Max Area, Min Circ, or Subpixel Factor changes."""
        current_cfg = self.manager.state.recon_config
        new_cfg = ReconstructionConfig(
            intden_low=current_cfg.intden_low,
            intden_high=current_cfg.intden_high,
            max_area=self.max_area_spin.value(),
            min_circ=self.min_circ_spin.value(),
            subpixel_factor=self.subpixel_spin.value(),
        )
        recon = self.manager.get_reconstruction(new_cfg)
        self._render_dashboard_event_map(recon.event_map)
        self._update_kpi_cards()

        if self._current_mode == "Frame Inspector":
            self._render_frame_inspector()
        elif self._current_mode == "Chunk Inspector":
            self._render_chunk_inspector()

    def _handle_dash_cmap_changed(self, cmap_name: str) -> None:
        if self._im_dashboard_event is not None:
            self._im_dashboard_event.set_cmap(cmap_name)
            self._canvas_dashboard.draw_idle()

    # ------------------------------------------------------------------
    # Dashboard Zoom & Intensity Clamping Handlers
    # ------------------------------------------------------------------

    def _handle_zoom_in(self) -> None:
        """Double zoom level up to 32× centered on axes center."""
        if self._current_zoom_level >= 32.0:
            return
        self._current_zoom_level = min(32.0, self._current_zoom_level * 2.0)
        self._apply_zoom()

    def _handle_zoom_out(self) -> None:
        """Halve zoom level down to 1× centered on axes center."""
        if self._current_zoom_level <= 1.0:
            return
        self._current_zoom_level = max(1.0, self._current_zoom_level / 2.0)
        self._apply_zoom()

    def _handle_zoom_reset(self) -> None:
        """Reset view to full detector bounds and 1× zoom."""
        self._current_zoom_level = 1.0
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        """Apply current zoom level to the 2D dashboard event map axes."""
        self._zoom_lbl.setText(f"Zoom: {int(self._current_zoom_level)}×")
        h, w = self.manager.state.image_shape
        if self._im_dashboard_event is not None:
            arr = self._im_dashboard_event.get_array()
            if arr is not None and hasattr(arr, "shape") and len(arr.shape) == 2:
                h, w = int(arr.shape[0]), int(arr.shape[1])

        if self._current_zoom_level <= 1.0:
            self._ax_dashboard_event.set_xlim(0.0, float(w))
            self._ax_dashboard_event.set_ylim(0.0, float(h))
        else:
            xlim = self._ax_dashboard_event.get_xlim()
            ylim = self._ax_dashboard_event.get_ylim()
            if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
                cx, cy = float(w) / 2.0, float(h) / 2.0
            else:
                cx = (float(xlim[0]) + float(xlim[1])) / 2.0
                cy = (float(ylim[0]) + float(ylim[1])) / 2.0

            hw = (float(w) / self._current_zoom_level) / 2.0
            hh = (float(h) / self._current_zoom_level) / 2.0

            x0 = max(0.0, cx - hw)
            x1 = x0 + 2.0 * hw
            if x1 > float(w):
                x1 = float(w)
                x0 = max(0.0, x1 - 2.0 * hw)

            y0 = max(0.0, cy - hh)
            y1 = y0 + 2.0 * hh
            if y1 > float(h):
                y1 = float(h)
                y0 = max(0.0, y1 - 2.0 * hh)

            self._ax_dashboard_event.set_xlim(x0, x1)
            self._ax_dashboard_event.set_ylim(y0, y1)

        self._canvas_dashboard.draw_idle()

    def _handle_clamping_changed(self, floor: float, ceiling: float) -> None:
        """Fast interactive response: clamp intensity contrast without recomputing event map."""
        self._clamping_floor = float(floor)
        self._clamping_ceiling = float(ceiling)
        self._floor_entry.setText(f"{self._clamping_floor:.2f}")
        self._ceiling_entry.setText(f"{self._clamping_ceiling:.2f}")
        if self._im_dashboard_event is not None:
            self._im_dashboard_event.set_clim(self._clamping_floor, self._clamping_ceiling)
            self._canvas_dashboard.draw_idle()

    def _on_floor_entry_submitted(self) -> None:
        """Handle manual text entry submission for clamping floor."""
        try:
            val = float(self._floor_entry.text())
        except ValueError:
            self._floor_entry.setText(f"{self._clamping_floor:.2f}")
            return
        val = max(0.0, min(val, self._clamping_ceiling))
        self._clamping_floor = val
        self._floor_entry.setText(f"{self._clamping_floor:.2f}")
        self._clamping_slider.set_values(self._clamping_floor, self._clamping_ceiling)
        if self._im_dashboard_event is not None:
            self._im_dashboard_event.set_clim(self._clamping_floor, self._clamping_ceiling)
            self._canvas_dashboard.draw_idle()

    def _on_ceiling_entry_submitted(self) -> None:
        """Handle manual text entry submission for clamping ceiling."""
        try:
            val = float(self._ceiling_entry.text())
        except ValueError:
            self._ceiling_entry.setText(f"{self._clamping_ceiling:.2f}")
            return
        val = max(self._clamping_floor, val)
        if val > self._clamping_slider_max:
            self._clamping_slider_max = max(1.0, val)
            self._clamping_slider.configure_range(0.0, self._clamping_slider_max)
        self._clamping_ceiling = val
        self._ceiling_entry.setText(f"{self._clamping_ceiling:.2f}")
        self._clamping_slider.set_values(self._clamping_floor, self._clamping_ceiling)
        if self._im_dashboard_event is not None:
            self._im_dashboard_event.set_clim(self._clamping_floor, self._clamping_ceiling)
            self._canvas_dashboard.draw_idle()

    # ------------------------------------------------------------------
    # Rendering: Frame Inspector Mode (Bounding Boxes & Centroids)
    # ------------------------------------------------------------------

    def _render_frame_inspector(self) -> None:
        n_frames = self.manager.total_frames
        if n_frames == 0 or self._current_frame_idx < 0 or self._current_frame_idx >= n_frames:
            return

        self._ax_frame.clear()
        self._ax_frame.set_facecolor("#16213e")

        slice_num = self._current_frame_idx + 1
        # Load dark-subtracted frame
        try:
            frame_img = self.manager.get_frame_image(slice_num, dark_subtracted=True)
            self._ax_frame.imshow(frame_img, cmap="viridis", origin="upper")
        except Exception as exc:
            logger.warning("Failed to render frame %d: %s", slice_num, exc)
            self._ax_frame.text(
                0.5, 0.5, f"Could not load frame {slice_num}",
                color="#ef4444", ha="center", va="center", transform=self._ax_frame.transAxes
            )
            self._canvas_frame.draw_idle()
            return

        # Fetch frame clusters
        frame_df = self.manager.get_frame_clusters(slice_num)
        cfg = self.manager.state.recon_config

        n_accepted = 0
        n_rejected = 0

        if not frame_df.empty:
            for _, row in frame_df.iterrows():
                xm = float(row["XM"])
                ym = float(row["YM"])
                area = int(row["Area"])
                circ = float(row["Circ."])
                int_den = float(row["IntDen"])

                is_accepted = (
                    (int_den >= cfg.intden_low)
                    & (int_den <= cfg.intden_high)
                    & (area <= cfg.max_area)
                    & (circ >= cfg.min_circ)
                )

                # Bounding box estimate based on area/extent
                box_half = max(1.5, np.sqrt(area) / 2.0 + 1.0)
                box_color = "#22c55e" if is_accepted else "#ef4444"

                if is_accepted:
                    n_accepted += 1
                else:
                    n_rejected += 1

                # Draw bounding box rectangle
                rect = patches.Rectangle(
                    (xm - box_half, ym - box_half),
                    box_half * 2.0,
                    box_half * 2.0,
                    linewidth=1.4,
                    edgecolor=box_color,
                    facecolor="none",
                )
                self._ax_frame.add_patch(rect)

                # Draw sub-pixel centroid marker
                self._ax_frame.plot(xm, ym, "+", color="#06b6d4", markersize=7, markeredgewidth=1.4)

        self._ax_frame.set_title(
            f"Frame {self._current_frame_idx + 1}/{n_frames} — "
            f"{len(frame_df)} clusters ({n_accepted} accepted, {n_rejected} rejected)",
            color="#f8fafc",
            fontsize=11,
        )
        self._ax_frame.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_frame.spines.values():
            spine.set_color("#2d3561")

        self._fig_frame.tight_layout()
        self._canvas_frame.draw_idle()
        self._update_kpi_cards()

    def prev_frame(self) -> None:
        """Scrub to previous frame."""
        if self._current_frame_idx > 0:
            self.frame_spin.setValue(self._current_frame_idx)

    def next_frame(self) -> None:
        """Scrub to next frame."""
        if self._current_frame_idx < max(0, self.manager.total_frames - 1):
            self.frame_spin.setValue(self._current_frame_idx + 2)

    def _handle_frame_first(self) -> None:
        self.frame_spin.setValue(1)

    def _handle_frame_last(self) -> None:
        self.frame_spin.setValue(max(1, self.manager.total_frames))

    def _handle_frame_spin_changed(self, value: int) -> None:
        self._current_frame_idx = max(0, value - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        if self._current_mode == "Frame Inspector":
            self._render_frame_inspector()

    # ------------------------------------------------------------------
    # Rendering: Chunk Inspector Mode
    # ------------------------------------------------------------------

    def _render_chunk_inspector(self) -> None:
        ranges = self.manager.get_chunk_frame_ranges()
        total_chunks = len(ranges)
        if total_chunks == 0 or self._current_chunk_idx < 0 or self._current_chunk_idx >= total_chunks:
            return

        self._ax_chunk.clear()
        self._ax_chunk.set_facecolor("#16213e")

        start_f, end_f = ranges[self._current_chunk_idx]
        chunk_recon = self.manager.get_chunk_reconstruction(self._current_chunk_idx)
        cmap = self._chunk_cmap_combo.currentText()

        self._im_chunk = self._ax_chunk.imshow(
            chunk_recon.event_map, cmap=cmap, origin="upper"
        )
        self._ax_chunk.set_title(
            f"Chunk {self._current_chunk_idx + 1}/{total_chunks} (Frames {start_f}-{end_f}) — "
            f"{chunk_recon.accepted_events:,} photons",
            color="#f8fafc",
            fontsize=11,
        )
        self._ax_chunk.tick_params(colors="#94a3b8", labelsize=9)
        for spine in self._ax_chunk.spines.values():
            spine.set_color("#2d3561")

        self._fig_chunk.tight_layout()
        self._canvas_chunk.draw_idle()
        self._update_kpi_cards()

    def prev_chunk(self) -> None:
        """Scrub to previous chunk."""
        if self._current_chunk_idx > 0:
            self.chunk_spin.setValue(self._current_chunk_idx)

    def next_chunk(self) -> None:
        """Scrub to next chunk."""
        if self._current_chunk_idx < self.manager.total_chunks - 1:
            self.chunk_spin.setValue(self._current_chunk_idx + 2)

    def _handle_chunk_first(self) -> None:
        self.chunk_spin.setValue(1)

    def _handle_chunk_last(self) -> None:
        self.chunk_spin.setValue(max(1, self.manager.total_chunks))

    def _handle_chunk_spin_changed(self, value: int) -> None:
        self._current_chunk_idx = value - 1
        self.chunk_slider.blockSignals(True)
        self.chunk_slider.setValue(value)
        self.chunk_slider.blockSignals(False)
        if self._current_mode == "Chunk Inspector":
            self._render_chunk_inspector()

    def _handle_chunk_cmap_changed(self, cmap_name: str) -> None:
        if self._im_chunk is not None:
            self._im_chunk.set_cmap(cmap_name)
            self._canvas_chunk.draw_idle()

    # ------------------------------------------------------------------
    # Actions: Re-Run & Chunk Export
    # ------------------------------------------------------------------

    def _handle_rerun_clicked(self) -> None:
        """Re-run Stage 2 cluster extraction."""
        if not self.manager.state.signal_paths:
            return
        self.start_pipeline()

    def _handle_cancel_clicked(self) -> None:
        """Cancel background Stage 2 extraction."""
        if self._pipeline_worker is not None:
            self._pipeline_worker.cancel()
            self._status_lbl.setText("Canceling extraction...")

    def _handle_save_chunks_clicked(self) -> None:
        """Open export folder dialog and launch ChunkSaveWorker."""
        if not self.manager.has_clusters:
            QMessageBox.information(self, "No Clusters", "Run cluster extraction before exporting products.")
            return

        default_dir = ""
        if self.manager.state.signal_paths:
            default_dir = str(self.manager.state.signal_paths[0].parent / "clusters")

        target_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory for Chunks and Products", default_dir
        )
        if not target_dir:
            return

        self._status_lbl.setText("Exporting chunks and products...")
        self._progress_bar.setValue(0)

        worker = ChunkSaveWorker(
            manager=self.manager,
            output_dir=target_dir,
            recon_config=self.manager.state.recon_config,
        )
        self._save_worker = worker

        worker.signals.progress.connect(
            lambda cur, tot: self._progress_bar.setValue(int(cur / tot * 100) if tot > 0 else 0)
        )
        worker.signals.progress_msg.connect(self._status_lbl.setText)
        worker.signals.finished.connect(
            lambda out_d: QMessageBox.information(
                self, "Export Complete", f"Successfully exported all products to:\n{out_d}"
            )
        )
        worker.signals.error.connect(
            lambda err: QMessageBox.critical(self, "Export Error", f"Export failed:\n{err}")
        )

        QThreadPool.globalInstance().start(worker)

    def _handle_back_clicked(self) -> None:
        """Handle ❮ Back to Home button press."""
        if self.manager.state.is_processing:
            reply = QMessageBox.question(
                self,
                "Cancel Processing?",
                "Cluster extraction is still running. Do you want to cancel and return home?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                if self._pipeline_worker is not None:
                    self._pipeline_worker.cancel()
                if self.on_back is not None:
                    self.on_back()
        else:
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
    # Teardown and Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up Matplotlib figures and background workers to prevent memory leaks."""
        if hasattr(self, "_accum_timer") and self._accum_timer.isActive():
            self._accum_timer.stop()

        if self._pipeline_worker is not None:
            self._pipeline_worker.cancel()
            self._pipeline_worker = None

        for canvas in [self._canvas_dashboard, self._canvas_hist, self._canvas_frame, self._canvas_chunk]:
            if canvas is not None:
                try:
                    canvas._destroyed = True
                except Exception:
                    pass
                if hasattr(canvas, "_idle_timer") and canvas._idle_timer is not None:
                    try:
                        canvas._idle_timer.stop()
                    except Exception:
                        pass

        try:
            for fig in [self._fig_dashboard, self._fig_hist, self._fig_frame, self._fig_chunk]:
                if fig is not None:
                    try:
                        fig.clear()
                    except Exception:
                        pass
        except Exception:
            pass

    def _teardown_mpl(self) -> None:
        """Teardown Matplotlib figures and resources."""
        self.cleanup()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Handle widget close event with resource cleanup."""
        self.cleanup()
        super().closeEvent(event)
