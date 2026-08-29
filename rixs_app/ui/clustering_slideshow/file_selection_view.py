"""Single-Photon Clustering File Selection and Ingestion View.

Provides:
- Dark calibration verification banner (Green OK with metrics vs Red missing with Calibrate link).
- Drag-and-drop signal TIFF folder and file ingest with natural numerical sorting.
- Chunk size configuration spinbox (default 80, range 20-1000) with live chunk count feedback.
- Advanced parameter configuration for Stage 2 cluster extraction and Stage 3 filtering.
- Launch validation enforcing presence of TIFF frames and active dark calibration.
- Navigation navbar with ❮ Back to Home button and Co-Pilot docking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rixs_app.core import calibration_store
from rixs_app.core.cli_utils import glob_tifs
from rixs_app.core.photon_clustering import ClusterConfig, ReconstructionConfig
from rixs_app.core.utils import natural_sort
from rixs_app.ui import theme

logger = logging.getLogger(__name__)


class SignalDropZone(QFrame):
    """Interactive drag-and-drop dropzone for signal TIFF folders and files."""

    def __init__(
        self,
        on_paths_dropped: Callable[[list[str]], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_paths_dropped = on_paths_dropped
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
                self._on_paths_dropped(paths)
        else:
            super().dropEvent(event)


class ClusteringFileSelectionView(QWidget):
    """File Selection & Ingest workstation for Single-Photon Event Clustering.

    Args:
        parent: Optional parent QWidget.
        on_back: Callback when user clicks '❮ Back to Home'.
        on_launch_studio: Callback when launching studio with (signal_paths, chunk_size, cluster_cfg, recon_cfg).
        on_navigate_dark_cal: Callback when user clicks 'Calibrate Now' / 'Recalibrate'.
        cal_dir: Optional custom storage directory for dark calibrations.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        on_back: Callable[[], None] | None = None,
        on_launch_studio: Callable[[list[str], int, ClusterConfig, ReconstructionConfig], None] | None = None,
        on_navigate_dark_cal: Callable[[], None] | None = None,
        cal_dir: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_back = on_back
        self.on_launch_studio = on_launch_studio
        self.on_navigate_dark_cal = on_navigate_dark_cal
        self.cal_dir = cal_dir

        self.signal_paths: list[str] = []
        self._copilot_btn: QPushButton | None = None
        self._has_valid_cal: bool = False

        self._init_ui()
        self._status_banner = self._cal_banner
        self._launch_btn = self.launch_btn
        self._chunk_spinbox = self.chunk_size_spin
        self.refresh_calibration_status()

    @property
    def chunk_size(self) -> int:
        """Configured frames per chunk."""
        return self.chunk_size_spin.value()

    # ------------------------------------------------------------------
    # UI Initialization
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Navigation Bar
        navbar = self._create_navbar()
        main_layout.addWidget(navbar)

        # 2. Dark Calibration Status Banner
        self._cal_banner = self._create_cal_banner()
        main_layout.addWidget(self._cal_banner)

        # 3. Main Content Split / Columns (Scrollable Area)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # Left Column: File Ingest Dropzone and File List
        left_col = self._create_left_column()
        content_layout.addWidget(left_col, stretch=3)

        # Right Column: Chunk Size & Advanced Parameters
        right_col = self._create_right_column()
        content_layout.addWidget(right_col, stretch=2)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, stretch=1)

        # 4. Bottom Launch Action Bar
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_navbar(self) -> QWidget:
        navbar = QWidget(self)
        layout = QHBoxLayout(navbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._back_btn = QPushButton("❮ Back to Home", navbar)
        theme.set_tool_btn(self._back_btn)
        self._back_btn.clicked.connect(self._handle_back_clicked)
        layout.addWidget(self._back_btn)

        title_lbl = QLabel("Single-Photon Event Clustering — File Selection", navbar)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title_lbl)

        layout.addStretch(1)

        # Co-Pilot button docking container
        self._copilot_container = QWidget(navbar)
        self._copilot_container_layout = QHBoxLayout(self._copilot_container)
        self._copilot_container_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._copilot_container)

        return navbar

    def _create_cal_banner(self) -> QFrame:
        banner = QFrame(self)
        banner.setObjectName("cal_status_ok")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        self._cal_status_icon = QLabel("✓", banner)
        self._cal_status_icon.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self._cal_status_icon)

        self._cal_status_text = QLabel(banner)
        self._cal_status_text.setWordWrap(True)
        self._cal_status_text.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._cal_status_text, stretch=1)

        self._cal_action_btn = QPushButton(banner)
        theme.set_tool_btn(self._cal_action_btn)
        self._cal_action_btn.clicked.connect(self._handle_cal_action_clicked)
        layout.addWidget(self._cal_action_btn)

        return banner

    def _create_left_column(self) -> QWidget:
        col = QWidget(self)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Dropzone Box
        self._drop_frame = SignalDropZone(self.handle_dropped_paths, col)
        theme.set_squircle_card(self._drop_frame)
        self._drop_frame.setMinimumHeight(120)

        drop_layout = QVBoxLayout(self._drop_frame)
        drop_layout.setContentsMargins(20, 16, 20, 16)
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_layout.setSpacing(6)

        drop_icon = QLabel("📥", self._drop_frame)
        drop_icon.setStyleSheet("font-size: 28px;")
        drop_icon.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(drop_icon)

        drop_label = QLabel("Drag & Drop Signal TIFF Folder or Files Here", self._drop_frame)
        drop_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(drop_label)

        drop_sub = QLabel("Supports single frames or raw image sequences (.tif, .tiff)", self._drop_frame)
        drop_sub.setStyleSheet("font-size: 11px; color: #94a3b8;")
        drop_sub.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(drop_sub)

        layout.addWidget(self._drop_frame)

        # Browse & Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._browse_dir_btn = QPushButton("📂 Browse Folder...", col)
        theme.set_tool_btn(self._browse_dir_btn)
        self._browse_dir_btn.clicked.connect(self._handle_browse_dir)
        btn_row.addWidget(self._browse_dir_btn)

        self._browse_files_btn = QPushButton("📄 Add Files...", col)
        theme.set_tool_btn(self._browse_files_btn)
        self._browse_files_btn.clicked.connect(self._handle_browse_files)
        btn_row.addWidget(self._browse_files_btn)

        btn_row.addStretch(1)

        self._clear_btn = QPushButton("🗑 Clear All", col)
        theme.set_cancel_btn(self._clear_btn)
        self._clear_btn.clicked.connect(self.clear_files)
        btn_row.addWidget(self._clear_btn)

        layout.addLayout(btn_row)

        # File List Section
        list_header_row = QHBoxLayout()
        self._file_count_lbl = QLabel("No Signal TIFF Files Loaded", col)
        self._file_count_lbl.setStyleSheet("font-weight: bold; color: #e2e8f0;")
        list_header_row.addWidget(self._file_count_lbl)
        list_header_row.addStretch(1)
        layout.addLayout(list_header_row)

        self._file_list = QListWidget(col)
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self._file_list, stretch=1)

        return col

    def _create_right_column(self) -> QWidget:
        col = QWidget(self)
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Chunk Partitioning Group
        chunk_box = QGroupBox("📦 Chunk Partitioning", col)
        chunk_layout = QVBoxLayout(chunk_box)
        chunk_layout.setContentsMargins(14, 14, 14, 14)
        chunk_layout.setSpacing(10)

        chunk_row = QHBoxLayout()
        chunk_lbl = QLabel("Frames per Chunk:", chunk_box)
        chunk_lbl.setStyleSheet("color: #e2e8f0;")
        chunk_row.addWidget(chunk_lbl)

        self.chunk_size_spin = QSpinBox(chunk_box)
        self.chunk_size_spin.setRange(1, 1000)
        self.chunk_size_spin.setValue(80)
        self.chunk_size_spin.setSingleStep(10)
        self.chunk_size_spin.valueChanged.connect(self._update_chunk_summary)
        chunk_row.addWidget(self.chunk_size_spin)
        chunk_layout.addLayout(chunk_row)

        self._chunk_summary_lbl = QLabel("0 frames = 0 chunks", chunk_box)
        self._chunk_summary_lbl.setStyleSheet("font-size: 12px; color: #38bdf8; font-weight: bold;")
        chunk_layout.addWidget(self._chunk_summary_lbl)

        chunk_desc = QLabel(
            "Dividing sequences into chunks allows independent drift evaluation and modular export.",
            chunk_box,
        )
        chunk_desc.setWordWrap(True)
        chunk_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        chunk_layout.addWidget(chunk_desc)

        layout.addWidget(chunk_box)

        # Stage 2 Extraction Parameters
        stage2_box = QGroupBox("🔬 Stage 2: Signal Conditioning & Clustering", col)
        s2_grid = QGridLayout(stage2_box)
        s2_grid.setContentsMargins(14, 14, 14, 14)
        s2_grid.setSpacing(10)

        s2_grid.addWidget(QLabel("Signal Threshold Low (ADU):", stage2_box), 0, 0)
        self.sig_low_spin = QDoubleSpinBox(stage2_box)
        self.sig_low_spin.setRange(1.0, 10000.0)
        self.sig_low_spin.setValue(45.0)
        self.sig_low_spin.setSingleStep(5.0)
        s2_grid.addWidget(self.sig_low_spin, 0, 1)

        s2_grid.addWidget(QLabel("Signal Threshold High (ADU):", stage2_box), 1, 0)
        self.sig_high_spin = QDoubleSpinBox(stage2_box)
        self.sig_high_spin.setRange(100.0, 1e7)
        self.sig_high_spin.setValue(1e6)
        self.sig_high_spin.setSingleStep(1000.0)
        s2_grid.addWidget(self.sig_high_spin, 1, 1)

        s2_grid.addWidget(QLabel("Pixel Connectivity:", stage2_box), 2, 0)
        self.connectivity_spin = QSpinBox(stage2_box)
        self.connectivity_spin.setRange(4, 8)
        self.connectivity_spin.setValue(8)
        self.connectivity_spin.setSingleStep(4)
        s2_grid.addWidget(self.connectivity_spin, 2, 1)

        layout.addWidget(stage2_box)

        # Stage 3 Filtering Defaults
        stage3_box = QGroupBox("🎯 Stage 3: Initial Reconstruction Cuts", col)
        s3_grid = QGridLayout(stage3_box)
        s3_grid.setContentsMargins(14, 14, 14, 14)
        s3_grid.setSpacing(10)

        s3_grid.addWidget(QLabel("Default IntDen Low (ADU):", stage3_box), 0, 0)
        self.intden_low_spin = QDoubleSpinBox(stage3_box)
        self.intden_low_spin.setRange(0.0, 5000.0)
        self.intden_low_spin.setValue(120.0)
        self.intden_low_spin.setSingleStep(10.0)
        s3_grid.addWidget(self.intden_low_spin, 0, 1)

        s3_grid.addWidget(QLabel("Default IntDen High (ADU):", stage3_box), 1, 0)
        self.intden_high_spin = QDoubleSpinBox(stage3_box)
        self.intden_high_spin.setRange(0.0, 10000.0)
        self.intden_high_spin.setValue(320.0)
        self.intden_high_spin.setSingleStep(10.0)
        s3_grid.addWidget(self.intden_high_spin, 1, 1)

        s3_grid.addWidget(QLabel("Max Cluster Area (px):", stage3_box), 2, 0)
        self.max_area_spin = QSpinBox(stage3_box)
        self.max_area_spin.setRange(1, 100)
        self.max_area_spin.setValue(9)
        s3_grid.addWidget(self.max_area_spin, 2, 1)

        s3_grid.addWidget(QLabel("Min Circularity:", stage3_box), 3, 0)
        self.min_circ_spin = QDoubleSpinBox(stage3_box)
        self.min_circ_spin.setRange(0.0, 1.0)
        self.min_circ_spin.setSingleStep(0.05)
        self.min_circ_spin.setValue(0.3)
        s3_grid.addWidget(self.min_circ_spin, 3, 1)

        layout.addWidget(stage3_box)
        layout.addStretch(1)

        return col

    def _create_bottom_bar(self) -> QWidget:
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

        self._launch_hint_lbl = QLabel(bar)
        self._launch_hint_lbl.setStyleSheet("font-size: 12px; color: #94a3b8;")
        layout.addWidget(self._launch_hint_lbl, stretch=1)

        self.launch_btn = QPushButton("🚀 Launch Clustering Studio", bar)
        self.launch_btn.setMinimumHeight(42)
        self.launch_btn.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px 20px;")
        theme.set_success_btn(self.launch_btn)
        self.launch_btn.clicked.connect(self._handle_launch_clicked)
        layout.addWidget(self.launch_btn)

        return bar

    # ------------------------------------------------------------------
    # Public Ingestion API
    # ------------------------------------------------------------------

    def load_folder(self, folder_path: str | Path) -> list[str]:
        """Ingest all TIFF frames from a directory with natural numerical sorting.

        Args:
            folder_path: Path to directory containing signal TIFFs.

        Returns:
            List of sorted absolute filepath strings.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            logger.warning("Specified path is not a directory: %s", folder_path)
            return []

        tif_paths = glob_tifs(folder)
        sorted_paths = natural_sort([str(Path(p).resolve()) for p in tif_paths])
        self.set_files(sorted_paths)
        return sorted_paths

    def load_files(self, file_paths: Sequence[str | Path]) -> list[str]:
        """Append a list of file paths to the current selection.

        Args:
            file_paths: Sequence of file paths.

        Returns:
            Consolidated list of loaded file paths.
        """
        valid_paths = []
        for p in file_paths:
            path = Path(p)
            if path.suffix.lower() in (".tif", ".tiff"):
                valid_paths.append(str(path.resolve()) if path.exists() else str(path))

        combined = list(set(self.signal_paths + valid_paths))
        sorted_paths = natural_sort(combined)
        self.set_files(sorted_paths)
        return sorted_paths

    def set_files(self, file_paths: Sequence[str]) -> None:
        """Set the exact list of signal files and refresh the UI."""
        self.signal_paths = list(file_paths)
        self._file_list.clear()
        for p in self.signal_paths:
            self._file_list.addItem(Path(p).name)

        n = len(self.signal_paths)
        if n == 0:
            self._file_count_lbl.setText("No Signal TIFF Files Loaded")
        elif n == 1:
            self._file_count_lbl.setText("Loaded 1 Signal TIFF Frame")
        else:
            self._file_count_lbl.setText(f"Loaded {n:,} Signal TIFF Frames")

        self._update_chunk_summary()
        self._update_launch_state()

    def clear_files(self) -> None:
        """Clear all loaded signal frames."""
        self.set_files([])

    def handle_dropped_paths(self, paths: list[str]) -> None:
        """Process dropped file or folder paths."""
        all_tifs: list[str] = []
        for p_str in paths:
            p = Path(p_str)
            if p.is_dir():
                tifs = glob_tifs(p)
                all_tifs.extend(str(f.resolve()) for f in tifs)
            elif p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                all_tifs.append(str(p.resolve()))

        if all_tifs:
            self.load_files(all_tifs)

    # ------------------------------------------------------------------
    # Calibration Verification & Status
    # ------------------------------------------------------------------

    def refresh_calibration_status(self) -> bool:
        """Check calibration store and update banner styling and launch readiness.

        Returns:
            True if valid dark calibration is active; False otherwise.
        """
        is_ok = calibration_store.has_calibration(cal_dir=self.cal_dir)
        self._has_valid_cal = is_ok
        if is_ok:
            summary = calibration_store.get_calibration_summary(cal_dir=self.cal_dir) or "Calibration Active"
            theme.set_cal_status_ok(self._cal_banner)
            self._cal_status_icon.setText("✓")
            self._cal_status_icon.setStyleSheet("font-size: 18px; font-weight: bold; color: #34d399;")
            self._cal_status_text.setText(
                f"<b>Dark Calibration Verified:</b> {summary}. Ready for single-photon clustering."
            )
            self._cal_status_text.setStyleSheet("color: #34d399;")
            self._cal_action_btn.setText("⚙ Recalibrate...")
        else:
            theme.set_cal_status_missing(self._cal_banner)
            self._cal_status_icon.setText("⚠️")
            self._cal_status_icon.setStyleSheet("font-size: 18px; font-weight: bold; color: #f87171;")
            self._cal_status_text.setText(
                "<b>No Dark Calibration Found:</b> Detector dark baseline and bad-pixel mask are required."
            )
            self._cal_status_text.setStyleSheet("color: #f87171;")
            self._cal_action_btn.setText("🔧 Calibrate Now")

        self._update_launch_state()
        return is_ok

    def _update_chunk_summary(self) -> None:
        n_frames = len(self.signal_paths)
        chunk_size = max(1, self.chunk_size_spin.value())
        if n_frames == 0:
            self._chunk_summary_lbl.setText("0 frames = 0 chunks")
        else:
            n_chunks = (n_frames + chunk_size - 1) // chunk_size
            self._chunk_summary_lbl.setText(
                f"{n_frames:,} frames ÷ {chunk_size} = {n_chunks} chunk{'s' if n_chunks > 1 else ''}"
            )

    def _update_launch_state(self) -> None:
        has_cal = calibration_store.has_calibration(cal_dir=self.cal_dir)
        has_files = len(self.signal_paths) > 0

        can_launch = has_cal and has_files
        self.launch_btn.setEnabled(can_launch)

        if not has_cal and not has_files:
            self._launch_hint_lbl.setText("Load signal TIFF frames and run dark calibration to launch.")
        elif not has_cal:
            self._launch_hint_lbl.setText("Dark calibration required before launching studio.")
        elif not has_files:
            self._launch_hint_lbl.setText("Load at least one signal TIFF frame to proceed.")
        else:
            self._launch_hint_lbl.setText(
                f"Ready: {len(self.signal_paths):,} frames configured into "
                f"{(len(self.signal_paths) + self.chunk_size_spin.value() - 1) // self.chunk_size_spin.value()} chunks."
            )

    # ------------------------------------------------------------------
    # Handlers & Callbacks
    # ------------------------------------------------------------------

    def _handle_browse_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Signal TIFF Folder")
        if folder:
            self.load_folder(folder)

    def _handle_browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Signal TIFF Files", "", "TIFF Images (*.tif *.tiff)"
        )
        if files:
            self.load_files(files)

    def _handle_back_clicked(self) -> None:
        if self.on_back is not None:
            self.on_back()

    def _handle_cal_action_clicked(self) -> None:
        if self.on_navigate_dark_cal is not None:
            self.on_navigate_dark_cal()

    def _handle_launch_clicked(self) -> None:
        if not self.signal_paths:
            return

        cluster_cfg = ClusterConfig(
            sig_thresh_low=self.sig_low_spin.value(),
            sig_thresh_high=self.sig_high_spin.value(),
            connectivity=self.connectivity_spin.value(),
        )

        recon_cfg = ReconstructionConfig(
            intden_low=self.intden_low_spin.value(),
            intden_high=self.intden_high_spin.value(),
            max_area=self.max_area_spin.value(),
            min_circ=self.min_circ_spin.value(),
        )

        chunk_size = self.chunk_size_spin.value()

        if self.on_launch_studio is not None:
            self.on_launch_studio(self.signal_paths, chunk_size, cluster_cfg, recon_cfg)

    def set_copilot_button(self, btn: QPushButton) -> None:
        """Dock Co-Pilot toggle button into navbar header."""
        if self._copilot_btn is not None and self._copilot_btn is not btn:
            self._copilot_container_layout.removeWidget(self._copilot_btn)
            self._copilot_btn.setParent(None)

        self._copilot_btn = btn
        self._copilot_container_layout.addWidget(btn)
        btn.setParent(self._copilot_container)
        btn.show()
