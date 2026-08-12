"""Sorting view — PySide6 port.

Allows users to select, sort, and manage a list of TIFF files before
launching the alignment slideshow or zeroth-order calibration.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QDialog,
    QScrollArea, QDialogButtonBox, QMessageBox,
)

from rixs_app.core import natural_sort
from rixs_app.ui.theme import PALETTE, accent_style


class SortingView(QWidget):
    """Main sorting/workspace view for the mRIXS Super-App.

    Provides file selection, natural sort, manual up/down reorder,
    file removal, and buttons to launch the alignment slideshow or
    zeroth-order calibration.

    Args:
        parent: Parent widget.
        on_start_slideshow: Callback invoked with a file list when the user
            clicks Start Alignment Slideshow.
        on_zeroth_order: Callback invoked with (file_list, txt_path=...) when
            the user clicks Zeroth-Order Calibration.
    """

    def __init__(
        self,
        parent=None,
        *,
        on_start_slideshow=None,
        on_zeroth_order=None,
    ):
        """Initialise the sorting view.

        Args:
            parent: Parent QWidget.
            on_start_slideshow: Callback for launching alignment slideshow.
            on_zeroth_order: Callback for launching zeroth-order calibration.
        """
        super().__init__(parent)
        self.on_start_slideshow = on_start_slideshow
        self.on_zeroth_order = on_zeroth_order
        self.file_list: list[str] = []
        self.selected_index: int = -1

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build and lay out all widgets."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(8)

        # Header
        header = QLabel("mRIXS Super-App Workspace")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        header.setFont(font)
        header.setAlignment(Qt.AlignCenter)
        outer.addWidget(header)

        # --- Button row 1: Select / Sort ---
        row1 = QHBoxLayout()
        self.select_button = QPushButton("\U0001f4c1 Select Files")
        self.select_button.clicked.connect(self.select_files)
        row1.addWidget(self.select_button)

        self.sort_button = QPushButton("\u2195 Sort Files")
        self.sort_button.setStyleSheet(
            "background-color: #1F6AA5; color: white; font-weight: bold;"
        )
        self.sort_button.clicked.connect(self.sort_files)
        row1.addWidget(self.sort_button)
        outer.addLayout(row1)

        # --- Button row 2: Up / Down / Remove / Clear ---
        row2 = QHBoxLayout()
        self.up_button = QPushButton("\u25b2 Up")
        self.up_button.setFixedWidth(80)
        self.up_button.clicked.connect(self.move_up)
        row2.addWidget(self.up_button)

        self.down_button = QPushButton("\u25bc Down")
        self.down_button.setFixedWidth(80)
        self.down_button.clicked.connect(self.move_down)
        row2.addWidget(self.down_button)

        self.remove_button = QPushButton("\u2715 Remove")
        self.remove_button.setFixedWidth(90)
        self.remove_button.setStyleSheet("background-color: #aa3333; color: white;")
        self.remove_button.clicked.connect(self.remove_file)
        row2.addWidget(self.remove_button)

        self.clear_button = QPushButton("\U0001f5d1 Clear All")
        self.clear_button.setFixedWidth(100)
        self.clear_button.setStyleSheet("background-color: #883333; color: white;")
        self.clear_button.clicked.connect(self.clear_all)
        row2.addWidget(self.clear_button)
        outer.addLayout(row2)

        # --- Launch buttons ---
        self.start_button = QPushButton("\u25ba Start Alignment Slideshow")
        self.start_button.setFixedHeight(40)
        self.start_button.setStyleSheet(
            f"background-color: {PALETTE['accent_green']}; color: white; "
            "font-size: 14px; font-weight: bold;"
        )
        self.start_button.clicked.connect(self.start_slideshow)
        outer.addWidget(self.start_button)

        self.zeroth_order_button = QPushButton(
            "\U0001f52c Zeroth-Order Focus & FWHM Calibration"
        )
        self.zeroth_order_button.setFixedHeight(40)
        self.zeroth_order_button.setStyleSheet(
            "background-color: #1F6AA5; color: white; "
            "font-size: 14px; font-weight: bold;"
        )
        self.zeroth_order_button.clicked.connect(self.start_zeroth_order)
        outer.addWidget(self.zeroth_order_button)

        self.help_button = QPushButton("\u2753 Help / Guide")
        self.help_button.setFixedWidth(140)
        self.help_button.setStyleSheet("background-color: #555; color: white;")
        self.help_button.clicked.connect(self.show_help)
        outer.addWidget(self.help_button, alignment=Qt.AlignLeft)

        # --- File list ---
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(
            lambda item: self._select_item(self.list_widget.row(item))
        )
        outer.addWidget(self.list_widget, stretch=1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def select_files(self) -> None:
        """Open a file dialog to choose multiple TIFF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select TIFF files", "",
            "TIFF Files (*.tif *.tiff)"
        )
        if files:
            self.file_list.extend(list(files))
            self.update_listbox()

    def sort_files(self) -> None:
        """Sort the file list using natural sorting."""
        self.file_list = natural_sort(self.file_list)
        self.update_listbox()

    def _select_item(self, idx: int) -> None:
        """Select an item by index.

        Args:
            idx: The item index.
        """
        self.selected_index = idx
        self.update_listbox()

    def move_up(self) -> None:
        """Move the selected file one position up in the list."""
        idx = self.selected_index
        if 0 < idx < len(self.file_list):
            self.file_list[idx], self.file_list[idx - 1] = (
                self.file_list[idx - 1], self.file_list[idx]
            )
            self.selected_index = idx - 1
            self.update_listbox()

    def move_down(self) -> None:
        """Move the selected file one position down in the list."""
        idx = self.selected_index
        if 0 <= idx < len(self.file_list) - 1:
            self.file_list[idx], self.file_list[idx + 1] = (
                self.file_list[idx + 1], self.file_list[idx]
            )
            self.selected_index = idx + 1
            self.update_listbox()

    def remove_file(self) -> None:
        """Remove the currently selected file."""
        idx = self.selected_index
        if 0 <= idx < len(self.file_list):
            self.file_list.pop(idx)
            if self.file_list:
                self.selected_index = min(idx, len(self.file_list) - 1)
            else:
                self.selected_index = -1
            self.update_listbox()

    def clear_all(self) -> None:
        """Clear all files from the list."""
        self.file_list.clear()
        self.selected_index = -1
        self.update_listbox()

    def start_slideshow(self) -> None:
        """Trigger the alignment slideshow callback."""
        if self.on_start_slideshow and self.file_list:
            self.on_start_slideshow(self.file_list)

    def start_zeroth_order(self) -> None:
        """Prompt for an optional scan log, then trigger zeroth-order callback."""
        if not self.on_zeroth_order or not self.file_list:
            return
        txt_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Motor Scan Log (.txt) — Cancel to skip",
            "",
            "Text files (*.txt)",
        )
        if not txt_path:
            QMessageBox.information(
                self, "No Scan Log Selected",
                "Proceeding without motor scan metadata.\n"
                "Export will not include mirror pitch vs. FWHM focus curve."
            )
        self.on_zeroth_order(self.file_list, txt_path=txt_path if txt_path else None)

    def update_listbox(self) -> None:
        """Refresh the visual file list."""
        self.list_widget.clear()
        for idx, filepath in enumerate(self.file_list):
            item = QListWidgetItem(os.path.basename(filepath))
            if idx == self.selected_index:
                item.setBackground(Qt.blue)
                item.setForeground(Qt.white)
            self.list_widget.addItem(item)
        if 0 <= self.selected_index < self.list_widget.count():
            self.list_widget.setCurrentRow(self.selected_index)

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------

    def show_help(self) -> None:
        """Open the help/guide dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Spectroscopy Alignment — Quick Guide")
        dlg.resize(660, 580)

        dlg_layout = QVBoxLayout(dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setSpacing(8)
        scroll.setWidget(scroll_content)
        dlg_layout.addWidget(scroll)

        guide_sections = [
            ("Step 1: Load Files",
             "Click '\U0001f4c1 Select Files' to choose your TIFF spectroscopy images.\n"
             "You can select multiple files at once.\n"
             "Use '\U0001f5d1 Clear All' to remove all files and start over."),
            ("Step 2: Sort Files",
             "Click '\u2195 Sort Files' to auto-sort by filename (natural sorting).\n"
             "Use '\u25b2 Up' / '\u25bc Down' to manually reorder if needed.\n"
             "Frame 1 (top of list) is the REFERENCE frame — all other frames "
             "will be aligned to it."),
            ("Step 3: Start Slideshow",
             "Click '\u25ba Start Alignment Slideshow' to begin analysis.\n"
             "The default alignment engine is ECC (Enhanced Correlation "
             "Coefficient), which works well for most datasets. Warp is "
             "ON by default — frames are translated to align with Frame 1."),
            ("Zeroth-Order Calibration",
             "Click '\U0001f52c Zeroth-Order Calibration' from the main menu to analyze mirror pitch.\n"
             "Optionally select a scan log TXT file to map motor pitch positions to frames,\n"
             "then Export to generate the focus curve (mirror pitch vs FWHM)."),
            ("Tips",
             "\u2022 ECC is recommended for most workflows\n"
             "\u2022 Use viridis colormap for best visibility of faint features\n"
             "\u2022 Each frame stores its own PCA threshold independently"),
        ]

        for title, body in guide_sections:
            title_lbl = QLabel(title)
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            title_lbl.setFont(f)
            content_layout.addWidget(title_lbl)

            body_lbl = QLabel(body)
            body_lbl.setWordWrap(True)
            content_layout.addWidget(body_lbl)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        dlg_layout.addWidget(buttons)

        dlg.exec()

    # ------------------------------------------------------------------
    # Compatibility shim (tests call this to refresh the displayed list)
    # ------------------------------------------------------------------

    def select_item(self, idx: int) -> None:
        """Select an item (public API shim used by unit tests).

        Args:
            idx: Item index to select.
        """
        self._select_item(idx)
