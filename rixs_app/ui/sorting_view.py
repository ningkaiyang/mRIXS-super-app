"""Sorting view — PySide6 port.

Allows users to select and manage a list of TIFF files before launching
the alignment slideshow or zeroth-order calibration. Files are auto-sorted
naturally on selection and support native drag-and-drop item reordering.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QDialog,
    QScrollArea, QDialogButtonBox, QMessageBox, QAbstractItemView,
)

from rixs_app.core import natural_sort
from rixs_app.ui.theme import (
    set_sort_btn, set_danger_btn, set_danger_secondary_btn,
    set_success_btn, set_tool_btn, set_accent_btn,
)


class DragDropListWidget(QListWidget):
    """Custom QListWidget with native drag-and-drop item reordering."""

    def __init__(self, parent=None, on_reordered=None):
        super().__init__(parent)
        self.on_reordered = on_reordered
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def dropEvent(self, event) -> None:
        """Synchronise parent file_list when an item is dropped."""
        super().dropEvent(event)
        if self.on_reordered is not None:
            self.on_reordered()


class SortingView(QWidget):
    """Main sorting/workspace view for the mRIXS Super-App.

    Provides file selection, automatic natural sorting, drag-and-drop reorder,
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
        outer.setSpacing(10)

        # Header
        header = QLabel("mRIXS Super-App Workspace")
        header.setObjectName("header_title")
        header.setAlignment(Qt.AlignCenter)
        outer.addWidget(header)

        # --- Row 1: Primary Select Files ---
        self.select_button = QPushButton("\U0001f4c1 Select TIFF Files")
        self.select_button.setFixedHeight(42)
        set_accent_btn(self.select_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.select_button.setFont(f)
        self.select_button.clicked.connect(self.select_files)
        outer.addWidget(self.select_button)

        # --- Caption ---
        self.caption_label = QLabel("(Files auto-sorted naturally by name \u2014 drag items up/down to reorder)")
        self.caption_label.setObjectName("dim_label")
        self.caption_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.caption_label)

        # --- File list ---
        self.list_widget = DragDropListWidget(self, on_reordered=self._on_list_reordered)
        self.list_widget.itemClicked.connect(
            lambda item: self._select_item(self.list_widget.row(item))
        )
        outer.addWidget(self.list_widget, stretch=1)

        # --- Row 2: Remove / Clear All ---
        row2 = QHBoxLayout()
        self.remove_button = QPushButton("\u2715 Remove Selected")
        self.remove_button.setFixedHeight(36)
        set_danger_btn(self.remove_button)
        self.remove_button.clicked.connect(self.remove_file)
        row2.addWidget(self.remove_button)

        self.clear_button = QPushButton("\U0001f5d1 Clear All")
        self.clear_button.setFixedHeight(36)
        set_danger_secondary_btn(self.clear_button)
        self.clear_button.clicked.connect(self.clear_all)
        row2.addWidget(self.clear_button)
        outer.addLayout(row2)

        # --- Launch buttons ---
        self.start_button = QPushButton("\u25ba Start Alignment Slideshow")
        self.start_button.setFixedHeight(40)
        set_success_btn(self.start_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.start_button.setFont(f)
        self.start_button.clicked.connect(self.start_slideshow)
        outer.addWidget(self.start_button)

        self.zeroth_order_button = QPushButton(
            "\U0001f52c Zeroth-Order Focus & FWHM Calibration"
        )
        self.zeroth_order_button.setFixedHeight(40)
        set_sort_btn(self.zeroth_order_button)
        f = QFont(); f.setPointSize(14); f.setBold(True); self.zeroth_order_button.setFont(f)
        self.zeroth_order_button.clicked.connect(self.start_zeroth_order)
        outer.addWidget(self.zeroth_order_button)

        self.help_button = QPushButton("\u2753 Guide + Credits")
        self.help_button.setFixedWidth(170)
        set_tool_btn(self.help_button)
        self.help_button.clicked.connect(self.show_help)
        outer.addWidget(self.help_button, alignment=Qt.AlignCenter)

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
            self.file_list = natural_sort(self.file_list)
            self.update_listbox()

    def _on_list_reordered(self) -> None:
        """Synchronise internal file_list order with visual drag-drop item order."""
        new_list = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            filepath = item.data(Qt.UserRole)
            if filepath:
                new_list.append(filepath)
        self.file_list = new_list

    def _select_item(self, idx: int) -> None:
        """Select an item by index.

        Args:
            idx: The item index.
        """
        self.selected_index = idx

    def remove_file(self) -> None:
        """Remove the currently selected file."""
        idx = self.list_widget.currentRow()
        if idx < 0 and 0 <= self.selected_index < len(self.file_list):
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
            "Select Motor Scan Log (.txt) \u2014 Cancel to skip",
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
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for idx, filepath in enumerate(self.file_list):
            item = QListWidgetItem(os.path.basename(filepath))
            item.setData(Qt.UserRole, filepath)
            item.setTextAlignment(Qt.AlignCenter)
            self.list_widget.addItem(item)
        if 0 <= self.selected_index < self.list_widget.count():
            self.list_widget.setCurrentRow(self.selected_index)
        self.list_widget.blockSignals(False)

    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------

    def show_help(self) -> None:
        """Open the help/guide dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("mRIXS Super-App \u2014 Guide + Credits")
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
             "Click '\U0001f4c1 Select TIFF Files' to choose your spectroscopy images.\n"
             "Files are automatically sorted naturally by name upon import.\n"
             "Use '\U0001f5d1 Clear All' to remove all files and start over."),
            ("Step 2: Reorder Files (Drag & Drop)",
             "Click and drag any file up or down in the list to reorder.\n"
             "Frame 1 (top of list) is the REFERENCE frame \u2014 all other frames "
             "will be aligned to it."),
            ("Step 3: Start Slideshow",
             "Click '\u25ba Start Alignment Slideshow' to begin analysis.\n"
             "The default alignment engine is ECC (Enhanced Correlation "
             "Coefficient), which works well for most datasets. Warp is "
             "ON by default \u2014 frames are translated to align with Frame 1."),
            ("Zeroth-Order Calibration",
             "Click '\U0001f52c Zeroth-Order Calibration' from the main menu to analyze mirror pitch.\n"
             "Optionally select a scan log TXT file to map motor pitch positions to frames,\n"
             "then Export to generate the focus curve (mirror pitch vs FWHM)."),
            ("Tips",
             "\u2022 ECC is recommended for most workflows\n"
             "\u2022 Use viridis colormap for best visibility of faint features\n"
             "\u2022 Drag files in the list to change frame alignment sequence"),
            ("Credits",
             "mRIXS Super-App developed at Lawrence Berkeley National Laboratory.\n"
             "App Credits:\n"
             "\u2022 Nickolas Yang \u2014 Computing Student Assistant"),
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
